"""原生 XLA、官方 Pallas 与本文写法的对照：同一进程、同一个 TensorCore。

time 给出三方的计时区间（表 4 至表 7），fixes 给出 f32[8,128]、k = 8 上各种修正的计时区间（表 3），rows 比较多行时链的两种写法，check 给出逐位正确性（表 1、表 2）。
计时一律使用 shell.py 的外壳：起点和终点都在 TC VMEM。结果写到 --output 目录。
"""
import chip
import argparse
import json
from pathlib import Path

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tpuasm import tools as asm

import fixes
import shell
import total_order

# 标签、形状、轴、k。
TIME_CASES = (
    *((f'basic{k}', (8, 128), 1, k) for k in (1, 8, 16, 32, 128)),
    *((f'rows{rows}', (rows, 128), 1, 8) for rows in (16, 32, 64, 96, 128, 256)),
    ('rows16_16', (16, 128), 1, 16),
    *((f'wide{width}_8', (8, width), 1, 8) for width in (256, 1024, 4096)),
    *((f'wide{width}_32', (8, width), 1, 32) for width in (1024, 4096, 8192)),
    ('vertical8_8', (8, 128), 0, 8),
    ('vertical16_8', (16, 128), 0, 8),
    ('vertical32_32', (32, 128), 0, 32),
    ('vertical64_32', (64, 128), 0, 32),
    ('vertical128_8', (128, 128), 0, 8),
    ('vertical128_32', (128, 128), 0, 32),
    ('vertical256_32', (256, 128), 0, 32),
)
CHECK_CASES = (
    ((8, 128), 1, 1),
    ((8, 128), 1, 8),
    ((8, 128), 1, 16),
    ((8, 128), 1, 32),
    ((8, 128), 1, 128),
    ((1, 100), 1, 8),
    ((3, 17), 1, 17),
    ((8, 1), 1, 1),
    ((16, 128), 1, 8),
    ((16, 128), 1, 16),
    ((40, 128), 1, 32),
    ((64, 128), 1, 8),
    ((70, 128), 1, 8),
    ((96, 128), 1, 8),
    ((128, 128), 1, 8),
    ((256, 128), 1, 8),
    ((8, 256), 1, 8),
    ((8, 300), 1, 32),
    ((8, 1024), 1, 8),
    ((8, 4096), 1, 32),
    ((8, 8192), 1, 32),
    ((8, 128), 0, 8),
    ((16, 128), 0, 8),
    ((24, 128), 0, 8),
    ((33, 128), 0, 32),
    ((64, 128), 0, 32),
    ((128, 128), 0, 32),
    ((256, 128), 0, 32),
    ((2, 8, 128), 2, 8),
    ((2, 8, 128), 1, 8),
    ((2, 8, 128), 0, 2),
)
# 21 种特殊位型，按完整位型顺序从小到大：负号 NaN、-inf、负的正常数与次正规数、±0、正的次正规数与正常数、+inf、正号 NaN。
SPECIAL = np.array(
    [
        0xffffffff,
        0xfffffffe,
        0xffc00002,
        0xffc00001,
        0xff800001,
        0xff800000,
        0xff7fffff,
        0x80800000,
        0x807fffff,
        0x80000001,
        0x80000000,
        0,
        1,
        0x007fffff,
        0x00800000,
        0x7f7fffff,
        0x7f800000,
        0x7f800001,
        0x7fc00001,
        0x7fc00002,
        0x7fffffff,
    ],
    np.uint32,
)

def reference(x: np.ndarray, k: int, axis: int) -> np.ndarray:
    """参照：按可逆整数键做稳定的降序排序，取前 k 个的位型。"""
    bits = x.view(np.uint32)
    key = np.where(bits.view(np.int32) < 0, bits ^ 0x7fffffff, bits).view(np.int32)
    index = np.argsort(-key.astype(np.int64), axis=axis, kind='stable').take(range(k), axis=axis)
    return np.take_along_axis(bits, index, axis=axis)

def errors(x: np.ndarray, actual: tuple[jax.Array, jax.Array], k: int, axis: int) -> dict[str, int]:
    """沿 top-k 轴的每一条输入算一条，四类错误各有多少条，以及至少有一类错误的条数。"""
    values, index = (np.asarray(item) for item in actual)
    outside = np.any((index < 0) | (index >= x.shape[axis]), axis=axis)
    paired = np.take_along_axis(x.view(np.uint32), np.clip(index, 0, x.shape[axis] - 1), axis=axis)
    found = {
        'out_of_range': outside,
        'duplicate': np.any(np.diff(np.sort(index, axis=axis), axis=axis) == 0, axis=axis),
        'unpaired_bits': np.any(values.view(np.uint32) != paired, axis=axis),
        'wrong_bits': np.any(values.view(np.uint32) != reference(x, k, axis), axis=axis),
    }
    return {
        'rows': int(np.prod(x.shape) // x.shape[axis]),
        **{name: int(np.count_nonzero(wrong)) for name, wrong in found.items()},
        'any': int(np.count_nonzero(np.logical_or.reduce(list(found.values())))),
    }

def distributions(rng: np.random.Generator, shape: tuple[int, ...], axis: int, k: int, batches: int) -> list[tuple[str, np.ndarray]]:
    result = []
    for _ in range(batches):
        result.extend((
            ('normal', rng.normal(size=shape).astype(np.float32)),
            ('bits', rng.integers(0, 2**32, shape, np.uint32).view(np.float32)),
            ('special', rng.choice(SPECIAL, shape).view(np.float32)),
            ('ties', rng.integers(-3, 4, shape).astype(np.float32)),
        ))
    result += [('constant', np.full(shape, bit, np.uint32).view(np.float32)) for bit in SPECIAL]
    for count in range(min(k + 2, shape[axis] + 1)):
        x = -np.abs(rng.normal(size=shape).astype(np.float32))
        moved = np.moveaxis(x, axis, -1)
        moved[..., :count] = np.abs(moved[..., :count])
        result.append(('phase_count', np.ascontiguousarray(x)))
    corner = np.full(shape, -np.inf, np.float32)
    moved = np.moveaxis(corner, axis, -1)
    moved[..., 0] = 3.0
    moved[..., -1] = 2.0
    return result + [('finite_at_last', corner)]

def kernel_program(shape: tuple[int, ...], k: int, axis: int, using: shell.Implementation) -> asm.Compiled:
    """正确性核对用的 Pallas 程序：任意维数的输入，kernel 自己把输入读进 TC VMEM、把结果写出。"""
    out_shape = tuple(k if dimension == axis else size for dimension, size in enumerate(shape))

    @jax.shard_map(
        mesh=shell.MESH,
        in_specs=P(),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def function(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        @pl.kernel(
            out_type=(jax.ShapeDtypeStruct(out_shape, jnp.float32), jax.ShapeDtypeStruct(out_shape, jnp.int32)),
            mesh=shell.TC,
            scratch_types=(pltpu.VMEM(shape, jnp.float32), pltpu.VMEM(out_shape, jnp.float32), pltpu.VMEM(out_shape, jnp.int32), pltpu.SemaphoreType.DMA),
            name='top_k',
            compiler_params=shell.PARAMS,
        )
        def kernel(x_hbm: Ref, v_hbm: Ref, i_hbm: Ref, x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            v_vmem[...], i_vmem[...] = jax.lax.top_k(x_vmem[...], k, axis=axis, is_stable=False)
            pltpu.async_copy(v_vmem, v_hbm, sem).wait()
            pltpu.async_copy(i_vmem, i_hbm, sem).wait()

        return kernel(x)

    with shell.implementation(using):
        return asm.compile(function, np.zeros(shape, np.float32), mesh=shell.MESH)

def time(args: argparse.Namespace) -> None:
    clock = asm.KernelClock(num_cores=1)
    records = []
    for label, shape, axis, k in TIME_CASES:
        if args.cases and label not in args.cases:
            continue
        candidates = {
            'xla': (shell.xla_top_k(k, axis), shell.OFFICIAL),
            'xla_stable': (shell.xla_top_k(k, axis, stable=True), shell.OFFICIAL),
            'official': (shell.pallas_top_k(k, axis), shell.OFFICIAL),
            'ours': (shell.pallas_top_k(k, axis), total_order.fast),
        }
        for name, (top_k, using) in candidates.items():
            records.append({'label': label, 'variant': name, **shell.measure(clock, top_k, shape, k, axis, using=using, output=args.output, label=f'{label}-{name}')})
            (args.output / 'interval.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')

def fix(args: argparse.Namespace) -> None:
    """f32[8,128]、k = 8：只修正、不加速的几种写法。pairs 另用一组触发回退的输入（每行只有三个有限值）。"""
    clock = asm.KernelClock(num_cores=1)
    rng = np.random.default_rng(6604)
    sparse = []
    for _ in range(8):
        x = np.full((8, 128), -np.inf, np.float32)
        x[:, :3] = rng.normal(size=(8, 3))
        sparse.append(x)
    variants = {'official': shell.OFFICIAL, 'mask': fixes.mask, 'lift': fixes.lift, 'suffix': fixes.suffix, 'phased': total_order.phased, 'pairs': total_order.pairs}
    records = []
    for name, using in variants.items():
        records.append({'variant': name, 'inputs': 'permutation', **shell.measure(clock, shell.pallas_top_k(8, 1), (8, 128), 8, 1, using=using, label=name)})
        if name in ('phased', 'pairs'):
            records.append({'variant': name, 'inputs': 'three finite values per row', **shell.measure(clock, shell.pallas_top_k(8, 1), (8, 128), 8, 1, using=using, inputs=sparse, label=f'{name}-sparse')})
        (args.output / 'fixes.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')

def rows(args: argparse.Namespace) -> None:
    """行宽 128、k = 8，行数从 16 到 256：官方的链、逐位正确的链、只求下标并在结束后取值的链，以及转置之后沿子通道归并。"""
    clock = asm.KernelClock(num_cores=1)
    records = []
    for count in (16, 32, 64, 96, 128, 256):
        for name, using in (('official', shell.OFFICIAL), ('phased', total_order.phased), ('phased_gather', total_order.phased_gather), ('transposed', total_order.transposed)):
            if name == 'transposed' and count > 128:
                continue
            records.append({'variant': name, **shell.measure(clock, shell.pallas_top_k(8, 1), (count, 128), 8, 1, using=using, label=f'{count}-{name}')})
            (args.output / 'rows.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')

def ks(args: argparse.Namespace) -> None:
    """行宽 128、沿最后一维，k 取 8 以外的值：官方、不用转置时的通用分派，以及转置之后沿子通道归并。--cases 给出行数。"""
    clock = asm.KernelClock(num_cores=1)
    records = []

    def chain(operand: jax.Array, **options: object) -> tuple[jax.Array, jax.Array]:
        saved, total_order.use_transposed = total_order.use_transposed, lambda rows, k: False
        try:
            return total_order.fast(operand, **options)
        finally:
            total_order.use_transposed = saved

    for count in map(int, args.cases or ('16', '64', '128')):
        for k in (2, 4, 16, 32):
            for name, using in (('official', shell.OFFICIAL), ('chain', chain), ('transposed', total_order.transposed)):
                records.append({'variant': name, **shell.measure(clock, shell.pallas_top_k(k, 1), (count, 128), k, 1, using=using, label=f'{count}-k{k}-{name}')})
                (args.output / f'ks-{count}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')

def check(args: argparse.Namespace) -> None:
    """三个实现在同一批输入上的四类错误；不因出错而停止。"""
    rng = np.random.default_rng(6602)
    cpu = jax.local_devices(backend='cpu')[0]
    # 参照先与 CPU 逐位核对：固定的特殊位型与随机原始位型。
    for x in (SPECIAL.view(np.float32)[None, :], rng.integers(0, 2**32, (8, 128), np.uint32).view(np.float32)):
        values, _ = jax.jit(lambda x: jax.lax.top_k(x, x.shape[-1]))(jax.device_put(x, cpu))
        np.testing.assert_array_equal(np.asarray(values).view(np.uint32), reference(x, x.shape[-1], -1))
    records = []
    for shape, axis, k in CHECK_CASES:
        data = distributions(rng, shape, axis, k, args.batches)
        programs = {
            'xla': lambda: asm.compile(lambda x: jax.lax.top_k(x, k, axis=axis, is_stable=False), np.zeros(shape, np.float32)),
            'official': lambda: kernel_program(shape, k, axis, shell.OFFICIAL),
            'ours': lambda: kernel_program(shape, k, axis, total_order.fast),
        }
        for name, build in programs.items():
            compiled = build()
            totals: dict[str, dict[str, int]] = {}
            for distribution, x in data:
                wrong = errors(x, compiled(x), k, axis)
                total = totals.setdefault(distribution, dict.fromkeys(wrong, 0))
                for field, number in wrong.items():
                    total[field] += number
            summed = {field: sum(total[field] for total in totals.values()) for field in next(iter(totals.values()))}
            records.append({'shape': shape, 'axis': axis, 'k': k, 'variant': name, **summed, 'distributions': totals})
            print(json.dumps({key: value for key, value in records[-1].items() if key != 'distributions'}), flush=True)
            (args.output / 'correctness.json').write_text(json.dumps(records, indent=2) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('time', 'fix', 'rows', 'ks', 'check'))
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--output', type=Path, default=Path('/tmp/top-k-paper'))
    parser.add_argument('--batches', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {'time': time, 'fix': fix, 'rows': rows, 'ks': ks, 'check': check}[args.mode](args)

if __name__ == '__main__':
    main()
