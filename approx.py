"""近似 top-k：Pallas 的 approx_max_k 与原生 XLA，以及把本文的写法放进它的两个阶段。

Pallas 的 approx_max_k 分两段：先把一行按 num_bins 切片，逐元素比较，留下每个位置上各片的最大值与下标；再对这 num_bins 个候选调用 _top_k_impl。四种实现：

- xla：原生 XLA 的 jax.lax.approx_max_k，分桶方式与 Pallas 不同；
- official：未经修改的 Pallas approx_max_k；
- stage2：只把第二阶段换成本文的通用分派 total_order.fast；
- ours：两个阶段都换掉，第一阶段在整数键上比较，第二阶段用 fast。

check 在 compare.py 的同一批输入上数错误：下标越界、下标重复、值与下标不配对、全局最大值（按完整位型的顺序）没有排在第一，以及与参照不同。参照是同一个分桶算法在整数键上的精确结果：每个位置取各片中键最大、相等时下标最小的元素，再对候选做稳定的降序排序；它只适用于 Pallas 的三种实现。time 在 shell.py 的外壳中计时。
"""
import chip
import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path

import jax
from jax import lax, Ref
from jax._src.lax import ann
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tpuasm import tools as asm

import compare
import shell
import total_order

RECALL = 0.95
CHECK_CASES = (((8, 1024), 8), ((8, 1000), 8), ((16, 2048), 8), ((8, 4096), 32))
TIME_CASES = (((8, 1024), 8), ((8, 4096), 8), ((64, 4096), 8), ((8, 4096), 32), ((8, 8192), 32))
OFFICIAL_RULE = lowering.lowering_rules[tpu_core.CoreType.TC][ann.approx_top_k_p]

def num_bins(n: int, k: int, recall: float) -> int:
    """与 Pallas 的 approx_max_k 相同的分桶数：按召回率求出，向上取整到 128 的倍数，不超过 n。"""
    bins = 1 if k <= 1 else n if recall == 1.0 else math.ceil((k - 1) / (1.0 - recall))
    return min(-(-bins // 128) * 128, n)

def ours_impl(operand: jax.Array, *, k: int, recall: float) -> tuple[jax.Array, jax.Array]:
    """与 Pallas 的 approx_max_k 同样分桶，第一阶段在整数键上比较，第二阶段用 total_order.fast。只支持沿最后一维。"""
    axis = operand.ndim - 1
    n = operand.shape[axis]
    bins = num_bins(n, k, recall)
    if bins == n:
        return total_order.fast(operand, k=k, axis=axis)
    key = total_order.to_key(operand)
    full, remainder = divmod(n, bins)
    iota = lax.broadcasted_iota(jnp.int32, (*operand.shape[:-1], bins), axis)
    # 只保留键和下标；键是可逆的，候选的值在最后由键还原，第一阶段每个元素少一次选择。
    best_key, best_idx = key[..., :bins], iota
    pads = [(0, 0, 0)] * axis + [(0, bins - remainder, 0)]
    for i in range(1, full + (remainder > 0)):
        seg = lax.slice_in_dim(key, i * bins, min((i + 1) * bins, n), axis=axis)
        if i == full:
            # 填充的键是 INT_MIN，严格大于才替换，所以它不会被选中。
            seg = lax.pad(seg, jnp.int32(total_order.INT_MIN), pads)
        take = seg > best_key
        best_key, best_idx = jnp.where(take, seg, best_key), jnp.where(take, i * bins + iota, best_idx)
    best_val = total_order.from_key(best_key)
    return total_order.fast(best_val, k=k, axis=axis, carried_idx=best_idx)

def ours_rule(ctx: lowering.LoweringRuleContext, x: object, *, k: int, reduction_dimension: int, recall_target: float, is_max_k: bool, reduction_input_size_override: int, aggregate_to_topk: bool) -> object:
    assert is_max_k and aggregate_to_topk and reduction_dimension % len(ctx.avals_in[0].shape) == len(ctx.avals_in[0].shape) - 1
    return lowering.lower_fun(lambda operand: ours_impl(operand, k=k, recall=recall_target))(ctx, x)

@contextmanager
def variant(name: str) -> Iterator[None]:
    """编译期间按 name 替换 Pallas 中 approx_max_k 的 lowering rule 或其中第二阶段的 _top_k_impl。"""
    rules = lowering.lowering_rules[tpu_core.CoreType.TC]
    rules[ann.approx_top_k_p] = ours_rule if name == 'ours' else OFFICIAL_RULE
    try:
        with shell.implementation(total_order.fast) if name == 'stage2' else nullcontext():
            yield
    finally:
        rules[ann.approx_top_k_p] = OFFICIAL_RULE

def approx(k: int) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    return lambda x: jax.lax.approx_max_k(x, k, recall_target=RECALL)

def keys(x: np.ndarray) -> np.ndarray:
    bits = x.view(np.uint32)
    return np.where(bits.view(np.int32) < 0, bits ^ 0x7fffffff, bits).view(np.int32).astype(np.int64)

def reference(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Pallas 的分桶算法在整数键上的精确结果：(值的位型, 下标)，沿最后一维。"""
    n = x.shape[-1]
    bins = num_bins(n, k, RECALL)
    slices = -(-n // bins)
    key = np.pad(keys(x), [(0, 0)] * (x.ndim - 1) + [(0, slices * bins - n)], constant_values=-(1 << 40))
    grouped = key.reshape(*x.shape[:-1], slices, bins)
    # argmax 取第一次出现，即键相等时取下标最小的一片。
    winner = np.argmax(grouped, axis=-2)
    index = winner * bins + np.arange(bins)
    candidate = np.take_along_axis(grouped, winner[..., None, :], axis=-2)[..., 0, :]
    order = np.argsort(-candidate, axis=-1, kind='stable')[..., :k]
    chosen = np.take_along_axis(index, order, axis=-1)
    return np.take_along_axis(x.view(np.uint32), chosen, axis=-1), chosen

def errors(x: np.ndarray, actual: tuple[jax.Array, jax.Array], k: int, exact: bool) -> dict[str, int]:
    """沿最后一维的每一行算一条；exact 为假时不与参照比较。"""
    values, index = (np.asarray(item) for item in actual)
    n = x.shape[-1]
    paired = np.take_along_axis(x.view(np.uint32), np.clip(index, 0, n - 1), axis=-1)
    largest = np.take_along_axis(x.view(np.uint32), np.argmax(keys(x), axis=-1)[..., None], axis=-1)[..., 0]
    found = {
        'out_of_range': np.any((index < 0) | (index >= n), axis=-1),
        'duplicate': np.any(np.diff(np.sort(index, axis=-1), axis=-1) == 0, axis=-1),
        'unpaired_bits': np.any(values.view(np.uint32) != paired, axis=-1),
        'max_missing': values.view(np.uint32)[..., 0] != largest,
    }
    if exact:
        found['wrong_bits'] = np.any(values.view(np.uint32) != reference(x, k)[0], axis=-1)
    return {'rows': int(np.prod(x.shape[:-1])), **{name: int(np.count_nonzero(wrong)) for name, wrong in found.items()}, 'any': int(np.count_nonzero(np.logical_or.reduce(list(found.values()))))}

def kernel_program(shape: tuple[int, ...], k: int) -> asm.Compiled:
    """正确性核对用的 Pallas 程序：kernel 自己把输入读进 TC VMEM，调用 approx_max_k，再把结果写出。"""
    out_shape = (*shape[:-1], k)

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
            name='approx_max_k',
            compiler_params=shell.PARAMS,
        )
        def kernel(x_hbm: Ref, v_hbm: Ref, i_hbm: Ref, x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            v_vmem[...], i_vmem[...] = approx(k)(x_vmem[...])
            pltpu.async_copy(v_vmem, v_hbm, sem).wait()
            pltpu.async_copy(i_vmem, i_hbm, sem).wait()

        return kernel(x)

    return asm.compile(function, np.zeros(shape, np.float32), mesh=shell.MESH)

def check(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(6610)
    records = []
    for shape, k in CHECK_CASES:
        data = compare.distributions(rng, shape, len(shape) - 1, k, args.batches)
        for name in ('xla', 'official', 'stage2', 'ours'):
            if name == 'xla':
                compiled = asm.compile(approx(k), np.zeros(shape, np.float32))
            else:
                with variant(name):
                    compiled = kernel_program(shape, k)
            totals: dict[str, dict[str, int]] = {}
            for distribution, x in data:
                wrong = errors(x, compiled(x), k, exact=name != 'xla')
                total = totals.setdefault(distribution, dict.fromkeys(wrong, 0))
                for field, number in wrong.items():
                    total[field] += number
            summed = {field: sum(total[field] for total in totals.values()) for field in next(iter(totals.values()))}
            records.append({'shape': shape, 'k': k, 'bins': num_bins(shape[-1], k, RECALL), 'variant': name, **summed, 'distributions': totals})
            print(json.dumps({key: value for key, value in records[-1].items() if key != 'distributions'}), flush=True)
            (args.output / 'approx-correctness.json').write_text(json.dumps(records, indent=2) + '\n')

def check_binned(result: tuple[jax.Array, jax.Array], x: np.ndarray, k: int, axis: int) -> None:
    """计时输入没有并列，Pallas 的三种实现都应与参照的值和下标完全相同。"""
    values, indices = reference(x, k)
    np.testing.assert_array_equal(np.asarray(result[0]).view(np.uint32), values)
    np.testing.assert_array_equal(np.asarray(result[1]), indices)

def check_valid(result: tuple[jax.Array, jax.Array], x: np.ndarray, k: int, axis: int) -> None:
    """原生 XLA 的分桶不同，只核对合法性：下标合法、互不相同、与值配对，值降序，全局最大值排第一。"""
    assert not any(errors(x, result, k, exact=False)[field] for field in ('out_of_range', 'duplicate', 'unpaired_bits', 'max_missing'))
    assert np.all(np.diff(keys(np.asarray(result[0])), axis=-1) <= 0)

def time(args: argparse.Namespace) -> None:
    clock = asm.KernelClock(num_cores=1)
    records = []
    for shape, k in TIME_CASES:
        for name in ('xla', 'official', 'stage2', 'ours'):
            top_k = approx(k) if name == 'xla' else shell.pallas_top_k(k, 1, body=approx(k))
            with variant(name):
                # shell.measure 编译时会按 using 设置 _top_k_impl，所以第二阶段的替换要经由它传入。
                record = shell.measure(clock, top_k, shape, k, 1, using=total_order.fast if name == 'stage2' else shell.OFFICIAL, check=check_valid if name == 'xla' else check_binned, label=f'{shape[0]}x{shape[1]}-k{k}-{name}')
            records.append({'variant': name, 'bins': num_bins(shape[-1], k, RECALL), **record})
            (args.output / 'approx.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'time'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/top-k-paper/approx'))
    parser.add_argument('--batches', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {'check': check, 'time': time}[args.mode](args)

if __name__ == '__main__':
    main()
