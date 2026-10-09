"""把一个 top-k 实现放进起点和终点都在 TC VMEM 的外壳，并测这段连续区间的周期数。

外壳对原生 XLA 和 Pallas kernel 相同：输入先 Pin 到 TC VMEM 并约束布局，两个结果同样 Pin 并约束布局，再共同经过 optimization_barrier 后返回。
计时区间从输入的 Unpin 结束到较晚的那个输出 Unpin 结束，只在两端各读一次 LCC；区间内不允许出现进出 HBM 的 DMA。
"""
import chip
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import re

import jax
from jax import Ref
from jax._src.pallas.mosaic import lowering
from jax._src.state import primitives as state
from jax.experimental import pallas as pl
from jax.experimental.layout import Layout, with_layout_constraint
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tpuasm import tools as asm

import pin_propagation

TopK = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
Implementation = Callable[..., tuple[jax.Array, jax.Array]]
MESH = jax.make_mesh((1,), ('device',))
TC = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
PARAMS = pltpu.CompilerParams(
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)
# Pallas kernel 在 TC VMEM 中的二维 f32 数组所用的布局：8 个子通道乘 128 个通道的 tile，行优先。Layout 的参数是 major-to-minor 顺序，对应 HLO 中的 {1,0:T(8,128)}。
ROW = Layout(major_to_minor=(0, 1), tiling=((8, 128),))
OFFICIAL = lowering._top_k_impl

def pinned_row(x: jax.Array) -> jax.Array:
    """把一个值固定在 TC VMEM（HLO layout 中的 S(1)），并约束为行优先、(8,128) tile 的布局。布局约束必须加在 Unpin 之后，否则 XLA 会在 Pin 与 Unpin 之间插入转置。"""
    return with_layout_constraint(state.unpin(state.pin(x, to='vmem')), ROW)

def xla_top_k(k: int, axis: int, stable: bool = False) -> TopK:
    return lambda x: jax.lax.top_k(x, k, axis=axis, is_stable=stable)

def pallas_top_k(k: int, axis: int, body: Callable[..., tuple[jax.Array, jax.Array]] | None = None, scratch_rows: int = 0) -> TopK:
    """操作数和结果都在 TC VMEM 的 Pallas kernel，没有 DMA 和信号量。默认调用 jax.lax.top_k，由编译时的 _top_k_impl 展开；body 给出时改用它。

    scratch_rows 不为零时 kernel 另有一块 int32 的 TC VMEM 工作区，它的 Ref 作为第二个参数传给 body。
    """
    def top_k(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        out_shape = tuple(k if dimension == axis else size for dimension, size in enumerate(x.shape))

        @pl.kernel(
            out_type=(pltpu.VMEM(out_shape, jnp.float32), pltpu.VMEM(out_shape, jnp.int32)),
            mesh=TC,
            scratch_types=(pltpu.VMEM((scratch_rows, 128), jnp.int32),) if scratch_rows else (),
            name='top_k',
            compiler_params=PARAMS,
        )
        def kernel(x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, *scratch: Ref) -> None:
            v_vmem[...], i_vmem[...] = body(x_vmem[...], *scratch) if body else jax.lax.top_k(x_vmem[...], k, axis=axis, is_stable=False)

        return kernel(pltpu.with_memory_space_constraint(x, pltpu.VMEM))

    return top_k

def program(top_k: TopK) -> TopK:
    """外壳：两个结果共同经过 barrier，程序向 HBM 写出之前它们都已经在 TC VMEM 中。"""
    def function(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        v, i = top_k(pinned_row(x))
        return jax.lax.optimization_barrier((pinned_row(v), pinned_row(i)))

    return jax.shard_map(function, mesh=MESH, in_specs=P(), out_specs=(P(), P()), check_vma=False)

@contextmanager
def implementation(function: Implementation) -> Iterator[None]:
    """编译期间把 Pallas 对 jax.lax.top_k 的展开换成 function，kernel 仍走公开的编译路径。"""
    lowering._top_k_impl = function
    try:
        yield
    finally:
        lowering._top_k_impl = OFFICIAL

def compile(top_k: TopK, shape: tuple[int, ...], using: Implementation = OFFICIAL) -> tuple[asm.Compiled, str]:
    """编译外壳中的程序。原版编译器因 Pin 的着色没有传播到 fusion 内部而失败时，装上 pin_propagation 的修复重试；返回值的第二项是原版编译器的报错（没有则为空）。"""
    with implementation(using):
        try:
            return asm.compile(program(top_k), np.zeros(shape, np.float32), mesh=MESH), ''
        except jax.errors.JaxRuntimeError as error:
            message = str(error).split(', metadata=')[0].split('Suspected Python')[0].strip()
            if 'Fused computation shape' not in message:
                raise
        with pin_propagation.propagate_after_pin():
            return asm.compile(program(top_k), np.zeros(shape, np.float32), mesh=MESH), message

def instruction_names(listing: str, bounds: dict[int, tuple[int, int]]) -> dict[int, str]:
    """每个 vtrace 标记对应的 HLO 指令名：标记区间内第一条 entry 或 function 注释；没有注释的用序号。"""
    notes, pc = [], 0
    for line in listing.splitlines():
        if line.startswith('{'):
            pc += 1
        elif line.startswith('# entry bundle: %'):
            notes.append((pc, line.split('%')[1].split(' ')[0]))
        elif line.startswith('# function '):
            notes.append((pc, re.search(r'\[symbol \d+, ([^\]]+)\]', line).group(1)))
    return {number: next((name for at, name in notes if start <= at <= end), f'#{number}') for number, (start, end) in bounds.items()}

def kernel_span(serialized: bytes) -> tuple[int, int]:
    """Pallas kernel 的起止标记所在的 bundle 编号。"""
    bounds = {number: span for number, span in asm.hlo_bounds(serialized).items() if number != asm.MODULE}
    names = instruction_names(asm.full_listing(serialized), bounds)
    span, = [bounds[number] for number, name in names.items() if name.startswith('top_k')]
    return span

def boundary(compiled: asm.Compiled, serialized: bytes) -> tuple[int, int, dict[str, object]]:
    """计时区间的两个 bundle 编号，以及核对用的证据：三个端点的 layout、区间内有代码的 HLO 指令、区间内的 DMA。serialized 是实际运行的 executable，可以是改写过的。"""
    text = compiled.as_text()
    lines = text[text.index('\nENTRY'):].split('\n}')[0].splitlines()[2:]
    name = lambda line: re.search(r'%([\w.\-]+) =', line).group(1)
    unpins = [line for line in lines if 'custom_call_target="Unpin"' in line]
    assert len(unpins) == 3, unpins
    layouts = [line.split(' = ', 1)[1].split(' custom-call(', 1)[0] for line in unpins]
    # 三个端点都必须是行优先、(8,128) tile 布局的 TC VMEM 数组。
    assert all('{1,0:T(8,128)S(1)}' in layout for layout in layouts), layouts
    listing = asm.full_listing(serialized)
    bounds = {number: span for number, span in asm.hlo_bounds(serialized).items() if number != asm.MODULE}
    names = instruction_names(listing, bounds)
    ops = {names[number]: span for number, span in bounds.items()}
    first = ops[name(unpins[0])][1] + 1
    last = max(ops[name(line)][1] for line in unpins[1:]) + 1
    # 属于 top_k 的指令（包括不依赖输入的 iota）不能被调度到区间之外。
    for line in lines:
        if '/top_k' in line and (match := re.search(r'%([\w.\-]+) =', line)) and match[1] in ops:
            start, end = ops[match[1]]
            assert first <= start <= end < last, (match[1], start, end, first, last)
    moves, pc = [], -1
    for line in listing.splitlines():
        if line.startswith('{'):
            pc += 1
        if first <= pc < last:
            moves += [f'{source}->{destination}' for destination, source in re.findall(r'dma\.\w+ \[(\w+):[^\]]*\], \[(\w+):', line)]
    assert not any('hbm' in move for move in moves), moves
    inside = [op for op, (start, end) in ops.items() if first <= start <= end < last and not op.startswith(('pin.', 'unpin.'))]
    return first, last, {'layouts': layouts, 'ops': inside, 'dma': sorted(set(moves))}

def permutations(shape: tuple[int, ...], axis: int, seed: int) -> np.ndarray:
    """每条排序轴上是 0 到 n - 1 的随机排列：有限、没有并列，下标的正确答案唯一。"""
    rng = np.random.default_rng(seed)
    others = tuple(size for dimension, size in enumerate(shape) if dimension != axis)
    rows = np.stack([rng.permutation(shape[axis]) for _ in range(int(np.prod(others)))]).astype(np.float32)
    return np.ascontiguousarray(np.moveaxis(rows.reshape(*others, shape[axis]), -1, axis))

def verify(result: tuple[jax.Array, jax.Array], x: np.ndarray, k: int, axis: int) -> None:
    """值的位型序列与按可逆整数键降序排序的结果相同，下标合法、互不相同并与值配对。"""
    values, indices = (np.asarray(item) for item in result)
    bits = x.view(np.uint32)
    key = np.where(bits.view(np.int32) < 0, bits ^ 0x7fffffff, bits).view(np.int32).astype(np.int64)
    order = np.take(np.argsort(-key, axis=axis, kind='stable'), np.arange(k), axis=axis)
    np.testing.assert_array_equal(values.view(np.uint32), np.take_along_axis(bits, order, axis=axis))
    assert indices.min() >= 0 and indices.max() < x.shape[axis]
    np.testing.assert_array_equal(np.take_along_axis(bits, indices, axis=axis), values.view(np.uint32))
    assert not np.any(np.diff(np.sort(indices, axis=axis), axis=axis) == 0)

def measure(clock: asm.KernelClock, top_k: TopK, shape: tuple[int, ...], k: int, axis: int, *, using: Implementation = OFFICIAL, inputs: list[np.ndarray] | None = None, output: Path | None = None, label: str = '', transform: Callable[[asm.Compiled], bytes] | None = None, check: Callable[[tuple[jax.Array, jax.Array], np.ndarray, int, int], None] | None = None) -> dict[str, object]:
    """编译、核对边界、核对全部输出，再测周期数：8 份输入各运行一次，舍弃前两次，报告后六次的中位数与范围。

    transform 给出时，它把编译结果改写成另一个 executable（换掉占位、插入手写片段或重新排程），边界核对、数值核对和计时都在改写之后的程序上进行。check 给出时用它代替 verify 核对输出，用于语义不是精确 top-k 的程序。
    """
    check = check or verify
    compiled, error = compile(top_k, shape, using)
    serialized = transform(compiled) if transform else asm.serialize(compiled)
    plain = asm.load(serialized, compiled) if transform else compiled
    first, last, evidence = boundary(compiled, serialized)
    if output is not None:
        (output / f'{label}.hlo').write_text(compiled.as_text())
        (output / f'{label}.tpuasm').write_text(asm.full_listing(serialized))
    xs = inputs or [permutations(shape, axis, seed) for seed in range(8)]
    for x in xs:
        check(plain(jax.device_put(x)), x, k, axis)
    timed = asm.load(asm.insert_bundles(serialized, {first: asm.clock_read(0), last: asm.clock_read(1)}), compiled)
    samples = []
    for x in xs:
        result = timed(jax.device_put(x))
        jax.block_until_ready(result)
        reads = clock.read(2)[0].astype(np.int64)
        samples.append(int(reads[1] - reads[0] - asm.CLOCK_OVERHEAD))
        check(result, x, k, axis)
    record = {'shape': shape, 'k': k, 'axis': axis, 'median': float(np.median(samples[2:])), 'min': min(samples[2:]), 'max': max(samples[2:]), 'samples': samples, 'points': [first, last], **evidence}
    if error:
        record['repaired'] = error
    print(json.dumps({'label': label, **{key: record[key] for key in ('shape', 'k', 'axis', 'median', 'min', 'max', 'ops', 'dma')}, 'repaired': bool(error)}, ensure_ascii=False), flush=True)
    return record
