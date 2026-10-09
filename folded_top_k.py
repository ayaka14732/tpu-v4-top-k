"""折叠后数秩的 top-k，用普通的 Pallas 加 asm_intrinsics 的四个占位写成：形状、寄存器和排程都由编译器决定。

folded_top_k 对 f32[8n, 128] 沿最后一维取前 k 名（k 是 8 的倍数，不超过 32），结果与稳定排序逐位相同。
直接运行：参数为 check、time 或 stages（分段计时），以及行数、k。
"""
from pathlib import Path
import json
import os
import sys
import chip
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from tpuasm import tools as asm
import asm_intrinsics as ai
import reschedule
# 环境变量 VLIW_REORDER=<窗口> 时，在编译流程里面接管 LLO 的 VLIW 排程。
import vliw_reorder

INT_MIN = -(1 << 31)
# 实验开关：二分的第二步也用按索引读（少一次读）；k = 8 时值与下标并行送回（省掉最后的 vperm）；让阈值依赖旋转的结果。
GATHER_TWICE = os.environ.get('GATHER_TWICE', '0') == '1'
PARALLEL_VALUES = os.environ.get('PARALLEL_VALUES', '1') == '1'
LATE_THRESHOLD = os.environ.get('LATE_THRESHOLD', '0') == '1'
# 各组的计数怎样加起来：默认是一条链，tree 是先各自加好再成对相加。
SUM = os.environ.get('SUM', '')
PER_TILE = os.environ.get('PER_TILE', '0') == '1'
# 改写之后再用 reschedule.py 把计算段重新排程。
RESCHEDULE = os.environ.get('RESCHEDULE', '0') == '1'
# 按固定模式重排不用占位，改用 tpu.gather。
FRONT_DOOR_SHUFFLE = os.environ.get('FRONT_DOOR_SHUFFLE', '0') == '1'
shuffle = ai.sublane_shuffle_op if FRONT_DOOR_SHUFFLE else ai.sublane_shuffle
# 按索引读不用占位，直接写 jnp.take_along_axis，由打了补丁的 libtpu 编译成 vsetiar 加 vld.iar。
FRONT_DOOR_GATHER = os.environ.get('FRONT_DOOR_GATHER', '0') == '1'
# 折叠不用占位，由同一个补丁编译成 vsxpose。
FRONT_DOOR_FOLD = os.environ.get('FRONT_DOOR_FOLD', '0') == '1'
if FRONT_DOOR_GATHER or FRONT_DOOR_FOLD:
    import libtpu_patch
    libtpu_patch.allow_sublane_gather()
    # 再让排程器知道 IAR 是有状态的寄存器、vsetiar 之后要等几个周期，并且不再把 vsetiar 当作内存屏障。
    # TEACH_IAR=0 关掉；ANY_STORE_GAP、IAR_BARRIER 用来复现正文 8.3 节的对照。
    if os.environ.get('TEACH_IAR', '1') == '1':
        libtpu_patch.teach_iar(any_store_gap=int(os.environ.get('ANY_STORE_GAP', '0')), barrier=os.environ.get('IAR_BARRIER', '0') == '1')

# 四个 intrinsic 全部走正门（front_door.py）：不用占位，不改写 executable。
FRONT_DOOR_ALL = os.environ.get('FRONT_DOOR_ALL', '0') == '1'
if FRONT_DOOR_ALL:
    import front_door
    front_door.install()
    fold, shuffle, gather, scatter = front_door.fold, front_door.sublane_shuffle, front_door.sublane_gather, front_door.sublane_scatter
else:
    fold = ai.fold_op if FRONT_DOOR_FOLD else ai.fold
    gather = (lambda x, offset: jnp.take_along_axis(x, offset + (lax.broadcasted_iota(jnp.int32, x.shape, 0) & 7), axis=0)) if FRONT_DOOR_GATHER else ai.sublane_gather
    scatter = ai.sublane_scatter

def pattern(shift: int) -> int:
    """sublane_shuffle 的模式：第 s 个子通道读第 (s - shift) mod 8 行。"""
    return sum(((s - shift) % 8) << (4 * s) for s in range(8))

def folded_top_k(x: jax.Array, k: int, *, stop: str = '') -> tuple[jax.Array, jax.Array]:
    """stop 为 fold、rank、column、total、index 时提前返回到那一步为止的中间结果（裁成输出的形状），用来分段计时。"""
    if PER_TILE and x.shape[0] > 8:
        # 各行互不相干：一个 TC VREG 一个 TC VREG 地写，XLU 上各个 TC VREG 的提交就不会交错成一条链。
        parts = [folded_top_k(x[start:start + 8], k, stop=stop) for start in range(0, x.shape[0], 8)]
        return jnp.concatenate([values for values, _ in parts]), jnp.concatenate([indices for _, indices in parts])
    assert x.ndim == 2 and x.shape[1] == 128 and x.shape[0] % 8 == 0 and k % 8 == 0 and k <= 32, (x.shape, k)
    shape = x.shape
    lane = lax.broadcasted_iota(jnp.int32, shape, 1)
    sublane = lax.broadcasted_iota(jnp.int32, shape, 0) & 7
    group = lane & 120
    bits = lax.bitcast_convert_type(x, jnp.int32)
    # 保序的整数键；折叠后第 r 行的元素在通道 8g + r 上，每列 8 个子通道。
    key = fold(jnp.where(bits < 0, bits ^ 0x7fffffff, bits))
    early = lambda a, b: (lax.bitcast_convert_type(a, jnp.float32)[:, :k], b[:, :k])
    if stop == 'fold':
        return early(key, key)
    below = key - 1
    # 每个元素在自己那一列中的名次（同键时原通道号小的在前），再按名次把整列排成降序。
    rank = jnp.where(key == INT_MIN, sublane, 0)
    for e in range(1, 8):
        other = shuffle(key, pattern(e))
        rank = rank + jnp.where(other > jnp.where(sublane >= e, below, key), 1, 0)
    if stop == 'rank':
        return early(key, rank)
    column = scatter(key, rank - sublane)
    origin = scatter(sublane, rank - sublane)
    if stop == 'column':
        return early(column, origin)
    below = column - 1
    # 其余 15 组逐个旋转过来，在对方排好的列里二分，数出排在自己前面的个数。
    rows = {c: c - sublane for c in (0, 1, 2, 4, 5, 6)}
    total = sublane * 16
    counts: list[jax.Array] = []
    for d in range(1, 16):
        other = jnp.roll(column, 8 * d, axis=1)
        if LATE_THRESHOLD:
            # 对方的列是降序的，第 3 行不小于第 7 行恒成立，但编译器不知道：让阈值依赖旋转的结果，它就没法把 15 个阈值提前算好再溢出。
            seventh, third = shuffle(other, 0x77777777), shuffle(other, 0x33333333)
            threshold = jnp.where((group >= 8 * d) & (third >= seventh), below, column)
            whole = seventh > threshold
            half = third > threshold
        else:
            threshold = jnp.where(group >= 8 * d, below, column)
            whole = shuffle(other, 0x77777777) > threshold
            half = shuffle(other, 0x33333333) > threshold
        if GATHER_TWICE:
            quarter = gather(other, jnp.where(half, rows[5], rows[1])) > threshold
        else:
            quarter = jnp.where(half, shuffle(other, 0x55555555), shuffle(other, 0x11111111)) > threshold
        # 第三次要看的行是 0、2、4、6 之一；累加的是行偏移（行号减子通道号），15 次多减的 15 个子通道号最后一次补回。
        offset = jnp.where(half, jnp.where(quarter, rows[6], rows[4]), jnp.where(quarter, rows[2], rows[0]))
        last = gather(other, offset) > threshold
        if SUM == 'tree':
            # 每组的计数各自加好，最后成对相加：累加不再是一条 45 步的链，各组的中间结果也不用等着它。
            counts.append(offset + (jnp.where(last, 1, 0) + jnp.where(whole, 1, 0)))
        else:
            total = total + offset + jnp.where(last, 1, 0) + jnp.where(whole, 1, 0)
    while len(counts) > 1:
        counts = [a + b for a, b in zip(counts[::2], counts[1::2])] + counts[len(counts) & ~1:]
    if counts:
        total = total + counts[0]
    # 最小键减一回绕后，前面 g 个组的比较全部落空：补上这 8g 个。
    total = jnp.where(column == INT_MIN, total + group, total)
    if stop == 'total':
        return early(column, total + origin)
    position = group + origin
    mine = (lane & 7) == sublane
    parallel = PARALLEL_VALUES and k == 8
    if parallel:
        # 键的低 31 位换成一个有限的正常浮点数，符号位随下标带走：这样每个名次的值可以用一次 vmax.xlane 原样带回原布局。
        payload = column & 0x3fffffff
        proxy = jnp.where((column & 0x40000000) != 0, payload + 0x00800000, (0x407fffff - payload) | INT_MIN)
        position = position | jnp.where(column < 0, 128, 0)
    index = jnp.zeros(shape, jnp.int32)
    carried = jnp.zeros(shape, jnp.int32)
    for part in range(k // 8):
        # 名次在这 8 个之内的元素把原通道号写到自己那一列的第（名次）行，其余元素落到 8 行之后被丢弃。
        inside = (total >= 8 * part) & (total < 8 * part + 8)
        target = jnp.where(inside, total - 8 * part - sublane, 8)
        ranked = scatter(position, target)
        for start in (0, 3, 6):
            # 三个名次一个字：各自那一行广播出来，移到各自的字节上相加；只留通道号模 8 等于子通道号的那一类，沿通道求和。
            word = jnp.zeros(shape, jnp.int32)
            for byte, j in enumerate(range(start, min(start + 3, 8))):
                word = word + (shuffle(ranked, 0x11111111 * j) << (8 * byte))
            summed = jnp.sum(jnp.where(mine, word, 0).astype(jnp.float32), axis=1, keepdims=True).astype(jnp.int32)
            # 第 8·part + start 起的三个通道各取这个字的一个字节。
            inside = (lane >= 8 * part + start) & (lane < 8 * part + min(start + 3, 8))
            index = jnp.where(inside, (summed >> ((lane - (8 * part + start)) * 8)) & 255, index)
        if parallel:
            valued = scatter(proxy, target, fill=-8388608)
            for j in range(8):
                row = lax.bitcast_convert_type(shuffle(valued, 0x11111111 * j), jnp.float32)
                best = jnp.max(jnp.where(mine, row, -jnp.inf), axis=1, keepdims=True)
                carried = jnp.where(lane == j, lax.bitcast_convert_type(best + jnp.zeros(shape, jnp.float32), jnp.int32), carried)
    if stop == 'index':
        return early(index, index)
    if parallel:
        negative = index >= 128
        index = index & 127
        magnitude = carried & 0x7fffffff
        low = jnp.where(carried < 0, 0x407fffff - magnitude, (magnitude - 0x00800000) | 0x40000000)
        restored = jnp.where(negative, (low | INT_MIN) ^ 0x7fffffff, low)
        return lax.bitcast_convert_type(restored, jnp.float32)[:, :k], index[:, :k]
    values = jnp.take_along_axis(x, index, axis=1, mode='promise_in_bounds')
    return values[:, :k], index[:, :k]

def build(rows: int, k: int, stop: str = '') -> tuple[asm.Compiled, bytes, dict[str, int]]:
    """完整的程序：输入 DMA、top-k、两次同时发起的输出 DMA。返回编译结果（装载模板）、改写后的 executable 和改写统计。"""
    mesh = jax.make_mesh((1,), ('device',))

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def function(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        @pl.kernel(
            out_type=(jax.ShapeDtypeStruct((rows, k), jnp.float32), jax.ShapeDtypeStruct((rows, k), jnp.int32)),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            scratch_types=(pltpu.VMEM((rows, 128), jnp.float32), pltpu.VMEM((rows, k), jnp.float32), pltpu.VMEM((rows, k), jnp.int32), pltpu.VMEM((48 * rows, 128), jnp.int32), pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            name='folded_top_k',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, v_hbm: Ref, i_hbm: Ref, x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, work: Ref, sem: Ref, sem2: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            if not FRONT_DOOR_ALL:
                ai.scratch(work)
            v_vmem[...], i_vmem[...] = folded_top_k(x_vmem[...], k, stop=stop)
            first = pltpu.async_copy(v_vmem, v_hbm, sem)
            second = pltpu.async_copy(i_vmem, i_hbm, sem2)
            first.wait()
            second.wait()

        return kernel(x)

    ai.SITES.clear()
    compiled = asm.compile(function, np.zeros((rows, 128), np.float32), mesh=mesh)
    # 全部走正门时没有占位，编译出来的 executable 就是最终的程序。
    serialized, counts = (asm.serialize(compiled), dict(front_door.COUNTS)) if FRONT_DOOR_ALL else ai.rewrite(compiled)
    if vliw_reorder.STATISTICS:
        counts = {**counts, 'vliw reorder': max(vliw_reorder.STATISTICS, key=lambda entry: entry['nodes'])}
        vliw_reorder.STATISTICS.clear()
    if RESCHEDULE:
        serialized, statistics = reschedule.reschedule(serialized)
        counts = {**counts, **statistics}
    return compiled, serialized, counts

def reference(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    bits = x.view(np.uint32)
    key = np.where(bits.view(np.int32) < 0, bits ^ 0x7fffffff, bits).view(np.int32)
    order = np.argsort(-key.astype(np.int64), axis=1, kind='stable')[:, :k]
    return np.take_along_axis(bits, order, axis=1), order

def cases(rng: np.random.Generator, rows: int, k: int) -> list[tuple[str, np.ndarray]]:
    shape = (rows, 128)
    special = np.array([0xffffffff, 0xffc00001, 0xff800000, 0x80000000, 0, 1, 0x7f800000, 0x7fc00001, 0x7fffffff], np.uint32)
    result = []
    for _ in range(8):
        result += [('normal', rng.normal(size=shape).astype(np.float32)), ('bits', rng.integers(0, 2**32, shape, np.uint32).view(np.float32))]
        result += [('ties', rng.integers(-3, 4, shape).astype(np.float32)), ('special', rng.choice(special, shape).view(np.float32))]
    result += [('constant', np.full(shape, word, np.uint32).view(np.float32)) for word in special]
    for count in range(k + 2):
        x = -np.abs(rng.normal(size=shape).astype(np.float32))
        x[:, rng.permutation(128)[:count]] *= -1
        result.append(('phase', x))
    sparse = np.full(shape, -np.inf, np.float32)
    sparse[:, [0, 1, 127]] = [3.0, 1.0, 2.0]
    return result + [('sparse', sparse)]

def compute_cycles(compiled: asm.Compiled, serialized: bytes, rows: int, rng: np.random.Generator) -> list[int]:
    """输入 DMA 的等待之后到第一次输出 DMA 的等待之前的周期数。"""
    clock = asm.KernelClock(num_cores=1)
    waits = asm.find_bundles(serialized, 'vwait')
    timed = asm.load(asm.insert_bundles(serialized, {waits[0] + 1: asm.clock_read(0), waits[1]: asm.clock_read(1)}), compiled)
    samples = []
    for _ in range(11):
        jax.block_until_ready(timed(jax.device_put(rng.normal(size=(rows, 128)).astype(np.float32))))
        first, last = clock.read(2)[0].astype(np.int64)
        samples.append(int(last - first - asm.CLOCK_OVERHEAD))
    return sorted(set(samples[3:]))

def main() -> None:
    mode, rows, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(6502)
    if mode == 'stages':
        for stop in ('fold', 'rank', 'column', 'total', 'index', ''):
            compiled, serialized, counts = build(rows, k, stop)
            print(json.dumps({'through': stop or 'values', 'compute cycles': compute_cycles(compiled, serialized, rows, rng)}), flush=True)
        return
    compiled, serialized, counts = build(rows, k)
    function = asm.load(serialized, compiled)
    print(json.dumps({'rows': rows, 'k': k, 'rewrite': counts}), flush=True)
    wrong: dict[str, list[int]] = {}
    for name, x in cases(rng, rows, k):
        values, index = (np.asarray(item) for item in function(x))
        want, order = reference(x, k)
        entry = wrong.setdefault(name, [0, 0])
        entry[0] += rows
        entry[1] += int(np.count_nonzero(np.any(values.view(np.uint32) != want, axis=1) | np.any(index != order, axis=1)))
    print(json.dumps({'rows checked and wrong, by distribution': wrong}), flush=True)
    if mode == 'time':
        print(json.dumps({'compute cycles': compute_cycles(compiled, serialized, rows, rng)}), flush=True)

if __name__ == '__main__':
    main()
