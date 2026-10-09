"""LCC 的计数频率，以主机的 CLOCK_MONOTONIC 为参照，用两种互相独立的办法测。

单芯片会话中这颗芯片自己就是时间源，GTC 由同一个时钟驱动，恰好是 LCC 的 32/3 倍，不能给出独立的参照，所以参照只能是主机时钟。第 3 部分核对这个倍数。

1. 时间轴。LCC 在两次运行之间也持续计数。一个短 kernel 在约 20 秒内运行 200 次，每次在 kernel 两端读 LCC，主机在调用前后各读一次 CLOCK_MONOTONIC。第一次读数与最后一次读数的真实时刻都夹在主机读数之间，所以整段 LCC 差除以主机区间的上下界，就得到不依赖派发延迟的频率包络。
2. 差值。同一个 kernel 分别循环 10⁷ 次和 10⁸ 次，各运行 8 次取主机耗时的中位数；周期数之差除以耗时之差，派发、完成通知等固定开销在差值中抵消。

3. GTC。LCC、GTC、向量一侧停 H 个周期、sfence、GTC、LCC；两次 GTC 之间是 ΔL − 2 个周期。

不直接用一次调用的主机耗时去除周期数：短调用的耗时主要是派发和唤醒，会被误当作频率。
"""
import time

import chip
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tpuasm import tools as asm
from tpuasm.tools import bundle, read_gtc, read_lcc

MESH = jax.make_mesh((1,), ('device',))
TIMELINE_LOOPS = 1_000
TIMELINE_CALLS = 200
TIMELINE_GAP = 0.1
DIFFERENCE_LOOPS = (10_000_000, 100_000_000)
GTC_DELAYS = (1_000, 1_000_000, 100_000_000)

def loop_program(n: int) -> asm.Compiled:
    """每次迭代读一个 tile、加一、写回，周期数与 n 成正比。"""
    @jax.shard_map(
        mesh=MESH,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), jnp.int32),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            scratch_types=(pltpu.VMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA),
            name='clock_rate',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: jax.Ref, o_hbm: jax.Ref, work: jax.Ref, sem: jax.Ref) -> None:
            pltpu.async_copy(x_hbm, work, sem).wait()

            @pl.loop(0, n)
            def _(_: jax.Array) -> None:
                work[...] = work[...] + 1

            pltpu.async_copy(work, o_hbm, sem).wait()

        return kernel(x)

    return asm.compile(function, np.zeros((8, 128), np.int32), mesh=MESH)

def timed(clock: asm.KernelClock, n: int) -> asm.Compiled:
    """在 kernel 的起始标记之前、结束标记之后各读一次 LCC。"""
    compiled = loop_program(n)
    start, end = min(span for number, span in asm.hlo_bounds(asm.serialize(compiled)).items() if number != asm.MODULE)
    return clock.instrument(compiled, [start, end + 1])

def main() -> None:
    clock = asm.KernelClock(num_cores=1)
    x = jax.device_put(np.zeros((8, 128), np.int32))
    # 1. 时间轴。
    program = timed(clock, TIMELINE_LOOPS)
    jax.block_until_ready(program(x))
    samples = []
    for _ in range(TIMELINE_CALLS):
        host_start = time.monotonic_ns()
        jax.block_until_ready(program(x))
        host_stop = time.monotonic_ns()
        begin, end = (int(value) for value in clock.read(2)[0])
        samples.append((host_start, host_stop, begin, end))
        time.sleep(TIMELINE_GAP)
    first, last = samples[0], samples[-1]
    span = last[3] - first[2]
    shortest, longest = last[0] - first[1], last[1] - first[0]
    middle = np.array([((start + stop) / 2, (begin + end) / 2) for start, stop, begin, end in samples])
    fit = float(np.polyfit(middle[:, 0], middle[:, 1], 1)[0])
    print(f'timeline: {TIMELINE_CALLS} calls over {longest / 1e9:.3f} s, {span} cycles; LCC rate {span / longest * 1e3:.4f} to {span / shortest * 1e3:.4f} MHz (envelope width {(longest - shortest) / longest * 1e6:.1f} ppm), fit of midpoints {fit * 1e3:.4f} MHz', flush=True)
    # 2. 差值。
    measured = []
    for n in DIFFERENCE_LOOPS:
        program = timed(clock, n)
        assert int(jax.block_until_ready(program(x))[0, 0]) == n
        cycles = int(clock.read(2)[0][1] - clock.read(2)[0][0]) - asm.CLOCK_OVERHEAD
        seconds = []
        for _ in range(8):
            host_start = time.monotonic_ns()
            jax.block_until_ready(program(x))
            seconds.append((time.monotonic_ns() - host_start) / 1e9)
        measured.append((cycles, float(np.median(seconds))))
        print(f'n = {n}: {cycles} cycles, median host time {measured[-1][1] * 1e3:.3f} ms over 8 calls (range {min(seconds) * 1e3:.3f} to {max(seconds) * 1e3:.3f} ms)', flush=True)
    (c0, s0), (c1, s1) = measured
    print(f'difference: LCC rate {(c1 - c0) / (s1 - s0) / 1e6:.4f} MHz')
    # 3. GTC 与 LCC。
    probe = asm.LccProbe()
    body = read_lcc(20) + read_gtc(21) + bundle('misc: vdelay s24') + bundle('s0: sfence') + read_gtc(22) + read_lcc(23)
    for delay in GTC_DELAYS:
        readings = probe.run_raw(body, 4, repeats=8, setup=bundle(f's0: simm.s32 s24, {delay}'))[:, 0].astype(np.int64)
        lcc = sorted({int(row[3] - row[0]) - 2 for row in readings})
        gtc = sorted({int(row[2] - row[1]) for row in readings})
        print(f'GTC: delay {delay}, cycles between the two GTC reads {lcc}, GTC difference {gtc}, ratio {gtc[0] / lcc[0]:.7f} (32/3 = {32 / 3:.7f})', flush=True)

if __name__ == '__main__':
    main()
