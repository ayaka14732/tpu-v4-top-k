"""逐点测量重排后的 top-k：计算段的第 p 个 bundle 实际在第几个周期发射，与 reschedule.py 的时序模型比较。

模型漏掉的等待会表现为“实测减模型”在某个 bundle 上突然变大。每个测点只是在同一个 executable 里换一个读时钟的位置，不用重新编译。
用法：issue_profile.py <行数> <k> <步长> [起点 终点]；环境变量与 folded_top_k.py 相同。
"""
import os
import sys
os.environ['RESCHEDULE'] = '1'
import folded_top_k
import jax
import numpy as np
import reschedule
from tpuasm import tools as asm

def main() -> None:
    rows, k, step = (int(argument) for argument in sys.argv[1:4])
    compiled, serialized, counts = folded_top_k.build(rows, k)
    issued = reschedule.ISSUED[counts['window'], bool(counts['hold stores'])]
    waits = asm.find_bundles(serialized, 'vwait')
    clock = asm.KernelClock(num_cores=1)
    x = jax.device_put(np.random.default_rng(0).normal(size=(rows, 128)).astype(np.float32))
    points = range(int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 4 else [*range(4, counts['bundles after'], step), counts['bundles after'] - 1]
    previous = 0
    print(f'$ issue_profile.py {" ".join(sys.argv[1:])}')
    for point in points:
        timed = asm.load(asm.insert_bundles(serialized, {waits[0] + 1: asm.clock_read(0), waits[0] + 1 + point: asm.clock_read(1)}), compiled)
        for _ in range(4):
            jax.block_until_ready(timed(x))
            first, last = clock.read(2)[0].astype(np.int64)
        measured = int(last - first - asm.CLOCK_OVERHEAD)
        drift = measured - issued[point]
        text = ' | '.join(part.strip() for part in asm.bundle_text(serialized, waits[0] + 1 + point).strip('{} \n').split(';') if 'encoding' not in part)
        print(f'bundle {point}: model {issued[point]}, measured {measured}, difference {drift} ({drift - previous:+d}) {text[:200]}', flush=True)
        previous = drift

if __name__ == '__main__':
    main()
