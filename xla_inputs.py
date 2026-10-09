"""原生 XLA 的周期数是否随输入变化：每个形状取 4 份不同的输入，每份在同一个 executable 上重复运行 6 次。

计时区间与 shell.py 相同。同一份输入重复运行仍有波动，说明波动来自程序之外（区间内的 DMA）；同一份输入恒定、不同输入不同，说明程序里有按数据的分支。
用法：xla_inputs.py --cases <标签>...（标签见 compare.TIME_CASES），结果写到 --output 目录的 xla-inputs.json。
"""
import chip
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from tpuasm import tools as asm

import compare
import shell

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--output', type=Path, default=Path('/tmp/top-k-paper/xla-inputs'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clock = asm.KernelClock(num_cores=1)
    records = []
    for label, shape, axis, k in compare.TIME_CASES:
        if args.cases and label not in args.cases:
            continue
        compiled, _ = shell.compile(shell.xla_top_k(k, axis), shape)
        first, last, evidence = shell.boundary(compiled, asm.serialize(compiled))
        timed = clock.instrument(compiled, [first, last])
        samples = []
        for seed in range(4):
            x = jax.device_put(shell.permutations(shape, axis, seed))
            runs = []
            for _ in range(8):
                jax.block_until_ready(timed(x))
                reads = clock.read(2)[0].astype(np.int64)
                runs.append(int(reads[1] - reads[0] - asm.CLOCK_OVERHEAD))
            samples.append(runs[2:])
        records.append({'label': label, 'shape': shape, 'axis': axis, 'k': k, 'samples': samples, 'dma': evidence['dma'], 'ops': evidence['ops']})
        print(json.dumps({'label': label, 'per_input': [sorted(set(runs)) for runs in samples], 'dma': evidence['dma']}), flush=True)
        (args.output / f'xla-inputs-{args.cases[0] if args.cases else "all"}.json').write_text(json.dumps(records, ensure_ascii=False, indent=1) + '\n')

if __name__ == '__main__':
    main()
