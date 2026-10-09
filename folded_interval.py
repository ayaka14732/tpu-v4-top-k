"""折叠后数秩放进 shell.py 的外壳，与 compare.py 的三方对照用同一个计时区间。同一个 Pallas 函数（folded_top_k.py）有四种编译方式，由第一个参数选择：

placeholder  四个 intrinsic 写成占位，编译之后由 asm_intrinsics.rewrite 换成真正的指令；
rescheduled  在 placeholder 之上，再用 reschedule.py 按实测的时序把 kernel 重新排程；
compiler     四个 intrinsic 由打过补丁的编译流程直接生成，排程由编译器自己完成；
reorder      在 compiler 之上，再在编译流程里面按时间重排（vliw_reorder.py）。

libtpu 的补丁在导入时装上，对整个进程生效，所以每种方式单独一个进程运行。
"""
import argparse
import json
import os
from pathlib import Path
import sys

MODE = sys.argv[1]
assert MODE in ('placeholder', 'rescheduled', 'compiler', 'reorder'), MODE
if MODE in ('compiler', 'reorder'):
    os.environ.update(FRONT_DOOR_ALL='1', SUM='tree')
if MODE == 'reorder':
    os.environ['VLIW_REORDER'] = '100000'
import folded_top_k as folded
import jax
from jax import Ref
import numpy as np
from tpuasm import tools as asm

import asm_intrinsics as ai
import reschedule
import shell

CASES = (
    ((8, 128), 8),
    ((8, 128), 16),
    ((8, 128), 32),
    ((16, 128), 8),
    ((16, 128), 16),
    ((32, 128), 8),
)

def transform(compiled: asm.Compiled) -> bytes:
    """占位换成指令；rescheduled 再重排 kernel 中起止标记与结尾的 cfence 之间的那一段。"""
    serialized, _ = ai.rewrite(compiled)
    if MODE == 'rescheduled':
        start, end = shell.kernel_span(serialized)
        assert 'cfence' in asm.bundle_text(serialized, end - 1)
        serialized, _ = reschedule.reschedule(serialized, span=(start + 1, end - 1))
    return serialized

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--output', type=Path, default=Path('/tmp/top-k-paper/folded'))
    parser.add_argument('--rows', type=int, nargs='+', default=[8, 16, 32])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clock = asm.KernelClock(num_cores=1)
    rng = np.random.default_rng(6603)
    placeholder = MODE in ('placeholder', 'rescheduled')
    records = []
    for shape, k in CASES:
        if shape[0] not in args.rows:
            continue

        def body(x: jax.Array, work: Ref) -> tuple[jax.Array, jax.Array]:
            if placeholder:
                ai.scratch(work)
            return folded.folded_top_k(x, k)

        top_k = shell.pallas_top_k(k, 1, body, scratch_rows=48 * shape[0])
        ai.SITES.clear()
        record = shell.measure(clock, top_k, shape, k, 1, transform=transform if placeholder else None, output=args.output, label=f'{MODE}-{shape[0]}x128-k{k}')
        # 计时用的输入没有并列；这里在同一个外壳里用特殊值再核对一遍值的位型和稳定下标。
        ai.SITES.clear()
        compiled, _ = shell.compile(top_k, shape)
        function = asm.load(transform(compiled), compiled) if placeholder else compiled
        rows = wrong = 0
        for _, x in folded.cases(rng, shape[0], k):
            values, index = (np.asarray(item) for item in function(jax.device_put(x)))
            want, order = folded.reference(x, k)
            rows += shape[0]
            wrong += int(np.count_nonzero(np.any(values.view(np.uint32) != want, axis=1) | np.any(index != order, axis=1)))
        records.append({**record, 'variant': f'folded-{MODE}', 'bitwise_rows': rows, 'bitwise_wrong': wrong})
        print(json.dumps({'mode': MODE, 'shape': shape, 'k': k, 'median': record['median'], 'rows': rows, 'wrong': wrong}), flush=True)
        (args.output / f'folded-{MODE}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    main()
