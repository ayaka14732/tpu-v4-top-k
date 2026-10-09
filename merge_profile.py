"""转置之后的两种合并各用了多少条指令：Pallas 源码写的 transposed（比较网络把候选列两两归并）与 transpose_merge.py 的败者树。

两者的转置和候选列排序相同，kernel 里指令条数的差别就是合并和转回那一段的差别。按物理槽统计 kernel 起止标记之间的指令：向量 ALU、读、写、XLU 提交、XLU 取回，以及 bundle 数。
用法：merge_profile.py <行数>...
"""
import chip
from collections import Counter
import re
import sys

from tpuasm import tools as asm

import shell
import total_order
import transpose_merge

SLOTS = {'va': '向量 ALU', 'vld': '读', 'vst': '写', 'vx': 'XLU 提交', 'vr': 'XLU 取回'}

def profile(serialized: bytes) -> dict[str, int]:
    start, end = shell.kernel_span(serialized)
    (_, _, image), = asm.executable_programs(serialized)
    lines = asm.format_assembly(image, target=asm.TARGET, encoding='canonical').splitlines()
    spans = asm._bundle_spans(lines)
    counts: Counter[str] = Counter()
    for pc in range(start + 1, end):
        text = '\n'.join(line.split('#')[0] for line in lines[spans[pc][0]:spans[pc][1] + 1])
        for slot, mnemonic in re.findall(r'\b(va|vld|vst|vx|vr)\d?: (?:@!?p\d+ )?([\w.]+)', text):
            counts[SLOTS[slot]] += 1
            if mnemonic.startswith(('vld.iar', 'vld.sshfl')):
                counts['其中按索引或按模式的读'] += 1
    return {'bundle': end - start - 1, **counts}

def main() -> None:
    for rows in map(int, sys.argv[1:]):
        compiled, _ = shell.compile(shell.pallas_top_k(8, 1), (rows, 128), total_order.transposed)
        print(f'[{rows},128] transposed:', profile(asm.serialize(compiled)), flush=True)
        saved: dict[str, object] = {}
        compiled, _ = shell.compile(shell.pallas_top_k(8, 1, transpose_merge.carrier(rows, 8), scratch_rows=transpose_merge.layout(8)[1] + 96), (rows, 128))
        print(f'[{rows},128] loser tree:', profile(transpose_merge.transform(rows, 8, saved)(compiled)), flush=True)

if __name__ == '__main__':
    main()
