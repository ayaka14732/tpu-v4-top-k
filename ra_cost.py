"""编译器自己怎样估计寄存器分配的代价：让 libtpu 把各阶段的 LLO 转储出来，读其中的寄存器压力和几份排程分析。

寄存器分配前后编译器各打包一次 bundle，并且额外算两份“假如”：不计寄存器带来的先后限制（no-ra-deps）、去掉溢出与读回（no-spills-fills）。
每份分析的 total scheduled bundles 含空 bundle，是打包器按延迟表排出来的静态排程的长度（整个 kernel，含两头的 DMA）；编译器没有周期级的模拟器。
用法：ra_cost.py <行数> <k>；环境变量与 folded_top_k.py 相同。
"""
from pathlib import Path
import os
import re
import sys
import tempfile
DUMP = tempfile.mkdtemp(prefix='ra_cost_')
os.environ['LIBTPU_INIT_ARGS'] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true --xla_jf_dump_llo_pass_label_regex=.*".strip()
import folded_top_k
import numpy as np

def main() -> None:
    rows, k = int(sys.argv[1]), int(sys.argv[2])
    compiled, serialized, _ = folded_top_k.build(rows, k)
    files = {re.sub(r'^\d+-folded_top_k\.\d+-\d+-', '', path.name): path for path in sorted(Path(DUMP).glob('*folded_top_k*'))}
    pressure = ', '.join(re.findall(r'^Register pressure for (\w+ is \d+)', files['register-pressure.txt'].read_text(), re.M))
    stages = ('packed-bundles-pre-ra', 'packed-bundles-no-ra-deps', 'packed-bundles-no-spills-fills', 'packed-bundles-post-ra')
    totals = {stage: int(re.search(r'total scheduled bundles:\s+(\d+)', files[f'schedule-analysis_{stage}.txt'].read_text()).group(1)) for stage in stages}
    measured = folded_top_k.compute_cycles(compiled, serialized, rows, np.random.default_rng(6502))
    print(f'$ ra_cost.py {rows} {k}')
    print(f'register pressure: {pressure}; total scheduled bundles: {totals}; measured compute cycles {measured}', flush=True)

if __name__ == '__main__':
    main()
