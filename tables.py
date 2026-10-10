"""把 results/ 中的原始记录填进 report.template.md，生成 report.md。模板中的 <<...>> 是占位，表格和正文里引用的数字都由这里给出。"""
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
load = lambda name: json.loads((RESULTS / name).read_text())

def number(value: float) -> str:
    return f'{value:.1f}'.rstrip('0').rstrip('.')

def change(new: float, old: float) -> str:
    return '0%' if new == old else f'{(new / old - 1) * 100:+.1f}%'.replace('-', '−')

def less(new: float, old: float) -> str:
    return f'{(1 - new / old) * 100:.0f}%'

def geomean_speedup(pairs: list[tuple[float, float]]) -> str:
    """基线周期数除以本文周期数的几何平均，pairs 是 (基线, 本文)。"""
    return f'{float(np.exp(np.mean([np.log(old / new) for old, new in pairs]))):.2f}×'

def nanoseconds(cycles: float) -> str:
    """按 3.3 节实测的 1050 MHz 把周期数换算成纳秒。"""
    return f'约 {cycles / 1.05:.0f} ns'

def table(header: list[str], rows: list[list[str]], left: int = 1) -> str:
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * left + ['---:'] * (len(header) - left)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(row) + ' |' for row in rows])

def main() -> None:
    values: dict[str, str] = {}
    # 正确性。
    fields = ('out_of_range', 'duplicate', 'unpaired_bits', 'wrong_bits')
    totals: dict[str, dict[str, int]] = {}
    classes: dict[str, dict[str, int]] = {}
    for record in load('correctness.json'):
        total = totals.setdefault(record['variant'], dict.fromkeys(('rows', 'any', *fields), 0))
        for field in total:
            total[field] += record[field]
        if record['variant'] == 'official':
            for name, found in record['distributions'].items():
                entry = classes.setdefault(name, dict.fromkeys(('rows', *fields), 0))
                for field in entry:
                    entry[field] += found[field]
    assert len({total['rows'] for total in totals.values()}) == 1
    values['check:rows'] = str(totals['xla']['rows'])
    values['check:official'] = str(totals['official']['any'])
    values['check:groups'] = str(len(load('correctness.json')) // 3)
    names = {'xla': '原生 XLA', 'official': '官方 Pallas', 'ours': '本文'}
    values['TABLE:correct'] = table(['实现', '下标越界', '下标重复', '值与下标不配对', '值的位型序列与参照不同'], [[names[name], *(str(totals[name][field]) for field in fields)] for name in names])
    kinds = {'normal': '正态随机数', 'ties': '小整数并列', 'phase_count': '非负元素个数的定向构造', 'bits': '原始 32 位位型', 'special': '特殊位型的随机混合', 'constant': '恒定位型', 'finite_at_last': '有限值位于最后一个位置'}
    values['TABLE:official'] = table(['输入类别', '条数', '下标越界', '下标重复', '值与下标不配对', '值的位型序列不同'], [[label, str(classes[kind]['rows']), *(str(classes[kind][field]) for field in fields)] for kind, label in kinds.items()])
    # 三方对照。
    cases: dict[str, dict[str, dict[str, object]]] = {}
    for record in load('interval.json'):
        cases.setdefault(record['label'], {})[record['variant']] = record
    best: dict[str, tuple[float, float, float]] = {}

    def row(label: str) -> list[str]:
        found = cases[label]
        xla, stable, official, ours = (found[name]['median'] for name in ('xla', 'xla_stable', 'official', 'ours'))
        fastest = min(xla, stable)
        mark = ('†' if stable < xla else '') + ('‡' if 'repaired' in found['xla' if xla <= stable else 'xla_stable'] else '')
        best[label] = fastest, official, ours
        shape = found['xla']['shape']
        return [f'`[{shape[0]},{shape[1]}]`', str(found['xla']['k']), number(fastest) + mark, number(official), number(ours), change(ours, fastest), change(ours, official)]

    header = ['形状', '$k$', '原生 XLA', '官方 Pallas', '本文', '本文相对原生 XLA', '本文相对官方 Pallas']
    groups = {
        'basic': ['basic1', 'basic8', 'basic16', 'basic32', 'basic128'],
        'rows': ['rows16', 'rows32', 'rows64', 'rows96', 'rows128', 'rows256', 'rows16_16'],
        'wide': ['wide256_8', 'wide1024_8', 'wide4096_8', 'wide1024_32', 'wide4096_32', 'wide8192_32'],
        'vertical': ['vertical8_8', 'vertical16_8', 'vertical32_32', 'vertical64_32', 'vertical128_8', 'vertical128_32', 'vertical256_32'],
    }
    for name, labels in groups.items():
        values[f'TABLE:{name}'] = table(header, [row(label) for label in labels])
    assert len(best) == 25
    for label, (xla, official, ours) in best.items():
        values[f't:{label}:xla'], values[f't:{label}:official'], values[f't:{label}:ours'] = number(xla), number(official), number(ours)
        values[f't:{label}:ours-official'] = number(ours - official)
    assert {label for label, (_, official, ours) in best.items() if ours > official} == {'basic1', 'basic8', 'rows16', 'rows32', 'rows64'}
    faster_xla = [1 - ours / xla for xla, _, ours in best.values() if ours < xla]
    faster = [1 - ours / official for _, official, ours in best.values() if ours < official]
    slower = [ours / official - 1 for _, official, ours in best.values() if ours > official]
    official_slower = [label for label, (xla, official, _) in best.items() if official > xla]
    assert len(faster_xla) == 25
    values['SUMMARY:q3'] = '\n'.join((
        f'- 相对原生 XLA，本文在全部 25 个形状上更快，周期数少 {min(faster_xla) * 100:.0f}% 至 {max(faster_xla) * 100:.0f}%。',
        f'- 相对官方 Pallas，本文在 {len(faster)} 个形状上更快，少 {min(faster) * 100:.0f}% 至 {max(faster) * 100:.0f}%；在 {len(slower)} 个形状上更慢，多 {min(slower) * 100:.0f}% 至 {max(slower) * 100:.0f}%。这 {len(slower)} 个形状是同一种情形，即行宽 128、沿最后一维、行数不超过 64 并且仍然用链的那些：`[8,128]` 的 $k = 1$ 与 8，16、32、64 行的 $k = 8$。在这些形状上官方 Pallas 的链已经贴着 XLU 的延迟，本文多出来的主要是逐位正确所需的指令。',
        f'- 官方 Pallas 自己在 {len(official_slower)} 个形状上比原生 XLA 慢，都是 $k$ 较大的情形。',
    ))
    values['SUMMARY:officialslower'] = str(len(official_slower))
    # 原生 XLA 的周期数与输入：同一份输入重复运行的波动，与不同输入之间的差别。
    dma, constant, data = [], [], []
    for record in load('xla-inputs.json'):
        runs = record['samples']
        if any(len(set(found)) > 1 for found in runs):
            assert record['dma'], record['label']
            dma.append(max(max(found) - min(found) for found in runs))
        elif len({found[0] for found in runs}) == 1:
            constant.append(record['label'])
        else:
            data.append((record, max(found[0] for found in runs) - min(found[0] for found in runs)))
    assert len(data) == 1 and all(not record['dma'] for record, _ in data)
    (record, difference), = data
    shape = f"`[{record['shape'][0]},{record['shape'][1]}]` 取前 {record['k']} 名"
    values['xi:data-short'] = shape
    values['xi:datadiff'] = str(difference)
    values['xi:summary'] = (
        f'区间里有 DMA 的 {len(dma)} 个形状，同一份输入重复运行也有 1 至 {max(dma)} 个周期的波动，波动与输入无关；'
        f'没有 DMA 的 {len(constant) + len(data)} 个形状里，{len(constant)} 个的读数无论换不换输入都恒定。'
        f'剩下的一个是 {shape}：XLA 在这个形状上用的是一个专门的 top-$k$ custom call，同一份输入重复运行恒定，不同输入之间相差 {difference} 个周期，说明它里面有按数据决定的分支。'
        f'其余 {len(dma) + len(constant)} 个形状的周期数都与输入无关：它们多数是排序，区间里没有按数据决定的分支。'
    )
    values['q3:faster'] = str(len(faster))
    values['q3:geo-official'] = geomean_speedup([(official, ours) for _, official, ours in best.values()])
    values['q3:geo-xla'] = geomean_speedup([(xla, ours) for xla, _, ours in best.values()])
    band = [best[label][2] / best[label][1] - 1 for label in ('rows16', 'rows32', 'rows64')]
    assert min(band) > 0
    values['band:range'] = f'{min(band) * 100:.0f}% 至 {max(band) * 100:.0f}%'
    # 多行时链的两种写法。
    chains: dict[int, dict[str, float]] = {}
    for record in load('rows.json'):
        chains.setdefault(record['shape'][0], {})[record['variant']] = record['median']
    for rows, found in chains.items():
        values[f'chain:{rows}:phased-official'] = number(found['phased'] - found['official'])
    for rows, found in chains.items():
        for name, value in found.items():
            values[f'chain:{rows}:{name}'] = number(value)
    values['TABLE:chains'] = table(['行数', '官方 Pallas', '`phased`', '`phased_gather`', '`transposed`'], [[str(rows), *(number(found[name]) if name in found else '—' for name in ('official', 'phased', 'phased_gather', 'transposed'))] for rows, found in chains.items()])
    # 其他的 k。
    sweep_k: dict[tuple[int, int], dict[str, float]] = {}
    for record in load('ks.json'):
        sweep_k.setdefault((record['shape'][0], record['k']), {})[record['variant']] = record['median']
    values['TABLE:ks'] = table(['行数', '$k$', '官方 Pallas', '不转置', '`transposed`', '`transposed` 相对官方 Pallas'], [[str(rows), str(k), *(number(found[name]) for name in ('official', 'chain', 'transposed')), change(found['transposed'], found['official'])] for (rows, k), found in sorted(sweep_k.items(), key=lambda item: (item[0][1], item[0][0]))], left=2)
    assert all(found['transposed'] < min(found['official'], found['chain']) for (_, k), found in sweep_k.items() if k >= 16)
    assert all(found['transposed'] > found['chain'] > found['official'] for (_, k), found in sweep_k.items() if k < 8)
    small = [found['chain'] / found['official'] - 1 for (_, k), found in sweep_k.items() if k < 8]
    values['ks:small'] = f'{min(small) * 100:.0f}% 至 {max(small) * 100:.0f}%'
    # 只做修正。
    fix = {(record['variant'], record['inputs'] == 'permutation'): record['median'] for record in load('fixes.json')}
    official = fix['official', True]
    for name in ('official', 'mask', 'lift', 'suffix', 'phased', 'pairs'):
        values[f'fix:{name}'] = number(fix[name, True])
        values[f'fix:{name}-official'] = number(fix[name, True] - official)
    values['fix:pairs-sparse'] = number(fix['pairs', False])
    values['fix:phased-percent'] = f"{(fix['phased', True] / official - 1) * 100:.1f}%"
    assert fix['phased', False] == fix['phased', True]
    values['fix:suffix-vs'] = '没有增加' if fix['suffix', True] == official else f'多 {number(fix["suffix", True] - official)} 个'
    kinds = (('official', '官方 Pallas', '—'), ('mask', '最朴素的掩码', '重选（含 NaN 时下标越界）'), ('lift', '抬高', '重选'), ('suffix', 'MXU 后缀补位', '重选'), ('phased', '`phased`', '逐位正确'))
    values['TABLE:fixes'] = table(['写法', '修正的范围', '周期数', '相对官方 Pallas'], [[label, scope, number(fix[name, True]), '—' if name == 'official' else change(fix[name, True], official)] for name, label, scope in kinds], left=2)
    # 折叠后数秩。
    folded: dict[str, dict[tuple[int, int], float]] = {}
    for mode in ('placeholder', 'rescheduled', 'compiler', 'reorder'):
        for record in load(f'folded-{mode}.json'):
            assert record['bitwise_wrong'] == 0
            folded.setdefault(mode, {})[record['shape'][0], record['k']] = record['median']
    hand = {record['k']: record['median'] for record in load('handwritten.json')}
    assert all(record['special'] == record['median'] for record in load('handwritten.json'))
    label = {8: 'basic8', 16: 'basic16', 32: 'basic32'}
    values['TABLE:handwritten'] = table(['$k$', '原生 XLA', '官方 Pallas', '本文的通用分派', '手写的折叠后数秩', '相对原生 XLA', '相对官方 Pallas', '相对通用分派'], [[str(k), *(number(v) for v in best[label[k]]), number(hand[k]), *(change(hand[k], v) for v in best[label[k]])] for k in label])
    values['hand:8:official'], values['hand:8:xla'] = less(hand[8], best['basic8'][1]), less(hand[8], best['basic8'][0])
    modes = (('placeholder', '占位加改写'), ('rescheduled', '再重排'), ('compiler', '编译流程生成'), ('reorder', '再加流程内重排'))
    values['TABLE:folded8'] = table(['$k$', '原生 XLA', '官方 Pallas', '本文的通用分派', '手写', *(text for _, text in modes)], [[str(k), *(number(v) for v in best[label[k]]), number(hand[k]), *(number(folded[mode][8, k]) for mode, _ in modes)] for k in label])
    for mode, _ in modes:
        for k in label:
            values[f'f:{mode}:{k}'] = number(folded[mode][8, k])
            values[f'f:{mode}:{k}:official'] = less(folded[mode][8, k], best[label[k]][1])
    gains = [folded['placeholder'][8, k] - folded['rescheduled'][8, k] for k in label]
    values['f:rescheduled-gain'] = f'{number(min(gains))} 至 {number(max(gains))} 个周期'
    ratios = [folded['rescheduled'][8, k] / hand[k] - 1 for k in label]
    values['f:rescheduled-vs-hand'] = f'与手写的版本相差 {min(ratios) * 100:+.1f}% 至 {max(ratios) * 100:+.1f}%'.replace('-', '−')
    ratios = [folded['compiler'][8, k] / hand[k] - 1 for k in label]
    assert min(ratios) > 0
    values['f:compiler-vs-hand'] = f'{min(ratios) * 100:.0f}% 至 {max(ratios) * 100:.0f}%'
    ratios = [folded['reorder'][8, k] / hand[k] - 1 for k in label]
    assert max(ratios) < 0
    values['ns:basic8:official'], values['ns:basic8:folded'] = nanoseconds(best['basic8'][1]), nanoseconds(folded['reorder'][8, 8])
    values['SUMMARY:folded8'] = f'最后一种方式下，编译器从一个六十行的 Pallas 函数生成的程序，在三个 $k$ 上都比手写的版本快（少 {-max(ratios) * 100:.0f}% 至 {-min(ratios) * 100:.0f}%），比官方 Pallas 少 {less(folded["reorder"][8, 8], best["basic8"][1])} 至 {less(folded["reorder"][8, 32], best["basic32"][1])}，比原生 XLA 少 {less(folded["reorder"][8, 32], best["basic32"][0])} 至 {less(folded["reorder"][8, 8], best["basic8"][0])}。四种编译方式的次序也说明了差距的来源：从“占位加改写”到“再重排”、从“编译流程生成”到“再加流程内重排”，指令没有变，变的只是先后。'
    shapes = {(16, 8): 'rows16', (16, 16): 'rows16_16', (32, 8): 'rows32'}
    values['fr:16_16:official'], values['fr:16_16:ours'], values['fr:16_16:folded'] = less(folded['reorder'][16, 16], best['rows16_16'][1]), less(folded['reorder'][16, 16], best['rows16_16'][2]), number(folded['reorder'][16, 16])
    values['TABLE:foldedrows'] = table(['形状', '$k$', '原生 XLA', '官方 Pallas', '本文的通用分派', '折叠后数秩', '折叠后数秩，重排', '重排相对官方 Pallas', '重排相对通用分派'], [[f'`[{rows},128]`', str(k), *(number(v) for v in best[name]), number(folded['compiler'][rows, k]), number(folded['reorder'][rows, k]), change(folded['reorder'][rows, k], best[name][1]), change(folded['reorder'][rows, k], best[name][2])] for (rows, k), name in shapes.items()])
    # 整体转置加败者树。
    records = load('transpose.json')
    sweep = {record['rows']: (record['transpose']['median'], record['official']['median']) for record in records}
    assert sorted(sweep) == list(range(8, 129)) and all(new < old for new, old in sweep.values())
    values['tm:count'] = str(len(sweep))
    values['tm:rows'] = str(sum(record['bitwise_rows'] for record in records))
    saving = {rows: 1 - new / old for rows, (new, old) in sweep.items()}
    values['tm:min'], values['tm:max'] = f'{min(saving.values()) * 100:.1f}%', f'{max(saving.values()) * 100:.1f}%'
    gap = [saving[rows] for rows in range(16, 65)]
    values['tm:gap'] = f'{min(gap) * 100:.1f}% 至 {max(gap) * 100:.1f}%'
    values['tm:96'] = f'{saving[96] * 100:.1f}%'
    values['tm:128'] = number(sweep[128][0])
    values['tm:growth'] = number(sweep[128][0] - sweep[16][0])
    values['tm:128:ours'] = less(sweep[128][0], best['rows128'][2])
    values['tm:8:saving'] = number(sweep[8][1] - sweep[8][0])
    aligned = {8: 'basic8', 16: 'rows16', 32: 'rows32', 64: 'rows64', 96: 'rows96', 128: 'rows128'}
    assert all(sweep[rows][1] == best[name][1] for rows, name in aligned.items())
    values['TABLE:transpose'] = table(['行数', '原生 XLA', '官方 Pallas', '本文的通用分派', '`transposed`', '转置加败者树', '相对原生 XLA', '相对官方 Pallas', '相对 `transposed`'], [[str(rows), *(number(v) for v in best[name]), number(chains[rows]['transposed']), number(sweep[rows][0]), change(sweep[rows][0], best[name][0]), change(sweep[rows][0], best[name][1]), change(sweep[rows][0], chains[rows]['transposed'])] for rows, name in aligned.items() if rows != 8])
    grouped = []
    for last in range(8, 129, 8):
        members = [rows for rows in sweep if last - 8 < rows <= last]
        ours = sorted({sweep[rows][0] for rows in members})
        theirs = [sweep[rows][1] for rows in members]
        span = lambda low, high: number(low) if low == high else f'{number(low)}–{number(high)}'
        percent = [saving[rows] * 100 for rows in members]
        grouped.append([str(last) if len(members) == 1 else f'{members[0]}–{last}', span(ours[0], ours[-1]), span(min(theirs), max(theirs)), f'{min(percent):.1f}%' if len(members) == 1 else f'{min(percent):.1f}%–{max(percent):.1f}%'])
    values['TABLE:sweep'] = table(['行数', '转置加败者树', '官方 Pallas', '比官方少'], grouped)
    loser16 = {record['rows']: (record['transpose']['median'], record['official']['median']) for record in load('transpose16.json')}
    assert sorted(loser16) == list(range(8, 129, 8)) and all(new < old for new, old in loser16.values())
    values['TABLE:loser16'] = table(['行数', '官方 Pallas', '转置加败者树', '相对官方 Pallas'], [[str(rows), number(old), number(new), change(new, old)] for rows, (new, old) in loser16.items()])
    found = [1 - new / old for new, old in loser16.values()]
    values['tm16:range'] = f'{min(found) * 100:.0f}% 至 {max(found) * 100:.0f}%'
    values['tm16:16'] = number(loser16[16][0])
    values['tm16:rows'] = str(sum(record['bitwise_rows'] for record in load('transpose16.json')))
    # 各种办法合在一起：每个形状取本文最快的一种。
    fastest = {name: ours for name, (_, _, ours) in best.items()}
    for (rows, k), name in {(8, 8): 'basic8', (8, 16): 'basic16', (8, 32): 'basic32', **shapes}.items():
        fastest[name] = min(fastest[name], folded['reorder'][rows, k])
    for rows, name in aligned.items():
        fastest[name] = min(fastest[name], sweep[rows][0])
    behind = [name for name, value in fastest.items() if value >= best[name][1]]
    assert all(value < best[name][0] for name, value in fastest.items())
    values['all:faster'] = str(len(fastest) - len(behind))
    gain_official = [1 - fastest[name] / best[name][1] for name in fastest if name not in behind]
    gain_xla = [1 - fastest[name] / best[name][0] for name in fastest]
    values['all:official-range'] = f'{min(gain_official) * 100:.0f}% 至 {max(gain_official) * 100:.0f}%'
    values['all:xla-range'] = f'{min(gain_xla) * 100:.0f}% 至 {max(gain_xla) * 100:.0f}%'
    values['all:geo-official'] = geomean_speedup([(best[name][1], value) for name, value in fastest.items()])
    values['all:geo-xla'] = geomean_speedup([(best[name][0], value) for name, value in fastest.items()])
    assert behind == ['basic1'], behind
    values['all:remaining'] = f'唯一的例外是 `[8,128]` 取前 1 名：只有一轮，没有可以分摊、也没有可以换路的地方，逐位正确多出的 {number(best["basic1"][2] - best["basic1"][1])} 个周期（官方 {number(best["basic1"][1])}，本文 {number(best["basic1"][2])}）还在。表 <<T:ks>> 中 $k$ 为 2 和 4 的情形是同一回事：链太短，转置和折叠两头的固定开销都比它长，逐位正确的代价无处可藏。'
    # 全景：每种情形下本文最快的办法。
    def spans(labels: list[str], which: int) -> str:
        found = [fastest[label] / best[label][which] - 1 for label in labels]
        low, high = min(found) * 100, max(found) * 100
        return f'{low:+.0f}%'.replace('-', '−') if round(low) == round(high) else f'{low:+.0f}% 至 {high:+.0f}%'.replace('-', '−')

    overview = (
        ('行宽超过 128', '通道内排出候选列（5.1 节）', 'Pallas 源码', groups['wide']),
        ('沿子通道', '整数键的秩计数、候选层、归并（5.2 节）', 'Pallas 源码', groups['vertical']),
        ('行宽 128、8 行、$k$ 为 128', '秩计数（5.3 节）', 'Pallas 源码', ['basic128']),
        ('行宽 128、256 行、$k = 8$', '只求下标、结束后取值的链（5.4 节）', 'Pallas 源码', ['rows256']),
        ('行宽 128、8 行、$k$ 为 8 至 32', '折叠后数秩（6.2 节、第 7 章）', '进程内补丁（最快）；改写 executable 也快于两个基线', ['basic8', 'basic16', 'basic32']),
        ('行宽 128、16 行、$k = 16$', '折叠后数秩（7.4 节）', '进程内补丁', ['rows16_16']),
        ('行宽 128、16 至 128 行、$k = 8$', '整体转置加败者树（6.3 至 6.5 节）', '改写 executable', ['rows16', 'rows32', 'rows64', 'rows96', 'rows128']),
        ('行宽 128、8 行、$k = 1$', '`phased`', 'Pallas 源码', ['basic1']),
    )
    assert sorted(label for *_, labels in overview for label in labels) == sorted(best)
    values['TABLE:overview'] = table(['情形', '最快的办法', '部署形式', '形状数', '相对官方 Pallas', '相对原生 XLA'], [[case, method, means, str(len(labels)), spans(labels, 1), spans(labels, 0)] for case, method, means, labels in overview], left=3)
    # 近似 top-k（8.3 节）。
    timed: dict[tuple[int, int, int], dict[str, dict[str, object]]] = {}
    for record in load('approx.json'):
        timed.setdefault((*record['shape'], record['k']), {})[record['variant']] = record
    exact = {(8, 1024, 8): 'wide1024_8', (8, 4096, 8): 'wide4096_8', (8, 4096, 32): 'wide4096_32', (8, 8192, 32): 'wide8192_32'}
    rows = []
    for (count, width, k), found in timed.items():
        cycles = {name: found[name]['median'] for name in ('xla', 'official', 'stage2', 'ours')}
        mark = '‡' if 'repaired' in found['xla'] else ''
        exact_ours = number(best[exact[count, width, k]][2]) if (count, width, k) in exact else '—'
        rows.append([f'`[{count},{width}]`', str(k), str(found['ours']['bins']), number(cycles['xla']) + mark, number(cycles['official']), number(cycles['stage2']), number(cycles['ours']), change(cycles['ours'], cycles['official']), exact_ours])
    values['TABLE:approxtime'] = table(['形状', '$k$', '候选数', '原生 XLA', '官方 Pallas', '只换第二阶段', '两阶段都换', '两阶段都换相对官方', '精确：本文的通用分派'], rows, left=3)
    saving = {shape: 1 - found['ours']['median'] / found['official']['median'] for shape, found in timed.items()}
    assert [shape for shape, value in saving.items() if value < 0] == [(64, 4096, 8)]
    faster = [value for value in saving.values() if value > 0]
    values['approx:saving'] = f'{min(faster) * 100:.0f}% 至 {max(faster) * 100:.0f}%'
    values['approx:64:ours'] = f"{-saving[64, 4096, 8] * 100:.1f}%"
    values['approx:64:stage2'] = f"{(1 - timed[64, 4096, 8]['stage2']['median'] / timed[64, 4096, 8]['official']['median']) * 100:.1f}%"
    values['approx:xla'] = f"{min(1 - found['ours']['median'] / found['xla']['median'] for found in timed.values()) * 100:.0f}% 至 {max(1 - found['ours']['median'] / found['xla']['median'] for found in timed.values()) * 100:.0f}%"
    gap = {shape: found['ours']['median'] / best[name][2] - 1 for shape, name in exact.items() if (found := timed.get(shape))}
    values['approx:exact'] = '、'.join(f'`[{c},{w}]` 取前 {k} 名 {value * 100:+.0f}%'.replace('-', '−') for (c, w, k), value in gap.items())
    fields = ('out_of_range', 'duplicate', 'unpaired_bits', 'max_missing', 'wrong_bits')
    checked: dict[str, dict[str, int]] = {}
    for record in load('approx-correctness.json'):
        total = checked.setdefault(record['variant'], dict.fromkeys(('rows', 'any', *fields), 0))
        for field in total:
            total[field] += record.get(field, 0)
    names = {'xla': '原生 XLA', 'official': '官方 Pallas', 'stage2': '只换第二阶段', 'ours': '两阶段都换'}
    values['TABLE:approxcheck'] = table(['实现', '下标越界', '下标重复', '值与下标不配对', '最大值不在第一位', '与参照不同'], [[label, *(('—' if name == 'xla' and field == 'wrong_bits' else str(checked[name][field])) for field in fields)] for name, label in names.items()])
    values['approx:rows'] = str(checked['ours']['rows'])
    values['approx:cases'] = str(len(load('approx-correctness.json')) // 4)
    for name in names:
        values[f'approx:{name}:any'] = str(checked[name]['any'])
    # 分析框架给出的下限与实测（8.2 节）。下限由表 3 的延迟算出：归约 79，旋转与按下标重排 69，分段转置 126，同一个队列相邻两次提交隔 8。
    chain = (cases['basic128']['official']['median'] - cases['basic32']['official']['median']) / 96
    wide_chain = (cases['wide1024_32']['official']['median'] - cases['wide1024_8']['official']['median']) / 24
    issue = (-(-127 // 2) - 1) * 8 + 69
    model = (
        ('官方的链，每轮', '`[8,128]`', 'XLU 往返', 79, chain),
        ('官方的链，每轮', '`[8,1024]`', 'XLU 往返，两次', 79 + 69, wide_chain),
        ('秩计数，$k = 16$', '`[8,128]`', 'XLU 发射', issue, best['basic16'][2]),
        ('折叠后数秩，$k = 8$', '`[8,128]`', 'XLU 往返，三趟', 126 + 69 + 79, folded['reorder'][8, 8]),
    )
    values['TABLE:model'] = table(['写法', '形状', '框架判断的瓶颈', '下限', '实测', '下限占实测'], [[name, shape, kind, number(bound), number(measured), f'{bound / measured * 100:.0f}%'] for name, shape, kind, bound, measured in model], left=3)
    values['model:chain-extra'] = number(chain - 79)
    for line in (RESULTS / 'latency.txt').read_text().splitlines():
        if line.startswith('vxpose width 128'):
            first, total = re.search(r'\[\((\d+), (\d+)\)\]', line).groups()
            values['lat:xpose'] = f'{first}、{total}'
    text = (HERE / 'report.template.md').read_text()
    values.update({f'T:{name}': str(index) for index, name in enumerate(re.findall(r'^表 <<T:(\w+)>>：', text, re.M), 1)})
    values.update({f'F:{name}': str(index) for index, name in enumerate(re.findall(r'^图 <<F:(\w+)>>：', text, re.M), 1)})
    for _ in range(2):
        text = re.sub(r'<<([^<>]+)>>', lambda match: values[match[1]], text)
    (HERE / 'report.md').write_text(text)
    for key in ('SUMMARY:q3', 'SUMMARY:folded8', 'f:rescheduled-gain', 'f:rescheduled-vs-hand', 'TABLE:rows', 'TABLE:chains', 'TABLE:foldedrows'):
        print(values[key], end='\n\n')

if __name__ == '__main__':
    main()
