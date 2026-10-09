"""生成 report.md 与 report.en.md 中的图：数据图读 results/ 中的记录，示意图按正文的算法画出。中文版输出到 figures/，英文版输出到 figures/en/，只依赖标准库。

颜色按实体固定：官方 Pallas 橙色，原生 XLA 灰色虚线，Pallas 源码写成的本文写法蓝色，改写 executable 的写法青色，进程内补丁的写法紫色。
"""
from collections.abc import Callable
import json
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
FIGURES = HERE / 'figures'
load = lambda name: json.loads((RESULTS / name).read_text())

FONT = "'Noto Sans CJK SC', 'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif"
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#ffffff'
OFFICIAL = '#eb6834'
XLA = '#8f8e88'
SOURCE = '#2a78d6'
REWRITE = '#1baf7a'
PATCH = '#4a3aa7'
HIGHLIGHT = '#cde2fb'
# 英文版的图把文字逐条换掉；f-string 生成的文字用正则匹配。
ENGLISH = {
    '逐元素：向量 ALU': 'Elementwise: vector ALU',
    '结果': 'Result',
    '两个寄存器的同一子通道、同一通道相遇': 'Operands meet at the same sublane and lane of two registers',
    '结果下一个周期可用，每周期两条': 'Result ready next cycle; two per cycle',
    '跨通道：XLU': 'Cross-lane: XLU',
    '提交': 'Push',
    '相邻两次提交至少隔 8 个周期': 'Pushes at least 8 cycles apart',
    '同一个队列上连续提交三次：提交后 79 个周期才能取回': 'Three pushes on one queue: each result pops 79 cycles after its push',
    '周期': 'cycles',
    '原生 XLA': 'Native XLA',
    '官方 Pallas': 'Official Pallas',
    '本文的通用分派': 'Our dispatcher',
    '周期数（对数刻度）': 'Cycles (log scale)',
    '周期数': 'Cycles',
    '行数': 'Rows',
    '输入': 'Input',
    '1 折叠（vsxpose）': '1 Fold (vsxpose)',
    '同一行变成 16 列，每列 8 个': 'One row becomes 16 columns of 8',
    '2 列内排序': '2 Sort within columns',
    '数出列内名次，vst.iar 按名次写': 'Rank within column; vst.iar writes by rank',
    '3 跨组数秩：vrot 把别的组的列对齐过来，对方的列降序，三次比较二分数出有几个排在前面': '3 Cross-group rank: vrot aligns another group\'s column; it is descending, so three comparisons binary-search how many precede',
    '自己的列': 'Own column',
    '旋转过来的列': 'Rotated column',
    '比较前的区间': 'Before comparing',
    '第 1 次比较后': 'After comparison 1',
    '第 2 次比较后': 'After comparison 2',
    '1 整体转置': '1 Full transpose',
    '[R,128]：第 r 行': '[R,128]: row r',
    '[128,R]：': '[128,R]:',
    '第 r 个通道': 'lane r',
    '2 每个子通道排出候选列': '2 Candidate columns per sublane',
    '一个通道上的 8 条候选列，各 8 层，降序': '8 descending columns of 8 per lane',
    '列首（第 0 层）参加比赛': 'Heads (level 0) compete',
    '3 败者树：每轮只重赛三场': '3 Loser tree: three replays per round',
    '新列首': 'New head',
    '上一轮的胜者来自第 5 列，第 5 列推进一格；': 'The last winner came from column 5, which advances;',
    '新列首沿路径与三个节点记下的败者比赛，': 'the new head replays the losers stored on its path;',
    '其余四个节点不变': 'the other four nodes stay valid',
    '转置加败者树': 'Transpose + loser tree',
    '行宽': 'Row width',
    '方向': 'Direction',
    '基本形状': 'Basic shape',
    'Pallas 源码': 'Pallas source',
    '改写 executable': 'Executable rewriting',
    '进程内补丁': 'In-process patch',
    '本文 / 官方 Pallas（周期数之比，对数刻度）': 'Ours / official Pallas (cycle ratio, log scale)',
}
ENGLISH_PATTERNS = ((r'^队列 (\d+)$', r'Queue \1'), (r'^第 (\d+) 行的 128 个元素$', r'The 128 elements of row \1'), (r' 沿 axis 0$', r' along axis 0'))
LANGUAGE = 'zh'

def translate(content: str) -> str:
    """LANGUAGE 为 en 时把图中的中文换成英文。"""
    if LANGUAGE == 'zh':
        return content
    if content in ENGLISH:
        return ENGLISH[content]
    for pattern, replacement in ENGLISH_PATTERNS:
        content = re.sub(pattern, replacement, content)
    assert not re.search(r'[\u4e00-\u9fff]', content), content
    return content

class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.parts: list[str] = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="{SURFACE}"/>']

    def text(self, x: float, y: float, content: str, *, size: float = 13, anchor: str = 'start', color: str = INK, weight: str = 'normal') -> None:
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{escape(translate(content))}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str = MUTED, width: float = 1, dash: str = '', arrow: bool = False) -> None:
        extra = (f' stroke-dasharray="{dash}"' if dash else '') + (' marker-end="url(#arrow)"' if arrow else '')
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def rect(self, x: float, y: float, width: float, height: float, *, fill: str = 'none', stroke: str = 'none', radius: float = 0) -> None:
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def path(self, points: list[tuple[float, float]], *, color: str, width: float = 2, dash: str = '') -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        d = ' '.join(f'{"M" if i == 0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(points))
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{extra}/>')

    def dot(self, x: float, y: float, *, color: str, radius: float = 4, hollow: bool = False) -> None:
        fill = SURFACE if hollow else color
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{color if hollow else SURFACE}" stroke-width="2"/>')

    def save(self, name: str) -> None:
        marker = f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>'
        body = '\n'.join(self.parts)
        folder = FIGURES if LANGUAGE == 'zh' else FIGURES / LANGUAGE
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" font-family="{FONT}">\n{marker}\n{body}\n</svg>\n')

def grid(svg: Svg, x: float, y: float, rows: int, columns: int, cell: float, *, fill: Callable[[int, int], str] = lambda r, c: SURFACE) -> None:
    """rows × columns 个方格，fill(r, c) 给出每格的颜色。"""
    for r in range(rows):
        for c in range(columns):
            svg.rect(x + c * cell, y + r * cell, cell, cell, fill=fill(r, c), stroke=GRID)

def axes(svg: Svg, left: float, top: float, width: float, height: float, yticks: list[tuple[float, str]], xticks: list[tuple[float, str]], ylabel: str, xlabel: str) -> None:
    """坐标轴与浅色网格；ticks 已换算成像素坐标。"""
    for y, label in yticks:
        svg.line(left, y, left + width, y, color=GRID)
        svg.text(left - 8, y + 4, label, size=12, anchor='end', color=MUTED)
    for x, label in xticks:
        svg.line(x, top + height, x, top + height + 4, color=MUTED)
        svg.text(x, top + height + 18, label, size=12, anchor='middle', color=MUTED)
    svg.line(left, top + height, left + width, top + height, color=MUTED)
    svg.text(left - 8, top - 10, ylabel, size=12, anchor='start', color=MUTED)
    svg.text(left + width / 2, top + height + 38, xlabel, size=12, anchor='middle', color=MUTED)

def results() -> dict[str, dict[str, float]]:
    """三方对照的 25 个形状：{label: {variant: 周期数}}，另含 shape 与 k。"""
    cases: dict[str, dict[str, float]] = {}
    for record in load('interval.json'):
        found = cases.setdefault(record['label'], {'rows': record['shape'][0], 'width': record['shape'][1], 'k': record['k'], 'axis': record['axis']})
        found[record['variant']] = record['median']
    return cases

def xlu_figure() -> None:
    svg = Svg(860, 300)
    cell = 9
    # 左：逐元素。
    svg.text(20, 28, '逐元素：向量 ALU', size=15, weight='bold')
    for i, name in enumerate(('TC VREG a', 'TC VREG b')):
        y = 60 + i * 100
        grid(svg, 20, y, 8, 16, cell, fill=lambda r, c: HIGHLIGHT if (r, c) == (2, 5) else SURFACE)
        svg.text(20, y - 8, name, size=12, color=MUTED)
    grid(svg, 220, 110, 8, 16, cell, fill=lambda r, c: SOURCE if (r, c) == (2, 5) else SURFACE)
    svg.line(170, 96, 214, 124, arrow=True)
    svg.line(170, 196, 214, 152, arrow=True)
    svg.text(220, 102, '结果', size=12, color=MUTED)
    svg.text(20, 262, '两个寄存器的同一子通道、同一通道相遇', size=12, color=MUTED)
    svg.text(20, 280, '结果下一个周期可用，每周期两条', size=12, color=MUTED)
    # 右：XLU。
    left = 430
    svg.text(left, 28, '跨通道：XLU', size=15, weight='bold')
    grid(svg, left, 60, 8, 16, cell, fill=lambda r, c: HIGHLIGHT if r == 2 else SURFACE)
    svg.text(left, 52, 'TC VREG', size=12, color=MUTED)
    svg.line(left + 150, 96, left + 200, 96, arrow=True)
    svg.text(left + 175, 88, '提交', size=12, anchor='middle', color=MUTED)
    for q in range(2):
        y = 70 + q * 40
        svg.rect(left + 205, y, 150, 26, fill=SURFACE, stroke=MUTED, radius=4)
        svg.text(left + 280, y + 18, f'队列 {q}', size=12, anchor='middle')
    svg.text(left + 205, 160, '相邻两次提交至少隔 8 个周期', size=12, color=MUTED)
    # 时间线：同一个队列上提交三次，79 个周期后依次取回。
    top, scale = 210, 2.8
    svg.text(left, top - 12, '同一个队列上连续提交三次：提交后 79 个周期才能取回', size=12, color=MUTED)
    svg.line(left, top + 40, left + 120 * scale, top + 40, color=MUTED)
    for t in (0, 40, 80, 120):
        svg.line(left + t * scale, top + 40, left + t * scale, top + 44, color=MUTED)
        svg.text(left + t * scale, top + 58, str(t), size=11, anchor='middle', color=MUTED)
    svg.text(left + 120 * scale + 10, top + 44, '周期', size=11, color=MUTED)
    for i, start in enumerate((0, 8, 16)):
        y = top + 8 + i * 10
        svg.line(left + start * scale, y, left + (start + 79) * scale, y, color=OFFICIAL, width=2)
        svg.dot(left + start * scale, y, color=OFFICIAL, radius=3)
        svg.dot(left + (start + 79) * scale, y, color=OFFICIAL, radius=3, hollow=True)
    svg.save('xlu.svg')

def basic_figure(cases: dict[str, dict[str, float]]) -> None:
    labels = ['basic1', 'basic8', 'basic16', 'basic32', 'basic128']
    ks = [cases[label]['k'] for label in labels]
    series = (
        ('原生 XLA', XLA, '6 4', [min(cases[label]['xla'], cases[label]['xla_stable']) for label in labels]),
        ('官方 Pallas', OFFICIAL, '', [cases[label]['official'] for label in labels]),
        ('本文的通用分派', SOURCE, '', [cases[label]['ours'] for label in labels]),
    )
    svg = Svg(640, 360)
    left, top, width, height = 70, 40, 420, 260
    low, high = math.log10(100), math.log10(20000)
    x = lambda k: left + math.log2(k) / 7 * width
    y = lambda v: top + (high - math.log10(v)) / (high - low) * height
    axes(svg, left, top, width, height, [(y(v), str(v)) for v in (100, 300, 1000, 3000, 10000)], [(x(k), str(k)) for k in ks], '周期数（对数刻度）', 'k')
    for name, color, dash, values in series:
        points = [(x(k), y(v)) for k, v in zip(ks, values)]
        svg.path(points, color=color, dash=dash)
        for px, py in points:
            svg.dot(px, py, color=color)
        svg.text(points[-1][0] + 10, points[-1][1] + 4, name, size=12)
    svg.save('basic.svg')

def fold_figure() -> None:
    svg = Svg(860, 270)
    cell, groups = 8, 3
    tone = lambda rank: ['#104281', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb'][rank]
    r = 2

    def panel(x: float, title: str, note: str, fill: Callable[[int, int, int], str]) -> None:
        svg.text(x, 28, title, size=14, weight='bold')
        for g in range(groups):
            grid(svg, x + g * (8 * cell + 6), 50, 8, 8, cell, fill=lambda s, l, g=g: fill(s, l, g))
        svg.text(x + groups * (8 * cell + 6), 50 + 4.5 * cell, '…', size=14, color=MUTED)
        svg.text(x, 50 + 8 * cell + 22, note, size=12, color=MUTED)

    width = groups * (8 * cell + 6) + 24
    panel(20, '输入', f'第 {r} 行的 128 个元素', lambda s, l, g: SOURCE if s == r else SURFACE)
    panel(20 + width + 20, '1 折叠（vsxpose）', '同一行变成 16 列，每列 8 个', lambda s, l, g: SOURCE if l == r else SURFACE)
    # 列内排序：各列按名次从深到浅；列内名次随组变化，这里只示意降序。
    panel(20 + 2 * (width + 20), '2 列内排序', '数出列内名次，vst.iar 按名次写', lambda s, l, g: tone(s) if l == r else SURFACE)
    for i in range(2):
        svg.line(20 + (i + 1) * width + i * 20 + 2, 86, 20 + (i + 1) * width + i * 20 + 16, 86, arrow=True)
    # 第三步：一列与旋转过来的另一列比较，二分查找。
    x0, y0 = 40, 180
    svg.text(20, y0 - 8, '3 跨组数秩：vrot 把别的组的列对齐过来，对方的列降序，三次比较二分数出有几个排在前面', size=13)
    for i, name in enumerate(('自己的列', '旋转过来的列')):
        for s in range(8):
            svg.rect(x0 + i * 160 + s * 18, y0 + 14, 18, 18, fill=tone(s), stroke=SURFACE)
        svg.text(x0 + i * 160, y0 + 50, name, size=12, color=MUTED)
    for step, (lo, hi) in enumerate(((0, 8), (4, 8), (4, 6))):
        svg.rect(x0 + 340 + step * 150, y0 + 14, 8 * 18, 18, fill=SURFACE, stroke=GRID)
        for s in range(lo, hi):
            svg.rect(x0 + 340 + step * 150 + s * 18, y0 + 14, 18, 18, fill=HIGHLIGHT, stroke=SURFACE)
        svg.text(x0 + 340 + step * 150, y0 + 50, ('比较前的区间', '第 1 次比较后', '第 2 次比较后')[step], size=12, color=MUTED)
    svg.save('fold.svg')

def loser_figure() -> None:
    svg = Svg(940, 330)
    # 1：整体转置。
    svg.text(20, 28, '1 整体转置', size=14, weight='bold')
    svg.rect(20, 50, 120, 40, fill=SURFACE, stroke=MUTED)
    svg.rect(20, 62, 120, 8, fill=SOURCE)
    svg.text(20, 108, '[R,128]：第 r 行', size=12, color=MUTED)
    svg.line(150, 70, 180, 70, arrow=True)
    svg.rect(190, 40, 40, 120, fill=SURFACE, stroke=MUTED)
    svg.rect(202, 40, 8, 120, fill=SOURCE)
    svg.text(190, 178, '[128,R]：', size=12, color=MUTED)
    svg.text(190, 196, '第 r 个通道', size=12, color=MUTED)
    # 2：候选列。
    x = 320
    svg.text(x, 28, '2 每个子通道排出候选列', size=14, weight='bold')
    tone = ['#104281', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb']
    for column in range(8):
        for layer in range(8):
            svg.rect(x + column * 22, 50 + layer * 14, 20, 12, fill=tone[layer], stroke=SURFACE)
        svg.text(x + column * 22 + 10, 178, str(column), size=11, anchor='middle', color=MUTED)
    svg.text(x, 200, '一个通道上的 8 条候选列，各 8 层，降序', size=12, color=MUTED)
    svg.text(x, 218, '列首（第 0 层）参加比赛', size=12, color=MUTED)
    # 3：败者树。
    x = 600
    svg.text(x, 28, '3 败者树：每轮只重赛三场', size=14, weight='bold')
    leaves = [(x + 20 + i * 38, 200) for i in range(8)]
    level1 = [((leaves[2 * i][0] + leaves[2 * i + 1][0]) / 2, 150) for i in range(4)]
    level2 = [((level1[2 * i][0] + level1[2 * i + 1][0]) / 2, 100) for i in range(2)]
    root = ((level2[0][0] + level2[1][0]) / 2, 55)
    winner = 5
    path = {('leaf', winner), ('l1', winner // 2), ('l2', winner // 4), ('root', 0)}
    edges = [(leaves[i], level1[i // 2], ('leaf', i) in path and ('l1', i // 2) in path) for i in range(8)]
    edges += [(level1[i], level2[i // 2], ('l1', i) in path) for i in range(4)]
    edges += [(level2[i], root, ('l2', i) in path) for i in range(2)]
    for (x1, y1), (x2, y2), on in edges:
        svg.line(x1, y1, x2, y2, color=REWRITE if on else GRID, width=2.5 if on else 1.5)
    for kind, nodes in (('leaf', leaves), ('l1', level1), ('l2', level2), ('root', [root])):
        for i, (nx, ny) in enumerate(nodes):
            on = (kind, i) in path
            svg.dot(nx, ny, color=REWRITE if on else MUTED, radius=7 if kind != 'leaf' else 6, hollow=not on)
    svg.text(leaves[winner][0], 224, '新列首', size=12, anchor='middle', color=INK)
    svg.text(x, 260, '上一轮的胜者来自第 5 列，第 5 列推进一格；', size=12, color=MUTED)
    svg.text(x, 278, '新列首沿路径与三个节点记下的败者比赛，', size=12, color=MUTED)
    svg.text(x, 296, '其余四个节点不变', size=12, color=MUTED)
    svg.save('loser.svg')

def rows_figure() -> None:
    sweeps = {8: load('transpose.json'), 16: load('transpose16.json')}
    transposed = {record['shape'][0]: record['median'] for record in load('rows.json') if record['variant'] == 'transposed'}
    svg = Svg(1120, 360)
    for panel, k in enumerate((8, 16)):
        records = sweeps[k]
        rows = [record['rows'] for record in records]
        official = [record['official']['median'] for record in records]
        loser = [record['transpose']['median'] for record in records]
        left, top, width, height = 70 + panel * 560, 50, 320, 240
        ceiling = 1400 if k == 8 else 2800
        x = lambda r: left + (r - 8) / 120 * width
        y = lambda v: top + (1 - v / ceiling) * height
        step = 200 if k == 8 else 500
        axes(svg, left, top, width, height, [(y(v), str(v)) for v in range(0, ceiling + 1, step)], [(x(r), str(r)) for r in (8, 32, 64, 96, 128)], '周期数', '行数')
        svg.text(left + width / 2, 24, f'k = {k}', size=14, anchor='middle', weight='bold')
        svg.path([(x(r), y(v)) for r, v in zip(rows, official)], color=OFFICIAL)
        svg.path([(x(r), y(v)) for r, v in zip(rows, loser)], color=REWRITE)
        svg.text(x(rows[-1]) + 6, y(official[-1]) + 4, '官方 Pallas', size=12)
        svg.text(x(rows[-1]) + 6, y(loser[-1]) + 14, '转置加败者树', size=12)
        if k == 8:
            for r, v in sorted(transposed.items()):
                svg.dot(x(r), y(v), color=SOURCE, hollow=True)
            last = max(transposed)
            svg.text(x(last) + 6, y(transposed[last]) - 6, 'transposed', size=12)
    svg.save('rows.svg')

def overview_figure(cases: dict[str, dict[str, float]]) -> None:
    folded = {(record['shape'][0], record['k']): record['median'] for record in load('folded-reorder.json')}
    sweep = {record['rows']: record['transpose']['median'] for record in load('transpose.json')}
    groups = (
        ('行宽', ['wide256_8', 'wide1024_8', 'wide4096_8', 'wide1024_32', 'wide4096_32', 'wide8192_32']),
        ('方向', ['vertical8_8', 'vertical16_8', 'vertical32_32', 'vertical64_32', 'vertical128_8', 'vertical128_32', 'vertical256_32']),
        ('基本形状', ['basic1', 'basic8', 'basic16', 'basic32', 'basic128']),
        ('行数', ['rows16', 'rows32', 'rows64', 'rows96', 'rows128', 'rows256', 'rows16_16']),
    )
    candidates = {'basic8': folded[8, 8], 'basic16': folded[8, 16], 'basic32': folded[8, 32], 'rows16_16': folded[16, 16]}
    candidates.update({f'rows{r}': sweep[r] for r in (16, 32, 64, 96, 128)})
    legend = (('Pallas 源码', SOURCE), ('改写 executable', REWRITE), ('进程内补丁', PATCH))
    svg = Svg(720, 40 + 25 * 22 + 4 * 18 + 70)
    left, top, width = 210, 50, 380
    low, high = math.log10(0.1), math.log10(2)
    x = lambda ratio: left + (math.log10(ratio) - low) / (high - low) * width
    rows_used = 25 * 22 + 4 * 18
    for ratio in (0.1, 0.2, 0.5, 1, 2):
        svg.line(x(ratio), top - 6, x(ratio), top + rows_used, color=MUTED if ratio == 1 else GRID)
        svg.text(x(ratio), top + rows_used + 18, f'{ratio:g}', size=12, anchor='middle', color=MUTED)
    svg.text(left + width / 2, top + rows_used + 38, '本文 / 官方 Pallas（周期数之比，对数刻度）', size=12, anchor='middle', color=MUTED)
    for i, (name, color) in enumerate(legend):
        svg.rect(120 + i * 175, 14, 12, 12, fill=color, radius=2)
        svg.text(120 + i * 175 + 18, 25, name, size=12)
    y = top
    for group, labels in groups:
        svg.text(20, y + 12, group, size=12, weight='bold')
        y += 18
        for label in labels:
            found = cases[label]
            ours, kind = found['ours'], SOURCE
            if label in candidates and candidates[label] < ours:
                ours, kind = candidates[label], PATCH if label in ('basic8', 'basic16', 'basic32', 'rows16_16') else REWRITE
            ratio = ours / found['official']
            shape = f"[{found['rows']},{found['width']}] k={found['k']}" + (' 沿 axis 0' if found['axis'] == 0 else '')
            svg.text(left - 10, y + 13, shape, size=12, anchor='end', color=MUTED)
            start, end = sorted((x(1), x(ratio)))
            svg.rect(start, y + 3, max(end - start, 1), 14, fill=kind, radius=2)
            svg.text(max(end, x(1)) + 6, y + 14, f'{ratio:.2f}', size=11, color=INK)
            y += 22
    svg.save('overview.svg')

def main() -> None:
    global LANGUAGE
    cases = results()
    for LANGUAGE in ('zh', 'en'):
        xlu_figure()
        basic_figure(cases)
        fold_figure()
        loser_figure()
        rows_figure()
        overview_figure(cases)

if __name__ == '__main__':
    main()
