"""把英文版报告 report.en.md 转成 ACM acmart（sigconf，nonacm）的 LaTeX 源文件 main.tex，并把 figures/en/ 中的 SVG 转成 PDF。

转换只处理 report.en.md 用到的 Markdown：标题、段落、列表、代码块、表格、图、粗体、行内代码和链接。正文中的 [n] 换成 \\cite，按 REFERENCES 的顺序对应 references.bib 中的条目；“Section 3.4”“Table 6”“Figure 2”“Appendix C.1”换成 \\ref。表格按估计的宽度放进单栏的 table 或通栏的 table*。生成的源文件只含 ASCII，pdflatex、xelatex 与 lualatex 都能编译。

SVG 转 PDF 需要 cairosvg（pip install cairosvg）。
"""
import argparse
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# report.en.md 参考文献 1 到 38 在 references.bib 中的键。
REFERENCES = (
    'jax-topk-impl',
    'jax-issue-34620',
    'xla-compare',
    'batcher1968',
    'tpuasm',
    'libtpu',
    'jax-layout',
    'pallas-async',
    'xla-memory-space-propagation',
    'shanbhag2018',
    'johnson2021faiss',
    'gaihre2021drtopk',
    'li2024radik',
    'xie2025rtopk',
    'guo2026sonicmoe',
    'key2024approx',
    'chern2022tpuknn',
    'samaga2026',
    'chhugani2008',
    'wassenberg2022vqsort',
    'jia2018volta',
    'hayes2019decoding',
    'abel2019uops',
    'he2024sip',
    'he2025cuasmrl',
    'norrie2021tpuv2',
    'jouppi2023tpuv4',
    'jouppi2026ironwood',
    'qwen3',
    'ieee754',
    'herf2001radix',
    'knuth1998',
    'zhang2023air',
    'zhang2017kepleras',
    'maxas',
    'turingas',
    'kaufman2021',
    'qwen3-config',
)
# sigconf 的通栏与单栏宽度（pt）。
TEXT_WIDTH = 506
COLUMN_WIDTH = 241
PREAMBLE = r'''\documentclass[sigconf,nonacm,balance=false]{acmart}

\usepackage{listings}
\usepackage{makecell}
\lstset{basicstyle=\ttfamily\scriptsize, columns=fullflexible, keepspaces=true, breaklines=true, breakatwhitespace=false, showstringspaces=false, upquote=true, frame=none, aboveskip=4pt, belowskip=4pt}
\newcommand{\code}[1]{\texttt{#1}}
\setlength{\emergencystretch}{3em}

\begin{document}
'''
AUTHORS = r'''\author{Muhan Zhong}
\affiliation{%
  \institution{Independent Researcher}
  \country{}}
'''
REPOSITORY = 'https://github.com/ayaka14732/tpu-v4-top-k'
# 首页的脚注，排在致谢之前。
NOTES = (
    f'Code, raw results, and the Chinese version of this paper: \\url{{{REPOSITORY}}}.',
    'The code, experiments, and text of this paper were produced with Claude Code (Opus 5.5) and Codex (GPT-5.6 Sol, GPT-6 Astra), prompted by Muhan Zhong.',
)
# report.en.md 中只适用于仓库内阅读的句子，在论文里换掉。
PAPER_TEXT = {
    'All code for this report is in this directory and depends only on': f'All code for this paper is available at <{REPOSITORY}> and depends only on',
    'The Chinese version `report.md` is generated from `report.template.md` and `results/` by `tables.py`; this English version is a translation of it. Run in this directory:': 'The repository also contains a Chinese version of this paper, `report.md`, which `tables.py` generates from `report.template.md` and `results/`; this paper is a translation of it. Run in the root of the repository:',
}
TEXT = {
    '\u2212': '$-$',
    '\u2013': '--',
    '\u2014': '---',
    '\u00d7': '$\\times$',
    '\u2264': '$\\le$',
    '\u2265': '$\\ge$',
    '\u2021': '\\ddag{}',
    '\u2020': '\\dag{}',
    '\u00a7': '\\S{}',
    '\u03c1': '$\\rho$',
    '\u03b5': '$\\epsilon$',
    '\u2032': '$\'$',
    '\u00e9': '\\\'e',
    '\u2308': '$\\lceil$',
    '\u2309': '$\\rceil$',
    '\u2082': '$_2$',
}
SUPERSCRIPT = dict(zip('\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079', '0123456789'))
SPECIAL = {'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}', '^': '\\textasciicircum{}'}

def escape(text: str) -> str:
    return ''.join(SPECIAL.get(ch, ch) for ch in text)

def code(text: str) -> str:
    """行内代码：转义，并在常见的分隔符之后允许断行。"""
    out = escape(text)
    out = re.sub(r'(\\_|\.|/|,|\(|=|::)', lambda m: m[1] + '\\allowbreak{}', out)
    out = re.sub(r'(?<=[a-z])(?=[A-Z])', r'\\allowbreak{}', out)
    return f'\\code{{{out}}}'

def cite(match: re.Match[str]) -> str:
    keys = []
    for part in re.split(r',\s*', match[1]):
        first, _, last = part.partition('\u2013')
        keys += [REFERENCES[n - 1] for n in range(int(first), int(last or first) + 1)]
    return f'~\\cite{{{",".join(keys)}}}'

def references(text: str) -> str:
    """“Section 3.4”“Chapters 6 and 7”“Table 6”“Figure 2”“Appendix C.1”一类的编号换成 \\ref；章与节在论文中都叫 Section。"""
    kinds = {'Section': 'sec', 'Sections': 'sec', 'Chapter': 'sec', 'Chapters': 'sec', 'Table': 'tab', 'Tables': 'tab', 'Figure': 'fig', 'Figures': 'fig', 'Appendix': 'sec'}
    number = r'(?:[A-C](?:\.\d+)?|\d+(?:\.\d+)?)'
    pattern = rf'\b(Sections?|Chapters?|Tables?|Figures?|Appendix) ({number}(?:(?:, | and |, and | to ){number})*)\b'

    def replace(match: re.Match[str]) -> str:
        word, kind = match[1].replace('Chapter', 'Section'), kinds[match[1]]
        if kind == 'sec' and match[1] == 'Appendix':
            word = 'Appendix'
        numbers = re.sub(number, lambda m: f'\\ref{{{kind}:{m[0]}}}', match[2])
        return f'{word}~{numbers}'

    return re.sub(pattern, replace, text)

def prose(text: str) -> str:
    """不含行内代码的一段文字。"""
    text = re.sub(r'\b[Tt]his report\b', lambda m: m[0][0] + 'his paper', text)
    text = text.replace('in the report', 'in the paper')
    # 正文中零星的下标写法，如 A_b、max_b。
    text = re.sub(r'\b(max|min)_([0-9a-z])\b', r'@@\\\1_{\2}@@', text)
    text = re.sub(r'\b([A-Z])_([0-9a-z])\b', r'@@\1_{\2}@@', text)
    text = re.sub(r'\b([A-Z]+(?:_[A-Z]+)+)\b', lambda m: '@@CODE' + m[1] + '@@', text)
    parts = re.split(r'(@@.*?@@)', text)
    out = []
    for part in parts:
        if part.startswith('@@CODE'):
            out.append(code(part[6:-2]))
        elif part.startswith('@@'):
            out.append(f'${part[2:-2]}$')
        else:
            out.append(plain(part))
    return ''.join(out)

def plain(text: str) -> str:
    text = re.sub(r'\[(\d+(?:(?:\u2013|,\s*)\d+)*)\]', lambda m: '\x00' + m[1] + '\x01', text)
    text = re.sub(r'<(https?://[^>]+)>', lambda m: '\x02' + m[1] + '\x03', text)
    text = escape(text)
    text = re.sub(r'\s*\x00([^\x01]*)\x01', cite, text)
    text = re.sub(r'[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+', lambda m: '$^{' + ''.join(SUPERSCRIPT[ch] for ch in m[0]) + '}$', text)
    text = ''.join(TEXT.get(ch, ch) for ch in text)
    text = re.sub(r'\x02([^\x03]*)\x03', lambda m: f'\\url{{{m[1]}}}', text)
    text = text.replace('"', "''").replace(" ''", ' ``').replace("(''", '(``')
    text = re.sub(r"^''", '``', text)
    assert all(ord(ch) < 128 for ch in text), [ch for ch in text if ord(ch) >= 128]
    return references(text)

def inline(text: str) -> str:
    """一行 Markdown 文字：行内代码原样转义，其余按 prose 处理。"""
    # 粗体可能跨过行内代码，先换成控制字符，转换完再换回。
    text = re.sub(r'\*\*(.+?)\*\*', '\x04\\1\x05', text)
    parts = re.split(r'(`[^`]+`)', text)
    out = ''.join(code(part[1:-1]) if part.startswith('`') else prose(part) for part in parts)
    return out.replace('\x04', '\\textbf{').replace('\x05', '}')

def wrap(words: list[str], width: int) -> list[str]:
    lines: list[str] = []
    for word in words:
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += ' ' + word
        else:
            lines.append(word)
    return lines

def table(rows: list[str], caption: str | None) -> str:
    """Markdown 表格转成 booktabs 表格。

    宽度按字符数估计。先不拆表头，单栏放得下就用 table；否则把比内容宽得多的表头拆成两行再试；仍放不下就用通栏的 table*，通栏也放不下时把最宽的左对齐列改成可换行的 p 列。没有标题的表格不加标题，照样浮动。
    """
    cells = [[cell.strip() for cell in row.strip().strip('|').split('|')] for row in rows]
    header, spec, body = cells[0], cells[1], cells[2:]
    align = ['r' if column.endswith(':') else 'l' for column in spec]
    visible = lambda cell: len(re.sub(r'`|\*\*', '', cell))
    content = [max(visible(row[i]) for row in body) for i in range(len(header))]
    measure = lambda heads: sum(max(content[i], max(map(visible, heads[i]))) for i in range(len(header))) + 3 * len(header)
    heads = [[cell] for cell in header]
    split = []
    for i, cell in enumerate(header):
        words = cell.split(' ')
        lines = wrap(words, max(content[i], -(-visible(cell) // 2))) if visible(cell) > content[i] + 4 and len(words) > 1 and '`' not in cell else [cell]
        split.append(lines if max(map(len, lines)) < len(cell) else [cell])
    if measure(heads) > 52 and measure(split) <= 52:
        heads = split
    wide = measure(heads) > 52
    if wide and measure(heads) > 120:
        heads = split
    width = [max(content[i], max(map(visible, heads[i]))) for i in range(len(header))]
    total = sum(width) + 3 * len(width)
    room = 120 if wide else 52
    columns = list(align)
    for i in sorted(range(len(width)), key=lambda i: -width[i]):
        if total <= room or align[i] != 'l':
            continue
        target = max(16, width[i] - (total - room))
        columns[i] = f'>{{\\raggedright\\arraybackslash}}p{{{target / room:.3f}\\linewidth}}'
        heads[i] = [' '.join(heads[i])]
        total -= width[i] - target
    environment = 'table*' if wide else 'table'
    head = ['\\centering']
    if caption is not None:
        head += [f'\\caption{{{inline(caption.split(": ", 1)[1])}}}', f'\\label{{tab:{re.match(r"Table (\d+):", caption)[1]}}}']
    head.append('\\footnotesize' if wide else '\\small')
    title = lambda i: inline(heads[i][0]) if len(heads[i]) == 1 else f'\\makecell[{align[i]}b]{{{"\\\\".join(inline(line) for line in heads[i])}}}'
    lines = [f'\\begin{{{environment}}}', *head, f'\\begin{{tabular}}{{{"".join(columns)}}}', '\\toprule']
    lines.append(' & '.join(title(i) for i in range(len(header))) + ' \\\\')
    lines.append('\\midrule')
    lines += [' & '.join(inline(cell) for cell in row) + ' \\\\' for row in body]
    lines += ['\\bottomrule', '\\end{tabular}', f'\\end{{{environment}}}']
    return '\n'.join(lines)

def figure(path: str, caption: str) -> str:
    """图的宽度让 SVG 中 12 px 的文字印出来约 7 pt，最宽到通栏；比单栏宽的放进 figure*。"""
    name = Path(path).stem
    number = re.match(r'Figure (\d+):', caption)[1]
    pixels = float(re.search(r'<svg[^>]* width="([\d.]+)"', (ROOT / 'figures' / 'en' / f'{name}.svg').read_text())[1])
    points = min(TEXT_WIDTH, pixels * 7 / 12)
    environment = 'figure*' if points > COLUMN_WIDTH else 'figure'
    width = f'{points / TEXT_WIDTH:.2f}\\textwidth' if environment == 'figure*' else '\\columnwidth'
    return '\n'.join((
        f'\\begin{{{environment}}}',
        '\\centering',
        f'\\includegraphics[width={width}]{{figures/{name}.pdf}}',
        f'\\caption{{{inline(caption.split(": ", 1)[1])}}}',
        f'\\label{{fig:{number}}}',
        f'\\end{{{environment}}}',
    ))

def listing(lines: list[str], wide: bool) -> str:
    """代码块；wide 时放进不加标题的通栏浮动体，用于附录中较长的命令清单。"""
    assert all(ord(ch) < 128 for line in lines for ch in line), lines
    block = '\n'.join(('\\begin{lstlisting}', *lines, '\\end{lstlisting}'))
    return f'\\begin{{figure*}}\n{block}\n\\end{{figure*}}' if wide else block

def convert(markdown: str) -> str:
    for old, new in PAPER_TEXT.items():
        assert markdown.count(old) == 1, old
        markdown = markdown.replace(old, new)
    # 致谢不单独成节，与 NOTES 一起作为首页的脚注。
    head, _, rest = markdown.partition('\n## Acknowledgments\n\n')
    acknowledgment, _, rest = rest.partition('\n\n')
    markdown = head + '\n' + rest
    notes = '\n'.join(f'\\footnotetext{{{note}}}%' for note in (*NOTES, inline(acknowledgment)))
    lines = markdown.split('\n')
    out = [PREAMBLE]
    abstract: list[str] = []
    keywords = ''
    section = ''
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('```'):
            end = lines.index('```', i + 1)
            body = lines[i + 1:end]
            out.append(listing(body, section == 'appendix' and len(body) > 10 and max(map(len, body)) > 70))
            i = end + 1
            continue
        if line.startswith('# '):
            out.append(f'\\title{{{inline(line[2:])}}}\n')
            out.append(AUTHORS)
            i += 1
            continue
        if line.startswith('## '):
            title = line[3:]
            if section == 'abstract':
                out.append('\\begin{abstract}\n' + '\n\n'.join(abstract) + '\n\\end{abstract}\n')
                out.append(f'\\keywords{{{keywords}}}\n')
                out.append('\\maketitle\n')
                out.append('{\\renewcommand{\\thefootnote}{}%\n' + notes + '\n}\n')
            if title == 'Abstract':
                section = 'abstract'
            elif title == 'References':
                out.append('\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{references}\n')
                section = 'references'
            elif title.startswith('Appendix '):
                letter, name = title[len('Appendix '):].split(' ', 1)
                if letter == 'A':
                    out.append('\\appendix\n')
                out.append(f'\\section{{{inline(name)}}}\n\\label{{sec:{letter}}}\n')
                section = 'appendix'
            else:
                number, name = title.split(' ', 1)
                out.append(f'\\section{{{inline(name)}}}\n\\label{{sec:{number}}}\n')
                section = 'body'
            i += 1
            continue
        if section == 'references':
            i += 1
            continue
        if line.startswith('### '):
            number, name = line[4:].split(' ', 1)
            out.append(f'\\subsection{{{inline(name)}}}\n\\label{{sec:{number}}}\n')
            i += 1
            continue
        if section == 'abstract':
            if line.startswith('**Keywords**: '):
                keywords = ', '.join(inline(word) for word in line[len('**Keywords**: '):].split('; '))
            else:
                abstract.append(inline(line))
            i += 1
            continue
        if line.startswith('|'):
            end = i
            while end < len(lines) and lines[end].startswith('|'):
                end += 1
            caption = lines[end + 1] if end + 1 < len(lines) and lines[end + 1].startswith('Table ') else None
            out.append(table(lines[i:end], caption) + '\n')
            i = end + (2 if caption else 0)
            continue
        if line.startswith('!['):
            path = re.match(r'!\[\]\(([^)]+)\)', line)[1]
            out.append(figure(path, lines[i + 2]) + '\n')
            i += 3
            continue
        if line.startswith('- ') or re.match(r'\d+\. ', line):
            ordered = not line.startswith('- ')
            items = []
            while i < len(lines) and (lines[i].startswith('- ') if not ordered else re.match(r'\d+\. ', lines[i])):
                items.append(re.sub(r'^(- |\d+\. )', '', lines[i]))
                i += 1
            environment = 'enumerate' if ordered else 'itemize'
            out.append(f'\\begin{{{environment}}}\n' + '\n'.join(f'\\item {inline(item)}' for item in items) + f'\n\\end{{{environment}}}\n')
            continue
        out.append(inline(line) + '\n')
        i += 1
    out.append('\\end{document}\n')
    return '\n'.join(out)

def figures(output: Path) -> None:
    import cairosvg
    output.mkdir(exist_ok=True)
    for svg in sorted((ROOT / 'figures' / 'en').glob('*.svg')):
        cairosvg.svg2pdf(url=str(svg), write_to=str(output / f'{svg.stem}.pdf'))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-figures', action='store_true', help='只生成 main.tex，不转换图')
    args = parser.parse_args()
    (HERE / 'main.tex').write_text(convert((ROOT / 'report.en.md').read_text()))
    if not args.no_figures:
        figures(HERE / 'figures')

if __name__ == '__main__':
    main()
