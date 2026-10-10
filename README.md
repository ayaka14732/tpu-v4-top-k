# TPU v4 上逐位正确且更快的 Top-$k$

本仓库包含 TPU v4 单个 TensorCore 上 top-$k$ 的算法实现、正确性检查、性能实验和论文源文件。阅读 [中文报告](report.md) 或 [英文报告](report.en.md)；论文的 LaTeX 源文件在 [paper/](paper/)。

## 文件一览

| 文件 | 内容 |
| --- | --- |
| [total_order.py](total_order.py) | 逐位正确的各种写法与通用分派 `fast` |
| [fixes.py](fixes.py) | 只修重选的三种写法：掩码、抬高、MXU 后缀补位 |
| [shell.py](shell.py) | 起终点都在 TC VMEM 的外壳、边界核对与计时 |
| [pin_propagation.py](pin_propagation.py)、[pin_propagation.cpp](pin_propagation.cpp) | Pin 着色之后补上内存空间传播的进程内修复 |
| [compare.py](compare.py) | 三方对照的计时与正确性 |
| [folded_select.py](folded_select.py) | 手写的折叠后数秩 |
| [asm_intrinsics.py](asm_intrinsics.py)、[check_intrinsics.py](check_intrinsics.py) | 四个占位 intrinsic、改写器及其核对 |
| [folded_top_k.py](folded_top_k.py)、[folded_interval.py](folded_interval.py) | 用 intrinsic 写的折叠后数秩，及其在外壳中的四种编译方式 |
| [reschedule.py](reschedule.py)、[issue_profile.py](issue_profile.py) | 编译之后的重排工具与时序模型的校准 |
| [transpose_merge.py](transpose_merge.py)、[merge_profile.py](merge_profile.py) | 整体转置之后用败者树做八路合并；两种合并的指令条数 |
| [libtpu_patch.py](libtpu_patch.py)、[front_door.py](front_door.py)、[vliw_reorder.py](vliw_reorder.py)、[libtpu_latency.cpp](libtpu_latency.cpp)、[libtpu_sublane_gather.cpp](libtpu_sublane_gather.cpp) | 进程内给 libtpu 打的补丁、由编译流程生成的四个 intrinsic、流程内的重排 |
| [llo_op_probe.py](llo_op_probe.py)、[tpu_op_probe.py](tpu_op_probe.py)、[ra_cost.py](ra_cost.py) | 编译链路各层的探针，以及编译器自己的寄存器压力报告 |
| [memory_capacity.py](memory_capacity.py) | 各层存储的容量：CMEM 的回绕实测与 HBM 的分配上限 |
| [semantics.py](semantics.py)、[latency.py](latency.py)、[latency_probe.py](latency_probe.py) | 指令语义、延迟的实测，以及编译器的延迟表 |
| [tables.py](tables.py)、[report.template.md](report.template.md) | 正文的模板，以及把 `results/` 中的数字填进模板的脚本 |
| [approx.py](approx.py) | Pallas 与原生 XLA 的 `approx_max_k`，以及把本文的写法放进它的两个阶段 |
| [figures.py](figures.py) | 由 `results/` 生成 `figures/`（中文）与 `figures/en/`（英文）中的图 |
| [clock_rate.py](clock_rate.py) | LCC 的计数频率 |
| [chip.py](chip.py) | 选择本进程使用的芯片 |

## 实验环境

论文使用 Python 3.14.7t、JAX `0.12.0.dev20261002+7fc69a22c2`、jaxlib `0.12.0.dev20261002`、libtpu `0.0.49` 和 tpuasm `0.2.1`。机器拓扑为 TPU v4 (2x2x2)，包含 2 个 host，每个 host 有 4 颗芯片；实验只使用 host 0 本地一颗芯片的一个 TensorCore。

`chip.py` 在导入 JAX 前选择芯片，将两个 TensorCore 分别暴露为设备。环境变量 `TOP_K_CHIP` 选择本地芯片，默认值为 `1`。不同芯片上的进程可以同时运行；`folded_interval.py` 的四种编译方式各自启动一个进程，因为编译器补丁对整个进程生效。

## 复现实验

在已安装上述依赖的 Python 环境中，从仓库根目录运行。下面的注释给出命令对应的论文表格或章节；可指定的输出路径使用 `/tmp/top-k-paper/`，没有输出参数的探针直接查看终端输出。论文使用的原始记录归档在 `results/`。

```sh
# 创建实验输出目录
mkdir -p /tmp/top-k-paper

# 表 4、22：精确 top-k 的正确性与按输入类别统计的错误
python compare.py check --output /tmp/top-k-paper/check

# 表 5、附录 C.4：重选修复与成对提取
python compare.py fix --output /tmp/top-k-paper/fix

# 表 9：不同行数下的链
python compare.py rows --output /tmp/top-k-paper/rows

# 表 10：不同 k 下的结果
python compare.py ks --output /tmp/top-k-paper/ks

# 表 6、7、8、11：三方对照的计时
python compare.py time --output /tmp/top-k-paper/time

# 表 12：手写折叠后数秩
python folded_select.py shell /tmp/top-k-paper/handwritten.json

# 表 16：占位 intrinsic，编译后替换成机器指令
python folded_interval.py placeholder --rows 8

# 表 16：替换机器指令后重新排程
python folded_interval.py rescheduled --rows 8

# 表 16、17：由打过补丁的编译流程生成 intrinsic
python folded_interval.py compiler

# 表 16、17：在编译流程内按实测时序再次重排
python folded_interval.py reorder

# 表 13、23：k = 8，逐个测量 8 至 128 行；可把行数分给不同芯片
python transpose_merge.py $(seq 8 128) --output /tmp/top-k-paper/transpose

# 表 24：k = 16，测量 8、16、24、…、128 行
python transpose_merge.py --k 16 $(seq 8 8 128) --output /tmp/top-k-paper/transpose16

# 表 20：近似 top-k 的正确性
python approx.py check --output /tmp/top-k-paper/approx

# 表 21：近似 top-k 的计时
python approx.py time --output /tmp/top-k-paper/approx

# 6.3、6.5 节：两种合并的指令条数
python merge_profile.py 16 128

# 表 1：存储容量
python memory_capacity.py

# 表 2：指令语义
python semantics.py

# 表 3：指令延迟
python latency.py

# 3.3 节：LCC 的计数频率
python clock_rate.py

# 编译器自己的延迟表
python latency_probe.py compare.py time --cases basic32 --output /tmp/top-k-paper/probe
```

## 生成报告与论文

`tables.py` 从 `results/` 读取归档记录并填入 `report.template.md`，生成中文报告 `report.md`；英文报告 `report.en.md` 是中文报告的翻译。`figures.py` 生成中英文 SVG。生成表格需要 NumPy，生成 SVG 只依赖 Python 标准库，这两步都不需要 TPU。

```sh
# 从归档记录生成中文报告及其中的表格
python tables.py

# 从归档记录生成 figures/ 和 figures/en/ 中的插图
python figures.py
```

编译论文需要带 `acmart` 文档类的 TeX 环境、XeLaTeX 和 `latexmk`，以及 SVG 导出工具 `rsvg-convert`。插图使用 Linux Libertine、Inconsolata 和 Noto Serif CJK SC；Ubuntu 对应的导出与字体软件包为 `librsvg2-bin`、`fonts-linuxlibertine`、`fonts-inconsolata`、`fonts-noto-cjk`。

```sh
# 将英文报告转换为 LaTeX，并将英文 SVG 转换为嵌入字体的 PDF
python paper/md2tex.py

# 编译论文，PDF 和中间文件输出到 /tmp/top-k-paper/pdf/
latexmk -cd -xelatex -outdir=/tmp/top-k-paper/pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```
