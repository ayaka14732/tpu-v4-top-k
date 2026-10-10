# Bitwise-Correct and Faster Top-*k* on TPU v4

[中文](README.md)

This repository contains the algorithm implementations, correctness checks, performance experiments, and paper sources for top-*k* on a single TensorCore of TPU v4. Read the [English report](report.en.md) or the [Chinese report](report.md); the LaTeX source of the paper is in [paper/](paper/). The PDF of the paper is published on Zenodo: <https://doi.org/10.5281/zenodo.23287738>.

## Files

| File | Contents |
| --- | --- |
| [total_order.py](total_order.py) | Bitwise-correct formulations and the general dispatcher `fast` |
| [fixes.py](fixes.py) | Three formulations that fix only the reselection: naive masking, lifting, and MXU suffix ranking |
| [shell.py](shell.py) | The harness whose start and end are both in TC VMEM, with its boundary checks and timing |
| [pin_propagation.py](pin_propagation.py), [pin_propagation.cpp](pin_propagation.cpp) | An in-process fix that adds memory-space propagation after Pin coloring |
| [compare.py](compare.py) | Timing and correctness for the three-way comparison |
| [folded_select.py](folded_select.py) | Handwritten fold-and-rank |
| [asm_intrinsics.py](asm_intrinsics.py), [check_intrinsics.py](check_intrinsics.py) | Four placeholder intrinsics, the rewriter, and their checks |
| [folded_top_k.py](folded_top_k.py), [folded_interval.py](folded_interval.py) | Fold-and-rank written with the intrinsics, and its four ways of compilation inside the harness |
| [reschedule.py](reschedule.py), [issue_profile.py](issue_profile.py) | The post-compilation rescheduling tool and the calibration of its timing model |
| [transpose_merge.py](transpose_merge.py), [merge_profile.py](merge_profile.py) | Eight-way merge with a loser tree after a full transpose; instruction counts of the two merges |
| [libtpu_patch.py](libtpu_patch.py), [front_door.py](front_door.py), [vliw_reorder.py](vliw_reorder.py), [libtpu_latency.cpp](libtpu_latency.cpp), [libtpu_sublane_gather.cpp](libtpu_sublane_gather.cpp) | In-process patches to libtpu, the four intrinsics emitted by the compilation pipeline, and the in-pipeline reorder |
| [llo_op_probe.py](llo_op_probe.py), [tpu_op_probe.py](tpu_op_probe.py), [ra_cost.py](ra_cost.py) | Probes for each layer of the compilation pipeline, and the compiler's own register-pressure report |
| [memory_capacity.py](memory_capacity.py) | Capacity of each storage level: the measured wrap-around of CMEM and the allocation limit of HBM |
| [semantics.py](semantics.py), [latency.py](latency.py), [latency_probe.py](latency_probe.py) | Measurements of instruction semantics and latencies, and the compiler's latency table |
| [tables.py](tables.py), [report.template.md](report.template.md) | The template of the report, and the script that fills the numbers from `results/` into it |
| [approx.py](approx.py) | `approx_max_k` in Pallas and native XLA, and our formulations placed in its two stages |
| [figures.py](figures.py) | Generates the figures in `figures/` (Chinese) and `figures/en/` (English) from `results/` |
| [clock_rate.py](clock_rate.py) | The counting frequency of LCC |
| [chip.py](chip.py) | Selects the chip used by the process |

## Environment

The paper uses Python 3.14.7t, JAX `0.12.0.dev20261002+7fc69a22c2`, jaxlib `0.12.0.dev20261002`, libtpu `0.0.49`, and tpuasm `0.2.1`. The machine has the topology TPU v4 (2x2x2), with 2 hosts of 4 chips each; the experiments use only one TensorCore of one chip local to host 0.

`chip.py` selects the chip before JAX is imported and exposes the two TensorCores as separate devices. The environment variable `TOP_K_CHIP` selects the local chip and defaults to `1`. Processes on different chips can run at the same time; each of the four ways of compilation in `folded_interval.py` starts its own process, because the compiler patches apply to the whole process.

## Reproducing the Experiments

Run from the repository root in a Python environment with the dependencies above installed. The comments below give the table or section of the paper that each command corresponds to; output paths, where they can be given, use `/tmp/top-k-paper/`, and probes without an output argument print to the terminal. The raw records used in the paper are archived in `results/`.

```sh
# Create the output directory
mkdir -p /tmp/top-k-paper

# Tables 4 and 22: correctness of exact top-k and errors by input class
python compare.py check --output /tmp/top-k-paper/check

# Table 5 and Appendix C.4: reselection fixes and pairwise extraction
python compare.py fix --output /tmp/top-k-paper/fix

# Table 9: the chain at different numbers of rows
python compare.py rows --output /tmp/top-k-paper/rows

# Table 10: results for different k
python compare.py ks --output /tmp/top-k-paper/ks

# Tables 6, 7, 8, and 11: timing of the three-way comparison
python compare.py time --output /tmp/top-k-paper/time

# Table 12: handwritten fold-and-rank
python folded_select.py shell /tmp/top-k-paper/handwritten.json

# Table 16: placeholder intrinsics, replaced by machine instructions after compilation
python folded_interval.py placeholder --rows 8

# Table 16: rescheduling after the replacement
python folded_interval.py rescheduled --rows 8

# Tables 16 and 17: intrinsics emitted by the patched compilation pipeline
python folded_interval.py compiler

# Tables 16 and 17: one more reorder by measured timing inside the pipeline
python folded_interval.py reorder

# Tables 13 and 23: k = 8, every row count from 8 to 128; the row counts can be split across chips
python transpose_merge.py $(seq 8 128) --output /tmp/top-k-paper/transpose

# Table 24: k = 16, rows 8, 16, 24, ..., 128
python transpose_merge.py --k 16 $(seq 8 8 128) --output /tmp/top-k-paper/transpose16

# Table 20: correctness of approximate top-k
python approx.py check --output /tmp/top-k-paper/approx

# Table 21: timing of approximate top-k
python approx.py time --output /tmp/top-k-paper/approx

# Sections 6.3 and 6.5: instruction counts of the two merges
python merge_profile.py 16 128

# Table 1: storage capacities
python memory_capacity.py

# Table 2: instruction semantics
python semantics.py

# Table 3: instruction latencies
python latency.py

# Section 3.3: the counting frequency of LCC
python clock_rate.py

# The compiler's own latency table
python latency_probe.py compare.py time --cases basic32 --output /tmp/top-k-paper/probe
```

## Generating the Reports and the Paper

`tables.py` reads the archived records from `results/` and fills them into `report.template.md` to produce the Chinese report `report.md`; the English report `report.en.md` is a translation of the Chinese one. `figures.py` generates the SVGs in both languages. Generating the tables needs NumPy and generating the SVGs needs only the Python standard library; neither step needs a TPU.

```sh
# Generate the Chinese report and its tables from the archived records
python tables.py

# Generate the figures in figures/ and figures/en/ from the archived records
python figures.py
```

Compiling the paper needs a TeX environment with the `acmart` document class, XeLaTeX, and `latexmk`, plus the SVG exporter `rsvg-convert`. The figures use Linux Libertine, Inconsolata, and Noto Serif CJK SC; on Ubuntu the exporter and fonts are in the packages `librsvg2-bin`, `fonts-linuxlibertine`, `fonts-inconsolata`, and `fonts-noto-cjk`.

```sh
# Convert the English report to LaTeX and the English SVGs to PDFs with embedded fonts
python paper/md2tex.py

# Compile the paper; the PDF and intermediate files go to /tmp/top-k-paper/pdf/
latexmk -cd -xelatex -outdir=/tmp/top-k-paper/pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

## Citation

```bibtex
@misc{zhong2026topk,
  author = {Muhan Zhong},
  title = {Bitwise-Correct and Faster Top-k on {TPU} v4},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23287738},
  url = {https://doi.org/10.5281/zenodo.23287738},
}
```
