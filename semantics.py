"""本文用到而 Mosaic 不生成的四条指令的语义怎样确认：在设备上执行，把结果与 NumPy 写的模型逐元素比较。

指令的名字、操作数和编码来自 tpuasm 的指令索引（它由公开发行的 libtpu 中内嵌的 ISA 描述生成）；名字只是猜测语义的线索，下面每个模型都是先猜、再用随机输入在真机上核对得到的。
片段插在一个载体 kernel 里：输入 x 是 u32[8,128]，已经在 v10 和 TC VMEM 的第 0 个 tile 中；片段把结果写到第 1 个 tile，载体把它送回主机。
"""
import chip
import jax
import numpy as np
from tpuasm import tools as asm
from tpuasm.tools import bundle as b

GAP = b('misc: vnop') * 10
# v3 是子通道号 s（0 到 7）：vlaneseq 给出 128 s + l。
SUBLANE = b('va0: vlaneseq.8x128.u32 v1') + b('va1: vshrl.8x128.s32 v3, v1, 7')
STORE = GAP + b('vst: vst.8x128 [vmem:0x8], v1') + GAP

def fold(x: np.ndarray) -> tuple[str, np.ndarray]:
    """vsxpose 宽度 8：一个 TC VREG 内 16 个 8×8 的小块各自转置，y[s, 8g + r] = x[r, 8g + s]。"""
    body = b('vx0: vsxpose.0.start.end.8x128 trf0, v10, 8') + b('vr0: vpop.8x128 v1, trf0')
    return body + STORE, x.reshape(8, 16, 8).transpose(2, 1, 0).reshape(8, 128)

def shuffle(x: np.ndarray) -> tuple[str, np.ndarray]:
    """vld.sshfl：读一个 tile，第 s 个子通道取模式的第 s 个十六进制位指定的那一行。"""
    rows = [3, 3, 0, 7, 1, 6, 6, 2]
    pattern = sum(row << (4 * s) for s, row in enumerate(rows))
    return b(f'vld: vld.sshfl.8x128 v1, [vmem:0x0], 0x{pattern:x}') + STORE, x[rows]

def gather(x: np.ndarray) -> tuple[str, np.ndarray]:
    """vsetiar.raw 加 vld.iar：每个元素按自己的行偏移读，y[s, l] = M[A + s + offset[s, l], l]。这里让目的行为 x[s, l] 的低 3 位。"""
    body = SUBLANE + b('va0: vand.8x128.u32 v5, 7, v10') + b('va0: vsub.8x128.s32 v4, v5, v3')
    body += GAP + b('vst: vsetiar.raw iar0, v4') + GAP + b('vld: vld.iar0.8x128 v1, [vmem:0x0]')
    return body + STORE, np.take_along_axis(x, (x & 7).astype(np.int64), axis=0)

def scatter(x: np.ndarray) -> tuple[str, np.ndarray]:
    """vsetiar.raw 加 vst.iar：每个元素按自己的行偏移写，M[A + s + offset[s, l], l] = v[s, l]。这里的偏移是 7 - 2s，第 s 个子通道写到第 7 - s 行。"""
    body = SUBLANE + b('va0: vadd.8x128.s32 v6, v3, v3') + b('va0: vsub.8x128.s32 v6, 7, v6')
    body += GAP + b('vst: vsetiar.raw iar0, v6') + GAP + b('vst: vst.iar.8x128 [vmem:0x8], iar0, v10') + GAP
    return body, x[::-1]

def main() -> None:
    probe = asm.LccProbe()
    rng = np.random.default_rng(6605)
    for test in (fold, shuffle, gather, scatter):
        wrong = 0
        for _ in range(4):
            x = rng.integers(0, 1 << 32, (8, 128), dtype=np.uint64).astype(np.uint32)
            raw = probe.host.copy()
            raw[:8] = x
            probe.x = jax.device_put(raw)
            body, expected = test(x)
            wrong += int(np.count_nonzero(probe.run_tiles(body)[0, 0, 0] != expected))
        print(f'{test.__name__}: 4 random inputs, {wrong} of {4 * 1024} elements differ from the model', flush=True)

if __name__ == '__main__':
    main()
