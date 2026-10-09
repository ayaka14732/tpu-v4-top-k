"""表 1 的容量从哪里来：Megacore Shared CMEM 用地址回绕实测，HBM 读设备的 memory_stats()，再与 pltpu.get_tpu_info() 的报告对照。

CMEM 不能由 Pallas 分配，容量只能直接测。CMEM 以 512 B 的 granule 编址。片段插在一个载体 kernel 里：把一个 tile（记作 P）写到 CMEM 地址 0，把它的按位取反（记作 Q）写到 granule 地址 A，再把两处读回 TC VMEM 的第 1、2 个 tile，由载体送回主机。A 在容量之内时地址 0 仍是 P；A 超出容量、回绕到地址 0 时，P 被改写成 Q。
"""
import chip
import jax
from jax.experimental.pallas import tpu as pltpu
from tpuasm import tools as asm
from tpuasm.tools import bundle as b

GAP = b('misc: vnop') * 16
ADDRESSES = (0x1000, 0x20000, 0x3FFF8, 0x40000, 0x7FFF8, 0x80000)

def dma(destination: str, source: str) -> str:
    """搬 8 个 granule（一个 f32 tile）并等待完成。"""
    return b(f's0: dma.simple {destination}, {source}, length=8, dst_flag=[sflag:52]') + b('misc: vwait.ge [sflag:52], 8') + b('misc: vsyncadd.s32 [sflag:52], -8')

def body(address: int) -> str:
    return (
        # P 是输入的第 0 个 tile（v10），写到 TC VMEM 的 0x100；Q 是它的按位取反，写到 0x108。
        b('vst: vst.8x128 [vmem:0x100], v10')
        + b('va0: vxor.8x128.u32 v11, 0xffffffff, v10')
        + GAP
        + b('vst: vst.8x128 [vmem:0x108], v11')
        + GAP
        + b('s0: simm.s32 s20, 0')
        + b(f's0: simm.s32 s21, {address}')
        + b('s0: simm.s32 s22, 256')
        + b('s0: simm.s32 s23, 264')
        + b('s0: simm.s32 s24, 8')
        + b('s0: simm.s32 s26, 16')
        # P → CMEM 地址 0，Q → CMEM 地址 A；再把地址 0 读回第 1 个 tile，地址 A 读回第 2 个 tile。
        + dma('[cmem:s20]', '[vmem:s22]')
        + dma('[cmem:s21]', '[vmem:s23]')
        + dma('[vmem:s24]', '[cmem:s20]')
        + dma('[vmem:s26]', '[cmem:s21]')
        + b('s0: sfence')
    )

def main() -> None:
    probe = asm.LccProbe()
    pattern = probe.host[:8]
    for address in ADDRESSES:
        tiles = probe.run_tiles(body(address))[0, 0]
        print(f'CMEM A = 0x{address:x} ({address * 512 / 2**20:.3f} MiB): address A reads Q {bool((tiles[1] == ~pattern).all())}; address 0 still P {bool((tiles[0] == pattern).all())}, became Q {bool((tiles[0] == ~pattern).all())}', flush=True)
    for device in jax.local_devices():
        print(f'{device}: memory_stats bytes_limit = {device.memory_stats()["bytes_limit"]}')
    info = pltpu.get_tpu_info()
    print(f'get_tpu_info: vmem_capacity_bytes = {info.vmem_capacity_bytes}, smem_capacity_bytes = {info.smem_capacity_bytes}, cmem_capacity_bytes = {info.cmem_capacity_bytes}, hbm_capacity_bytes = {info.hbm_capacity_bytes}')

if __name__ == '__main__':
    main()
