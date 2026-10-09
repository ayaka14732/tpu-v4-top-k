"""本文用到的各种延迟怎样测得：把手写的几条指令插进一个载体 kernel，前后各读一次设备上的周期计数器 LCC。

每个实验的片段是：读数 R0、被测指令、读数 R1、sfence、读数 R2。向量指令严格按序发射，sfence 等此前的向量指令全部发射完，所以 R2 - R0 是这段指令从第一条到最后一条发射完的周期数，其中有固定的开销（见第一行的空片段）。
结果要等的指令（取回 XLU 的结果、读刚写过的 tile）会被硬件挡到结果就绪，延迟就体现在读数里：把“提交、取回”与空片段的读数相减，就是提交到可以取回的周期数；同一个队列上连着提交 n 次，读数随 n 的增量就是相邻两次提交的间隔。
"""
import chip
from tpuasm import tools as asm
from tpuasm.tools import bundle as b, read_lcc

def rotations(units: list[int]) -> str:
    """按给定的 XLU 编号依次提交旋转，再按提交的顺序取回。编号的最低位决定结果进哪个队列。"""
    pushes = ''.join(b(f'vx0: vrot.{unit}.8x128 trf{unit & 1}, v10, 8') for unit in units)
    return pushes + ''.join(b(f'vr0: vpop.8x128 v1, trf{unit & 1}') for unit in units)

def main() -> None:
    probe = asm.LccProbe()
    nop = b('misc: vnop')
    pop = b('vr0: vpop.8x128 v1, trf0')
    both = b('vr0: vpop.8x128 v1, trf0 ; vr1: vpop.8x128 v2, trf1') * 8
    two_queues = b('vx0: vrot.0.8x128 trf0, v10, 8 ; vx1: vrot.1.8x128 trf1, v10, 16') * 8 + both
    fold = b('vx0: vsxpose.0.start.end.8x128 trf0, v10, 8') + b('vr0: vpop.8x128 v3, trf0')
    push = lambda unit: b(f'vx0: vsxpose.{unit}.start.end.8x128 trf{unit & 1}, v10, 8')
    take = lambda unit, register=3: b(f'vr0: vpop.8x128 v{register}, trf{unit & 1}')
    tests = {
        # 固定开销与逐元素指令。
        'empty': '',
        '16 vnop': nop * 16,
        '16 dependent vadd': b('va0: vadd.8x128.s32 v1, 1, v1') * 16,
        '16 dependent vrot.slane.down': b('va0: vrot.slane.down.8x128.u32 v1, v1') * 16,
        '8 x (vgt, vsel on that mask)': (b('va0: vgt.8x128.s32 vm0, v1, v10') + b('va0: vsel.8x128 v1, vm0, v10, v1')) * 8,
        # XLU：提交到取回，以及同一个队列上连着提交的间隔。
        'vmax.xlane, 1 push 1 pop': b('vx0: vmax.xlane.0.8x128.f32 trf0, v9') + pop,
        'vmax.index.xlane, 1 push 1 pop': b('vx0: vmax.index.xlane.0.8x128.f32 trf0, v9') + pop,
        'vadd.xlane, 1 push 1 pop': b('vx0: vadd.xlane.0.8x128.f32 trf0, v9') + pop,
        'vmax.xlane, 2 pushes 2 pops on one queue': b('vx0: vmax.xlane.0.8x128.f32 trf0, v9') * 2 + pop * 2,
        'vmax.xlane, 8 pushes 8 pops on one queue': b('vx0: vmax.xlane.0.8x128.f32 trf0, v9') * 8 + pop * 8,
        'vmax.xlane, 16 pushes 16 pops on two queues': b('vx0: vmax.xlane.0.8x128.f32 trf0, v9 ; vx1: vmax.xlane.1.8x128.f32 trf1, v9') * 8 + both,
        'vrot, 1 push 1 pop': rotations([0]),
        'vrot, 8 pushes 8 pops on unit 0': rotations([0] * 8),
        'vrot, 8 pushes alternating units 0 and 2 (same queue)': rotations([0, 2] * 4),
        'vrot, 8 pushes alternating units 0 and 1 (two queues)': rotations([0, 1] * 4),
        'vrot, 16 pushes on two queues': two_queues,
        # 分段转置：延迟，以及它之后队列被占用多久。
        'vsxpose width 8, 1 push 1 pop': fold,
        'vsxpose then 16 vrot on two queues': fold + two_queues,
        'vsxpose, 80 vnop, then 16 vrot on two queues': fold + nop * 80 + two_queues,
        'vsxpose on unit 0, then 4 vrot on unit 0': fold + rotations([0] * 4),
        'vsxpose on unit 0, then 4 vrot on unit 1': fold + rotations([1] * 4),
        'vsxpose on unit 0, 100 vnop, then 4 vrot on unit 0': fold + nop * 100 + rotations([0] * 4),
        'two vsxpose on unit 0': push(0) * 2 + take(0) + take(0, 4),
        'two vsxpose on units 0 and 1 (two queues)': push(0) + push(1) + take(0) + take(1, 4),
        'four vsxpose on unit 0': push(0) * 4 + take(0) * 4,
        # TC VMEM 的写后读与 IAR。
        '8 x (vst tile A, vld tile A)': (b('vst: vst.8x128 [vmem:0xf0], v1') + b('vld: vld.8x128 v1, [vmem:0xf0]')) * 8,
        '8 x (vst tile A, vld tile B)': (b('vst: vst.8x128 [vmem:0xf0], v1') + b('vld: vld.8x128 v2, [vmem:0xe0]')) * 8,
        '8 x (vsetiar, vld.iar)': (b('vst: vsetiar.raw iar0, v8') + b('vld: vld.iar0.8x128 v1, [vmem:0xf0]')) * 8,
        '8 x (vst tile B, vld.iar tile A)': (b('vst: vst.8x128 [vmem:0xe0], v2') + b('vld: vld.iar0.8x128 v1, [vmem:0xf0]')) * 8,
        '8 x vld.iar tile A': b('vld: vld.iar0.8x128 v1, [vmem:0xf0]') * 8,
    }
    setup = b('va0: vimm.8x128.s32 v8, 0') + b('va0: vimm.8x128.s32 v1, 0 ; va1: vimm.8x128.s32 v2, 0') + b('va0: vcvt.8x128.s32.f32 v9, v10') + b('vst: vst.8x128 [vmem:0xf0], v1') + b('vst: vsetiar.raw iar0, v8')
    for name, body in tests.items():
        readings = sorted({int(row[1]) for row in probe.run(read_lcc(20) + body + read_lcc(21) + b('s0: sfence') + read_lcc(22), setup=setup).tolist()})
        print(f'{name}: R2 - R0 = {readings}, bundles = {body.count("{")}', flush=True)
    # 整体转置：一个 TC VREG 转成 16 个。第一个结果取回之后读一次数，16 个都取回之后再读一次；中间那次读数自己带一条 sfence，多占 7 个周期左右。
    body = read_lcc(20) + b('vx0: vxpose.0.start.end.8x128 trf0, v10, 128') + pop + b('s0: sfence') + read_lcc(21) + pop * 15 + b('s0: sfence') + read_lcc(22)
    readings = sorted({tuple(int(value) for value in row) for row in probe.run(body, setup=setup).tolist()})
    print(f'vxpose width 128, 1 push 16 pops: (R1 - R0 after the first pop, R2 - R0 after all) = {readings}', flush=True)

if __name__ == '__main__':
    main()
