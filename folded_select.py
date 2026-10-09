"""TPU v4 f32[8,128] 的 top-8：折叠后数秩，不走八轮 XLU 链。

fragment() 是输入在 v10、结果写到 TC VMEM 两个 tile 的手写片段：vsxpose 把通道号的低三位换到子通道，
vst.iar 把每列按键降序排好，15 次跨通道旋转把其余各组对齐过来，vld.sshfl 与 vld.iar 在排好的列里二分数出名次，
三次 vadd.xlane 把前 8 名的位置按字段打包送回原布局，vst.iar 与 vld.iar 把它们的键按名次分开，八次 vmax.xlane 带回来。
"""
from pathlib import Path
import os
import sys
import chip
import json
import re
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from tpuasm import tools as asm
from tpuasm.tools import bundle as b, read_lcc

# 五个工作 tile 的行地址；探针载体的 0x80 起存着被借用的寄存器，所以从 0xd8 开始。program() 换成自己的 scratch。
TY, TZ, TP, T0, T1 = 0xd8, 0xe0, 0xe8, 0xf0, 0xf8
# 取值用的两个相邻 tile：前一个按名次存放前 8 名的代理键，后一个始终是负无穷。探针里借用返回区域中没有用到的行。
VT = [0x18, 0x20]
# 输入所在的 tile；k > 8 时取值的 vperm 要再读一次原输入。
INPUT = 0
# k > 8 时按名次存放原通道号的 tile（k/8 个，后面再跟一个不写的），送回下标时逐行读出。
UP = 0x28
# v0 排好的键 Z，v1 Z-1，v2 子通道号 S，v3 组号乘 8，v4/v5 二分用的行偏移 0-S、4-S，v6/v7 两个累加器，v8 零。
Z, ZM1, S, G8, K0, K4, ACC_A, ACC_B, ZERO = (f'v{i}' for i in range(9))
ACC_C = 'v27'
INT_MIN = '0x80000000'
CAPACITY = {'va': 2}

def pattern(shift: int) -> int:
    """vld.sshfl 的模式：第 s 个子通道读第 (s - shift) mod 8 行。"""
    return sum(((s - shift) % 8) << (4 * s) for s in range(8))

class Plan:
    """按依赖和槽位把指令排进 bundle。向量发射严格按序，一条指令等待时后面的全部跟着等，所以把等待（同一个 tile 的写后读、vsetiar、XLU 的结果）写成最小间隔，用别的指令填满。"""

    def __init__(self) -> None:
        self.ops: list[tuple[str, str, list[tuple[int, int]], int]] = []

    def add(self, slot: str, text: str, after: list[tuple[int | None, int]] = [], release: int = 0) -> int:
        self.ops.append((slot, text, [(op, gap) for op, gap in after if op is not None], release))
        return len(self.ops) - 1

    def emit(self) -> str:
        time: dict[int, int] = {}
        pending = list(range(len(self.ops)))
        text, now = '', 0
        while pending:
            used: dict[str, int] = {}
            alu: list[tuple[str, str]] = []
            items, placed = [], []
            for index in pending:
                slot, operation, after, release = self.ops[index]
                kind = 'va' if slot == 'va1' else slot
                # 一个 bundle 能读的 TC VREG 端口有限：两条 ALU 加一条 store 可以，再加一次 XLU 提交就会冲突。
                reads = 2 if kind == 'va' else 1 if slot in ('vst', 'vx0', 'vx1') else 0
                if release > now or used.get(kind, 0) >= CAPACITY.get(kind, 1) or used.get(slot, 0) >= CAPACITY.get(kind, 1) and slot == 'va1' or used.get('reads', 0) + reads > 5 or any(op not in time or time[op] + gap > now for op, gap in after):
                    continue
                if slot == 'va1' and used.get('va1', 0):
                    continue
                if kind == 'va':
                    alu.append((slot, operation))
                    used[slot] = used.get(slot, 0) + (slot == 'va1')
                else:
                    items.append(f'{slot}: {operation}')
                used[kind] = used.get(kind, 0) + 1
                used['reads'] = used.get('reads', 0) + reads
                placed.append(index)
            # 移位只能放在 va1；其余 ALU 指令两个槽都可以。
            alu.sort(key=lambda item: item[0] == 'va1')
            if len(alu) == 1 and alu[0][0] == 'va1':
                items.append(f'va1: {alu[0][1]}')
            else:
                items += [f'va{number}: {operation}' for number, (_, operation) in enumerate(alu)]
            for index in placed:
                time[index] = now
            pending = [index for index in pending if index not in time]
            text += b(' ; '.join(items) or 'misc: vnop')
            now += 1
        return text

def local_rank(plan: Plan, stored: int) -> int:
    """每个元素在自己那一列中的名次（同键时原子通道号小的在前），结果在 v11。返回最后一条指令。"""
    start = plan.add('va', f'veq.8x128.s32 vm6, {Z}, {INT_MIN}')
    minus = plan.add('va', f'vadd.8x128.s32 {ZM1}, -1, {Z}')
    last = plan.add('va', f'vsel.8x128 v11, vm6, {S}, {ZERO}', [(start, 1)])
    compare: list[int | None] = [None, None]
    for e in range(1, 8):
        other, threshold, m0, m1 = f'v{12 + e % 2}', f'v{14 + e % 2}', f'vm{e % 2}', f'vm{2 + e % 2}'
        load = plan.add('vld', f'vld.sshfl.8x128 {other}, [vmem:0x{TY:x}], 0x{pattern(e):x}', [(stored, 7), (compare[e % 2], 0)])
        earlier = plan.add('va', f'vge.8x128.s32 {m0}, {S}, {e}', [(compare[e % 2], 1)])
        select = plan.add('va', f'vsel.8x128 {threshold}, {m0}, {ZM1}, {Z}', [(earlier, 1), (minus, 1)])
        compare[e % 2] = plan.add('va', f'vgt.8x128.s32 {m1}, {other}, {threshold}', [(load, 1), (select, 1)])
        more = plan.add('va', f'vadd.8x128.s32 v16, 1, v11', [(last, 1)])
        last = plan.add('va', f'vsel.8x128 v11, {m1}, v16, v11', [(compare[e % 2], 1), (more, 1)])
    return last

def search(plan: Plan, n: int, d: int, queue: int, arrival: int, state: dict[str, dict[int, int | None]], exact: bool) -> None:
    """第 n 个取回的旋转结果（对方是组 g-d 排好的列，来自 trf{queue}）：在其中二分出排在自己前面的个数，0 到 7 记入累加器。

    整列 8 个都排在前面时，k <= 8 只需记入淘汰掩码 vm7；exact 为真时（k > 8）改为在第三个累加器上加一。
    """
    group, tile, iar = n % 3, (TY, TZ, T0, T1)[n % 4], n % 2
    threshold, a, bb, c, e, f = (f'v{9 + 6 * group + i}' for i in range(6))
    ma, mb = f'vm{2 * group}', f'vm{2 * group + 1}'
    done = state['done'].get(n - 3)
    pop = plan.add(f'vr{queue}', f'vpop.8x128 {c}, trf{queue}', [(state['pop'].get(queue), 1), (state['merge'].get(n - 3), 1)], arrival)
    store = plan.add('vst', f'vst.8x128 [vmem:0x{tile:x}], {c}', [(pop, 1), (state['gather'].get(n - 4), 1)])
    earlier = plan.add('va', f'vge.8x128.s32 {ma}, {G8}, {8 * d}', [(done, 1), (store, 0)])
    select = plan.add('va', f'vsel.8x128 {threshold}, {ma}, {ZM1}, {Z}', [(earlier, 1)])
    load7 = plan.add('vld', f'vld.sshfl.8x128 {a}, [vmem:0x{tile:x}], 0x77777777', [(store, 7), (done, 1)])
    load3 = plan.add('vld', f'vld.sshfl.8x128 {bb}, [vmem:0x{tile:x}], 0x33333333', [(store, 7), (done, 1)])
    full = plan.add('va', f'vgt.8x128.s32 {ma}, {a}, {threshold}', [(load7, 1), (select, 1)])
    if exact:
        whole = plan.add('va', f'vadd.8x128.s32 {f}, 1, {ACC_C}', [(state['whole'].get(n - 1), 1), (done, 1)])
        dead = state['whole'][n] = plan.add('va', f'vsel.8x128 {ACC_C}, {ma}, {f}, {ACC_C}', [(full, 1), (whole, 1)])
    else:
        dead = plan.add('misc', f'vmor.8x128.u1 vm7, vm7, {ma}', [(full, 1)])
    half = plan.add('va', f'vgt.8x128.s32 {mb}, {bb}, {threshold}', [(load3, 1), (select, 1)])
    load1 = plan.add('vld', f'vld.sshfl.8x128 {a}, [vmem:0x{tile:x}], 0x11111111', [(full, 0)])
    load5 = plan.add('vld', f'vld.sshfl.8x128 {c}, [vmem:0x{tile:x}], 0x55555555', [(store, 7)])
    merge = plan.add('va', f'vsel.8x128 {bb}, {mb}, {c}, {a}', [(half, 1), (load1, 1), (load5, 1)])
    base = plan.add('va', f'vsel.8x128 {e}, {mb}, {K4}, {K0}', [(half, 1)])
    quarter = plan.add('va', f'vgt.8x128.s32 {ma}, {bb}, {threshold}', [(merge, 1), (dead, 1)])
    upper = plan.add('va', f'vadd.8x128.s32 {f}, 2, {e}', [(base, 1), (dead, 1)])
    offset = plan.add('va', f'vsel.8x128 {e}, {ma}, {f}, {e}', [(quarter, 1), (upper, 1)])
    set_iar = plan.add('vst', f'vsetiar.raw iar{iar}, {e}', [(offset, 1), (state['gather'].get(n - 2), 1)])
    gather = plan.add('vld', f'vld.iar{iar}.8x128 {a}, [vmem:0x{tile:x}]', [(set_iar, 7)])
    last = plan.add('va', f'vgt.8x128.s32 {ma}, {a}, {threshold}', [(gather, 1), (offset, 1)])
    plan.add('va', f'vadd.8x128.s32 {ACC_A}, {ACC_A}, {e}', [(offset, 1)])
    more = plan.add('va', f'vadd.8x128.s32 {f}, 1, {ACC_B}', [(state['done'].get(n - 1), 1), (offset, 0)])
    state['done'][n] = plan.add('va', f'vsel.8x128 {ACC_B}, {ma}, {f}, {ACC_B}', [(last, 1), (more, 1)])
    state['pop'][queue], state['merge'][n], state['gather'][n] = pop, merge, gather

def fragment(values: int, indices: int, *, k: int = 8, stop: str = '', dma: dict[str, tuple[str, str]] = {}) -> str:
    """输入位型在 v10（也在 [vmem:0x0]）；值写到 [vmem:values]，下标写到 [vmem:indices]，只写前 k 个通道。k 可以是 8、16 或 32。

    dma 给出两次输出 DMA 各自的两条 s0 槽指令（'indices'、'values'）：program() 用它把输出 DMA 提前到片段内部。
    """
    assert k in (8, 16, 32), k
    text = b('va0: vlt.8x128.s32 vm0, v10, 0 ; va1: vxor.8x128.u32 v0, 0x7fffffff, v10')
    text += b('va0: vsel.8x128 v0, vm0, v0, v10 ; va1: vlaneseq.8x128.u32 v9')
    text += b(f'vx0: vsxpose.0.start.end.8x128 trf0, v0, 8 ; va0: vand.8x128.u32 {G8}, 120, v9 ; va1: vshrl.8x128.s32 {S}, v9, 7')
    text += b(f'va0: vimm.8x128.s32 {ZERO}, 0 ; va1: vsub.8x128.s32 {K0}, 0, {S}')
    # 累加的行偏移是 c - S，共 15 次；自己那一列里的名次是 S。两项合起来是 16 S，放进累加器的初值。
    text += b(f'va0: vadd.8x128.s32 {K4}, 4, {K0} ; va1: vshll.8x128.s32 {ACC_A}, {S}, 4')
    # 折叠：Y[s, 8g+r] 是第 r 行第 8g+s 个元素的键。先数出每个元素在自己那一列中的名次。
    text += b('vr0: vpop.8x128 v0, trf0')
    plan = Plan()
    stored = plan.add('vst', f'vst.8x128 [vmem:0x{TY:x}], v0')
    ranked = local_rank(plan, stored)
    # 按名次把键和原子通道号写到对应的行：每列从上到下降序。
    offset = plan.add('va', f'vsub.8x128.s32 v13, v11, {S}', [(ranked, 1)])
    set_iar = plan.add('vst', 'vsetiar.raw iar0, v13', [(offset, 1)])
    key = plan.add('vst', f'vst.iar.8x128 [vmem:0x{TZ:x}], iar0, v0', [(set_iar, 1)])
    plan.add('vst', f'vst.iar.8x128 [vmem:0x{TP:x}], iar0, {S}', [(key, 1)])
    plan.add('vld', f'vld.8x128 v0, [vmem:0x{TZ:x}]', [(key, 7)])
    text += plan.emit()
    if stop == 'sorted':
        return text + b(f'vst: vst.8x128 [vmem:0x{values:x}], v0') + b(f'vld: vld.8x128 v1, [vmem:0x{TP:x}]') + b(f'vst: vst.8x128 [vmem:0x{indices:x}], v1')
    # vsxpose 之后它所在的 XLU 还要忙约一百个周期：前面的旋转都交给另一个 XLU，每 8 个周期提交一次；它空出来之后两边交替。
    plan = Plan()
    free = 70
    last = {0: None, 1: None}
    ready = {0: free, 1: 0}
    arrivals = []
    for d in range(1, 16):
        queue = 0 if ready[0] < ready[1] else 1
        last[queue] = plan.add(f'vx{queue}', f'vrot.{queue}.8x128 trf{queue}, v0, {8 * d}', [(last[queue], 8)], ready[queue])
        arrivals.append((ready[queue] + 70, d, queue))
        ready[queue] += 8
    minus = plan.add('va', f'vadd.8x128.s32 {ZM1}, -1, {Z}')
    lowest = plan.add('va', f'veq.8x128.s32 vm7, {Z}, {INT_MIN}')
    # 最小键减一会回绕：它在组 g >= 1 时前面至少有一整列，直接淘汰；组 0 不用减一的阈值。
    later = plan.add('va', f'vge.8x128.s32 vm6, {G8}, 8')
    zeroed = plan.add('va', f'vimm.8x128.s32 {ACC_B}, 0')
    cleared = None
    if k > 8:
        cleared = plan.add('va', f'vimm.8x128.s32 {ACC_C}, 0')
        for t in range(k // 8):
            plan.add('vst', f'vst.8x128 [vmem:0x{UP + 8 * t:x}], {ZERO}')
    else:
        # 等旋转结果的空档里准备取值用的东西：键的低 31 位换成一个有限的正常浮点数（v27），两个 tile 填成负无穷。
        dead = plan.add('misc', 'vmand.8x128.u1 vm7, vm7, vm6', [(lowest, 1), (later, 1)])
        payload = plan.add('va', f'vand.8x128.u32 v28, 0x3fffffff, {Z}')
        top = plan.add('va', f'vand.8x128.u32 v29, 0x40000000, {Z}')
        upper = plan.add('va', 'vne.8x128.s32 vm6, v29, 0', [(top, 1), (dead, 1)])
        positive = plan.add('va', 'vadd.8x128.s32 v30, 8388608, v28', [(payload, 1)])
        negative = plan.add('va', 'vsub.8x128.s32 v31, 1082130431, v28', [(payload, 1)])
        negative = plan.add('va', 'vor.8x128.u32 v31, 0x80000000, v31', [(negative, 1)])
        plan.add('va', 'vsel.8x128 v27, vm6, v30, v31', [(upper, 1), (positive, 1), (negative, 1)])
        fill = plan.add('va', 'vimm.8x128.s32 v28, -8388608', [(positive, 1), (negative, 1)])
        for tile in VT:
            plan.add('vst', f'vst.8x128 [vmem:0x{tile:x}], v28', [(fill, 1)])
    state: dict[str, dict[int, int | None]] = {'pop': {}, 'merge': {}, 'gather': {}, 'done': {-1: zeroed}, 'whole': {-1: cleared}}
    for n, (arrival, d, queue) in enumerate(sorted(arrivals)):
        search(plan, n, d, queue, arrival, state, k > 8)
    text += plan.emit()
    if k > 8:
        return text + gather_tail(values, indices, k, stop, dma)
    if stop == 'rank':
        text += b(f'va0: vadd.8x128.s32 v11, {ACC_A}, {ACC_B} ; vld: vld.8x128 v12, [vmem:0x{TP:x}]')
        return text + b('va0: vsel.8x128 v11, vm7, 99, v11') + b(f'vst: vst.8x128 [vmem:0x{values:x}], v11') + b(f'vst: vst.8x128 [vmem:0x{indices:x}], v12')
    plan = Plan()
    lane = plan.add('va', 'vlaneseq.8x128.u32 v9')
    rank = plan.add('va', f'vadd.8x128.s32 v11, {ACC_A}, {ACC_B}')
    payload = plan.add('vld', f'vld.8x128 v12, [vmem:0x{TP:x}]')
    survive = plan.add('misc', 'vmneg.8x128.u1 vm7, vm7')
    residue = plan.add('va', 'vand.8x128.u32 v14, 7, v9', [(lane, 1)])
    # 前 8 名把代理键写到自己那一列的第 t 行（t 是名次）；其余元素不写，但目的行仍要与放行的错开（见 gather_tail 的说明），统一指到下一个 tile。
    top = plan.add('va', 'vlt.8x128.s32 vm2, v11, 8', [(rank, 1)])
    top = plan.add('misc', 'vmand.8x128.u1 vm2, vm2, vm7', [(top, 1), (survive, 1)])
    rest = plan.add('misc', 'vmneg.8x128.u1 vm5, vm2', [(top, 1)])
    row = plan.add('va', f'vsub.8x128.s32 v28, v11, {S}', [(rank, 1)])
    row = plan.add('va', 'vsel.8x128 v28, vm5, 8, v28', [(row, 1), (rest, 1)])
    set_iar = plan.add('vst', 'vsetiar.raw iar0, v28', [(row, 1)])
    stored = plan.add('vst', f'vst.iar.msk.8x128 [vmem:0x{VT[0]:x}], iar0, vm2, v27', [(set_iar, 7)])
    # 再按名次逐个读回：第 r 个子通道在通道号模 8 等于 r 的列上读第 j 行，其余位置读到负无穷。读回的第 r 行里只有第 r 行的第 j 名，一次 vmax.xlane 就把它带回原布局。
    same = plan.add('va', f'veq.8x128.s32 vm6, v14, {S}', [(residue, 1)])
    away = plan.add('va', f'vsub.8x128.s32 v31, 8, {S}')
    home = plan.add('va', f'vsub.8x128.s32 {K4}, 0, {S}')
    step = plan.add('va', f'vsel.8x128 {K0}, vm6, 1, {ZERO}', [(same, 1)])
    offset = plan.add('va', f'vsel.8x128 {K4}, vm6, {K4}, v31', [(same, 1), (away, 1), (home, 1)])
    order = [('value', j) for j in range(5)] + [('word', k) for k in range(3)] + [('value', j) for j in range(5, 8)]
    queue = {item: number % 2 for number, item in enumerate(order)}
    ready: dict[tuple[str, int], tuple[int, str]] = {}
    gathers: dict[int, int] = {}
    for j in range(8):
        register = f'v{29 + j % 2}'
        set_j = plan.add('vst', f'vsetiar.raw iar{j % 2}, {K4}', [(offset, 1), (gathers.get(j - 2), 1), (stored, 1)])
        offset = plan.add('va', f'vadd.8x128.s32 {K4}, {K4}, {K0}', [(set_j, 1), (step, 1)])
        gathers[j] = plan.add('vld', f'vld.iar{j % 2}.8x128 {register}, [vmem:0x{VT[0]:x}]', [(set_j, 7), (stored, 7)])
        ready['value', j] = (gathers[j], register)
    # 下标：名次 t < 8 的元素把 原通道号（负键再加 128）左移 8·(t mod 3) 位，放进第 t // 3 个字；三个字节互不重叠，沿通道求和是精确的。
    field = plan.add('va', f'vadd.8x128.s32 v12, {G8}, v12', [(payload, 1)])
    sign = plan.add('va', f'vlt.8x128.s32 vm0, {Z}, 0')
    flagged = plan.add('va', 'vor.8x128.u32 v13, 128, v12', [(field, 1)])
    field = plan.add('va', 'vsel.8x128 v12, vm0, v13, v12', [(flagged, 1), (sign, 1)])
    less3 = plan.add('va', 'vadd.8x128.s32 v13, -3, v11', [(rank, 1), (field, 0)])
    less6 = plan.add('va', 'vadd.8x128.s32 v15, -6, v11', [(rank, 1)])
    second = plan.add('va', 'vge.8x128.s32 vm0, v11, 3', [(rank, 1), (field, 1)])
    third = plan.add('va', 'vge.8x128.s32 vm1, v11, 6', [(rank, 1)])
    byte = plan.add('va', 'vsel.8x128 v13, vm0, v13, v11', [(less3, 1), (second, 1)])
    byte = plan.add('va', 'vsel.8x128 v13, vm1, v15, v13', [(byte, 1), (less6, 1), (third, 1)])
    shift = plan.add('va1', 'vshll.8x128.s32 v13, v13, 3', [(byte, 1)])
    shifted = plan.add('va1', 'vshll.8x128.s32 v12, v12, v13', [(shift, 1), (field, 1)])
    kept = plan.add('va', f'vsel.8x128 v12, vm2, v12, {ZERO}', [(shifted, 1), (top, 1)])
    words = [plan.add('va', f'vsel.8x128 v16, vm0, {ZERO}, v12', [(kept, 1)])]
    middle = plan.add('va', f'vsel.8x128 v17, vm0, v12, {ZERO}', [(kept, 1)])
    words.append(plan.add('va', f'vsel.8x128 v17, vm1, {ZERO}, v17', [(middle, 1)]))
    words.append(plan.add('va', f'vsel.8x128 v18, vm1, v12, {ZERO}', [(kept, 1)]))
    for k, (word, spare) in enumerate((('v16', 'v19'), ('v17', 'v20'), ('v18', 'v21'))):
        # 沿子通道求和：子通道旋转在向量 ALU 中完成，结果两个周期后可用；移 1、2、4 行各加一次。
        last, gap = words[k], 1
        for amount in (1, 2, 4):
            source = word
            for _ in range(amount):
                last = plan.add('va', f'vrot.slane.down.8x128.u32 {spare}, {source}', [(last, gap)])
                source, gap = spare, 2
            last = plan.add('va', f'vadd.8x128.s32 {word}, {spare}, {word}', [(last, 2)])
            gap = 1
        # 只留下通道号模 8 等于子通道号的那一类，交给 XLU 沿通道求和。
        last = plan.add('va', f'vsel.8x128 {word}, vm6, {word}, {ZERO}', [(last, 1), (same, 1)])
        ready['word', k] = (plan.add('va', f'vcvt.8x128.s32.f32 {word}, {word}', [(last, 1)]), word)
    built = words + [middle]
    # 解码用的常数只与通道号有关：第 j 个通道取第 j // 3 个字的第 j % 3 个字节。
    column = plan.add('va', 'vand.8x128.u32 v22, 127, v9', [(lane, 1)])
    low = plan.add('va', 'vlt.8x128.s32 vm0, v22, 3', [(column, 1)] + [(op, 1) for op in built])
    high = plan.add('va', 'vlt.8x128.s32 vm1, v22, 6', [(column, 1)] + [(op, 1) for op in built])
    valid = plan.add('va', 'vlt.8x128.s32 vm2, v22, 8', [(column, 1), (kept, 1), (stored, 1)])
    less3 = plan.add('va', 'vadd.8x128.s32 v23, -3, v22', [(column, 1)])
    less6 = plan.add('va', 'vadd.8x128.s32 v24, -6, v22', [(column, 1)])
    byte = plan.add('va', 'vsel.8x128 v23, vm0, v22, v23', [(less3, 1), (low, 1)])
    byte = plan.add('va', 'vsel.8x128 v23, vm1, v23, v24', [(byte, 1), (less6, 1), (high, 1)])
    shift = plan.add('va1', 'vshll.8x128.s32 v23, v23, 3', [(byte, 1)])
    # 十一次 XLU 提交交替分给两个队列，每个队列 8 个周期一次；归约 79 个周期后可以取回。
    last_push: dict[int, int | None] = {0: None, 1: None}
    last_pop: dict[int, int | None] = {0: None, 1: None}
    pops: dict[tuple[str, int], int] = {}
    for item in order:
        q = queue[item]
        source, register = ready[item]
        operation = 'vmax.xlane' if item[0] == 'value' else 'vadd.xlane'
        last_push[q] = plan.add(f'vx{q}', f'{operation}.{q}.8x128.f32 trf{q}, {register}', [(source, 1), (last_push[q], 8)])
        if item[0] == 'value' and item[1] + 2 in gathers:
            # 两个读回用的寄存器轮流使用：下下个读回要等这一次提交读完寄存器。
            plan.ops[gathers[item[1] + 2]][2].append((last_push[q], 1))
        # 结果依次取进：三个字回到 v16-v18，名次 0 的值直接进 v26，其余各值先进 v12/v15 再按通道选入 v26。
        target = f'v{16 + item[1]}' if item[0] == 'word' else 'v26' if item[1] == 0 else f'v{12 + 3 * q}'
        after = [(last_push[q], 80), (last_pop[q], 8)]
        if item[0] == 'value' and item[1] > 0:
            after += [(pops.get(('merged', q)), 0), (kept, 1)]
        last_pop[q] = pops[item] = plan.add(f'vr{q}', f'vpop.8x128 {target}, trf{q}', after)
        if item[0] == 'value' and item[1] > 0:
            mask = f'vm{3 + q}'
            chosen = plan.add('va', f'veq.8x128.s32 {mask}, v22, {item[1]}', [(column, 1), (rest, 1), (pops.get(('merged', q)), 1)])
            pops['merged', q] = pops['assembled'] = plan.add('va', f'vsel.8x128 v26, {mask}, {target}, v26', [(chosen, 1), (pops[item], 1), (pops['value', 0], 1), (pops.get('assembled'), 1)])
    last = pops['value', 0]
    for line in (*dma.get('indices', ()), *dma.get('values', ())):
        last = plan.add('s0', line, [(last, 1)])
    word = plan.add('va', 'vsel.8x128 v17, vm1, v17, v18', [(pops['word', 1], 1), (pops['word', 2], 1), (high, 1)])
    word = plan.add('va', 'vsel.8x128 v16, vm0, v16, v17', [(word, 1), (pops['word', 0], 1), (low, 1)])
    word = plan.add('va', 'vcvt.8x128.f32.s32 v16, v16, 0xffffffff', [(word, 1)])
    word = plan.add('va1', 'vshrl.8x128.s32 v16, v16, v23', [(word, 1), (shift, 1)])
    word = plan.add('va', 'vand.8x128.u32 v16, 255, v16', [(word, 1)])
    index = plan.add('va', 'vand.8x128.u32 v25, 127, v16', [(word, 1)])
    plan.add('vst', f'vst.msk.8x128 [vmem:0x{indices:x}], vm2, v25', [(index, 1), (valid, 1)])
    sign = plan.add('va', 'vge.8x128.s32 vm6, v16, 128', [(word, 1)] + [(ready['word', k][0], 1) for k in range(3)])
    # 代理键换回低 31 位，补上符号位，再从键换回原来的位型。
    final = pops['assembled']
    magnitude = plan.add('va', 'vand.8x128.u32 v19, 0x7fffffff, v26', [(final, 1)] + [(ready['word', k][0], 1) for k in range(3)])
    lower = plan.add('va', 'vlt.8x128.s32 vm3, v26, 0', [(final, 1)])
    first = plan.add('va', 'vsub.8x128.s32 v20, 1082130431, v19', [(magnitude, 1)])
    other = plan.add('va', 'vadd.8x128.s32 v21, -8388608, v19', [(magnitude, 1)])
    other = plan.add('va', 'vor.8x128.u32 v21, 0x40000000, v21', [(other, 1)])
    key = plan.add('va', 'vsel.8x128 v19, vm3, v20, v21', [(first, 1), (other, 1), (lower, 1)])
    flipped = plan.add('va', 'vor.8x128.u32 v20, 0x80000000, v19', [(key, 1)])
    flipped = plan.add('va', 'vxor.8x128.u32 v20, 0x7fffffff, v20', [(flipped, 1)])
    value = plan.add('va', 'vsel.8x128 v19, vm6, v20, v19', [(flipped, 1), (sign, 1)])
    plan.add('vst', f'vst.msk.8x128 [vmem:0x{values:x}], vm2, v19', [(value, 1), (valid, 1)])
    return text + plan.emit()

def gather_tail(values: int, indices: int, k: int, stop: str, dma: dict[str, tuple[str, str]]) -> str:
    """k > 8 的送回阶段：下标按字段打包，每三个名次一个字，共 ceil(k/3) 次 vadd.xlane；值在下标回来之后用一次 vperm 从原输入取。"""
    plan = Plan()
    lane = plan.add('va', 'vlaneseq.8x128.u32 v9')
    rank = plan.add('va', f'vadd.8x128.s32 v11, {ACC_A}, {ACC_B}')
    rank = plan.add('va', f'vadd.8x128.s32 v11, v11, {ACC_C}', [(rank, 1)])
    # 最小键减一回绕后，前面 g 个组的比较全部落空：补上这 8g 个。
    lowest = plan.add('va', f'veq.8x128.s32 vm0, {Z}, {INT_MIN}')
    more = plan.add('va', f'vadd.8x128.s32 v13, v11, {G8}', [(rank, 1)])
    rank = plan.add('va', 'vsel.8x128 v11, vm0, v13, v11', [(more, 1), (lowest, 1)])
    payload = plan.add('vld', f'vld.8x128 v12, [vmem:0x{TP:x}]')
    if stop == 'rank':
        plan.add('vst', f'vst.8x128 [vmem:0x{values:x}], v11', [(rank, 1)])
        plan.add('vst', f'vst.8x128 [vmem:0x{indices:x}], v12', [(payload, 1)])
        return plan.emit()
    position = plan.add('va', f'vadd.8x128.s32 v12, {G8}, v12', [(payload, 1)])
    # 前 k 名把原通道号写到自己那一列的第 t 行（t 是名次）。带掩码的 vst.iar 实测有一条限制：目的行按 16 取模后，
    # 放行的元素不能与子通道号更小的元素（放行与否都算）相同。所以每条只放行 8 个名次，基址退后 8 行：
    # 放行的落在第 8 到 15 行，其余元素偏移为零、留在第 0 到 7 行，两边模 16 不会相遇。
    stores: dict[int, int] = {}
    for part in range(k // 8):
        low, high, offset, iar = f'vm{part % 2}', ('vm2', 'vm5')[part % 2], f'v{14 + part % 2}', part % 2
        before = [(stores.get(part - 2), 1)]
        above = plan.add('va', f'vge.8x128.s32 {low}, v11, {8 * part}', [(rank, 1)] + before)
        below = plan.add('va', f'vlt.8x128.s32 {high}, v11, {8 * part + 8}', [(rank, 1)] + before)
        inside = plan.add('misc', f'vmand.8x128.u1 {high}, {high}, {low}', [(above, 1), (below, 1)])
        row = plan.add('va', f'vsub.8x128.s32 {offset}, v11, {S}', [(rank, 1)] + before)
        row = plan.add('va', f'vadd.8x128.s32 {offset}, {8 - 8 * part}, {offset}', [(row, 1)])
        row = plan.add('va', f'vsel.8x128 {offset}, {high}, {offset}, {ZERO}', [(row, 1), (inside, 1)])
        set_iar = plan.add('vst', f'vsetiar.raw iar{iar}, {offset}', [(row, 1)] + before)
        stores[part] = plan.add('vst', f'vst.iar.msk.8x128 [vmem:0x{UP + 8 * part - 8:x}], iar{iar}, {high}, v12', [(set_iar, 7), (position, 1)])
    kept = stores[k // 8 - 1]

    def divide(source: str, quotient: str, shift: str, spare: str, after: int) -> tuple[int, int]:
        """source 小于 32：quotient = source // 3，shift = 8 * (source % 3)。没有整数乘法，11x 用移位和加法拼出。"""
        eight = plan.add('va1', f'vshll.8x128.s32 {quotient}, {source}, 3', [(after, 1)])
        two = plan.add('va1', f'vshll.8x128.s32 {spare}, {source}, 1', [(after, 1)])
        total = plan.add('va', f'vadd.8x128.s32 {quotient}, {quotient}, {spare}', [(eight, 1), (two, 1)])
        total = plan.add('va', f'vadd.8x128.s32 {quotient}, {quotient}, {source}', [(total, 1)])
        third = plan.add('va1', f'vshrl.8x128.s32 {quotient}, {quotient}, 5', [(total, 1)])
        double = plan.add('va1', f'vshll.8x128.s32 {spare}, {quotient}, 1', [(third, 1)])
        rest = plan.add('va', f'vsub.8x128.s32 {shift}, {source}, {spare}', [(double, 1)])
        rest = plan.add('va', f'vsub.8x128.s32 {shift}, {shift}, {quotient}', [(rest, 1)])
        return third, plan.add('va1', f'vshll.8x128.s32 {shift}, {shift}, 3', [(rest, 1)])

    residue = plan.add('va', 'vand.8x128.u32 v23, 7, v9', [(lane, 1)])
    same = plan.add('va', f'veq.8x128.s32 vm6, v23, {S}', [(residue, 1)])
    # 解码用的常数只与通道号有关：第 j 个通道取第 j // 3 个字的第 j % 3 个字节。
    column = plan.add('va', 'vand.8x128.u32 v22, 127, v9', [(lane, 1)])
    lane_word, lane_shift = divide('v22', 'v28', 'v29', 'v30', column)
    valid = plan.add('va', f'vlt.8x128.s32 vm7, v22, {k}', [(column, 1)])
    start = plan.add('va', 'vimm.8x128.s32 v26, 0')
    count = -(-k // 3)
    pushed: dict[int, int] = {}
    last_push: dict[int, int | None] = {0: None, 1: None}
    last_pop: dict[int, int | None] = {0: None, 1: None}
    decoded: dict[int, int | None] = {0: None, 1: None}
    merged = start
    for w in range(count):
        # 第 w 个字：名次 3w、3w+1、3w+2 的那三行各广播出来，移到各自的字节上相加；只留下通道号模 8 等于子通道号的那一类，交给 XLU 沿通道求和。
        word, spare, q = f'v{16 + w % 3}', f'v{19 + w % 3}', w % 2
        last = None
        for byte, j in enumerate(range(3 * w, min(3 * w + 3, k))):
            target = word if byte == 0 else spare
            last = plan.add('vld', f'vld.sshfl.8x128 {target}, [vmem:0x{UP + 8 * (j // 8):x}], 0x{0x11111111 * (j % 8):x}', [(stores[j // 8], 7), (pushed.get(w - 3), 1), (last, 0)])
            if byte:
                last = plan.add('va1', f'vshll.8x128.s32 {spare}, {spare}, {8 * byte}', [(last, 1)])
                last = plan.add('va', f'vadd.8x128.s32 {word}, {word}, {spare}', [(last, 1)])
        last = plan.add('va', f'vsel.8x128 {word}, vm6, {word}, {ZERO}', [(last, 1), (same, 1)])
        last = plan.add('va', f'vcvt.8x128.s32.f32 {word}, {word}', [(last, 1)])
        last_push[q] = pushed[w] = plan.add(f'vx{q}', f'vadd.xlane.{q}.8x128.f32 trf{q}, {word}', [(last, 1), (last_push[q], 8)])
        # 取回后各自解码，再按通道选进 v26。
        target, chosen = f'v{24 + q}', f'vm{3 + q}'
        last_pop[q] = plan.add(f'vr{q}', f'vpop.8x128 {target}, trf{q}', [(last_push[q], 80), (last_pop[q], 8), (decoded[q], 1)])
        field = plan.add('va', f'vcvt.8x128.f32.s32 {target}, {target}, 0xffffffff', [(last_pop[q], 1)])
        field = plan.add('va1', f'vshrl.8x128.s32 {target}, {target}, v29', [(field, 1), (lane_shift, 1)])
        field = plan.add('va', f'vand.8x128.u32 {target}, 127, {target}', [(field, 1)])
        mine = plan.add('va', f'veq.8x128.s32 {chosen}, v28, {w}', [(lane_word, 1), (decoded[q], 1)])
        merged = decoded[q] = plan.add('va', f'vsel.8x128 v26, {chosen}, {target}, v26', [(field, 1), (mine, 1), (merged, 1)])
    index = plan.add('va', f'vsel.8x128 v26, vm7, v26, {ZERO}', [(merged, 1), (valid, 1)])
    stored = plan.add('vst', f'vst.msk.8x128 [vmem:0x{indices:x}], vm7, v26', [(index, 1)])
    source = plan.add('vld', f'vld.8x128 v10, [vmem:0x{INPUT:x}]')
    permutation = plan.add('vx1', 'vsetperm.1.all.u8 pcr1, v26', [(index, 1), (last_push[1], 8)])
    gathered = plan.add('vx1', 'vperm.1.8x128 trf1, v10', [(permutation, 1), (source, 1)])
    result = plan.add('vr1', 'vpop.8x128 v17, trf1', [(gathered, 70), (last_pop[1], 8)])
    plan.add('vst', f'vst.msk.8x128 [vmem:0x{values:x}], vm7, v17', [(result, 1)])
    last = stored
    for line in dma.get('indices', ()):
        last = plan.add('s0', line, [(last, 1)])
    last = gathered
    for line in dma.get('values', ()):
        last = plan.add('s0', line, [(last, 1), (stored, 1)])
    return plan.emit()

def carrier(k: int) -> asm.Compiled:
    """只做搬运的 kernel：输入读进 TC VMEM 第 0 行起，两个输出各占一个 tile，之后是十二个工作 tile。"""
    mesh = jax.make_mesh((1,), ('device',))

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def function(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        @pl.kernel(
            out_type=(jax.ShapeDtypeStruct((8, k), jnp.float32), jax.ShapeDtypeStruct((8, k), jnp.int32)),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, k), jnp.float32), pltpu.VMEM((8, k), jnp.int32), pltpu.VMEM((96, 128), jnp.int32), pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            name='folded_select',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, v_hbm: Ref, i_hbm: Ref, x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, work: Ref, sem: Ref, sem2: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            # 载体自己的两条 store 只为确定各 scratch 的地址；program() 把它们改写到工作区，结果由手写片段写出。
            v_vmem[...] = x_vmem[:, :k]
            i_vmem[...] = work[:8, :k]
            first = pltpu.async_copy(v_vmem, v_hbm, sem)
            second = pltpu.async_copy(i_vmem, i_hbm, sem2)
            first.wait()
            second.wait()

        return kernel(x)

    return asm.compile(function, np.zeros((8, 128), np.float32), mesh=mesh)

def program(k: int = 8, *, early_output: bool) -> tuple[bytes, asm.Compiled, int]:
    """完整的 top-k 程序（k 为 8、16 或 32）：载体的搬运之后插入手写片段。返回 executable、装载用的模板和插入的 bundle 数。

    early_output 把两次输出 DMA 提到片段内部。k = 8 时两条都排在第一个值取回之后，距结果写出四五十个周期；k > 8 时下标的排在下标写出之后，值的排在取值的 vperm 提交之后，距值写出约 70 个周期。都在 76 个周期的窗口之内。
    """
    global TY, TZ, TP, T0, T1, VT, UP
    compiled = carrier(k)
    serialized = asm.serialize(compiled)
    listing = asm.kernel_listing(compiled)
    assert 'vld.8x128' in listing and '[vmem:0x18]' in listing, listing
    TY, TZ, TP, T0, T1 = (0x18 + 8 * i for i in range(5))
    VT = [0x40, 0x48]
    UP = 0x50
    outputs = asm.find_bundles(serialized, 'dma.simple [hbm:')
    changed: dict[int, str] = {}

    def replace(pc: int, pattern: str, new: str | None = None, address: int | None = None) -> str:
        """把 bundle pc 中含 pattern 的那条指令换掉（或把它的地址改到工作区），返回原指令。"""
        whole = changed.get(pc) or asm.bundle_text(serialized, pc, encoding='exact')
        item, = [part.strip(' {}\n') for part in whole.split(';') if pattern in part]
        changed[pc] = whole.replace(item, new if new is not None else item.replace(f'[vmem:0x{address:x}]', '[vmem:0x18]'))
        return item

    for address in (0x8, 0x10):
        pc, = asm.find_bundles(serialized, f'vst.msk.8x128 [vmem:0x{address:x}]')
        replace(pc, 'vst.msk', address=address)
    moved = {}
    for pc in outputs:
        item = replace(pc, 'dma.simple', 's0: simm.s32 s22, 0') if early_output else [part.strip(' {}\n') for part in asm.bundle_text(serialized, pc, encoding='exact').split(';') if 'dma.simple' in part][0]
        register = re.search(r'\[hbm:(s\d+)\], \[vmem:(s\d+)\]', item)
        # 序言把两个输出的 HBM 地址放在 s1（值）和 s2（下标）。
        name, address, scratch = {'s1': ('values', 0x8, 's20'), 's2': ('indices', 0x10, 's21')}[register.group(1)]
        moved[name] = (f'simm.s32 {scratch}, {address}', item.split(':', 1)[1].strip().replace(f'[vmem:{register.group(2)}]', f'[vmem:{scratch}]'))
    edits = {pc: (asm.bundle_text(serialized, pc, encoding='exact'), whole) for pc, whole in changed.items()}
    text = b('vld: vld.8x128 v10, [vmem:0x0]')
    text += fragment(0x8, 0x10, k=k, dma=moved if early_output else {})
    return asm.insert_bundles(asm.edit_bundles(serialized, edits), {outputs[0]: text}), compiled, text.count('{')

def shell_measure(clock: asm.KernelClock, k: int) -> dict[str, object]:
    """把手写片段放进 shell.py 的外壳：载体 kernel 的操作数、结果和工作区都在 TC VMEM 的固定地址上，片段插在 kernel 的结尾，直接读输入、写两个结果。"""
    import shell

    def carrier(x: jax.Array, work: Ref) -> tuple[jax.Array, jax.Array]:
        # 载体自己的读写只为让编译器给出各块 TC VMEM 的地址；它写出的内容随后被片段覆盖。
        return x[:, :k], lax.bitcast_convert_type(x, jnp.int32)[:, :k] ^ work[:8, :k]

    def insert(compiled: asm.Compiled) -> bytes:
        global TY, TZ, TP, T0, T1, VT, UP, INPUT
        serialized = asm.serialize(compiled)
        start, end = shell.kernel_span(serialized)
        text = ' '.join(asm.bundle_text(serialized, pc) for pc in range(start, end + 1))
        constants = {register: int(value) for register, value in re.findall(r'simm\.s32 (s\d+), (\d+)', text)}
        address = lambda operand: int(operand, 16) if operand.startswith('0x') else constants[operand]
        loads = {register: address(operand) for register, operand in re.findall(r'vld\.8x128 (v\d+), \[vmem:(\w+)\]', text)}
        stores = [(address(operand), register) for operand, register in re.findall(r'vst\.msk\.8x128 \[vmem:(\w+)\], vm\d+, (v\d+)', text)]
        assert len(loads) == 2 and len(stores) == 2 and len({*loads.values(), *(target for target, _ in stores)}) == 4, text
        # 值的那次写出直接来自输入的读入；另一次读入的是工作区。
        INPUT = loads[stores[0][1]]
        work, = [target for target in loads.values() if target != INPUT]
        stores = [target for target, _ in stores]
        TY, TZ, TP, T0, T1 = (work + 8 * i for i in range(5))
        VT = [work + 40, work + 48]
        UP = work + 56
        # 插在 kernel 结尾编译器自己的 cfence 之前，与编译器生成的 kernel 用同一种收尾。
        assert 'cfence' in asm.bundle_text(serialized, end - 1)
        return asm.insert_bundles(serialized, {end - 1: b(f'vld: vld.8x128 v10, [vmem:0x{INPUT:x}]') + fragment(stores[0], stores[1], k=k)})

    rng = np.random.default_rng(6466)
    record = shell.measure(clock, shell.pallas_top_k(k, 1, carrier, scratch_rows=96), (8, 128), k, 1, transform=insert, label=f'handwritten-k{k}')
    special = [x.view(np.float32) for x in cases(rng, k)]
    return {**record, 'variant': 'handwritten', 'special': shell.measure(clock, shell.pallas_top_k(k, 1, carrier, scratch_rows=96), (8, 128), k, 1, transform=insert, inputs=special, label=f'handwritten-k{k}-special')['median']}

def reference(x: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    key = np.where(x.view(np.int32) < 0, x ^ 0x7fffffff, x).view(np.int32)
    order = np.argsort(-key.astype(np.int64), axis=1, kind='stable')
    return np.take_along_axis(x, order[:, :k], axis=1), order[:, :k]

def cases(rng: np.random.Generator, k: int = 8) -> list[np.ndarray]:
    result = [rng.integers(0, 2**32, (8, 128), dtype=np.uint32) for _ in range(12)]
    result += [rng.integers(-3, 4, (8, 128)).astype(np.float32).view(np.uint32) for _ in range(4)]
    result += [np.full((8, 128), word, np.uint32) for word in (0xffffffff, 0, 0x80000000, 0x7fffffff)]
    result += [rng.normal(size=(8, 128)).astype(np.float32).view(np.uint32) for _ in range(4)]
    for count in range(k + 2):
        x = -np.abs(rng.normal(size=(8, 128)).astype(np.float32))
        x[:, rng.permutation(128)[:count]] *= -1
        result.append(x.view(np.uint32))
    sparse = np.full((8, 128), -np.inf, np.float32)
    sparse[:, [0, 1, 127]] = [3.0, 1.0, 2.0]
    result.append(sparse.view(np.uint32))
    result += [rng.choice(np.array([0xffffffff, 0xff800000, 0x80000000, 0, 1, 0x7f800000, 0x7fc00001], np.uint32), (8, 128)) for _ in range(4)]
    return result

def main() -> None:
    """参数：阶段（sorted、rank 或 full）和 k；第一个参数为 shell 时，改为在 shell.py 的外壳里对 k = 8、16、32 计时，结果写到第二个参数给出的 JSON 文件。"""
    if len(sys.argv) > 1 and sys.argv[1] == 'shell':
        clock = asm.KernelClock(num_cores=1)
        records = [shell_measure(clock, k) for k in (8, 16, 32)]
        Path(sys.argv[2]).write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
        return
    probe = asm.LccProbe()
    rng = np.random.default_rng(6465)
    stop = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'full' else ''
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    body = fragment(0x8, 0x10, k=k, stop=stop) + asm.reload_saved()
    wrong = 0
    for i, x in enumerate(cases(rng, k)):
        raw = probe.host.copy()
        raw[:8] = x
        probe.x = jax.device_put(raw)
        tiles = probe.run_tiles(body, setup=asm.spill_saved())[0, 0]
        key = np.where(x.view(np.int32) < 0, x ^ 0x7fffffff, x).view(np.int32)
        if stop == 'sorted':
            folded = key.reshape(8, 16, 8).transpose(2, 1, 0).reshape(8, 128)
            order = np.argsort(-folded.astype(np.int64), axis=0, kind='stable')
            ok = np.array_equal(tiles[0].view(np.int32), np.take_along_axis(folded, order, axis=0)) and np.array_equal(tiles[1], order)
        elif stop == 'rank':
            order = np.argsort(-key.astype(np.int64), axis=1, kind='stable')
            rank = np.argsort(order, axis=1)
            position = (np.arange(128)[None, :] & 120) + tiles[1].astype(np.int64)
            row = (np.arange(128)[None, :] & 7) + np.zeros((8, 1), np.int64)
            expected = rank[row, position]
            got = tiles[0].astype(np.int64)
            ok = np.array_equal(np.where(expected < k, expected, 99), np.where(got < k, got, 99))
        else:
            want, index = reference(x, k)
            ok = np.array_equal(tiles[0][:, :k], want) and np.array_equal(tiles[1][:, :k], index)
        wrong += not ok
        print('case', i, 'correct' if ok else 'WRONG', flush=True)
    print('k', k, 'stage', stop or 'full', 'cases', i + 1, 'wrong', wrong)
    program = read_lcc(20) + fragment(0x8, 0x10, k=k, stop=stop) + read_lcc(21) + b('s0: sfence') + read_lcc(22) + asm.reload_saved()
    print('cycles', sorted({tuple(row) for row in probe.run(program, setup=asm.spill_saved()).tolist()}), flush=True)

if __name__ == '__main__':
    main()
