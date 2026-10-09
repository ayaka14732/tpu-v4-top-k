"""让 Pallas kernel 用上 Mosaic 不生成的 TPU v4 指令：源码里写占位运算，编译之后把占位换成真正的指令。

每个 intrinsic 在 Pallas 里是一两条带唯一立即数的整数运算（占位），形状、寄存器和排程都由编译器决定；
rewrite() 在编译出的 executable 中按立即数找到占位，换成对应的指令。操作数都是 int32 的 [8n, 128] 数组，每个 TC VREG 各展开一次。
kernel 的指令必须是一条直线（没有循环和分支），rewrite() 按 bundle 顺序做数据流分析。
"""
from pathlib import Path
import re
import sys
import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jaxlib.mlir import ir
from jaxlib.mosaic.python import tpu
from tpuasm import tools as asm
from tpuasm.tools import bundle as b

MARK = 0x5eed0000
SITES: list[tuple[str, tuple[int, ...]]] = []

def _site(kind: str, *parameters: int) -> jax.Array:
    """登记一个调用点，返回它的占位立即数。"""
    SITES.append((kind, parameters))
    return jnp.int32(MARK + len(SITES) - 1)

def _float(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, jnp.float32)

def _int(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, jnp.int32)

def fold(x: jax.Array) -> jax.Array:
    """分段转置（vsxpose，宽度 8）：y[s, 8g+r] = x[r, 8g+s]。它是自己的逆运算。

    占位是对打了标记的值做一次跨通道旋转：编译器照 XLU 的提交与取回来排程，改写时只把提交的那条 vrot 换成 vsxpose。
    """
    return jnp.roll(x ^ _site('fold'), 1, axis=1)

def sublane_shuffle(x: jax.Array, pattern: int) -> jax.Array:
    """沿子通道按固定模式重排（vld.sshfl）：第 s 个子通道取 pattern 第 s 个十六进制位（从低位数）所指的行。

    占位只是一条带标记的异或。源值在它被定义之后立刻存进工作区（同一个值只存一次），占位换成一条读。
    """
    return x ^ _site('shuffle', pattern)

shuffle_p = jax_core.Primitive('sublane_shuffle')
shuffle_p.def_abstract_eval(lambda x, *, indices: x)

def _shuffle_rule(ctx: mosaic_lowering.LoweringRuleContext, x: ir.Value, *, indices: tuple[int, ...]) -> ir.Value:
    return tpu.gather(x.type, x, list(indices), 0)

mosaic_lowering.lowering_rules[tpu_core.CoreType.TC][shuffle_p] = _shuffle_rule

def sublane_shuffle_op(x: jax.Array, pattern: int) -> jax.Array:
    """与 sublane_shuffle 同义，但不用占位：TPU dialect 里现成的 tpu.gather（下标是编译期常量的 gather）在 TPU v4 上本来就编译成 vld.sshfl，
    只是 Pallas 没有哪条 lowering rule 会生成它。这里补一条 primitive，不需要改写 executable，也不需要给 libtpu 打补丁。
    """
    return shuffle_p.bind(x, indices=tuple(8 * (row // 8) + (pattern >> (4 * (row % 8)) & 15) for row in range(x.shape[0])))

fold_p = jax_core.Primitive('fold')
fold_p.def_abstract_eval(lambda x: x)
mosaic_lowering.lowering_rules[tpu_core.CoreType.TC][fold_p] = lambda ctx, x: tpu.dynamic_gather(x, x, [0])

def fold_op(x: jax.Array) -> jax.Array:
    """与 fold 同义，但不用占位：需要 libtpu_patch.allow_sublane_gather()。

    TPU dialect 里没有分段转置的 op，这里借一个不会与正常用法混淆的写法当载体：沿子通道、拿自己当下标的 gather。
    补丁在把它译成 C++ LLO 的地方认出这种写法，生成 vsxpose 和取回。编译器因此知道这是一次转置，会把它之后 XLU 被占用的时间算进排程。
    """
    return fold_p.bind(x)

def sublane_gather(x: jax.Array, offset: jax.Array) -> jax.Array:
    """沿子通道按索引读（vsetiar + vld.iar）：y[s, l] = x[s + offset[s, l], l]，行号须落在同一个 TC VREG 的 0 到 7 之内。

    占位是对打了标记的偏移求一次 exp2 再减去数据：EUP 的等待对应 vsetiar 到 vld.iar 之间要留的间隔。
    """
    return _int(jnp.exp2(_float(offset ^ _site('gather')))) - x

def sublane_scatter(x: jax.Array, offset: jax.Array, fill: int = 0) -> jax.Array:
    """沿子通道按索引写（vsetiar + vst.iar）：y[s + offset[s, l], l] = x[s, l]，没有被写到的位置是 fill。

    同一列的目的行按 16 取模必须互不相同；落在 8 到 15 行的元素被丢弃，可以用来筛掉不要的元素。
    """
    return (offset ^ _site('scatter', fill)) - x

def scratch(ref: jax.Ref) -> None:
    """声明一块供 intrinsic 使用的 TC VMEM 工作区（至少 96 行，越大可同时保存的中间值越多）：一次读改写让 rewrite() 认出它的地址。"""
    ref[0:8] = ref[0:8] ^ _site('scratch', ref.shape[0] // 8)

def _items(text: str) -> list[str]:
    body = re.sub(r'\.encoding \{[^}]*\}', '', text)
    return [part.strip(' {}\n') for part in body.split(';') if part.strip(' {}\n')]

def _written(item: str) -> str | None:
    """这条指令写入的 TC VREG。"""
    match = re.match(r'(?:va\d|vld|vr\d): \S+ (v\d+)\b', item)
    return match.group(1) if match else None

def _reads(item: str, register: str) -> bool:
    operands = item.split(None, 2)[2] if len(item.split(None, 2)) > 2 else ''
    if _written(item):
        operands = operands.split(',', 1)[1] if ',' in operands else ''
    return re.search(rf'\b{register}\b', operands) is not None

def rewrite(compiled: asm.Compiled) -> tuple[bytes, dict[str, int]]:
    """把 executable 中的占位换成真正的指令，返回新的 executable 和各类指令插入的条数。"""
    serialized = asm.serialize(compiled)
    pcs = asm.pallas_bundles(serialized)
    first, last = pcs[0], pcs[-1] + 1
    # 整个程序只反汇编一次；asm.bundle_text 每次调用都会重新反汇编。
    (_, _, image), = asm.executable_programs(serialized)
    lines = asm.format_assembly(image, target=asm.TARGET).splitlines()
    spans = asm._bundle_spans(lines)
    exact = {pc: '\n'.join(lines[spans[pc][0]:spans[pc][1] + 1]) for pc in range(first, last)}
    items = {pc: _items(text) for pc, text in exact.items()}
    counts: dict[str, int] = {}
    new_items = {pc: list(found) for pc, found in items.items()}
    before: dict[int, list[str]] = {}
    after: dict[int, list[str]] = {}
    stores: dict[int, list[str]] = {}

    def definition(register: str, pc: int) -> int:
        """bundle pc 读到的 register 是在哪个 bundle 写入的。"""
        for index in range(pc - 1, first - 1, -1):
            if any(_written(item) == register for item in items[index]):
                return index
        raise ValueError(f'{register} read at bundle {pc} is never written in the kernel')

    def origin(register: str, pc: int) -> tuple[str, int]:
        """bundle pc 读到的 register 里的那个值最初是在哪里算出来的：(寄存器, bundle)。

        编译器会把值溢出到 TC VMEM 再读回别的寄存器，或者复制到别的寄存器；沿着读回和复制往回追，直到真正算出它的那条指令。
        """
        defined = definition(register, pc)
        item, = [other for other in items[defined] if _written(other) == register]
        load = re.match(r'vld: vld\.8x128 v\d+, \[vmem:0x([0-9a-f]+)\]$', item)
        move = re.match(r'va\d: vmov\.8x128 v\d+, (v\d+)$', item)
        if move:
            return origin(move.group(1), defined)
        if load:
            for index in range(defined - 1, first - 1, -1):
                for other in items[index]:
                    store = re.match(rf'vst: vst\.8x128 \[vmem:0x{load.group(1)}\], (v\d+)$', other)
                    if store:
                        return origin(store.group(1), index)
        return register, defined

    def touched(register: str, start: int, stop: int, *, read: bool, skip: str = '') -> bool:
        """bundle [start, stop) 中是否有指令读（或写）register；skip 是不计入的那条指令。"""
        for index in range(start, stop):
            for other in items[index]:
                if other != skip and (_reads(other, register) if read else _written(other) == register):
                    return True
        return False

    def swap(pc: int, old: str, new: str | None) -> None:
        position = new_items[pc].index(old)
        if new is None:
            del new_items[pc][position]
        else:
            new_items[pc][position] = new
        counts[(new or 'removed').split(':', 1)[1].split()[0] if new else 'removed'] = counts.get((new or 'removed').split(':', 1)[1].split()[0] if new else 'removed', 0) + 1

    def add(where: dict[int, list[str]], pc: int, item: str) -> None:
        where.setdefault(pc, []).append(item)
        counts[item.split(':', 1)[1].split()[0]] = counts.get(item.split(':', 1)[1].split()[0], 0) + 1

    # 找出所有占位：(种类, 参数, bundle, 指令, 目的寄存器, 源寄存器)。
    found = []
    work = tiles = None
    for site, (kind, parameters) in enumerate(SITES):
        for pc in range(first, last):
            for item in items[pc]:
                marker = re.search(rf'vxor\S* (v\d+), 0x{MARK + site:x}, (v\d+)', item)
                if not marker:
                    continue
                if kind == 'scratch':
                    source = marker.group(2)
                    load, = [other for other in items[definition(source, pc)] if _written(other) == source]
                    address = int(re.search(r'\[vmem:0x([0-9a-f]+)\]', load).group(1), 16)
                    work, tiles = address, parameters[0]
                else:
                    found.append((kind, parameters, pc, item, marker.group(1), marker.group(2)))
    found.sort(key=lambda entry: entry[2])

    def reader(register: str, pc: int, pattern: str) -> tuple[int, str, re.Match, str]:
        """bundle pc 写入 register 的那个值，之后第一条匹配 pattern 的读者；pattern 中的 REG 代表当时装着这个值的寄存器。

        不能只按寄存器名找：编译器在寄存器紧张时会把值溢出到 TC VMEM、之后读回另一个寄存器，原来的寄存器则改作他用。
        所以沿着 bundle 跟踪这个值在哪些寄存器和哪些溢出地址里，直到找到读者。
        """
        holders, spilled = {register}, set()
        for index in range(pc + 1, last):
            for other in items[index]:
                for holder in holders:
                    match = re.search(pattern.replace('REG', holder), other)
                    if match and _reads(other, holder):
                        return index, other, match, holder
            arrived, departed = set(), set()
            for other in items[index]:
                store = re.match(r'vst: vst\.8x128 \[vmem:0x([0-9a-f]+)\], (v\d+)$', other)
                load = re.match(r'vld: vld\.8x128 (v\d+), \[vmem:0x([0-9a-f]+)\]$', other)
                move = re.match(r'va\d: vmov\.8x128 (v\d+), (v\d+)$', other)
                if store:
                    (spilled.add if store.group(2) in holders else spilled.discard)(store.group(1))
                if load and load.group(2) in spilled or move and move.group(2) in holders:
                    arrived.add((load or move).group(1))
                elif _written(other) in holders:
                    departed.add(_written(other))
            holders = (holders - departed) | arrived
            if not holders and not spilled:
                break
        raise ValueError(f'nothing matching {pattern!r} reads the value written to {register} at bundle {pc}')

    # EUP 的结果队列先进先出：第 n 次提交对应第 n 次取回。
    pushes = [(pc, other) for pc in range(first, last) for other in items[pc] if re.search(r': v\w+\.8x128\.f32 erf, v\d+', other)]
    pops = [(pc, other) for pc in range(first, last) for other in items[pc] if re.search(r'vpop\.8x128 v\d+, erf', other)]
    assert len(pushes) == len(pops), (len(pushes), len(pops))
    popped = dict(zip(pushes, pops))
    # 每个占位展开成一条“提交”和一条“取回”：(种类, 参数, 标记, 提交, 取回, 使用)。
    plans = []
    folds: list[tuple[int, int]] = []
    for kind, parameters, pc, item, target, source in found:
        slot = item.split(':')[0]
        if kind == 'shuffle':
            plans.append((kind, parameters, target, None, None, (pc, item, target, source)))
            continue
        swap(pc, item, f'{slot}: vmov.8x128 {target}, {source}')
        if kind == 'fold':
            at, push, match, holder = reader(target, pc, r'(vx\d): vrot\.(\d)\.8x128 (trf\d), REG\b')
            swap(at, push, f'{match.group(1)}: vsxpose.{match.group(2)}.start.end.8x128 {match.group(3)}, {holder}, 8')
            folds.append((at, int(match.group(3)[-1])))
        elif kind == 'gather':
            at, push, _, offset = reader(target, pc, r'vpow2\.8x128\.f32 erf, REG\b')
            back, pop = popped[at, push]
            result = re.search(r'vpop\.8x128 (v\d+), erf', pop).group(1)
            use, consumer, match, holder = reader(result, back, r'(\w+): vsub\.8x128\.s32 (v\d+), REG, (v\d+)')
            # 偏移的源寄存器在提交之前没有被改写时，标记处的复制可以省掉，直接从源寄存器装 IAR。
            if offset == target and not touched(source, pc, at, read=False):
                swap(pc, f'{slot}: vmov.8x128 {target}, {source}', None)
                offset = source
            plans.append((kind, parameters, offset, (at, push), (back, pop), (use, consumer, match.group(2), match.group(3), holder)))
        else:
            use, consumer, match, holder = reader(target, pc, r'(\w+): vsub\.8x128\.s32 (v\d+), REG, (v\d+)')
            plans.append((kind, parameters, holder, None, None, (use, consumer, match.group(2), match.group(3))))
    # vsxpose 的结果取回之后，它所在的 XLU 还要忙约一百个周期，编译器不知道这一点，仍会把随后的旋转平分给两个队列，
    # 整批都被堵在后面。把紧跟在折叠之后、赶在任何结果取回之前提交的那一批，连同它们的取回，全部改到另一个队列上。
    # 几个 TC VREG 的折叠如果被编译器分到了两个队列上，两边都被占用，就不改。
    for blocked in ({queue for _, queue in folds} if len({queue for _, queue in folds}) == 1 else ()):
        other = 1 - blocked
        fold_pushes = {at for at, _ in folds}
        events = [(pc, item) for pc in range(first, last) for item in sorted(items[pc]) if re.search(r'trf[01]', item)]
        matched: dict[tuple[int, str], tuple[int, str]] = {}
        outstanding = {0: [], 1: []}
        batch_open = True
        moved: list[tuple[int, str]] = []
        for pc, item in events:
            queue = int(re.search(r'trf([01])', item).group(1))
            if 'vpop' in item:
                push = outstanding[queue].pop(0)
                matched[push] = (pc, item)
                # 折叠自己的取回不算；第一次取回别的结果时这一批就结束了。
                batch_open = batch_open and push[0] in fold_pushes and 'vrot' in push[1]
            else:
                outstanding[queue].append((pc, item))
                if queue == blocked and batch_open and pc > max(fold_pushes) and re.search(r': vrot\.', item):
                    moved.append((pc, item))
        if not moved or any(push not in matched for push in moved):
            continue
        # 合并之后另一个队列上的提交顺序：按 bundle，同一个 bundle 里原来就在这个队列上的在前。取回必须是同样的顺序。
        first_moved, last_moved_pop = moved[0][0], max(matched[push][0] for push in moved)
        merged = sorted([push for push in matched if int(re.search(r'trf([01])', push[1]).group(1)) == other and first_moved <= push[0] <= last_moved_pop] + moved, key=lambda push: (push[0], push in moved))
        pops = [matched[push] for push in merged]
        if [pop[0] for pop in pops] != sorted(pop[0] for pop in pops):
            continue
        # 先定下每次提交的新位置（bundle，0 表示留在 bundle 里、1 表示挪到它之后），确认先后次序没有变再动手：
        # 队列先进先出，提交的次序一旦与取回的次序对不上，各次旋转的结果就会张冠李戴。
        placements = []
        for number, push in enumerate(merged):
            pc, item = push
            new = re.sub(r'vrot\.\d', f'vrot.{other}', item).replace(f'trf{blocked}', f'trf{other}')
            source = re.search(r'trf\d, (v\d+)', item).group(1)
            # 一个队列每 8 个周期收一次提交，结果 69 个周期后回来：前 9 次连着提交，之后的每次挪到一次取回之后，
            # 否则按序发射会让第一次取回排在全部提交的后面。
            late = pops[number - 9][0] if number >= 9 else None
            if late is not None and late > pc and not touched(source, pc, late + 1, read=False):
                placements.append((late, 1, new))
            elif any(f'trf{other}' in neighbor and 'vpop' not in neighbor for neighbor in items[pc] if neighbor != item):
                placements.append((pc, 1, new))
            else:
                placements.append((pc, 0, new))
        if [entry[:2] for entry in placements] != sorted(entry[:2] for entry in placements):
            continue
        for push, (where, phase, new) in zip(merged, placements):
            pc, item = push
            if phase:
                swap(pc, item, None)
                add(after, where, f'vx0: {new.split(": ", 1)[1]}')
            elif new != item:
                swap(pc, item, new)
            if push not in moved:
                continue
            back, pop = matched[push]
            new = pop.replace(f'trf{blocked}', f'trf{other}')
            if any(f'trf{other}' in neighbor and 'vpop' in neighbor for neighbor in items[back] if neighbor != pop):
                swap(back, pop, None)
                add(after, back, f'vr0: {new.split(": ", 1)[1]}')
            else:
                swap(back, pop, new)
    assert work is not None or not plans, 'call scratch() in the kernel before using memory intrinsics'
    # 要从 TC VMEM 读的值（shuffle 的源、gather 的数据）在定义之后立刻存进工作区的一个 tile，同一个值只存一次；tile 按“定义到最后一次读”的区间分配。第 0 个 tile 留给 scratch() 自己的读改写，最后八个留给 scatter。
    # 两个 IAR 按占用区间分配：按索引读从装入（原来的提交）占到读（原来的取回），按索引写只在使用的那一刻占用。
    # 编译器可能让三个以上的按索引读同时在途。装不进 IAR 的那一次改为“推迟”：偏移先存进工作区，轮到它读的时候再读回来装 IAR；
    # 有推迟的读在途时，后来的读也一律推迟，保证轮到它时 IAR 是空的。给一次读分配 IAR 之前还要确认，这不会让随后的某次按索引写无 IAR 可用。
    scatters = sorted(use[0] for kind, _, _, _, _, use in plans if kind == 'scatter')
    iar_busy = [first - 1, first - 1]
    decision: dict[int, tuple[str, int]] = {}
    deferred_until = first - 1
    requests = sorted((use[0] if kind == 'scatter' else push[0], number) for number, (kind, _, _, push, _, use) in enumerate(plans) if kind in ('gather', 'scatter'))
    # 编译器自己生成的按索引读（打了 libtpu 补丁之后的 jnp.take_along_axis）也占着 IAR：从它的 vsetiar 到 vld.iar。
    loads = [(pc, int(match.group(1))) for pc in range(first, last) for item in items[pc] for match in [re.search(r'vld\.iar(\d)', item)] if match]
    native = [(pc, min(load for load, other in loads if other == int(match.group(1)) and load > pc), int(match.group(1))) for pc in range(first, last) for item in items[pc] for match in [re.search(r'vsetiar\.raw iar(\d)', item)] if match]
    for at, number in requests:
        kind, _, _, push, pop, use = plans[number]
        free = [register for register in (0, 1) if iar_busy[register] < at and not any(start <= (at if kind == 'scatter' else pop[0]) and at <= end and register == other for start, end, other in native)]
        if kind == 'scatter':
            assert free, f'no index register is free for the scatter at bundle {at}'
            decision[number] = ('now', free[0])
            iar_busy[free[0]] = at
            continue
        crowded = any(push[0] <= other <= pop[0] for other in scatters) and len(free) < 2
        if not free or at <= deferred_until or crowded:
            decision[number] = ('deferred', 0)
            deferred_until = max(deferred_until, pop[0])
        else:
            decision[number] = ('now', free[0])
            iar_busy[free[0]] = pop[0]
    spans = []
    for number, (kind, _, _, push, pop, use) in enumerate(plans):
        if kind == 'shuffle':
            spans.append((origin(use[3], use[0])[1], use[0], ('data', *origin(use[3], use[0]))))
        elif kind == 'gather':
            spans.append((origin(use[3], use[0])[1], pop[0], ('data', *origin(use[3], use[0]))))
        if kind == 'gather' and decision[number][0] == 'deferred':
            spans.append((push[0], pop[0], ('offset', number)))
    tile_of: dict[tuple, int] = {}
    busy: dict[int, int] = {}
    for start, end, name in sorted(set(spans)):
        if name in tile_of:
            busy[(tile_of[name] - work) // 8] = max(busy[(tile_of[name] - work) // 8], end)
            continue
        free = [tile for tile in range(1, tiles - 8) if busy.get(tile, -1) < start]
        assert free, f'scratch() needs more than {tiles} tiles'
        tile_of[name] = work + 8 * free[0]
        busy[free[0]] = max(end, max((other_end for other_start, other_end, other in spans if other == name), default=end))
        if name[0] == 'data':
            # 定义这个值的指令本身可能是改写时挪到 bundle 之后的一条读，所以这些 store 单独记，排在同一位置的其他插入指令之后。
            add(stores, name[2], f'vst: vst.8x128 [vmem:0x{tile_of[name]:x}], {name[1]}')
    # scatter 各用相邻的两个 tile（后一个只接收被丢弃的元素）；两个 scatter 可能落在同一个 bundle，所以留四对轮流用。
    pair_busy = [first - 1] * 4

    def place(pc: int, old: str, new: str, where: dict[int, list[str]]) -> None:
        """把 bundle pc 中的 old 换成 new；new 要用的槽在这个 bundle 里已被占用时，把它挪到这个 bundle 之前或之后。"""
        slot = new.split(':')[0]
        if any(other.split(':')[0] == slot for other in new_items[pc] if other != old):
            swap(pc, old, None)
            add(where, pc, new)
        else:
            swap(pc, old, new)

    for number, (kind, parameters, target, push, pop, use) in enumerate(plans):
        if kind == 'shuffle':
            at, marker, result, source = use
            tile = tile_of['data', *origin(source, at)]
            place(at, marker, f'vld: vld.sshfl.8x128 {result}, [vmem:0x{tile:x}], 0x{parameters[0]:x}', after)
            continue
        mode, register = decision[number]
        if kind == 'gather':
            at, consumer, final, data, holder = use
            result = re.search(r'vpop\.8x128 (v\d+), erf', pop[1]).group(1)
            tile = tile_of['data', *origin(data, at)]
            if mode == 'deferred':
                # 偏移存起来；轮到它时结果寄存器正要被写，借它把偏移读回来装 IAR。这三条都排在原来取回的那个 bundle 之后。
                saved = tile_of['offset', number]
                place(push[0], push[1], f'vst: vst.8x128 [vmem:0x{saved:x}], {target}', before)
                swap(pop[0], pop[1], None)
                for text in (f'vld: vld.8x128 {result}, [vmem:0x{saved:x}]', f'vst: vsetiar.raw iar0, {result}', f'vld: vld.iar0.8x128 {result}, [vmem:0x{tile:x}]'):
                    add(after, pop[0], text)
                swap(at, consumer, f'{consumer.split(":")[0]}: vmov.8x128 {final}, {holder}')
                continue
            place(push[0], push[1], f'vst: vsetiar.raw iar{register}, {target}', before)
            # 最终的目的寄存器在取回到使用之间没有别的读写、取回的寄存器也没有别的读者时，直接读进目的寄存器，省掉使用处的复制。
            direct = holder == result and not touched(final, pop[0], at, read=False) and not touched(final, pop[0], at + 1, read=True) and not touched(result, pop[0] + 1, at + 1, read=True, skip=consumer)
            place(pop[0], pop[1], f'vld: vld.iar{register}.8x128 {final if direct else result}, [vmem:0x{tile:x}]', after)
            swap(at, consumer, None if direct else f'{consumer.split(":")[0]}: vmov.8x128 {final}, {holder}')
            continue
        at, consumer, result, data = use
        pair = min(range(4), key=lambda index: pair_busy[index])
        pair_busy[pair] = at
        scatter_tile = work + 8 * (tiles - 8 + 2 * pair)
        # 操作数寄存器只保证活到使用它的那个 bundle 为止（同一个 bundle 里别的指令可以改写它），所以整段都排在那个 bundle 之前；
        # 填充值借用已经装进 IAR 的偏移寄存器。结果寄存器在那个 bundle 之前可能还装着别的值，读回放在原来的位置。
        for text in (f'vst: vsetiar.raw iar{register}, {target}', f'va0: vimm.8x128.s32 {target}, {parameters[0]}', f'vst: vst.8x128 [vmem:0x{scatter_tile:x}], {target}', f'vst: vst.iar.8x128 [vmem:0x{scatter_tile:x}], iar{register}, {data}'):
            add(before, at, text)
        place(at, consumer, f'vld: vld.8x128 {result}, [vmem:0x{scatter_tile:x}]', after)
    edits = {}
    insertions: dict[int, str] = {}
    for pc in range(first, last):
        if new_items[pc] != items[pc]:
            edits[pc] = (exact[pc], '{ ' + ' ; '.join(new_items[pc] or ['misc: vnop']) + ' }')
        text = ''.join(b(item) for item in before.get(pc, []))
        if text:
            insertions[pc] = insertions.get(pc, '') + text
        text = ''.join(b(item) for item in after.get(pc, []) + stores.get(pc, []))
        if text:
            insertions[pc + 1] = text + insertions.get(pc + 1, '')
    serialized = asm.edit_bundles(serialized, edits)
    return asm.insert_bundles(serialized, insertions), counts

def compile(function, *arguments, mesh: jax.sharding.Mesh) -> tuple[asm.Compiled, dict[str, int]]:
    """编译一个用到 intrinsic 的程序并完成改写，返回可以直接调用的 executable 和改写统计。function 在这次调用中被追踪，调用点从零开始登记。"""
    SITES.clear()
    compiled = asm.compile(function, *arguments, mesh=mesh)
    serialized, counts = rewrite(compiled)
    return asm.load(serialized, compiled), counts
