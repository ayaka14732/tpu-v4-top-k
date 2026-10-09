"""在 libtpu 的编译流程里面重排指令：接管 LLO 的 VLIW 排程（ScheduleNodesForVliwBundles），在它排好的顺序上再按周期排一遍。

钩子是一个 Python 函数。原函数返回一串依赖图节点之后，让 libtpu 自己把这些节点连同依赖边打印成文本，
从文本里读出每条指令的助记符和“前驱、间隔”，按周期做一次表调度，再把节点指针按新的顺序写回去。
之后的打包、寄存器分配和溢出仍由编译器完成，所以这里不需要管寄存器，也不需要改写 executable。

install(window) 之后编译的 kernel 都会经过它；window 的含义与 reschedule.py 相同。只对 libtpu 0.0.49 有效。
"""
import ctypes
import os
import re
import struct
import libtpu_patch

SCHEDULE = 0xf5b07f0   # ScheduleNodesForVliwBundles(Span<LloDependencyGraphNode*>, LloDependencyGraph&)，返回 StatusOr<vector<节点指针>>
CALL_SITES = (0xf57d942, 0xf57da01)   # LloDependencyGraphOptimizations::OptimizeOnGraph 里调用它的两处
TO_STRING = 0x15797ce0   # LloDependencyGraph::ToString(Span<节点指针>, PrintOptions)
PRINT_EVERYTHING = 0x0101010101010101
CAPACITY = {'va': 2, 'vld': 1, 'vst': 1, 'vx': 2, 'vr': 2, 's': 2, 'misc': 1}
SET_TO_INDEXED_LOAD = 7
ANY_STORE_TO_INDEXED_LOAD = 5
FOLD_BUSY = 224
FOLD_INTERVAL = 128
IAR_GAP = int(os.environ.get('VLIW_IAR_GAP', '1'))
REASSIGN_IARS = os.environ.get('VLIW_IARS', '1') == '1'
REASSIGN_XLUS = os.environ.get('VLIW_XLUS', '1') == '1'
PUSH_INTERVAL = 8
FIFO_DEPTH = int(os.environ.get('VLIW_FIFO_DEPTH', '1000'))
# 同时活着的值的上限。TC VREG 有 32 个，但寄存器分配之后的清单里同时装着有用的值的最多 23 到 25 个（register_occupancy.py），超出的由编译器溢出；实测 28 最好。
BUDGET = {'v': int(os.environ.get('VLIW_VREGS', '28')), 'vm': int(os.environ.get('VLIW_MASKS', '1000'))}
# 把溢出放进排程：先排一遍，一个值相邻两次被读（或算出来到第一次被读）隔得超过 SPLIT 个周期，就在依赖图里真的加一条 vst 和一条 vld，
# 后面那批读者改读 vld 的结果。之后这两条指令与别的指令一样参加排程，寄存器分配器看到的是已经拆短的生命期。0 是不拆。
SPLIT = int(os.environ.get('VLIW_SPLIT', '0'))
# 拆之前那一遍用的上限：这一遍只是用来看每个值隔多久才被读，可以比真正排的时候宽。
SPLIT_REGISTERS = int(os.environ.get('VLIW_SPLIT_VREGS', '28'))
STORE_TO_LOAD = 7
STATISTICS: list[dict[str, int]] = []
KEEP: list = []

def _slot(mnemonic: str) -> str:
    """指令占哪一类槽，按 LLO 文本里的助记符判断。"""
    if mnemonic.startswith('vpop.') or mnemonic.startswith('vpop'):
        return 'vr'
    if re.search(r'\.xlu\d', mnemonic):
        return 'vx'
    if mnemonic.startswith('vld'):
        return 'vld'
    if mnemonic.startswith(('vst', 'vset.iar', 'vsetiar')):
        return 'vst'
    if mnemonic.startswith(('smov', 'dma', 's', 'int', 'ptr')) and not mnemonic.startswith('v'):
        return 's'
    return 'va' if mnemonic.startswith('v') else 'misc'

def parse(text: str) -> tuple[list[str], dict[str, str], dict[str, list[tuple[str, int]]], dict[str, list[str]]]:
    """从依赖图的文本里读出节点（按打印的顺序）、助记符、每个节点的前驱与间隔，以及每个节点读的寄存器值（%v、%vm 开头的操作数）。"""
    names: list[str] = []
    mnemonics: dict[str, str] = {}
    before: dict[str, list[tuple[str, int]]] = {}
    reads: dict[str, list[str]] = {}
    declared = found = 0
    current = ''
    for line in text.splitlines():
        head = re.match(r'(%\S+) = (\S+)', line)
        # 每个节点下面列的是它的后继：“↳ (间隔) :: 后继”；紧跟着的“↴”一行只是把这条边反过来再写一遍。
        edge = re.match(r'\s+↳ \((\d+)\) :: (%\S+) = ', line)
        if head:
            current = head.group(1)
            names.append(current)
            mnemonics[current] = head.group(2)
            reads[current] = re.findall(r'%vm?\d+', line.split('/*')[0].split(' = ', 1)[1])
            declared += int(re.search(r'out_degree = (\d+)', line).group(1))
            before.setdefault(current, [])
        elif edge:
            found += 1
            before.setdefault(edge.group(2), []).append((current, int(edge.group(1))))
    # 每个节点声明的出边数要与读到的边数对得上；对不上说明有的边没认出来，这时不能重排。
    assert declared == found, (declared, found)
    return names, mnemonics, before, reads

def reorder(names: list[str], mnemonics: dict[str, str], before: dict[str, list[tuple[str, int]]], reads: dict[str, list[str]], window: int, reassign: bool = False, requeue: bool = False, lazy: frozenset[int] = frozenset(), registers: int | None = None) -> tuple[list[int], int, int, dict[int, int], dict[int, int], dict[int, int]]:
    """按周期做表调度，返回新的顺序（原下标的排列）、估计的周期数和 bundle 数，以及重新分配的 IAR（节点下标 -> IAR 编号）。

    reassign：IAR 的编号是生成指令时轮流定下的，依赖图里每个 IAR 的各次使用因此连成一条按源码顺序的链；
    哪一组的数据晚到，同一条链上排在它后面的各组都得等。打开之后忽略这两条链，把两个 IAR 当作两个名额，排到谁就给谁。
    requeue：XLU 也一样。分配 XLU 的那一遍给每次提交定了队列，同一个队列上的提交、取回各自按源码顺序连成一条链。
    打开之后去掉这些链，每次提交现挑一个队列，取回跟着它的提交走，只保证每个队列先进先出。返回值的倒数第二项是节点下标 -> 队列。
    lazy：这些节点是拆生命期时加的 vld，不自己排，由读者带出来：读者别的条件都满足的那个周期读回来，读者下个周期发射。最后一项是每个节点的发射时刻。
    """
    index = {name: number for number, name in enumerate(names)}
    count = len(names)
    preds = [[(index[other], gap) for other, gap in before[name] if other in index] for name in names]
    sets = ['vsetiar' in mnemonics[name] for name in names]
    uses = [mnemonics[name].startswith(('vld.iar', 'vst.iar')) for name in names]
    # 大的 kernel 会被分成几段交给这一步；一次装入和用它的读写不在同一段里时，这一段的 IAR 都保持原样。
    complete = all(any(sets[other] for other, _ in preds[number]) for number in range(count) if uses[number]) and all(any(uses[number] and (owner, gap) in preds[number] for number in range(count) for gap in (1, 3, 7)) for owner in range(count) if sets[owner])
    if os.environ.get('VLIW_DEBUG') and window > 1:
        print('IAR groups:', sum(sets), 'sets', sum(uses), 'uses; set predecessors per use:', sorted({sum(1 for other, _ in preds[number] if sets[other]) for number in range(count) if uses[number]}), 'uses per set:', sorted({sum(1 for number in range(count) if uses[number] and any(other == owner for other, _ in preds[number])) for owner in range(count) if sets[owner]}), 'complete', complete, flush=True)
    if not (reassign and complete):
        sets, uses = [False] * count, [False] * count
    # 每次按索引读写用的是哪一次装入；装入指令之间、读写与下一次装入之间的边去掉。
    group = {number: next(other for other, _ in preds[number] if sets[other]) for number in range(count) if uses[number]}
    left = {number: sum(1 for owner in group.values() if owner == number) for number in range(count) if sets[number]}
    preds = [[(other, gap) for other, gap in preds[number] if not (sets[number] and (sets[other] or uses[other]))] for number in range(count)]
    # 每次取回对应哪一次提交：前驱里间隔不小于 69 的那次提交。对不齐（一次转置取回多次、vsetperm）时不动队列。
    pushes = [_slot(mnemonics[name]) == 'vx' for name in names]
    taken = [_slot(mnemonics[name]) == 'vr' and 'xlu' in mnemonics[name] for name in names]
    # vperm 用的是它那个 XLU 上最近一次 vset.pattern.permute 装的模式：这两种指令留在原来的队列上，彼此的先后也不动。
    patterns = [pushes[number] and 'vset.pattern' in mnemonics[name] for number, name in enumerate(names)]
    pinned = [pushes[number] and 'perm' in mnemonics[name] for number, name in enumerate(names)]
    home = [int(re.search(r'xlu(\d)', mnemonics[name]).group(1)) & 1 if pinned[number] else None for number, name in enumerate(names)]
    source = {number: [other for other, gap in preds[number] if pushes[other] and gap >= 69] for number in range(count) if taken[number]}
    requeue = requeue and all(len(found) == 1 for found in source.values()) and sorted(found[0] for found in source.values()) == [number for number in range(count) if pushes[number] and not patterns[number]]
    if requeue:
        preds = [[(other, gap) for other, gap in preds[number] if not (pushes[number] and pushes[other] and not (pinned[number] and pinned[other]) or taken[number] and taken[other])] for number in range(count)]
    else:
        pushes, taken = [False] * count, [False] * count
    fifo: dict[int, list[int]] = {0: [], 1: []}
    next_push = {0: 0, 1: 0}
    next_pop = {0: 0, 1: 0}
    units: dict[int, int] = {}
    holder: list[int | None] = [None, None]
    free_at = [0, 0]
    assigned: dict[int, int] = {}
    # 到结尾的最长路径；原顺序是拓扑序，倒着扫一遍即可。
    tail = [0] * count
    for number in range(count - 1, -1, -1):
        for other, gap in preds[number]:
            tail[other] = max(tail[other], tail[number] + gap)
    slots = [_slot(mnemonics[name]) for name in names]
    indexed = [mnemonics[name].startswith('vld') and 'iar' in mnemonics[name] or mnemonics[name].startswith('vld.idx') for name in names]
    stores = [mnemonics[name].startswith('vst') for name in names]
    # 每次 XLU 提交用的是哪个队列（编号的最低位）；转置提交之后，它的队列要忙很久：下一次转置要等 128 个周期，别的操作要等 224 个。
    queue = [int(re.search(r'xlu(\d)', mnemonics[name]).group(1)) & 1 if slots[number] == 'vx' and not requeue else None for number, name in enumerate(names)]
    busy_until = {0: 0, 1: 0}
    fold_until = {0: 0, 1: 0}
    # 每个队列上已经提交、还没有取回的结果个数。队列是有深度的：编译器自己排的清单里一个队列上最多 9 个在途。
    pops = [int(re.search(r'xlu(\d)', mnemonics[name]).group(1)) & 1 if slots[number] == 'vr' and 'xlu' in mnemonics[name] and not requeue else None for number, name in enumerate(names)]
    in_flight = {0: 0, 1: 0}
    order = sorted(range(count), key=lambda number: (-tail[number], number))
    # 寄存器压力：一个值从算出来到最后一个读者发射之前占着一个 TC VREG（%v）或掩码（%vm）。
    # 这一步之后编译器才分配寄存器，放不下的会被它溢出到 TC VMEM，每次溢出都是一次写加一次读，所以这里自己留出余量。
    budget = {**BUDGET, 'v': registers or BUDGET['v']}
    kind = lambda name: 'vm' if name.startswith('%vm') else 'v' if name.startswith('%v') else None
    users = {name: 0 for name in names}
    for name in names:
        for value in set(reads[name]):
            if value in users:
                users[value] += 1
    live = {'v': 0, 'vm': 0}
    readers: list[list[int]] = [[] for _ in range(count)]
    for number, name in enumerate(names):
        for value in set(reads[name]):
            if value in index:
                readers[index[value]].append(number)
    waiting = set(range(count))

    def commit(number: int, used: dict[str, int], placed: list[int]) -> None:
        """把一条指令记进这个周期：占槽，它的结果开始活着，它读的值少一个读者。"""
        name = names[number]
        used[slots[number]] = used.get(slots[number], 0) + 1
        placed.append(number)
        if users[name] and kind(name):
            live[kind(name)] += 1
        for value in set(reads[name]):
            if value in users:
                users[value] -= 1
                if users[value] == 0:
                    live[kind(value)] -= 1

    def demand(number: int, used: dict[str, int], placed: list[int], now: int) -> None:
        """读者要用 lazy 里的一条指令了：前驱都到位就排在这个周期；它自己还缺一条 lazy 的前驱时先排那一条。"""
        for other, _ in preds[number]:
            if other in waiting and other not in placed and other in lazy:
                demand(other, used, placed, now)
        if number not in placed and used.get(slots[number], 0) < CAPACITY[slots[number]] and all(other not in waiting and time[other] + gap <= now for other, gap in preds[number]):
            commit(number, used, placed)
    time: dict[int, int] = {}
    result: list[int] = []
    now = bundles = 0
    last_store = -100
    relaxed = False
    release_lazy = False
    while waiting:
        oldest = min(waiting)
        used: dict[str, int] = {}
        placed: list[int] = []
        blocked: list[int] = []
        touched: set[tuple[str, int]] = set()
        deferred = False
        for number in order:
            if number not in waiting or number >= oldest + window:
                continue
            # lazy 里的 vld 不自己排：读者除了它以外什么都齐了、这个周期就能发射的时候才把它读回来，读者下个周期再排。
            if number in lazy and not release_lazy:
                continue
            missing = [other for other, _ in preds[number] if other in waiting or other in placed]
            if any(other not in lazy or other in placed or release_lazy for other in missing):
                continue
            ready = max([0] + [time[other] + gap for other, gap in preds[number] if other not in missing])
            if indexed[number]:
                ready = max(ready, last_store + ANY_STORE_TO_INDEXED_LOAD)
            if queue[number] is not None:
                ready = max(ready, fold_until[queue[number]] if 'xpose' in mnemonics[names[number]] else busy_until[queue[number]])
                if in_flight[queue[number]] >= FIFO_DEPTH and 'setperm' not in mnemonics[names[number]]:
                    continue
            if pushes[number]:
                folding = 'xpose' in mnemonics[names[number]]
                options = {unit: max(ready, next_push[unit], fold_until[unit] if folding else busy_until[unit]) for unit in ((0, 1) if home[number] is None else (home[number],)) if ('push', unit) not in touched}
                if not options:
                    continue
                unit = min(options, key=lambda unit: (options[unit], len(fifo[unit])))
                ready = options[unit]
            if taken[number]:
                unit = units[source[number][0]]
                if fifo[unit][0] != source[number][0] or ('pop', unit) in touched:
                    continue
                ready = max(ready, next_pop[unit])
            if sets[number]:
                available = [register for register in (0, 1) if holder[register] is None]
                if not available:
                    continue
                register = min(available, key=lambda register: free_at[register])
                ready = max(ready, free_at[register])
            # 还缺 lazy 的 vld 的读者提前一个周期看：下个周期能发射，这个周期就该把值读回来。
            if ready > now + bool(missing):
                blocked.append(ready - bool(missing))
                continue
            if not missing and used.get(slots[number], 0) == CAPACITY[slots[number]]:
                continue
            # 下面两段会改 IAR 的占用情况，所以放不放得下必须先判断完。
            name = names[number]
            defined = kind(name) if users[name] else None
            freed = [kind(value) for value in set(reads[name]) if value in users and users[value] == 1]
            if defined and not relaxed and live[defined] + 1 - freed.count(defined) > budget[defined] and freed.count(defined) == 0:
                deferred = True
                continue
            if missing:
                for other in missing:
                    demand(other, used, placed, now)
                blocked.append(now + 1)
                continue
            if pushes[number]:
                units[number] = unit
                if not patterns[number]:
                    fifo[unit].append(number)
                touched.add(('push', unit))
                next_push[unit] = now + PUSH_INTERVAL
                if folding:
                    busy_until[unit] = now + FOLD_BUSY
                    fold_until[unit] = now + FOLD_INTERVAL
            if taken[number]:
                units[number] = unit
                fifo[unit].pop(0)
                touched.add(('pop', unit))
                next_pop[unit] = now + PUSH_INTERVAL
            if sets[number]:
                holder[register] = number
                assigned[number] = register
            if uses[number]:
                owner = group[number]
                assigned[number] = assigned[owner]
                left[owner] -= 1
                if left[owner] == 0:
                    holder[assigned[owner]] = None
                    free_at[assigned[owner]] = now + IAR_GAP
            commit(number, used, placed)
        # 什么都没排上、又有指令只是因为寄存器上限被跳过：放开上限再试一次，而不是空等到下一个时刻。
        if not placed and (deferred or not blocked) and not relaxed:
            relaxed = True
            continue
        relaxed = False
        if placed:
            for number in placed:
                time[number] = now
                waiting.discard(number)
                if stores[number]:
                    last_store = now
                if queue[number] is not None and 'setperm' not in mnemonics[names[number]]:
                    in_flight[queue[number]] += 1
                if pops[number] is not None:
                    in_flight[pops[number]] -= 1
                if queue[number] is not None and 'xpose' in mnemonics[names[number]]:
                    busy_until[queue[number]] = now + FOLD_BUSY
                    fold_until[queue[number]] = now + FOLD_INTERVAL
            result += placed
            release_lazy = False
            now += 1
            bundles += 1
        elif not blocked and not release_lazy:
            # 实在排不下去了（读者被别的规则卡着）：这一遍把 lazy 的 vld 当普通指令排。
            release_lazy = True
        else:
            assert blocked, f'stuck at {names[oldest]} {mnemonics[names[oldest]]} (position {oldest}); waiting predecessors: {[(names[other], mnemonics[names[other]], other, gap) for other, gap in preds[oldest] if other in waiting]}'
            now = min(blocked)
    if os.environ.get('VLIW_DEBUG') == '2' and window > 1:
        by_time: dict[int, list[str]] = {}
        for number in result:
            by_time.setdefault(time[number], []).append(mnemonics[names[number]])
        print('reorder bundles:', ' | '.join(f'{moment}: {" ".join(members)}' for moment, members in by_time.items()), flush=True)
    if os.environ.get('VLIW_DEBUG') and window > 1:
        print('reorder plan:', 'requeue', requeue, ' '.join(f'{time[number]}:{mnemonics[names[number]]}/{units.get(number)}' for number in result if slots[number] in ('vx', 'vr')), flush=True)
    return result, now, bundles, assigned, units, time

def install(window: int) -> None:
    base = libtpu_patch.base()
    original = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)(base + SCHEDULE)
    set_unit = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)(base + 0xf59b990)   # LloValue::set_unit_id
    to_string = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64)(base + TO_STRING)

    function = lambda address, result, *arguments: ctypes.CFUNCTYPE(result, *arguments)(base + address)
    pointer, word = ctypes.c_void_p, ctypes.c_uint64
    make_builder = function(0x19758478, None, pointer, pointer, pointer)   # LloRegionBuilder::LloRegionBuilder(LloRegion*, HloInstruction const*)
    make_shape = function(0xc3e80b0, pointer, pointer, ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_size_t)   # xla::ShapeUtil::MakeShape
    allocate = function(0x1975ac90, pointer, pointer, pointer, ctypes.c_char_p, ctypes.c_size_t)   # LloRegionBuilder::AllocateScopedVmem
    # 这两个只造指令、不挂进 region：挂进去的指令不在正在排的这个基本块里，后面打包时没有 bundle 编号。
    store = function(0x1971f880, pointer, pointer, pointer, pointer, pointer, word, pointer)   # LloInstruction::CreateVectorStore(地址, 位移, 值, 掩码, StrideSpec, region)
    load = function(0x1971e6b0, pointer, pointer, pointer, pointer, word, pointer)   # LloInstruction::CreateVectorLoad(地址, 位移, 掩码, StrideSpec, region)
    operand = function(0xed3d1c0, pointer, pointer, ctypes.c_int)   # LloValue::operands(int)
    operand_count = function(0x18acc390, ctypes.c_long, pointer)   # LloValue::operands_size
    set_operand = function(0x19724900, None, pointer, ctypes.c_int, pointer)   # LloInstruction::SetOperandAt(int, LloValue*)
    add_node = function(0x1578d9d0, pointer, pointer, pointer, pointer, ctypes.c_long, pointer)   # LloDependencyGraph::AddNode(LloInstruction*, long, RangeSimplifier*)
    get_node = function(0x1578f5a0, pointer, pointer, pointer)   # LloDependencyGraph::GetNode(LloValue*)
    remove_edge = function(0x157906c0, None, pointer, pointer, pointer)   # LloDependencyGraph::RemoveEdge(节点, 节点)
    latency = function(0x1578d900, ctypes.c_int, pointer, pointer, pointer)   # LloDependencyGraph::GetLatencyBetween(值, 值)
    update_edge = function(0x1578d7e0, None, pointer, pointer, pointer, ctypes.c_int)   # LloDependencyGraph::UpdateEdge(节点, 节点, 间隔)

    def describe(nodes: int, count: int, graph: int) -> str:
        # 逐个节点打印：整段一起打印时文本的顺序是图里的顺序，不是这串节点的顺序。返回的字符串不释放。
        pieces = []
        for position in range(count):
            buffer = (ctypes.c_char * 24)()
            to_string(buffer, graph, nodes + 8 * position, 1, PRINT_EVERYTHING, 0)
            raw = bytes(buffer)
            pieces.append((ctypes.string_at(int.from_bytes(raw[0:8], 'little'), int.from_bytes(raw[8:16], 'little')) if raw[23] & 0x80 else raw[:raw[23]]).decode(errors='replace'))
        return '\n'.join(pieces)

    def split(nodes: int, count: int, graph: int) -> tuple[list[int], set[int], int]:
        """先排一遍，按排出来的时刻把隔得远的读者拆到一条新的 vld 后面。返回加了新节点之后的整串节点、其中新加的 vld，以及存了几个值。"""
        names, mnemonics, before, reads = parse(describe(nodes, count, graph))
        *_, time = reorder(names, mnemonics, before, reads, window, REASSIGN_IARS, REASSIGN_XLUS, registers=SPLIT_REGISTERS)
        index = {name: number for number, name in enumerate(names)}
        node = lambda number: pointer.from_address(nodes + 8 * number).value
        instruction = lambda number: pointer.from_address(node(number) + 0x10).value
        # 指令的第一个字段是它所在的 region；builder 用 region 就能构造。
        builder = ctypes.create_string_buffer(512)
        region = pointer.from_address(instruction(0)).value
        make_builder(builder, region, None)
        shape = ctypes.create_string_buffer(1024)
        make_shape(shape, 4, (ctypes.c_long * 2)(8, 128), 2)
        KEEP.extend([builder, shape])
        scratch = ctypes.create_string_buffer(64)

        def attach(created: int) -> int:
            """把一条新造的指令加进依赖图，返回它的节点。节点偏移 0x18 处指向基本块的成员，成员是一个打了标记的指令指针，这里自己造一个。"""
            add_node(scratch, graph, created, count + len(added), None)
            member = ctypes.c_uint64(created | 1)
            KEEP.append(member)
            pointer.from_address(get_node(graph, created) + 0x18).value = ctypes.addressof(member)
            # 指令偏移 8 处反过来指向它的成员（LloRegionMember 的构造函数也是这样连的）。
            pointer.from_address(created + 8).value = ctypes.addressof(member)
            return get_node(graph, created)

        added: list[int] = []
        loads: set[int] = set()
        # 原函数要求交给它的节点是拓扑序：vst 紧跟在算出值的指令后面，vld 放在那一批里最靠前的读者前面。
        after: dict[int, list[int]] = {}
        ahead: dict[int, list[int]] = {}
        values = 0
        for number, name in enumerate(names):
            readers = sorted((reader for reader, other in enumerate(names) if name in reads[other]), key=lambda reader: time[reader])
            if not name.startswith('%v') or name.startswith('%vm') or not readers or mnemonics[name].startswith(('vmov', 'vimm', 'vld')):
                continue
            # 按时刻把读者分批：与前一次读（第一批是与算出来的时刻）隔得超过 SPLIT 就另起一批；第一批之后的各批改读 vld。
            batches: list[list[int]] = []
            previous = time[number]
            for reader in readers:
                if time[reader] - previous > SPLIT or not batches:
                    batches.append([])
                batches[-1].append(reader)
                previous = time[reader]
            later = batches[1:] if time[readers[0]] - time[number] <= SPLIT else batches
            if not later:
                continue

            def redirect(batch: list[int], source: int) -> None:
                for reader in batch:
                    for position in range(operand_count(instruction(reader))):
                        if operand(instruction(reader), position) == instruction(number):
                            set_operand(instruction(reader), position, pointer.from_address(source + 0x10).value)
                    remove_edge(graph, node(number), node(reader))
                    update_edge(graph, source, node(reader), 1)

            values += 1
            address = allocate(builder, shape, b'park', 4)
            stored = store(address, None, instruction(number), None, 0x100000001, region)
            added.append(attach(stored))
            kept = added[-1]
            after.setdefault(number, []).append(kept)
            # 新节点的边要自己连：值到 vst，vst 到每条 vld（写后读同一个 tile 隔 7 个周期）。
            update_edge(graph, node(number), kept, latency(graph, instruction(number), stored))
            for batch in later:
                loaded = load(address, None, None, 0x100000001, region)
                added.append(attach(loaded))
                loads.add(added[-1])
                ahead.setdefault(min(batch), []).append(added[-1])
                update_edge(graph, kept, added[-1], STORE_TO_LOAD)
                redirect(batch, added[-1])
        return [member for number in range(count) for member in (*ahead.get(number, []), node(number), *after.get(number, []))], loads, values

    @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    def hook(result: int, nodes: int, count: int, graph: int) -> int:
        loads: set[int] = set()
        values = 0
        if SPLIT and count >= 32:
            sequence, loads, values = split(nodes, count, graph)
            if loads:
                extended = (ctypes.c_void_p * len(sequence))(*sequence)
                KEEP.append(extended)
                nodes, count = ctypes.addressof(extended), len(sequence)
        value = original(result, nodes, count, graph)
        # StatusOr<vector>：状态（1 是成功）、数据指针、元素个数。
        status, data, size = struct.unpack('<3Q', ctypes.string_at(result, 24))
        if status != 1 or size < 32:
            return value
        try:
            names, mnemonics, before, reads = parse(describe(data, size, graph))
        except (AssertionError, AttributeError):
            names = []
        if len(names) != size or len(set(names)) != size:
            STATISTICS.append({'nodes': size, 'parsed': len(names)})
            return value
        pointers = (ctypes.c_void_p * size).from_address(data)
        saved = list(pointers)
        new, cycles, bundles, assigned, units, _ = reorder(names, mnemonics, before, reads, window, REASSIGN_IARS, REASSIGN_XLUS, frozenset(position for position in range(size) if saved[position] in loads))
        _, cycles_before, bundles_before, *_ = reorder(names, mnemonics, before, reads, 1)
        for number, register in assigned.items():
            # 节点偏移 0x10 处是指令；IAR 的编号在指令偏移 0x38 指向的附属结构的 0x80 处，unit 编号用 set_unit_id 设。
            instruction = ctypes.c_void_p.from_address(saved[number] + 0x10).value
            ctypes.c_uint32.from_address(ctypes.c_void_p.from_address(instruction + 0x38).value + 0x80).value = register
            set_unit(instruction, register)
        for number, unit in units.items():
            # unit 编号之外，提交指令还登记了 source bus（指令偏移 0x1c 那个字的第 10、11 位，第 12 位表示有值），它的奇偶要与 XLU 一致。
            instruction = ctypes.c_void_p.from_address(saved[number] + 0x10).value
            set_unit(instruction, unit)
            flags = ctypes.c_uint16.from_address(instruction + 0x1c)
            if flags.value & 0x1000:
                flags.value = flags.value & 0xe3ff | unit << 10 | 0x1000
        for position, number in enumerate(new):
            pointers[position] = saved[number]
        STATISTICS.append({'nodes': size, 'modelled cycles before': cycles_before, 'modelled cycles after': cycles, 'bundles before': bundles_before, 'bundles after': bundles, 'values stored': values, 'loads added': len(loads)})
        return value

    cave = libtpu_patch.find_cave(SCHEDULE)
    old = lambda site: b'\xe8' + struct.pack('<i', SCHEDULE - (site + 5))
    libtpu_patch.redirect_call(CALL_SITES[0], old(CALL_SITES[0]), cave, hook)
    libtpu_patch.write(CALL_SITES[1], old(CALL_SITES[1]), b'\xe8' + struct.pack('<i', cave - (CALL_SITES[1] + 5)))
    KEEP.append(hook)

if os.environ.get('VLIW_REORDER'):
    install(int(os.environ['VLIW_REORDER']))
