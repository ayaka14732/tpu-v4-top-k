"""给编译出的直线型 kernel 重新排程：指令不变，只改它们所在的 bundle 和用的寄存器。

TensorCore 的向量指令严格按序发射，一个 bundle 在等（写后读同一个 tile、vsetiar 之后的 vld.iar、XLU 的结果还没回来），后面的全部跟着等。
编译器的排程器不知道其中几种等待，排出来的清单里有大量空等。这里把两次 DMA 等待之间的那一段读成一张依赖图，
按周期重新填 bundle：每条指令在它依赖的结果真正可用之后才发射，等待的周期用别的就绪指令填上；TC VREG、掩码、IAR 和 XLU 队列都重新分配。

reschedule(serialized) 返回新的 executable 和一份统计；simulate() 用同一套时序规则估计一段清单的周期数，用来核对模型。
"""
from dataclasses import dataclass, field
import os
import re
from tpuasm import BundleInsertion, insert_executable_bundles
from tpuasm.assembly_syntax import parse_assembly
from tpuasm.tc_solver import BundleSolver
from tpuasm.tpu_v4_tc_assembler import ISA
from tpuasm import tools as asm

# 时序规则，单位是周期。“软”间隔只影响快慢：早发射的指令会被硬件的计分板挡住，后面的跟着等。
# 结果写进寄存器之后，多少个周期才能被别的指令读：按产生它的槽分。vcvt 是 2（libtpu 自己的延迟表里的数）。
RESULT_LATENCY = {'va': int(os.environ.get('VA_LATENCY', '1')), 'vld': int(os.environ.get('VLD_LATENCY', '1')), 'vr': int(os.environ.get('VR_LATENCY', '1')), 'misc': 1, 's': 1, 'vst': 1}
STORE_TO_LOAD = int(os.environ.get('STORE_TO_LOAD', '7'))   # 写一个 tile 之后读同一个 tile
ANY_STORE_TO_INDEXED_LOAD = int(os.environ.get('ANY_STORE', '5'))   # 任何一次写（不论地址）之后的 vld.iar
SET_TO_INDEXED_LOAD = 7   # vsetiar 之后的 vld.iar
SET_TO_INDEXED_STORE = 3   # vsetiar 之后的 vst.iar；按 3 个周期处理
PUSH_INTERVAL = 8   # 同一个 XLU 队列上相邻两次提交
XLU_LATENCY = {'vrot': 69, 'vperm': 69, 'vsxpose': int(os.environ.get('FOLD_LATENCY', '126')), 'vxpose': 126}   # 其余（归约）是 79
VIF_ENTRIES = 20
VIF_RELEASE = 10
FOLD_BUSY = 224   # vsxpose 提交之后，它的队列这么久不能再收别的提交
FOLD_INTERVAL = 128   # 同一个队列上相邻两次 vsxpose
SLOTS = {'va': ('va0', 'va1'), 'vld': ('vld',), 'vst': ('vst',), 'vx': ('vx0', 'vx1'), 'vr': ('vr0', 'vr1'), 's': ('s0', 's1'), 'misc': ('misc',)}
POOLS = {'v': [f'v{n}' for n in range(32)], 'vm': [f'vm{n}' for n in range(8)], 'iar': ['iar0', 'iar1']}
READ_PORTS = 5
FITS: dict[str, bool] = {}
ISSUED: dict[tuple[int, bool], list[int]] = {}

@dataclass
class Instruction:
    slot: str
    mnemonic: str
    operands: list[str]
    defines: list[int] = field(default_factory=list)   # 操作数下标
    uses: list[int] = field(default_factory=list)
    loads: list[int] = field(default_factory=list)   # 读的 tile 地址
    stores: list[int] = field(default_factory=list)
    push: int | None = None   # 提交到的队列
    pop: int | None = None   # 这次取回对应的提交（指令编号）
    flags: list[str] = field(default_factory=list)
    # (前驱, 软间隔, 硬间隔)
    before: list[tuple[int, int, int]] = field(default_factory=list)
    # 每个被读的操作数对应的值（定义它的指令编号与操作数下标）；窗口之外定义的是 None。
    sources: dict[int, tuple[int, int] | None] = field(default_factory=dict)

def _kind(operand: str) -> str | None:
    for kind, pattern in (('v', r'v\d+'), ('vm', r'vm\d+'), ('s', r's\d+'), ('iar', r'iar\d'), ('trf', r'trf\d'), ('pcr', r'pcr\d')):
        if re.fullmatch(pattern, operand):
            return kind
    return None

def _parse(item: str, constants: dict[str, int | str]) -> Instruction:
    slot, body = item.split(': ', 1)
    mnemonic, _, rest = body.partition(' ')
    # 方括号里的逗号属于地址（[vmem:0x1fb, ss=0]），不分隔操作数。
    operands = [part.strip() for part in re.split(r',(?![^\[]*\])', rest)] if rest else []
    instruction = Instruction(re.sub(r'\d$', '', slot), mnemonic, operands)
    head = mnemonic.split('.')[0]
    for index, operand in enumerate(operands):
        if _kind(operand) in ('v', 'vm', 's', 'iar', 'pcr'):
            defines = index == 0 and (instruction.slot in ('va', 'vld', 'vr') or head in ('simm', 'vsetiar', 'vsetperm', 'vmand', 'vmmov', 'vmneg', 'vmor', 'vmxor'))
            (instruction.defines if defines else instruction.uses).append(index)
        memory = re.fullmatch(r'\[vmem:(0x[0-9a-f]+|s\d+)((?:, s[ms]=\w+)*)\]', operand)
        if memory:
            # 依赖按 tile 记：地址去掉行号。
            address = (constants[memory.group(1)] if memory.group(1).startswith('s') else int(memory.group(1), 16)) // 8 * 8
            if head == 'vst':
                # vst.iar 可以落到下一个 tile；它和带掩码的写都只改一部分，仍然排在上一次写之后。
                instruction.stores += [address, address + 8] if '.iar' in mnemonic else [address]
            else:
                # vld.iar 可以读到相邻的 tile；DMA 读的是源。
                instruction.loads += [address - 8, address, address + 8] if '.iar' in mnemonic else [address]
        elif '[' in operand:
            flag = re.search(r'\[sflag:(\d+)\]', operand)
            assert flag or re.match(r'\[(hbm|cmem):s\d+\]', operand), item
            if flag:
                instruction.flags.append(flag.group(1))
        if re.search(r'\[\w+:s\d+\]', operand):
            instruction.uses.append(index)
    if head == 'vld' and 'iar' in mnemonic:
        instruction.operands.append('iar' + re.search(r'iar(\d)', mnemonic).group(1))
        instruction.uses.append(len(instruction.operands) - 1)
    if head == 'simm':
        constants[operands[0]] = int(operands[1], 0)
    # vperm 用的是同一个队列上最近一次 vsetperm 装好的置换（pcr0 对 trf0，pcr1 对 trf1）：记成对那个置换寄存器的读。
    if head == 'vsetperm':
        constants['permutation ' + operands[0][-1]] = operands[0]
    if head == 'vperm':
        instruction.operands.append(constants['permutation ' + str(int(mnemonic.split('.')[1]) & 1)])
        instruction.uses.append(len(instruction.operands) - 1)
    # 只接受已经核对过读写关系的指令；别的一律拒绝，免得把依赖建错。
    known = instruction.slot in ('va', 'vld', 'vst', 'vx', 'vr') or head in ('simm', 'dma', 'vsyncadd', 'vmand', 'vmmov', 'vmneg', 'vmor', 'vmxor')
    assert known and head not in ('vpush', 'spop') and 'erf' not in body and not mnemonic.startswith('dma.general'), f'unsupported instruction: {item}'
    return instruction

def _register(operand: str) -> str:
    """操作数里的寄存器名：[hbm:s2] 这样的地址取里面的标量寄存器。"""
    scalar = re.search(r'\[\w+:(s\d+)\]', operand)
    return scalar.group(1) if scalar else operand

def _slots(kind: str, mnemonic: str) -> list[str]:
    """这条指令可以放进哪些槽。两个向量 ALU 槽并不对称（移位只能在 va1），能通用的指令先用限制多的那个槽之外的。"""
    names = [name for name in SLOTS[kind] if (name, mnemonic) in ISA.signatures_by_mnemonic]
    return names[::-1] if kind == 's' else names

def _latency(mnemonic: str) -> int:
    return XLU_LATENCY.get(mnemonic.split('.')[0], 79)

def build(bundles: list[list[str]]) -> tuple[list[Instruction], list[int]]:
    """把一段直线型清单读成依赖图。返回指令表，以及每条指令原来所在的 bundle。"""
    instructions: list[Instruction] = []
    home: list[int] = []
    constants: dict[str, int] = {}
    writer: dict[str, tuple[int, int]] = {}   # 寄存器 -> 最近一次定义
    readers: dict[str, list[int]] = {}
    stored: dict[int, int] = {}
    loaded: dict[int, list[int]] = {}
    pushed: dict[int, list[int]] = {0: [], 1: []}
    flagged: dict[str, int] = {}
    last_dma = None
    for number, items in enumerate(bundles):
        members = []
        # 同一个 bundle 里读先于写：先让所有指令认领它们读的值，再登记这个 bundle 写的。
        for item in items:
            if item in ('misc: vnop', ''):
                continue
            instruction = _parse(item, constants)
            me = len(instructions)
            members.append(me)
            for index in instruction.uses:
                name = _register(instruction.operands[index])
                source = writer.get(name)
                instruction.sources[index] = source
                if source is not None:
                    gap = (SET_TO_INDEXED_LOAD, 1) if _kind(name) == 'iar' and instruction.slot == 'vld' else (SET_TO_INDEXED_STORE, SET_TO_INDEXED_STORE) if _kind(name) == 'iar' else (2 if instructions[source[0]].mnemonic.startswith('vcvt') else RESULT_LATENCY.get(instructions[source[0]].slot, 1), 1)
                    instruction.before.append((source[0], *gap))
                readers.setdefault(name, []).append(me)
            for address in instruction.loads:
                if address in stored:
                    instruction.before.append((stored[address], STORE_TO_LOAD, 1))
                loaded.setdefault(address, []).append(me)
            queue = next((int(operand[-1]) for operand in instruction.operands if _kind(operand) == 'trf'), None)
            if instruction.slot == 'vx':
                # vsetperm 占用它那个 XLU 的提交间隔，但没有结果要取回。
                instruction.push = queue if queue is not None else int(instruction.mnemonic.split('.')[1]) & 1
                if queue is not None:
                    pushed[queue].append(me)
            elif instruction.slot == 'vr':
                instruction.pop = pushed[queue].pop(0)
                instruction.before.append((instruction.pop, _latency(instructions[instruction.pop].mnemonic) + 1, 1))
            for flag in instruction.flags:
                if flag in flagged:
                    instruction.before.append((flagged[flag], 1, 1))
                flagged[flag] = me
            if instruction.mnemonic.startswith('dma.'):
                if last_dma is not None:
                    instruction.before.append((last_dma, 1, 1))
                last_dma = me
            instructions.append(instruction)
            home.append(number)
        for me in members:
            instruction = instructions[me]
            for index in instruction.defines:
                name = instruction.operands[index]
                if _kind(name) in ('s', 'pcr'):
                    # 标量寄存器和置换寄存器不改名：新的定义排在旧值的全部读者之后。
                    instruction.before += [(other, 1, 1) for other in readers.get(name, []) if other != me] + ([(writer[name][0], 1, 1)] if name in writer else [])
                writer[name] = (me, index)
                readers[name] = []
            for address in instruction.stores:
                instruction.before += [(other, 1, 1) for other in loaded.get(address, []) if other != me] + ([(stored[address], 1, 1)] if address in stored else [])
                stored[address] = me
                loaded[address] = []
    assert not pushed[0] and not pushed[1], 'an XLU result is still in flight at the end of the window'
    return instructions, home

def simulate(instructions: list[Instruction], home: list[int], vif: bool = False, issued: list[int] | None = None) -> int:
    """按原来的 bundle 划分，用时序规则估计这段清单要多少个周期。

    默认只算向量发射一侧（计分板：源操作数、单元和队列）。vif=True 时把标量发射也接上，规则取自教程第三章第 5 节：
    含向量指令的 bundle 标量发射后进入 VIF，向量发射后 10 个周期才释放，未释放的达到 20 项时标量发射停住；
    向量发射最早在标量发射后 1 个周期，含 vld、vst 的 bundle 是 2 个周期。issued 收每个 bundle 的向量发射时刻。
    """
    time: dict[int, int] = {}
    now = -1
    scalar = 0
    releases: list[int] = []
    next_push = {0: 0, 1: 0}
    next_fold = {0: 0, 1: 0}
    last_store = -100
    for number in sorted(set(home)):
        members = [index for index, bundle in enumerate(home) if bundle == number]
        now += 1
        if vif and any(instructions[index].slot[0] != 's' for index in members):
            pending = sorted(release for release in releases if release > scalar)
            if len(pending) >= VIF_ENTRIES:
                scalar = pending[len(pending) - VIF_ENTRIES]
            releases = pending
            now = max(now, scalar + 1 + any(instructions[index].slot in ('vld', 'vst') for index in members))
        for index in members:
            instruction = instructions[index]
            now = max([now] + [time[other] + soft for other, soft, _ in instruction.before if home[other] != number])
            if instruction.push is not None:
                now = max(now, next_fold[instruction.push] if instruction.mnemonic.startswith('vsxpose') else next_push[instruction.push])
            if instruction.mnemonic.startswith('vld.iar'):
                now = max(now, last_store + ANY_STORE_TO_INDEXED_LOAD)
        for index in members:
            time[index] = now
            instruction = instructions[index]
            if instruction.stores:
                last_store = now
            if instruction.push is not None:
                next_push[instruction.push] = now + (FOLD_BUSY if instruction.mnemonic.startswith('vsxpose') else PUSH_INTERVAL)
                next_fold[instruction.push] = now + (FOLD_INTERVAL if instruction.mnemonic.startswith('vsxpose') else PUSH_INTERVAL)
        releases.append(now + VIF_RELEASE)
        scalar += 1
        if issued is not None:
            issued.append(now)
    return now + 1

def _fits(members: list[str]) -> bool:
    """这些指令能不能编码进同一个 bundle：立即数单元、读写端口等限制都交给汇编器判断。"""
    text = '{ ' + ' ; '.join(members) + ' }'
    if text not in FITS:
        fragment = parse_assembly(f'.target {asm.TARGET}\n{text}\n', fragment=True)
        try:
            BundleSolver(ISA, fragment.bundles[0], fragment, 0).solve()
            FITS[text] = True
        except ValueError:
            FITS[text] = False
    return FITS[text]

def schedule(instructions: list[Instruction], window: int, hold: bool = True) -> tuple[list[list[str]], int]:
    """重新排程。返回新的 bundle 列表（每个是指令文本的列表）和模型估计的周期数；每个 bundle 的估计发射时刻记在 ISSUED 里。

    window：只在“原清单里最早的那条还没排的指令”往后这么多条之内挑。原来的顺序用 32 个 TC VREG 是排得下的，
    离它越远，同时活着的值越多；窗口太大时寄存器会用尽（这里不会溢出），太小时可挑的指令不够。
    hold：一次按索引读快到期时先不写内存。排程是贪心的，两种做法哪个好因程序而异。
    """
    count = len(instructions)
    after: list[list[int]] = [[] for _ in range(count)]
    for index, instruction in enumerate(instructions):
        for other, _, _ in instruction.before:
            after[other].append(index)
    # 到结尾的最长路径（按软间隔），既是优先级，也给出最晚该发射的时刻。
    tail = [0] * count
    for index in range(count - 1, -1, -1):
        for successor in after[index]:
            soft = max(soft for other, soft, _ in instructions[successor].before if other == index)
            tail[index] = max(tail[index], tail[successor] + soft)
    # 每个值的读者，用来在最后一个读者发射之后收回寄存器。
    pop_of = {instruction.pop: index for index, instruction in enumerate(instructions) if instruction.pop is not None}
    pending: dict[tuple[int, int], int] = {}
    for instruction in instructions:
        for index, source in instruction.sources.items():
            if source is not None and _kind(_register(instruction.operands[index])) not in ('s', 'pcr'):
                pending[source] = pending.get(source, 0) + 1
    # 窗口之外定义的寄存器保持原名，不参与分配。
    pinned = {_register(instruction.operands[index]) for instruction in instructions for index, source in instruction.sources.items() if source is None}
    free = {kind: [name for name in names if name not in pinned] for kind, names in POOLS.items()}
    holding: dict[tuple[int, int], str] = {}
    time: dict[int, int] = {}
    placed: dict[int, int] = {}
    waiting = set(range(count))
    next_push = {0: 0, 1: 0}
    last_store = -100
    in_flight: dict[int, list[int]] = {0: [], 1: []}
    queue_of: dict[int, int] = {}
    bundles: list[list[str]] = []
    issued: list[int] = []
    order = sorted(range(count), key=lambda index: (-tail[index], index))
    now = 0
    while waiting:
        taken: set[str] = set()
        popped: set[int] = set()
        ports = 0
        members: list[str] = []
        blocked_until: list[int] = []
        hard_blocked = False
        dead: list[str] = []
        # 已经装好 IAR、还没读的按索引读里，最早到期的那个时刻。
        due = [max([last_store + ANY_STORE_TO_INDEXED_LOAD, now] + [time[other] + soft for other, soft, _ in instructions[index].before]) for index in waiting if instructions[index].mnemonic.startswith('vld.iar') and not any(other in waiting for other, _, _ in instructions[index].before)]
        indexed_due = min(due, default=10**9)
        # 三遍：先只排不让寄存器吃紧的；什么都排不了时，放行那些能让某个读者凑齐操作数的；最后才是任意一条。
        oldest = min(waiting)
        # 队列先进先出：窗口里有一次取回被队首的结果挡着时，队首的那次取回不受窗口限制。
        urgent = {pop_of[queue[0]] for queue in in_flight.values() if any(pop_of[push] < oldest + window for push in queue[1:])}
        for relaxed in (0, 1, 2):
            for index in order:
                instruction = instructions[index]
                if index not in waiting or index >= oldest + window and index not in urgent:
                    continue
                if any(other in waiting for other, _, _ in instruction.before):
                    continue
                if any(placed[other] + hard > len(bundles) for other, _, hard in instruction.before):
                    hard_blocked = True
                    continue
                ready = max([0] + [time[other] + soft for other, soft, _ in instruction.before])
                queue = None
                if instruction.push is not None:
                    # 折叠留在原来的队列；其余的提交挑先空出来的那个。
                    choices = [instruction.push] if instruction.mnemonic.startswith(('vsxpose', 'vsetperm', 'vperm')) else sorted((0, 1), key=lambda queue: next_push[queue])
                    queue = choices[0]
                    ready = max(ready, next_push[queue])
                if instruction.mnemonic.startswith('vld.iar'):
                    ready = max(ready, last_store + ANY_STORE_TO_INDEXED_LOAD)
                if ready > now:
                    blocked_until.append(ready)
                    continue
                # 一次按索引读就要到期时先不写内存：任何一次写都会让它再等几个周期。
                if instruction.stores and hold and relaxed == 0 and indexed_due < now + ANY_STORE_TO_INDEXED_LOAD:
                    continue
                # 同一个 bundle 不能从同一个队列取回两次。
                if instruction.pop is not None and (in_flight[queue_of[instruction.pop]][0] != instruction.pop or queue_of[instruction.pop] in popped):
                    continue
                mnemonic = instruction.mnemonic
                if queue is not None:
                    mnemonic = re.sub(r'\.(\d)\.', lambda match: f'.{int(match.group(1)) & 2 | queue}.', mnemonic, count=1)
                slot = next((name for name in _slots(instruction.slot, mnemonic) if name not in taken), None)
                if slot is None:
                    continue
                reads = sum(1 for operand_index in instruction.uses if _kind(instruction.operands[operand_index]) == 'v')
                if ports + reads > READ_PORTS:
                    continue
                # 这条指令发射之后能收回的寄存器，和它要新占的。
                released = {source for source in instruction.sources.values() if source in pending and pending[source] == sum(1 for other in instruction.sources.values() if other == source)}
                wanted = [_kind(instruction.operands[operand_index]) for operand_index in instruction.defines]
                grows = {kind: wanted.count(kind) - sum(1 for source in released if _kind(holding[source]) == kind) for kind in ('v', 'vm')}
                if relaxed == 0 and any(grows[kind] > 0 and len(free[kind]) - grows[kind] < 1 for kind in grows):
                    continue
                if relaxed == 1 and max(grows.values()) > 0 and not any(all(other == index or other not in waiting for other, _, _ in instructions[successor].before) for successor in after[index]):
                    continue
                if any(wanted.count(kind) > len(free[kind]) + sum(1 for source in released if _kind(holding[source]) == kind) for kind in ('v', 'vm', 'iar')):
                    continue
                snapshot = {kind: list(names) for kind, names in free.items()}
                dead_before = len(dead)
                operands = list(instruction.operands)
                for operand_index, source in instruction.sources.items():
                    if source in holding:
                        operands[operand_index] = operands[operand_index].replace(_register(operands[operand_index]), holding[source])
                for source in released:
                    # TC VREG 和掩码在同一个 bundle 里读先于写，读完就可以让给这个 bundle 里的新结果；IAR 要等到下一个 bundle。
                    (dead if _kind(holding[source]) == 'iar' else free[_kind(holding[source])]).append(holding[source])
                for operand_index, source in instruction.sources.items():
                    if source in pending:
                        pending[source] -= 1
                for operand_index in instruction.defines:
                    kind = _kind(operands[operand_index])
                    if kind not in ('s', 'pcr'):
                        holding[index, operand_index] = operands[operand_index] = free[kind].pop(0)
                        if (index, operand_index) not in pending:
                            # 没有人读的结果：这个 bundle 之后就可以覆盖。
                            dead.append(holding[index, operand_index])
                target = queue if instruction.push is not None else queue_of[instruction.pop] if instruction.pop is not None else None
                operands = [f'trf{target}' if _kind(operand) == 'trf' else operand for operand in operands]
                if mnemonic.startswith('vld.iar'):
                    mnemonic = re.sub(r'iar\d', operands.pop(), mnemonic)
                if mnemonic.startswith('vperm'):
                    operands.pop()
                text = f'{slot}: {mnemonic} ' + ', '.join(operands)
                if not _fits(members + [text]):
                    # 放不进这个 bundle：把刚才的寄存器分配退回去。
                    free.update(snapshot)
                    del dead[dead_before:]
                    for operand_index, source in instruction.sources.items():
                        if source in pending:
                            pending[source] += 1
                    for operand_index in instruction.defines:
                        holding.pop((index, operand_index), None)
                    continue
                if instruction.push is not None:
                    queue_of[index] = queue
                    if not mnemonic.startswith('vsetperm'):
                        in_flight[queue].append(index)
                    next_push[queue] = now + (FOLD_BUSY if mnemonic.startswith('vsxpose') else PUSH_INTERVAL)
                if instruction.pop is not None:
                    in_flight[target].pop(0)
                    popped.add(target)
                taken.add(slot)
                if instruction.stores:
                    last_store = now
                ports += reads
                members.append(text)
                time[index] = now
                placed[index] = len(bundles)
                waiting.discard(index)
            if members or blocked_until:
                break
        for name in dead:
            free[_kind(name)].append(name)
        if members:
            bundles.append(members)
            issued.append(now)
            now += 1
        elif blocked_until:
            # 没有可以发射的指令：下一个 bundle 会被计分板挡到那个时刻，不用垫空 bundle。
            now = min(blocked_until)
        else:
            if not hard_blocked:
                ready = [index for index in order if index in waiting and index < oldest + window and not any(other in waiting for other, _, _ in instructions[index].before)]
                detail = [(instructions[index].slot, instructions[index].mnemonic, instructions[index].operands) for index in ready[:12]]
                raise AssertionError(f'the schedule is stuck at cycle {now}, bundle {len(bundles)}: {len(waiting)} instructions left, free registers { {kind: len(names) for kind, names in free.items()} }, in flight {in_flight}, ready {detail}')
            bundles.append(['misc: vnop'])
            issued.append(now)
            now += 1
    ISSUED[window, hold] = issued
    return bundles, now

def reschedule(serialized: bytes, window: int | None = None, span: tuple[int, int] | None = None) -> tuple[bytes, dict[str, int]]:
    """重排输入 DMA 的等待之后、第一次输出 DMA 的等待之前的那一段。不给 window 时试几个，取模型估计最短的。"""
    if span is None:
        waits = asm.find_bundles(serialized, 'vwait')
        span = waits[0] + 1, waits[1]
    first, last = span
    (_, _, image), = asm.executable_programs(serialized)
    lines = asm.format_assembly(image, target=asm.TARGET).splitlines()
    spans = asm._bundle_spans(lines)
    exact = {pc: '\n'.join(lines[spans[pc][0]:spans[pc][1] + 1]) for pc in range(first, last)}
    original = [[part.strip(' {}\n') for part in re.sub(r'\.encoding \{[^}]*\}', '', exact[pc]).split(';') if part.strip(' {}\n')] for pc in range(first, last)]
    instructions, home = build(original)
    before = simulate(instructions, home)
    results = []
    for trial in [(size, hold) for size in ([window] if window else (1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256)) for hold in (True, False)]:
        try:
            estimate, bundles = schedule(instructions, *trial)[::-1]
            results.append((estimate, bundles, trial))
        except AssertionError:
            # 这个窗口把寄存器用尽了；只给了一个窗口时把错误报出来。
            if window:
                raise
    if not results:
        # 每个窗口都把寄存器用尽了：保持原样。
        return serialized, {'bundles before': last - first, 'bundles after': last - first, 'modelled cycles before': before, 'modelled cycles after': before, 'window': 0, 'hold stores': 0}
    estimate, bundles, chosen = min(results, key=lambda result: (result[0], len(result[1])))
    (record, index, _), = asm.executable_programs(serialized)
    source = f'.target {asm.TARGET}\n' + ''.join('{ ' + ' ; '.join(members) + ' }\n' for members in bundles)
    # 新的清单整段换掉原来的那一段；它更短，多出来的 bundle 直接删除。
    patched = insert_executable_bundles(serialized, {(record, index): [BundleInsertion(first, source, delete=last - first)]})
    return patched, {'bundles before': last - first, 'bundles after': len(bundles), 'modelled cycles before': before, 'modelled cycles after': estimate, 'window': chosen[0], 'hold stores': int(chosen[1])}
