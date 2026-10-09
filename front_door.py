"""四个 intrinsic 的“正门”版本：不用占位、不改写 executable，由打了补丁的 libtpu 0.0.49 在编译时生成指令。只适用于 TPU v4。

Pallas 一侧只用到 TPU dialect 里现成的一个 op：tpu.dynamic_gather（沿子通道，TPU v4 上原本被拒绝）。
它在 libtpu 把 LLO dialect 译成 C++ LLO 的地方被一个 Python 钩子接住，按下标的写法分成五种：

- 下标就是源：分段转置（vsxpose，宽度 8）；
- 下标是“偏移 异或 RAW”：按相对偏移读，y[s, l] = x[s + offset[s, l], l]（vsetiar.raw + vld.iar）；
- 下标是常数：按固定模式重排（vld.sshfl），常数的第 s 个十六进制位是第 s 个子通道要取的行；小于 8 的常数表示把那一行广播出来；
- 下标是“偏移 异或 SCATTER + n”：按相对偏移写，y[s + offset[s, l], l] = x[s, l]，没写到的位置是 FILLS[n]（vsetiar.raw + vst.iar）；
- 其他：普通的 gather，y[s, l] = x[index[s, l], l]，也就是 jnp.take_along_axis(x, index, axis=0)。

要经过 TC VMEM 的源只存一次：同一个值的几次重排和按偏移读共用一个 tile，按偏移写的结果直接从它写进去的那个 tile 读。

install() 之后编译的 kernel 都经过这个钩子；它同时调用 libtpu_patch.teach_iar()，让排程器把 IAR 当作有状态的寄存器。
"""
import ctypes
import os
import re
import struct
import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jaxlib.mlir import ir
from jaxlib.mosaic.python import tpu
import libtpu_patch

RAW = 0x5eed0100
SCATTER = 0x5eed0200
FILLS = (0, -8388608)
# StrideSpec 按值放在一个寄存器里；普通的整 TC VREG 读写是高、低 32 位各为 1。
UNIT_STRIDE = 0x100000001
S32 = 4
SEGMENTED_B32 = 3
# LLO 的 opcode 编号。
TRANSPOSE, ROTATE = 0xa5, 0x39
FOLD_BUSY = int(os.environ.get('FOLD_BUSY', '224'))
FOLD_LOAD = int(os.environ.get('FOLD_LOAD', '0'))
# 用几个 IAR。
IARS = int(os.environ.get('IARS', '2'))
KEEP: list = []
DEBUG = os.environ.get('FRONT_DOOR_DEBUG') == '1'
COUNTS: dict[str, int] = {}

gather_p = jax_core.Primitive('front_door_gather')
gather_p.def_abstract_eval(lambda x, index: x)
mosaic_lowering.lowering_rules[tpu_core.CoreType.TC][gather_p] = lambda ctx, x, index: tpu.dynamic_gather(x, index, [0])
def _gather(x: jax.Array, index: jax.Array) -> jax.Array:
    """tpu.dynamic_gather 沿子通道时源只能是一个 TC VREG：行数多于 8 时逐个 TC VREG 做。"""
    if x.shape[0] == 8:
        return gather_p.bind(x, index)
    return jnp.concatenate([gather_p.bind(x[row:row + 8], index[row:row + 8]) for row in range(0, x.shape[0], 8)], axis=0)

def fold(x: jax.Array) -> jax.Array:
    """分段转置：y[s, 8g+r] = x[r, 8g+s]。x 是 int32 的一个 TC VREG。"""
    return _gather(x, x)

def sublane_shuffle(x: jax.Array, pattern: int) -> jax.Array:
    """沿子通道按固定模式重排：第 s 个子通道取 pattern 第 s 个十六进制位所指的行。"""
    assert pattern == 0 or pattern >= 8, 'a pattern below 8 means "broadcast that row"'
    return _gather(x, jnp.full(x.shape, pattern, jnp.int32))

def sublane_gather(x: jax.Array, offset: jax.Array) -> jax.Array:
    """沿子通道按相对偏移读：y[s, l] = x[s + offset[s, l], l]，行号须落在 0 到 7 之内。"""
    return _gather(x, offset ^ RAW)

def sublane_scatter(x: jax.Array, offset: jax.Array, fill: int = 0) -> jax.Array:
    """沿子通道按相对偏移写：y[s + offset[s, l], l] = x[s, l]，没有被写到的位置是 fill。目的行的限制与 asm_intrinsics.sublane_scatter 相同。"""
    return _gather(x, offset ^ (SCATTER + FILLS.index(fill)))

def install() -> None:
    base = libtpu_patch.base()
    function = lambda address, result, *arguments: ctypes.CFUNCTYPE(result, *arguments)(base + address)
    pointer, word = ctypes.c_void_p, ctypes.c_uint64
    make_shape = function(0xc3e80b0, pointer, pointer, ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_size_t)   # xla::ShapeUtil::MakeShape
    allocate = function(0x1975ac90, pointer, pointer, pointer, ctypes.c_char_p, ctypes.c_size_t)   # LloRegionBuilder::AllocateScopedVmem
    store = function(0x19792850, pointer, pointer, pointer, pointer, pointer, word, pointer)   # LloRegionBuilder::Vst(地址, 值, 掩码, StrideSpec, 位移)
    load = function(0x1977d650, pointer, pointer, word, pointer, pointer, word)   # LloRegionBuilder::VldHelper(optional<IAR 编号>, 地址, 掩码, StrideSpec)
    sublane_id = function(0x1975faa0, pointer, pointer)   # LloRegionBuilder::Vslaneid
    subtract = function(0x19762580, pointer, pointer, pointer, pointer)   # LloRegionBuilder::VsubS32
    immediate = function(0x19762570, pointer, pointer, ctypes.c_int)   # LloRegionBuilder::VimmS32
    set_iar = function(0x1971f860, pointer, ctypes.c_uint, pointer, pointer)   # LloInstruction::CreateVectorSetIarRaw(IAR 编号, 值, region)
    store_indexed = function(0x197200c0, pointer, ctypes.c_uint, pointer, pointer, pointer, pointer, word, pointer)   # LloInstruction::CreateVectorStoreIndexed(IAR 编号, 地址, 位移, 值, 掩码, StrideSpec, region)
    append = function(0x19752860, pointer, pointer, pointer, word)   # LloRegion::AppendInstruction
    transpose = function(0x197935e0, pointer, pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, word, word)   # LloRegionBuilder::Vxpose(值, 模式, 宽度, 第几块, 一共几块, XLU, optional<int>)
    transpose_result = function(0x19794230, pointer, pointer, ctypes.c_int, ctypes.c_int)   # LloRegionBuilder::Vxposeres(XLU, 第几块)
    to_string = function(0x197eb840, pointer, pointer, pointer)   # LloValue::ToString
    operand = function(0xed3d1c0, pointer, pointer, ctypes.c_int)   # LloValue::operands(int)
    operand_count = function(0x18acc390, ctypes.c_long, pointer)   # LloValue::operands_size
    load_indexed = function(0x1971ee50, pointer, ctypes.c_uint, pointer, pointer, pointer, word, pointer)   # LloInstruction::CreateVectorLoadIndexed(IAR 编号, 地址, 位移, 掩码, StrideSpec, region)
    set_unit = function(0xf59b990, None, pointer, ctypes.c_int)   # LloValue::set_unit_id
    shuffle_pattern = function(0x1977e390, pointer, pointer, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t)   # LloRegionBuilder::CreateShufflePattern(Span<int const>)
    load_shuffled = function(0x1977f200, pointer, pointer, pointer, pointer, pointer)   # LloRegionBuilder::VldSshfl(地址, 模式, 掩码)
    # (region, 值) -> 存着它的 tile。
    stored: dict[tuple[int, int], int] = {}

    def text(value: int) -> str:
        buffer = (ctypes.c_char * 24)()
        to_string(buffer, value)
        raw = bytes(buffer)
        return (ctypes.string_at(int.from_bytes(raw[0:8], 'little'), int.from_bytes(raw[8:16], 'little')) if raw[23] & 0x80 else raw[:raw[23]]).decode(errors='replace')

    def opcode(value: int) -> str:
        return re.search(r'opcode=([\w.-]+)', text(value)).group(1)

    def scratch(builder: int, rows: int) -> int:
        # xla::Shape 按值返回到调用者给的位置；留足空间，用完不析构。
        shape = ctypes.create_string_buffer(1024)
        KEEP.append(shape)
        make_shape(shape, S32, (ctypes.c_long * 2)(rows, 128), 2)
        return allocate(builder, shape, b'front_door', 10)

    def memory(builder: int, region: int, value: int) -> int:
        """存着 value 的 tile；还没有存过就分配一个存进去。"""
        if (region, value) not in stored:
            stored[region, value] = scratch(builder, 8)
            store(builder, stored[region, value], value, None, UNIT_STRIDE, None)
            count('store')
        return stored[region, value]

    def indexed(builder: int, region: int, offset: int, address: int, value: int | None = None) -> int:
        """装一个 IAR，再按它读（value 为 None）或写。两个 IAR 轮流用；指令的 unit 编号设成 IAR 的编号，排程器据此把两个 IAR 当作两个寄存器。"""
        number = COUNTS.get('indexed', 0) % IARS
        count('indexed')
        # unit 编号要在指令挂进 region 之前设好：挂进去的时候依赖图就会来问它读写哪个寄存器。
        instructions = [set_iar(number, offset, region), load_indexed(number, address, None, None, UNIT_STRIDE, region) if value is None else store_indexed(number, address, None, value, None, UNIT_STRIDE, region)]
        for instruction in instructions:
            set_unit(instruction, number)
            result = append(region, instruction, 0)
        return result

    def count(kind: str) -> None:
        COUNTS[kind] = COUNTS.get(kind, 0) + 1

    @ctypes.CFUNCTYPE(pointer, pointer, pointer, pointer, ctypes.c_int)
    def hook(builder: int, source: int, index: int, granularity: int) -> int:
        region = ctypes.c_void_p.from_address(builder).value
        # 下标先经过一次到“任意类型”的转换；往回找到真正算出它的那条指令。
        actual = index
        while opcode(actual) == 'vector-type-cast':
            actual = operand(actual, 0)
        if DEBUG:
            print('front door:', opcode(source), '|', opcode(actual), [text(operand(actual, position))[:200] for position in range(operand_count(actual))], flush=True)
        if actual == source:
            count('fold')
            transpose(builder, source, SEGMENTED_B32, 8, 0, 1, 0, 0)
            return transpose_result(builder, 0, 0)
        constant = re.match(r'LloConstant\{.*?value=(0x[0-9a-f]+|-?\d+)', text(actual))
        if DEBUG and not constant and opcode(actual).startswith('vector-imm'):
            print('front door: unparsed constant', text(actual), flush=True)
        if constant:
            count('shuffle')
            code = int(constant.group(1), 0) & 0xffffffff
            rows = [code] * 8 if code < 8 else [code >> (4 * position) & 15 for position in range(8)]
            return load_shuffled(builder, memory(builder, region, source), shuffle_pattern(builder, (ctypes.c_int * 8)(*rows), 8), None)
        code = None
        if opcode(actual) == 'vector-xor-u32':
            for position in range(2):
                constant = re.match(r'LloConstant\{.*?value=(0x[0-9a-f]+)', text(operand(actual, position)))
                if constant:
                    code, offset = int(constant.group(1), 0) & 0xffffffff, operand(actual, 1 - position)
        if code is not None and code & ~0xff in (RAW, SCATTER):
            if code & ~0xff == RAW:
                count('gather by offset')
                return indexed(builder, region, offset, memory(builder, region, source))
            count('scatter')
            # 两个 tile：按偏移落到 8 行之后的元素写进后一个，等于被丢弃。
            address = scratch(builder, 16)
            store(builder, address, immediate(builder, FILLS[code & 0xff]), None, UNIT_STRIDE, None)
            indexed(builder, region, offset, address, source)
            result = load(builder, 0, address, None, UNIT_STRIDE)
            # 结果就在这个 tile 的前 8 行里，之后对它的重排直接从这里读。
            stored[region, result] = address
            return result
        count('gather')
        # vsetiar.raw 装的是相对自己所在行的偏移，gather 的下标是绝对行号。
        return indexed(builder, region, subtract(builder, index, sublane_id(builder)), memory(builder, region, source))

    # tpu.dynamic_gather 的 lower-to-LLO 里“这一代 TPU 是否支持沿子通道 gather”的判断：失败分支的 je 换成 nop。
    libtpu_patch.write(0xf16ab3c, bytes.fromhex('0f8495020000'), b'\x90' * 6)
    # llo.vector_sublane_permute 译成 C++ LLO 时调用 LloRegionBuilder::VpermuteSlane 的地方。
    libtpu_patch.redirect_call(0x151cb51f, bytes.fromhex('e87c415a04'), 0x1971f872, hook)
    library = libtpu_patch.teach_iar(per_unit=IARS > 1, any_store_gap=int(os.environ.get('ANY_STORE_GAP', '0')))
    # 延迟表认为转置提交 104 个周期之后同一个 XLU 就能再收旋转；对分段转置，实测要到 224 个周期之后。
    # 规则按 opcode 登记，所以普通的 vxpose 也会被当成这样，偏保守。
    # 这条规则只在排程时生效：分配 XLU 的那一步若也看到它，会把后面的归约全部挤到另一个 XLU 上。
    library.set_late_rule(TRANSPOSE, ROTATE, FOLD_BUSY)
    if FOLD_LOAD:
        library.set_rule(TRANSPOSE, ROTATE, FOLD_LOAD)
    KEEP.append(hook)
