"""在当前进程里改写已加载的 libtpu.so 的机器码（不改磁盘上的文件）。地址都是 libtpu 0.0.49 的 ELF 虚拟地址。"""
from importlib.metadata import distribution
from pathlib import Path
import ctypes
import os
import struct
import subprocess

assert distribution('libtpu').version == '0.0.49', distribution('libtpu').version
PATH = Path(str(distribution('libtpu').locate_file('libtpu/libtpu.so'))).resolve()
LIBRARY = ctypes.CDLL(str(PATH))
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

def base() -> int:
    """libtpu 的装载基址。它的代码段的 ELF 虚拟地址等于文件偏移，所以任何一段映射的“起始地址减文件偏移”都是基址。"""
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(PATH) and fields[1] == 'r-xp':
            return int(fields[0].split('-')[0], 16) - int(fields[2], 16)
    raise RuntimeError('libtpu is not mapped')

def write(address: int, old: bytes, new: bytes) -> None:
    """把 ELF 虚拟地址 address 处的 old 换成等长的 new。"""
    assert len(old) == len(new)
    target = base() + address
    assert ctypes.string_at(target, len(old)) == old, (hex(address), ctypes.string_at(target, len(old)).hex())
    page = target & -4096
    length = (target + len(new) - page + 4095) & -4096
    assert LIBC.mprotect(page, length, 7) == 0
    ctypes.memmove(target, new, len(new))
    assert LIBC.mprotect(page, length, 5) == 0

def opcode_names() -> list[str]:
    """LLO 的全部 opcode 名，下标是编号：读 LloOpcodeString() 用的那张表。"""
    return [name.decode() for name in (ctypes.c_char_p * 0x1d1).from_address(base() + 0x1e110c10)]

def native(source: str) -> ctypes.CDLL:
    """编译并装载同目录下的一个 C++ 文件，把 libtpu 的装载基址告诉它。产物放在 /tmp。"""
    # 每个进程编译到自己的文件：几个进程同时跑时，共用一个文件名会读到别人写了一半的产物。
    output = Path('/tmp') / f'{Path(source).stem}.{os.getpid()}.so'
    subprocess.run(['g++', '-std=c++17', '-O2', '-fno-omit-frame-pointer', '-fPIC', '-shared', '-Wall', '-Wextra', str(Path(__file__).with_name(source)), '-o', str(output)], check=True)
    library = ctypes.CDLL(str(output))
    library.configure.argtypes = [ctypes.c_size_t]
    library.configure(base())
    return library

def find_cave(near: int, taken: tuple[int, ...] = ()) -> int:
    """在 near 附近的代码段里找一段 14 字节的 int3 填充（函数之间的对齐空隙），返回它的 ELF 虚拟地址。"""
    start = near - (near % 16)
    for address in range(start, start + (64 << 20), 16):
        for candidate in (address - 16, address - 14):
            if candidate not in taken and ctypes.string_at(base() + candidate, 14) == b'\xcc' * 14 and ctypes.string_at(base() + candidate + 14, 2) != b'\xcc\xcc':
                return candidate
    raise RuntimeError('no cave found')

def redirect_call(site: int, old: bytes, cave: int, function: ctypes._CFuncPtr) -> None:
    """把 site 处的一条 5 字节 call 改为调用 function。function 在别的库里，rel32 够不着，
    所以先在 libtpu 里找一段 14 字节的 int3 填充（cave），放一条跳到绝对地址的 jmp，call 改为指向它。"""
    target = ctypes.cast(function, ctypes.c_void_p).value
    write(cave, b'\xcc' * 14, bytes.fromhex('ff2500000000') + struct.pack('<Q', target))
    write(site, old, b'\xe8' + struct.pack('<i', cave - (site + 5)))

def allow_sublane_gather() -> None:
    """tpu.dynamic_gather 沿子通道（维度 0）在 TPU v4 上可用：见 libtpu_sublane_gather.cpp。"""
    # lower-to-LLO 里“这一代 TPU 是否支持沿子通道 gather”的判断：失败分支的 je 换成 nop。
    write(0xf16ab3c, bytes.fromhex('0f8495020000'), b'\x90' * 6)
    # llo.vector_sublane_permute 译成 C++ LLO 时调用 LloRegionBuilder::VpermuteSlane 的地方；cave 是 CreateVectorSetIarRaw 后面的填充。
    library = native('libtpu_sublane_gather.cpp')
    redirect_call(0x151cb51f, bytes.fromhex('e87c415a04'), 0x1971f872, library.sublane_gather)
    KEEP.append(library)
    GATHER.append(library)

def hook_latency() -> ctypes.CDLL:
    """接管 TPU v4 的指令延迟表：见 libtpu_latency.cpp。返回的库有 set_rule(前, 后, 周期数) 和 dump(路径)。"""
    library = native('libtpu_latency.cpp')
    library.dump.argtypes = [ctypes.c_char_p]
    # TPU v4 用的是 xla::pufferfish::LatencyTablePufferfish，虚表在 0x1e05cfa0，前 16 字节是到顶偏移和类型信息，LatencyBetweenInternal 是第 4 个虚函数。
    slot = base() + 0x1e05cfa0 + 0x10 + 0x18
    assert ctypes.c_uint64.from_address(slot).value == base() + 0x191361f0, hex(ctypes.c_uint64.from_address(slot).value)
    page = slot & -4096
    assert LIBC.mprotect(page, 4096, 3) == 0
    ctypes.c_uint64.from_address(slot).value = ctypes.cast(library.latency_between, ctypes.c_void_p).value
    assert LIBC.mprotect(page, 4096, 1) == 0
    KEEP.append(library)
    return library

# LLO 的 opcode 编号（libtpu 0.0.49）。
SET_IAR = (2, 3, 4)
LOAD_INDEXED, STORE_INDEXED, STORE_INDEXED_MASKED = 0x31, 0x3f, 0x43
STORES = (0x3e, 0x3f, 0x40, 0x43, 0x44, 0x45)
# 借用的“体系结构寄存器”编号：原本只有 vector-cmem-store 写它、没有指令读它。
IAR_REGISTER = 10
# 第二个借用的编号（原本只有 BarnaCore 的 scalar pop 用）：表示“有一次写内存还在途”。
STORE_IN_FLIGHT = 9

def teach_iar(load_gap: int = 7, store_gap: int = 3, any_store_gap: int = 0, barrier: bool = False, per_unit: bool = False) -> ctypes.CDLL:
    """让 LLO 的排程器知道 IAR 是一个有状态的寄存器。

    libtpu 的每个 opcode 有一条 28 字节的属性记录（internal::opcode_info_big，在 0x1ebbb8a0），第 8 到 19 字节是它读的体系结构寄存器，
    第 20 到 27 字节是它写的，0xff 结尾。置换寄存器、分段模式、随机数种子都登记在这里，排程器据此保证“先写后读”并向延迟表询问间隔；
    IAR 没有登记；编译器原本只在程序开头生成 vsetiar。这里补上登记，再在延迟表里给出 vsetiar 到按索引读写的间隔。

    硬件还有一条延迟表里没有的规则：任何一次写内存之后的几个周期内，vld.iar 都要等（它的地址到执行时才知道）。
    any_store_gap 不为零时，把每种向量写登记为写一个假想的寄存器、vld.iar 登记为读它，用同样的机制表达这条规则。

    libtpu 原来处理 IAR 的办法是把 vsetiar 当作内存屏障（LloOpcodeIsMemoryBarrier，与 fence、DMA、vwait 同类）：
    它之前的内存读写必须先做完，之后的每一次读都要等够写后读的 7 个周期。barrier=False 时把这四个 opcode 从屏障里拿掉，
    先后次序改由上面登记的寄存器依赖保证。
    """
    # per_unit=True 时改用编号 11：它按指令的 unit 编号分成几个独立的寄存器（原本是累加器用的），
    # 生成 IAR 指令的一方把 unit 编号设成 IAR 的编号，两个 IAR 就互不牵连。
    register = 11 if per_unit else IAR_REGISTER
    if per_unit:
        # 编号 11 要求指令带 unit 编号，没有就中止编译；编译器的某些遍会重建指令而不带上它，改为没有时按 0 号处理。
        write(0xedde590, bytes.fromhex('0f84ce000000'), b'\x90' * 6)
    table = base() + 0x1ebbb8a0
    page = table & -4096
    assert LIBC.mprotect(page, 28 * 0x1d1 + (table - page), 3) == 0
    for opcode in SET_IAR:
        assert ctypes.string_at(table + 28 * opcode + 20, 1) == b'\xff'
        ctypes.memmove(table + 28 * opcode + 20, bytes([register, 0xff]), 2)
    for opcode in (LOAD_INDEXED, STORE_INDEXED, STORE_INDEXED_MASKED):
        assert ctypes.string_at(table + 28 * opcode + 8, 1) == b'\xff'
        ctypes.memmove(table + 28 * opcode + 8, bytes([register, 0xff]), 2)
    library = hook_latency()
    if not barrier:
        # LloOpcodeIsMemoryBarrier 是一张按 opcode 减一索引的跳转表（0x9219a64，每项是相对表头的偏移）：1 到 4 号原来跳到“返回真”，改为跳到“返回假”。
        jumps = 0x9219a64
        write(jumps, struct.pack('<4i', *[0xe67b01c - jumps] * 4), struct.pack('<4i', *[0xe67b01d - jumps] * 4))
    if any_store_gap:
        ctypes.memmove(table + 28 * LOAD_INDEXED + 8, bytes([register, STORE_IN_FLIGHT, 0xff]), 3)
        for opcode in STORES:
            assert ctypes.string_at(table + 28 * opcode + 20, 1) == b'\xff'
            ctypes.memmove(table + 28 * opcode + 20, bytes([STORE_IN_FLIGHT, 0xff]), 2)
            library.set_rule(opcode, LOAD_INDEXED, any_store_gap)
    for opcode in SET_IAR:
        library.set_rule(opcode, LOAD_INDEXED, load_gap)
        library.set_rule(opcode, STORE_INDEXED, store_gap)
        library.set_rule(opcode, STORE_INDEXED_MASKED, store_gap)
        # 一个 IAR 在同一个 bundle 里被读、同时又被装入新值，在设备上会出错：延迟表原来给的间隔是 0，改成 1。
        for reader in (LOAD_INDEXED, STORE_INDEXED, STORE_INDEXED_MASKED):
            library.set_rule(reader, opcode, 1)
    return library

KEEP: list[ctypes.CDLL] = []
GATHER: list[ctypes.CDLL] = []
