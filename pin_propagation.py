"""libtpu 0.0.49 的进程内研究用 hook；只在专用进程中同步编译，不允许其他线程并发编译。

在 PinPrecoloring 返回之前补上内存空间传播。只修改 private mapping 中的一个虚表指针，退出时恢复；不修改 wheel 或机器代码。
"""
from collections.abc import Iterator
from contextlib import contextmanager
import ctypes
import importlib.metadata
import os
from pathlib import Path
import platform
import struct
import subprocess
import threading

import libtpu

PATH = Path(libtpu.__file__).with_name('libtpu.so')
SLOT = 0x1dd5d0f0 + 0x38
RUN = 0x0f44cb60
_lock = threading.RLock()
_depth = 0
_library: ctypes.CDLL | None = None
_tpu: ctypes.CDLL | None = None
_slot = 0
_original = b''
_replacement = b''
_libc = ctypes.CDLL(None, use_errno=True)
_libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.mprotect.restype = ctypes.c_int

def mapping(address: int) -> tuple[int, int, int]:
    """返回包含地址的 mapping 边界与实际权限，避免把数据页恢复为代码页权限。"""
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        start, end = (int(part, 16) for part in fields[0].split('-'))
        if start <= address < end:
            assert fields[1][3] == 'p', fields
            protection = sum(value for char, value in zip(fields[1][:3], (1, 2, 4)) if char != '-')
            return start, end, protection
    raise RuntimeError(f'没有映射 {address:#x}')

def prepare() -> None:
    global _library, _tpu, _slot, _original, _replacement
    if _library is not None:
        return
    assert importlib.metadata.version('libtpu') == '0.0.49'
    assert platform.machine() == 'x86_64'
    _tpu = ctypes.CDLL(str(PATH))
    # 当前 ELF 第一个 PT_LOAD 的 p_offset=p_vaddr=0；其余数据段的文件偏移不等于 VA。
    with PATH.open('rb') as file:
        header = file.read(64)
        offset = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        file.seek(offset)
        segments = [struct.unpack('<IIQQQQQQ', file.read(size)) for _ in range(count)]
    first = next(segment for segment in segments if segment[0] == 1)
    assert first[2] == first[3] == 0
    base = next(int(line.split()[0].split('-')[0], 16) for line in Path('/proc/self/maps').read_text().splitlines() if line.split()[-1] == str(PATH) and line.split()[2] == '00000000')
    _slot = base + SLOT
    _original = struct.pack('<Q', base + RUN)
    assert ctypes.string_at(_slot, 8) == _original
    assert ctypes.string_at(base + RUN, 8) == bytes.fromhex('554889e541574156')
    assert ctypes.c_uint64.from_address(base + 0x1e132790 + 0x38).value == base + 0x1a213190
    source = Path(__file__).with_suffix('.cpp')
    output = Path('/tmp') / f'pin_propagation-{os.getpid()}.so'
    subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra', str(source), '-o', str(output)], check=True)
    library = ctypes.CDLL(str(output))
    library.configure.argtypes = [ctypes.c_size_t]
    library.configure(base)
    library.statistics.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    _replacement = struct.pack('<Q', ctypes.cast(library.after_pin, ctypes.c_void_p).value)
    _library = library

def swap(expected: bytes, replacement: bytes) -> None:
    assert ctypes.string_at(_slot, 8) == expected
    page_size = os.sysconf('SC_PAGE_SIZE')
    page = _slot & -page_size
    start, end, protection = mapping(_slot)
    assert start <= page and _slot + 8 <= min(end, page + page_size)
    assert _libc.mprotect(page, page_size, protection | 2) == 0, ctypes.get_errno()
    try:
        ctypes.memmove(_slot, replacement, 8)
        assert ctypes.string_at(_slot, 8) == replacement
    except BaseException:
        ctypes.memmove(_slot, expected, 8)
        raise
    finally:
        if _libc.mprotect(page, page_size, protection) != 0:
            ctypes.memmove(_slot, expected, 8)
            _libc.mprotect(page, page_size, protection)
            raise OSError(ctypes.get_errno(), '恢复虚表页面权限失败')
    assert mapping(_slot)[2] == protection

@contextmanager
def propagate_after_pin() -> Iterator[None]:
    """锁只协调本模块调用者；调用者必须保证本进程没有其他并发编译。"""
    global _depth
    with _lock:
        prepare()
        if _depth == 0:
            swap(_original, _replacement)
        _depth += 1
        try:
            yield
        finally:
            _depth -= 1
            if _depth == 0:
                swap(_replacement, _original)

def statistics() -> dict[str, int]:
    assert _library is not None
    values = (ctypes.c_uint64 * 4)()
    _library.statistics(values)
    return dict(zip(('pin_calls', 'propagation_runs', 'propagation_changed', 'propagation_errors'), values))

def restored() -> bool:
    return _depth == 0 and _slot != 0 and ctypes.string_at(_slot, 8) == _original
