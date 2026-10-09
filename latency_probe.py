"""运行另一个脚本，同时记录编译时 LLO 的排程器向 TPU v4 的延迟表问过哪些指令对、得到的回答是多少。

用法：latency_probe.py [teach] <脚本> [参数...]。带 teach 时先用 libtpu_patch.teach_iar() 让排程器认识 IAR。
脚本结束后打印回答大于 1 个周期的指令对，以及所有与 IAR、按索引读写有关的指令对。
"""
import atexit
import runpy
import sys
import tempfile
import libtpu_patch

def report(library, path: str) -> None:
    library.dump(path.encode())
    names = libtpu_patch.opcode_names()
    rows = [[int(word) for word in line.split()] for line in open(path)]
    print(f'{len(rows)} opcode pairs asked')
    for source, destination, count, low, high in sorted(rows, key=lambda row: (-row[4], names[row[0]], names[row[1]])):
        if high > 1 or any(word in names[source] + names[destination] for word in ('iar', 'indexed')):
            print(f'{names[source]:36s} -> {names[destination]:36s} asked {count:5d}  latency {low}..{high}')

def main() -> None:
    teach = sys.argv[1] == 'teach'
    library = libtpu_patch.teach_iar() if teach else libtpu_patch.hook_latency()
    atexit.register(report, library, tempfile.mktemp(prefix='latency-probe-', dir='/tmp'))
    sys.argv = sys.argv[2 if teach else 1:]
    runpy.run_path(sys.argv[0], run_name='__main__')

if __name__ == '__main__':
    main()
