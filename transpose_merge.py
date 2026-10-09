"""行宽 128、k 为 8 或 16、行数较多时的 top-k：整体转置之后，用败者树做八路合并。

f32[R,128] 先整体转置成 [128,R]，每条输入行落在一个通道上；每个子通道上的 16 个候选用比较网络排好，k = 8 时保留前 8 层，k = 16 时 16 层全留，得到每行八条有序的候选列。
之后 k 轮合并，每轮输出八个列首中最大的一个并推进它所在的列：比较结果保存在一棵败者树里，每轮只沿上一次胜者的路径重赛三场，按索引读（IAR）取下一个列首和底层的败者，中层和根的败者留在 TC VREG 中。
转置与候选列的排序由 Pallas 生成；合并和转回是手写的片段，插在编译出的 executable 里，再用 reschedule.py 重排。放在 shell.py 的外壳里计时，并在同一个进程里量官方 Pallas。

用法：transpose_merge.py [--k 16] <行数>...，行数在 8 到 128 之间。
"""
import chip
import argparse
from collections.abc import Callable
import json
from pathlib import Path
import re

import jax
from jax import Ref, lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from tpuasm import BundleInsertion, insert_executable_bundles
from tpuasm import tools as asm
from tpuasm.tools import bundle as b

import compare
import reschedule
import shell
import total_order

def layout(k: int) -> tuple[int, int, int, int, int]:
    """工作区的布局，行号相对工作区的起点：候选表的行数 T（候选的键在 [0, T)，下标在 [T, 2T)），底层败者的键与下标所在的行，按名次排好的值与下标所在的行（各 k / 8 个 tile）。工作区共 2T + 96 行。"""
    assert k in (8, 16), k
    table = 8 * k
    return table, 2 * table, 2 * table + 24, 2 * table + 48, 2 * table + 64

def game(winner: int, challenger: int, loser: int = 14) -> str:
    """一场比赛：v{winner}、v{winner + 1} 与 v{challenger}、v{challenger + 1} 是两组（键，下标）；胜者留在前一组，败者写到 v{loser}、v{loser + 1}。"""
    text = b(f'va0: vgt.8x128.s32 vm0, v{challenger}, v{winner}')
    text += b(f'va0: vsel.8x128 v{loser}, vm0, v{winner}, v{challenger} ; va1: vsel.8x128 v{winner}, vm0, v{challenger}, v{winner}')
    return text + b(f'va0: vsel.8x128 v{loser + 1}, vm0, v{winner + 1}, v{challenger + 1} ; va1: vsel.8x128 v{winner + 1}, vm0, v{challenger + 1}, v{winner + 1}')

def merge(work: int, k: int) -> str:
    """八路合并的片段，共 k 轮。结果按名次写到值与下标两块输出区：第 r 名在第 r // 8 个 tile 的第 r % 8 行。

    每个候选的下标字是 (原下标 << 7) | (8 × 层 + 列)：低七位是它在候选表中的行号，加 8 就是同一列的下一个候选。v17 是子通道号 s。
    底层四个节点的败者在 TC VMEM 中，每个节点的败者复制到它的两个子通道上，所以胜者列为 c 时用同一个偏移 c − s 就能读到它所在的节点。
    中层两个节点的败者在 v22–v25，根的败者在 v6、v7。v8、v9 是当前的全局胜者。
    """
    table, tree_key, tree_index, output_value, output_index = layout(k)
    text = b('va0: vlaneseq.8x128.u32 v17') + b('va1: vshrl.8x128.s32 v17, v17, 7') + b('va1: vshrl.8x128.s32 v18, v17, 1')
    # 第一轮：八个列首两两比赛，建好整棵树。
    for pair in range(4):
        for register, base, column in ((2 * pair, 0, 2 * pair), (2 * pair + 1, table, 2 * pair), (10, 0, 2 * pair + 1), (11, table, 2 * pair + 1)):
            text += b(f'vld: vld.sshfl.8x128 v{register}, [vmem:0x{work + base:x}], 0x{0x11111111 * column:x}')
        text += game(2 * pair, 10)
        text += b(f'va0: veq.8x128.s32 vm2, v18, {pair}')
        text += b(f'vst: vst.msk.8x128 [vmem:0x{work + tree_key:x}], vm2, v14') + b(f'vst: vst.msk.8x128 [vmem:0x{work + tree_index:x}], vm2, v15')
    text += game(0, 2, 22) + game(4, 6, 24) + game(0, 4, 6)
    text += b('va0: vmov.8x128 v8, v0 ; va1: vmov.8x128 v9, v1')
    for rank in range(k):
        # 输出胜者：键变回 f32 的位型，下标去掉低七位，写到第 rank // 8 个 tile 的第 rank % 8 行。
        text += b('va0: vlt.8x128.s32 vm0, v8, 0 ; va1: vxor.8x128.u32 v14, 0x7fffffff, v8')
        text += b('va0: vsel.8x128 v14, vm0, v14, v8 ; va1: vshrl.8x128.s32 v15, v9, 7')
        text += b(f'va0: veq.8x128.s32 vm7, v17, {rank % 8}')
        text += b(f'vst: vst.msk.8x128 [vmem:0x{work + output_value + rank // 8 * 8:x}], vm7, v14') + b(f'vst: vst.msk.8x128 [vmem:0x{work + output_index + rank // 8 * 8:x}], vm7, v15')
        if rank == k - 1:
            break
        # 胜者列 c 的下一个候选在表的第（低七位 + 8）行；一条列最多有 16 个候选，只有最后一轮才可能取完，那时不再读。
        # 底层节点的败者在偏移 c − s 处。
        text += b('va0: vand.8x128.u32 v10, 127, v9 ; va1: vand.8x128.u32 v18, 7, v9')
        text += b('va0: vadd.8x128.s32 v10, 8, v10 ; va1: vsub.8x128.s32 v11, v18, v17')
        text += b('va0: vsub.8x128.s32 v10, v10, v17 ; va1: vxor.8x128.u32 v19, v17, v18')
        text += b('vst: vsetiar.raw iar0, v10 ; va0: vand.8x128.u32 v20, 6, v19')
        text += b('vst: vsetiar.raw iar1, v11 ; va0: veq.8x128.s32 vm4, v20, 0')
        text += b(f'vld: vld.iar0.8x128 v8, [vmem:0x{work:x}]') + b(f'vld: vld.iar0.8x128 v9, [vmem:0x{work + table:x}]')
        text += b(f'vld: vld.iar1.8x128 v0, [vmem:0x{work + tree_key:x}]') + b(f'vld: vld.iar1.8x128 v1, [vmem:0x{work + tree_index:x}]')
        # 中层：胜者列小于 4 时重赛 v22、v23 那个节点，否则重赛 v24、v25。
        text += b('va0: vlt.8x128.s32 vm6, v18, 4')
        text += b('va0: vsel.8x128 v2, vm6, v22, v24 ; va1: vsel.8x128 v3, vm6, v23, v25')
        text += game(8, 0)
        text += b(f'vst: vst.msk.8x128 [vmem:0x{work + tree_key:x}], vm4, v14') + b(f'vst: vst.msk.8x128 [vmem:0x{work + tree_index:x}], vm4, v15')
        text += game(8, 2)
        text += b('va0: vsel.8x128 v22, vm6, v14, v22 ; va1: vsel.8x128 v23, vm6, v15, v23')
        text += b('va0: vsel.8x128 v24, vm6, v24, v14 ; va1: vsel.8x128 v25, vm6, v25, v15')
        text += game(8, 6, 6)
    return text

def carrier(rows: int, k: int) -> Callable[[jax.Array, Ref], tuple[jax.Array, jax.Array]]:
    """Pallas 生成的前半段：整体转置、键的变换、每个子通道上 16 个候选排出前 k 层，写进工作区。

    之后的几行只为让编译器保留候选表并给出各块 TC VMEM 的地址，带着三个唯一的立即数作标记；transform 把它们整段删掉，换成手写的合并。
    """
    table, base, *_ = layout(k)

    def body(x: jax.Array, work: Ref) -> tuple[jax.Array, jax.Array]:
        key = total_order.to_key(jnp.pad(x.T, ((0, 0), (0, 128 - rows))))
        position = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
        groups = [(key[t:t + 8], position[t:t + 8]) for t in range(0, 128, 8)]
        for level, (keys, indices) in enumerate(total_order.sort_columns(groups, k)):
            work[level * 8:level * 8 + 8] = keys
            work[table + level * 8:table + level * 8 + 8] = (indices << 7) | (level * 8 + position[:8])
        work[base + 16:base + 24] = work[base + 16:base + 24] ^ jnp.int32(0x2468ace0)

        @pl.loop(0, 2 * k)
        def retain(t: jax.Array) -> None:
            work[base:base + 8] = work[pl.ds(t * 8, 8)] ^ jnp.int32(0x13579bdf)

        dummy = work[base:base + 1, :k] ^ work[base + 16:base + 17, :k]
        values = lax.bitcast_convert_type(lax.bitcast_convert_type(x[:, :k], jnp.int32) ^ jnp.int32(0x11223344), jnp.float32)
        return values, jnp.broadcast_to(dummy, (rows, k))

    return body

def addresses(serialized: bytes, start: int, end: int, k: int) -> tuple[int, int, int, int]:
    """从载体的清单里读出：工作区的地址、两个结果的地址、候选表写完之后的第一个 bundle。

    地址要么是立即数，要么在 simm 装好的标量寄存器里。工作区由 0x2468ace0 那次读改写认出；值的输出是带着 0x11223344 标记的那几次带掩码的写，另一组带掩码的写是下标的输出。
    """
    (_, _, image), = asm.executable_programs(serialized)
    lines = asm.format_assembly(image, target=asm.TARGET).splitlines()
    spans = asm._bundle_spans(lines)
    constants: dict[str, int] = {}
    loads: list[tuple[int, str, int | None]] = []
    stores: list[tuple[int, int | None, str]] = []
    masked: set[tuple[int, int | None, str]] = set()
    tagged: dict[str, bool] = {}
    tagged_memory: dict[int | None, bool] = {}
    value_stores: list[tuple[int, int | None, str]] = []
    work = None
    table, base, *_ = layout(k)

    def address(expression: str) -> int | None:
        pieces = expression.split(',', 1)[0].split(' + ')
        base = int(pieces[0], 16) if pieces[0].startswith('0x') else constants.get(pieces[0])
        return None if base is None else base + (int(pieces[1], 16) if len(pieces) == 2 else 0)

    for pc in range(start, end):
        text = '\n'.join(lines[spans[pc][0]:spans[pc][1] + 1])
        pending: dict[str, bool] = {}
        for item in re.sub(r'\.encoding \{[^}]*\}', '', text).split(';'):
            if match := re.search(r'(?:va\d|vld|vr\d): (?:@!?p\d+ )?([\w.]+) (v\d+), (.*)', item):
                operation, destination, operands = match.groups()
                if operation.startswith('vld'):
                    memory = re.search(r'\[vmem:([^]]+)\]', operands)
                    pending[destination] = bool(memory and tagged_memory.get(address(memory[1]), False))
                elif operation.startswith('vmov'):
                    pending[destination] = tagged.get(operands.strip(' }\n'), False)
                else:
                    pending[destination] = '0x11223344' in operands
        for register, expression in re.findall(r'vld\.8x128 (v\d+), \[vmem:([^]]+)\]', text):
            loads.append((pc, register, address(expression)))
        for expression, register in re.findall(r'vst(?:\.msk)?\.8x128 \[vmem:([^]]+)\], (?:vm\d+, )?(v\d+)', text):
            stores.append((pc, address(expression), register))
            tagged_memory[address(expression)] = tagged.get(register, False)
            if tagged.get(register, False):
                value_stores.append((pc, address(expression), register))
        for expression, register in re.findall(r'vst\.msk\.8x128 \[vmem:([^]]+)\], vm\d+, (v\d+)', text):
            masked.add((pc, address(expression), register))
        if '0x2468ace0' in text:
            source = re.search(r'vxor\.8x128\.u32 v\d+, 0x2468ace0, (v\d+)', text)[1]
            work = next(found for at, register, found in reversed(loads) if register == source and at <= pc) - (base + 16)
        tagged.update(pending)
        for register, value in re.findall(r'simm\.s32 (s\d+), (-?\d+)', text):
            constants[register] = int(value)
    assert work is not None
    filled = [item for item in stores if item[1] in range(work, work + 2 * table, 8)]
    assert {target for _, target, _ in filled} == set(range(work, work + 2 * table, 8)), (work, stores)
    cut = max(pc for pc, _, _ in filled) + 1
    outputs = sorted({target for pc, target, register in stores if target is not None and not work <= target < work + base + 96 and pc >= cut and (pc, target, register) in masked})
    values = min(target for pc, target, register in value_stores if (pc, target, register) in masked and pc >= cut)
    indices = min(target for target in outputs if not values <= target < values + len(outputs) * 4)
    assert set(outputs) == set(range(values, values + len(outputs) * 4, 8)) | set(range(indices, indices + len(outputs) * 4, 8)), (outputs, values, indices)
    return work, values, indices, cut

def transform(rows: int, k: int, saved: dict[str, object]) -> Callable[[asm.Compiled], bytes]:
    """把载体中候选表写完之后的部分换成：八路合并，再把按名次排好的值和下标各整体转置一次，写到两个结果里；合并那一段随后重排。"""
    padded = (rows + 7) // 8 * 8

    def rewrite(compiled: asm.Compiled) -> bytes:
        serialized = asm.serialize(compiled)
        start, end = shell.kernel_span(serialized)
        work, values, indices, cut = addresses(serialized, start, end, k)
        *_, output_value, output_index = layout(k)
        body = merge(work, k)
        merge_bundles = body.count('{')
        # 按名次排好的值和下标各是 k / 8 个 TC VREG，分块送进转置：第一块带 start，最后一块带 end。
        for tile in range(k // 8):
            flags = '.start' * (tile == 0) + '.end' * (tile == k // 8 - 1)
            body += b(f'vld: vld.8x128 v0, [vmem:0x{work + output_value + 8 * tile:x}]') + b(f'vld: vld.8x128 v1, [vmem:0x{work + output_index + 8 * tile:x}]')
            body += b(f'vx0: vxpose.0{flags}.8x128 trf0, v0, {padded}') + b(f'vx1: vxpose.1{flags}.8x128 trf1, v1, {padded}')
        for offset in range(0, padded, 8):
            body += b('vr0: vpop.8x128 v2, trf0 ; vr1: vpop.8x128 v3, trf1')
            body += b(f'vst: vst.8x128 [vmem:0x{values + offset:x}], v2') + b(f'vst: vst.8x128 [vmem:0x{indices + offset:x}], v3')
        assert 'cfence' in asm.bundle_text(serialized, end - 1)
        # 被删掉的那一段里有 retain 循环的分支；先换成空操作，删除时就没有需要重定位的分支。
        (_, _, image), = asm.executable_programs(serialized)
        lines = asm.format_assembly(image, target=asm.TARGET, encoding='canonical').splitlines()
        spans = asm._bundle_spans(lines)
        branches = [pc for pc in range(cut, end - 1) if re.search(r'\bsbr\.', '\n'.join(lines[spans[pc][0]:spans[pc][1] + 1]))]
        serialized = asm.edit_bundles(serialized, {pc: (asm.bundle_text(serialized, pc, encoding='exact'), '{ misc: vnop }') for pc in branches})
        (record, index, _), = asm.executable_programs(serialized)
        serialized = insert_executable_bundles(serialized, {(record, index): [BundleInsertion(cut, f'.target {asm.TARGET}\n' + body, delete=end - 1 - cut)]})
        serialized, _ = reschedule.reschedule(serialized, span=(cut, cut + merge_bundles))
        saved.update(compiled=compiled, serialized=serialized)
        return serialized

    return rewrite

def directed(rows: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """定向输入：让前 k 名全部来自同一条候选列（连续推进到表的最后一层），或者逐轮换一条候选列（每轮换一条胜者路径）。"""
    key = np.arange(128, dtype=np.int64) * (2**32 // 128) - 2**31
    bits = np.where(key < 0, key.astype(np.int32).view(np.uint32) ^ np.uint32(0x7fffffff), key.astype(np.uint32))
    result = []
    for column in range(8):
        for interleaved in (False, True):
            positions = (np.arange(k) + column) % k if interleaved else np.arange(k) * 8 + column
            remaining = np.setdiff1d(np.arange(128), positions)
            x = np.empty((rows, 128), np.uint32)
            for row in range(rows):
                x[row, remaining] = rng.permutation(bits[:-k])
                x[row, positions] = bits[-k:][::-1]
            result.append(x.view(np.float32))
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=int, nargs='+')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('/tmp/top-k-paper/transpose'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clock = asm.KernelClock(num_cores=1)
    k = args.k
    for rows in args.rows:
        assert 8 <= rows <= 128, rows
        saved: dict[str, object] = {}
        record = shell.measure(clock, shell.pallas_top_k(k, 1, carrier(rows, k), scratch_rows=layout(k)[1] + 96), (rows, 128), k, 1, transform=transform(rows, k, saved), label=f'transpose-{rows}-k{k}')
        # 计时用的输入没有并列；这里在同一个程序上再核对含特殊值的输入和定向输入。
        program = asm.load(saved['serialized'], saved['compiled'])
        cases = [x for _, x in compare.distributions(np.random.default_rng(9127), (rows, 128), 1, k, 8)] + directed(rows, k, np.random.default_rng(70200 + rows))
        for x in cases:
            shell.verify(program(jax.device_put(x)), x, k, 1)
        official = shell.measure(clock, shell.pallas_top_k(k, 1), (rows, 128), k, 1, label=f'official-{rows}-k{k}')
        result = {'rows': rows, 'k': k, 'transpose': record, 'official': official, 'bitwise_rows': rows * len(cases)}
        (args.output / (f'{rows}.json' if k == 8 else f'{rows}-k{k}.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'rows': rows, 'k': k, 'transpose': record['median'], 'official': official['median'], 'bitwise_rows': rows * len(cases)}), flush=True)

if __name__ == '__main__':
    main()
