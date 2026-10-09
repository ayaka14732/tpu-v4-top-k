"""TPU v4 f32 top-k：保留完整位型顺序，签名与 Pallas 的 _top_k_impl 相同，可以直接替换它。

phased 把完整键分成两段，在同一条 XLU 链上分时选取，phased_gather 是它在行数多时的变体；rank 用整数秩计数；
columns/vertical 用比较网络准备候选列，vertical_merge 再用归并网络合并候选列。fast 是按形状和 k 选择写法的通用分派。仅 rank 与 vertical 保证同位型时下标稳定。
"""
import math

import jax
from jax import lax
from jax._src.pallas.mosaic import lowering
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

official = lowering._top_k_impl
# 沿通道、单个 lane tile 时，行数不少于此值改用 phased_gather。
GATHER_ROWS = 64
def use_transposed(rows: int, k: int) -> bool:
    """f32[rows,128] 沿最后一维取前 k 名时是否改用 transposed。界线来自 k 为 2、4、8、16、32 的实测：k 不小于 16 时从 16 行起就划算，k = 8 时要到 64 行，k 更小时不划算。"""
    return 16 <= rows <= 128 and (16 <= k <= 32 or 8 <= k < 16 and rows >= 64)

INT_MIN = -(1 << 31)

def comparators(size: int) -> list[tuple[int, int]]:
    """Batcher 奇偶归并网络，按降序比较；size 是 2 的幂。"""
    result: list[tuple[int, int]] = []

    def merge(start: int, length: int, stride: int) -> None:
        twice = 2 * stride
        if twice < length:
            merge(start, length, twice)
            merge(start + stride, length, twice)
            result.extend((i, i + stride) for i in range(start + stride, start + length - stride, twice))
        else:
            result.append((start, start + stride))

    def sort(start: int, length: int) -> None:
        if length > 1:
            sort(start, length // 2)
            sort(start + length // 2, length // 2)
            merge(start, length, 1)

    sort(0, size)
    return result

def schedule(size: int, k: int) -> list[tuple[int, int, bool, bool]]:
    """反向传播前 k 个输出的依赖，删除不需要的比较和选择。"""
    live = set(range(k))
    result = []
    for a, b in reversed(comparators(size)):
        keep_a, keep_b = a in live, b in live
        if keep_a or keep_b:
            result.append((a, b, keep_a, keep_b))
            live.update((a, b))
    return result[::-1]

def sort_columns(groups: list[tuple[jax.Array, ...]], k: int) -> list[tuple[jax.Array, ...]]:
    count = len(groups)
    size = 1 << (count - 1).bit_length()
    # None 是比有效输入更小的填充；在 tracing 时消去它参与的比较。
    wires = groups + [None] * (size - count)
    for a, b, keep_a, keep_b in schedule(size, k):
        left, right = wires[a], wires[b]
        if right is None:
            continue
        if left is None:
            wires[a], wires[b] = right, None
            continue
        swap = right[0] > left[0]
        if keep_a:
            wires[a] = tuple(jnp.where(swap, y, x) for x, y in zip(left, right))
        if keep_b:
            wires[b] = tuple(jnp.where(swap, x, y) for x, y in zip(left, right))
    return wires[:k]

def to_key(x: jax.Array) -> jax.Array:
    """f32 的可逆 total-order 键；保留 NaN 符号、payload 与有符号零。"""
    bits = lax.bitcast_convert_type(x, jnp.int32)
    return jnp.where(bits < 0, bits ^ 0x7fffffff, bits)

def from_key(key: jax.Array) -> jax.Array:
    return lax.bitcast_convert_type(jnp.where(key < 0, key ^ 0x7fffffff, key), jnp.float32)

def encode_half(key: jax.Array) -> jax.Array:
    """低 31 位严格保序映射到正常有限 f32，分段号由 key 的符号单独保存。"""
    payload = key & 0x3fffffff
    bits = jnp.where((key & 0x40000000) != 0, payload + 0x00800000, (0x407fffff - payload) | jnp.int32(INT_MIN))
    return lax.bitcast_convert_type(bits, jnp.float32)

def decode_half(value: jax.Array, high: jax.Array) -> jax.Array:
    bits = lax.bitcast_convert_type(value, jnp.int32)
    magnitude = bits & 0x7fffffff
    lower = jnp.where(bits < 0, 0x407fffff - magnitude, (magnitude - 0x00800000) | 0x40000000)
    return from_key(lower | jnp.where(high, 0, jnp.int32(INT_MIN)))

def masked_word(word: jax.Array, mask: jax.Array, axis: int, *, keepdims: bool = False) -> jax.Array:
    """从每行唯一选中的位置取完整 32 位字；两个 16 位字段都能精确归约。"""
    word = lax.bitcast_convert_type(word, jnp.uint32)
    lo = jnp.max(jnp.where(mask, (word & 65535).astype(jnp.float32), 0.0), axis=axis, keepdims=keepdims).astype(jnp.uint32)
    hi = jnp.max(jnp.where(mask, (word >> 16).astype(jnp.float32), 0.0), axis=axis, keepdims=keepdims).astype(jnp.uint32)
    return lax.bitcast_convert_type(lo | (hi << 16), jnp.int32)

def phased(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """先选非负键，再激活负键；第一轮同时准备全负键行的结果。"""
    axis %= operand.ndim
    key = to_key(operand)
    proxy = encode_half(key)
    high = key >= 0
    m = jnp.sum(high.astype(jnp.float32), axis=axis)
    lower = jnp.where(high, -jnp.inf, proxy)
    current = jnp.where(high, proxy, -jnp.inf)
    position = lax.broadcasted_iota(jnp.int32, operand.shape, axis)
    when = jnp.where(high, -1.0, jnp.expand_dims(jnp.maximum(m, 1.0), axis))
    values, indices = [], []
    for j in range(k):
        index = lax.argmax(current, axis=axis, index_dtype=jnp.int32)
        selected = jnp.max(current, axis=axis)
        if j == 0:
            low_index = lax.argmax(lower, axis=axis, index_dtype=jnp.int32)
            low_value = jnp.max(lower, axis=axis)
            index = jnp.where(m == 0, low_index, index)
            selected = jnp.where(m == 0, low_value, selected)
        values.append(decode_half(selected, m > j))
        hit = position == jnp.expand_dims(index, axis)
        indices.append(index if carried_idx is None else masked_word(carried_idx, hit, axis))
        current = jnp.where(hit, -jnp.inf, jnp.where(when == j + 1, proxy, current))
    return jnp.stack(values, axis=axis), jnp.stack(indices, axis=axis)

def phased_gather(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """与 phased 相同的两段分时链，但每轮只求下标，归约从每轮两条减到一条；k 个值在结束后用一次沿通道的 gather 从原输入取回，取回的就是原位型，不需要解码。"""
    axis %= operand.ndim
    assert carried_idx is None and axis == operand.ndim - 1 and operand.shape[axis] <= 128
    key = to_key(operand)
    proxy = encode_half(key)
    high = key >= 0
    m = jnp.sum(high.astype(jnp.float32), axis=axis)
    current = jnp.where(high, proxy, -jnp.inf)
    position = lax.broadcasted_iota(jnp.int32, operand.shape, axis)
    when = jnp.where(high, -1.0, jnp.expand_dims(jnp.maximum(m, 1.0), axis))
    indices = []
    for j in range(k):
        index = lax.argmax(current, axis=axis, index_dtype=jnp.int32)
        if j == 0:
            index = jnp.where(m == 0, lax.argmax(jnp.where(high, -jnp.inf, proxy), axis=axis, index_dtype=jnp.int32), index)
        indices.append(index)
        current = jnp.where(position == jnp.expand_dims(index, axis), -jnp.inf, jnp.where(when == j + 1, proxy, current))
    indices = jnp.stack(indices, axis=axis)
    return jnp.take_along_axis(operand, indices, axis=axis), indices

def rank(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """稳定秩计数；INT_MIN 的下溢由初始秩修正，不改每次比较。"""
    axis %= operand.ndim
    n = operand.shape[axis]
    position = lax.broadcasted_iota(jnp.int32, operand.shape, axis)
    key = to_key(operand)
    lower = key - 1
    count = jnp.where(key == INT_MIN, position, 0)
    for distance in range(1, n):
        other = jnp.roll(key, distance, axis=axis)
        count += jnp.where(other > jnp.where(position >= distance, lower, key), 1, 0)
    local = jnp.stack([jnp.max(jnp.where(count == r, position.astype(jnp.float32), -1.0), axis=axis) for r in range(k)], axis=axis).astype(jnp.int32)
    if axis == operand.ndim - 1 and n <= 128:
        values = jnp.take_along_axis(operand, local, axis=axis)
    else:
        values = jnp.stack([lax.bitcast_convert_type(masked_word(operand, count == r, axis), jnp.float32) for r in range(k)], axis=axis)
    if carried_idx is not None:
        local = jnp.stack([masked_word(carried_idx, count == r, axis) for r in range(k)], axis=axis)
    return values, local

def columns(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """按完整整数键排序 lane tile，列首按当前分段编码后交给 XLU。"""
    axis %= operand.ndim
    if axis != operand.ndim - 1 or operand.shape[-1] <= 128:
        return phased(operand, k=k, axis=axis, carried_idx=carried_idx)
    n = operand.shape[-1]
    count = -(-n // 128)
    lead = operand.shape[:-1]
    pads = [(0, 0)] * len(lead) + [(0, count * 128 - n)]
    key = jnp.pad(to_key(operand), pads, constant_values=INT_MIN)
    position = jnp.pad(lax.broadcasted_iota(jnp.int32, operand.shape, axis), pads, constant_values=-1)
    lane = lax.broadcasted_iota(jnp.int32, (*lead, 128), axis)
    groups = [(key[..., t * 128:(t + 1) * 128], position[..., t * 128:(t + 1) * 128]) for t in range(count)]
    if carried_idx is not None:
        carried = jnp.pad(carried_idx, pads)
        groups = [(*group, carried[..., t * 128:(t + 1) * 128]) for t, group in enumerate(groups)]
    # 所有填充都在输入之后；比较网络在相等时不交换，故 INT_MIN 实值仍先于填充。
    column = sort_columns(groups, min(count, k))
    m = jnp.sum((to_key(operand) >= 0).astype(jnp.float32), axis=axis, keepdims=True)
    bottom = (jnp.int32(INT_MIN), jnp.int32(-1)) + (() if carried_idx is None else (jnp.int32(0),))
    values, indices = [], []
    for j in range(k):
        head, position, *extra = column[0]
        proxy = encode_half(head)
        active = (position >= 0) & ((head >= 0) == (m > j))
        value = jnp.where(active, proxy, -jnp.inf)
        index = jnp.expand_dims(lax.argmax(value, axis=axis, index_dtype=jnp.int32), axis)
        best = jnp.max(value, axis=axis, keepdims=True)
        hit = lane == index
        values.append(decode_half(best, m > j))
        indices.append(masked_word(extra[0], hit, axis, keepdims=True) if extra else jnp.max(jnp.where(hit, position.astype(jnp.float32), -1.0), axis=axis, keepdims=True).astype(jnp.int32))
        for level in range(min(len(column), k - j - 1)):
            below = column[level + 1] if level + 1 < len(column) else bottom
            column[level] = tuple(jnp.where(hit, b, a) for a, b in zip(column[level], below))
    return jnp.concatenate(values, axis=axis), jnp.concatenate(indices, axis=axis)

def vertical(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """整数淘汰树；删除时同步使位置无效，INT_MIN 输入不会被当成填充。"""
    axis %= operand.ndim
    n = operand.shape[axis]
    key = to_key(operand)
    position = lax.broadcasted_iota(jnp.int32, operand.shape, axis)
    values, indices = [], []
    for _ in range(k):
        best = jnp.max(key, axis=axis, keepdims=True)
        index = jnp.min(jnp.where(key == best, position, n), axis=axis, keepdims=True)
        hit = position == index
        values.append(from_key(best))
        indices.append(index if carried_idx is None else masked_word(carried_idx, hit, axis, keepdims=True))
        key = jnp.where(hit, INT_MIN, key)
        position = jnp.where(hit, n, position)
    return jnp.concatenate(values, axis=axis), jnp.concatenate(indices, axis=axis)

def fast(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """仅对 f32 提供完整位型语义，按已有开销模型选择候选。"""
    if operand.dtype != jnp.float32:
        return official(operand, k=k, axis=axis, carried_idx=carried_idx)
    axis %= operand.ndim
    n = operand.shape[axis]
    if axis == operand.ndim - 1:
        rows = -(-(math.prod(operand.shape) // n) // 8)
        if n == 128 and operand.ndim == 2 and carried_idx is None and use_transposed(operand.shape[0], k):
            function = transposed
        elif n <= 128 and rows <= 8 and 90 + rows * (670 + 4.4 * k) < 83 * k:
            function = rank
        elif 1 < n <= 128 and carried_idx is None and 8 * rows >= GATHER_ROWS:
            function = phased_gather
        else:
            function = phased if n <= 128 else columns
    else:
        # 实测 16 至 64 行时子通道归并不慢于逐个弹出，128 行起寄存器溢出使它变慢。
        # 归并同时保留的候选更多，每层超过一个 TC VREG 时溢出会用尽 TC VMEM，只在每层恰为一个 TC VREG 时选用。
        function = rank if n <= 8 else vertical_merge if n <= 64 and math.prod(operand.shape) // n <= 128 else vertical_columns
    return function(operand, k=k, axis=axis, carried_idx=carried_idx)

def vertical_columns(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """先排序各 sublane tile 的候选层；无效位置为 -1，键可取全部 int32 位型。"""
    axis %= operand.ndim
    if axis != operand.ndim - 2:
        return vertical(operand, k=k, axis=axis, carried_idx=carried_idx)
    n = operand.shape[axis]
    count = -(-n // 8)
    pads = [(0, 0)] * operand.ndim
    pads[axis] = (0, count * 8 - n)
    key = jnp.pad(to_key(operand), pads, constant_values=INT_MIN)
    position = jnp.pad(lax.broadcasted_iota(jnp.int32, operand.shape, axis), pads, constant_values=-1)
    arrays = (key, position) if carried_idx is None else (key, position, jnp.pad(carried_idx, pads))
    groups = [tuple(lax.slice_in_dim(x, i * 8, (i + 1) * 8, axis=axis) for x in arrays) for i in range(count)]
    column = sort_columns(groups, min(count, k))
    bottom = (jnp.int32(INT_MIN), jnp.int32(-1)) + (() if carried_idx is None else (jnp.int32(0),))
    values, indices = [], []
    for j in range(k):
        key, position, *extra = column[0]
        best = jnp.max(key, axis=axis, keepdims=True)
        index = jnp.max(jnp.where(key == best, position, -1), axis=axis, keepdims=True)
        hit = position == index
        values.append(from_key(best))
        indices.append(index if carried_idx is None else masked_word(extra[0], hit, axis, keepdims=True))
        for level in range(min(len(column), k - j - 1)):
            below = column[level + 1] if level + 1 < len(column) else bottom
            column[level] = tuple(jnp.where(hit, b, a) for a, b in zip(column[level], below))
    return jnp.concatenate(values, axis=axis), jnp.concatenate(indices, axis=axis)

def merge_schedule(half: int, k: int) -> list[tuple[int, int, bool, bool]]:
    """两条长度 half 的降序列（线 0 起与线 half 起）的奇偶归并，只保留前 k 个输出所需的比较。"""
    network: list[tuple[int, int]] = []

    def merge(start: int, length: int, stride: int) -> None:
        twice = 2 * stride
        if twice < length:
            merge(start, length, twice)
            merge(start + stride, length, twice)
            network.extend((i, i + stride) for i in range(start + stride, start + length - stride, twice))
        else:
            network.append((start, start + stride))

    merge(0, 2 * half, 1)
    live = set(range(k))
    result = []
    for a, b in reversed(network):
        keep_a, keep_b = a in live, b in live
        if keep_a or keep_b:
            result.append((a, b, keep_a, keep_b))
            live.update((a, b))
    return result[::-1]

def vertical_merge(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """列排序后不逐个弹出：把八条候选列沿子通道归并三次，每个子通道都得到完整结果，输出取第 0 个。"""
    axis %= operand.ndim
    n = operand.shape[axis]
    if axis != operand.ndim - 2 or n % 8 or n < 16:
        return vertical_columns(operand, k=k, axis=axis, carried_idx=carried_idx)
    count = n // 8
    key = to_key(operand)
    position = lax.broadcasted_iota(jnp.int32, operand.shape, axis)
    arrays = (key, position) if carried_idx is None else (key, position, carried_idx)
    groups = [tuple(lax.slice_in_dim(x, i * 8, (i + 1) * 8, axis=axis) for x in arrays) for i in range(count)]
    column = sort_columns(groups, min(count, k))
    for shift in (4, 6, 7):
        # 子通道 s 的原列与来自子通道 s - shift 的列归并；三次之后每个子通道都覆盖全部八条列。
        # 硬件的子通道旋转每条指令只移一行，方向相当于 roll 7；位移 4、6、7 分别是 4、2、1 条指令。
        half = 1 << (len(column) - 1).bit_length()
        fill = [None] * (half - len(column))
        wires = [tuple(pltpu.roll(x, shift, axis) for x in group) for group in column] + fill + column + fill
        keep = min(2 * len(column), k)
        for a, b, keep_a, keep_b in merge_schedule(half, keep):
            left, right = wires[a], wires[b]
            if right is None:
                continue
            if left is None:
                wires[a], wires[b] = right, None
                continue
            swap = right[0] > left[0]
            if keep_a:
                wires[a] = tuple(jnp.where(swap, y, x) for x, y in zip(left, right))
            if keep_b:
                wires[b] = tuple(jnp.where(swap, x, y) for x, y in zip(left, right))
        column = wires[:keep]
    # 各子通道在同键时的次序可能不同，只取第 0 个子通道：逐个接入并旋转一行，第 r 名落到第 r % 8 行。
    sublane = lax.broadcasted_iota(jnp.int32, column[0][0].shape, axis)
    tiles = []
    for start in range(0, k, 8):
        tile = column[start]
        for r in range(start, start + 8):
            tile = tuple(pltpu.roll(jnp.where(sublane == 0, y, x), 7, axis) for x, y in zip(tile, column[min(r, k - 1)]))
        tiles.append(tuple(lax.slice_in_dim(x, 0, min(8, k - start), axis=axis) for x in tile))
    values = from_key(jnp.concatenate([tile[0] for tile in tiles], axis=axis))
    indices = jnp.concatenate([tile[-1 if carried_idx is not None else 1] for tile in tiles], axis=axis)
    return values, indices

def transposed(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """把横向问题转成纵向问题：f32[R,128] 整体转置，每条输入行落在一个通道上，沿子通道用 vertical_merge 取前 k 名，再把两个结果转回来。中间不经过 XLU，各行共用同一批逐元素指令。"""
    rows, n = operand.shape
    assert carried_idx is None and axis % 2 == 1 and n == 128 and rows <= 128
    values, indices = vertical_merge(jnp.pad(operand.T, ((0, 0), (0, 128 - rows))), k=k, axis=0)
    return values.T[:rows], indices.T[:rows]

def pairs(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """非负键段的成对提取；未通过唯一性与数量审计时回退到完整位型链。"""
    if operand.shape != (8, 128) or operand.dtype != jnp.float32 or k != 8 or axis % 2 != 1 or carried_idx is not None:
        return fast(operand, k=k, axis=axis, carried_idx=carried_idx)
    position = lax.broadcasted_iota(jnp.int32, operand.shape, 1)
    key = to_key(operand)
    original = jnp.where(key >= 0, encode_half(key), -jnp.inf)
    current = original
    values, indices = [], []
    bad = jnp.zeros((8, 1), jnp.bool_)
    for pair in range(4):
        if pair == 3:
            audit = jnp.sum(jnp.where(original != -jnp.inf, jnp.where(current == -jnp.inf, 256.0, 1.0), 0.0), axis=1, keepdims=True)
            bad |= (audit < 1538) | (audit >= 1792)
            first_index = jnp.expand_dims(lax.argmax(current, axis=1, index_dtype=jnp.int32), 1)
            first = jnp.max(current, axis=1, keepdims=True)
            current = jnp.where(position == first_index, -jnp.inf, current)
            second_index = jnp.expand_dims(lax.argmax(current, axis=1, index_dtype=jnp.int32), 1)
            second = jnp.max(current, axis=1, keepdims=True)
        else:
            halves = []
            for bit in range(7):
                zero = jnp.max(jnp.where((position & (1 << bit)) == 0, current, -jnp.inf), axis=1, keepdims=True)
                one = jnp.max(jnp.where((position & (1 << bit)) != 0, current, -jnp.inf), axis=1, keepdims=True)
                halves.append((zero, one))
            first = jnp.maximum(*halves[0])
            runners = [jnp.minimum(a, b) for a, b in halves]
            second = runners[0]
            for runner in runners[1:]:
                second = jnp.maximum(second, runner)
            first_index = jnp.zeros((8, 1), jnp.int32)
            difference = jnp.zeros((8, 1), jnp.int32)
            for bit, ((a, b), runner) in enumerate(zip(halves, runners)):
                first_index |= jnp.where(b > a, 1 << bit, 0)
                difference |= jnp.where(runner == second, 1 << bit, 0)
            second_index = first_index ^ difference
            bad |= first == second
            current = jnp.where((current == first) | (current == second), -jnp.inf, current)
        values.extend((first, second))
        indices.extend((first_index, second_index))
    quick = decode_half(jnp.concatenate(values, axis=1), jnp.array(True)), jnp.concatenate(indices, axis=1)
    # 返回值分支无法用 pl.when 表达；保留 lax.cond。
    return lax.cond(jnp.sum(bad.astype(jnp.float32), axis=0)[0] > 0, lambda x: phased(x, k=8, axis=1), lambda _: quick, operand)
