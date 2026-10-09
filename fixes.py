"""只修正“已选位置被重新选中”、不改变官方实现其余部分的三种写法，签名与 Pallas 的 _top_k_impl 相同。它们不保证完整位型的顺序。"""
import jax
from jax import lax
import jax.numpy as jnp

def mask(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """最朴素的防御性掩码：已选位置记在单独的布尔掩码中，每轮先求未选位置的最大值，再在等于它的未选位置中取最小的下标。两次归约都在链上。"""
    del carried_idx
    axis = axis % operand.ndim
    position = lax.broadcasted_iota(jnp.int32, operand.shape, axis).astype(operand.dtype)
    taken = jnp.zeros(operand.shape, jnp.bool_)
    vals, idxs = [], []
    for _ in range(k):
        best = jnp.max(jnp.where(taken, -jnp.inf, operand), axis=axis, keepdims=True)
        index = jnp.min(jnp.where(~taken & (operand == best), position, float(operand.shape[axis])), axis=axis, keepdims=True)
        taken = taken | (position == index)
        vals.append(jnp.squeeze(best, axis))
        idxs.append(jnp.squeeze(index, axis).astype(jnp.int32))
    return jnp.stack(vals, axis=axis), jnp.stack(idxs, axis=axis)

def lift(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """抬高：数出每条中不等于 -inf 的元素个数 m，第 m 轮起把剩下的 -inf 抬到最小有限值。只多一次不在链上的归约。"""
    axis = axis % operand.ndim
    iota = lax.broadcasted_iota(jnp.int32, operand.shape, axis)
    neg_inf = jnp.array(-jnp.inf, dtype=operand.dtype)
    lowest = jnp.finfo(operand.dtype).min
    is_neg_inf = operand == neg_inf
    m = jnp.sum(jnp.where(is_neg_inf, 0.0, 1.0).astype(jnp.float32), axis=axis, keepdims=True)
    lift_round = jnp.where(is_neg_inf, jnp.maximum(m, 1.0), -1.0)
    vals, idxs = [], []
    curr = operand
    for j in range(k):
        idx = lax.argmax(curr, axis=axis, index_dtype=jnp.int32)
        if j == 0:
            idx = jnp.minimum(idx, operand.shape[axis] - 1)
        val = jnp.where(jnp.squeeze(m, axis) <= j, neg_inf, jnp.max(curr, axis=axis))
        hit = iota == jnp.expand_dims(idx, axis)
        idxs.append(idx if carried_idx is None else jnp.max(jnp.where(hit, carried_idx.astype(jnp.int32), -1), axis=axis))
        vals.append(val)
        curr = jnp.where(hit, neg_inf, jnp.where(lift_round == j + 1, lowest, curr))
    return jnp.stack(vals, axis=axis), jnp.stack(idxs, axis=axis)

def suffix(operand: jax.Array, *, k: int, axis: int = -1, carried_idx: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """把修正移出链：官方的链原样执行，原有的 -inf 在输出中的名次是 n - 1 减去它右侧 -inf 的个数，这个后缀和由 MXU 精确算出。只支持 f32[8,128] 沿最后一维。"""
    assert operand.shape == (8, 128) and operand.dtype == jnp.float32 and axis % 2 == 1 and carried_idx is None
    lane = lax.broadcasted_iota(jnp.int32, operand.shape, 1)
    missing = operand == -jnp.inf
    row = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
    column = lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    triangular = (row > column).astype(jnp.bfloat16)
    count = lax.dot_general(missing.astype(jnp.bfloat16), triangular, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    current = operand
    values, indices = [], []
    for j in range(k):
        index = jnp.expand_dims(lax.argmax(current, axis=1, index_dtype=jnp.int32), 1)
        value = jnp.max(current, axis=1, keepdims=True)
        repair = jnp.max(jnp.where(missing & (count == 127 - j), lane.astype(jnp.float32), -1.0), axis=1, keepdims=True).astype(jnp.int32)
        indices.append(jnp.where(repair >= 0, repair, index))
        values.append(value)
        current = jnp.where(lane == index, -jnp.inf, current)
    return jnp.concatenate(values, axis=1), jnp.concatenate(indices, axis=1)
