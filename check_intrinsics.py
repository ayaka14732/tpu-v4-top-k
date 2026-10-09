"""逐个核对 asm_intrinsics 的四个 intrinsic：普通 Pallas kernel 加占位，改写后与 NumPy 参照比较。参数是输入的行数（8 的倍数）。"""
from pathlib import Path
import os
import sys
import chip
import jax
import jax.numpy as jnp
import numpy as np
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from tpuasm import tools as asm
import asm_intrinsics as ai

def build(rows: int, body) -> tuple:
    mesh = jax.make_mesh((1,), ('device',))

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array, p: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((rows, 128), jnp.int32),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            scratch_types=(pltpu.VMEM((rows, 128), jnp.int32), pltpu.VMEM((rows, 128), jnp.int32), pltpu.VMEM((rows, 128), jnp.int32), pltpu.VMEM((256, 128), jnp.int32), pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
            name='intrinsic',
        )
        def kernel(x_hbm: Ref, p_hbm: Ref, o_hbm: Ref, x_vmem: Ref, p_vmem: Ref, o_vmem: Ref, work: Ref, sem: Ref, sem2: Ref) -> None:
            first = pltpu.async_copy(x_hbm, x_vmem, sem)
            second = pltpu.async_copy(p_hbm, p_vmem, sem2)
            first.wait()
            second.wait()
            ai.scratch(work)
            o_vmem[...] = body(x_vmem[...], p_vmem[...])
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x, p)

    ai.SITES.clear()
    compiled = asm.compile(function, np.zeros((rows, 128), np.int32), np.zeros((rows, 128), np.int32), mesh=mesh)
    serialized, counts = ai.rewrite(compiled)
    return asm.load(serialized, compiled), counts

def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rng = np.random.default_rng(6501)
    x = rng.integers(-2**31, 2**31, (rows, 128), dtype=np.int64).astype(np.int32)
    tiles = x.reshape(rows // 8, 8, 128)
    sublane = np.arange(8)[None, :, None]
    # 每列一个随机排列，作为 gather 与 scatter 的目的行。
    permutation = np.argsort(rng.random((rows // 8, 8, 128)), axis=1)
    order = permutation.reshape(rows, 128).astype(np.int32)
    reverse = sum((7 - s) << (4 * s) for s in range(8))
    row = lambda shape: jax.lax.broadcasted_iota(jnp.int32, shape, 0) % 8
    scattered = np.zeros_like(tiles)
    np.put_along_axis(scattered, permutation, tiles, axis=1)
    half = np.where(permutation < 4, permutation, permutation + 4)
    kept = np.full_like(tiles, -7)
    for s in range(8):
        target = half[:, s, :]
        t, lane = np.nonzero(target < 8)
        kept[t, target[t, lane], lane] = tiles[t, s, lane]
    cases = {
        'fold': (lambda v, p: ai.fold(v), tiles.reshape(-1, 8, 16, 8).transpose(0, 3, 2, 1).reshape(rows, 128)),
        'fold twice': (lambda v, p: ai.fold(ai.fold(v)), x),
        'shuffle (reverse)': (lambda v, p: ai.sublane_shuffle(v, reverse), tiles[:, ::-1].reshape(rows, 128)),
        'gather': (lambda v, p: ai.sublane_gather(v, p - row(v.shape)), np.take_along_axis(tiles, permutation, axis=1).reshape(rows, 128)),
        'scatter': (lambda v, p: ai.sublane_scatter(v, p - row(v.shape)), scattered.reshape(rows, 128)),
        'scatter keeping half': (lambda v, p: ai.sublane_scatter(v, jnp.where(p < 4, p, p + 4) - row(v.shape), fill=-7), kept.reshape(rows, 128)),
        'fold, then column maximum by Pallas': (lambda v, p: jnp.max(ai.fold(v).reshape(rows // 8, 8, 128), axis=1, keepdims=True).repeat(8, axis=1).reshape(rows, 128), tiles.reshape(-1, 8, 16, 8).transpose(0, 3, 2, 1).reshape(-1, 8, 128).max(axis=1, keepdims=True).repeat(8, axis=1).reshape(rows, 128)),
    }
    for name, (body, expected) in cases.items():
        try:
            function, counts = build(rows, body)
            got = np.asarray(function(x, order))
            print(f'{name}: expansions {counts}, correct {bool((got == expected).all())}', flush=True)
        except Exception as error:
            print(f'{name}: FAILED {type(error).__name__}: {str(error).splitlines()[0][:300]}', flush=True)

if __name__ == '__main__':
    main()
