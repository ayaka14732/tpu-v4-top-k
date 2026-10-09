"""TPU dialect 里现成的 op 在 TPU v4 的 TensorCore 上能走多远：在 primitive 的 lowering rule 里生成 tpu.* op（不是 llo.* op），不打任何补丁，看编译结果。

用法：tpu_op_probe.py <kind> [补丁名...]，kind 见 RULES。成功时打印 kernel 里与这个 op 有关的指令和数值是否正确，失败时打印编译器的报错。
"""
from pathlib import Path
import os
import re
import sys
import chip
import jax
import jax.numpy as jnp
import numpy as np
from jax import Ref
from jax._src import core as jax_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mosaic.python import tpu
from tpuasm import tools as asm
import libtpu_patch

KIND = sys.argv[1]
PATCHES = sys.argv[2:]
if 'check_only' in PATCHES:
    # 只去掉 tpu.dynamic_gather 的 lower-to-LLO 里“这一代 TPU 是否支持沿子通道 gather”的判断。
    libtpu_patch.write(0xf16ab3c, bytes.fromhex('0f8495020000'), b'\x90' * 6)
if 'allow_sublane_gather' in PATCHES:
    libtpu_patch.allow_sublane_gather()
for name, opcode in (('set_iar_sublane', 4), ('set_iar_lane', 2)):
    if name in PATCHES:
        libtpu_patch.GATHER[0].use_set_iar(opcode)
REVERSE = [7, 6, 5, 4, 3, 2, 1, 0]
# 对值的 op：(x, index) -> y。
value_p = jax_core.Primitive('tpu_value_probe')
value_p.def_abstract_eval(lambda x, index: x)
# 对 Ref 的 op：(ref, index) -> y，读 ref 的前 8 行。
ref_p = jax_core.Primitive('tpu_ref_probe')
ref_p.is_effectful = lambda params: True
ref_p.def_effectful_abstract_eval(lambda ref, index: (jax_core.ShapedArray((8, 128), ref.dtype), {state_types.ReadEffect(0)}))

def _zero_index() -> ir.Value:
    return arith.constant(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0))

def _value_rule(ctx: mosaic_lowering.LoweringRuleContext, x: ir.Value, index: ir.Value) -> ir.Value:
    if KIND == 'gather_sublanes':
        return tpu.dynamic_gather(x, index, [0])
    if KIND == 'gather_lanes':
        return tpu.dynamic_gather(x, index, [1])
    if KIND == 'sublane_shuffle':
        return tpu.sublane_shuffle(x, x, REVERSE)
    if KIND == 'static_gather':
        return tpu.gather(x.type, x, REVERSE, 0)
    raise ValueError(KIND)

def _ref_rule(ctx: mosaic_lowering.LoweringRuleContext, ref: ir.Value, index: ir.Value) -> ir.Value:
    result = ir.VectorType.get((8, 128), ir.IntegerType.get_signless(32))
    if KIND == 'shuffled_load':
        return tpu.shuffled_load(result, ref, [_zero_index(), _zero_index()], [True] * 8, REVERSE)
    if KIND == 'load_idx':
        # 行号来自 index，列号是通道自己。
        lanes = tpu.iota(result, [1])
        return tpu.vector_load_idx(result, ref, [index, lanes])
    raise ValueError(KIND)

mosaic_lowering.lowering_rules[tpu_core.CoreType.TC][value_p] = _value_rule
mosaic_lowering.lowering_rules[tpu_core.CoreType.TC][ref_p] = _ref_rule
RULES = {
    'gather_sublanes': lambda x, p: np.take_along_axis(x, p, axis=0),
    'gather_lanes': lambda x, p: np.take_along_axis(x, p * 16 % 128, axis=1),
    'sublane_shuffle': lambda x, p: x[::-1],
    'static_gather': lambda x, p: x[::-1],
    'shuffled_load': lambda x, p: x[::-1],
    'load_idx': lambda x, p: np.take_along_axis(x, p, axis=0),
}

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array, index: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), jnp.int32),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            scratch_types=(pltpu.VMEM((8, 128), jnp.int32), pltpu.VMEM((8, 128), jnp.int32), pltpu.VMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
            name='tpu_op_probe',
        )
        def kernel(x_hbm: Ref, p_hbm: Ref, o_hbm: Ref, x_vmem: Ref, p_vmem: Ref, o_vmem: Ref, sem: Ref, sem2: Ref) -> None:
            first = pltpu.async_copy(x_hbm, x_vmem, sem)
            second = pltpu.async_copy(p_hbm, p_vmem, sem2)
            first.wait()
            second.wait()
            index = p_vmem[...] * 16 % 128 if KIND == 'gather_lanes' else p_vmem[...]
            if KIND in ('shuffled_load', 'load_idx'):
                o_vmem[...] = ref_p.bind(x_vmem, index)
            else:
                o_vmem[...] = value_p.bind(x_vmem[...], index)
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x, index)

    rng = np.random.default_rng(6502)
    x = rng.integers(-2**31, 2**31, (8, 128), dtype=np.int64).astype(np.int32)
    order = np.argsort(rng.random((8, 128)), axis=0).astype(np.int32)
    print(f'$ tpu_op_probe.py {KIND}', *PATCHES)
    try:
        compiled = asm.compile(function, x, order, mesh=mesh)
    except Exception as error:
        print('FAILED:', ' | '.join(line for line in str(error).splitlines()[:3] if line)[:500])
        return
    listing = asm.kernel_listing(compiled, pallas_only=True)
    interesting = sorted({re.sub(r'v\d+', 'v', match) for match in re.findall(r'\b(?:vld|vst|vx\d|vr\d|va\d|misc): ((?:vld|vst|vsetiar|vperm|vxpose|vsxpose|vrot|vshfl|vpop|vsort)[\w.]*)', listing)})
    got = np.asarray(asm.load(asm.serialize(compiled), compiled)(x, order))
    expected = RULES[KIND]
    print('compiled; instructions:', ', '.join(interesting), '| correct:', bool((got == expected(x, order)).all()))
    if os.environ.get('SHOW'):
        print('index column 5:', order[:, 5].tolist(), 'x column 5:', x[:, 5].tolist(), 'got column 5:', got[:, 5].tolist())

if __name__ == '__main__':
    main()
