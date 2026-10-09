"""在 JAX primitive 的 Mosaic lowering rule 里直接生成带向量操作数或结果的 llo.* op，看 Mosaic 是否接受。\n\n用法：llo_op_probe.py result_only | operand_and_result | slane_shuffle | transpose <mode>\n"""
from pathlib import Path
import os
import sys
import chip
import jax
import jax.numpy as jnp
import numpy as np
from jax import Ref
from jax._src import core as jax_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith

KIND, MODE = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'b32')
p = jax_core.Primitive('front_probe')
p.def_abstract_eval(lambda x: x)

def _rule(ctx, x):
    ir.Context.current.allow_unregistered_dialects = True
    if KIND == 'result_only':
        return ir.Operation.create('llo.vlaneseq', results=[x.type]).result
    if KIND == 'operand_and_result':
        return ir.Operation.create('llo.sublane_reverse', operands=[x], results=[x.type]).result
    if KIND == 'transpose':
        i32 = ir.IntegerType.get_signless(32)
        const = lambda value: ir.IntegerAttr.get(i32, value)
        width = arith.constant(i32, const(8))
        ir.Operation.create('llo.vxpose', operands=[x, width], attributes={'chunk_id': const(0), 'mode': ir.Attribute.parse(f'#llo.vxpose_mode<{MODE}>'), 'number_of_chunks': const(1), 'xlu_id': const(0)})
        return ir.Operation.create('llo.vxpose.result', results=[x.type], attributes={'xlu_id': const(0)}).result
    if KIND == 'slane_shuffle':
        return ir.Operation.create('llo.slane.shuffle', operands=[x], results=[x.type], attributes={'indices': ir.DenseI32ArrayAttr.get([7, 6, 5, 4, 3, 2, 1, 0])}).result

mosaic_lowering.lowering_rules[tpu_core.CoreType.TC][p] = _rule

@jax.jit
def f(x):
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
        scratch_types=(pltpu.VMEM((8, 128), jnp.int32), pltpu.VMEM((8, 128), jnp.int32), pltpu.SemaphoreType.DMA),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
        name='front_probe',
    )
    def kernel(x_hbm: Ref, o_hbm: Ref, xv: Ref, ov: Ref, sem: Ref) -> None:
        pltpu.async_copy(x_hbm, xv, sem).wait()
        ov[...] = p.bind(xv[...])
        pltpu.async_copy(ov, o_hbm, sem).wait()
    return kernel(x)

x = np.arange(1024, dtype=np.int32).reshape(8, 128)
try:
    y = np.asarray(f(x)); print(KIND, 'ok; column 0:', y[:, 0].tolist(), 'row 0[:4]:', y[0, :4].tolist())
except Exception as e:
    print(KIND, MODE if KIND == 'transpose' else '', 'FAILED:', ' | '.join(line for line in str(e).splitlines()[:2] if line)[:400])
