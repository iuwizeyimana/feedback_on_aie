#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.helpers.dialects.ext.scf import _for as range_

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE element_wise addition MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix e-wise addition design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="bf16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="bf16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    my_matadd(
        args.dev,
        args.M,
        args.N,
        args.dtype_in,
        args.dtype_out,
        args.trace_size,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matadd(
    dev, M, N, dtype_in_str, dtype_out_str, trace_size
):

    vectorized = True

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    A_sz = N * M
    B_sz = M 
    C_sz = N * M


    with mlir_mod_ctx() as ctx:

        C_sz_in_bytes = C_sz * np.dtype(dtype_out).itemsize

        if dev == "npu":
            dev_ty = AIEDevice.npu1_1col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            #ToDO -- fixme 
            a_ty = np.ndarray[(M,), np.dtype[dtype_in]]
            b_ty = np.ndarray[(M,), np.dtype[dtype_in]]
            c_ty = np.ndarray[(M,), np.dtype[dtype_out]]

            # AIE Core Function declarations
            func_type = "" if vectorized else "scalar_"
            add_func_name = (
                f"ewise_add_{func_type}{dtype_in_str}_{dtype_out_str}"
            )
            ewise_add = external_func(
                add_func_name,
                inputs=[a_ty, b_ty, c_ty],
            )

            store_func_name = (
                f"store_{func_type}{dtype_in_str}_{dtype_out_str}"
            )
            store = external_func(
                store_func_name,
                inputs=[c_ty, c_ty],
            )

            # Tile declarations
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile = tile(0, 2)

            # AIE-array data movement with object fifos
            # Input A
            inA = object_fifo("inA", shim_tile, mem_tile, 2, a_ty)
            memA = object_fifo(
                "memA",
                mem_tile,
                compute_tile,
                2,
                a_ty,
            )
            object_fifo_link(inA, memA)

            # Input B
            inB = object_fifo("inB", shim_tile, mem_tile, 2, b_ty)
            memB = object_fifo(
                "memB",
                mem_tile,
                compute_tile,
                2,
                b_ty,
            )
            #object_fifo_link(inB, memB)

            # Output C
            memC = object_fifo("memC", compute_tile, mem_tile, 2, c_ty)
            memC_cpy = object_fifo("memC_cpy", compute_tile, mem_tile, 2, c_ty)
            outC = object_fifo(
                "outC",
                mem_tile,
                shim_tile,
                2,
                c_ty,
            )
            object_fifo_link(memC, outC)

            object_fifo_link([inB, memC_cpy], memB, [0, 0])

            # Set up compute tiles
            @core(compute_tile, f"add.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(N) if N > 1 else range(1):
                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        elem_out_cp = memC_cpy.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                        ewise_add(elem_in_a, elem_in_b, elem_out)
                        store(elem_out, elem_out_cp)
                        memA.release(ObjectFifoPort.Consume, 1)
                        memB.release(ObjectFifoPort.Consume, 1)
                        memC.release(ObjectFifoPort.Produce, 1)
                        memC_cpy.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
            )
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=inA,
                    bd_id = 0,
                    mem=A,
                    sizes=[1, 1, N, M],
                    strides=[0, 0, M, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=inB,
                    bd_id = 1,
                    mem=B,
                    sizes=[1,1,1,M],
                    strides=[0,0,M,1],
                )
                npu_dma_memcpy_nd(
                    metadata=outC,
                    bd_id = 2,
                    mem=C,
                    sizes=[1,1,N,M],
                    strides=[0,0,M,1],
                )
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
