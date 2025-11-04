srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

all: build/final.xclbin

VPATH := ${srcdir}/kernels/

targetname ?= single_core
NPU2 ?= 1
devicename ?= $(if $(filter 1,$(NPU2)), npu2, npu)

# final GEMM dims
M ?= 8
N ?= 8 

# Microkernel dims
m ?= 8
n ?= 8

use_linear_buf_alloc ?= 1
buffer_aloc_flag := $(if $(filter 1,$(use_linear_buf_alloc)),basic-sequential,bank-aware)

dtype_in ?= bf16
dtype_out ?= bf16
trace_size ?= 65536
runargs ?= -v 2 --warmup 1 --iters 1
aie_py_src ?= $(targetname).py

kernels := add 

KERNEL_CC := ${PEANO_INSTALL_DIR}/bin/clang++
KERNEL_CFLAGS := ${PEANOWRAP2P_FLAGS}

aiecc_peano_flags=--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}

KERNEL_DEFINES_E := -DDIM_M=${m}

# --------------- dtype mapping for host build ------------------------
dtype_acc_cpp := int16_t
ifeq ($(dtype_in),bf16)
	dtype_in_cpp=std::bfloat16_t
endif
ifeq ($(dtype_out),bf16)
	dtype_out_cpp=std::bfloat16_t
	dtype_acc_cpp=float
endif
ifeq ($(dtype_in),i16)
	dtype_in_cpp=int16_t
endif
ifeq ($(dtype_out),i16)
	dtype_out_cpp=int16_t
	dtype_acc_cpp=int16_t
endif
ifeq ($(dtype_out),i32)
	dtype_out_cpp=int32_t
	dtype_acc_cpp=int32_t
endif
ifeq ($(dtype_out),f32)
	dtype_out_cpp=float
	dtype_acc_cpp=float
endif
ifeq ($(dtype_in),i8)
	dtype_in_cpp=int8_t
endif
ifeq ($(dtype_out),i8)
	dtype_out_cpp=int8_t
	dtype_acc_cpp=int8_t
endif



# -------- aie args ----------------
aieargs += --dev ${devicename} -M ${M} -N ${N} 
aieargs += --dtype_in ${dtype_in} --dtype_out ${dtype_out}

target_suffix ?= ${M}x${N}
mlir_target ?= build/aie.mlir
trace_mlir_target ?= build/aie_trace.mlir
xclbin_target ?= build/final.xclbin
trace_xclbin_target ?= build/trace.xclbin
insts_target ?= build/insts.bin

host_out := ${targetname}.exe
powershell ?=
getwslpath ?= 

.PHONY: all run trace parse_trace clean clean_trace 

all: ${xclbin_target} ${host_out}


# --------- Kernel objects -----------------------
build/add.o: add.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES_E} -c $< -o ${@F}

# ------------- MLIR generation --------------------------

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size 0 > $@

${trace_mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size ${trace_size} > $

# --------------- XCLBIN build -------------------
${xclbin_target}: ${mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				${aiecc_peano_flags} \
				--aie-generate-npu-insts --npu-insts-name=${insts_target:build/%=%} $(^:%=../%)

${trace_xclbin_target}: ${trace_mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				${aiecc_peano_flags} \
				--aie-generate-npu-insts --npu-insts-name=${insts_target:build/%=%} $(^:%=../%)


# --------------- Host build -------------------

${host_out}: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -ggdb -DDTYPE_IN=${dtype_in_cpp} -DDTYPE_OUT=${dtype_out_cpp} -DDTYPE_ACC=${dtype_acc_cpp}" \
		cmake `${getwslpath} ${srcdir}` -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

#-------------------- Run -----------------
run: ${host_out} ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./${host_out} -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M ${M}  -N ${N} ${runargs}

#-------------------- Trace -----------------
trace: ${host_out} ${trace_xclbin_target}  ${insts_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./${host_out} -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -M ${M} -N ${N} ${runargs} -t ${trace_size}

clean: clean_trace
	rm -rf build _build ${host_out}

clean_trace:
	rm -rf tmpTrace parse*.json trace*json trace.txt
