//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 64
#endif

template <typename T_in, typename T_out, const int M>
void eltwise_add(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < M; i++) {
    c[i] = a[i] + b[i];
  }
}

template <typename T_in, typename T_out, const int M>
void eltwise_vadd(T_in *a, T_in *b, T_out *c) {

  constexpr int vec_factor = 8;
  event0();
  T_in *__restrict pA1 = a;
  T_in *__restrict pB1 = b;
  T_out *__restrict pC1 = c;
  const int F = M / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int i = 0; i < F; i++) {
    aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
    pB1 += vec_factor;
    aie::vector<T_out, vec_factor> cout = aie::add(A0, B0);
    aie::store_v(pC1, cout);
    pC1 += vec_factor;
  }
  event1();
}

template <typename T_in, typename T_out, const int M>
void vstore(T_in *a, T_out *c) {
  constexpr int vec_factor = 8;
  event0();
  T_in *__restrict pA1 = a;
  T_out *__restrict pC1 = c;
  const int F = M / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int i = 0; i < F; i++) {
    aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::store_v(pC1, A0);
    pC1 += vec_factor;
  }
  event1();
}
extern "C" {


void ewise_add_i16_i16 (int16 *a, int16 *b, int16 *c){
  eltwise_vadd<int16, int16, DIM_M>(a, b, c);
}
void ewise_add_bf16_bf16 (bfloat16 *a, bfloat16 *b, bfloat16 *c){
  eltwise_vadd<bfloat16, bfloat16, DIM_M>(a, b, c);
}

void store_i16_i16 (int16 *c_in, int16 *c_out){
  vstore<int16, int16, DIM_M>(c_in, c_out);
}

void store_bf16_bf16 (bfloat16 *c_in, bfloat16 *c_out){
  vstore<bfloat16, bfloat16, DIM_M>(c_in, c_out);
}

void ewise_add_f32_f32 (float *a, float *b, float *c){
  eltwise_vadd<float, float, DIM_M>(a, b, c);
}

} // extern "C"
