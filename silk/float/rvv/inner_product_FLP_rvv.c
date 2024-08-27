
/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
              2023 Amazon
              2024 Bernard Gingold, Samsung Electronics
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include "SigProc_FLP.h"
#include "include/opus_sse2rvv_hal.h"

/* inner product of two silk_float arrays, with result as double */
__attribute__((hot))
__attribute__((aligned(32)))
double silk_inner_product_FLP_rvv(const silk_float * __restrict__ data1,
                                  const silk_float * __restrict__ data2,
                                  opus_int                        dataSize) {
    vfloat64m1_t                    accum1;
    vfloat64m1_t                    accum2;
    const silk_float * __restrict__ pd1 = data1;
    const silk_float * __restrict__ pd2 = data2;
    size_t                          vn  = (size_t)dataSize;
    size_t                          vl;
    size_t                          vlmax;
    double                          result;
    int32_t                         i;

    vlmax    = __riscv_vsetvlmax_e64m1();
    accum1   =  __riscv_vfmv_s_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    accum2   = accum1;
    result   = 0.0;
    for(;vn > 0ULL;pd1 += vl,pd2 += vl) {
        register vfloat32m1_t x1f;
        register vfloat32m1_t x2f;
        register vfloat64m1_t x1d;
        register vfloat64m1_t x2d;
        vl    = __riscv_vsetvl_e32m1();
        __builtin_prefetch(pd1+vl,0,1);
        __builtin_prefetch(pd2+vl,0,1);
        x1f   = __riscv_vle32_v_f32m1(pd1,vl);
        x1d   = __riscv_vfwcvt_f_f_v_f64m1(x1f,vl);
        x2f   = __riscv_vle32_v_f32m1(pd2,vl);
        x2d   = __riscv_vfwcvt_f_f_v_f64m1(x2f,vl);
        accum1= __riscv_vfmadd_vv_f64m1(x1d,x2d,accum1,vl);
        x1f   = __riscv_vle32_v_f32m1(pd1+4,vl);
        x1d   = __riscv_vfwcvt_f_f_v_f64m1(x1f,vl);
        x2f   = __riscv_vle32_v_f32m1(pd2+4,vl);
        x2d   = __riscv_vfwcvt_f_f_v_f64m1(x2f,vl);
        accum2= __riscv_vfmadd_vv_f64m1(x1d,x2d,accum2,vl);
    }
    accum1 = __riscv_vfadd_vv_f64m1(accum1,accum1,vlmax);
    accum1 = __riscv_vfadd_vv_f64m1(accum1,__riscv_vslideup_vx_f64m1(accum1,accum1,1,vlmax),vlmax);
    accum1 = __riscv_vhadd_vv_f64m1(accum1,accum1);
    result = __riscv_vfmv_f_s_f64m1_f64(accum1);
    return (result);
}
