
#ifndef __OPUS_SSE2RVV_HAL_H__
#define __OPUS_SSE2RVV_HAL_H__

/*
 * sse2rvv is freely redistributable under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <riscv_vector.h>
#include <stddef.h>
#include <omp.h>

__attribute__((always_inline))
static inline
vfloat64m1_t __riscv_vhadd_vv_f64m1(const vfloat64m1_t a,
                                    const vfloat64m1_t b){

    //size_t vlmax = __riscv_vsetvlmax_e64m1();
    register vfloat64m2_t _a = __riscv_vlmul_ext_v_f64m1_f64m2(a);
    register vfloat64m2_t _b = __riscv_vlmul_ext_v_f64m1_f64m2(b);
    register vfloat64m2_t ab = __riscv_vslideup_vx_f64m2_tu(_a, _b, 2, 8);
    register vfloat64m2_t ab_s = __riscv_vslidedown_vx_f64m2(ab, 1, 8);
    register vfloat64m2_t ab_add = __riscv_vfadd_vv_f64m2(ab, ab_s, 8);
    vbool32_t mask = __riscv_vreinterpret_v_u8m1_b32(__riscv_vmv_s_x_u8m1(85, 4));
     return __riscv_vlmul_trunc_v_f64m2_f64m1(
      __riscv_vcompress_vm_f64m2(ab_add, mask, 4));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_cvtepi16_epi32(const vint32m1_t a,
                                  const size_t vl) {
      register vint16m1_t x = __riscv_vreinterpret_v_i32m1_i16m1(a);
      return (__riscv_vsext_vf2_i32m1(__riscv_vlmul_trunc_v_i16m1_i16mf2(x),vl));
}


__attribute__((always_inline))
static inline
vint32m1_t __riscv_loadl_epi64(const vint32m1_t * maddr) {
      register vint64m1_t addr = __riscv_vreinterpret_v_i32m1_i64m1(*maddr);
      register vint64m1_t vzero= __riscv_vmv_v_x_i64m1(0,2);
      return (__riscv_vreinterpret_v_i64m1_i32m1(__riscv_vslideup_vx_i64m1(addr,vzero,1,2)));
}

/*
   #  define OP_CVTEPI16_EPI32_M64(x) \
 (_mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(void*)(x))))
*/

#define RISCV_CVTEPI16_EPI32_M64(x,vl) \
 (__riscv_cvtepi16_epi32(__riscv_loadl_epi64((vint32m1_t *)(void*)(x)),(vl)))

__attribute__((always_inline))
static inline
vint32m1_t __riscv_shuffle_epi32(const vint32m1_t a,
                                 int imm8,
                                 const size_t vl) {
      register vuint32m1_t x        = __riscv_vreinterpret_v_i32m1_u32m1(a);
      register vuint32m1_t imm8_dup = __riscv_vmv_v_x_u32m1(imm8,vl);
      register vuint32m1_t vid      = __riscv_vsll_vx_u32m1(__riscv_vid_v_u32m1(4),1,vl);
      register vuint32m1_t idxs     = __riscv_vand_vx_u32m1(__riscv_vsrl_vv_u32m1(imm8_dup,vid,vl),0x3,vl);
      return (__riscv_vreinterpret_v_u32m1_i32m1(__riscv_vrgather_vv_u32m1(x,idxs,vl)));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_shufflelo_epi16(const vint32m1_t a,
                                   int imm8) {
      register vuint16m1_t x         = __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vreinterpret_v_i32m1_u32m1(a));
      register vuint16m1_t imm8_dup  = __riscv_vmv_v_x_u16m1(imm8,4);
      register vuint16m1_t vid       = __riscv_vsll_vx_u16m1(__riscv_vid_v_u16m1(4),1,4);
      register vuint16m1_t idxs      = __riscv_vand_vx_u16m1(__riscv_vsrl_vv_u16m1(imm8_dup,vid,4),0x3,4);
      register vuint16m1_t shuffle   = __riscv_vrgather_vv_u16m1(x,idxs,4);
      return (__riscv_vreinterpret_v_u32m1_i32m1(
                                  __riscv_vreinterpret_v_u16m1_u32m1(
                                                  __riscv_vslideup_vx_u16m1_tu(x,shuffle,0,4))));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_srai_epi32(const vint32m1_t a,
                              int imm8,
                              const size_t vl) {
      register vint32m1_t x          = a;
      int64_t             imm8_shift = imm8 >> 1;
      register vint32m1_t x_s        = __riscv_vsra_vx_i32m1(x,imm8_shift,vl);
      return (__riscv_vsra_vx_i32m1(x_s,imm8-imm8_shift,vl));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_unpackhi_epi64(const vint32m1_t a,
                                  const vint32m1_t b) {
      register vint64m1_t    x   = __riscv_vreinterpret_v_u32m1_u64m1(
                                             __riscv_vreinterpret_v_i32m1_u32m1(a));
      register vint64m1_t    y   = __riscv_vreinterpret_v_u32m1_u64m1(
                                             __riscv_vreinterpret_v_i32m1_u32m1(b));
      register vuint64m1_t   x_s = __riscv_vslidedown_vx_u64m1(x,1,2);
      return (__riscv_vreinterpret_v_u32m1_i32m1(
                     __riscv_vreinterpret_v_u64m1_u32m1(
                                  __riscv_vslideup_vx_u64m1_tu(y,x_s,0,1))));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_slli_epi32(const vint32m1_t a,
                              int imm8,
                              const size_t vl) {
      register vint32m1_t x = a;
      const int cimm8       = imm8 & 0xff;
      if(cimm8 > 31)
         return (__riscv_vmv_v_x_i32m1(0,vl));
      else
         return (__riscv_vsll_vx_i32m1(x,cimm8,vl));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_slli_epi64(const vint32m1_t a,
                              int imm8,
                              const size_t vl) {
      register vint64m1_t x = __riscv_vreinterpret_v_i32m1_i64m1(a);
      const int cimm8       = imm8 & 0xff;
      if(cimm8 > 63)
         return (__riscv_vreinterpret_v_i64m1_i32m1(__riscv_vmv_v_x_i64m1(0,vl)));
      else
         return (__riscv_vreinterpret_v_i64m1_i32m1(__riscv_vsll_vx_i32m1(x,cimm8,vl)));
}



__attribute__((always_inline))
static inline
vint32m1_t __riscv_srli_epi32(const vint32m1_t a,
                              int imm8,
                              const size_t vl) {
      register vuint32m1_t x = __riscv_vreinterpret_v_i32m1_u32m1(a);
      const int cimm8       = imm8 & 0xff;
      if(cimm8 > 31)
         return (__riscv_vreinterpret_v_u32m1_i32m1(__riscv_vmv_v_x_u32m1(0,vl)));
      else
         return (__riscv_vreinterpret_v_u32m1_i32m1(__riscv_vsll_vx_u32m1(x,cimm8,vl)));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_srli_epi64(const vint32m1_t a,
                              int imm8,
                              const size_t vl) {
      register vuint64m1_t x = __riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_i32m1_u32m1(a));
      const int cimm8        = imm8 & 0xff;
      if(cimm8 > 63)
         return (__riscv_vreinterpret_v_u32m1_i32m1(__riscv_vmv_v_x_u32m1(0,vl)));
      else
         return (__riscv_vreinterpret_v_u32m1_i32m1(__riscv_vreinterpret_v_u64m1_u32m1(__riscv_vsll_vx_u64m1(x,cimm8,vl))));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_blend_epi16(const vint32m1_t a,
                               const vint32m1_t b,
                               const int imm8,
                               const size_t vl) {
      register vint16m1_t x   = __riscv_vreinterpret_v_i32m1_i16m1(a);
      register vint16m1_t y   = __riscv_vreinterpret_v_i32m1_i16m1(b);
      register vbool16_t  b8  = __riscv_vreinterpret_v_i8m1_b64(__riscv_vmv_s_x_i8m1(imm8, 2));
      return (__riscv_vreinterpret_v_i16m1_i32m1(__riscv_vmerge_vvm_i16m1(x,y,b8,vl)));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_srli_si128(const vint32m1_t a,
                              int imm8) {
      register vuint8m1_t x = __riscv_vreinterpret_v_u32m1_u8m1(
                                            __riscv_vreinterpret_v_i32m1_u32m1(a));
      return (__riscv_vreinterpret_v_u32m1_i32m1(
                  __riscv_vreinterpret_v_u8m1_u32m1(
                                 __riscv_vslidedown_vx_u8m1(x,imm8 & 0xff,__riscv_vsetvlmax_e8m1()))));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_blendv_epi8(const vint32m1_t a,
                               const vint32m1_t b,
                               const vint32m1_t mask) {
      register vint8m1_t x   = __riscv_vreinterpret_v_i32m1_i8m1(a);
      register vint8m1_t y   = __riscv_vreinterpret_v_i32m1_i8m1(b);
      register vint8m1_t msk = __riscv_vreinterpret_v_i32m1_i8m1(mask);
      register vint8m1_t msk_sra = __riscv_vsra_vx_i8m1(msk,7,16);
      register vbool8_t  msk_b8  = __riscv_vmsne_vx_i8m1_b8(msk_sra,0,16);
      return (__riscv_vreinterpret_v_i8m1_i32m1(__riscv_vmerge_vvm_i8m1(x,y,msk_b8,16)));
}

__attribute__((always_inline))
static inline
vint32m1_t __riscv_andnot_epi32(const vint32m1_t a,
                                const vint32m1_t b,
                                const size_t     vl) {
      return (__riscv_vand_vv_i32m1(__riscv_vnot_v_i32m1(a,vl),b,vl));
}

__attribute__((always_inline))
static inline
 vint32m1_t __riscv_sign_epi32(vint32m1_t a,
                               vint32m1_t b,
                               const size_t vl) {
  vbool32_t lt_mask = __riscv_vmslt_vx_i32m1_b32(b, 0, vl);
  vbool32_t zero_mask = __riscv_vmseq_vx_i32m1_b32(b, 0, vl);
  vint32m1_t a_neg = __riscv_vneg_v_i32m1(a, vl);
  vint32m1_t res_lt = __riscv_vmerge_vvm_i32m1(a, a_neg, lt_mask, vl);
  return (__riscv_vmerge_vxm_i32m1(res_lt, 0, zero_mask, vl));
 }

__attribute__((always_inline))
static inline
vint32m1_t __riscv_permute8x32_epi32(vint32m1_t a,vint32m1_t idx) {
   int32_t a_vec[8];
   __riscv_vse32_v_i32m1(&a_vec[0],a,8);
   int32_t idx_vec[8];
   __riscv_vse32_v_i32m1(&idx_vec[0],idx,8);
   int32_t dst_vec[8];
   int j;
   for(j = 0; j <= 7; ++j) {
       const int32_t id = idx_vec[j] & 0x7;
       dst_vec[j]       = a_vec[id];
   }
   return (__riscv_vle32_v_i32m1(dst_vec,8));
}

// 128-bit returned.
__attribute__((always_inline))
static inline
vint32m1_t __riscv_castsi256_si128(const vint32m1_t a) {
   int32_t mem[4];
   __riscv_vse32_v_i32m1(mem,a,4);
   return (__riscv_vle32_v_i32m1(mem,4));
}

__attribute__((always_inline))
static inline int64_t SignExtend(int64_t val64) {
       return val64;
}

__attribute__((always_inline))
static inline
vint64m1_t __riscv_cvtepi32_epi64(const vint32m1_t a) {
   int32_t mem[4];
   __riscv_vse32_v_i32m1(mem,a,4);
   int64_t dst[4];
   for(int j = 0; j <= 3; ++j) {
       dst[j] = SignExtend(mem[j]);
   }
   return (__riscv_vle64_v_i64m1(dst,4));
}




#endif /*__OPUS_SSE2RVV_HAL_H__*/
