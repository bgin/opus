
/* Copyright (c) 2014-2020, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang FrancisQuiers
   Converted to RVV version by Bernard Gingold, Samsung Electronics 2024
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <riscv_vector.h>
#include "include/opus_sse2rvv_hal.h"
#include "SigProc_FIX.h"
#include "define.h"
#include "tuning_parameters.h"
#include "pitch.h"

#define MAX_FRAME_SIZE              384             /* subfr_length * nb_subfr = ( 0.005 * 16000 + 16 ) * 4 = 384 */
#define QA                          25
#define N_BITS_HEAD_ROOM            3
#define MIN_RSHIFTS                 -16
#define MAX_RSHIFTS                 (32 - QA)

/* Compute reflection coefficients from input signal */
__attribute__((hot))
__attribute__((aligned(32)))
void silk_burg_modified_rvv(
    opus_int32                  * __restrict__ res_nrg,           /* O    Residual energy                                             */
    opus_int                    * __restrict__ res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */
)
{
       
    opus_int32       C_first_row[ SILK_MAX_ORDER_LPC ];
    opus_int32       C_last_row[  SILK_MAX_ORDER_LPC ];
    opus_int32       Af_QA[       SILK_MAX_ORDER_LPC ];
    opus_int32       CAf[ SILK_MAX_ORDER_LPC + 1 ];
    opus_int32       CAb[ SILK_MAX_ORDER_LPC + 1 ];
    opus_int32       xcorr[ SILK_MAX_ORDER_LPC ];
    vint32m1_t       FIRST_3210, LAST_3210, ATMP_3210, TMP1_3210, TMP2_3210, T1_3210, T2_3210, PTR_3210, SUBFR_3210, X1_3210, X2_3210;
    vint32m1_t       CONST1;
    const opus_int16 * __restrict__ x_ptr;
    opus_int32       * __restrict__ pC_first_row;
    opus_int32       * __restrict__ pC_last_row;
    opus_int32       * __restrict__ pAf_QA;
    opus_int32       * __restrict__ pCAf;
    opus_int32       * __restrict__ pCAb;
    opus_int64       C0_64;
    size_t           vn;
    size_t           vl;
    opus_int         k, n, s, lz, rshifts, reached_max_gain;
    opus_int32       C0, num, nrg, rc_Q31, invGain_Q30, Atmp_QA, Atmp1, tmp1, tmp2, x1, x2;

    CONST1 = __riscv_vmv_v_x_i32m1(1,__riscv_vsetvlmax_e32m1());
    celt_assert( subfr_length * nb_subfr <= MAX_FRAME_SIZE );
     /* Compute autocorrelations, added over subframes */
    C0_64 = silk_inner_prod16( x, x, subfr_length*nb_subfr, arch );
    lz = silk_CLZ64(C0_64);
    rshifts = 32 + 1 + N_BITS_HEAD_ROOM - lz;
    if (rshifts > MAX_RSHIFTS) rshifts = MAX_RSHIFTS;
    if (rshifts < MIN_RSHIFTS) rshifts = MIN_RSHIFTS;
    
    if (rshifts > 0) {
        C0 = (opus_int32)silk_RSHIFT64(C0_64, rshifts );
    } else {
        C0 = silk_LSHIFT32((opus_int32)C0_64, -rshifts );
    }
    CAb[ 0 ] = CAf[ 0 ] = C0 + silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ) + 1;                                /* Q(-rshifts) */
    silk_memset( C_first_row, 0, SILK_MAX_ORDER_LPC * sizeof( opus_int32 ) );
    if( rshifts > 0 ) {
        for( s = 0; s < nb_subfr; s++ ) {
            x_ptr = x + s * subfr_length;
            for( n = 1; n < D + 1; n++ ) {
                C_first_row[ n - 1 ] += (opus_int32)silk_RSHIFT64(
                    silk_inner_prod16( x_ptr, x_ptr + n, subfr_length - n, arch ), rshifts );
            }
        }
    } else {
        for( s = 0; s < nb_subfr; s++ ) {
            int i;
            opus_int32 d;
            x_ptr = x + s * subfr_length;
            celt_pitch_xcorr(x_ptr, x_ptr + 1, xcorr, subfr_length - D, D, arch );
            for( n = 1; n < D + 1; n++ ) {
               for ( i = n + subfr_length - D, d = 0; i < subfr_length; i++ )
                  d = MAC16_16( d, x_ptr[ i ], x_ptr[ i - n ] );
               xcorr[ n - 1 ] += d;
            }
            for( n = 1; n < D + 1; n++ ) {
                C_first_row[ n - 1 ] += silk_LSHIFT32( xcorr[ n - 1 ], -rshifts );
            }
        }
    }
    silk_memcpy( C_last_row, C_first_row, SILK_MAX_ORDER_LPC * sizeof( opus_int32 ) );
    /* Initialize */
    CAb[ 0 ] = CAf[ 0 ] = C0 + silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ) + 1;                                /* Q(-rshifts) */
    
    invGain_Q30 = (opus_int32)1 << 30;
    reached_max_gain = 0;
    pC_first_row = &C_first_row[0];
    pC_last_row  = &C_last_row[0];
    pAf_QA       = &Af_QA[0];
    pCAf         = &CAf[0];
    pCAb         = &CAb[0];
    for( n = 0; n < D; n++ ) {
        /* Update first row of correlation matrix (without first element) */
        /* Update last row of correlation matrix (without last element, stored in reversed order) */
        /* Update C * Af */
        /* Update C * flipud(Af) (stored in reversed order) */
        if( rshifts > -2 ) {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                x1  = -silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    16 - rshifts );        /* Q(16-rshifts) */
                x2  = -silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], 16 - rshifts );        /* Q(16-rshifts) */
                tmp1 = silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    QA - 16 );             /* Q(QA-16) */
                tmp2 = silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], QA - 16 );             /* Q(QA-16) */
                for( k = 0; k < n; k++ ) {
                    C_first_row[ k ] = silk_SMLAWB( C_first_row[ k ], x1, x_ptr[ n - k - 1 ]            ); /* Q( -rshifts ) */
                    C_last_row[ k ]  = silk_SMLAWB( C_last_row[ k ],  x2, x_ptr[ subfr_length - n + k ] ); /* Q( -rshifts ) */
                    Atmp_QA = Af_QA[ k ];
                    tmp1 = silk_SMLAWB( tmp1, Atmp_QA, x_ptr[ n - k - 1 ]            );                 /* Q(QA-16) */
                    tmp2 = silk_SMLAWB( tmp2, Atmp_QA, x_ptr[ subfr_length - n + k ] );                 /* Q(QA-16) */
                }
                tmp1 = silk_LSHIFT32( -tmp1, 32 - QA - rshifts );                                       /* Q(16-rshifts) */
                tmp2 = silk_LSHIFT32( -tmp2, 32 - QA - rshifts );                                       /* Q(16-rshifts) */
                for( k = 0; k <= n; k++ ) {
                    CAf[ k ] = silk_SMLAWB( CAf[ k ], tmp1, x_ptr[ n - k ]                    );        /* Q( -rshift ) */
                    CAb[ k ] = silk_SMLAWB( CAb[ k ], tmp2, x_ptr[ subfr_length - n + k - 1 ] );        /* Q( -rshift ) */
                }
            }
        } else {
            vn = (size_t)(n-3);
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                x1  = -silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    -rshifts );            /* Q( -rshifts ) */
                x2  = -silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], -rshifts );            /* Q( -rshifts ) */
                tmp1 = silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    17 );                  /* Q17 */
                tmp2 = silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], 17 );                  /* Q17 */

                X1_3210   =  __riscv_vmv_v_x_i32m1(x1,__riscv_vsetvlmax_e32m1());
                X2_3210   =  __riscv_vmv_v_x_i32m1(x2,__riscv_vsetvlmax_e32m1());
                TMP1_3210 =  __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
                TMP2_3210 =  __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
                for(;vn>0ULL;vn -= vl,x_ptr += vl,pC_first_row += vl,pC_last_row += vl,pAf_QA += vl ) {
                    vl         = __riscv_vsetvl_e32m1(vn);
                    PTR_3210   = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ n - (int)vl - 1 - 3 ],vl );
                    SUBFR_3210 = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ subfr_length - n + (int)vl ],vl );
                    FIRST_3210 = __riscv_vle32_v_i32m1(pC_first_row,vl); 
                    PTR_3210   = __riscv_shuffle_epi32(PTR_3210,27,vl);
                    LAST_3210  = __riscv_vle32_v_i32m1(pC_last_row,vl);
                    ATMP_3210  = __riscv_vle32_v_i32m1(pAf_QA,vl);

                    T1_3210    = __riscv_vmul_vv_i32m1(PTR_3210,X1_3210,vl);
                    T2_3210    = __riscv_vmul_vv_i32m1(SUBFR_3210,X2_3210,vl);

                    ATMP_3210  = __riscv_srai_epi32(ATMP_3210,7,vl);
                    ATMP_3210  = __riscv_vadd_vv_i32m1(ATMP_3210,CONST1,vl);
                    ATMP_3210  = __riscv_srai_epi32(ATMP_3210,1,vl);

                    FIRST_3210 = __riscv_vadd_vv_i32m1( FIRST_3210, T1_3210,vl );
                    LAST_3210  = __riscv_vadd_vv_i32m1( LAST_3210,  T2_3210,vl );

                    PTR_3210   = __riscv_vmul_vv_i32m1(ATMP_3210,PTR_3210,vl);
                    SUBFR_3210 = __riscv_vmul_vv_i32m1(ATMP_3210,SUBFR_3210,vl);

                    __riscv_vse32_v_i32m1(pC_first_row,FIRST_3210,vl);
                    __riscv_vse32_v_i32m1(pC_last_row,LAST_3210,vl);

                    TMP1_3210  = __riscv_vadd_v_i32m1(TMP1_3210,PTR_3210,vl);
                    TMP2_3210  = __riscv_vadd_v_i32m1(TMP2_3210,SUBFR_3210,vl);

                }

                TMP1_3210 = __riscv_vadd_vv_i32m1(TMP1_3210,__riscv_unpackhi_epi64(TMP1_3210,TMP1_3210),__riscv_vsetvlmax_e32m1());
                TMP2_3210 = __riscv_vadd_vv_i32m1(TMP2_3210,__riscv_unpackhi_epi64(TMP2_3210,TMP2_3210),__riscv_vsetvlmax_e32m1());
                TMP1_3210 = __riscv_vadd_vv_i32m1(TMP1_3210,__riscv_shufflelo_epi16(TMP1_3210,0xE),__riscv_vsetvlmax_e32m1());
                TMP2_3210 = __riscv_vadd_vv_i32m1(TMP2_3210,__riscv_shufflelo_epi16(TMP2_3210,0xE),__riscv_vsetvlmax_e32m1());

                tmp1 += __riscv_vmv_x_s_i32m1_i32(TMP1_3210);
                tmp2 += __riscv_vmv_x_s_i32m1_i32(TMP2_3210);

                tmp1 = -tmp1;                /* Q17 */
                tmp2 = -tmp2;                /* Q17 */

                {
                    vint32m1_t xmm_tmp1, xmm_tmp2;
                    vint32m1_t xmm_x_ptr_n_k_x2x0, xmm_x_ptr_n_k_x3x1;
                    vint32m1_t xmm_x_ptr_sub_x2x0, xmm_x_ptr_sub_x3x1;

                    xmm_tmp1 = __riscv_vmv_v_x_i32m1(tmp1,__riscv_vsetvlmax_e32m1());
                    xmm_tmp2 = __riscv_vmv_v_x_i32m1(tmp2,__riscv_vsetvlmax_e32m1());

                    for(;vn>0ULL;vn -= vl,x_ptr += vl,pCAf += vl,pCAb += vl) {
                        vl         = __riscv_vsetvl_e32m1(vn);
                        xmm_x_ptr_n_k_x2x0 = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ n - (int)vl - 3 ],vl );
                        xmm_x_ptr_sub_x2x0 = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ subfr_length - n + (int)vl - 1 ],vl );

                        xmm_x_ptr_n_k_x2x0 = __riscv_shuffle_epi32( xmm_x_ptr_n_k_x2x0,27,vl);

                        xmm_x_ptr_n_k_x2x0 = __riscv_slli_epi32( xmm_x_ptr_n_k_x2x0, -rshifts - 1,vl );
                        xmm_x_ptr_sub_x2x0 = __riscv_slli_epi32( xmm_x_ptr_sub_x2x0, -rshifts - 1,vl );

                        /* equal shift right 4 bytes, xmm_x_ptr_n_k_x3x1 = _mm_srli_si128(xmm_x_ptr_n_k_x2x0, 4)*/
                        xmm_x_ptr_n_k_x3x1 = __riscv_shuffle_epi32( xmm_x_ptr_n_k_x2x0,57,vl);
                        xmm_x_ptr_sub_x3x1 = __riscv_shuffle_epi32( xmm_x_ptr_sub_x2x0,57,vl);

                        xmm_x_ptr_n_k_x2x0 = __riscv_vmul_vv_i32m1(xmm_x_ptr_n_k_x2x0, xmm_tmp1,vl);
                        xmm_x_ptr_n_k_x3x1 = __riscv_vmul_vv_i32m1(xmm_x_ptr_n_k_x3x1, xmm_tmp1,vl);
                        xmm_x_ptr_sub_x2x0 = __riscv_vmul_vv_i32m1(xmm_x_ptr_sub_x2x0, xmm_tmp2,vl);
                        xmm_x_ptr_sub_x3x1 = __riscv_vmul_vv_i32m1(xmm_x_ptr_sub_x3x1, xmm_tmp2,vl);

                        xmm_x_ptr_n_k_x2x0 = __riscv_srli_epi64( xmm_x_ptr_n_k_x2x0, 16,vl);
                        xmm_x_ptr_n_k_x3x1 = __riscv_slli_epi64( xmm_x_ptr_n_k_x3x1, 16,vl);
                        xmm_x_ptr_sub_x2x0 = __riscv_srli_epi64( xmm_x_ptr_sub_x2x0, 16,vl);
                        xmm_x_ptr_sub_x3x1 = __riscv_slli_epi64( xmm_x_ptr_sub_x3x1, 16,vl);

                        xmm_x_ptr_n_k_x2x0 = __riscv_blend_epi16( xmm_x_ptr_n_k_x2x0, xmm_x_ptr_n_k_x3x1, 0xCC,vl);
                        xmm_x_ptr_sub_x2x0 = __riscv_blend_epi16( xmm_x_ptr_sub_x2x0, xmm_x_ptr_sub_x3x1, 0xCC,vl);

                        X1_3210  = __riscv_vle32_v_i32m1(pCAf,vl);
                        PTR_3210 = __riscv_vle32_v_i32m1(pCAb,vl);

                        X1_3210  = __riscv_vadd_vv_i32m1(X1_3210,xmm_x_ptr_n_k_x2x0,vl);
                        PTR_3210 = __riscv_vadd_vv_i32m1(PTR_3210,xmm_x_ptr_sub_x2x0,vl);

                        __riscv_vse32_v_i32m1(pCAf,X1_3210,vl);
                        __riscv_vse32_v_i32m1(pCAb,PTR_3210,vl);
                    }

                }
            }
        }

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        tmp1 = C_first_row[ n ];                                                                        /* Q( -rshifts ) */
        tmp2 = C_last_row[ n ];                                                                         /* Q( -rshifts ) */
        num  = 0;                                                                                       /* Q( -rshifts ) */
        nrg  = silk_ADD32( CAb[ 0 ], CAf[ 0 ] );                                                        /* Q( 1-rshifts ) */
        for( k = 0; k < n; k++ ) {
            Atmp_QA = Af_QA[ k ];
            lz = silk_CLZ32( silk_abs( Atmp_QA ) ) - 1;
            lz = silk_min( 32 - QA, lz );
            Atmp1 = silk_LSHIFT32( Atmp_QA, lz );                                                       /* Q( QA + lz ) */

            tmp1 = silk_ADD_LSHIFT32( tmp1, silk_SMMUL( C_last_row[  n - k - 1 ], Atmp1 ), 32 - QA - lz );  /* Q( -rshifts ) */
            tmp2 = silk_ADD_LSHIFT32( tmp2, silk_SMMUL( C_first_row[ n - k - 1 ], Atmp1 ), 32 - QA - lz );  /* Q( -rshifts ) */
            num  = silk_ADD_LSHIFT32( num,  silk_SMMUL( CAb[ n - k ],             Atmp1 ), 32 - QA - lz );  /* Q( -rshifts ) */
            nrg  = silk_ADD_LSHIFT32( nrg,  silk_SMMUL( silk_ADD32( CAb[ k + 1 ], CAf[ k + 1 ] ),
                                                                                Atmp1 ), 32 - QA - lz );    /* Q( 1-rshifts ) */
        }
        CAf[ n + 1 ] = tmp1;                                                                            /* Q( -rshifts ) */
        CAb[ n + 1 ] = tmp2;                                                                            /* Q( -rshifts ) */
        num = silk_ADD32( num, tmp2 );                                                                  /* Q( -rshifts ) */
        num = silk_LSHIFT32( -num, 1 );                                                                 /* Q( 1-rshifts ) */

        /* Calculate the next order reflection (parcor) coefficient */
        if( silk_abs( num ) < nrg ) {
            rc_Q31 = silk_DIV32_varQ( num, nrg, 31 );
        } else {
            rc_Q31 = ( num > 0 ) ? silk_int32_MAX : silk_int32_MIN;
        }

        /* Update inverse prediction gain */
        tmp1 = ( (opus_int32)1 << 30 ) - silk_SMMUL( rc_Q31, rc_Q31 );
        tmp1 = silk_LSHIFT( silk_SMMUL( invGain_Q30, tmp1 ), 2 );
        if( tmp1 <= minInvGain_Q30 ) {
            /* Max prediction gain exceeded; set reflection coefficient such that max prediction gain is exactly hit */
            tmp2 = ( (opus_int32)1 << 30 ) - silk_DIV32_varQ( minInvGain_Q30, invGain_Q30, 30 );            /* Q30 */
            rc_Q31 = silk_SQRT_APPROX( tmp2 );                                                  /* Q15 */
            if( rc_Q31 > 0 ) {
                 /* Newton-Raphson iteration */
                rc_Q31 = silk_RSHIFT32( rc_Q31 + silk_DIV32( tmp2, rc_Q31 ), 1 );                   /* Q15 */
                rc_Q31 = silk_LSHIFT32( rc_Q31, 16 );                                               /* Q31 */
                if( num < 0 ) {
                    /* Ensure adjusted reflection coefficients has the original sign */
                    rc_Q31 = -rc_Q31;
                }
            }
            invGain_Q30 = minInvGain_Q30;
            reached_max_gain = 1;
        } else {
            invGain_Q30 = tmp1;
        }

        /* Update the AR coefficients */
        for( k = 0; k < (n + 1) >> 1; k++ ) {
            tmp1 = Af_QA[ k ];                                                                  /* QA */
            tmp2 = Af_QA[ n - k - 1 ];                                                          /* QA */
            Af_QA[ k ]         = silk_ADD_LSHIFT32( tmp1, silk_SMMUL( tmp2, rc_Q31 ), 1 );      /* QA */
            Af_QA[ n - k - 1 ] = silk_ADD_LSHIFT32( tmp2, silk_SMMUL( tmp1, rc_Q31 ), 1 );      /* QA */
        }
        Af_QA[ n ] = silk_RSHIFT32( rc_Q31, 31 - QA );                                          /* QA */

        if( reached_max_gain ) {
            /* Reached max prediction gain; set remaining coefficients to zero and exit loop */
            for( k = n + 1; k < D; k++ ) {
                Af_QA[ k ] = 0;
            }
            break;
        }

        /* Update C * Af and C * Ab */
        for( k = 0; k <= n + 1; k++ ) {
            tmp1 = CAf[ k ];                                                                    /* Q( -rshifts ) */
            tmp2 = CAb[ n - k + 1 ];                                                            /* Q( -rshifts ) */
            CAf[ k ]         = silk_ADD_LSHIFT32( tmp1, silk_SMMUL( tmp2, rc_Q31 ), 1 );        /* Q( -rshifts ) */
            CAb[ n - k + 1 ] = silk_ADD_LSHIFT32( tmp2, silk_SMMUL( tmp1, rc_Q31 ), 1 );        /* Q( -rshifts ) */
        }
    }

    if( reached_max_gain ) {
        for( k = 0; k < D; k++ ) {
            /* Scale coefficients */
            A_Q16[ k ] = -silk_RSHIFT_ROUND( Af_QA[ k ], QA - 16 );
        }
        /* Subtract energy of preceding samples from C0 */
        if( rshifts > 0 ) {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                C0 -= (opus_int32)silk_RSHIFT64( silk_inner_prod16( x_ptr, x_ptr, D, arch ), rshifts );
            }
        } else {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                C0 -= silk_LSHIFT32( silk_inner_prod_aligned( x_ptr, x_ptr, D, arch ), -rshifts );
            }
        }
        /* Approximate residual energy */
        *res_nrg = silk_LSHIFT( silk_SMMUL( invGain_Q30, C0 ), 2 );
        *res_nrg_Q = -rshifts;
    } else {
        /* Return residual energy */
        nrg  = CAf[ 0 ];                                                                            /* Q( -rshifts ) */
        tmp1 = (opus_int32)1 << 16;                                                                             /* Q16 */
        for( k = 0; k < D; k++ ) {
            Atmp1 = silk_RSHIFT_ROUND( Af_QA[ k ], QA - 16 );                                       /* Q16 */
            nrg  = silk_SMLAWW( nrg, CAf[ k + 1 ], Atmp1 );                                         /* Q( -rshifts ) */
            tmp1 = silk_SMLAWW( tmp1, Atmp1, Atmp1 );                                               /* Q16 */
            A_Q16[ k ] = -Atmp1;
        }
        *res_nrg = silk_SMLAWW( nrg, silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ), -tmp1 );/* Q( -rshifts ) */
        *res_nrg_Q = -rshifts;
    }

#ifdef OPUS_CHECK_ASM
    {
        opus_int32 res_nrg_c = 0;
        opus_int res_nrg_Q_c = 0;
        opus_int32 A_Q16_c[ MAX_LPC_ORDER ] = {0};

        silk_burg_modified_c(
            &res_nrg_c,
            &res_nrg_Q_c,
            A_Q16_c,
            x,
            minInvGain_Q30,
            subfr_length,
            nb_subfr,
            D,
            0
        );

        silk_assert( *res_nrg == res_nrg_c );
        silk_assert( *res_nrg_Q == res_nrg_Q_c );
        silk_assert( !memcmp( A_Q16, A_Q16_c, D * sizeof( *A_Q16 ) ) );
    }
#endif
   

}


/* Compute reflection coefficients from input signal */
// RVV version limited to vector length of 128-bit.
// This is a fallback version in case of any kind of errors due to RVV-VL version.

__attribute__((hot))
__attribute__((aligned(32)))
void silk_burg_modified_rvv_128b(
    opus_int32                  * __restrict__ res_nrg,           /* O    Residual energy                                             */
    opus_int                    * __restrict__ res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */
)
{
       
    opus_int32       C_first_row[ SILK_MAX_ORDER_LPC ];
    opus_int32       C_last_row[  SILK_MAX_ORDER_LPC ];
    opus_int32       Af_QA[       SILK_MAX_ORDER_LPC ];
    opus_int32       CAf[ SILK_MAX_ORDER_LPC + 1 ];
    opus_int32       CAb[ SILK_MAX_ORDER_LPC + 1 ];
    opus_int32       xcorr[ SILK_MAX_ORDER_LPC ];
    vint32m1_t       FIRST_3210, LAST_3210, ATMP_3210, TMP1_3210, TMP2_3210, T1_3210, T2_3210, PTR_3210, SUBFR_3210, X1_3210, X2_3210;
    vint32m1_t       CONST1;
    const opus_int16 * __restrict__ x_ptr;
    opus_int32       * __restrict__ pC_first_row;
    opus_int32       * __restrict__ pC_last_row;
    opus_int32       * __restrict__ pAf_QA;
    opus_int32       * __restrict__ pCAf;
    opus_int32       * __restrict__ pCAb;
    opus_int64       C0_64;
    //size_t           vn;
    //size_t           vl;
    opus_int         k, n, s, lz, rshifts, reached_max_gain;
    opus_int32       C0, num, nrg, rc_Q31, invGain_Q30, Atmp_QA, Atmp1, tmp1, tmp2, x1, x2;

    CONST1 = __riscv_vmv_v_x_i32m1(1,4);
    celt_assert( subfr_length * nb_subfr <= MAX_FRAME_SIZE );
     /* Compute autocorrelations, added over subframes */
    C0_64 = silk_inner_prod16( x, x, subfr_length*nb_subfr, arch );
    lz = silk_CLZ64(C0_64);
    rshifts = 32 + 1 + N_BITS_HEAD_ROOM - lz;
    if (rshifts > MAX_RSHIFTS) rshifts = MAX_RSHIFTS;
    if (rshifts < MIN_RSHIFTS) rshifts = MIN_RSHIFTS;
    
    if (rshifts > 0) {
        C0 = (opus_int32)silk_RSHIFT64(C0_64, rshifts );
    } else {
        C0 = silk_LSHIFT32((opus_int32)C0_64, -rshifts );
    }
    CAb[ 0 ] = CAf[ 0 ] = C0 + silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ) + 1;                                /* Q(-rshifts) */
    silk_memset( C_first_row, 0, SILK_MAX_ORDER_LPC * sizeof( opus_int32 ) );
    if( rshifts > 0 ) {
        for( s = 0; s < nb_subfr; s++ ) {
            x_ptr = x + s * subfr_length;
            for( n = 1; n < D + 1; n++ ) {
                C_first_row[ n - 1 ] += (opus_int32)silk_RSHIFT64(
                    silk_inner_prod16( x_ptr, x_ptr + n, subfr_length - n, arch ), rshifts );
            }
        }
    } else {
        for( s = 0; s < nb_subfr; s++ ) {
            int i;
            opus_int32 d;
            x_ptr = x + s * subfr_length;
            celt_pitch_xcorr(x_ptr, x_ptr + 1, xcorr, subfr_length - D, D, arch );
            for( n = 1; n < D + 1; n++ ) {
               for ( i = n + subfr_length - D, d = 0; i < subfr_length; i++ )
                  d = MAC16_16( d, x_ptr[ i ], x_ptr[ i - n ] );
               xcorr[ n - 1 ] += d;
            }
            for( n = 1; n < D + 1; n++ ) {
                C_first_row[ n - 1 ] += silk_LSHIFT32( xcorr[ n - 1 ], -rshifts );
            }
        }
    }
    silk_memcpy( C_last_row, C_first_row, SILK_MAX_ORDER_LPC * sizeof( opus_int32 ) );
    /* Initialize */
    CAb[ 0 ] = CAf[ 0 ] = C0 + silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ) + 1;                                /* Q(-rshifts) */
    
    invGain_Q30 = (opus_int32)1 << 30;
    reached_max_gain = 0;
    pC_first_row = &C_first_row[0];
    pC_last_row  = &C_last_row[0];
    pAf_QA       = &Af_QA[0];
    pCAf         = &CAf[0];
    pCAb         = &CAb[0];
    for( n = 0; n < D; n++ ) {
        /* Update first row of correlation matrix (without first element) */
        /* Update last row of correlation matrix (without last element, stored in reversed order) */
        /* Update C * Af */
        /* Update C * flipud(Af) (stored in reversed order) */
        if( rshifts > -2 ) {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                x1  = -silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    16 - rshifts );        /* Q(16-rshifts) */
                x2  = -silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], 16 - rshifts );        /* Q(16-rshifts) */
                tmp1 = silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    QA - 16 );             /* Q(QA-16) */
                tmp2 = silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], QA - 16 );             /* Q(QA-16) */
                for( k = 0; k < n; k++ ) {
                    C_first_row[ k ] = silk_SMLAWB( C_first_row[ k ], x1, x_ptr[ n - k - 1 ]            ); /* Q( -rshifts ) */
                    C_last_row[ k ]  = silk_SMLAWB( C_last_row[ k ],  x2, x_ptr[ subfr_length - n + k ] ); /* Q( -rshifts ) */
                    Atmp_QA = Af_QA[ k ];
                    tmp1 = silk_SMLAWB( tmp1, Atmp_QA, x_ptr[ n - k - 1 ]            );                 /* Q(QA-16) */
                    tmp2 = silk_SMLAWB( tmp2, Atmp_QA, x_ptr[ subfr_length - n + k ] );                 /* Q(QA-16) */
                }
                tmp1 = silk_LSHIFT32( -tmp1, 32 - QA - rshifts );                                       /* Q(16-rshifts) */
                tmp2 = silk_LSHIFT32( -tmp2, 32 - QA - rshifts );                                       /* Q(16-rshifts) */
                for( k = 0; k <= n; k++ ) {
                    CAf[ k ] = silk_SMLAWB( CAf[ k ], tmp1, x_ptr[ n - k ]                    );        /* Q( -rshift ) */
                    CAb[ k ] = silk_SMLAWB( CAb[ k ], tmp2, x_ptr[ subfr_length - n + k - 1 ] );        /* Q( -rshift ) */
                }
            }
        } else {
            //vn = (size_t)(n-3);
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                x1  = -silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    -rshifts );            /* Q( -rshifts ) */
                x2  = -silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], -rshifts );            /* Q( -rshifts ) */
                tmp1 = silk_LSHIFT32( (opus_int32)x_ptr[ n ],                    17 );                  /* Q17 */
                tmp2 = silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n - 1 ], 17 );                  /* Q17 */

                X1_3210   =  __riscv_vmv_v_x_i32m1(x1,4);
                X2_3210   =  __riscv_vmv_v_x_i32m1(x2,4);
                TMP1_3210 =  __riscv_vmv_v_x_i32m1(0, 4);
                TMP2_3210 =  __riscv_vmv_v_x_i32m1(0, 4);
                for(k = 0; k < n-3; k += 4) {
                    //vl         = __riscv_vsetvl_e32m1(vn);
                    PTR_3210   = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ n - k - 1 - 3 ],4 );
                    SUBFR_3210 = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ subfr_length - n + k ],4 );
                    FIRST_3210 = __riscv_vle32_v_i32m1(&pC_first_row[k],4); 
                    PTR_3210   = __riscv_shuffle_epi32(PTR_3210,27,4);
                    LAST_3210  = __riscv_vle32_v_i32m1(&pC_last_row[k],4);
                    ATMP_3210  = __riscv_vle32_v_i32m1(&pAf_QA[k],4);

                    T1_3210    = __riscv_vmul_vv_i32m1(PTR_3210,X1_3210,4);
                    T2_3210    = __riscv_vmul_vv_i32m1(SUBFR_3210,X2_3210,4);

                    ATMP_3210  = __riscv_srai_epi32(ATMP_3210,7,4);
                    ATMP_3210  = __riscv_vadd_vv_i32m1(ATMP_3210,CONST1,4);
                    ATMP_3210  = __riscv_srai_epi32(ATMP_3210,1,4);

                    FIRST_3210 = __riscv_vadd_vv_i32m1( FIRST_3210, T1_3210,4 );
                    LAST_3210  = __riscv_vadd_vv_i32m1( LAST_3210,  T2_3210,4 );

                    PTR_3210   = __riscv_vmul_vv_i32m1(ATMP_3210,PTR_3210,4);
                    SUBFR_3210 = __riscv_vmul_vv_i32m1(ATMP_3210,SUBFR_3210,4);

                    __riscv_vse32_v_i32m1(&pC_first_row[k],FIRST_3210,4);
                    __riscv_vse32_v_i32m1(&pC_last_row[k],LAST_3210,4);

                    TMP1_3210  = __riscv_vadd_v_i32m1(TMP1_3210,PTR_3210,4);
                    TMP2_3210  = __riscv_vadd_v_i32m1(TMP2_3210,SUBFR_3210,4);

                }

                TMP1_3210 = __riscv_vadd_vv_i32m1(TMP1_3210,__riscv_unpackhi_epi64(TMP1_3210,TMP1_3210),4);
                TMP2_3210 = __riscv_vadd_vv_i32m1(TMP2_3210,__riscv_unpackhi_epi64(TMP2_3210,TMP2_3210),4);
                TMP1_3210 = __riscv_vadd_vv_i32m1(TMP1_3210,__riscv_shufflelo_epi16(TMP1_3210,0xE),4);
                TMP2_3210 = __riscv_vadd_vv_i32m1(TMP2_3210,__riscv_shufflelo_epi16(TMP2_3210,0xE),4);

                tmp1 += __riscv_vmv_x_s_i32m1_i32(TMP1_3210);
                tmp2 += __riscv_vmv_x_s_i32m1_i32(TMP2_3210);

                for( ; k < n; k++ ) {
                    C_first_row[ k ] = silk_MLA( pC_first_row[ k ], x1, x_ptr[ n - k - 1 ]            ); /* Q( -rshifts ) */
                    C_last_row[ k ]  = silk_MLA( pC_last_row[ k ],  x2, x_ptr[ subfr_length - n + k ] ); /* Q( -rshifts ) */
                    Atmp1 = silk_RSHIFT_ROUND( pAf_QA[ k ], QA - 17 );                                   /* Q17 */
                    /* We sometimes get overflows in the multiplications (even beyond +/- 2^32),
                       but they cancel each other and the real result seems to always fit in a 32-bit
                       signed integer. This was determined experimentally, not theoretically (unfortunately). */
                    tmp1 = silk_MLA_ovflw( tmp1, x_ptr[ n - k - 1 ],            Atmp1 );                      /* Q17 */
                    tmp2 = silk_MLA_ovflw( tmp2, x_ptr[ subfr_length - n + k ], Atmp1 );                      /* Q17 */
                }


                tmp1 = -tmp1;                /* Q17 */
                tmp2 = -tmp2;                /* Q17 */

                {
                    vint32m1_t xmm_tmp1, xmm_tmp2;
                    vint32m1_t xmm_x_ptr_n_k_x2x0, xmm_x_ptr_n_k_x3x1;
                    vint32m1_t xmm_x_ptr_sub_x2x0, xmm_x_ptr_sub_x3x1;

                    xmm_tmp1 = __riscv_vmv_v_x_i32m1(tmp1,4);
                    xmm_tmp2 = __riscv_vmv_v_x_i32m1(tmp2,4);

                    for(k = 0; k < n-3; k += 4) {
                        //vl         = __riscv_vsetvl_e32m1(vn);
                        xmm_x_ptr_n_k_x2x0 = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ n - k - 3 ],4 );
                        xmm_x_ptr_sub_x2x0 = RISCV_CVTEPI16_EPI32_M64( &x_ptr[ subfr_length - n + k - 1 ],4);

                        xmm_x_ptr_n_k_x2x0 = __riscv_shuffle_epi32( xmm_x_ptr_n_k_x2x0,27,4);

                        xmm_x_ptr_n_k_x2x0 = __riscv_slli_epi32( xmm_x_ptr_n_k_x2x0, -rshifts - 1,4 );
                        xmm_x_ptr_sub_x2x0 = __riscv_slli_epi32( xmm_x_ptr_sub_x2x0, -rshifts - 1,4 );

                        /* equal shift right 4 bytes, xmm_x_ptr_n_k_x3x1 = _mm_srli_si128(xmm_x_ptr_n_k_x2x0, 4)*/
                        xmm_x_ptr_n_k_x3x1 = __riscv_shuffle_epi32( xmm_x_ptr_n_k_x2x0,57,4);
                        xmm_x_ptr_sub_x3x1 = __riscv_shuffle_epi32( xmm_x_ptr_sub_x2x0,57,4);

                        xmm_x_ptr_n_k_x2x0 = __riscv_vmul_vv_i32m1(xmm_x_ptr_n_k_x2x0, xmm_tmp1,4);
                        xmm_x_ptr_n_k_x3x1 = __riscv_vmul_vv_i32m1(xmm_x_ptr_n_k_x3x1, xmm_tmp1,4);
                        xmm_x_ptr_sub_x2x0 = __riscv_vmul_vv_i32m1(xmm_x_ptr_sub_x2x0, xmm_tmp2,4);
                        xmm_x_ptr_sub_x3x1 = __riscv_vmul_vv_i32m1(xmm_x_ptr_sub_x3x1, xmm_tmp2,4);

                        xmm_x_ptr_n_k_x2x0 = __riscv_srli_epi64( xmm_x_ptr_n_k_x2x0, 16,4);
                        xmm_x_ptr_n_k_x3x1 = __riscv_slli_epi64( xmm_x_ptr_n_k_x3x1, 16,4);
                        xmm_x_ptr_sub_x2x0 = __riscv_srli_epi64( xmm_x_ptr_sub_x2x0, 16,4);
                        xmm_x_ptr_sub_x3x1 = __riscv_slli_epi64( xmm_x_ptr_sub_x3x1, 16,4);

                        xmm_x_ptr_n_k_x2x0 = __riscv_blend_epi16( xmm_x_ptr_n_k_x2x0, xmm_x_ptr_n_k_x3x1, 0xCC,4);
                        xmm_x_ptr_sub_x2x0 = __riscv_blend_epi16( xmm_x_ptr_sub_x2x0, xmm_x_ptr_sub_x3x1, 0xCC,4);

                        X1_3210  = __riscv_vle32_v_i32m1(&pCAf[k],4);
                        PTR_3210 = __riscv_vle32_v_i32m1(&pCAb[k],4);

                        X1_3210  = __riscv_vadd_vv_i32m1(X1_3210,xmm_x_ptr_n_k_x2x0,4);
                        PTR_3210 = __riscv_vadd_vv_i32m1(PTR_3210,xmm_x_ptr_sub_x2x0,4);

                        __riscv_vse32_v_i32m1(&pCAf[k],X1_3210,4);
                        __riscv_vse32_v_i32m1(&pCAb[k],PTR_3210,4);
                    }

                    for( ; k <= n; k++ ) {
                        CAf[ k ] = silk_SMLAWW( pCAf[ k ], tmp1,
                            silk_LSHIFT32( (opus_int32)x_ptr[ n - k ], -rshifts - 1 ) );                    /* Q( -rshift ) */
                        CAb[ k ] = silk_SMLAWW( pCAb[ k ], tmp2,
                            silk_LSHIFT32( (opus_int32)x_ptr[ subfr_length - n + k - 1 ], -rshifts - 1 ) ); /* Q( -rshift ) */
                    }


                }
            }
        }

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        tmp1 = C_first_row[ n ];                                                                        /* Q( -rshifts ) */
        tmp2 = C_last_row[ n ];                                                                         /* Q( -rshifts ) */
        num  = 0;                                                                                       /* Q( -rshifts ) */
        nrg  = silk_ADD32( CAb[ 0 ], CAf[ 0 ] );                                                        /* Q( 1-rshifts ) */
        for( k = 0; k < n; k++ ) {
            Atmp_QA = Af_QA[ k ];
            lz = silk_CLZ32( silk_abs( Atmp_QA ) ) - 1;
            lz = silk_min( 32 - QA, lz );
            Atmp1 = silk_LSHIFT32( Atmp_QA, lz );                                                       /* Q( QA + lz ) */

            tmp1 = silk_ADD_LSHIFT32( tmp1, silk_SMMUL( C_last_row[  n - k - 1 ], Atmp1 ), 32 - QA - lz );  /* Q( -rshifts ) */
            tmp2 = silk_ADD_LSHIFT32( tmp2, silk_SMMUL( C_first_row[ n - k - 1 ], Atmp1 ), 32 - QA - lz );  /* Q( -rshifts ) */
            num  = silk_ADD_LSHIFT32( num,  silk_SMMUL( CAb[ n - k ],             Atmp1 ), 32 - QA - lz );  /* Q( -rshifts ) */
            nrg  = silk_ADD_LSHIFT32( nrg,  silk_SMMUL( silk_ADD32( CAb[ k + 1 ], CAf[ k + 1 ] ),
                                                                                Atmp1 ), 32 - QA - lz );    /* Q( 1-rshifts ) */
        }
        CAf[ n + 1 ] = tmp1;                                                                            /* Q( -rshifts ) */
        CAb[ n + 1 ] = tmp2;                                                                            /* Q( -rshifts ) */
        num = silk_ADD32( num, tmp2 );                                                                  /* Q( -rshifts ) */
        num = silk_LSHIFT32( -num, 1 );                                                                 /* Q( 1-rshifts ) */

        /* Calculate the next order reflection (parcor) coefficient */
        if( silk_abs( num ) < nrg ) {
            rc_Q31 = silk_DIV32_varQ( num, nrg, 31 );
        } else {
            rc_Q31 = ( num > 0 ) ? silk_int32_MAX : silk_int32_MIN;
        }

        /* Update inverse prediction gain */
        tmp1 = ( (opus_int32)1 << 30 ) - silk_SMMUL( rc_Q31, rc_Q31 );
        tmp1 = silk_LSHIFT( silk_SMMUL( invGain_Q30, tmp1 ), 2 );
        if( tmp1 <= minInvGain_Q30 ) {
            /* Max prediction gain exceeded; set reflection coefficient such that max prediction gain is exactly hit */
            tmp2 = ( (opus_int32)1 << 30 ) - silk_DIV32_varQ( minInvGain_Q30, invGain_Q30, 30 );            /* Q30 */
            rc_Q31 = silk_SQRT_APPROX( tmp2 );                                                  /* Q15 */
            if( rc_Q31 > 0 ) {
                 /* Newton-Raphson iteration */
                rc_Q31 = silk_RSHIFT32( rc_Q31 + silk_DIV32( tmp2, rc_Q31 ), 1 );                   /* Q15 */
                rc_Q31 = silk_LSHIFT32( rc_Q31, 16 );                                               /* Q31 */
                if( num < 0 ) {
                    /* Ensure adjusted reflection coefficients has the original sign */
                    rc_Q31 = -rc_Q31;
                }
            }
            invGain_Q30 = minInvGain_Q30;
            reached_max_gain = 1;
        } else {
            invGain_Q30 = tmp1;
        }

        /* Update the AR coefficients */
        for( k = 0; k < (n + 1) >> 1; k++ ) {
            tmp1 = Af_QA[ k ];                                                                  /* QA */
            tmp2 = Af_QA[ n - k - 1 ];                                                          /* QA */
            Af_QA[ k ]         = silk_ADD_LSHIFT32( tmp1, silk_SMMUL( tmp2, rc_Q31 ), 1 );      /* QA */
            Af_QA[ n - k - 1 ] = silk_ADD_LSHIFT32( tmp2, silk_SMMUL( tmp1, rc_Q31 ), 1 );      /* QA */
        }
        Af_QA[ n ] = silk_RSHIFT32( rc_Q31, 31 - QA );                                          /* QA */

        if( reached_max_gain ) {
            /* Reached max prediction gain; set remaining coefficients to zero and exit loop */
            for( k = n + 1; k < D; k++ ) {
                Af_QA[ k ] = 0;
            }
            break;
        }

        /* Update C * Af and C * Ab */
        for( k = 0; k <= n + 1; k++ ) {
            tmp1 = CAf[ k ];                                                                    /* Q( -rshifts ) */
            tmp2 = CAb[ n - k + 1 ];                                                            /* Q( -rshifts ) */
            CAf[ k ]         = silk_ADD_LSHIFT32( tmp1, silk_SMMUL( tmp2, rc_Q31 ), 1 );        /* Q( -rshifts ) */
            CAb[ n - k + 1 ] = silk_ADD_LSHIFT32( tmp2, silk_SMMUL( tmp1, rc_Q31 ), 1 );        /* Q( -rshifts ) */
        }
    }

    if( reached_max_gain ) {
        for( k = 0; k < D; k++ ) {
            /* Scale coefficients */
            A_Q16[ k ] = -silk_RSHIFT_ROUND( Af_QA[ k ], QA - 16 );
        }
        /* Subtract energy of preceding samples from C0 */
        if( rshifts > 0 ) {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                C0 -= (opus_int32)silk_RSHIFT64( silk_inner_prod16( x_ptr, x_ptr, D, arch ), rshifts );
            }
        } else {
            for( s = 0; s < nb_subfr; s++ ) {
                x_ptr = x + s * subfr_length;
                C0 -= silk_LSHIFT32( silk_inner_prod_aligned( x_ptr, x_ptr, D, arch ), -rshifts );
            }
        }
        /* Approximate residual energy */
        *res_nrg = silk_LSHIFT( silk_SMMUL( invGain_Q30, C0 ), 2 );
        *res_nrg_Q = -rshifts;
    } else {
        /* Return residual energy */
        nrg  = CAf[ 0 ];                                                                            /* Q( -rshifts ) */
        tmp1 = (opus_int32)1 << 16;                                                                             /* Q16 */
        for( k = 0; k < D; k++ ) {
            Atmp1 = silk_RSHIFT_ROUND( Af_QA[ k ], QA - 16 );                                       /* Q16 */
            nrg  = silk_SMLAWW( nrg, CAf[ k + 1 ], Atmp1 );                                         /* Q( -rshifts ) */
            tmp1 = silk_SMLAWW( tmp1, Atmp1, Atmp1 );                                               /* Q16 */
            A_Q16[ k ] = -Atmp1;
        }
        *res_nrg = silk_SMLAWW( nrg, silk_SMMUL( SILK_FIX_CONST( FIND_LPC_COND_FAC, 32 ), C0 ), -tmp1 );/* Q( -rshifts ) */
        *res_nrg_Q = -rshifts;
    }

#ifdef OPUS_CHECK_ASM
    {
        opus_int32 res_nrg_c = 0;
        opus_int res_nrg_Q_c = 0;
        opus_int32 A_Q16_c[ MAX_LPC_ORDER ] = {0};

        silk_burg_modified_c(
            &res_nrg_c,
            &res_nrg_Q_c,
            A_Q16_c,
            x,
            minInvGain_Q30,
            subfr_length,
            nb_subfr,
            D,
            0
        );

        silk_assert( *res_nrg == res_nrg_c );
        silk_assert( *res_nrg_Q == res_nrg_Q_c );
        silk_assert( !memcmp( A_Q16, A_Q16_c, D * sizeof( *A_Q16 ) ) );
    }
#endif
   

}


