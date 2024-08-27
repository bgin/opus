
#include <riscv_vector.h>
#include "include/opus_sse2rvv_hal.h"
#include "main.h"
#include "stack_alloc.h"

typedef struct {
    opus_int32 sLPC_Q14[ MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH ];
    opus_int32 RandState[ DECISION_DELAY ];
    opus_int32 Q_Q10[     DECISION_DELAY ];
    opus_int32 Xq_Q14[    DECISION_DELAY ];
    opus_int32 Pred_Q15[  DECISION_DELAY ];
    opus_int32 Shape_Q14[ DECISION_DELAY ];
    opus_int32 sAR2_Q14[ MAX_SHAPE_LPC_ORDER ];
    opus_int32 LF_AR_Q14;
    opus_int32 Diff_Q14;
    opus_int32 Seed;
    opus_int32 SeedInit;
    opus_int32 RD_Q10;
} NSQ_del_dec_struct;

typedef struct {
    opus_int32 Q_Q10;
    opus_int32 RD_Q10;
    opus_int32 xq_Q14;
    opus_int32 LF_AR_Q14;
    opus_int32 Diff_Q14;
    opus_int32 sLTP_shp_Q14;
    opus_int32 LPC_exc_Q14;
} NSQ_sample_struct;

typedef NSQ_sample_struct  NSQ_sample_pair[ 2 ];


__attribute__((always_inline))
static OPUS_INLINE void silk_nsq_del_dec_scale_states_rvv(
    const silk_encoder_state * __restrict__ psEncC,               /* I    Encoder State                       */
    silk_nsq_state      * __restrict__ NSQ,                       /* I/O  NSQ state                           */
    NSQ_del_dec_struct  psDelDec[],                 /* I/O  Delayed decision states             */
    const opus_int16    x16[],                      /* I    Input                               */
    opus_int32          x_sc_Q10[],                 /* O    Input scaled with 1/Gain in Q10     */
    const opus_int16    sLTP[],                     /* I    Re-whitened LTP state in Q0         */
    opus_int32          sLTP_Q15[],                 /* O    LTP state matching scaled input     */
    opus_int            subfr,                      /* I    Subframe number                     */
    opus_int            nStatesDelayedDecision,     /* I    Number of del dec states            */
    const opus_int      LTP_scale_Q14,              /* I    LTP state scaling                   */
    const opus_int32    Gains_Q16[ MAX_NB_SUBFR ],  /* I                                        */
    const opus_int      pitchL[ MAX_NB_SUBFR ],     /* I    Pitch lag                           */
    const opus_int      signal_type,                /* I    Signal type                         */
    const opus_int      decisionDelay               /* I    Decision delay                      */
);

/******************************************/
/* Noise shape quantizer for one subframe */
/******************************************/

__attribute__((always_inline))
static OPUS_INLINE void silk_noise_shape_quantizer_del_dec_rvv(
    silk_nsq_state      * __restrict__ NSQ,                   /* I/O  NSQ state                           */
    NSQ_del_dec_struct  psDelDec[],             /* I/O  Delayed decision states             */
    opus_int            signalType,             /* I    Signal type                         */
    const opus_int32    x_Q10[],                /* I                                        */
    opus_int8           pulses[],               /* O                                        */
    opus_int16          xq[],                   /* O                                        */
    opus_int32          sLTP_Q15[],             /* I/O  LTP filter state                    */
    opus_int32          delayedGain_Q10[],      /* I/O  Gain delay buffer                   */
    const opus_int16    a_Q12[],                /* I    Short term prediction coefs         */
    const opus_int16    b_Q14[],                /* I    Long term prediction coefs          */
    const opus_int16    AR_shp_Q13[],           /* I    Noise shaping coefs                 */
    opus_int            lag,                    /* I    Pitch lag                           */
    opus_int32          HarmShapeFIRPacked_Q14, /* I                                        */
    opus_int            Tilt_Q14,               /* I    Spectral tilt                       */
    opus_int32          LF_shp_Q14,             /* I                                        */
    opus_int32          Gain_Q16,               /* I                                        */
    opus_int            Lambda_Q10,             /* I                                        */
    opus_int            offset_Q10,             /* I                                        */
    opus_int            length,                 /* I    Input length                        */
    opus_int            subfr,                  /* I    Subframe number                     */
    opus_int            shapingLPCOrder,        /* I    Shaping LPC filter order            */
    opus_int            predictLPCOrder,        /* I    Prediction filter order             */
    opus_int            warping_Q16,            /* I                                        */
    opus_int            nStatesDelayedDecision, /* I    Number of states in decision tree   */
    opus_int            * __restrict__ smpl_buf_idx,          /* I/O  Index to newest samples in buffers  */
    opus_int            decisionDelay           /* I                                        */
);
