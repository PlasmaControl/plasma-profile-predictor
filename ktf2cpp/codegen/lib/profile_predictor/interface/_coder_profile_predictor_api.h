/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_profile_predictor_api.h
 *
 * Code generation for function '_coder_profile_predictor_api'
 *
 */

#ifndef _CODER_PROFILE_PREDICTOR_API_H
#define _CODER_PROFILE_PREDICTOR_API_H

/* Include files */
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"
#include <stddef.h>
#include <stdlib.h>
#include "_coder_profile_predictor_api.h"

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void profile_predictor(real_T input[256], real_T prediction[30]);
extern void profile_predictor_api(const mxArray * const prhs[1], int32_T nlhs,
  const mxArray *plhs[1]);
extern void profile_predictor_atexit(void);
extern void profile_predictor_initialize(void);
extern void profile_predictor_terminate(void);
extern void profile_predictor_xil_shutdown(void);
extern void profile_predictor_xil_terminate(void);

#endif

/* End of code generation (_coder_profile_predictor_api.h) */
