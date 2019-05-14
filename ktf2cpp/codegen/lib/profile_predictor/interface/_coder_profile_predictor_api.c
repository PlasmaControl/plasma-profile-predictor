/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_profile_predictor_api.c
 *
 * Code generation for function '_coder_profile_predictor_api'
 *
 */

/* Include files */
#include "tmwtypes.h"
#include "_coder_profile_predictor_api.h"
#include "_coder_profile_predictor_mex.h"

/* Variable Definitions */
emlrtCTX emlrtRootTLSGlobal = NULL;
emlrtContext emlrtContextGlobal = { true,/* bFirstTime */
  false,                               /* bInitialized */
  131482U,                             /* fVersionInfo */
  NULL,                                /* fErrorFunction */
  "profile_predictor",                 /* fFunctionName */
  NULL,                                /* fRTCallStack */
  false,                               /* bDebugMode */
  { 2045744189U, 2170104910U, 2743257031U, 4284093946U },/* fSigWrd */
  NULL                                 /* fSigMem */
};

/* Function Declarations */
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, real_T y[256]);
static void c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId, real_T ret[256]);
static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *input, const
  char_T *identifier, real_T y[256]);
static const mxArray *emlrt_marshallOut(const real_T u[30]);

/* Function Definitions */
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId, real_T y[256])
{
  real_T dv0[256];
  int32_T i0;
  int32_T i1;
  c_emlrt_marshallIn(sp, emlrtAlias(u), parentId, dv0);
  for (i0 = 0; i0 < 8; i0++) {
    for (i1 = 0; i1 < 32; i1++) {
      y[i1 + (i0 << 5)] = dv0[i0 + (i1 << 3)];
    }
  }

  emlrtDestroyArray(&u);
}

static void c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId, real_T ret[256])
{
  static const int32_T dims[2] = { 8, 32 };

  real_T (*r0)[256];
  int32_T i3;
  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 2U, dims);
  r0 = (real_T (*)[256])emlrtMxGetData(src);
  for (i3 = 0; i3 < 256; i3++) {
    ret[i3] = (*r0)[i3];
  }

  emlrtDestroyArray(&src);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *input, const
  char_T *identifier, real_T y[256])
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = (const char *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  b_emlrt_marshallIn(sp, emlrtAlias(input), &thisId, y);
  emlrtDestroyArray(&input);
}

static const mxArray *emlrt_marshallOut(const real_T u[30])
{
  const mxArray *y;
  const mxArray *m0;
  static const int32_T iv0[2] = { 1, 30 };

  real_T *pData;
  int32_T i2;
  int32_T i;
  y = NULL;
  m0 = emlrtCreateNumericArray(2, iv0, mxDOUBLE_CLASS, mxREAL);
  pData = emlrtMxGetPr(m0);
  i2 = 0;
  for (i = 0; i < 30; i++) {
    pData[i2] = u[i];
    i2++;
  }

  emlrtAssign(&y, m0);
  return y;
}

void profile_predictor_api(const mxArray * const prhs[1], int32_T nlhs, const
  mxArray *plhs[1])
{
  real_T input[256];
  real_T prediction[30];
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  (void)nlhs;
  st.tls = emlrtRootTLSGlobal;

  /* Marshall function inputs */
  emlrt_marshallIn(&st, emlrtAliasP(prhs[0]), "input", input);

  /* Invoke the target function */
  profile_predictor(input, prediction);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(prediction);
}

void profile_predictor_atexit(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  profile_predictor_xil_terminate();
  profile_predictor_xil_shutdown();
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void profile_predictor_initialize(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, 0);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

void profile_predictor_terminate(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (_coder_profile_predictor_api.c) */
