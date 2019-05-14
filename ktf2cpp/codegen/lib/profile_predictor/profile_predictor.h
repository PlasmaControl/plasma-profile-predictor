/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * profile_predictor.h
 *
 * Code generation for function 'profile_predictor'
 *
 */

#ifndef PROFILE_PREDICTOR_H
#define PROFILE_PREDICTOR_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "profile_predictor_types.h"

/* Function Declarations */
extern void profile_predictor(const double input[256], double prediction[30]);
extern void profile_predictor_initialize(void);
extern void profile_predictor_terminate(void);

#endif

/* End of code generation (profile_predictor.h) */
