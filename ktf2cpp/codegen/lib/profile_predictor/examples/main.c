/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * main.c
 *
 * Code generation for function 'main'
 *
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/
/* Include files */
#include "profile_predictor.h"
#include "main.h"

/* Function Declarations */
static void argInit_8x32_real32_T(float result[256]);
static float argInit_real32_T(void);
static void main_profile_predictor(void);

/* Function Definitions */
static void argInit_8x32_real32_T(float result[256])
{
  int idx0;
  int idx1;

  /* Loop over the array to initialize each element. */
  for (idx0 = 0; idx0 < 8; idx0++) {
    for (idx1 = 0; idx1 < 32; idx1++) {
      /* Set the value of the array element.
         Change this value to the value that the application requires. */
      result[idx1 + (idx0 << 5)] = argInit_real32_T();
    }
  }
}

static float argInit_real32_T(void)
{
  return 0.0F;
}

static void main_profile_predictor(void)
{
  float fv17[256];
  float prediction[30];

  /* Initialize function 'profile_predictor' input arguments. */
  /* Initialize function input argument 'input'. */
  /* Call the entry-point 'profile_predictor'. */
  argInit_8x32_real32_T(fv17);
  profile_predictor(fv17, prediction);
}

int main(int argc, const char * const argv[])
{
  (void)argc;
  (void)argv;

  /* Initialize the application.
     You do not need to do this more than one time. */
  profile_predictor_initialize();

  /* Invoke the entry-point functions.
     You can call entry-point functions multiple times. */
  main_profile_predictor();

  /* Terminate the application.
     You do not need to do this more than one time. */
  profile_predictor_terminate();
  return 0;
}

/* End of code generation (main.c) */
