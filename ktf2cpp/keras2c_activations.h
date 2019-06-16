#ifndef KERAS2C_ACTIVATIONS_H
#define KERAS2C_ACTIVATIONS_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>

void linear(float x[], size_t size){
  /* linear activation. Doesn't do anything, just a blank fn */

}

void relu(float x[], size_t size) {
  /* Rectified Linear Unit activation (ReLU) */
  /*   y = max(x,0) */
  /* x is overwritten with the activated values */

  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = 0.0f;
    }
  }
}

void hard_sigmoid(float x[], size_t size) {
  /* Hard Sigmoid activation */
  /*   y = {1 if x> 2.5 */
  /*        0.2*x+0.5 if -2.5<x<2.5 */
  /*        0 if x<2.5 */
  /* x is overwritten with the activated values */

  for (size_t i=0; i < size; i++) {
    if (x[i] <= -2.5f){
      x[i] = 0.0f;
    }
    else if (x[i]>=2.5f) {
      x[i] = 1.0f;
    }
    else {
      x[i] = 0.2f*x[i] + 0.5f;
    }
  }
}

void arrtanh(float x[], size_t size) {
  /* standard tanh activation */
  /* x is overwritten with the activated values */
  for (size_t i=0; i<size; i++){
    x[i] = tanh(x[i]);
  }
}

void sigmoid(float x[], size_t size) {
  /* Sigmoid activation */
  /*   y = 1/(1+exp(-x)) */
 /* x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = 1/(1+exp(-x[i]));
  }
}

void softmax(float *x, size_t size) {
/* Softmax activation */
/*     z[i] = exp(x[i]-max(x)) */
/*     y = z/sum(z) */
 /* x is overwritten with the activated values */
  float xmax = x[0];
  float sum = 0;
  for (size_t i=0; i < size; i++) {
    if (x[i]>xmax) {
      xmax = x[i];}
  }
  for (size_t i=0; i < size; i++) {
    x[i] = exp(x[i]-xmax);
  }
  for (size_t i=0; i < size; i++) {
    sum += x[i];
  }
  sum = 1.0f/sum; // divide once and multiply -> fast
  for (size_t i=0; i < size; i++) {
    x[i] = x[i]*sum;
  }
}

void softplus(float x[], size_t size) {
  /* Softplus activation */
  /*   y = ln(1+exp(x)) */
  /*   x is overwritten with the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = log(1.0f + exp(x[i]));
  }
}

void softsign(float x[], size_t size) {
  /* Softsign activation */
  /*   y = x/(1+|x|) */
  /*   x is overwritten by the activated values */
  for (size_t i=0; i < size; i++) {
    x[i] = x[i]/(1.0f + fabs(x[i]));
  }
}

void elu(float x[], size_t size) {
  /* Exponential Linear Unit activation (ELU) */
  /*   y = {x if x>0 */
  /* 	 alpha*(e^x - 1) else} */
  /* x is overwritten with the activated values */

  float alpha = 1.0f; // change this if needed
    
  for (size_t i=0; i < size; i++) {
    if (x[i] <= 0.0f){
      x[i] = alpha*(exp(x[i])-1.0f);
    }
  }
}

#endif /* KERAS2C_ACTIVATIONS_H */
