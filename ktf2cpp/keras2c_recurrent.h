#ifndef KERAS2C_RECURRENT_H
#define KERAS2C_RECURRENT_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "keras2c_helper_functions.h"

void lstmcell(float inputs[], float states[], float Wi[],
	      float Wf[], float Wc[], float Wo[], float bi[],
	      float bf[], float bc[], float bo[], float Ui[],
	      float Uf[], float Uc[], float Uo[], size_t units,
	      size_t in_height, size_t in_width, float xi[],
	      float xf[], float xc[], float xo[], float yi[],
	      float yf[], float yc[], float yo[],
	      void (*recurrent_activation) (float*, size_t),
	      void (*output_activation)(float*, size_t)){

  //printf("in lstmcell \n");

  float *h_tm1 = &states[0];  // previous memory state
  float *c_tm1 = &states[units];  // previous carry state
  size_t xoutrows = 1;
  size_t youtrows = 1;

  /* printf("in lstm cell, h_tm1: \n"); */
  /* for (int i=0; i<units;i++){ */
  /*   printf("%.2f  ", h_tm1[i]);} */
  /* printf("\n"); */
  /* printf("in lstm cell, c_tm1: \n"); */
  /* for (int i=0; i<units;i++){ */
  /*   printf("%.2f  ", c_tm1[i]);} */
  /* printf("\n"); */

  //xi = inputs*Wi + bi;
  affine_matmul(xi, inputs, Wi, bi, xoutrows, units, in_width);
  //xf = inputs*Wf + bf;
  affine_matmul(xf, inputs, Wf, bf, xoutrows, units, in_width);
  //xc = inputs*Wc + bc;
  affine_matmul(xc, inputs, Wc, bc, xoutrows, units, in_width);
  //xo = inputs*Wo + bo;
  affine_matmul(xo, inputs, Wo, bo, xoutrows, units, in_width);
 
  // yi = recurrent_activation(xi + h_tm1*Ui);
  affine_matmul(yi, h_tm1, Ui, xi, youtrows, units, units);
  recurrent_activation(yi, units);

  // yf = recurrent_activation(xf + h_tm1*Uf); 
  affine_matmul(yf, h_tm1, Uf, xf, youtrows, units, units);
  recurrent_activation(yf, units);

  // yc = yf.*c_tm1 + yi.*output_activation(xc + h_tm1*Uc);
  affine_matmul(yc, h_tm1, Uc, xc, youtrows, units, units);
  output_activation(yc, units);
  for (size_t i=0; i < units; i++){
    yc[i] = yf[i]*c_tm1[i] + yi[i]*yc[i];}

  
  // yo = recurrent_activation(xo + h_tm1*Uo); 
  affine_matmul(yo, h_tm1, Uo, xo, youtrows, units, units);
  recurrent_activation(yo, units);

  // h = yo.*output_activation(yc); 
  // state = [h;yc];
  for (size_t i=0; i < units; i++){
    states[units+i] = yc[i];}

  output_activation(yc, units);

  for (size_t i=0; i < units; i++){
    states[i] = yo[i]*yc[i];}

}

void lstm(float inputs[], float states[], float Wi[], float Wf[],
	  float Wc[], float Wo[], float bi[], float bf[], float bc[],
	  float bo[], float Ui[], float Uf[], float Uc[], float Uo[],
	  size_t units, size_t in_height, size_t in_width, float xi[],
	  float xf[], float xc[], float xo[], float yi[], float yf[],
	  float yc[], float yo[],
	  void (*recurrent_activation) (float*, size_t),
	  void (*output_activation)(float*, size_t), float *outputs){

  // printf("in lstm \n");
  /*     printf("start lstm, inputs: \n"); */
  /* for (int i=0; i<in_height*in_width;i++){ */
  /*   printf("%.2f  ", inputs[i]);} */
  /* printf("\n"); */
    
    
  for (size_t i=0; i < in_height; i++){
    lstmcell(&inputs[i*in_width], states, Wi, Wf, Wc, Wo, bi, bf, bc, bo,
	     Ui, Uf, Uc, Uo, units, in_height, in_width, xi, xf, xc, xo,
	     yi, yf, yc, yo, recurrent_activation, output_activation);
  }
  for (size_t i=0; i < units; i++){
    outputs[i] = states[i];
  }
}

#endif /* KERAS2C_RECURRENT_H */
