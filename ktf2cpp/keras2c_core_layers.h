#ifndef KERAS2C_CORE_LAYERS_H
#define KERAS2C_CORE_LAYERS_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include "keras2c_helper_functions.h"

void dense(float output[], float input[], float kernel[], float bias[],
	   size_t outrows, size_t outcols, size_t innerdim,
	   void (*activation) (float[], size_t)){
  //  printf("in dense \n");
  size_t outsize = outrows*outcols;
  affine_matmul(output,input,kernel,bias,outrows,outcols,innerdim);
  activation(output,outsize);
}



#endif /* KERAS2C_CORE_LAYERS_H */
