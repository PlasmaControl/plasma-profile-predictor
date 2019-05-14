#include <stdio.h>

int main()
{
  /* printf("hello world"); */
  /*  printf("%f",fmax(1.0,0.0)); */
  double inputs[256] = {0};
  double outputs[30] = {0};
  profile_predictor(inputs, outputs);
    for (int i=0 ; i<30 ; i++)
      printf("%f\n", outputs[i]);
   return 0;

}
