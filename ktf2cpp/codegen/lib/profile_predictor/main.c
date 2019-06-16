#include <stdio.h>
#include <time.h>
#include <math.h>

int main()
{
  /* printf("hello world"); */
  /*  printf("%f",fmax(1.0,0.0)); */
  float inputs[256] = {0};
  float outputs[30] = {0};
  clock_t start, end, start1;
  double cpu_time_used;
  double cpu_times[100000];
  double avg_time = 0;
  double std_time = 0;
  double max_time = 0;
  
     start = clock();
     for (int i=0; i<100000; i++)
     {
       start1 = clock();
       profile_predictor(inputs, outputs);
       end = clock();
       cpu_times[i] = ((double) (end - start1)) / CLOCKS_PER_SEC;
     }
     end = clock();
     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
     
     avg_time = cpu_time_used / 100000.0;
     max_time = cpu_times[0];
     for (int i=0; i<1000; i++)
     {
       std_time += (cpu_times[i] - avg_time)*(cpu_times[i] - avg_time);
       if (cpu_times[i] > max_time)
	 {max_time = cpu_times[i];}
     }
     std_time = sqrt(std_time/99999.0);
     
     printf("Avg time per call: \n");
     printf("%f  ", avg_time*1000000.0);
     printf("us \n");

     printf("std time per call: \n");
     printf("%f  ", std_time*1000000.0);
     printf("us \n");
     
     printf("Worst time per call: \n");
     printf("%f  ", max_time*1000000.0);
     printf("us \n");
     
     /* for (int i=0 ; i<1000 ; i++) */
     /*   printf("%f\n", cpu_times[i]*1000000.0); */
   return 0;

}
