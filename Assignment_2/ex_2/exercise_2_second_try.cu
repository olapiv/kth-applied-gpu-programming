#include <stdio.h>
#include <iostream>
using namespace std;
#define TPB 256
#define ARRAY_SIZE 10
#define N (ARRAY_SIZE/TPB + 1)

// __device__ float scale(int i, int n)
// {
//   return ((float)i)/(n - 1);
// }
//
// __device__ float distance(float x1, float x2)
// {
//   return sqrt((x2 - x1)*(x2 - x1));
// }
//
 __global__ void saxpy(float *x, float *y, const float a)
 {

   const int i = blockIdx.x*blockDim.x + threadIdx.x;

   if (i<ARRAY_SIZE) {
      y[i] = a*x[i] + y[i];
   }
}

int main()
{

  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(ARRAY_SIZE*sizeof(float));
  y = (float*)malloc(ARRAY_SIZE*sizeof(float));
  const int a = 3;

  cudaMalloc(&d_x, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&d_y, ARRAY_SIZE*sizeof(float));

  for (int i = 0; i < ARRAY_SIZE; i++) {
    x[i] = rand() % 1000;
    y[i] = rand() % 1000;
        cout << x[i] << "\n";
            cout << y[i] << "\n\n";
  }
  cout << "---------------------" <<"\n";

  cudaMemcpy(d_x, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  saxpy<<<N, TPB>>>(d_x, d_y, a);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < ARRAY_SIZE; i++) {
    cout << x[i] << "\n";
    cout << y[i] << "\n\n";
  }

  free(x);
  free(y);
  cudaFree(d_x);
  cudaFree(d_y);




  // float x_array[ARRAY_SIZE];
  // float y_array_gpu[ARRAY_SIZE];
  // float y_array_cpu[ARRAY_SIZE];
  // const int a = 0;
  //
  // cout << x_array << "\n";
  // cout << &x_array << "\n";
  // cout << *x_array << "\n";
  //
  // for(int i = 0; i < ARRAY_SIZE; i++)
  // {
  //   x_array[i] = rand();
  //   // printf("values %lf\n", x_array[i]);
  // }
  //
  // cout << x_array << "\n";
  // cout << &x_array << "\n";
  // cout << *x_array << "\n";
  //
  // cudaMalloc(&x_array, ARRAY_SIZE * sizeof(float));
  // // cudaMalloc(&y_array_gpu, ARRAY_SIZE * sizeof(float));
  //
  // free(x_array);
  // // free(y_array_gpu);
  //
  //
  // float *x_cpu = NULL, *y_cpu, *x_gpu = NULL, *y_gpu = NULL;
  //
  //
  // float *d_out = 0;
  // cout << d_out << "\n";
  // cout << &d_out << "\n";
  // cout << *d_out << "\n";

  // Declare a pointer for an array of floats
  //float *d_out = 0;

  // y_cpu = (float*) malloc(ARRAY_SIZE * sizeof(float));
  // x_cpu = (float*) malloc(ARRAY_SIZE * sizeof(float));

  // cudaMalloc(&x_gpu, ARRAY_SIZE * sizeof(float));
  // cudaMalloc(&y_gpu, ARRAY_SIZE * sizeof(float));

  // for(int i = 0; i < ARRAY_SIZE; i++)
  // {
  //   x_cpu[i] = rand();
  //   y_cpu[i] = rand();
  //   x_gpu[i] = rand();
  //   y_gpu[i] = rand();
  //   printf("values%lf", x_cpu[i]);
  //
  // }
  // Allocate device memory to store the output array

  // Launch kernel to compute and store distance values
  //mainKernel<<<N, TPB>>>(x_gpu, y_gpu, a);

  // free(x_cpu);
  // free(y_cpu);
  // cudaFree(x_gpu);
  // cudaFree(y_gpu); // Free the memory
  return 0;
}
