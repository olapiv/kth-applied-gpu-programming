#include <stdio.h>
#include <iostream>
using namespace std;
#define TPB 256
#define ARRAY_SIZE 10
#define N (ARRAY_SIZE/TPB + 1)


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

  return 0;
}
