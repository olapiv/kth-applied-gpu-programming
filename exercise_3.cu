#include <stdio.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;
#define TPB 256
#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 1000
#define N (NUM_PARTICLES/TPB + 1)


struct particle {
    float position[3];
    float velocity[3];
};

struct seed {
    int x;
    int y;
    int z;
};

__host__ __device__ float gen_random(int seed, int particle_id, int iteration)
{
  float rand_num = (seed * particle_id + iteration) % NUM_PARTICLES;
  // printf("seed = %d, particle_id = %d, iteration = %d, rand_num = %e\n",
  //   seed,
  //   particle_id,
  //   iteration,
  //   rand_num);
  return rand_num;
}

 __global__ void timestep(particle *particles, seed seed, int iteration) {
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   if (i<NUM_PARTICLES) {
      // printf("Old threadId = %d, velocity.x = %e, position.x = %e\n",
      //    threadIdx.x, particles[i].velocity[0], particles[i].position[0]);

     // Velocity update:
      particles[i].velocity[0] = gen_random(seed.x, i, iteration);
      particles[i].velocity[1] = gen_random(seed.y, i, iteration);
      particles[i].velocity[2] = gen_random(seed.z, i, iteration);

      // Position update:
      particles[i].position[0] = particles[i].position[0] + particles[i].velocity[0];
      particles[i].position[1] = particles[i].position[1] + particles[i].velocity[1];
      particles[i].position[2] = particles[i].position[2] + particles[i].velocity[2];

      // printf("New threadId = %d, velocity.x = %e\n", threadIdx.x, particles[i].velocity[0]);
   }
}

int main()
{
  seed seed = {5,6,7};
  auto start = high_resolution_clock::now();
  particle *particlesCPU = new particle[NUM_PARTICLES];
  particle *particlesGPU = new particle[NUM_PARTICLES];

  cudaMalloc(&particlesGPU, NUM_PARTICLES*6*sizeof(float));

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    // cout << "iteration: " << i <<"\n";
    timestep<<<N, TPB>>>(particlesGPU, seed, i);
    // cudaDeviceSynchronize();
  }

  cudaMemcpy(particlesCPU, particlesGPU, NUM_PARTICLES*6*sizeof(float), cudaMemcpyDeviceToHost);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Duration in microseconds: " << duration.count() << endl;

  // Print output:
  // for (int ii = 0; ii < NUM_PARTICLES; ii++) {
  //   cout << particlesCPU[ii].position[0] << "\n";
  // }

  delete[] particlesCPU;
  cudaFree(particlesGPU);

  return 0;
}
