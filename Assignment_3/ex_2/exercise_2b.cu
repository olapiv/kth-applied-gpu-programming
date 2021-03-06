#include <stdio.h>
#include <iostream>
#include <chrono>
#include <math.h>
using namespace std::chrono;
using namespace std;
#define TPB 256
#define NUM_PARTICLES 100000
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

__host__ __device__ void updateVelAndPos(particle *particles, seed seed, int iteration, int particle_id)
{
  // Velocity update:
   particles[particle_id].velocity[0] = gen_random(seed.x, particle_id, iteration);
   particles[particle_id].velocity[1] = gen_random(seed.y, particle_id, iteration);
   particles[particle_id].velocity[2] = gen_random(seed.z, particle_id, iteration);

   // Position update:
   particles[particle_id].position[0] = particles[particle_id].position[0] + particles[particle_id].velocity[0];
   particles[particle_id].position[1] = particles[particle_id].position[1] + particles[particle_id].velocity[1];
   particles[particle_id].position[2] = particles[particle_id].position[2] + particles[particle_id].velocity[2];
}

 __global__ void timestepGPU(particle *particles, seed seed, int iteration) {
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   if (i<NUM_PARTICLES) {
      // printf("Old threadId = %d, velocity.x = %e, position.x = %e\n",
      //    threadIdx.x, particles[i].velocity[0], particles[i].position[0]);

      updateVelAndPos(particles, seed, iteration, i);

      // printf("New threadId = %d, velocity.x = %e\n", threadIdx.x, particles[i].velocity[0]);
   }
}

void timestepCPU(particle *particles, seed seed, int iteration) {
  for (int i = 0; i < NUM_PARTICLES; i++) {
    updateVelAndPos(particles, seed, iteration, i);
  }
}

int main()
{
  seed seed = {5,6,7};

  particle* particlesSharedCPU = NULL;
  cudaMallocManaged(&particlesSharedCPU, NUM_PARTICLES * sizeof(particle));
  particle* particlesSharedGPU = NULL;
  cudaMallocManaged(&particlesSharedGPU, NUM_PARTICLES * sizeof(particle));

  //////// CPU calculations ////////
  auto startCPU = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    // cout << "iteration: " << i <<"\n";
    timestepCPU(particlesSharedCPU, seed, i);
  }

  //Print output:
  // for (int ii = 0; ii < 10; ii++) {
  //   cout << particlesCPU[ii].position[0] << "\n";
  // }

  auto stopCPU = high_resolution_clock::now();
  auto durationCPU = duration_cast<milliseconds>(stopCPU - startCPU);
  cout << "---------------\n";
  //////////////////////////////////

  //////// GPU calculations ////////
  auto startGPU = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERATIONS; i++) {

    // cout << "iteration: " << i <<"\n";
    timestepGPU<<<N, TPB>>>(particlesSharedGPU, seed, i);
    cudaDeviceSynchronize();

  }

  // Print output:
  for (int ii = 0; ii < 10; ii++) {
    cout << particlesSharedGPU[ii].position[0] << "\n";
  }

  auto stopGPU = high_resolution_clock::now();
  auto durationGPU = duration_cast<milliseconds>(stopGPU - startGPU);
  //////////////////////////////////

  cudaFree(particlesSharedCPU);
  cudaFree(particlesSharedGPU);

  cout << "CPU duration in milliseconds: " << durationCPU.count() << endl;
  cout << "GPU duration in milliseconds: " << durationGPU.count() << endl;

  return 0;
}
