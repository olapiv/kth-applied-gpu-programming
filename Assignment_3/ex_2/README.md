# Exercise 2 - Pinned and Managed Memory

## Assignment instructions for code

Reuse your solution from Assignment 2 - Exercise 3 (particle mover) for this exercise.

### Exercise 2a - Pinned Memory

1. Modify the program, such that
   1. All particles are copied to the GPU at the beginning of a time step.
   2. All the particles are copied back to the host after the kernel completes, before proceeding to the next time step.
2. Use nvprof to study the time spent on data movement and actual computation, with a large number of particles that can the GPU.
3. Change the appropriate memory allocator to use cudaMallocHost().
4. Use nvprof to study the time spent on data movement and actual computation, with a large number of particles that can fill the GPU memory. Also, note for the time spent on allocation.

### Exercise 2b - Managed Memory

1. Change the GPU memory allocators to use cudaMallocManaged().
1. Eliminate explicit data copy and device pointers.
1. Study the breakdown of timing using nvprof.

## Q&A for report

