# All Assignments for KTH-course Applied GPU Programming

*See course description [here](https://www.kth.se/student/kurser/kurs/DD2360?l=en).*

## Quick commands

Switch to Lustre (cfs) work-space (instead of AFS) on PDC-supercomputer:

```bash
cd /cfs/klemming/scratch/(initial of username)/(username)/
```

Prepare Environment:

```bash
# Lab machine:
export PATH=/usr/local/cuda/bin:$PATH  # Load nvcc command

# PDC supercomputer:
module add cuda/8.0
module load gcc/6.2.0
module load openmpi/3.0-gcc-6.2
module load cuda/10.0
module load pgi
```

Compile with:

```bash
nvcc -arch=sm_50 exercise_3.cu -g -o exercise_3.out  # lab machine
nvcc -arch=sm_30 exercise_3.cu -g -o exercise_3.out  # PDC supercomputer

# Optimized:
nvcc -O3 -arch=sm_50 exercise_3.cu -o exercise_3.out  # lab machine
nvcc -O3 -arch=sm_30 exercise_3.cu -o exercise_3.out  # PDC supercomputer
```

Request node:

```bash
salloc --nodes=1 -t 01:00:00 -A edu19.DD2360 --reservation=reservation_here
```

Run program:

Important - do not run program on entry node. It has no CUDA-capable device, so __device__ functions are simply not called and nvprof will return an error.
Using srun will make sure that a node on Tegner is used, which has a CUDA-capable device.

```bash
srun -n 1 ./exercise_3.out

# With profiling:
srun -n 1 nvprof ./exercise_3.out
```

## Detecting errors

* If __ device __ functions are not being called, try implementing [this](https://stackoverflow.com/questions/21990904/cuda-global-function-not-called).
