# All Assignments for KTH-course *Applied GPU Programming*

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
module add cuda/8.0  # PDC supercomputer
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

```bash
srun -n 1 ./exercise_3.out

# With profiling:
srun nvprof -n 1 ./exercise_3.out
```
