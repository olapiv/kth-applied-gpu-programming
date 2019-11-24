# All Assignments for KTH-course *Applied GPU Programming*

*See course description [here](https://www.kth.se/student/kurser/kurs/DD2360?l=en).*

## Quick commands

Compile with:

```bash
nvcc -arch=sm_50 exercise_3.cu -g -o exercise_3.out  # lab machine
nvcc -arch=sm_30 exercise_3.cu -g -o exercise_3.out  # PDC supercomputer

# Optimized:
nvcc -O3 -arch=sm_50 exercise_3.cu -o exercise_3.out  # lab machine
nvcc -O3 -arch=sm_30 exercise_3.cu -o exercise_3.out  # PDC supercomputer
```

Load nvc compiler:

```bash
export PATH=/usr/local/cuda/bin:$PATH  # lab machine
module add cuda/8.0  # PDC supercomputer
```
