#include "EMfield.h"


/** allocate electric and magnetic field */
void field_allocate_gpu(struct grid* grd, struct EMfield* field, struct EMfield* fieldGPU)
{

    FPfield *dev_fieldEx, *dev_fieldEy, *dev_fieldEz, *dev_fieldBxn, *dev_fieldByn, *dev_fieldBzn;

    cudaMalloc(&dev_fieldEx, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldEy, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldEz, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldBxn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldByn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&dev_fieldBzn, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    cudaMemcpy(dev_fieldEx, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldEy, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldEz, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldBxn, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldByn, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fieldBzn, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMemcpy(&(fieldGPU->Ex), &dev_fieldEx, sizeof(fieldGPU->Ex), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Ey), &dev_fieldEy, sizeof(fieldGPU->Ey), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Ez), &dev_fieldEz, sizeof(fieldGPU->Ez), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Bxn), &dev_fieldBxn, sizeof(fieldGPU->Bxn), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Byn), &dev_fieldByn, sizeof(fieldGPU->Byn), cudaMemcpyHostToDevice);
    cudaMemcpy(&(fieldGPU->Bzn), &dev_fieldBzn, sizeof(fieldGPU->Bzn), cudaMemcpyHostToDevice);

}

/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field)
{
    // E on nodes
    field->Ex  = newArr3<FPfield>(&field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ey  = newArr3<FPfield>(&field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ez  = newArr3<FPfield>(&field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);
    // B on nodes
    field->Bxn = newArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Byn = newArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Bzn = newArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field)
{
    // E deallocate 3D arrays
    delArr3(field->Ex, grd->nxn, grd->nyn);
    delArr3(field->Ey, grd->nxn, grd->nyn);
    delArr3(field->Ez, grd->nxn, grd->nyn);

    // B deallocate 3D arrays
    delArr3(field->Bxn, grd->nxn, grd->nyn);
    delArr3(field->Byn, grd->nxn, grd->nyn);
    delArr3(field->Bzn, grd->nxn, grd->nyn);
}
