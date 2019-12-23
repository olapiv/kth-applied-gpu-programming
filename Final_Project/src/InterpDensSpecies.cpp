#include "InterpDensSpecies.h"


/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, struct interpDensSpecies* idsGPU)
{
    FPinterp *dev_ids_rhon, *dev_ids_rhoc, *dev_ids_Jx, *dev_ids_Jy, *dev_ids_Jz, *dev_ids_pxx, *dev_ids_pxy, *dev_ids_pxz, *dev_ids_pyy, *dev_ids_pyz, *dev_ids_pzz;

    size_t size_array_nodes = grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp);
    size_t size_array_cells = grd->nxc * grd->nyc * grd->nzc * sizeof(FPinterp);

    cudaMalloc(&dev_ids_rhon, size_array_nodes);
    cudaMalloc(&dev_ids_rhoc, size_array_cells);
    cudaMalloc(&dev_ids_Jx, size_array_nodes);
    cudaMalloc(&dev_ids_Jy, size_array_nodes);
    cudaMalloc(&dev_ids_Jz, size_array_nodes);
    cudaMalloc(&dev_ids_pxx, size_array_nodes);
    cudaMalloc(&dev_ids_pxy, size_array_nodes);
    cudaMalloc(&dev_ids_pxz, size_array_nodes);
    cudaMalloc(&dev_ids_pyy, size_array_nodes);
    cudaMalloc(&dev_ids_pyz, size_array_nodes);
    cudaMalloc(&dev_ids_pzz, size_array_nodes);

    cudaMemcpy(dev_ids_rhon, ids->rhon_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_rhoc, ids->rhoc_flat, size_array_cells, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_Jx, ids->Jx_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_Jy, ids->Jy_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_Jz, ids->Jz_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_pxx, ids->pxx_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_pxy, ids->pxy_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_pxz, ids->pxz_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_pyy, ids->pyy_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_pyz, ids->pyz_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ids_pzz, ids->pzz_flat, size_array_nodes, cudaMemcpyHostToDevice);

    // Binding pointers
    cudaMemcpy(&(idsGPU->rhon_flat), &dev_ids_rhon, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->rhoc_flat), &dev_ids_rhoc, size_array_cells, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->Jx_flat), &dev_ids_Jx, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->Jy_flat), &dev_ids_Jy, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->Jz_flat), &dev_ids_Jz, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->pxx_flat), &dev_ids_pxx, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->pxy_flat), &dev_ids_pxy, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->pxz_flat), &dev_ids_pxz, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->pyy_flat), &dev_ids_pyy, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->pyz_flat), &dev_ids_pyz, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(&(idsGPU->pzz_flat), &dev_ids_pzz, size_array_nodes, cudaMemcpyHostToDevice);
}

/** allocated interpolated densities per species */
void interp_dens_species_cpu2gpu(struct grid* grd, struct interpDensSpecies* ids, struct interpDensSpecies* idsGPU)
{
    size_t size_array_nodes = grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp);
    size_t size_array_cells = grd->nxc * grd->nyc * grd->nzc * sizeof(FPinterp);

    cudaMemcpy(idsGPU->rhon_flat, ids->rhon_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->rhoc_flat, ids->rhoc_flat, size_array_cells, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->Jy_flat, ids->Jy_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->Jx_flat, ids->Jx_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->Jz_flat, ids->Jz_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->pxx_flat, ids->pxx_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->pxy_flat, ids->pxy_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->pxz_flat, ids->pxz_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->pyy_flat, ids->pyy_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->pyz_flat, ids->pyz_flat, size_array_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(idsGPU->pzz_flat, ids->pzz_flat, size_array_nodes, cudaMemcpyHostToDevice);
}

/** allocated interpolated densities per species */
void interp_dens_species_gpu2cpu(struct grid* grd, struct interpDensSpecies* ids, struct interpDensSpecies* idsGPU)
{
    size_t size_array_nodes = grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp);
    size_t size_array_cells = grd->nxc * grd->nyc * grd->nzc * sizeof(FPinterp);

    cudaMemcpy(ids->rhon_flat, idsGPU->rhon_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhoc_flat, idsGPU->rhoc_flat, size_array_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, idsGPU->Jy_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jx_flat, idsGPU->Jx_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, idsGPU->Jz_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx_flat, idsGPU->pxx_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, idsGPU->pxy_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, idsGPU->pxz_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, idsGPU->pyy_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, idsGPU->pyz_flat, size_array_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, idsGPU->pzz_flat, size_array_nodes, cudaMemcpyDeviceToHost);
}

/** allocated interpolated densities per species */
void ids_gpu_deallocate(struct interpDensSpecies* idsGPU)
{
    cudaFree(idsGPU->rhon_flat);
    cudaFree(idsGPU->rhoc_flat);
    cudaFree(idsGPU->Jy_flat);
    cudaFree(idsGPU->Jx_flat);
    cudaFree(idsGPU->Jz_flat);
    cudaFree(idsGPU->pxx_flat);
    cudaFree(idsGPU->pxy_flat);
    cudaFree(idsGPU->pxz_flat);
    cudaFree(idsGPU->pyy_flat);
    cudaFree(idsGPU->pyz_flat);
    cudaFree(idsGPU->pzz_flat);
}

/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, int is)
{
    // set species ID
    ids->species_ID = is;
    
    // allocate 3D arrays
    // rho: 1
    ids->rhon = newArr3<FPinterp>(&ids->rhon_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    ids->rhoc = newArr3<FPinterp>(&ids->rhoc_flat, grd->nxc, grd->nyc, grd->nzc); // center
    // Jx: 2
    ids->Jx   = newArr3<FPinterp>(&ids->Jx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jy: 3
    ids->Jy   = newArr3<FPinterp>(&ids->Jy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jz: 4
    ids->Jz   = newArr3<FPinterp>(&ids->Jz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxx: 5
    ids->pxx  = newArr3<FPinterp>(&ids->pxx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxy: 6
    ids->pxy  = newArr3<FPinterp>(&ids->pxy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxz: 7
    ids->pxz  = newArr3<FPinterp>(&ids->pxz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyy: 8
    ids->pyy  = newArr3<FPinterp>(&ids->pyy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyz: 9
    ids->pyz  = newArr3<FPinterp>(&ids->pyz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pzz: 10
    ids->pzz  = newArr3<FPinterp>(&ids->pzz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    
}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid* grd, struct interpDensSpecies* ids)
{
    
    // deallocate 3D arrays
    delArr3(ids->rhon, grd->nxn, grd->nyn);
    delArr3(ids->rhoc, grd->nxc, grd->nyc);
    // deallocate 3D arrays: J - current
    delArr3(ids->Jx, grd->nxn, grd->nyn);
    delArr3(ids->Jy, grd->nxn, grd->nyn);
    delArr3(ids->Jz, grd->nxn, grd->nyn);
    // deallocate 3D arrays: pressure
    delArr3(ids->pxx, grd->nxn, grd->nyn);
    delArr3(ids->pxy, grd->nxn, grd->nyn);
    delArr3(ids->pxz, grd->nxn, grd->nyn);
    delArr3(ids->pyy, grd->nxn, grd->nyn);
    delArr3(ids->pyz, grd->nxn, grd->nyn);
    delArr3(ids->pzz, grd->nxn, grd->nyn);
    
    
}

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies* ids, struct grid* grd){
    for (register int i = 1; i < grd->nxc - 1; i++)
        for (register int j = 1; j < grd->nyc - 1; j++)
            for (register int k = 1; k < grd->nzc - 1; k++){
                ids->rhoc[i][j][k] = .125 * (ids->rhon[i][j][k] + ids->rhon[i + 1][j][k] + ids->rhon[i][j + 1][k] + ids->rhon[i][j][k + 1] +
                                       ids->rhon[i + 1][j + 1][k]+ ids->rhon[i + 1][j][k + 1] + ids->rhon[i][j + 1][k + 1] + ids->rhon[i + 1][j + 1][k + 1]);
            }
}
