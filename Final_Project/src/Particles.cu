#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define TpBx 512

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** allocate particle arrays */
void particle_allocate_gpu(struct particles* part, struct particles* particlesGPU)
{    
    FPpart *dev_x, *dev_y, *dev_z, *dev_u, *dev_v, *dev_w, *dev_q;

    cudaMalloc(&dev_x, part->npmax * sizeof(FPpart));
    cudaMalloc(&dev_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&dev_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&dev_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&dev_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&dev_w, part->npmax * sizeof(FPpart));
    cudaMalloc(&dev_q, part->npmax * sizeof(FPpart));

    cudaMemcpy(dev_x, part->x, part->npmax * sizeof(*dev_x), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, part->y, part->npmax * sizeof(*dev_y), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, part->z, part->npmax * sizeof(*dev_z), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, part->u, part->npmax * sizeof(*dev_u), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, part->v, part->npmax * sizeof(*dev_v), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, part->w, part->npmax * sizeof(*dev_w), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_q, part->q, part->npmax * sizeof(*dev_q), cudaMemcpyHostToDevice);

    // Binding pointers
    cudaMemcpy(&(particlesGPU->x), &dev_x, sizeof(particlesGPU->x), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->y), &dev_y, sizeof(particlesGPU->y), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->z), &dev_z, sizeof(particlesGPU->z), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->u), &dev_u, sizeof(particlesGPU->u), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->v), &dev_v, sizeof(particlesGPU->v), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->w), &dev_w, sizeof(particlesGPU->w), cudaMemcpyHostToDevice);
    cudaMemcpy(&(particlesGPU->q), &dev_q, sizeof(particlesGPU->q), cudaMemcpyHostToDevice);
}

/** allocated interpolated densities per species */
void particle_deallocate_gpu(struct particles* part)
{
    cudaFree(part->x);
    cudaFree(part->y);
    cudaFree(part->z);
    cudaFree(part->u);
    cudaFree(part->v);
    cudaFree(part->w);
    cudaFree(part->q);
}

__device__ void subcycle_single_particle(particles* part, EMfield* field, grid* grd, parameters* param, int particle_index) {

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = part->x[particle_index];
    yptilde = part->y[particle_index];
    zptilde = part->z[particle_index];
    // calculate the average velocity iteratively
    for(int innter=0; innter < part->NiterMover; innter++){
        // interpolation G-->P
        // 2 + to create boundary conditions
        // Index of the cells:
        ix = 2 +  int((part->x[particle_index] - grd->xStart)*grd->invdx);
        iy = 2 +  int((part->y[particle_index] - grd->yStart)*grd->invdy);
        iz = 2 +  int((part->z[particle_index] - grd->zStart)*grd->invdz);
        
        // calculate weights
        long xi0_index_flat = get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn);
        xi[0]   = part->x[particle_index] - grd->XN_flat[xi0_index_flat];

        long eta0_index_flat = get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn);
        eta[0]  = part->y[particle_index] - grd->YN_flat[eta0_index_flat];

        long zeta0_index_flat = get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn);
        zeta[0] = part->z[particle_index] - grd->ZN_flat[zeta0_index_flat];

        long index_flat_1 = get_idx(ix, iy, iz, grd->nyn, grd->nzn);
        xi[1]   = grd->XN_flat[index_flat_1] - part->x[particle_index];
        eta[1]  = grd->YN_flat[index_flat_1] - part->y[particle_index];
        zeta[1] = grd->ZN_flat[index_flat_1] - part->z[particle_index];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
        
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    long index_flat = get_idx(ix- ii, iy -jj, iz- kk, grd->nyn, grd->nzn);
                    Exl += weight[ii][jj][kk]*field->Ex_flat[index_flat];
                    Eyl += weight[ii][jj][kk]*field->Ey_flat[index_flat];
                    Ezl += weight[ii][jj][kk]*field->Ez_flat[index_flat];
                    Bxl += weight[ii][jj][kk]*field->Bxn_flat[index_flat];
                    Byl += weight[ii][jj][kk]*field->Byn_flat[index_flat];
                    Bzl += weight[ii][jj][kk]*field->Bzn_flat[index_flat];
                }
        
        // end interpolation
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= part->u[particle_index] + qomdt2*Exl;
        vt= part->v[particle_index] + qomdt2*Eyl;
        wt= part->w[particle_index] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
        // update position
        part->x[particle_index] = xptilde + uptilde*dto2;
        part->y[particle_index] = yptilde + vptilde*dto2;
        part->z[particle_index] = zptilde + wptilde*dto2;
        
        
    } // end of iteration
    // update the final position and velocity
    part->u[particle_index]= 2.0*uptilde - part->u[particle_index];
    part->v[particle_index]= 2.0*vptilde - part->v[particle_index];
    part->w[particle_index]= 2.0*wptilde - part->w[particle_index];
    part->x[particle_index] = xptilde + uptilde*dt_sub_cycling;
    part->y[particle_index] = yptilde + vptilde*dt_sub_cycling;
    part->z[particle_index] = zptilde + wptilde*dt_sub_cycling;
    
    
    //////////
    //////////
    ////////// BC
                       
    // X-DIRECTION: BC particles
    if (part->x[particle_index] > grd->Lx){
        if (param->PERIODICX==true){ // PERIODIC
            part->x[particle_index] = part->x[particle_index] - grd->Lx;
        } else { // REFLECTING BC
            part->u[particle_index] = -part->u[particle_index];
            part->x[particle_index] = 2*grd->Lx - part->x[particle_index];
        }
    }

    if (part->x[particle_index] < 0){
        if (param->PERIODICX==true){ // PERIODIC
           part->x[particle_index] = part->x[particle_index] + grd->Lx;
        } else { // REFLECTING BC
            part->u[particle_index] = -part->u[particle_index];
            part->x[particle_index] = -part->x[particle_index];
        }
    }
    
    // Y-DIRECTION: BC particles
    if (part->y[particle_index] > grd->Ly){
        if (param->PERIODICY==true){ // PERIODIC
            part->y[particle_index] = part->y[particle_index] - grd->Ly;
        } else { // REFLECTING BC
            part->v[particle_index] = -part->v[particle_index];
            part->y[particle_index] = 2*grd->Ly - part->y[particle_index];
        }
    }
    
    if (part->y[particle_index] < 0){
        if (param->PERIODICY==true){ // PERIODIC
            part->y[particle_index] = part->y[particle_index] + grd->Ly;
        } else { // REFLECTING BC
            part->v[particle_index] = -part->v[particle_index];
            part->y[particle_index] = -part->y[particle_index];
        }
    }

    // Z-DIRECTION: BC particles
    if (part->z[particle_index] > grd->Lz){
        if (param->PERIODICZ==true){ // PERIODIC
            part->z[particle_index] = part->z[particle_index] - grd->Lz;
        } else { // REFLECTING BC
            part->w[particle_index] = -part->w[particle_index];
            part->z[particle_index] = 2*grd->Lz - part->z[particle_index];
        }
    }

    if (part->z[particle_index] < 0){
        if (param->PERIODICZ==true){ // PERIODIC
            part->z[particle_index] = part->z[particle_index] + grd->Lz;
        } else { // REFLECTING BC
            part->w[particle_index] = -part->w[particle_index];
            part->z[particle_index] = -part->z[particle_index];
        }
    }
}

__global__ void gpu_mover_PC(particles* parts, EMfield* field, grid* grd, parameters* param) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;  // Particle number
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;  // Type of particle
    
    particles *part = &(parts[index_y]);
    if (index_x >= part->nop) {
        return;
    }

    subcycle_single_particle(part, field, grd, param, index_x);
}


void gpu_mover_PC_wrapper(particles* parts, EMfield* field, grid* grd, parameters* param, int largestNumParticles) {
    gpu_mover_PC<<<dim3(largestNumParticles / TpBx + 1, 1, 1), dim3(TpBx, param->ns, 1)>>>(parts, field, grd, param);
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            //subcycle_single_particle(part, field, grd, param, i);                                                            
        }  // end of subcycling
    } // end of one particle

    return(0); // exit succcesfully
} // end of the mover


__device__ void interpolate_single_particle(particles* part,interpDensSpecies* ids, grid* grd, int particle_index) {

    // index of the cell
    int ix, iy, iz;

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // determine cell: can we change to int()? is it faster?
    ix = 2 + int (floor((part->x[particle_index] - grd->xStart) * grd->invdx));
    iy = 2 + int (floor((part->y[particle_index] - grd->yStart) * grd->invdy));
    iz = 2 + int (floor((part->z[particle_index] - grd->zStart) * grd->invdz));

    // distances from node
    long xi0_index_flat = get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn);
    xi[0]   = part->x[particle_index] - grd->XN_flat[xi0_index_flat];

    long eta0_index_flat = get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn);
    eta[0]  = part->y[particle_index] - grd->YN_flat[eta0_index_flat];

    long zeta0_index_flat = get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn);
    zeta[0] = part->z[particle_index] - grd->ZN_flat[zeta0_index_flat];

    long index_flat_1 = get_idx(ix, iy, iz, grd->nyn, grd->nzn);
    xi[1]   = grd->XN_flat[index_flat_1] - part->x[particle_index];
    eta[1]  = grd->YN_flat[index_flat_1] - part->y[particle_index];
    zeta[1] = grd->ZN_flat[index_flat_1] - part->z[particle_index];

    // calculate the weights for different nodes
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weight[ii][jj][kk] = part->q[particle_index] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long rhon_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->rhon_flat[rhon_index_flat] += weight[ii][jj][kk] * grd->invVOL;
            }


    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part->u[particle_index] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long jx_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->Jx_flat[jx_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }

    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                temp[ii][jj][kk] = part->v[particle_index] * weight[ii][jj][kk];
            }

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long jy_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->Jy_flat[jy_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }


    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                temp[ii][jj][kk] = part->w[particle_index] * weight[ii][jj][kk];
            }

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long jz_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->Jz_flat[jz_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }

    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part->u[particle_index] * part->u[particle_index] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long pxx_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->pxx_flat[pxx_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }


    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part->u[particle_index] * part->v[particle_index] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long pxx_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->pxy_flat[pxx_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }


    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part->u[particle_index] * part->w[particle_index] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long pxz_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->pxz_flat[pxz_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }

    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part->v[particle_index] * part->v[particle_index] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long pyy_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->pyy_flat[pyy_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }

    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part->v[particle_index] * part->w[particle_index] * weight[ii][jj][kk];
                
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long pyz_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->pyz_flat[pyz_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }

    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = part->w[particle_index] * part->w[particle_index] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                long pzz_index_flat = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                ids->pzz_flat[pzz_index_flat] += temp[ii][jj][kk] * grd->invVOL;
            }

}

__global__ void gpu_interpP2G(particles* parts, interpDensSpecies* ids, grid* grd) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;  // Particle number
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;  // Type of particle

    particles* part = &(parts[index_y]);
    if (index_x > part->nop) {
        return;
    }

    interpolate_single_particle(part, ids, grd, index_x);
}

void gpu_interpP2G_wrapper(particles* parts, interpDensSpecies* ids, grid* grd, parameters* param, int largestNumParticles) {
    gpu_interpP2G<<<dim3(largestNumParticles / TpBx + 1, 1, 1), dim3(TpBx, param->ns, 1)>>>(parts, ids, grd);
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(particles* part, interpDensSpecies* ids, grid* grd)
{    
    for (register long long i = 0; i < part->nop; i++) {
        //interpolate_single_particle(part, ids, grd, i);
    }
}
