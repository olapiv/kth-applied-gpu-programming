/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include <cstring>


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd, &field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd, &field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param, &part[is], is);
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);

    ////////////////////
    // Alocation GPU: //
    ////////////////////

    particles *particlesGPU = new particles[param.ns];
    cudaMalloc(&particlesGPU, sizeof(particles) * param.ns);
    cudaMemcpy(particlesGPU, part, sizeof(particles) * param.ns, cudaMemcpyHostToDevice);
    for (int is=0; is < param.ns; is++){
        particle_allocate_gpu(&part[is], &particlesGPU[is]);  // Correct the pointers of the arrays
    }

    EMfield *fieldGPU;
    cudaMalloc(&fieldGPU, sizeof(EMfield));
    cudaMemcpy(fieldGPU, &field, sizeof(EMfield), cudaMemcpyHostToDevice);
    field_allocate_gpu(&grd, &field, fieldGPU);  // Correct the pointers of the arrays

    struct grid *grdGPU;
    cudaMalloc(&grdGPU, sizeof(grid));
    cudaMemcpy(grdGPU, &grd, sizeof(grid), cudaMemcpyHostToDevice);
    setGridGPU(&param, &grd, grdGPU);  // Correct the pointers of the arrays

    parameters *paramGPU;
    cudaMalloc(&paramGPU, sizeof(parameters));
    cudaMemcpy(paramGPU, &param, sizeof(parameters), cudaMemcpyHostToDevice);

    interpDensSpecies *idsGPU = new interpDensSpecies[param.ns];
    interpDensSpecies *idsGPU2CPU = new interpDensSpecies[param.ns];
    cudaMalloc(&idsGPU, sizeof(interpDensSpecies) * param.ns);
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate_gpu(&grd, &ids[is], &idsGPU[is]);  // Correct the pointers of the arrays
    std::memcpy(idsGPU2CPU, &ids, sizeof(interpDensSpecies) * param.ns);  // cudaMemcpy is done in every iteration

    int largestNumParticles = 0;
    for (int i = 0; i < param.ns; i++) {
        if (part[i].nop > largestNumParticles) {
            largestNumParticles = part[i].nop;
        }
    }

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // Set to zero the densities - needed for interpolation
        // Only idn and ids are changed
        // setZeroDensities(&idn, ids, &grd, param.ns);
        setZeroDensities(&idn, idsGPU2CPU, &grd, param.ns);  // New for GPU

        cudaMemcpy(idsGPU, idsGPU2CPU, sizeof(interpDensSpecies) * param.ns, cudaMemcpyHostToDevice);
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        // for (int is=0; is < param.ns; is++)
        //    mover_PC(&part[is],&field,&grd,&param);
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        // Only particlesGPU is changed
        gpu_mover_PC_wrapper(particlesGPU, fieldGPU, grdGPU, paramGPU, largestNumParticles);
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        // for (int is=0; is < param.ns; is++)
        //    interpP2G(&part[is],&ids[is],&grd);

        // Only ids is changed
        gpu_interpP2G_wrapper(particlesGPU, idsGPU, grdGPU, paramGPU, largestNumParticles);

        cudaMemcpy(idsGPU2CPU, idsGPU, sizeof(interpDensSpecies) * param.ns, cudaMemcpyDeviceToHost);

        // apply BC to interpolated densities
        // Only ids is changed
        for (int is=0; is < param.ns; is++)
            // applyBCids(&ids[is],&grd,&param);
            applyBCids(&idsGPU2CPU[is], &grd, &param);  // New for GPU

        // sum over species
        // Only idn is changed
        // sumOverSpecies(&idn, ids, &grd, param.ns);
        sumOverSpecies(&idn, idsGPU2CPU, &grd, param.ns);  // New for GPU

        // interpolate charge density from center to node
        // Only idn is changed 
        // TODO: Is idn changed though? Why not &idn.rhon?
        // applyBCscalarDensN(idn.rhon, &grd, &param);
        applyBCscalarDensN(idn.rhon, &grd, &param);  // New for GPU
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            // TODO: Comment this out
            // VTK_Write_Vectors(cycle, &grd, &field);
            // VTK_Write_Scalars(cycle, &grd, ids, &idn);

            VTK_Write_Vectors(cycle, &grd, &field);  // New for GPU
            VTK_Write_Scalars(cycle, &grd, idsGPU2CPU, &idn);  // New for GPU
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
    }  // end of one PIC cycle
    
    // GPU de-allocation:
    cudaFree(particlesGPU);
    cudaFree(fieldGPU);
    cudaFree(grdGPU);
    cudaFree(paramGPU);

    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        interp_dens_species_deallocate(&grd,&idsGPU2CPU[is]);  // New for GPU
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}


