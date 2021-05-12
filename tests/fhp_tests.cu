#include "velocity.hpp"
#include "helper_types.hpp"
#include "init.hpp"
#include "minunit.h"
#include "dbg.h"
#include "fhp.hpp"

#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include <bitset>

char *test_fhp_1step()
{
    int width = 8, height = 8;
    std::vector<velocity2> channels = \
    {
        velocity2{{1.0, 0.0}},
        velocity2{{0.5, 0.866025}},
        velocity2{{-0.5, 0.866025}},
        velocity2{{-1.0, 0.0}},
        velocity2{{-0.5, -0.866025}},
        velocity2{{0.5, -0.866025}}    
    };

    u8 buffer[] = {
        39, 6, 41, 51, 17, 63, 10, 44, 
        41, 77, 58, 43, 50, 59, 35, 6, 
        60, 2, 20, 56, 27, 40, 39, 13, 
        54, 26, 46, 35, 51, 31, 9, 26, 
        38, 50, 13, 55, 49, 24, 35, 26, 
        37, 29, 5, 23, 24, 41, 30, 20, 
        43, 50, 13, 6, 27, 52, 20, 17, 
        14, 2, 52, 1, 33, 61, 28, 7
    };

    fhp1_grid fhp(width, height, channels, buffer);
    u8* d_ptr = fhp.device_grid;

    dim3 block(8, 8);
    dim3 grid(width/8, height/8);
    evolve<<<grid, block>>>(fhp);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError( ));


    // gpuErrchk(cudaGetLastError( ));
    u8 *output = (u8*) malloc(128*sizeof(u8));
    // gpuErrchk(cudaGetLastError( ));
    // fhp.get_output(output);
    cudaMemcpy(output, d_ptr, 8*8*sizeof(u8),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));

    const int GRID_SIZE = 8;
    for(int i=0; i<GRID_SIZE; i++) 
    {
        for(int j=0; j<GRID_SIZE; j++){
            u8 t = output[i*GRID_SIZE+j];
            std::bitset<8> x(t);
            std::cout << (int)t <<" ";
        }
        std::cout << "\n";
    }

    free(output);

    return NULL;
}

char *all_tests()
{
    mu_suite_start();

    mu_run_test(test_fhp_1step);

    return NULL;
}

RUN_TESTS(all_tests);
