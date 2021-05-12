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
    long seed = 1234;
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

    fhp1_grid fhp(width, height, channels, buffer, seed);
    u8* d_ptr = fhp.device_grid;

    dim3 block(8, 8);
    dim3 grid(width/8, height/8);
    evolve<<<grid, block>>>(fhp.device_grid, fhp.state, fhp.width, fhp.height, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError( ));


    u8 *output = (u8*) malloc(128*sizeof(u8));
    cudaMemcpy(output, d_ptr, 8*8*sizeof(u8),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));

    // const int GRID_SIZE = 8;
    // std::cout << "\n\n";
    // for(int i=0; i<GRID_SIZE; i++) 
    // {
    //     for(int j=0; j<GRID_SIZE; j++){
    //         u8 t = output[i*GRID_SIZE+j];
    //         std::bitset<8> x(t);
    //         std::cout << (int)t <<"\t";
    //     }
    //     std::cout << std::endl;
    // }


    // THis is output for state initilized with `curand_init(1234, id, 0, &state[id]);`
    u8 expected[] = {
        4, 36, 2, 3, 38, 38, 15, 14, 
        8, 125, 36, 48, 59, 20, 39, 5, 
        2, 11, 44, 21, 62, 51, 40, 43, 
        44, 46, 20, 18, 57, 18, 43, 43, 
        20, 60, 4, 35, 57, 53, 14, 29, 
        10, 55, 19, 27, 63, 60, 1, 48, 
        2, 63, 0, 25, 20, 5, 25, 13, 
        2, 45, 32, 0, 29, 29, 53, 22
    };

    for(int i=0; i<64; i++){
        mu_assert(expected[i] == output[i], "Deviation from expected" );
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
