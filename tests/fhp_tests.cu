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
#include <algorithm>

double
flerror(double of, double against)
{
    return fabs(of - against); 
}

const char *test_fhp_1step()
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
    evolve<<<grid, block>>>(fhp.device_grid, fhp.state, fhp.width, fhp.height, 1,
        fhp.device_channels, fhp.mx, fhp.my, fhp.ocpy);
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
        4,	36,	18,	35,	38,	59,	63,	14,	
        40,	125,36,	48,	59,	20,	39,	5,	
        3,	11,	44,	21,	62,	51,	40,	43,	
        44,	46,	20,	18,	57,	18,	43,	43,	
        20,	60,	4,	35,	57,	53,	14,	29,	
        10,	55,	19,	27,	63,	60,	1,	48,	
        3,	63,	0,	25,	20,	5,	25,	13,	
        3,	45,	32,	0,	29,	29,	53,	22,
    };

    for(int i=0; i<64; i++){
        // fprintf(stderr, "val: %d vs %d\n", expected[i], output[i]);
        mu_assert(expected[i] == output[i], "Deviation from expected" );
    }

    free(output);

    return NULL;
}

const char *fhp_all1()
{
    const auto threshold = 0.0001l;

    int width = 16, height = 16;
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

    u8 buffer[width*height];
    std::fill_n(buffer, width*height, 3);

    fhp1_grid fhp(width, height, channels, buffer, seed);
    u8* d_ptr = fhp.device_grid;

    dim3 block(8, 8);
    dim3 grid(width/8, height/8);
    evolve<<<grid, block>>>(fhp.device_grid, fhp.state, fhp.width, fhp.height, 3,
        fhp.device_channels, fhp.mx, fhp.my, fhp.ocpy);
    momentum<<<grid, block>>>(fhp.device_grid, fhp.device_channels, 
        fhp.mx, fhp.my, fhp.ocpy, fhp.width);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError( ));


    u8 *output = (u8*) malloc(width*height*sizeof(u8));
    cudaMemcpy(output, d_ptr, width*height*sizeof(u8),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));

    double *mx = (double*) malloc(width*height*sizeof(double));
    double *my = (double*) malloc(width*height*sizeof(double));
    u8 *ocpy = (u8*) malloc(width*height*sizeof(u8));
    cudaMemcpy(mx, fhp.mx, width*height*sizeof(double),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));
    cudaMemcpy(my, fhp.my, width*height*sizeof(double),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));
    cudaMemcpy(ocpy, fhp.ocpy, width*height*sizeof(u8),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));

    for (int i=0; i<width*height; i++){
        mu_assert(2 == ocpy[i], "Occupancy deviation" );
    }

    for (int i=0; i<width*height; i++){
        mu_assert(flerror(mx[i], 1.5) < threshold, "Momentum x deviation" );
    }

    for (int i=0; i<width*height; i++){
        mu_assert(flerror(my[i], 0.86602540378) < threshold, "Momentum x deviation" );
    }

    free(mx); free(my);
    free(output);

    return NULL;

}

// NOT A TEST, USED FOR CHECKING EVOLUTION OUTPUT
const char *fhp_all3()
{
    int width = 16, height = 16;
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

    u8 buffer[width*height];
    std::fill_n(buffer, width*height, 3);

    buffer[5*width+5] |= 1<<6;
    buffer[5*width+6] |= 1<<6;
    buffer[5*width+7] |= 1<<6;

    const int GRID_SIZE = width;
    std::cout << "\n\n";
    for(int i=0; i<GRID_SIZE; i++) 
    {
        for(int j=0; j<GRID_SIZE; j++){
            u8 t = buffer[i*GRID_SIZE+j];
            std::bitset<8> x(t);
            std::cout << (int)t <<"\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\n\n";

    fhp1_grid fhp(width, height, channels, buffer, seed);
    u8* d_ptr = fhp.device_grid;

    dim3 block(8, 8);
    dim3 grid(width/8, height/8);
    evolve<<<grid, block>>>(fhp.device_grid, fhp.state, fhp.width, fhp.height, 1000,
        fhp.device_channels, fhp.mx, fhp.my, fhp.ocpy);
    momentum<<<grid, block>>>(fhp.device_grid, fhp.device_channels, 
        fhp.mx, fhp.my, fhp.ocpy, fhp.width);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError( ));


    u8 *output = (u8*) malloc(width*height*sizeof(u8));
    cudaMemcpy(output, d_ptr, width*height*sizeof(u8),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));

    double *mx = (double*) malloc(width*height*sizeof(double));
    double *my = (double*) malloc(width*height*sizeof(double));
    u8 *ocpy = (u8*) malloc(width*height*sizeof(u8));
    cudaMemcpy(mx, fhp.mx, width*height*sizeof(double),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));
    cudaMemcpy(my, fhp.my, width*height*sizeof(double),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));
    cudaMemcpy(ocpy, fhp.ocpy, width*height*sizeof(u8),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));

    std::cout << "\n\n";
    for(int i=0; i<GRID_SIZE; i++) 
    {
        for(int j=0; j<GRID_SIZE; j++){
            u8 t = output[i*GRID_SIZE+j];
            std::bitset<8> x(t);
            std::cout << (int)t <<"\t";
        }
        std::cout << std::endl;
    }


    free(mx); free(my);
    free(output);

    return NULL;

}

// NOT A TEST, USED FOR CHECKING RANDOM INITIALIZER
const char *fhp_generate_grid()
{
    int width = 16, height = 16;
    long seed = 2;
    std::vector<velocity2> channels = \
    {
        velocity2{{1.0, 0.0}},
        velocity2{{0.5, 0.866025}},
        velocity2{{-0.5, 0.866025}},
        velocity2{{-1.0, 0.0}},
        velocity2{{-0.5, -0.866025}},
        velocity2{{0.5, -0.866025}}    
    };

    // dim3 block(8, 8);
    // dim3 grid(width/8, height/8);

    u8 buffer[width*height];
    std::fill_n(buffer, width*height, 0);

    buffer[5*width+5] = 1;

    double h_prob[] = {0.9, 0.9, 0.4, 0.3, 0.4, 0.9};

    fhp1_grid fhp = fhp_grid<u8, 6, 8, 8>(width, height, channels, seed, h_prob, buffer);

    cudaMemcpy(buffer, fhp.device_grid, width*height*sizeof(u8),
        cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError( ));

    const int GRID_SIZE = width;
    std::cout << "\n\n";
    for(int i=0; i<GRID_SIZE; i++) 
    {
        for(int j=0; j<GRID_SIZE; j++){
            u8 t = buffer[i*GRID_SIZE+j];
            // std::bitset<8> x(t);
            std::cout << (int)t <<"\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\n\n";

    return NULL;

}

const char *all_tests()
{
    mu_suite_start();

    mu_run_test(test_fhp_1step);
    mu_run_test(fhp_all1);
    // mu_run_test(fhp_generate_grid);
    // mu_run_test(fhp_all3);

    return NULL;
}

RUN_TESTS(all_tests);
