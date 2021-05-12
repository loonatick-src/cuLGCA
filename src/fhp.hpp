# pragma once
#include <vector>
#include <cassert>
#include <curand_kernel.h>
#include "dbg.h"
#include "velocity.hpp"


typedef uint8_t u8;
typedef uint32_t u32;
__constant__ u8 d_eq_class_size[128];
__constant__ u8 d_state_to_eq[128];
__constant__ u8 d_eq_classes[128];


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
struct fhp_grid
{
    word *device_grid;
    double *device_channels;
    curandState *state;

    const size_t width, height;
    const std::vector<velocity2> channels;
    
    fhp_grid(size_t w, size_t h,
            std::vector<velocity2> velocities, word *buffer) :
        width {w}, height{h}, channels {velocities}
    {
        static_assert(sizeof(word)*8 > channel_count);
        assert(buffer != NULL);
        assert(channel_count == velocities.size());

        const auto grid_sz = width * height;
        const auto mem_sz = grid_sz * sizeof(word);
        const auto channel_mem_sz = 2 * sizeof(double) * channel_count;
        double *temp = new double [2 * channel_count];
        for (size_t i = 0; i < channels.size(); i+=2)
        {
            temp[i] = channels[i][0]; 
            temp[i+1] = channels[i][1];
        }
        cudaMalloc((void **) &device_channels, channel_mem_sz);
        cudaMalloc((void **) &device_grid, mem_sz);
        cudaMalloc((void **) &state, width*height*sizeof(curandState));
        cudaMemcpy(device_grid, buffer, mem_sz,
                cudaMemcpyHostToDevice);
        cudaMemcpy(device_channels, temp, channel_mem_sz,
                cudaMemcpyHostToDevice);

        // Setup curand states
        setup_kernel(state, width, height);
        delete[] temp;
    }

    ~fhp_grid()
    {
        cudaFree(device_grid);
        cudaFree(device_channels);
    }

    template <size_t timesteps>
    void start_evolution();
    
    __device__
    word stream(int, int);

    __device__
    void collide(curandState localstate, word state);
    
        
    __device__
    word occupancy(word state);

    __device__
    auto momentum_x(word state, double *device_channels)->double;

    __device__
    auto momentum_y(word state, double *device_channels)->double;

    __device__
    void setup_kernel(curandState *state, size_t width, size_t height);
    
};

typedef fhp_grid<uint8_t, 6, 8, 8> fhp1_grid;

// kernel
__global__
void
evolve(fhp1_grid);
