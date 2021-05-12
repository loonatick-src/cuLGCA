# pragma once
#include <vector>
#include <cassert>
#include "dbg.h"
#include "velocity.hpp"


typedef uint8_t u8;
typedef uint32_t u32;


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
struct fhp_grid
{
    word *device_grid;
    double *device_channels;

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
        cudaMalloc((void **) device_channels, channel_mem_sz);
        cudaMalloc((void **) device_grid, mem_sz);
        cudaMemcpy(device_grid, buffer, mem_sz,
                cudaMemcpyHostToDevice);
        cudaMemcpy(device_channels, temp, channel_mem_sz,
                cudaMemcpyHostToDevice);
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
    void stream();

    __device__
    void collide();
    
        
    __device__
    void occupancy(word state);

    __device__
    auto momentum_x(word state, double *device_channels)->decltype(device_channels[0]);

    __device__
    auto momentum_y(word state, double *device_channels)->decltype(device_channels[1]);
    
};

typedef fhp_grid, 6, defaul_bw, default_bh> fhp1_grid;

// kernel
__global__
void
evolve(fhp1_grid);
