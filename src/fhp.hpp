# pragma once
#include "dbg.h"
#include <thrust/device_vector>
#include <thrust/host_vector>
#include "velocity.hpp"


typedef uint8_t u8;
typedef uint32_t u32;
typedef velocity<2, double> velocity2;

template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
struct fhp_grid
{
    word *device_grid;
    const size_t width, height;
    const thrust::device_vector<velocity2> channels;

    fhp_grid(size_t w, size_t h,
            thrust::host_vector<velocity2> velocities, word *buffer) :
        width {w}, height{h}, channels {velocities}
    {
        static_assert(sizeof(word)*8 > channel_count);
        assert(buffer != NULL);
        assert(channel_count == velocities.size());
        const auto grid_sz = width * height;
        const auto mem_sz = grid_sz * sizeof(word);
        cudaMalloc((void **) device_grid, mem_sz);
        cudaMemcpy(device_grid, buffer, mem_sz);
    }

    ~fhp_grid()
    {
        cudaFree(device_grid);
    }

    template <size_t timesteps>
    void start_evolution();
    
    __device__
    void stream();

    __device__
    void collide();
    
    __device__
    void momentum();
    
    __device__
    void occupancy();
};

template <typename word, u8 channel_count, size_t timesteps,
         size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
__global__
void
evolve(fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT> grid);
