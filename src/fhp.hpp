# pragma once
#include "dbg.h"
#include <thrust/device_vector>
#include <thrust/host_vector>
#include "velocity.hpp"


typedef uint8_t u8;
typedef uint32_t u32;
typedef velocity<2, double> velocity2;

template <typename word, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
struct fhp_grid
{
    // do we want to go all in and make device_grid thrust::device_vector?
    // TOOD: initialize lookup tables (as members?)
    word *device_grid;
    const size_t width, height;
    const thrust::device_vector<velocity2> channels; // will this be directly acessible inside the __global__ and __device__ functions?
#if 0
    fhp_grid(size_t width, size_t height, thrust::host_vector<velocity2> velocities) :
        width{width}, height{height}, channels {velocities};
#endif
    fhp_grid(size_t w, size_t h, thrust::host_vector<velocity2> velocities, word *buffer) :
        width {w}, height{h}, channels {velocities}
    {
        assert(buffer != NULL);
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
    void start_evolution()
    {
        // TODO: port from master to this
        /*
        assert(channels.size() > 0);
        const dim3 block_config(BLOCK_HEIGHT, BLOCK_WIDTH);
        const dim3 grid_config(...);

        evolve<<<>>>();
        */
    }

    template <size_t timesteps>
    __global__
    void evolve()
    {
        /*
        #pragma unroll
        for (size_t t = 0; t < timesteps; t++)
        {
            
        }
        */
    
    }

    __device__
    void stream()
    {
        // apply streaming operator to grid
    }

    __device__
    void collide()
    {
        // apply collision operator to grid
    }

    __device__
    void momentum()
    {
        // calculate momentum at threadId, blockId
    }
    
    __device__
    void occupancy()
    {
        // calculate occupancy at threadId, blockId
        // not to be confused with CUDA occupancy
    }
};
