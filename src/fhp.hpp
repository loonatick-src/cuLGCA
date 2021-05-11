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
    // Change this to constant device arrays
    thrust::device_vector<u8> d_eq_class_size, d_state_to_eq, d_eq_classes;
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

        // TODO: INITIALIZE EQ CLASSES DATA
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
        #pragma unroll
        for (size_t t = 0; t < timesteps; t++)
        {
            // 1. load into shared memory
            s_in[local_row][local_col] = grid[row*width + col];
            // As such all these bool values can be stored in the same
            // register. Is nvcc smart enough to do this if the number
            // of registers at hand are low?
            const auto ubound_y {local_row == blockDim.y-1};
            const auto ubound_x {local_col == blockDim.x-1};
            const auto lbound_y {local_row == 0};
            const auto lbound_x {local_col == 0};
            size_t pad_row = row, pad_col = col;
            size_t local_pad_row = local_row, local_pad_col = local_col;
            // NSight is going to roast tf out of this kernel
            if (lbound_y)
            {
                pad_row = (row == 0) ? height-1 : row-1;
                local_pad_row = 0;
                s_in[local_pad_row][local_col] = grid[pad_row * width + col];
            } else if (ubound_y)
            {
                pad_row = (row+1) % height;
                local_pad_row = blockDim.y+1;
                s_in[blockDim.y+1][local_col] = grid[pad_row * width + col];
            }
            __syncthreads();
            if (lbound_x)
            {
                pad_col = (col == 0) ? width-1 : col-1;
                local_pad_col = 0;
                s_in[local_row][local_pad_col] = grid[row * width + pad_col];
            } else if (ubound_x)
            {
                pad_col = (col+1) % width;
                local_pad_col = blockDim.x+1;
                s_in[local_row][local_pad_col] = grid[row * width + pad_col];
            }
            __syncthreads();

            if (pad_row != row && pad_col != col)
            {
                s_in[local_pad_row][local_pad_col] = grid[pad_row*width + pad_col]; 
            }
            __syncthreads();

            stream();
            collide();

            // 4. send result back to main memory
            grid[row * width + col] = state;
        }
    
    }

    __device__
    void stream()
    {
        // ASSUMPTION: Shared memory copies are handled in evolve()
        // 1<<6 is set if current point is obstacle
        u8 state = 0 | (1<<6 & (s_in[local_row][local_col]));
        u8 bit = 0x1;
        // TODO: redo with iterator using positive modulo
        state |= (bit & s_in[local_row][local_col-1]);
        bit <<= 1;
        state |= (bit & s_in[local_row+1][local_col]);
        bit <<= 1;
        state |= (bit & s_in[local_row+1][local_col+1]);
        bit <<= 1;
        state |= (bit & s_in[local_row][local_col+1]);
        bit <<= 1;
        state |= (bit & s_in[local_row-1][local_col]);
        bit <<= 1;
        state |= (bit & s_in[local_row-1][local_col-1]);
    }

    __device__
    void collide()
    {
        u8 size = d_eq_class_size[state];
        u8 base_index = d_state_to_eq[state];

        // This is from [0,...,size-1]
        float rand = curand_uniform(&localstate);
        rand *= size-0.00001;
        u8 random_index = (u8)(rand) % size; // Require curand_init for each thread

        state = d_eq_classes[base_index + random_index];
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
