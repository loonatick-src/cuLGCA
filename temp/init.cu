#include <iostream>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <thrust/device_vector>
#include <thrust/host_vector>
#include "dbg.h"
#include "fhp.hpp"
#include "velocity.hpp"

typedef uint8_t u8;
typedef uint32_t u32;
typedef velocity<2, double> velocity2;

#define obstacle_bit(S, C) ((S) & (0x1<<(C)))

template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
auto
initialize_cylindrical_obstacle(word *grid, const size_t width, const size_t height, double radius)
{
    assert(radius < width/2 && radius < height/2);
    assert(grid != NULL);

    const dim3 block_config(BLOCK_HEIGHT, BLOCK_WIDTH);
    const size_t grid_width = (size_t) ceil(((double) width)/BLOCK_WIDTH);
    const size_t grid_height = (size_t) ceil(((double) height)/BLOCK_HEIGHT);
    const dim3 grid_config(grid_height, grid_width);

    const auto grid_sz = width * height;
    const auto mem_sz = grid_sz * sizeof(word);

    word *d_grid; 

    cudaMalloc((void **) &d_grid, mem_sz);
    cudaMemcpy(d_grid, grid, mem_sz, cudaMemcpyHostToDevice);

    cobst_kernel<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>
        <<<grid_config, block_config>>>(d_grid, width, height, radius);
    
    cudaMemcpy(grid, d_grid, mem_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
}


__device__
void
distance2(const dim3 thread_id, const dim3 block_id, const size_t centre_r, const size_t centre_c)
{
    
}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
__global__
void
cobst_kernel(word *grid, const size_t width, const size_t height, double radius)
{
    // TODO
    const auto centre_r = width / 2; 
    const auto centre_c = height / 2;
    return;
}


template <typename word, u8 channel_count>
__device__
void
clean_obstacle_nodes(word *grid, const const size_t width, const const size_t height)
{
    static_assert(sizeof(word)*8 > channel_count); 
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    __syncthreads();
    if (row < height && col < width)
    {
        const auto index = row * width + col;
        auto state = grid[index];
        const auto obbit = obstacle_bit(state, channel_count);
        state = (obbit) ? obbit : state; 
        grid[index] = state;
    }

    return;
}
