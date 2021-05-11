#include <tuple>
#include <curand_kernel.h>

// Constant memory lookup tables
__constant__ uint8_t d_state_to_eq[128];
__constant__ uint8_t d_eq_class_size[128];
__constant__ uint8_t d_eq_classes[128];

// TODO: Copying constant memories from host to device

typedef uint8_t u8;
typedef std::tuple<double, double> velocity2;
typedef velocity2 v2;
typedef std::array<velocity2, 6> fhp_channels_t;

const u8 full_node {0x3f};
const fhp_channels_t fhp_channels
{
    // might need for in-house momentum_calculation
    v2{1.0, 0.0},
    v2{0.5, 0.866025},
    v2{-0.5, 0.866025},
    v2{-1.0, 0.0},
    v2{-0.5, -0.866025},
    v2{0.5, -0.866025}
};



inline
auto
number_of_tiles(size_t width, size_t block_width)
{
    return ceil( ((double) width)/block_width );
}

// Note: state[] size is equal to total thread size
__global__ void setup_kernel(curandState *state, size_t width, size_t height)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int id = idy*width + idx;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

template <size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT, size_t timesteps>
__global__
void
fhp_kernel(u8 *grid, size_t width, size_t height, curandState *state)
{
    static_assert(BLOCK_WIDTH * BLOCK_WIDTH <= 1024);
    __shared__ u8 s_in[BLOCK_HEIGHT+2][BLOCK_WIDTH+2];
    __shared__ u8 s_out[BLOCK_HEIGHT][BLOCK_WIDTH];

    const auto local_row = threadIdx.y+1;
    const auto local_col = threadIdx.x+1;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localstate = state[row*width + col];
    __syncthreads();

#   pragma unroll
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


        // 2. %%%%%%%%%%%% streaming operator
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

        // 3. %%%%%%%%%%%% perform collision
        u8 size = d_eq_class_size[state];
        u8 base_index = d_state_to_eq[state];

        // This is from [0,...,size-1]
        float rand = curand_uniform(&localstate);
        rand *= size-0.00001;
        u8 random_index = (u8)(rand) % size; // Require curand_init for each thread

        state = d_eq_classes[base_index + random_index];

        // 4. send result back to main memory
        grid[row * width + col] = state;
    }
}
/*



<template size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
void
fhp(u8 *grid, size_t width, size_t height)
{
    static_assert(BLOCK_WIDTH * BLOCK_HEIGHT <= 1024);
    const dim3 block_config(BLOCK_HEIGHT, BLOCK_WIDTH);
    const size_t grid_height { number_of_tiles(height, BLOCK_HEIGHT) };
    const size_t grid_width { number_of_tiles(width, BLOCK_WIDTH) };
    const dim3 grid_config(grid_height, grid_width);
}
    
*/
