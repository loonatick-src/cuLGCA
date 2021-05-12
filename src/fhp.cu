#include "velocity.hpp"
#include "dbg.h"
#include "fhp.hpp"
#include "helper_types.hpp"

const size_t default_bw = 16;
const size_t default_bh = 4;

//TODO: do I need the template arguments in every method declaration?


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
word
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::stream(int local_row, int local_col)
{
    word state = 0 | (1<<6 & (device_grid[local_row][local_col]));
    word bit = 0x1;

    // TODO: redo with iterator using positive modulo
    state |= (bit & device_grid[local_row][local_col-1]);
    bit <<= 1;
    state |= (bit & device_grid[local_row+1][local_col]);
    bit <<= 1;
    state |= (bit & device_grid[local_row+1][local_col+1]);
    bit <<= 1;
    state |= (bit & device_grid[local_row][local_col+1]);
    bit <<= 1;
    state |= (bit & device_grid[local_row-1][local_col]);
    bit <<= 1;
    state |= (bit & device_grid[local_row-1][local_col-1]);

    return state;
}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
void
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::collide(curandState localstate, word state)
{
    word size = d_eq_class_size[state];
    word base_index = d_state_to_eq[state];

    // This is from [0,...,size-1]
    float rand = curand_uniform(&localstate);
    rand *= size-0.00001;
    word random_index = (word)(rand) % size; // Require curand_init for each thread

    state = d_eq_classes[base_index + random_index];
    return;

}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
double
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::momentum_x(word state, double *device_channels)
{
    double rv = 0.0l; 
    u8 bit = 0x1;
    for (size_t shift = 0; shift < channel_count; shift++)
    {
        u8 occ = bit & (state >> shift);
        if (occ)
        {
            rv += device_channels[shift*2];
        }
    }
    return rv;
}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
double
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::momentum_y(word state, double *device_channels)
{
    double rv = 0.0l; 
    const word bit = 0x1;
    for (size_t shift = 0; shift < channel_count; shift++)
    {
        word occ = bit & (state >> shift);
        if (occ)
        {
            rv += device_channels[shift*2+1];
        }
    }
    return rv;
}

template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
word
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::occupancy(word state)
{
    word count = 0;
    while (state)
    {
        state &= (state-1);
        count++;
    }
    return count;
}

template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
void setup_kernel(curandState *state, size_t width, size_t height) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int id = idy*width + idx;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
    return;
}


__global__
void
evolve(fhp1_grid)
{
    __shared__ int sdm[default_bh][default_bw];
    // TODO

    return;
}
