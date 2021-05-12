#include "velocity.hpp"
#include "dbg.h"
#include "fhp.hpp"
#include "helper_types.hpp"

const size_t default_bw = 16;
const size_t default_bh = 4;

//TODO: do I need the template arguments in every method declaration?


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
void
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::stream()
{
    // TODO
    return;
}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
void
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::collide()
{
    // TODO
    return;

}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
double
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::momentum_x()
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
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::momentum_y()
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
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::occupancy(state)
{
    word count = 0;
    while (state)
    {
        state &= (state-1);
        count++;
    }
    return count;
}


__global__
void
evolve(fhp1_grid)
{
    __shared__ sdm[BLOCK_HEIGHT][BLOCK_WIDTH];
    // TODO

    return;
}
