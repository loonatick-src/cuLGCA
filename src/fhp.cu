#include "velocity.hpp"
#include "dbg.h"
#include "fhp.hpp"
#include "helper_types.hpp"

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
void
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::momentum()
{
    // TODO
    return;
}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
void
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::occupancy()
{
    // TODO
    return;
}
