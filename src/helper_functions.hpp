#pragma once

inline
auto
make_tiles(const dim3& block_config, size_t width, size_t height)
{
    const size_t grid_width = (size_t) ceil( ((double)width/block_config.x));
    const size_t grid_height= (size_t) ceil( ((double) height)/block_config.y);
    const dim3 grid_config(grid_height, grid_width);
    return grid_config;
}
