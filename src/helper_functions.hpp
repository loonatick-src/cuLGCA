#pragma once

inline
const dim3
make_tiles(const dim3& block_config, size_t width, size_t height)
{
    const size_t grid_width = (size_t) ceil( ((double)width/block_config.x));
    const size_t grid_height= (size_t) ceil( ((double) height)/block_config.y);
    const dim3 grid_config(grid_height, grid_width);
    return grid_config;
}


double
flerror(double of, double against)
{
    return fabs(of - against); 
}


template <typename numeric_type>
inline numeric_type
sq(numeric_type v)
{
    return v* v;
}

const auto sqd = sq<double>;


