#pragma once
#include <iostream>

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


template<typename word>
void
print_grid(word *grid, size_t width, size_t height)
{
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width-1; col++)
        {
            std::cout << grid[row * width + col];
        }
        std::cout << grid[row * width + width-1] << '\n';
    }
}


template <typename numeric_type>
constexpr auto
max(constexpr numeric_type a, constexpr numeric_type b)->decltype(true ? a : b)
{
    return (a > b ? a : b); 
}


template <typename numeric_type>
constexpr auto
min(constexpr numeric_type a, constexpr numeric_type b)->decltype(true ? a : b)
{
    return (a < b ? a : b); 
}
