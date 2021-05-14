#pragma once

#include "helper_types.hpp"
#include "dbg.h"
#include "helper_functions.hpp"
#include <cassert>
#include <cmath>

#define BLOCK_WIDTH 16;
#define BLOCK_HEIGHT 16;
#ifdef PI
#undef PI
#endif
#define PI 3.141592653589793

/*      0       x
 *      ------->
 *     /
 *    /
 * y\/_
 *
 */
constexpr double u1_x {1.0};
constexpr double u1_y {0.0}; 
constexpr double u2_x {-0.5};
constexpr double u2_y {-0.8660254037844386};
// unfortunately the cmath functions do not return constexpr
// constexpr double u2_x {-cos(PI/3.0)};
// constexpr double u2_y {-sin(PI/3.0)};


double
lattice_vector_length(const size_t d_1, const size_t d_2, const double b1_x, const double b1_y,
                      const double b2_x, const double b2_y)
{
    const double rv_x { d_1*b1_x + d_2*b2_x };
    const double rv_y { d_1*b1_y + d_2*b2_y };
    return (sqrt(rv_x*rv_x + rv_y*rv_y));
}


__device__
double
lattice_vector_length_d(const size_t d_1, const size_t d_2, const double b1_x, const double b1_y,
                      const double b2_x, const double b2_y)
{
    const double rv_x { d_1*b1_x + d_2*b2_x };
    const double rv_y { d_1*b1_y + d_2*b2_y };
    return (sqrt(rv_x*rv_x + rv_y*rv_y));
}


template <typename word>
__global__
void
init_cyl_obst_kernel(word *buffer, size_t width, size_t height,
        size_t centre_x, size_t centre_y, double radius)
{
    const auto row = blockDim.y*blockIdx.y + threadIdx.y;
    const auto col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < height && col < width)
    {
        const auto d_1 = labs(col - centre_x);
        const auto d_2 = labs(row - centre_y);
        const auto c_dist = lattice_vector_length_d(d_1, d_2, u1_x, u1_y, u2_x, u2_y);
        buffer[row*width + col] = (c_dist <= radius ? 1 : 0);
    }

    return;
}

template <typename word>
auto
initialize_cylindrical_obstacle(word *buffer, const size_t width, const size_t height,
        size_t centre_x, size_t centre_y, double radius)
{
    assert(buffer != nullptr);
    const dim3 block_config(16, 16);
    const dim3 grid_config = make_tiles(block_config, width, height);
    const size_t mem_sz = width * height * sizeof(word);
   
    word *device_buffer;
    cudaMalloc((void **) &device_buffer, mem_sz);
    gpuErrchk( cudaGetLastError() );
    cudaMemcpy(device_buffer, buffer, mem_sz, cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );

    init_cyl_obst_kernel<word><<<grid_config, block_config>>>(device_buffer, width, height,
            centre_x, centre_y, radius);

    cudaMemcpy(buffer, device_buffer, mem_sz, cudaMemcpyDeviceToHost);
    gpuErrchk( cudaGetLastError() );
    cudaFree(device_buffer);
    gpuErrchk( cudaGetLastError() );
    return 0;
error:
    return 1;
}



