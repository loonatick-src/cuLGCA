#include "dbg.h"
#include "helper_functions.hpp"
#include "helper_types.hpp"
#include "obstacle.hpp"
#include "velocity.hpp"
#include "fhp.hpp"
#include "init.hpp"
#include <cstdlib>
#include <cassert>


constexpr std::array<double, 2> base_velocity_vec { 1.0, 0.0 };


int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <width> <height> <obst-radius>\n", argv[0]);
        exit(1);
    }

    // reading width, height and radius of obstacle
    // from command line arguments
    size_t width { atol(argv[1]) }, height { atol(argv[2]) };
    double radius;
    sscanf(argv[3], "%lf", &radius);

    const auto centre_x = width  / 2;
    const auto centre_y = height / 2;
    const auto grid_sz = width * height;

    // generate channel metadata (a.k.a velocities)
    const auto base_v = velocity2(base_velocity_vec);
    const auto ch = generate_fhp1_velocities(base_v);

    long seed = 3;

    // buffer for storing obstacle information
    u8 *buffer = new u8 [width * height];

    // generating an obstacle
    initialize_cylindrical_obstacle<u8>(buffer, width, height, centre_x, centre_y, radius);

    // channel-wise occupancy probabilities for initialization
    double h_prob[] = { 0.9, 0.9, 0.4, 0.3, 0.4, 0.9 };

    const dim3 block_config(8, 8);
    const dim3 grid_config = make_tiles(block_config, width, height);

    // initializing grid
    fhp1_grid fhp(width, height, ch, seed, h_prob, buffer);
    // time evolution
    evolve<<<grid_config, block_config>>>(fhp.device_grid, fhp.state, fhp.width, fhp.height, 1000);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    momentum<<<grid_config, block_config>>>(fhp.device_grid, fhp.device_channels, 
        fhp.mx, fhp.my, fhp.ocpy, fhp.width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // copying back to buffer
    cudaMemcpy(buffer, fhp.device_grid,  grid_sz * sizeof(u8), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaGetLastError());

    u8 *occup = new u8[width*height];
    double *mx = new double[width*height];
    double *my = new double[width*height];

    std::ofstream output("data/output.csv"), px("data/px.csv"), py("data/py.csv"),
        ocpy("data/occupancy.csv");

    fhp.get_output(buffer, mx, my, occup);

    fhp.create_csv(output, buffer);
    fhp.create_csv(px, mx);
    fhp.create_csv(py, my);
    fhp.create_csv(ocpy, occup);

    // TODO print output 
    delete[] occup;
    delete[] buffer;
    return 0;
}
