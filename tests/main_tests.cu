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

void 
aligned_rect(int width, int height, u8* buffer)
{
    const auto centre_x = width  / 2;
    const auto centre_y = height / 2;

    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
            buffer[i*width+j] = 0;
    }

    for(int i= centre_y - 100; i<centre_y+100; i++)
    {
        for(int j=centre_x-5; j<centre_x+5; j++)
            buffer[i*width+j] = 1;
    }
    return;
}

void 
true_rect(int width, int height, u8* buffer)
{
    const auto centre_x = width  / 2;
    const auto centre_y = height / 2;

    int rh = 63, rw = 9;
    
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
            buffer[i*width+j] = 0;
    }

    int i = centre_y+rh/2, j = centre_x;
    int base = centre_x;
    for(int x=0; x<rh/2; x++){
        for(j = -rw/2; j<= +rw/2; j++)
            buffer[i*width + base + j] = 1;
        i = i-1;
        for(j = -rw/2; j< +rw/2; j++)
            buffer[i*width + base + j] = 1;
        i = i-1;
        base = base-1;
    }
    for(j = -rw/2; j<= +rw/2; j++)
    {
        buffer[i*width + base + j] = 1;
    }
    return;
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s <width> <height> <obst-radius> <timesteps>\n", argv[0]);
        exit(1);
    }

    // reading width, height and radius of obstacle
    // from command line arguments
    size_t width { atol(argv[1]) }, height { atol(argv[2]) };
    double radius;
    sscanf(argv[3], "%lf", &radius);

    int timesteps = atoi(argv[4]);

    printf("width: %ld, height: %ld, radius %lf\n", width, height, radius); 

    const auto centre_x = width  / 2;
    const auto centre_y = height / 2;
    const auto grid_sz = width * height;

    // generate channel metadata (a.k.a velocities)
    const auto base_v = velocity2(base_velocity_vec);
    const auto ch = generate_fhp1_velocities(base_v);

    long seed = 1024;

    // buffer for storing obstacle information
    u8 *buffer = new u8 [width * height];
    double *occup = new double[width*height];
    double *mx = new double[width*height];
    double *my = new double[width*height];

    // generating an obstacle
    // initialize_cylindrical_obstacle<u8>(buffer, width, height, centre_x, centre_y, radius);
    true_rect(width, height, buffer);
    // aligned_rect(width, height, buffer);


    // channel-wise occupancy probabilities for initialization
    double h_prob[] = { 0.7, 0.5, 0.2, 0.1, 0.2, 0.5 };

    const dim3 block_config(8, 8);
    const dim3 grid_config = make_tiles(block_config, width, height);
    fprintf(stderr, "(%d, %d, %d), ", grid_config.x, grid_config.y, grid_config.z);
    fprintf(stderr, "(%d, %d, %d)\n", block_config.x, block_config.y, block_config.z);

    // initializing grid
    fhp1_grid fhp(width, height, ch, seed, h_prob, buffer);

    momentum<<<grid_config, block_config>>>(fhp.device_grid, fhp.device_channels, 
        fhp.mx, fhp.my, fhp.ocpy, fhp.width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError( ));

    std::ofstream ioutput("data/ioutput.csv"), ipx("data/ipx.csv"), ipy("data/ipy.csv"),
        iocpy("data/ioccupancy.csv");

    fhp.get_output(buffer, mx, my, occup);

    fhp.create_csv(ioutput, buffer);
    fhp.create_csv(ipx, mx);
    fhp.create_csv(ipy, my);
    fhp.create_csv(iocpy, occup);

    double *avx, *avy, *aoc;
    cudaMalloc((void **) &avx, grid_config.x*grid_config.y*sizeof(double));
    cudaMalloc((void **) &avy, grid_config.x*grid_config.y*sizeof(double));
    cudaMalloc((void **) &aoc, grid_config.x*grid_config.y*sizeof(double));

    // time evolution
    evolve_non_local<<<grid_config, block_config>>>(fhp.device_grid, fhp.state, fhp.width, 
        fhp.height, timesteps, fhp.device_channels, fhp.mx, fhp.my, fhp.ocpy,
        avx, avy, aoc);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    std::ofstream output("data/output.csv"), px("data/px.csv"), py("data/py.csv"),
        ocpy("data/occupancy.csv");

    std::ofstream metadata("data/meta.txt");
    metadata << width << ", " << height << "\n";

    fhp.get_output(buffer, mx, my, occup);

    fhp.create_csv(output, buffer);
    fhp.create_csv(px, mx);
    fhp.create_csv(py, my);
    fhp.create_csv(ocpy, occup);


    std::ofstream apx("data/avx.csv"), apy("data/avy.csv"), aocp("data/aocpy.csv");

    double* hpx = new double[grid_config.x*grid_config.y];
    double* hpy = new double[grid_config.x*grid_config.y];
    double* hoc = new double[grid_config.x*grid_config.y];

    cudaMemcpy(hpx, avx, grid_config.x*grid_config.y*sizeof(double),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(hpy, avy, grid_config.x*grid_config.y*sizeof(double),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(hoc, aoc, grid_config.x*grid_config.y*sizeof(double),
        cudaMemcpyDeviceToHost);

    // for(int i=0; i< grid_config.x*grid_config.y; i++)
    //     printf("%lf ", hpx[i]);
    // printf("\n");
    fhp.create_csv(apx, hpx, grid_config.x, grid_config.y);
    fhp.create_csv(apy, hpy, grid_config.x, grid_config.y);
    fhp.create_csv(aocp, hoc, grid_config.x, grid_config.y);
    
    // TODO print output 
    momentum<<<grid_config, block_config>>>(fhp.device_grid, fhp.device_channels, 
        fhp.mx, fhp.my, fhp.ocpy, fhp.width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError( ));

    std::ofstream fpx("data/fpx.csv"), fpy("data/fpy.csv"), focpy("data/foccupancy.csv");

    fhp.get_output(buffer, mx, my, occup);

    fhp.create_csv(fpx, mx);
    fhp.create_csv(fpy, my);
    fhp.create_csv(focpy, occup);

    cudaFree(avx);
    cudaFree(avy);
    cudaFree(aoc);
    delete[] hpx; delete[] hpy; delete[] hoc;
    delete[] occup;
    delete[] buffer;
    return 0;
}
