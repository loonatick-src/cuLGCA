# pragma once
#include <vector>
#include <cassert>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include "dbg.h"
#include "velocity.hpp"


typedef uint8_t u8;
typedef uint32_t u32;
__constant__ u8 d_eq_class_size[128];
__constant__ u8 d_state_to_eq[128];
__constant__ u8 d_eq_classes[128];


__global__
void setup_kernel(curandState_t *state, size_t width, size_t height, long seed=1234);


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
struct fhp_grid
{
    word *device_grid;
    double *device_channels;
    curandState_t *state;
    double *mx, *my;
    double* ocpy;
    double *probability = nullptr;

    const size_t width, height;
    long seed;
    const std::vector<velocity2> channels;

    
    fhp_grid(size_t w, size_t h,
            std::vector<velocity2> velocities, word *buffer, long seed) :
        width {w}, height{h}, channels {velocities}, seed {seed}
    {
        static_assert(sizeof(word)*8 > channel_count);
        assert(buffer != NULL);
        assert(channel_count == velocities.size());

        const auto grid_sz = width * height;
        const auto mem_sz = grid_sz * sizeof(word);
        const auto channel_mem_sz = 2 * sizeof(double) * channel_count;
        double *temp = new double [2 * channel_count];
        for (size_t i = 0; i < channels.size(); i+=1)
        {
            temp[2*i] = channels[i][0]; 
            temp[2*i+1] = channels[i][1];
        }

        cudaMalloc((void **) &device_channels, channel_mem_sz);
        // ALLOW OPTION FOR DEVICE TO DEVICE COPY?
        cudaMalloc((void **) &device_grid, mem_sz);
        cudaMalloc((void **) &ocpy, mem_sz);
        cudaMalloc((void **) &mx, grid_sz*sizeof(double));
        cudaMalloc((void **) &my, grid_sz*sizeof(double));
        cudaMalloc((void **) &state, width*height*sizeof(curandState_t));
        fprintf(stderr, "start addr: %p, bytes: %ld", state, width*height*sizeof(curandState_t));
        // If we already have grid, do we need to store this?
        // cudaMalloc((void **) &probability, channel_count*sizeof(double));

        // ALLOW OPTION FOR DEVICE TO DEVICE COPY?
        cudaMemcpy(device_grid, buffer, mem_sz,
                cudaMemcpyHostToDevice);
        cudaMemcpy(device_channels, temp, channel_mem_sz,
                cudaMemcpyHostToDevice);
        // cudaMemcpy(probability, probs, prob_sz,
        //         cudaMemcpyHostToDevice);

        // Setup curand states
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
        dim3 grid(width/BLOCK_WIDTH, height/BLOCK_HEIGHT);
        setup_kernel<<<grid, block>>>(state, width, height, seed);
        delete[] temp;

        // word* output = (word*) malloc(width*height*sizeof(word));
        // cudaMemcpy(output, device_grid, mem_sz,
        //         cudaMemcpyDeviceToHost);

        setup_constants(this);
        
        // const int GRID_SIZE = 8;
        // std::cout<<"In Initializer:\n";
        // for(int i=0; i<GRID_SIZE; i++) 
        // {
        //     for(int j=0; j<GRID_SIZE; j++){
        //         u8 t = output[i*GRID_SIZE+j];
        //         // std::bitset<8> x(t);
        //         std::cout << (int)t <<" ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "\n\n";

    }

    fhp_grid(size_t w, size_t h,
        std::vector<velocity2> velocities, long seed, double *prob, word* obstacle):
    width {w}, height{h}, channels {velocities}, seed {seed}
    {
        static_assert(sizeof(word)*8 > channel_count);
        assert(prob != NULL);
        assert(obstacle != NULL);
        assert(channel_count == velocities.size());

        const auto grid_sz = width * height;
        const auto mem_sz = grid_sz * sizeof(word);
        const auto channel_mem_sz = 2 * sizeof(double) * channel_count;
        double *temp = new double [2 * channel_count];
        for (size_t i = 0; i < channels.size(); i+=1)
        {
            temp[2*i] = channels[i][0]; 
            temp[2*i+1] = channels[i][1];
        }
        word* dev_obstacle;

        cudaMalloc((void **) &device_channels, channel_mem_sz);
        cudaMalloc((void **) &device_grid, mem_sz);
        cudaMalloc((void **) &ocpy, grid_sz*sizeof(double));
        cudaMalloc((void **) &mx, grid_sz*sizeof(double));
        cudaMalloc((void **) &my, grid_sz*sizeof(double));
        cudaMalloc((void **) &state, width*height*sizeof(curandState_t));
        cudaMalloc((void **) &probability, channel_count*sizeof(double));

        cudaMalloc((void **) &dev_obstacle, grid_sz*sizeof(word));

        gpuErrchk(
        cudaMemcpy(probability, prob, channel_count*sizeof(double),
            cudaMemcpyHostToDevice)
        );
        gpuErrchk(
        cudaMemcpy(dev_obstacle, obstacle, grid_sz*sizeof(word),
            cudaMemcpyHostToDevice)
        );
        gpuErrchk(
            cudaMemcpy(device_channels, temp, channel_mem_sz,
                cudaMemcpyHostToDevice)
        );

            // Setup curand states
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
        dim3 grid(width/BLOCK_WIDTH, height/BLOCK_HEIGHT);
        setup_kernel<<<grid, block>>>(state, width, height, seed);

        initialize_grid<<<grid, block>>>(device_grid, dev_obstacle, probability, state, width);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError( ));


        delete[] temp;
        cudaFree(dev_obstacle);

        setup_constants(this);

    }


    ~fhp_grid()
    {
        cudaFree(device_grid);
        cudaFree(mx); cudaFree(my);
        cudaFree(ocpy);
        cudaFree(state);
        if (probability != nullptr) cudaFree(probability);
        cudaFree(device_channels);
    }

    template <size_t timesteps>
    void start_evolution();
    
    __device__
    void collide(curandState_t *localstate, word *state);
    
        
    velocity2
    calculate_momentum(word state);

    uint32_t
    number_of_particles(word n);

    void 
    get_output(word* output, double *p_x, double* p_y, double* o)
    {
        assert(output != NULL);
        assert(p_x != NULL);
        assert(p_y != NULL);
        assert(o != NULL);
        const auto grid_sz = width * height;
        const auto mem_sz = grid_sz * sizeof(word);
        cudaMemcpy(output, device_grid, mem_sz,
                cudaMemcpyDeviceToHost);
        cudaMemcpy(p_x, mx, grid_sz*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemcpy(p_y, my, grid_sz*sizeof(double),
                cudaMemcpyDeviceToHost);
        cudaMemcpy(o, ocpy, grid_sz*sizeof(double),
                cudaMemcpyDeviceToHost);
        
        return;
    }

    void 
    create_csv(std::ofstream &stream, word *buf)
    {
        for(int i=0; i<height; i++)
        {
            int j;
            for(j=0; j<width-1; j++){
                std::bitset<8> t(buf[i*width + j]);
                stream << t << ", " ;
            }
            std::bitset<8> t(buf[i*width + j]);
            stream << t;
            stream << "\n";
        }
        stream << "\n";
    }

    void 
    create_csv(std::ofstream &stream, double *buf)
    {
        for(int i=0; i<height; i++)
        {
            int j;
            for(j=0; j<width-1; j++)
                stream << buf[i*width + j] << ", " ;
            stream << buf[i*width + j];
            stream << "\n";
        }
        stream << "\n";
    }

};
    

typedef fhp_grid<uint8_t, 6, 8, 8> fhp1_grid;

void
setup_constants(fhp1_grid *grid);

// Device helpers 
template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT = BLOCK_WIDTH>
__device__
word stream(int local_row, int local_col, word sdm[BLOCK_WIDTH+2][BLOCK_HEIGHT+2]);

template <typename word>
__device__
word occupancy(word state);

template <typename word>
__device__
auto momentum_x(word state, double *device_channels)->double;

template <typename word>
__device__
auto momentum_y(word state, double *device_channels)->double;


// kernels
__global__
void
evolve(u8* device_grid, curandState_t* randstate, int width, int height, int timesteps, 
    double* device_channels, double* mx, double *my, double* ocpy);

__global__
void 
momentum(u8* device_grid, double* device_channels, double* mx, double *my, double* ocpy, int width);

__global__
void 
initialize_grid(u8* device_grid, u8* device_obstacle, double* probability, 
    curandState_t *randstate, int width);
