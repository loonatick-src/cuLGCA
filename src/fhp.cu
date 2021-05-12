#include "velocity.hpp"
#include "dbg.h"
#include "fhp.hpp"
#include "helper_types.hpp"

const size_t default_bw = 8;
const size_t default_bh = 8;

//TODO: do I need the template arguments in every method declaration?

template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
word
stream(int local_row, int local_col, word sdm[BLOCK_WIDTH+2][BLOCK_HEIGHT+2])
{
    u8 state = 0 | (1<<6 & (sdm[local_row][local_col]));
    u8 bit = 0x1;

    // TODO: redo with iterator using positive modulo
    state |= (bit & sdm[local_row][local_col-1]);
    bit <<= 1;
    state |= (bit & sdm[local_row+1][local_col]);
    bit <<= 1;
    state |= (bit & sdm[local_row+1][local_col+1]);
    bit <<= 1;
    state |= (bit & sdm[local_row][local_col+1]);
    bit <<= 1;
    state |= (bit & sdm[local_row-1][local_col]);
    bit <<= 1;
    state |= (bit & sdm[local_row-1][local_col-1]);

    return state;
}


template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__device__
void
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::collide(curandState *localstate, word *state)
{
    word size = d_eq_class_size[*state];
    word base_index = d_state_to_eq[*state];

    // This is from [0,...,size-1]
    float rand = curand_uniform(localstate);
    rand *= size-0.00001;
    word random_index = (word)(rand) % size; // Require curand_init for each thread

    *state = d_eq_classes[base_index + random_index];
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

__global__
void 
setup_kernel(curandState *state, size_t width, size_t height) 
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
evolve(u8* device_grid, curandState* randstate, int width, int height)
{
    __shared__ u8 sdm[default_bh+2][default_bw+2];
    const auto local_row = threadIdx.y+1;
    const auto local_col = threadIdx.x+1;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localstate = randstate[row*width + col];
    __syncthreads();

    // 1. load into shared memory
    sdm[local_row][local_col] = device_grid[row*width + col];
    // As such all these bool values can be stored in the same
    // register. Is nvcc smart enough to do this if the number
    // of registers at hand are low?
    const auto ubound_y {local_row == blockDim.y-1};
    const auto ubound_x {local_col == blockDim.x-1};
    const auto lbound_y {local_row == 0};
    const auto lbound_x {local_col == 0};
    size_t pad_row = row, pad_col = col;
    size_t local_pad_row = local_row, local_pad_col = local_col;
    // NSight is going to roast tf out of this kernel
    if (lbound_y)
    {
        pad_row = (row == 0) ? height-1 : row-1;
        local_pad_row = 0;
        sdm[local_pad_row][local_col] = device_grid[pad_row * width + col];
    } else if (ubound_y)
    {
        pad_row = (row+1) % height;
        local_pad_row = blockDim.y+1;
        sdm[blockDim.y+1][local_col] = device_grid[pad_row * width + col];
    }
    __syncthreads();
    if (lbound_x)
    {
        pad_col = (col == 0) ? width-1 : col-1;
        local_pad_col = 0;
        sdm[local_row][local_pad_col] = device_grid[row * width + pad_col];
    } else if (ubound_x)
    {
        pad_col = (col+1) % width;
        local_pad_col = blockDim.x+1;
        sdm[local_row][local_pad_col] = device_grid[row * width + pad_col];
    }
    __syncthreads();

    if (pad_row != row && pad_col != col)
    {
        sdm[local_pad_row][local_pad_col] = device_grid[pad_row*width + pad_col]; 
    }
    __syncthreads();

    // 2. Streaming
    u8 state = stream<u8, 6, 8, 8>(local_row, local_col, sdm);

    // 3. Collision
    u8 size = d_eq_class_size[state];
    u8 base_index = d_state_to_eq[state];

    // This is from [0,...,size-1]
    float rand = curand_uniform(&localstate);
    rand *= size-0.00001;
    u8 random_index = (u8)(rand) % size; // Require curand_init for each thread

    state = d_eq_classes[base_index + random_index];

    device_grid[row*width + col] = state;
    // printf("row %d, col %d: collide: %d\n", row, col, device_grid[row*width + col]);

    return;
}

template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
velocity2
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::calculate_momentum(word state)
{
    const word bit = 0x1;
    velocity2 v {{0.0, 0.0}};
    for (auto i = 0; i < channel_count; i++)
    {
        const word temp = state & bit;
        if (temp == 1)
        {
            const auto vt = channels[i];
            auto vx = vt[0];
            auto vy = vt[1];
            v[0] += vx;
            v[1] += vy;
        }
        state >>= 1;
    }
    return v;
}

template <typename word, u8 channel_count, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
uint32_t
fhp_grid<word, channel_count, BLOCK_WIDTH, BLOCK_HEIGHT>::number_of_particles(word n)
{
    uint32_t count = 0;
    while (n)
    {
        n &= (n-1);
        count++;
    }
    return count;
}

void
setup_constants(fhp1_grid *grid)
{
    const double threshold = 1.0e-6;
    std::vector< std::vector< u8 > > equivalence_classes;

    const int array_length = 128;
    // Cuda copy-able memory
    u8* h_eq_classes = (uint8_t*) malloc(array_length*sizeof(uint8_t));
    u8* h_state_to_eq = (uint8_t*) malloc(array_length*sizeof(uint8_t));
    u8* h_eq_class_size = (uint8_t*) malloc(array_length*sizeof(uint8_t));

    u8 state = 0;

    // Populate equivalence class table
    do 
    {
        auto momentum = grid->calculate_momentum(state);
        auto particle_count = grid->number_of_particles(state);
        bool found = false;
        for (auto& eqclass : equivalence_classes)
        {
            auto it = eqclass.begin();
            if (particle_count == grid->number_of_particles(*it)
                    && ((grid->calculate_momentum(*it)).norm_diff(momentum)) < threshold)
            {
                found = true;
                eqclass.push_back(state);
                break;
            }
        }
        if (!found)
        {
            std::vector<u8> newclass;
            newclass.push_back(state);
            equivalence_classes.push_back(newclass);
        }
        state++;
    } while (state != (1<<6));

        // Convert vec < vec <> > to array and Populate state_to_eq_class table
    // for normal collision 
    int index = 0;
    for (auto& eqcl : equivalence_classes)
    {
        auto start_id = index;
        auto size = eqcl.size();
        for (auto& state : eqcl)
        {
            h_eq_classes[index] = state;
            h_state_to_eq[state] = start_id;
            h_eq_class_size[state] = size;
            index++;
        }
    }

    int BIT_MASK_3 = 0b111, BIT_MASK_6 = 0b111000;

    // Lookup table entries for bounce back collisions
    // ASSUMPTION: 7th bit is set to 1 if bounce back happens in this point
    for(int i=64; i<128; i++) 
    {
        uint8_t vel_vec = i - 64;
        uint8_t bounced = (1<<6) | (((vel_vec&BIT_MASK_3)<<3)|((vel_vec&BIT_MASK_6)>>3));
        h_eq_classes[i] = bounced;
        h_state_to_eq[i] = i;
        h_eq_class_size[i] = 1;
    }

    cudaMemcpyToSymbol(d_state_to_eq, h_state_to_eq, 128*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_eq_classes, h_eq_classes, 128*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_eq_class_size, h_eq_class_size, 128*sizeof(uint8_t));
    gpuErrchk(cudaGetLastError( ));

    // for (auto& eqcl : equivalence_classes)
    // {
    //     for (auto& state : eqcl)
    //     {
    //         std::bitset<8> x(state);
    //         std::cout << "(" << (int)state << ") " << x << ' ';
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << std::endl; 
    // for(int i=0; i<array_length; i++) 
    // {
    //     std::bitset<8> x(h_eq_classes[i]);
    //     std::cout << (int)h_eq_classes[i] << ' ';
    //     if (i==63)
    //         std::cout<<"\n";
    // }
    // std::cout<<"\n\n";
    
    // for(int i=0; i<array_length; i++) 
    // {
    //     std::cout << i << ":" << (int)(h_state_to_eq[i]) << "\t";
    // }
    // std::cout<<"\n";


    free(h_eq_classes);
    free(h_state_to_eq);
    free(h_eq_class_size);
}
