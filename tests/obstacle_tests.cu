#include "dbg.h"
#include "minunit.h"
#include "obstacle.hpp"
#include "helper_types.hpp"
#include "helper_functions.hpp"


constexpr size_t width  {100};
constexpr size_t height {100};
constexpr double radius { (double) (min<size_t>(width, height)/4) };
constexpr size_t c_x { width / 2 };
constexpr size_t c_y { height/ 2 };

const char *test_lattice_vector_length()
{
    const auto len1 {lattice_vector_length(1,0, u1_x, u1_y, u2_x, u2_y)};
    mu_assert(len1 == 1.0, "Incorrect lattice vector length calculation");
    const auto len2 = lattice_vector_length(1,1,u1_x, u1_y, u2_x, u2_y);
    mu_assert(fabs(len2 - 1.0) < 0.0001, "Incorrect lattice vector length calculation");
    const auto len3 { lattice_vector_length(3, 6, u1_x, u1_y, u2_x, u2_y) }; 
    mu_assert(flerror(len3, 5.19615242) < 0.001, "Incorrect lattice vector length calculation");
    return NULL;
}

const char *test_obstacle_creation()
{
    log_info("Not really a test. Check printed output");
    int *buff = new int[width * height];
    
    initialize_cylindrical_obstacle<int>(buff, width, height, c_x, c_y, radius);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError( ));

    print_grid(buff, width, height);
    delete[] buff;
    return NULL;
}


const char *all_tests()
{
    mu_suite_start();
    mu_run_test(test_lattice_vector_length);
    // mu_run_test(test_obstacle_creation);
    return NULL;
}

RUN_TESTS(all_tests);
