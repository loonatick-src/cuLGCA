#include "minunit.h"
#include "obstacle.hpp"
#include "helper_functions.hpp"


const char *test_lattice_vector_length()
{
    const auto len {lattice_vector_length(1,0, u1_x, u1_y, u2_x, u2_y)};
    mu_assert(len == 1.0, "Incorrect lattice vector length calculation");
    return NULL;
}

const char *test_obstacle_creation()
{
    return NULL;
}


const char *all_tests()
{
    mu_suite_start();
    mu_run_test(test_lattice_vector_length);
    return NULL;
}

RUN_TESTS(all_tests);
