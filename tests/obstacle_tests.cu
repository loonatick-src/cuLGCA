#include "minunit.h"
#include "obstacle.hpp"
#include "helper_functions.hpp"

std::string filename {"tests/obstacle_tests.dat"};

const char *test_lattice_vector_length()
{
    const auto len1 {lattice_vector_length(1,0, u1_x, u1_y, u2_x, u2_y)};
    mu_assert(len1 == 1.0, "Incorrect lattice vector length calculation");
    const auto len2 = lattice_vector_length(1,1,u1_x, u1_y, u2_x, u2_y);
    debug("1,1: %lf", len2);
    mu_assert(fabs(len2 - 1.0) < 0.0001, "Incorrect lattice vector length calculation");
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
