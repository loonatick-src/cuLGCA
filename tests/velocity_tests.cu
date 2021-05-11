#include "minunit.h"
#include "velocity.hpp"
#include <array>

typedef velocity<2, double> velocity2;

char *test_velocity_init()
{
    debug("test_velocity_init");
    const double vx = 0.5;
    const double vy = 0.8662;
    std::array<double, 2> arr {vx, vy};
    velocity2 v {arr};
    mu_assert(v[0] == vx,
            "Velocity x-component incorrectly assigned");
    mu_assert(v[1] == vy,
            "Velocity y-component incorrectly assigned");

    return NULL;
}

char *all_tests()
{
    mu_suite_start();

    mu_run_test(test_velocity_init);

    log_info("Testing complete");
    return NULL;
}

RUN_TESTS(all_tests);
