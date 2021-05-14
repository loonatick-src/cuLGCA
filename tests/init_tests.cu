#include "velocity.hpp"
#include "helper_types.hpp"
#include "helper_functions.hpp"
#include "init.hpp"
#include "minunit.h"
#include "dbg.h"

#include <iostream>
#include <array>
#include <cmath>
#include <vector>


const char *test_velocity_gen()
{
    const auto rad = M_PI/3.0;
    const auto threshold = 0.0001l;
    const double c = cos(rad);
    const double s = sin(rad);
    const auto c_error = flerror (c, 0.5);

    mu_assert(flerror(c, 0.5) < threshold, "Go and revise trigo");
    mu_assert(flerror(s, sqrt(3.0)/2.0) < 0.0001, "Go and revise trigo");

    velocity2 base_velocity( std::array<double, 2> {1.0, 0.0} );
    const auto velocities = generate_velocities2<6, 1, 1>(base_velocity);

    mu_assert(velocities.size() == 6, "Incorrect channel count");
    const auto v_0 = velocities[0];
    mu_assert(v_0 == base_velocity, "Base velocity not stored.");
    const auto v_1 = velocities[1];
    debug("v_1 %lf %lf", v_1[0], v_1[1]);
    mu_assert(flerror(v_1[0], c) < threshold, "Bad rotations 1");
    mu_assert(flerror(v_1[1], s) < threshold, "Bad rotations 1");
    const auto v_2 = velocities[2];
    mu_assert(flerror(v_2[0], -c) < threshold, "Bad rotations 2");
    mu_assert(flerror(v_2[1], s)  < threshold, "Bad rotations 2");
    const auto v_3 = velocities[3];
    mu_assert(v_3[0] == -1.0 && v_3[1] == 0, "Bad rotations 3");
    const auto v_4 = velocities[4];
    mu_assert(flerror(v_4[0], -c) < threshold, "Bad rotations 4");
    mu_assert(flerror(v_4[1], -s) < threshold, "Bad rotations 4");
    const auto v_5 = velocities[5];
    mu_assert(flerror(v_5[0], c)  < threshold, "Bad rotations 5");
    mu_assert(flerror(v_5[1], -s) < threshold, "Bad rotations 5");
    return NULL;
}

const char *all_tests()
{
    mu_suite_start();

    mu_run_test(test_velocity_gen);

    return NULL;
}

RUN_TESTS(all_tests);
