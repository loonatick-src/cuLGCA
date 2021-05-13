#include "dbg.h"
#include "minunit.h"
#include "velocity.hpp"
#include <array>
#include <cmath>

typedef std::array<double, 2> v2;

template <typename numeric_type>
inline numeric_type
sq(numeric_type v)
{
    return v* v;
}

const auto sqd = sq<double>;

const char *test_velocity_init()
{
    debug("test_velocity_init");
    const double vx = 0.5;
    const double vy = 0.8662;
    std::array<double, 2> arr {vx, vy};
    velocity2 v {arr};
    // warning: conversion to char * from string literal is deprecated
    mu_assert(v[0] == vx,
            "Velocity x-component incorrectly assigned");
    mu_assert(v[1] == vy,
            "Velocity y-component incorrectly assigned");

    return NULL;
}

const char *test_velocity_algebra()
{
    debug(" ");
    const double vx1 = 0.2;
    const double vy1 = 0.3;
    const double vx2 = -0.23;
    const double vy2 = 1.23;
    const double vax = vx1 + vx2;
    const double vay = vy1 + vy2;
    const double vdx = vx1 - vx2;
    const double vdy = vy1 - vy2;

    const v2 a1 {vx1, vy1};
    const v2 a2 {vx2, vy2};
    const velocity2 v_1(a1);
    const velocity2 v_2(a2);
    const auto va =  v_1 + v_2;
    const auto vd = v_1 - v_2;
    
    mu_assert(va[0] == vax,
            "Incorrect velocity addition");
    mu_assert(va[1] == vay,
            "Incorrect velocity addition");
    mu_assert(vd[0] == vdx,
            "Incorrect velocity subtraction");
    mu_assert(vd[1] == vdy,
            "Incorrect velocity subtraction");

    const auto speed1 = v_1.speed();
    const auto speed2 = v_2.speed();

    mu_assert(speed1 == sqrt(vx1*vx1 + vy1*vy1),
            "Incorrect speed calculation.");
    mu_assert(speed2 == sqrt(vx2*vx2 + vy2*vy2),
            "Incorrect speed calculation");

    const auto vdiff = v_1.norm_diff(v_2);
    const auto ke1 = v_1.kinetic_energy();
    const auto ke2 = v_2.kinetic_energy();

    mu_assert(vdiff == sqrt(sqd(vx1 - vx2) + sqd(vy1 - vy2)),
            "Incorrect norm_diff calculation");
    mu_assert(ke1 == 0.5 * (sqd(vx1)+sqd(vy1)),
            "Incorrect kinetic energy calculation");
    mu_assert(ke2 == 0.5 * (sqd(vx2)+sqd(vy2)),
            "Incorrect kinetic energy calculation");

    return NULL;
}


const char *test_rotate_and_clean()
{
    const auto rad = M_PI/3.0l;
    const auto c = cos(rad);
    const auto s = sin(rad);
    std::array<double, 2> base_v {1.0 , 0.0}; 
    const velocity2 v_1(base_v);
    velocity2 v_2(v_1.rotate(rad));
    velocity2 v_3(v_2.rotate(rad));
    velocity2 v_4(v_3.rotate(rad));
    v_4.clean_perturbations(0.01);
    velocity2 v_5(v_4.rotate(rad));
    velocity2 v_6(v_5.rotate(rad));
    velocity2 v_7(v_6.rotate(rad));
    v_7.clean_perturbations(0.01);

    mu_assert(v_7 == v_1, "Rotations did not return to base velocity.");
    v_4.rotate_inplace(M_PI);
    v_4.clean_perturbations(0.01);
    mu_assert(v_4 == v_1, "Rotation through 180 degrees incorrect");

    return NULL;
}


const char *all_tests()
{
    mu_suite_start();

    mu_run_test(test_velocity_init);
    mu_run_test(test_velocity_algebra);
    mu_run_test(test_rotate_and_clean);

    log_info("Testing complete");
    return NULL;
}

RUN_TESTS(all_tests);
