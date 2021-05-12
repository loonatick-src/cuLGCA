#pragma once

#include "helper_types.hpp"
#include "velocity.hpp"
#include <cmath>
#include <vector>


// can be specialized for fhp-1, fhp-2, fhp-3, gbl
template <u32 symmetry, int level_0, int level_max>
std::vector<velocity2>
generate_velocities2(velocity2 base_velocity);


inline
std::vector<velocity2>
generate_fhp1_velocities(velocity2 base_velocity);
