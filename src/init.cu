#include "helper_types.hpp"
#include "velocity.hpp"
#include <cmath>
#include <vector>
#include "init.hpp"


// can be specialized for fhp-1, fhp-2, fhp-3, gbl
template <u32 symmetry, int level_0, int level_max>
std::vector<velocity2>
generate_velocities2(velocity2 base_velocity)
{
    auto rad = 2 * M_PI / symmetry; 
    std::vector<velocity2> velocities;

    if (level_0 == 0)
    {
        velocities.push_back(velocity2(std::array<double, 2> {0.0, 0.0} ));
    }

    int lvl = 1;
    u32 sym = symmetry;
    while (lvl <= level_max)
    {
        velocities.push_back(base_velocity);
        velocity2 v(base_velocity);
        for (u32 i = 1; i < sym; i++)
        {
            v = v.rotate(rad);
            v.clean_perturbations(0.01);
            velocities.push_back(v);
        }
        sym *= 2;
        rad /= 2;
        lvl++;
    }

    return velocities;
}


auto
generate_fhp1_velocities2(velocity2 base_velocity)
{
    return generate_velocities2<6, 1, 1>(base_velocity);
}
