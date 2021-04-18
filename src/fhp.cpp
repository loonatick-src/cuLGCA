#include <dbg.h>
#include <iostream>
#include <cstdint>
#include <utility>
#include <vector>
#include <cmath>


typedef struct 
{
    uint8_t state: 6;
    uint8_t extra: 2;
} fhpcell_t;

typedef std::pair<double, double> velocity_t;

auto
momentum(fhpcell_t s, std::vector<velocity_t> v)
{
    // _ _ _ _ _ _ _ _
    velocity_t rv = std::make_pair(0.0l, 0.0l);
    // unit mass
    // (s & mask)>>i * v.f v.s
    for (uint8_t i = 0; i < v.size(); i++)
    {
        rv.first += v[i].first * ((s.state >> i) & 1);
        rv.second += v[i].second * ((s.state >> i) & 1);
    }
    return rv;
}


inline
auto
momentumCmp(velocity_t v1, velocity_t v2, double norm_threshold)
{
    // L2 norm of difference of momenta
    norm_threshold = fabs(norm_threshold);  // just in case
    auto diff = sqrt((v1.first - v2.first) * (v1.first - v2.first) 
            + (v1.second - v2.second) * (v1.second - v2.second));
    return (diff < norm_threshold);
}


velocity_t
rotate(velocity_t v, double angle)
{
    // assuming radians
    velocity_t rv;
    rv.first = cos(angle)*v.first
               + sin(angle) * v.second;
    rv.second = -sin(angle) * v.first
               + cos(angle) * v.second;
    return rv;
}


template<typename T>
inline
auto
countSetBits(T value)
{
    size_t number_of_set_bits = 0;
    while (value)
    {
        value &= (value-1);
        number_of_set_bits++;
    }
    return number_of_set_bits;
}


inline
auto
particleCount(fhpcell_t node)
{
    return countSetBits(node.state);
}

inline
auto
clean(velocity_t& v, double threshold)
{
    if (fabs(v.first) < threshold)
        v.first = 0.0l;
    
    if (fabs(v.second) < threshold)
        v.second = 0.0l;
}


int main(int argc, char *argv[])
{
    std::vector<velocity_t> velocities;
    velocity_t v0 = std::make_pair(1.0l, 0.0l);
    for (int i = 0; i < 6; i++)
    {
        velocities.push_back(v0);
        v0 = rotate(v0, M_PI/3.0l);
    }

    fhpcell_t x;
    x.state = 0b110100;
    x.extra = 0b00;
    for (auto v: velocities)
    {
        std::cout << v.first << ' ' << v.second << '\n';
    }
    std::cout << std::endl;

    velocity_t p = momentum(x, velocities);
    std::cout << p.first << ' ' << p.second << '\n';

    std::cerr << "Testing set bits counter" << std::endl;
    std::cout << "0b100101: " << countSetBits(0b100101) << '\n';
    std::cout << "0b0000: " << countSetBits(0b0000) << '\n';
    std::cout << "0b1: " << countSetBits(0b01) << '\n';

    std::cerr << "Testing particle counter" << std::endl;
    std::cout << "x {0b110100, 0b01}: " << particleCount(x) << '\n';
    return 0;
}
