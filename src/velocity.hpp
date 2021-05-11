#pragma once
#include <array>
#include <cmath>
#include "velocity.hpp"

template <size_t dim, typename float_type>
struct velocity
{
    // buffer for storing float values
    std::array<float_type, dim> velocity_vec;

    // constructor from std::array
    velocity(const std::array<float_type, dim>& arr) :
        velocity_vec {arr} {}

    velocity(const velocity& v) = default;
    velocity() = default;

    float_type&
    operator[](const size_t& index)
    {
        float_type& rv = velocity_vec[index];
        return rv;
    }

    const float_type&
    operator[](const size_t& index) const
    {
        return velocity_vec[index];
    }

    auto
    operator+(const velocity& v2) const
    {
        std::array<float_type, dim> arr;
        for (size_t i = 0; i < dim; i++)
        {
            arr[i] = velocity_vec[i] + v2[i];
        }
        return velocity(arr);
    }

    velocity<dim, float_type>
    operator-(const velocity& v2) const
    {
        velocity<dim, float_type> v_out;
        for (size_t i = 0; i < dim; i++)
        {
            v_out[i] = velocity_vec[i] - v2[i];
        }
        return v_out;
    }


    inline float_type
    speed()
    {
        float_type rv = 0;
        for (auto v : velocity_vec)
        {
            rv += v * v;     
        }
        return sqrt(rv);
    }


    inline
    float_type
    kinetic_energy()
    {
        float_type rv = 0;
        for (auto v : velocity_vec)
        {
            rv += v * v;
        }
        return rv / 2;
    }


    inline
    float_type
    norm_diff(const velocity<dim, float_type>& v2)
    {
        const auto v = (*this) - v2; 
        return v.speed();
    }
};
