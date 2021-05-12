#pragma once
#include <array>
#include <cmath>

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

    velocity(float_type arr[dim]) :
        velocity_vec {arr} { }

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
    operator==(const velocity& v2) const
    {
        for (size_t i = 0; i < dim; i++)
        {
            if (velocity_vec[i] != v2[i])
                return false;
        }
        return true;
    }

    velocity&
    operator=(const velocity&) = default;

    auto
    operator+(const velocity& v2) const
    {
        std::array<float_type, dim> arr;
        for (size_t i = 0; i < dim; i++) {
            arr[i] = velocity_vec[i] + v2[i];
        }
        return velocity(arr);
    }

    velocity<dim, float_type>
    operator-(const velocity& v2) const
    {
        velocity<dim, float_type> v_out;
        for (size_t i = 0; i < dim; i++) {
            v_out[i] = velocity_vec[i] - v2[i];
        }
        return v_out;
    }


    inline float_type
    speed() const
    {
        float_type rv = 0;
        for (auto v : velocity_vec) {
            rv += v * v;     
        }
        return sqrt(rv);
    }


    inline
    float_type
    kinetic_energy() const
    {
        float_type rv = 0;
        for (auto v : velocity_vec) {
            rv += v * v;
        }
        return rv / 2;
    }


    inline
    float_type
    norm_diff(const velocity<dim, float_type>& v2) const
    {
        const auto v = (*this) - v2; 
        return v.speed();
    }

    inline
    void clean_perturbations(float_type threshold)
    {
        for (auto& v : velocity_vec) {
            if (fabs(v) < threshold)
                v = 0;
        }
    }

    inline void
    rotate_inplace(float_type radians);

    inline velocity
    rotate(float_type radians) const;
};

typedef velocity<2, double> velocity2;


template <>
inline void
velocity2::rotate_inplace(double radians)
{
    const auto c = cos(radians);
    const auto s = sin(radians);
    const auto vx = std::get<0>(this->velocity_vec);
    const auto vy = std::get<1>(this->velocity_vec);

    const auto vx_r = c*vx - s*vy;
    const auto vy_r = s*vx + c*vy;
    std::get<0>(this->velocity_vec) = vx_r;
    std::get<1>(this->velocity_vec) = vy_r;
}


template <>
inline velocity2
velocity2::rotate(double radians) const
{
    const auto c = cos(radians);
    const auto s = sin(radians);
    const auto vx = std::get<0>(this->velocity_vec);
    const auto vy = std::get<1>(this->velocity_vec);

    const auto vx_r = c*vx - s*vy;
    const auto vy_r = s*vx + c*vy;
    
    return velocity2({vx_r, vy_r});
}
