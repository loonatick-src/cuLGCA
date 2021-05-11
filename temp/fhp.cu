#include <cassert>
#include <cmath>
#include <iostream>
#include <tuple>
#include <bitset>
#include <vector>


template <size_t dim, typename float_type>
struct velocity
{
    // buffer for storing float values
    std::array<float_type, dim> velocity_vec;

    // constructor from std::array
    velocity(const std::array<float_type, dim>& arr) :
        velocity_vec {arr} {}

    // index subscript operator
    float_type
    operator[](size_t index)
    {
        return this->velocity_vec[index];
    }

    velocity
    operator+(const velocity& v2)
    {
        velocity v_out;
        for (size_t i = 0; i < dim; i++)
        {
            v_out[i] = velocity_vec[i] + v2[i];
        }
        return v_out;
    }

    velocity
    operator-(const velocity& v2)
    {
        velocity v_out;
        for (size_t i = 0; i < dim; i++)
        {
            v_out[i] = velocity_vec[i] - v2[i];
        }
        return v_out;
    }


    inline
    float_type
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
    norm_diff(const velocity& v2)
    {
        const auto v = (*this) - v2; 
        return v.speed();
    }
};


velocity2 = velocity<double, 2>;


#if 0
auto
rotate(velocity2 v, double radians)
{
    auto c = cos(radians);
    auto s = sin(radians);
    auto vx = std::get<0>(v);
    auto vy = std::get<1>(v);

    return make_velocity(c*vx - s*vy, s*vx + c*vy);
}
#endif


template <size_t n>
auto
generate_velocities()
{
    velocity2 v0 = make_velocity(1.0, 0.0);
    double angle = 2.0 * M_PI/n;

    std::array<velocity2, n> velocities;
    velocities[0] = v0; 
    for (int i = 1; i < n; i++)
    {
        velocities[i] = rotate(velocities[i-1], angle);
    }
    return velocities;
}


template <long unsigned int channel_count, typename word>
auto
number_of_particles(std::array<velocity2, channel_count>& velocities,
        word n)
{
    uint32_t count = 0;
    while (n)
    {
        n &= (n-1);
        count++;
    }
    return count;
}


template <long unsigned int channel_count, typename word>
auto
calculate_momentum(std::array<velocity2, channel_count>& velocities,
        word state)
{
    const word bit = 0x1;
    velocity2 v = make_velocity(0.0, 0.0);
    for (auto i = 0; i < channel_count; i++)
    {
        const word temp = state & bit;
        if (temp == 1)
        {
            const auto vt = velocities[i];
            auto vx = std::get<0>(vt);
            auto vy = std::get<1>(vt);
            std::get<0>(v) += vx;
            std::get<1>(v) += vy;
        }
        state >>= 1;
    }
    return v;
}


template<size_t n>
auto
clean_perturbations(std::array<velocity2, n>& velocities, double threshold)
{
    for (auto& v : velocities)
    {
        auto vx = std::get<0>(v);
        auto vy = std::get<1>(v);
        if (fabs(vx) < threshold)
            vx = 0;
        if (fabs(vy) < threshold)
            vy = 0;
        v = make_velocity(vx, vy);
    }
}

int main()
{
    const size_t CC = 6;
    const double threshold = 1.0e-6;
    auto velocities = generate_velocities<CC>();  
    clean_perturbations<CC>(velocities, threshold);
    std::vector< std::vector< u8 > > equivalence_classes;

    const int array_length = 128;

    // Cuda copy-able memory
    uint8_t *h_eq_classes = (uint8_t*) malloc(array_length*sizeof(uint8_t));
    uint8_t *h_state_to_eq = (uint8_t*) malloc(array_length*sizeof(uint8_t));
    uint8_t *h_eq_class_size = (uint8_t*) malloc(array_length*sizeof(uint8_t));
    
    u8 state = 0;

    // Populate equivalence class table
    do 
    {
        auto momentum = calculate_momentum(velocities, state);
        auto particle_count = number_of_particles(velocities, state);
        bool found = false;
        for (auto& eqclass : equivalence_classes)
        {
            auto it = eqclass.begin();
            if (particle_count == number_of_particles(velocities, *it)
                    && norm_diff(calculate_momentum(velocities, *it), momentum) < threshold)
            {
                found = true;
                eqclass.push_back(state);
                break;
            }
        }
        if (!found)
        {
            std::vector<u8> newclass;
            newclass.push_back(state);
            equivalence_classes.push_back(newclass);
        }
        state++;
    } while (state != (1<<6));
    
    // Convert vec < vec <> > to array and Populate state_to_eq_class table
    // for normal collision 
    int index = 0;
    for (auto& eqcl : equivalence_classes)
    {
        auto start_id = index;
        auto size = eqcl.size();
        for (auto& state : eqcl)
        {
            h_eq_classes[index] = state;
            h_state_to_eq[state] = start_id;
            h_eq_class_size[state] = size;
            index++;
        }
    }

    int BIT_MASK_3 = 0b111, BIT_MASK_6 = 0b111000;

    // Lookup table entries for bounce back collisions
    // ASSUMPTION: 7th bit is set to 1 if bounce back happens in this point
    for(int i=64; i<128; i++) 
    {
        uint8_t vel_vec = i - 64;
        uint8_t bounced = (1<<6) | (((vel_vec&BIT_MASK_3)<<3)|((vel_vec&BIT_MASK_6)>>3));
        h_eq_classes[i] = bounced;
        h_state_to_eq[i] = i;
        h_eq_class_size[i] = 1;
    }

    for (auto& eqcl : equivalence_classes)
    {
        for (auto& state : eqcl)
        {
            std::bitset<8> x(state);
            std::cout << x << ' ';
        }
        std::cout << std::endl;
    }

    std::cout << std::endl; 
    for(int i=0; i<array_length; i++) 
    {
        std::bitset<8> x(h_eq_classes[i]);
        std::cout << x << ' ';
        if (i==63)
            std::cout<<"\n";
    }
    std::cout<<"\n\n";
    
    for(int i=0; i<array_length; i++) 
    {
        std::cout << i << ":" << (int)(h_state_to_eq[i]) << "\t";
    }
    std::cout<<"\n";

    return 0;
}
