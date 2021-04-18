#include <dbg.h>
#include <iostream>
#include <cinttypes>
#include <cstdint>


typedef struct
{
    uint32_t state;
} GBL_cell_t;


int
main(int argc, char *argv[])
{
    LGCA_cell_t my_cell {0, 1};
    std::cout << sizeof(LGCA_cell_t) << '\n';
    /*
    while (true)
    {
        std::cout << my_cell.m1 << ' ' << my_cell.m2 << '\n';
        my_cell.m1++;
        my_cell.m2++;
    }
    */

    return 0;
error:
    return -1;
}
