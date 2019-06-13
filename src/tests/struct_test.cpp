
#include <opencv2/opencv.hpp>
#include <iostream>

typedef struct
{
    int i = 0;
} fuu_t;


void foo(int & i)
{
    i = -1;
}

void doo(int * i)
{
    *i = -1;
}

void fuu(fuu_t & fuu_var)
{
    fuu_var.i = 1;
}

void fuu2(fuu_t & fuu_var)
{
    fuu_t fuu_var2;
    fuu_var2.i = 20;
    fuu_var = fuu_var2;
}

int main(int argc, char * argv[])
{   
    int a = 0;
    int b = 1;
    int c = 2;

    foo(a);
    std::cout << a << std::endl;
    doo(&b);
    std::cout << b << std::endl;

    fuu_t fuu_var;
    fuu(fuu_var);
    std::cout << fuu_var.i << std::endl;

    fuu_t fuu_var_2;
    fuu_var_2.i = 10;
    fuu2(fuu_var_2);
    std::cout << fuu_var_2.i << std::endl;

    return 0;
}