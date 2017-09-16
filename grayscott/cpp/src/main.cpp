#include <chrono>
#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xview.hpp"

xt::xarray<double> gray_scott(std::size_t counts, double du, double dv, double f, double k)
{
    int n = 300;
    xt::xarray<double> U = xt::zeros<double>({n+2, n+2});
    xt::xarray<double> V = xt::zeros<double>({n+2, n+2});

    auto u = xt::view(U, xt::range(1, n+1), xt::range(1, n+1));
    auto v = xt::view(V, xt::range(1, n+1), xt::range(1, n+1));

    int r = 20;
    u = 1.0;
    std::size_t nd2 = n / 2;
    xt::view(U, xt::range(nd2-r, nd2+r), xt::range(nd2-r, nd2+r)) = 0.5;
    xt::view(V, xt::range(nd2-r, nd2+r), xt::range(nd2-r, nd2+r)) = 0.25;
    xt::noalias(u) += 0.15 * xt::ones<double>({n, n});
    xt::noalias(v) += 0.15 * xt::ones<double>({n, n});

    xt::xarray<double> uvv(u.shape());
    for(std::size_t i = 0; i < counts; ++i)
    {
        auto lu =     xt::view(U, xt::range(0, n),   xt::range(1, n+1)) +
                      xt::view(U, xt::range(1, n+1), xt::range(0, n))   -
                  4 * xt::view(U, xt::range(1, n+1), xt::range(1, n+1)) +
                      xt::view(U, xt::range(1, n+1), xt::range(2, n+2)) +
                      xt::view(U, xt::range(2, n+2), xt::range(1, n+1));

        auto lv =     xt::view(V, xt::range(0, n),   xt::range(1, n+1)) +
                      xt::view(V, xt::range(1, n+1), xt::range(0, n))   -
                  4 * xt::view(V, xt::range(1, n+1), xt::range(1, n+1)) +
                      xt::view(V, xt::range(1, n+1), xt::range(2, n+2)) +
                      xt::view(V, xt::range(2, n+2), xt::range(1, n+1));

        xt::noalias(uvv) = u * v * v;
        u += du * lu - uvv + f * (1 - u);
        v += dv * lv + uvv - (f + k) * v;
    }
    return V;
}

void test()
{
    gray_scott(40, 0.16, 0.08, 0.04, 0.06);
}

int main(int argc, char* argv[])
{
    using duration_type = std::chrono::duration<double, std::milli>;
    std::size_t numbers = 100;
    duration_type res = duration_type::max();
    for(std::size_t i = 0; i < numbers; ++i)
    {
        auto start = std::chrono::steady_clock::now();
        test();
        auto end = std::chrono::steady_clock::now();
        auto tmp = end - start;
        res = tmp < res ? tmp : res;
    }

    std::cout << res.count() * numbers / 1000 << std::endl;
    return 0;
}

