#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"

#include <fmt/core.h>
#include <fstream>
#include <iomanip>
#include <llama/llama.hpp>
#include <omp.h>
#include <random>
#include <vector>

constexpr auto PROBLEM_SIZE = 1024 * 1024 * 16;
constexpr auto STEPS = 5;
constexpr auto alpha = 3.14;

void daxpy(std::ofstream& plotFile)
{
    Stopwatch watch;
    auto x = std::vector<double>(PROBLEM_SIZE);
    auto y = std::vector<double>(PROBLEM_SIZE);
    auto z = std::vector<double>(PROBLEM_SIZE);
    watch.printAndReset("alloc");

    std::default_random_engine engine;
    std::normal_distribution dist(0.0, 1.0);
    for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        x[i] = dist(engine);
        y[i] = dist(engine);
    }
    watch.printAndReset("init");

    double sum = 0;
    for(std::size_t s = 0; s < STEPS; ++s)
    {
#pragma omp parallel for
        for(std::ptrdiff_t i = 0; i < PROBLEM_SIZE; i++)
            z[i] = alpha * x[i] + y[i];
        sum = watch.printAndReset("daxpy");
    }
    plotFile << "\"std::vector\"\t" << sum / STEPS << '\n';
}

template<typename Mapping>
void daxpy_llama(std::string mappingName, Mapping mapping, std::ofstream& plotFile)
{
    auto title = "LLAMA " + std::move(mappingName);

    Stopwatch watch;
    auto x = llama::allocViewUninitialized(mapping);
    auto y = llama::allocViewUninitialized(mapping);
    auto z = llama::allocViewUninitialized(mapping);
    watch.printAndReset("alloc");

    std::default_random_engine engine;
    std::normal_distribution dist(0.0, 1.0);
    for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        x[i] = dist(engine);
        y[i] = dist(engine);
    }
    watch.printAndReset("init");

    double sum = 0;
    for(std::size_t s = 0; s < STEPS; ++s)
    {
#pragma omp parallel for
        for(std::ptrdiff_t i = 0; i < PROBLEM_SIZE; i++)
            z[i] = alpha * x[i] + y[i];
        sum = watch.printAndReset("daxpy");
    }
    plotFile << std::quoted(title) << "\t" << sum / STEPS << '\n';
}

auto main() -> int
try
{
    const auto numThreads = static_cast<std::size_t>(omp_get_max_threads());
    const char* affinity = std::getenv("GOMP_CPU_AFFINITY"); // NOLINT(concurrency-mt-unsafe)
    affinity = affinity == nullptr ? "NONE - PLEASE PIN YOUR THREADS!" : affinity;

    std::ofstream plotFile{"daxpy.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# threads: {} affinity: {}
set title "daxpy CPU {}Mi doubles on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key off
set yrange [0:*]
set ylabel "runtime [s]"
$data << EOD
)",
        numThreads,
        affinity,
        PROBLEM_SIZE / 1024 / 1024,
        common::hostname());

    daxpy(plotFile);

    const auto extents = llama::ArrayExtents{PROBLEM_SIZE};
    daxpy_llama("AoS", llama::mapping::AoS{extents, double{}}, plotFile);
    daxpy_llama("SoA", llama::mapping::SoA{extents, double{}}, plotFile);
    daxpy_llama(
        "Bytesplit",
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<1>, double, llama::mapping::PreconfiguredAoS<>::type>{
            extents},
        plotFile);
    daxpy_llama(
        "ChangeType D->F",
        llama::mapping::ChangeType<
            llama::ArrayExtentsDynamic<1>,
            double,
            llama::mapping::PreconfiguredAoS<>::type,
            boost::mp11::mp_list<boost::mp11::mp_list<double, float>>>{extents},
        plotFile);
    daxpy_llama("Bitpack 52^{11}", llama::mapping::BitPackedFloatSoA{extents, 11, 52, double{}}, plotFile);
    daxpy_llama("Bitpack 23^{8}", llama::mapping::BitPackedFloatSoA{extents, 8, 23, double{}}, plotFile);
    daxpy_llama("Bitpack 10^{5}", llama::mapping::BitPackedFloatSoA{extents, 5, 10, double{}}, plotFile);

    plotFile << R"(EOD
plot $data using 2:xtic(1)
)";
    std::cout << "Plot with: ./daxpy.sh\n";

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
