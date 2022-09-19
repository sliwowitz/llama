#include "../../common/Stopwatch.hpp"
#include "../../common/hostname.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <string>
#include <utility>

#if __has_include(<xsimd/xsimd.hpp>)
#    include <xsimd/xsimd.hpp>
#    define HAVE_XSIMD
#endif

using FP = float;

constexpr auto problemSize = 16 * 1024; ///< total number of particles
constexpr auto steps = 5; ///< number of steps to calculate
constexpr auto timestep = FP{0.0001};
constexpr auto allowRsqrt = true; // rsqrt can be way faster, but less accurate

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        error Cannot enable CUDA together with other backends
#    endif
constexpr auto desiredElementsPerThread = xsimd::batch<float>::size;
constexpr auto threadsPerBlock = 1;
constexpr auto aosoaLanes = xsimd::batch<float>::size; // vectors
#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
constexpr auto desiredElementsPerThread = 1;
constexpr auto threadsPerBlock = 256;
constexpr auto aosoaLanes = 32; // coalesced memory access
#else
#    error "Unsupported backend"
#endif

// makes our life easier for now
static_assert(problemSize % (desiredElementsPerThread * threadsPerBlock) == 0);

constexpr FP epS2 = 0.01;

#ifdef HAVE_XSIMD
template<typename Batch>
struct llama::SimdTraits<Batch, std::enable_if_t<xsimd::is_batch<Batch>::value>>
{
    using value_type = typename Batch::value_type;

    inline static constexpr std::size_t lanes = Batch::size;

    static LLAMA_FORCE_INLINE auto loadUnaligned(const value_type* mem) -> Batch
    {
        return Batch::load_unaligned(mem);
    }

    static LLAMA_FORCE_INLINE void storeUnaligned(Batch batch, value_type* mem)
    {
        batch.store_unaligned(mem);
    }
};

template<typename T>
using MakeBatch = xsimd::batch<T>;

template<typename T, std::size_t N>
struct MakeSizedBatchImpl
{
    using type = xsimd::make_sized_batch_t<T, N>;
    static_assert(!std::is_void_v<type>);
};

template<typename T, std::size_t N>
using MakeSizedBatch = typename MakeSizedBatchImpl<T, N>::type;
#endif

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
} // namespace tag

using Vec3 = llama::Record<
    llama::Field<tag::X, FP>,
    llama::Field<tag::Y, FP>,
    llama::Field<tag::Z, FP>>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Vel, Vec3>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

enum Mapping
{
    AoS,
    SoA,
    AoSoA
};

namespace stdext
{
    LLAMA_FN_HOST_ACC_INLINE auto rsqrt(FP f) -> FP
    {
        return 1.0f / std::sqrt(f);
    }
} // namespace stdext

template<typename ViewParticleI, typename ParticleRefJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(ViewParticleI& pis, ParticleRefJ pj)
{
    using std::sqrt;
    using stdext::rsqrt;
    using xsimd::rsqrt;
    using xsimd::sqrt;

    auto dist = pis(tag::Pos{}) - pj(tag::Pos{});
    dist *= dist;
    const auto distSqr = +epS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const auto distSixth = distSqr * distSqr * distSqr;
    const auto invDistCube = allowRsqrt ? rsqrt(distSixth) : (1.0f / sqrt(distSixth));
    const auto sts = (pj(tag::Mass{}) * timestep) * invDistCube;
    pis(tag::Vel{}) += dist * sts;
}

template<int ProblemSize, int Elems, int BlockSize, Mapping MappingSM>
struct UpdateKernel
{
    template<typename Acc, typename View>
    ALPAKA_FN_HOST_ACC void operator()(const Acc& acc, View particles) const
    {
        auto sharedView = [&]
        {
            // if there is only 1 thread per block, use just a variable (in registers) instead of shared memory
            if constexpr(BlockSize == 1)
                return llama::allocViewStack<View::ArrayExtents::rank, typename View::RecordDim>();
            else
            {
                constexpr auto sharedMapping = []
                {
                    using ArrayExtents = llama::ArrayExtents<int, BlockSize>;
                    if constexpr(MappingSM == AoS)
                        return llama::mapping::AoS<ArrayExtents, Particle>{};
                    if constexpr(MappingSM == SoA)
                        return llama::mapping::SoA<ArrayExtents, Particle, false>{};
                    if constexpr(MappingSM == AoSoA)
                        return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{};
                }();
                static_assert(decltype(sharedMapping)::blobCount == 1);

                constexpr auto sharedMemSize = llama::sizeOf<typename View::RecordDim> * BlockSize;
                auto& sharedMem = alpaka::declareSharedVar<std::byte[sharedMemSize], __COUNTER__>(acc);
                return llama::View{sharedMapping, llama::Array<std::byte*, 1>{&sharedMem[0]}};
            }
        }();

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        auto pis = llama::SimdN<typename View::RecordDim, Elems, MakeSizedBatch>{};
        llama::loadSimd(pis, particles(ti * Elems));

        LLAMA_INDEPENDENT_DATA
        for(int blockOffset = 0; blockOffset < ProblemSize; blockOffset += BlockSize)
        {
            LLAMA_INDEPENDENT_DATA
            for(int j = tbi; j < BlockSize; j += threadsPerBlock)
                sharedView(j) = particles(blockOffset + j);
            alpaka::syncBlockThreads(acc);

            LLAMA_INDEPENDENT_DATA
            for(int j = 0; j < BlockSize; ++j)
                pPInteraction(pis, sharedView(j));
            alpaka::syncBlockThreads(acc);
        }
        llama::storeSimd(particles(ti * Elems)(tag::Vel{}), pis(tag::Vel{}));
    }
};

template<int ProblemSize, int Elems>
struct MoveKernel
{
    template<typename Acc, typename View>
    ALPAKA_FN_HOST_ACC void operator()(const Acc& acc, View particles) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto i = ti * Elems;
        llama::SimdN<Vec3, Elems, MakeSizedBatch> pos;
        llama::SimdN<Vec3, Elems, MakeSizedBatch> vel;
        llama::loadSimd(pos, particles(i)(tag::Pos{}));
        llama::loadSimd(vel, particles(i)(tag::Vel{}));
        llama::storeSimd(particles(i)(tag::Pos{}), pos + vel * +timestep);
    }
};

template<template<typename, typename> typename AccTemplate, Mapping MappingGM, Mapping MappingSM>
void run(std::ostream& plotFile)
{
    using Dim = alpaka::DimInt<1>;
    using Size = int;
    using Acc = AccTemplate<Dim, Size>;
    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto mappingName = [](int m) -> std::string
    {
        if(m == 0)
            return "AoS";
        if(m == 1)
            return "SoA";
        if(m == 2)
            return "AoSoA" + std::to_string(aosoaLanes);
        std::abort();
    };
    const auto title = "GM " + mappingName(MappingGM) + " SM " + mappingName(MappingSM);
    std::cout << '\n' << title << '\n';

    const DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    auto mapping = []
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
        const auto extents = ArrayExtents{problemSize};
        if constexpr(MappingGM == AoS)
            return llama::mapping::AoS<ArrayExtents, Particle>{extents};
        if constexpr(MappingGM == SoA)
            return llama::mapping::SoA<ArrayExtents, Particle, false>{extents};
        // if constexpr (MappingGM == 2)
        //    return llama::mapping::SoA<ArrayExtents, Particle, true>{extents};
        if constexpr(MappingGM == AoSoA)
            return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{extents};
    }();

    Stopwatch watch;

    auto hostView
        = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devHost)>{devHost});
    auto accView = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devAcc)>{devAcc});
    watch.printAndReset("alloc views");

    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP{0}, FP{1});
    for(int i = 0; i < problemSize; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(generator);
        p(tag::Pos(), tag::Y()) = distribution(generator);
        p(tag::Pos(), tag::Z()) = distribution(generator);
        p(tag::Vel(), tag::X()) = distribution(generator) / FP{10};
        p(tag::Vel(), tag::Y()) = distribution(generator) / FP{10};
        p(tag::Vel(), tag::Z()) = distribution(generator) / FP{10};
        p(tag::Mass()) = distribution(generator) / FP{100};
        hostView(i) = p;
    }
    watch.printAndReset("init");

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, accView.storageBlobs[i], hostView.storageBlobs[i]);
    watch.printAndReset("copy H->D");

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{static_cast<Size>(problemSize / (threadsPerBlock * desiredElementsPerThread))},
        alpaka::Vec<Dim, Size>{static_cast<Size>(threadsPerBlock)},
        alpaka::Vec<Dim, Size>{static_cast<Size>(desiredElementsPerThread)}};

    double sumUpdate = 0;
    double sumMove = 0;
    for(int s = 0; s < steps; ++s)
    {
        auto updateKernel = UpdateKernel<problemSize, desiredElementsPerThread, threadsPerBlock, MappingSM>{};
        alpaka::exec<Acc>(queue, workdiv, updateKernel, llama::shallowCopy(accView));
        sumUpdate += watch.printAndReset("update", '\t');

        auto moveKernel = MoveKernel<problemSize, desiredElementsPerThread>{};
        alpaka::exec<Acc>(queue, workdiv, moveKernel, llama::shallowCopy(accView));
        sumMove += watch.printAndReset("move");
    }
    plotFile << std::quoted(title) << "\t" << sumUpdate / steps << '\t' << sumMove / steps << '\n';

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, hostView.storageBlobs[i], accView.storageBlobs[i]);
    watch.printAndReset("copy D->H");
}

auto main() -> int
try
{
#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 5
// nvcc <= 11.5 chokes on `pis(tag::Pos{}, tag::X{})` inside `pPInteraction()`
#    warning "alpaka nbody example disabled for nvcc <= 11.5, because the compiler segfaults"
    return -1;
#else

    std::cout << problemSize / 1000 << "k particles (" << problemSize * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << threadsPerBlock << " particles (" << threadsPerBlock * llama::sizeOf<Particle> / 1024
              << " kiB) in shared memory\n"
              << "Reducing on " << desiredElementsPerThread << " particles per thread\n"
              << "Using " << threadsPerBlock << " threads per block\n";
    std::cout << std::fixed;

    std::ofstream plotFile{"nbody.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "nbody alpaka {}ki particles on {} on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
set y2range [0:*]
set ylabel "update runtime [s]"
set y2label "move runtime [s]"
set y2tics auto
$data << EOD
)",
        problemSize / 1024,
        alpaka::getAccName<alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, int>>(),
        common::hostname());
    plotFile << "\"\"\t\"update\"\t\"move\"\n";

    run<alpaka::ExampleDefaultAcc, AoS, AoS>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoS, SoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoS, AoSoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, SoA, AoS>(plotFile);
    run<alpaka::ExampleDefaultAcc, SoA, SoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, SoA, AoSoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoSoA, AoS>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoSoA, SoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoSoA, AoSoA>(plotFile);

    plotFile << R"(EOD
plot $data using 2:xtic(1) ti col axis x1y1, "" using 3 ti col axis x1y2
)";
    std::cout << "Plot with: ./nbody.sh\n";

    return 0;
#endif
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
