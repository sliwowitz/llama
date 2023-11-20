// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

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

constexpr auto problemSize = 64 * 1024; ///< total number of particles
constexpr auto steps = 5; ///< number of steps to calculate
constexpr auto allowRsqrt = true; // rsqrt can be way faster, but less accurate
constexpr auto runUpate = true; // run update step. Useful to disable for benchmarking the move step.

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        error Cannot enable CUDA together with other backends
#    endif
constexpr auto elementsPerThread = xsimd::batch<float>::size;
constexpr auto threadsPerBlock = 1;
constexpr auto sharedElementsPerBlock = 1;
constexpr auto aosoaLanes = xsimd::batch<float>::size; // vectors
#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
constexpr auto threadsPerBlock = 256;
constexpr auto sharedElementsPerBlock = 512;
constexpr auto elementsPerThread = 1;
constexpr auto aosoaLanes = 32; // coalesced memory access
#else
#    error "Unsupported backend"
#endif

// makes our life easier for now
static_assert(problemSize % sharedElementsPerBlock == 0);
static_assert(sharedElementsPerBlock % threadsPerBlock == 0);

constexpr auto timestep = FP{0.0001};
constexpr auto eps2 = FP{0.01};

constexpr auto rngSeed = 42;
constexpr auto referenceParticleIndex = 1338;

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
    struct Padding{};
} // namespace tag

using Vec3 = llama::Record<
    llama::Field<tag::X, FP>,
    llama::Field<tag::Y, FP>,
    llama::Field<tag::Z, FP>>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Vel, Vec3>,
    llama::Field<tag::Mass, FP>>;

using ParticleJ = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

// using SharedMemoryParticle = Particle;
using SharedMemoryParticle = ParticleJ;

enum Mapping
{
    AoS,
    SoA_SB,
    SoA_MB,
    AoSoA,
    SplitGpuGems
};

template<typename Acc, typename ParticleRefI, typename ParticleRefJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(const Acc& acc, ParticleRefI& pis, ParticleRefJ pj)
{
    auto dist = pis(tag::Pos{}) - pj(tag::Pos{});
    dist *= dist;
    const auto distSqr = +eps2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const auto distSixth = distSqr * distSqr * distSqr;
    const auto invDistCube
        = allowRsqrt ? alpaka::math::rsqrt(acc, distSixth) : (1.0f / alpaka::math::sqrt(acc, distSixth));
    const auto sts = (pj(tag::Mass{}) * timestep) * invDistCube;
    pis(tag::Vel{}) += dist * sts;
}

template<int Elems, Mapping MappingSM>
struct UpdateKernel
{
    template<typename Acc, typename View>
    ALPAKA_FN_HOST_ACC void operator()(const Acc& acc, View particles) const
    {
        auto sharedView = [&]
        {
            // if there is only 1 shared element per block, use just a variable (in registers) instead of shared memory
            if constexpr(sharedElementsPerBlock == 1)
                return llama::allocViewStackUninitialized<View::ArrayExtents::rank, typename View::RecordDim>();
            else
            {
                constexpr auto sharedMapping = []
                {
                    using ArrayExtents = llama::ArrayExtents<int, sharedElementsPerBlock>;
                    if constexpr(MappingSM == AoS)
                        return llama::mapping::AoS<ArrayExtents, SharedMemoryParticle>{};
                    if constexpr(MappingSM == SoA_SB)
                        return llama::mapping::
                            SoA<ArrayExtents, SharedMemoryParticle, llama::mapping::Blobs::Single>{};
                    if constexpr(MappingSM == SoA_MB)
                        return llama::mapping::
                            SoA<ArrayExtents, SharedMemoryParticle, llama::mapping::Blobs::OnePerField>{};
                    if constexpr(MappingSM == AoSoA)
                        return llama::mapping::AoSoA<ArrayExtents, SharedMemoryParticle, aosoaLanes>{};
                }();

                llama::Array<std::byte*, decltype(sharedMapping)::blobCount> sharedMems{};
                boost::mp11::mp_for_each<boost::mp11::mp_iota_c<decltype(sharedMapping)::blobCount>>(
                    [&](auto i)
                    {
                        auto& sharedMem = alpaka::declareSharedVar<std::byte[sharedMapping.blobSize(i)], i>(acc);
                        sharedMems[i] = &sharedMem[0];
                    });
                return llama::View{sharedMapping, sharedMems};
            }
        }();

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        auto pis = llama::SimdN<typename View::RecordDim, Elems, MakeSizedBatch>{};
        llama::loadSimd(particles(ti * Elems), pis);

        for(int blockOffset = 0; blockOffset < problemSize; blockOffset += sharedElementsPerBlock)
        {
            for(int j = 0; j < sharedElementsPerBlock; j += threadsPerBlock)
                sharedView(j) = particles(blockOffset + tbi + j);
            alpaka::syncBlockThreads(acc);
            for(int j = 0; j < sharedElementsPerBlock; ++j)
                pPInteraction(acc, pis, sharedView(j));
            alpaka::syncBlockThreads(acc);
        }
        llama::storeSimd(pis(tag::Vel{}), particles(ti * Elems)(tag::Vel{}));
    }
};

template<int Elems>
struct MoveKernel
{
    template<typename Acc, typename View>
    ALPAKA_FN_HOST_ACC void operator()(const Acc& acc, View particles) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto i = ti * Elems;
        llama::SimdN<Vec3, Elems, MakeSizedBatch> pos;
        llama::SimdN<Vec3, Elems, MakeSizedBatch> vel;
        llama::loadSimd(particles(i)(tag::Pos{}), pos);
        llama::loadSimd(particles(i)(tag::Vel{}), vel);
        llama::storeSimd(pos + vel * +timestep, particles(i)(tag::Pos{}));
    }
};

template<typename Acc>
constexpr auto hasSharedMem = alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt, alpaka::TagGpuHipRt>;

template<typename Acc, Mapping MappingGM, Mapping MappingSM>
void run(std::ostream& plotFile)
{
    using Dim = alpaka::Dim<Acc>;
    using Size = alpaka::Idx<Acc>;

    auto mappingName = [](int m) -> std::string
    {
        if(m == 0)
            return "AoS";
        if(m == 1)
            return "SoA SB";
        if(m == 2)
            return "SoA MB";
        if(m == 3)
            return "AoSoA" + std::to_string(aosoaLanes);
        if(m == 4)
            return "SplitGpuGems";
        std::abort();
    };
    const auto title = "GM " + mappingName(MappingGM) + (hasSharedMem<Acc> ? " SM " + mappingName(MappingSM) : "");
    std::cout << '\n' << title << '\n';

    const auto platformAcc = alpaka::Platform<Acc>{};
    const auto platformHost = alpaka::PlatformCpu{};
    const auto devAcc = alpaka::getDevByIdx(platformAcc, 0);
    const auto devHost = alpaka::getDevByIdx(platformHost, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{devAcc};

    auto mapping = []
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
        const auto extents = ArrayExtents{problemSize};
        if constexpr(MappingGM == AoS)
            return llama::mapping::AoS<ArrayExtents, Particle>{extents};
        if constexpr(MappingGM == SoA_SB)
            return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::Single>{extents};
        if constexpr(MappingGM == SoA_MB)
            return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::OnePerField>{extents};
        if constexpr(MappingGM == AoSoA)
            return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{extents};
        using boost::mp11::mp_list;
        if constexpr(MappingGM == SplitGpuGems)
        {
            using Vec4 = llama::Record<
                llama::Field<tag::X, FP>,
                llama::Field<tag::Y, FP>,
                llama::Field<tag::Z, FP>,
                llama::Field<tag::Padding, FP>>; // dummy
            using ParticlePadded = llama::
                Record<llama::Field<tag::Pos, Vec3>, llama::Field<tag::Vel, Vec4>, llama::Field<tag::Mass, FP>>;
            return llama::mapping::Split<
                ArrayExtents,
                ParticlePadded,
                mp_list<
                    mp_list<tag::Pos, tag::X>,
                    mp_list<tag::Pos, tag::Y>,
                    mp_list<tag::Pos, tag::Z>,
                    mp_list<tag::Mass>>,
                llama::mapping::BindAoS<>::fn,
                llama::mapping::BindAoS<>::fn,
                true>{extents};
        }
    }();
    std::ofstream{"nbody_alpaka_mapping_" + mappingName(MappingGM) + ".svg"}
        << llama::toSvg(decltype(mapping){llama::ArrayExtentsDynamic<int, 1>{32}});

    Stopwatch watch;

    auto hostView
        = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devHost)>{devHost});
    auto accView = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devAcc)>{devAcc});
    watch.printAndReset("alloc views");

    std::mt19937_64 engine{rngSeed};
    std::normal_distribution distribution(FP{0}, FP{1});
    for(int i = 0; i < problemSize; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(engine);
        p(tag::Pos(), tag::Y()) = distribution(engine);
        p(tag::Pos(), tag::Z()) = distribution(engine);
        p(tag::Vel(), tag::X()) = distribution(engine) / FP{10};
        p(tag::Vel(), tag::Y()) = distribution(engine) / FP{10};
        p(tag::Vel(), tag::Z()) = distribution(engine) / FP{10};
        p(tag::Mass()) = distribution(engine) / FP{100};
        hostView(i) = p;
    }
    watch.printAndReset("init");

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, accView.blobs()[i], hostView.blobs()[i]);
    watch.printAndReset("copy H->D");

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{static_cast<Size>(problemSize / (threadsPerBlock * elementsPerThread))},
        alpaka::Vec<Dim, Size>{static_cast<Size>(threadsPerBlock)},
        alpaka::Vec<Dim, Size>{static_cast<Size>(elementsPerThread)}};

    double sumUpdate = 0;
    double sumMove = 0;
    for(int s = 0; s < steps; ++s)
    {
        if constexpr(runUpate)
        {
            auto updateKernel = UpdateKernel<elementsPerThread, MappingSM>{};
            alpaka::exec<Acc>(queue, workdiv, updateKernel, llama::shallowCopy(accView));
            sumUpdate += watch.printAndReset("update", '\t');
        }

        auto moveKernel = MoveKernel<elementsPerThread>{};
        alpaka::exec<Acc>(queue, workdiv, moveKernel, llama::shallowCopy(accView));
        sumMove += watch.printAndReset("move");
    }
    plotFile << std::quoted(title) << "\t" << sumUpdate / steps << '\t' << sumMove / steps << '\n';

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, hostView.blobs()[i], accView.blobs()[i]);
    watch.printAndReset("copy D->H");

    const auto [x, y, z] = hostView(referenceParticleIndex)(tag::Pos{});
    fmt::print("reference pos: {{{} {} {}}}\n", x, y, z);
}

auto main() -> int
try
{
    using Dim = alpaka::DimInt<1>;
    using Size = int;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;

    std::cout << problemSize / 1000 << "k particles (" << problemSize * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << threadsPerBlock << " particles (" << threadsPerBlock * llama::sizeOf<Particle> / 1024
              << " kiB) in shared memory\n"
              << "Reducing on " << elementsPerThread << " particles per thread\n"
              << "Using " << threadsPerBlock << " threads per block\n";
    std::cout << std::fixed;

    std::ofstream plotFile{"nbody_alpaka.sh"};
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
        alpaka::getAccName<Acc>(),
        common::hostname());
    plotFile << "\"\"\t\"update\"\t\"move\"\n";

    run<Acc, AoS, AoS>(plotFile);
    if constexpr(hasSharedMem<Acc>)
        run<Acc, AoS, SoA_SB>(plotFile);
    if constexpr(hasSharedMem<Acc>)
        run<Acc, AoS, AoSoA>(plotFile);
    run<Acc, SoA_SB, AoS>(plotFile);
    if constexpr(hasSharedMem<Acc>)
        run<Acc, SoA_SB, SoA_SB>(plotFile);
    if constexpr(hasSharedMem<Acc>)
        run<Acc, SoA_SB, AoSoA>(plotFile);
    run<Acc, AoSoA, AoS>(plotFile);
    if constexpr(hasSharedMem<Acc>)
        run<Acc, AoSoA, SoA_SB>(plotFile);
    if constexpr(hasSharedMem<Acc>)
        run<Acc, AoSoA, AoSoA>(plotFile);
    run<Acc, SplitGpuGems, AoS>(plotFile);

    plotFile << R"(EOD
plot $data using 2:xtic(1) ti col axis x1y1, "" using 3 ti col axis x1y2
)";
    std::cout << "Plot with: ./nbody_alpaka.sh\n";

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}
