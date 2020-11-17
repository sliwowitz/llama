#include "../../common/Stopwatch.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

using FP = float;

constexpr auto PROBLEM_SIZE = 32 * 1024; ///< total number of particles
constexpr auto SHARED_ELEMENTS_PER_BLOCK = 256;
constexpr auto STEPS = 5; ///< number of steps to calculate
constexpr FP TIMESTEP = 0.0001f;

constexpr FP ts = 0.0001;

constexpr auto THREADS_PER_BLOCK = 256;
constexpr auto AOSOA_LANES = 32; // coalesced memory access

// makes our life easier for now
static_assert(PROBLEM_SIZE % SHARED_ELEMENTS_PER_BLOCK == 0);
static_assert(SHARED_ELEMENTS_PER_BLOCK % THREADS_PER_BLOCK == 0);

constexpr FP EPS2 = 0.01;

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>>>,
    llama::DE<tag::Vel, llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>>>,
    llama::DE<tag::Mass, FP>>;
// clang-format on

template <typename VirtualParticleI, typename VirtualParticleJ>
__device__ void pPInteraction(VirtualParticleI pi, VirtualParticleJ pj)
{
    auto dist = pi(tag::Pos()) - pj(tag::Pos());
    dist *= dist;
    const FP distSqr = EPS2 + dist(tag::X()) + dist(tag::Y()) + dist(tag::Z());
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP sts = pj(tag::Mass()) * invDistCube * +TIMESTEP;
    pi(tag::Vel()) += dist * sts;
}

template <std::size_t ProblemSize, std::size_t BlockSize, typename View>
__global__ void updateSM(View particles)
{
    // FIXME: removing this lambda makes nvcc 11 segfault
    auto sharedView = [] {
        const auto sharedMapping
            = llama::mapping::AoSoA<llama::ArrayDomain<1>, Particle, 32>(llama::ArrayDomain{BlockSize}, Particle{});
        static_assert(decltype(sharedMapping)::blobCount == 1);
        constexpr auto sharedMemSize = llama::sizeOf<typename View::DatumDomain> * BlockSize;
        __shared__ std::byte sharedMem[sizeof(std::byte[sharedMemSize])];
        return llama::View{sharedMapping, llama::Array<std::byte*, 1>{&sharedMem[0]}};
    }();

    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;
    const auto tbi = blockIdx.x;

    for (std::size_t blockOffset = 0; blockOffset < ProblemSize; blockOffset += BlockSize)
    {
        LLAMA_INDEPENDENT_DATA
        for (auto j = std::size_t{0}; j + ti < BlockSize; j += BlockSize)
            sharedView(j) = particles(blockOffset + j);
        __syncthreads();

        LLAMA_INDEPENDENT_DATA
        for (auto j = std::size_t{0}; j < BlockSize; ++j)
            pPInteraction(particles(ti), sharedView(j));
        __syncthreads();
    }
}

template <std::size_t ProblemSize, typename View>
__global__ void update(View particles)
{
    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;

    LLAMA_INDEPENDENT_DATA
    for (auto j = std::size_t{0}; j < ProblemSize; ++j)
        pPInteraction(particles(ti), particles(j));
}

template <std::size_t ProblemSize, typename View>
__global__ void move(View particles)
{
    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;
    particles(ti)(tag::Pos()) += particles(ti)(tag::Vel()) * +TIMESTEP;
}

void checkError(cudaError_t code)
{
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}

template <std::size_t Mapping>
void run(const char* name, bool useSharedMemory)
try
{
    const auto mappingName = [] {
        if constexpr (Mapping == 0)
            return "AoS";
        if constexpr (Mapping == 1)
            return "SoA";
        if constexpr (Mapping == 2)
            return "AoSoA";
    }();
    std::cout << '\n' << name << ' ' << mappingName << '\n';

    auto mapping = [] {
        const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
        if constexpr (Mapping == 0)
            return llama::mapping::AoS{arrayDomain, Particle{}};
        if constexpr (Mapping == 1)
            return llama::mapping::SoA{arrayDomain, Particle{}};
        if constexpr (Mapping == 2)
            return llama::mapping::AoSoA<std::decay_t<decltype(arrayDomain)>, Particle, AOSOA_LANES>{arrayDomain};
    }();

    Stopwatch chrono;

    const auto bufferSize = mapping.getBlobSize(0);
    std::byte* accBuffer;
    checkError(cudaMalloc(&accBuffer, bufferSize));

    chrono.printAndReset("alloc");

    auto hostView = llama::allocView(mapping);
    auto accView
        = llama::View<std::decay_t<decltype(mapping)>, std::byte*>{mapping, llama::Array<std::byte*, 1>{accBuffer}};

    chrono.printAndReset("views");

    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        auto temp = llama::allocVirtualDatumStack<Particle>();
        temp(tag::Pos(), tag::X()) = distribution(generator);
        temp(tag::Pos(), tag::Y()) = distribution(generator);
        temp(tag::Pos(), tag::Z()) = distribution(generator);
        temp(tag::Vel(), tag::X()) = distribution(generator) / FP(10);
        temp(tag::Vel(), tag::Y()) = distribution(generator) / FP(10);
        temp(tag::Vel(), tag::Z()) = distribution(generator) / FP(10);
        temp(tag::Mass()) = distribution(generator) / FP(100);
        hostView(i) = temp;
    }

    chrono.printAndReset("init");

    static_assert(hostView.storageBlobs.rank == 1);
    checkError(cudaMemcpy(accBuffer, hostView.storageBlobs[0].data(), bufferSize, cudaMemcpyHostToDevice));
    chrono.printAndReset("copy H->D");

    const auto blocks = PROBLEM_SIZE / THREADS_PER_BLOCK;

    for (std::size_t s = 0; s < STEPS; ++s)
    {
        if (useSharedMemory)
            updateSM<PROBLEM_SIZE, SHARED_ELEMENTS_PER_BLOCK><<<blocks, THREADS_PER_BLOCK>>>(accView);
        else
            update<PROBLEM_SIZE><<<blocks, THREADS_PER_BLOCK>>>(accView);
        checkError(cudaDeviceSynchronize());
        chrono.printAndReset("update", '\t');

        move<PROBLEM_SIZE><<<blocks, THREADS_PER_BLOCK>>>(accView);
        checkError(cudaDeviceSynchronize());
        chrono.printAndReset("move");
    }

    checkError(cudaMemcpy(hostView.storageBlobs[0].data(), accBuffer, bufferSize, cudaMemcpyDeviceToHost));
    chrono.printAndReset("copy D->H");

    checkError(cudaFree(accBuffer));
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << std::endl;
}

int main()
{
    std::cout << PROBLEM_SIZE / 1000 << "k particles (" << PROBLEM_SIZE * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << SHARED_ELEMENTS_PER_BLOCK << " particles ("
              << SHARED_ELEMENTS_PER_BLOCK * llama::sizeOf<Particle> / 1024 << " kiB) in shared memory\n"
              << "Using " << THREADS_PER_BLOCK << " per block\n";
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Running on " << prop.name << " " << prop.sharedMemPerBlock / 1024 << "kiB SM\n";

    run<0>("LLAMA", false);
    run<1>("LLAMA", false);
    run<2>("LLAMA", false);
    run<0>("LLAMA SM", true);
    run<1>("LLAMA SM", true);
    run<2>("LLAMA SM", true);

    return 0;
}