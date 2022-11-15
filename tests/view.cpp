#include "common.hpp"

#include <atomic>
#include <deque>

// clang-format off
namespace tag
{
    struct Value {};
} // namespace tag

using RecordDim = llama::Record<
    llama::Field<tag::Value, int>
>;
// clang-format on

TEST_CASE("view.default-ctor")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    [[maybe_unused]] const llama::View<llama::mapping::AlignedAoS<ArrayExtents, RecordDim>, std::byte*> view1{};
    [[maybe_unused]] const llama::View<llama::mapping::PackedAoS<ArrayExtents, RecordDim>, std::byte*> view2{};
    [[maybe_unused]] const llama::View<llama::mapping::AlignedSingleBlobSoA<ArrayExtents, RecordDim>, std::byte*>
        view3{};
    [[maybe_unused]] const llama::View<llama::mapping::PackedSingleBlobSoA<ArrayExtents, RecordDim>, std::byte*>
        view4{};
    [[maybe_unused]] const llama::View<llama::mapping::MultiBlobSoA<ArrayExtents, RecordDim>, std::byte*> view5{};
    [[maybe_unused]] const llama::View<llama::mapping::One<ArrayExtents, RecordDim>, std::byte*> view6{};
    [[maybe_unused]] const llama::
        View<llama::mapping::tree::Mapping<ArrayExtents, RecordDim, llama::Tuple<>>, std::byte*>
            view7{};
}

TEST_CASE("view.move")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    constexpr ArrayExtents viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayExtents, RecordDim>;
    auto view1 = llama::allocView(Mapping(viewSize));

    decltype(view1) view2;
    view1({3, 3}) = 1;
    view2 = std::move(view1);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view.swap")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    constexpr ArrayExtents viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayExtents, RecordDim>;
    auto view1 = llama::allocView(Mapping(viewSize));
    auto view2 = llama::allocView(Mapping(viewSize));

    view1({3, 3}) = 1;
    view2({3, 3}) = 2;

    std::swap(view1, view2);

    CHECK(view1({3, 3}) == 2);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view.non-memory-owning")
{
    auto test = [](auto typeHolder)
    {
        using byte = typename decltype(typeHolder)::type;

        using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 1>;
        const ArrayExtents extents{256};

        using Mapping = llama::mapping::SoA<ArrayExtents, RecordDim>;
        const Mapping mapping{extents};

        const auto blobSize = mapping.blobSize(0);
        const auto storage = std::make_unique<byte[]>(blobSize);
        auto view = llama::View{mapping, llama::Array{storage.get()}};

        for(auto i = 0u; i < 256u; i++)
        {
            auto* v = reinterpret_cast<byte*>(&view(i)(tag::Value{}));
            CHECK(storage.get() <= v);
            CHECK(v <= storage.get() + blobSize);
        }
    };
    test(boost::mp11::mp_identity<std::byte>{});
    test(boost::mp11::mp_identity<const std::byte>{});
}

TEST_CASE("view.subscript")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 2>;
    const ArrayExtents extents{16, 16};

    using Mapping = llama::mapping::SoA<ArrayExtents, Particle>;
    const Mapping mapping{extents};
    auto view = llama::allocView(mapping);
    auto l = [](auto& view)
    {
        const llama::ArrayIndex pos{0, 0};
        CHECK((view(pos) == view[pos]));
        CHECK((view(pos) == view[{0, 0}]));
        CHECK((view(pos) == view({0, 0})));

        const auto& x = view(pos)(llama::RecordCoord<0, 0>{});
        CHECK(&x == &view(pos)(llama::RecordCoord<0>{})(llama::RecordCoord<0>{}));
        CHECK(&x == &view(pos)(llama::RecordCoord<0>{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{})(llama::RecordCoord<0>{}));
        CHECK(&x == &view(pos)(tag::Pos{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{}, tag::X{}));

        // also test arrays
        using namespace llama::literals;
        const bool& o0 = view(pos)(tag::Flags{})(llama::RecordCoord<0>{});
        CHECK(&o0 == &view(pos)(tag::Flags{})(0_RC));
    };
    l(view);
    l(std::as_const(view));
}

TEST_CASE("view.assign-one-record")
{
    using namespace llama::literals;

    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    const ArrayExtents extents{16, 16};

    using Mapping = llama::mapping::SoA<ArrayExtents, Particle>;
    const Mapping mapping{extents};
    auto view = llama::allocView(mapping);

    llama::One<Particle> record;
    record(tag::Pos{}, tag::X{}) = 14.0f;
    record(tag::Pos{}, tag::Y{}) = 15.0f;
    record(tag::Pos{}, tag::Z{}) = 16.0f;
    record(tag::Vel{}) = 0;
    record(tag::Mass{}) = 500.0f;
    record(tag::Flags{})(0_RC) = true;
    record(tag::Flags{})(1_RC) = false;
    record(tag::Flags{})(2_RC) = true;
    record(tag::Flags{})(3_RC) = false;

    view({3, 4}) = record;

    CHECK(record(tag::Pos{}, tag::X{}) == 14.0f);
    CHECK(record(tag::Pos{}, tag::Y{}) == 15.0f);
    CHECK(record(tag::Pos{}, tag::Z{}) == 16.0f);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Mass{}) == 500.0f);
    CHECK(record(tag::Flags{})(0_RC) == true);
    CHECK(record(tag::Flags{})(1_RC) == false);
    CHECK(record(tag::Flags{})(2_RC) == true);
    CHECK(record(tag::Flags{})(3_RC) == false);
}

TEST_CASE("view.addresses")
{
    using namespace llama::literals;

    using ArrayExtents = llama::ArrayExtentsDynamic<int, 2>;
    const ArrayExtents extents{16, 16};

    using Mapping = llama::mapping::PackedSingleBlobSoA<ArrayExtents, Particle>;
    const Mapping mapping{extents};
    auto view = llama::allocView(mapping);

    const llama::ArrayIndex pos{0, 0};
    auto& x = view(pos)(tag::Pos{}, tag::X{});
    auto& y = view(pos)(tag::Pos{}, tag::Y{});
    auto& z = view(pos)(tag::Pos{}, tag::Z{});
    auto& w = view(pos)(tag::Mass{});
    auto& mx = view(pos)(tag::Vel{}, tag::X{});
    auto& my = view(pos)(tag::Vel{}, tag::Y{});
    auto& mz = view(pos)(tag::Vel{}, tag::Z{});
    auto& o0 = view(pos)(tag::Flags{})(0_RC);
    auto& o1 = view(pos)(tag::Flags{})(1_RC);
    auto& o2 = view(pos)(tag::Flags{})(2_RC);
    auto& o3 = view(pos)(tag::Flags{})(3_RC);

    CHECK(reinterpret_cast<std::byte*>(&y) - reinterpret_cast<std::byte*>(&x) == 2048);
    CHECK(reinterpret_cast<std::byte*>(&z) - reinterpret_cast<std::byte*>(&x) == 4096);
    CHECK(reinterpret_cast<std::byte*>(&mx) - reinterpret_cast<std::byte*>(&x) == 7168);
    CHECK(reinterpret_cast<std::byte*>(&my) - reinterpret_cast<std::byte*>(&x) == 9216);
    CHECK(reinterpret_cast<std::byte*>(&mz) - reinterpret_cast<std::byte*>(&x) == 11264);
    CHECK(reinterpret_cast<std::byte*>(&w) - reinterpret_cast<std::byte*>(&x) == 6144);
    CHECK(reinterpret_cast<std::byte*>(&o0) - reinterpret_cast<std::byte*>(&x) == 13312);
    CHECK(reinterpret_cast<std::byte*>(&o1) - reinterpret_cast<std::byte*>(&x) == 13568);
    CHECK(reinterpret_cast<std::byte*>(&o2) - reinterpret_cast<std::byte*>(&x) == 13824);
    CHECK(reinterpret_cast<std::byte*>(&o3) - reinterpret_cast<std::byte*>(&x) == 14080);
}

template<typename RecordRef>
struct SetZeroFunctor
{
    template<typename RecordCoord>
    void operator()(RecordCoord rc)
    {
        vd(rc) = 0;
    }
    RecordRef vd;
};

TEST_CASE("view.iteration-and-access")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 2>;
    const ArrayExtents extents{16, 16};

    using Mapping = llama::mapping::SoA<ArrayExtents, Particle>;
    const Mapping mapping{extents};
    auto view = llama::allocView(mapping);

    for(int x = 0; x < extents[0]; ++x)
        for(int y = 0; y < extents[1]; ++y)
        {
            SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
            llama::forEachLeafCoord<Particle>(szf, llama::RecordCoord<0, 0>{});
            llama::forEachLeafCoord<Particle>(szf, tag::Vel{});
            view({x, y}) = static_cast<double>(x + y) / static_cast<double>(extents[0] + extents[1]);
        }

    double sum = 0.0;
    for(int x = 0; x < extents[0]; ++x)
        for(int y = 0; y < extents[1]; ++y)
            sum += view(x, y)(llama::RecordCoord<2, 0>{});
    CHECK(sum == 120.0);
}

TEST_CASE("view.record-access")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    const ArrayExtents extents{16, 16};

    using Mapping = llama::mapping::SoA<ArrayExtents, Particle>;
    const Mapping mapping{extents};
    auto view = llama::allocView(mapping);

    for(size_t x = 0; x < extents[0]; ++x)
        for(size_t y = 0; y < extents[1]; ++y)
        {
            auto record = view(x, y);
            record(tag::Pos(), tag::X()) += record(llama::RecordCoord<2, 0>{});
            record(tag::Pos(), tag::Y()) += record(llama::RecordCoord<2, 1>{});
            record(tag::Pos(), tag::Z()) += record(llama::RecordCoord<1>());
            record(tag::Pos()) += record(tag::Vel());
        }

    double sum = 0.0;
    for(size_t x = 0; x < extents[0]; ++x)
        for(size_t y = 0; y < extents[1]; ++y)
            sum += view(x, y)(llama::RecordCoord<2, 0>{});

    CHECK(sum == 0.0);
}

TEST_CASE("view.indexing")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}});
    view(0u, 0u)(tag::Mass{}) = 42.0f;

    using integrals = boost::mp11::mp_list<
        char,
        unsigned char,
        signed char,
        short, // NOLINT(google-runtime-int)
        unsigned short, // NOLINT(google-runtime-int)
        int,
        unsigned int,
        long, // NOLINT(google-runtime-int)
        unsigned long>; // NOLINT(google-runtime-int)

    boost::mp11::mp_for_each<integrals>(
        [&](auto i)
        {
            boost::mp11::mp_for_each<integrals>(
                [&](auto j)
                {
                    const float& w = view(i, j)(tag::Mass{});
                    CHECK(w == 42.0f);
                });
        });

    llama::VirtualView virtualView{view, {0, 0}};
    boost::mp11::mp_for_each<integrals>(
        [&](auto i)
        {
            boost::mp11::mp_for_each<integrals>(
                [&](auto j)
                {
                    const float& w = virtualView(i, j)(tag::Mass{});
                    CHECK(w == 42.0f);
                });
        });
}

TEMPLATE_TEST_CASE("view.transformBlobs", "", llama::bloballoc::Vector)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}});
    iotaFillView(view);

    auto copy = llama::transformBlobs(
        view,
        [](auto& vector) { return std::deque<std::byte>(vector.begin(), vector.end()); });
    STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(copy.storageBlobs[0])>, std::deque<std::byte>>);
    iotaCheckView(copy);
}

TEMPLATE_TEST_CASE("view.shallowCopy", "", llama::bloballoc::Vector, llama::bloballoc::SharedPtr)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}}, TestType{});
    auto checkCopy = [&](const auto& copy)
    {
        STATIC_REQUIRE(std::is_same_v<
                       typename std::decay_t<decltype(view)>::Mapping,
                       typename std::decay_t<decltype(copy)>::Mapping>);
        // check that blob start address is the same
        for(std::size_t i = 0; i < view.storageBlobs.size(); i++)
            CHECK(&view.storageBlobs[i][0] == &copy.storageBlobs[i][0]);
    };

    const auto copy = llama::shallowCopy(view);
    checkCopy(copy);
    const auto copyOfCopy = llama::shallowCopy(copy);
    checkCopy(copyOfCopy);
    const auto constCopy = llama::shallowCopy(std::as_const(view));
    checkCopy(constCopy);
    const auto constCopyOfCopy = llama::shallowCopy(std::as_const(copyOfCopy));
    checkCopy(constCopyOfCopy);
    const auto constCopyOfConstCopy = llama::shallowCopy(std::as_const(constCopy));
    checkCopy(constCopyOfConstCopy);
}

TEST_CASE("view.allocViewStack")
{
    auto v0 = llama::allocViewStack<0, Vec3I>();
    v0(llama::ArrayIndex<int, 0>{})(tag::X{}) = 42;

    auto v1 = llama::allocViewStack<1, Vec3I>();
    v1(llama::ArrayIndex{0})(tag::X{}) = 42;

    auto v4 = llama::allocViewStack<4, Vec3I>();
    v4(llama::ArrayIndex{0, 0, 0, 0})(tag::X{}) = 42;
}

TEST_CASE("view.allocView.Default")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Particle{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::Default{});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("view.allocView.ReadOnlyByValue")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Vec3I{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::ReadOnlyByValue{});
    STATIC_REQUIRE(std::is_same_v<decltype(view(1, 2)(tag::X{})), int>);
}

TEST_CASE("view.allocView.Const")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Vec3I{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::Const{});
    STATIC_REQUIRE(std::is_same_v<decltype(view(1, 2)(tag::X{})), const int&>);
}

#ifdef __cpp_lib_atomic_ref
TEST_CASE("view.allocView.Atomic")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{3, 4}, Vec3I{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Vector{}, llama::accessor::Atomic{});
    STATIC_REQUIRE(std::is_same_v<decltype(view(1, 2)(tag::X{})), std::atomic_ref<int>>);
    iotaFillView(view);
    iotaCheckView(view);
}
#endif

TEST_CASE("view.withAccessor.Default.Vector")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Vec3I{}}, llama::bloballoc::Vector{});
    auto* addr = &view(1, 2)(tag::X{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(view); // copies
    auto* addr2 = &view2(1, 2)(tag::X{});
    CHECK(addr != addr2);
}

TEST_CASE("view.withAccessor.Default.Vector.move")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Vec3I{}}, llama::bloballoc::Vector{});
    auto* addr = &view(1, 2)(tag::X{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(std::move(view));
    auto* addr2 = &view2(1, 2)(tag::X{});
    CHECK(addr == addr2);
}

TEST_CASE("view.withAccessor.Default.SharedPtr")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Vec3I{}}, llama::bloballoc::SharedPtr{});
    auto* addr = &view(1, 2)(tag::X{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(view); // copies shared pointers, but not memory chunks
    auto* addr2 = &view2(1, 2)(tag::X{});
    CHECK(addr == addr2);
}

TEMPLATE_TEST_CASE("view.withAccessor.shallowCopy.Default", "", llama::bloballoc::Vector, llama::bloballoc::SharedPtr)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}}, TestType{});
    auto view2 = llama::withAccessor<llama::accessor::Default>(llama::shallowCopy(view));
    iotaFillView(view2);
    iotaCheckView(view);
}

#ifdef __cpp_lib_atomic_ref
TEMPLATE_TEST_CASE("view.withAccessor.shallowCopy.Atomic", "", llama::bloballoc::Vector, llama::bloballoc::SharedPtr)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}}, TestType{});
    auto view2 = llama::withAccessor<llama::accessor::Atomic>(llama::shallowCopy(view));
    iotaFillView(view2);
    iotaCheckView(view);
}
#endif

TEMPLATE_TEST_CASE("view.withAccessor.shallowCopy.Restrict", "", llama::bloballoc::Vector, llama::bloballoc::SharedPtr)
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16, 16}, Particle{}}, TestType{});
    auto view2 = llama::withAccessor<llama::accessor::Restrict>(llama::shallowCopy(view));
    iotaFillView(view2);
    iotaCheckView(view);
}

namespace
{
    struct OffsetFloatAccessor
    {
        float offset;

        template<typename T>
        auto operator()(T& ref) -> decltype(auto)
        {
            if constexpr(std::is_same_v<T, float>)
                return ref + offset;
            else
                return ref;
        }
    };
} // namespace

TEST_CASE("view.withAccessor.OffsetFloatAccessor")
{
    auto view
        = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{4}, Particle{}}, llama::bloballoc::SharedPtr{});
    view(0)(tag::Pos{})(tag::X{}) = 2.0;
    view(0)(tag::Mass{}) = 2.0f;

    auto view2 = llama::withAccessor(view, OffsetFloatAccessor{3});

    CHECK(view2(0)(tag::Pos{})(tag::X{}) == 2.0);
    CHECK(view2(0)(tag::Mass{}) == 5.0f);

    view2.accessor().offset = 10;

    CHECK(view2(0)(tag::Pos{})(tag::X{}) == 2.0);
    CHECK(view2(0)(tag::Mass{}) == 12.0f);
}
