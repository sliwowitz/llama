// copy into https://cppx.godbolt.org
// flags: -std=c++2a -O3 -mavx2 -mfma -ffast-math

#include <vector>
#include <tuple>
#include <cmath>
#include <random>

namespace meta = std::experimental::meta;

namespace lib {
    namespace internal {
        consteval void genSoRMembers(meta::info sov) {
            for (meta::info member : meta::data_member_range(sov)) {
                auto type = meta::type_of(member);
                auto name = meta::name_of(member);
                -> fragment struct {
                    typename(%{type})& unqualid(%{name});
                };
            }
        }

        template<typename T, typename U>
        void assignEquallyNamedMembers(T& t, const U& u) {
            consteval {
                for (meta::info member : meta::data_member_range(reflexpr(T))) {
                    auto name = meta::name_of(member);
                    -> fragment {
                        idexpr(%{reflexpr(t)}).unqualid(%{name}) = idexpr(%{reflexpr(u)}).unqualid(%{name});
                    };
                }
            }
        }

        template <typename T>
        struct SoR {
            consteval {
                -> fragment struct {
                    using type = struct Type {
                        consteval {
                            genSoRMembers(reflexpr(T));
                        }

                        operator T() const {
                            T t;
                            assignEquallyNamedMembers(t, *this);
                            return t;
                        }

                        auto operator=(const T& t) -> Type& {
                            assignEquallyNamedMembers(*this, t);
                            return *this;
                        }
                    };
                };
            }
        };

        template <typename T>
        inline constexpr std::size_t memberCount = []() consteval {
            auto i = 0;
            for (meta::info member : meta::data_member_range(reflexpr(T)))
                i++;
            return i;
        }();

        template <typename T, std::size_t I>
        struct MemberTypeAt {
            consteval {
                auto i = 0;
                for (meta::info member : meta::data_member_range(reflexpr(T))) {
                    if (i == I) {
                        auto type = meta::type_of(member);
                        -> fragment struct {
                            using type = typename(%{type});
                        };
                    }
                    i++;
                }

            }
        };

        template <typename T, std::size_t... Is>
        auto tupleOfVectorsType(std::index_sequence<Is...>) {
            return std::tuple{(std::vector<typename MemberTypeAt<T, Is>::type>{})...};
        }
    }

    template <typename T>
    struct AoSContainer {
        using value_type = T;
        using reference = typename internal::SoR<T>::type;

        AoSContainer(std::size_t size) {
            resize(size);
        }

        void resize(std::size_t size) {
            resizeVectors(size, TupleIndexSeq{});
        }

        inline auto operator[](std::size_t i) -> reference {
            return constructReferenceAt(i, TupleIndexSeq{});
        }

        inline auto operator[](std::size_t i) const -> value_type {
            return constructValueAt(i, TupleIndexSeq{});
        }

    private:
        using TupleIndexSeq = std::make_index_sequence<internal::memberCount<T>>;
        using TupleOfVectors = decltype(internal::tupleOfVectorsType<T>(TupleIndexSeq{}));

        template <std::size_t... Is>
        inline auto constructReferenceAt(std::size_t i, std::index_sequence<Is...>) {
            return reference{std::get<Is>(tupleOfVectors)[i]...};
        }

        template <std::size_t... Is>
        inline auto constructValueAt(std::size_t i, std::index_sequence<Is...>) const {
            return value_type{std::get<Is>(tupleOfVectors)[i]...};
        }

        template <std::size_t... Is>
        inline void resizeVectors(std::size_t size, std::index_sequence<Is...>) {
            ((std::get<Is>(tupleOfVectors).resize(size)),...);
        }

        TupleOfVectors tupleOfVectors;
    };
}

constexpr auto PROBLEM_SIZE = 256;
constexpr auto TIMESTEP = 0.0001f;
constexpr auto EPS2 = 0.01f;

struct Particle {
    float posx, posy, posz;
    float velx, vely, velz;
    float mass;
};

inline void pPInteraction(auto& pi, const auto& pj) {
    auto xdistance = pi.posx - pj.posx;
    auto ydistance = pi.posy - pj.posy;
    auto zdistance = pi.posz - pj.posz;
    xdistance *= xdistance;
    ydistance *= ydistance;
    zdistance *= zdistance;
    const float distSqr = EPS2 + xdistance + ydistance + zdistance;
    const float distSixth = distSqr * distSqr * distSqr;
    const float invDistCube = 1.0f / std::sqrt(distSixth);
    const float sts = pj.mass * invDistCube * TIMESTEP;
    pi.velx += xdistance * sts;
    pi.vely += ydistance * sts;
    pi.velz += zdistance * sts;
}

void update(lib::AoSContainer<Particle>& particles) {
    #pragma clang loop vectorize(enable)
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
        Particle pi = particles[i]; // SoR<Particle>::operator Particle()
        //#pragma clang loop vectorize(enable)
        for (std::size_t j = 0; j < PROBLEM_SIZE; ++j)
            pPInteraction(pi, particles[j]);
        particles[i] = pi; // SoR<Particle>::operator=(Particle)
    }
}

int main() {
    lib::AoSContainer<Particle> c(PROBLEM_SIZE);

    std::default_random_engine engine;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i) {
        auto p = c[i]; // this is a reference!
        p.posx = dist(engine);
        p.posy = dist(engine);
        p.posz = dist(engine);
        p.velx = dist(engine);
        p.vely = dist(engine);
        p.velz = dist(engine);
        p.mass = dist(engine);
    }

    update(c);

    return c[42].posz;
}
