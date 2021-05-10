// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "View.hpp"

#include <type_traits>

namespace llama
{
    template <typename View, typename BoundRecordCoord, bool OwnView>
    struct VirtualRecord;

    template <typename View>
    inline constexpr auto is_VirtualRecord = false;

    template <typename View, typename BoundRecordCoord, bool OwnView>
    inline constexpr auto is_VirtualRecord<VirtualRecord<View, BoundRecordCoord, OwnView>> = true;

    /// A \ref VirtualRecord that owns and holds a single value.
    template <typename RecordDim>
    using One = VirtualRecord<decltype(allocViewStack<1, RecordDim>()), RecordCoord<>, true>;

    /// Creates a single \ref VirtualRecord owning a view with stack memory and
    /// copies all values from an existing \ref VirtualRecord.
    template <typename VirtualRecord>
    LLAMA_FN_HOST_ACC_INLINE auto copyVirtualRecordStack(const VirtualRecord& vd) -> decltype(auto)
    {
        One<typename VirtualRecord::AccessibleRecordDim> temp;
        temp = vd;
        return temp;
    }

    namespace internal
    {
        template <
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(
            LeftRecord& left,
            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> LeftRecord&
        {
            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                [&](auto leftCoord)
                {
                    using LeftInnerCoord = decltype(leftCoord);
                    forEachLeaf<typename RightRecord::AccessibleRecordDim>(
                        [&](auto rightCoord)
                        {
                            using RightInnerCoord = decltype(rightCoord);
                            if constexpr (hasSameTags<
                                              typename LeftRecord::AccessibleRecordDim,
                                              LeftInnerCoord,
                                              typename RightRecord::AccessibleRecordDim,
                                              RightInnerCoord>)
                            {
                                Functor{}(left(leftCoord), right(rightCoord));
                            }
                        });
                });
            return left;
        }

        template <typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(LeftRecord& left, const T& right) -> LeftRecord&
        {
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>([&](auto leftCoord)
                                                                  { Functor{}(left(leftCoord), right); });
            return left;
        }

        template <
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(
            const LeftRecord& left,
            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> bool
        {
            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
            bool result = true;
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                [&](auto leftCoord)
                {
                    using LeftInnerCoord = decltype(leftCoord);
                    forEachLeaf<typename RightRecord::AccessibleRecordDim>(
                        [&](auto rightCoord)
                        {
                            using RightInnerCoord = decltype(rightCoord);
                            if constexpr (hasSameTags<
                                              typename LeftRecord::AccessibleRecordDim,
                                              LeftInnerCoord,
                                              typename RightRecord::AccessibleRecordDim,
                                              RightInnerCoord>)
                            {
                                result &= Functor{}(left(leftCoord), right(rightCoord));
                            }
                        });
                });
            return result;
        }

        template <typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(const LeftRecord& left, const T& right) -> bool
        {
            bool result = true;
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                [&](auto leftCoord) {
                    result &= Functor{}(
                        left(leftCoord),
                        static_cast<std::remove_reference_t<decltype(left(leftCoord))>>(right));
                });
            return result;
        }

        struct Assign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a = b;
            }
        };

        struct PlusAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a += b;
            }
        };

        struct MinusAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a -= b;
            }
        };

        struct MultiplyAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a *= b;
            }
        };

        struct DivideAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a /= b;
            }
        };

        struct ModuloAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a %= b;
            }
        };

        template <typename TWithOptionalConst, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(TWithOptionalConst& leaf, T) -> std::
            enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::reference_wrapper<TWithOptionalConst>>
        {
            return leaf;
        }

        template <typename VirtualRecord, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(VirtualRecord&& vd, Record<Fields...>)
        {
            return std::make_tuple(asTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template <typename TWithOptionalConst, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(TWithOptionalConst& leaf, T)
            -> std::enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::tuple<TWithOptionalConst&>>
        {
            return {leaf};
        }

        template <typename VirtualRecord, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(VirtualRecord&& vd, Record<Fields...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template <typename T, typename = void>
        constexpr inline auto isTupleLike = false;

        // get<I>(t) and std::tuple_size<T> must be available
        using std::get; // make sure a get<0>() can be found, so the compiler can compile the trait
        template <typename T>
        constexpr inline auto
            isTupleLike<T, std::void_t<decltype(get<0>(std::declval<T>())), std::tuple_size<T>>> = true;

        template <typename... Ts>
        constexpr inline auto dependentFalse = false;

        template <typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>);

        template <typename T1, typename T2>
        LLAMA_FN_HOST_ACC_INLINE void assignTupleElement(T1&& dst, T2&& src)
        {
            if constexpr (isTupleLike<std::decay_t<T1>> && isTupleLike<std::decay_t<T2>>)
            {
                static_assert(std::tuple_size_v<std::decay_t<T1>> == std::tuple_size_v<std::decay_t<T2>>);
                assignTuples(dst, src, std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>{});
            }
            else if constexpr (!isTupleLike<std::decay_t<T1>> && !isTupleLike<std::decay_t<T2>>)
                std::forward<T1>(dst) = std::forward<T2>(src);
            else
                static_assert(dependentFalse<T1, T2>, "Elements to assign are not tuple/tuple or non-tuple/non-tuple.");
        }

        template <typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>)
        {
            static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>);
            using std::get;
            (assignTupleElement(get<Is>(std::forward<Tuple1>(dst)), get<Is>(std::forward<Tuple2>(src))), ...);
        }

        template <typename T, typename Tuple, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeFromTuple(Tuple&& src, std::index_sequence<Is...>)
        {
            using std::get;
            return T{get<Is>(std::forward<Tuple>(src))...};
        }

        template <typename T, typename SFINAE, typename... Args>
        constexpr inline auto isDirectListInitializableImpl = false;

        template <typename T, typename... Args>
        constexpr inline auto
            isDirectListInitializableImpl<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...> = true;

        template <typename T, typename... Args>
        constexpr inline auto isDirectListInitializable = isDirectListInitializableImpl<T, void, Args...>;

        template <typename T, typename Tuple>
        constexpr inline auto isDirectListInitializableFromTuple = false;

        template <typename T, template <typename...> typename Tuple, typename... Args>
        constexpr inline auto
            isDirectListInitializableFromTuple<T, Tuple<Args...>> = isDirectListInitializable<T, Args...>;
    } // namespace internal

    /// Virtual record type returned by \ref View after resolving an array dimensions coordinate or partially resolving
    /// a \ref RecordCoord. A virtual record does not hold data itself (thus named "virtual"), it just binds enough
    /// information (array dimensions coord and partial record coord) to retrieve it from a \ref View later. Virtual
    /// records should not be created by the user. They are returned from various access functions in \ref View and
    /// VirtualRecord itself.
    template <typename T_View, typename BoundRecordCoord, bool OwnView>
    struct VirtualRecord
    {
        using View = T_View; ///< View this virtual record points into.

    private:
        using ArrayDims = typename View::Mapping::ArrayDims;
        using RecordDim = typename View::Mapping::RecordDim;

        const ArrayDims arrayDimsCoord;
        std::conditional_t<OwnView, View, View&> view;

    public:
        /// Subtree of the record dimension of View starting at
        /// BoundRecordCoord. If BoundRecordCoord is `RecordCoord<>` (default)
        /// AccessibleRecordDim is the same as `Mapping::RecordDim`.
        using AccessibleRecordDim = GetType<RecordDim, BoundRecordCoord>;

        LLAMA_FN_HOST_ACC_INLINE VirtualRecord()
            /* requires(OwnView) */
            : arrayDimsCoord({})
            , view{allocViewStack<1, RecordDim>()}
        {
            static_assert(OwnView, "The default constructor of VirtualRecord is only available if the ");
        }

        LLAMA_FN_HOST_ACC_INLINE
        VirtualRecord(ArrayDims arrayDimsCoord, std::conditional_t<OwnView, View&&, View&> view)
            : arrayDimsCoord(arrayDimsCoord)
            , view{static_cast<decltype(view)>(view)}
        {
        }

        VirtualRecord(const VirtualRecord&) = default;
        VirtualRecord(VirtualRecord&&) = default;

        /// Access a record in the record dimension underneath the current virtual
        /// record using a \ref RecordCoord. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual record.
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            if constexpr (isRecord<GetType<RecordDim, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<const View, AbsolutCoord>{arrayDimsCoord, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayDimsCoord, AbsolutCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> coord = {}) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            if constexpr (isRecord<GetType<RecordDim, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<View, AbsolutCoord>{arrayDimsCoord, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayDimsCoord, AbsolutCoord{});
            }
        }

        /// Access a record in the record dimension underneath the current virtual
        /// record using a series of tags. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual record.
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        // we need this one to disable the compiler generated copy assignment
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const VirtualRecord& other) -> VirtualRecord&
        {
            return this->operator=<VirtualRecord>(other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::Assign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator+=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::PlusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator-=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::MinusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator*=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::MultiplyAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator/=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::DivideAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator%=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::ModuloAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) += t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const T& t, const VirtualRecord& vd)
        {
            return vd + t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator-(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) -= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) *= t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const T& t, const VirtualRecord& vd)
        {
            return vd * t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator/(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) /= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator%(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) %= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd == t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::not_equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd != t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::less<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd > t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::less_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd >= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::greater<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd < t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::greater_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd <= t;
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple()
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple() const
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple()
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple() const
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() const -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        struct Loader
        {
            VirtualRecord& vd;

            template <typename T>
            LLAMA_FN_HOST_ACC_INLINE operator T()
            {
                return vd.loadAs<T>();
            }
        };

        struct LoaderConst
        {
            const VirtualRecord& vd;

            template <typename T>
            LLAMA_FN_HOST_ACC_INLINE operator T() const
            {
                return vd.loadAs<T>();
            }
        };

        LLAMA_FN_HOST_ACC_INLINE auto load() -> Loader
        {
            return {*this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto load() const -> LoaderConst
        {
            return {*this};
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE void store(const TupleLike& t)
        {
            internal::assignTuples(asTuple(), t, std::make_index_sequence<std::tuple_size_v<TupleLike>>{});
        }
    };
} // namespace llama

template <typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_size<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
    : boost::mp11::mp_size<typename llama::VirtualRecord<View, BoundRecordCoord, OwnView>::AccessibleRecordDim>
{
};

template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
{
    using type = decltype(std::declval<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
};

template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
{
    using type
        = decltype(std::declval<const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
};