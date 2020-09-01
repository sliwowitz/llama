/* Copyright 2018 Alexander Matthes, Rene Widera
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

#include "Array.hpp"
#include "ForEach.hpp"
#include "Functions.hpp"
#include "Allocators.hpp"
#include "macros.hpp"
#include "mapping/One.hpp"

#include <boost/preprocessor/cat.hpp>
#include <type_traits>

namespace llama
{
    template<typename Mapping, typename BlobType>
    struct View;

    namespace internal
    {
        template<typename Allocator>
        using AllocatorBlobType
            = decltype(std::declval<Allocator>().allocate(0));

        template<typename Allocator, typename Mapping, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE static auto makeBlobArray(
            const Allocator & alloc,
            Mapping mapping,
            std::integer_sequence<std::size_t, Is...>)
            -> Array<AllocatorBlobType<Allocator>, Mapping::blobCount>
        {
            return {alloc.allocate(mapping.getBlobSize(Is))...};
        }
    }

    /** Creates a view based on the given mapping, e.g. \ref mapping::AoS or
     * \ref mapping::SoA. For allocating the view's underlying memory, the
     * specified allocator is used (or the default one, which is \ref
     * allocator::Vector). This function is the preferred way to create a \ref
     * View.
     */
    template<typename Mapping, typename Allocator = allocator::Vector<>>
    LLAMA_FN_HOST_ACC_INLINE auto
    allocView(Mapping mapping = {}, const Allocator & alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator>>
    {
        return {
            mapping,
            internal::makeBlobArray<Allocator>(
                alloc,
                mapping,
                std::make_index_sequence<Mapping::blobCount>{})};
    }

    /** Uses the \ref OneOnStackFactory to allocate one (probably temporary)
     * element for a given dimension and datum domain on the stack (no costly
     * allocation). \tparam Dimension dimension of the view \tparam DatumDomain
     * the datum domain for the one element mapping \return the allocated view
     * \see OneOnStackFactory
     */
    template<std::size_t Dim, typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
    {
        using Mapping = llama::mapping::One<UserDomain<Dim>, DatumDomain>;
        return allocView(
            Mapping{}, llama::allocator::Stack<SizeOf<DatumDomain>>{});
    }

    template<typename View>
    inline constexpr auto IsView = false;

    template<typename Mapping, typename BlobType>
    inline constexpr auto IsView<View<Mapping, BlobType>> = true;

    namespace internal
    {
        template<typename View>
        struct ViewByRefHolder
        {
            View & view;
        };

        template<typename View>
        struct ViewByValueHolder
        {
            View view;
        };
    }

    template<
        typename View,
        typename BoundDatumDomain = DatumCoord<>,
        bool OwnView = false>
    struct VirtualDatum;

    template<typename View>
    inline constexpr auto is_VirtualDatum = false;

    template<typename View, typename BoundDatumDomain, bool OwnView>
    inline constexpr auto
        is_VirtualDatum<VirtualDatum<View, BoundDatumDomain, OwnView>> = true;

    /** Uses the \ref allocViewStack to allocate a virtual datum with an own
     * bound view "allocated" on the stack. \tparam DatumDomain the datum
     * domain for the virtual datum \return the allocated virtual datum \see
     * allocViewStack
     */
    template<typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto stackVirtualDatumAlloc() -> VirtualDatum<
        decltype(llama::allocViewStack<1, DatumDomain>()),
        DatumCoord<>,
        true>
    {
        return {UserDomain<1>{}, llama::allocViewStack<1, DatumDomain>()};
    }

    /** Uses the \ref stackVirtualDatumAlloc to allocate a virtual datum with an
     * own bound view "allocated" on the stack bases on an existing virtual
     * datum, whose data is copied into the new virtual datum. \tparam
     * VirtualDatum type of the input virtual datum \return the virtual datum
     * copy on stack \see stackVirtualDatumAlloc
     */
    template<typename VirtualDatum>
    LLAMA_FN_HOST_ACC_INLINE auto stackVirtualDatumCopy(const VirtualDatum & vd)
        -> decltype(auto)
    {
        auto temp = stackVirtualDatumAlloc<
            typename VirtualDatum::AccessibleDatumDomain>();
        temp = vd;
        return temp;
    }

    template<
        typename LeftDatum,
        typename RightDatum,
        typename Source,
        typename LeftOuterCoord,
        typename LeftInnerCoord,
        typename OP>
    struct GenericInnerFunctor
    {
        template<typename RightOuterCoord, typename RightInnerCoord>
        LLAMA_FN_HOST_ACC_INLINE void
        operator()(RightOuterCoord, RightInnerCoord)
        {
            if constexpr(CompareUID<
                             typename LeftDatum::AccessibleDatumDomain,
                             LeftOuterCoord,
                             LeftInnerCoord,
                             typename RightDatum::AccessibleDatumDomain,
                             RightOuterCoord,
                             RightInnerCoord>)
            {
                using Dst = Cat<LeftOuterCoord, LeftInnerCoord>;
                using Src = Cat<RightOuterCoord, RightInnerCoord>;
                OP{}(left(Dst()), right(Src()));
            }
        }
        LeftDatum & left;
        const RightDatum & right;
    };

    template<
        typename LeftDatum,
        typename RightDatum,
        typename Source,
        typename OP>
    struct GenericFunctor
    {
        template<typename LeftOuterCoord, typename LeftInnerCoord>
        LLAMA_FN_HOST_ACC_INLINE void operator()(LeftOuterCoord, LeftInnerCoord)
        {
            ForEach<typename RightDatum::AccessibleDatumDomain, Source>::apply(
                GenericInnerFunctor<
                    LeftDatum,
                    RightDatum,
                    Source,
                    LeftOuterCoord,
                    LeftInnerCoord,
                    OP>{left, right});
        }
        LeftDatum & left;
        const RightDatum & right;
    };

    template<typename LeftDatum, typename RightType, typename OP>
    struct GenericTypeFunctor
    {
        template<typename OuterCoord, typename InnerCoord>
        LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
        {
            using Dst = Cat<OuterCoord, InnerCoord>;
            OP{}(left(Dst()), right);
        }
        LeftDatum & left;
        const RightType & right;
    };

    /** Macro that defines an operator overloading inside of \ref
     * llama::VirtualDatum for itself and a second virtual datum. \param OP
     * operator, e.g. operator += \param FUNCTOR used for calling the internal
     * needed functor to operate on the virtual datums, e.g. if FUNCTOR is
     * "Addition", the AdditionFunctor will be used internally. \param REF may
     * be & or && to determine whether it is an overloading for lvalue or rvalue
     * references
     * */
    /** Macro that defines an operator overloading inside of \ref
     * llama::VirtualDatum for itself and a view. Internally the virtual datum
     * at the first postion (all zeros) will be taken. This is useful for
     * one-element views (e.g. temporary views). \param OP operator, e.g.
     * operator += \param FUNCTOR used for calling the internal needed functor
     * to operate on the virtual datums, e.g. if FUNCTOR is "Addition", the
     * AdditionFunctor will be used internally. \param REF may be & or && to
     * determine whether it is an overloading for lvalue or rvalue references
     * */
    /** Macro that defines an operator overloading inside of \ref
     * llama::VirtualDatum for itself and some other type. \param OP operator,
     * e.g. operator += \param FUNCTOR used for calling the internal needed
     * functor to operate on the virtual datums, e.g. if FUNCTOR is "Addition",
     * the AdditionTypeFunctor will be used internally. \param REF may be & or
     * && to determine whether it is an overloading for lvalue or rvalue
     * references
     * */
#define __LLAMA_VIRTUALDATUM_OPERATOR(OP, FUNCTOR) \
    template< \
        typename OtherView, \
        typename OtherBoundDatumDomain, \
        bool OtherOwnView> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView> & \
            other) \
        ->VirtualDatum & \
    { \
        GenericFunctor< \
            std::remove_reference_t<decltype(*this)>, \
            VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView>, \
            DatumCoord<>, \
            FUNCTOR> \
            functor{*this, other}; \
        ForEach<AccessibleDatumDomain, DatumCoord<>>::apply(functor); \
        return *this; \
    } \
\
    template<typename OtherMapping, typename OtherBlobType> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const llama::View<OtherMapping, OtherBlobType> & other) \
        ->VirtualDatum & \
    { \
        return *this OP other( \
            llama::UserDomain<OtherMapping::UserDomain::count>{}); \
    } \
\
    template< \
        typename OtherType, \
        typename \
        = std::enable_if_t<!IsView<OtherType> && !is_VirtualDatum<OtherType>>> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP(const OtherType & other) \
        ->VirtualDatum & \
    { \
        GenericTypeFunctor<decltype(*this), OtherType, FUNCTOR> functor{ \
            *this, other}; \
        ForEach<AccessibleDatumDomain, DatumCoord<>>::apply(functor); \
        return *this; \
    }

    template<
        typename LeftDatum,
        typename LeftBase,
        typename LeftLocal,
        typename RightDatum,
        typename OP>
    struct GenericBoolInnerFunctor
    {
        template<typename OuterCoord, typename InnerCoord>
        LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
        {
            if constexpr(CompareUID<
                             typename LeftDatum::Mapping::DatumDomain,
                             LeftBase,
                             LeftLocal,
                             typename RightDatum::Mapping::DatumDomain,
                             OuterCoord,
                             InnerCoord>)
            {
                using Dst = Cat<LeftBase, LeftLocal>;
                using Src = Cat<OuterCoord, InnerCoord>;
                result &= OP{}(left(Dst()), right(Src()));
            }
        }
        const LeftDatum & left;
        const RightDatum & right;
        bool result;
    };

    template<
        typename LeftDatum,
        typename RightDatum,
        typename Source,
        typename OP>
    struct GenericBoolFunctor
    {
        template<typename OuterCoord, typename InnerCoord>
        LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
        {
            GenericBoolInnerFunctor<
                LeftDatum,
                OuterCoord,
                InnerCoord,
                RightDatum,
                OP>
                functor{left, right, true};
            ForEach<typename RightDatum::AccessibleDatumDomain, Source>::apply(
                functor);
            result &= functor.result;
        }
        const LeftDatum & left;
        const RightDatum & right;
        bool result;
    };

    template<typename LeftDatum, typename RightType, typename OP>
    struct GenericBoolTypeFunctor
    {
        template<typename OuterCoord, typename InnerCoord>
        LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
        {
            using Dst = Cat<OuterCoord, InnerCoord>;
            result &= OP{}(
                left(Dst()),
                static_cast<std::remove_reference_t<decltype(left(Dst()))>>(
                    right));
        }
        const LeftDatum & left;
        const RightType & right;
        bool result;
    };

/** Macro that defines a boolean operator overloading inside of
 *  \ref llama::VirtualDatum for itself and a second virtual datum.
 * \param OP operator, e.g. operator >=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "BiggerSameThan", the
 *        BiggerSameThanBoolFunctor will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * \return result of the boolean operation for every combination with the same
 *  UID
 * */
/** Macro that defines a boolean operator overloading inside of
 *  \ref llama::VirtualDatum for itself and a view. Internally the virtual datum
 * at the first postion (all zeros) will be taken. This is useful for
 * one-element views (e.g. temporary views). \param OP operator, e.g. operator
 * >= \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "BiggerSameThan", the
 *        BiggerSameThanBoolFunctor will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * \return result of the boolean operation for every combination with the same
 *  UID
 * */
/** Macro that defines a boolean operator overloading inside of
 *  \ref llama::VirtualDatum for itself and some other type.
 * \param OP operator, e.g. operator >=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "BiggerSameThan", the
 *        BiggerSameThanBoolTypeFunctor will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * \return result of the boolean operation for every combination
 * */
#define __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(OP, FUNCTOR) \
    template< \
        typename OtherView, \
        typename OtherBoundDatumDomain, \
        bool OtherOwnView> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView> & \
            other) const->bool \
    { \
        GenericBoolFunctor< \
            std::remove_reference_t<decltype(*this)>, \
            VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView>, \
            DatumCoord<>, \
            FUNCTOR> \
            functor{*this, other, true}; \
        ForEach<AccessibleDatumDomain, DatumCoord<>>::apply(functor); \
        return functor.result; \
    } \
\
    template<typename OtherMapping, typename OtherBlobType> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const llama::View<OtherMapping, OtherBlobType> & other) const->bool \
    { \
        return *this OP other( \
            llama::UserDomain<OtherMapping::UserDomain::count>{}); \
    } \
\
    template<typename OtherType> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP(const OtherType & other) \
        const->bool \
    { \
        GenericBoolTypeFunctor<decltype(*this), OtherType, FUNCTOR> functor{ \
            *this, other, true}; \
        ForEach<AccessibleDatumDomain, DatumCoord<>>::apply(functor); \
        return functor.result; \
    }

    struct Assignment
    {
        template<typename A, typename B>
        LLAMA_FN_HOST_ACC_INLINE decltype(auto)
        operator()(A & a, const B & b) const
        {
            return a = b;
        }
    };

    struct Addition
    {
        template<typename A, typename B>
        LLAMA_FN_HOST_ACC_INLINE decltype(auto)
        operator()(A & a, const B & b) const
        {
            return a += b;
        }
    };

    struct Subtraction
    {
        template<typename A, typename B>
        LLAMA_FN_HOST_ACC_INLINE decltype(auto)
        operator()(A & a, const B & b) const
        {
            return a -= b;
        }
    };

    struct Multiplication
    {
        template<typename A, typename B>
        LLAMA_FN_HOST_ACC_INLINE decltype(auto)
        operator()(A & a, const B & b) const
        {
            return a *= b;
        }
    };

    struct Division
    {
        template<typename A, typename B>
        LLAMA_FN_HOST_ACC_INLINE decltype(auto)
        operator()(A & a, const B & b) const
        {
            return a /= b;
        }
    };

    struct Modulo
    {
        template<typename A, typename B>
        LLAMA_FN_HOST_ACC_INLINE decltype(auto)
        operator()(A & a, const B & b) const
        {
            return a %= b;
        }
    };

    template<typename T>
    auto as_mutable(const T & t) -> T &
    {
        return const_cast<T &>(t);
    }

    /** Virtual data type returned by \ref View after resolving user domain
     * address, being "virtual" in that sense that the data of the virtual datum
     * are not part of the struct itself but a helper object to address them in
     * the compile time datum domain. Beside the user domain, also a part of the
     * compile time domain may be resolved for access like `datum( Pos ) +=
     * datum( Vel )`. \tparam T_View parent view of the virtual datum \tparam
     * BoundDatumDomain optional \ref DatumCoord which restricts the virtual
     * datum to a smaller part of the datum domain
     */
    template<typename T_View, typename BoundDatumDomain, bool OwnView>
    struct VirtualDatum
    {
        using View = T_View; ///< parent view of the virtual datum
        using Mapping =
            typename View::Mapping; ///< mapping of the underlying view
        using UserDomain = typename Mapping::UserDomain;
        using DatumDomain = typename Mapping::DatumDomain;

        const UserDomain
            userDomainPos; ///< resolved position in the user domain
        std::conditional_t<OwnView, View, View &> view;

        /** Sub part of the datum domain of the view/mapping relative to
         *  \ref BoundDatumDomain. If BoundDatumDomain is `DatumCoord<>`
         * (default) AccessibleDatumDomain is the same as
         * `Mapping::DatumDomain`.
         */
        using AccessibleDatumDomain = GetType<DatumDomain, BoundDatumDomain>;

        LLAMA_FN_HOST_ACC_INLINE
        VirtualDatum(
            UserDomain userDomainPos,
            std::conditional_t<OwnView, View &&, View &> view) :
                userDomainPos(userDomainPos),
                view{static_cast<decltype(view)>(view)}
        {}

        VirtualDatum(const VirtualDatum &) = default;
        VirtualDatum(VirtualDatum &&) = default;
        // VirtualDatum & operator=(const VirtualDatum &) = delete;
        // VirtualDatum & operator=(VirtualDatum &&) = delete;

        /** Explicit access function for a coordinate in the datum domain given
         * as tree position indexes. If the address -- independently whether
         * given as datum coord or UID -- is not a leaf but, a new virtual datum
         * with a bound datum coord is returned. \tparam Coord... variadic
         * number std::size_t numbers as tree coordinates \return reference to
         * element at resolved user domain and given datum domain coordinate or
         * a new virtual datum with a bound datum coord
         */
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto access(DatumCoord<Coord...> = {}) const
            -> decltype(auto)
        {
            if constexpr(IsDatumStruct<GetType<
                             DatumDomain,
                             Cat<BoundDatumDomain, DatumCoord<Coord...>>>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<
                    const View,
                    Cat<BoundDatumDomain, DatumCoord<Coord...>>>{
                    userDomainPos, this->view};
            }
            else
            {
                using DatumCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, DatumCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto access(DatumCoord<Coord...> coord = {})
            -> decltype(auto)
        {
            if constexpr(IsDatumStruct<GetType<
                             DatumDomain,
                             Cat<BoundDatumDomain, DatumCoord<Coord...>>>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<
                    View,
                    Cat<BoundDatumDomain, DatumCoord<Coord...>>>{
                    userDomainPos, this->view};
            }
            else
            {
                using DatumCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, DatumCoord{});
            }
        }

        /** Explicit access function for a coordinate in the datum domain given
         * as unique identifier or \ref DatumCoord. If the address --
         * independently whether given as datum coord or UID -- is not a leaf
         * but, a new virtual datum with a bound datum coord is returned.
         * \tparam UIDs... variadic number of types as unique
         *  identifier
         * \return reference to element at resolved user domain and given datum
         *  domain coordinate or a new virtual datum with a bound datum coord
         */
        template<typename... UIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access(UIDs...) const -> decltype(auto)
        {
            using DatumCoord = GetCoordFromUIDRelative<
                DatumDomain,
                BoundDatumDomain,
                UIDs...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template<typename... UIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access(UIDs... uids) -> decltype(auto)
        {
            using DatumCoord = GetCoordFromUIDRelative<
                DatumDomain,
                BoundDatumDomain,
                UIDs...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoord{});
        }

        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access() const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access() -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

        /** operator overload() for a coordinate in the datum domain given as
         *  unique identifier or \ref DatumCoord. If the address --
         * independently whether given as datum coord or UID -- is not a leaf
         * but, a new virtual datum with a bound datum coord is returned. \param
         * unnamed instantiation of variadic number of unique
         *  identifier types **or** \ref DatumCoord with tree coordinates as
         *  template parameters inside
         * \return reference to element at resolved user domain and given datum
         *  domain coordinate or a new virtual datum with a bound datum coord
         */
        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoordOrUIDs...) const
            -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoordOrUIDs...)
            -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

        __LLAMA_VIRTUALDATUM_OPERATOR(=, Assignment)
        __LLAMA_VIRTUALDATUM_OPERATOR(+=, Addition)
        __LLAMA_VIRTUALDATUM_OPERATOR(-=, Subtraction)
        __LLAMA_VIRTUALDATUM_OPERATOR(*=, Multiplication)
        __LLAMA_VIRTUALDATUM_OPERATOR(/=, Division)
        __LLAMA_VIRTUALDATUM_OPERATOR(%=, Modulo)

        // we need this one to disable the compiler generated copy assignment
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const VirtualDatum & other)
            -> VirtualDatum &
        {
            return this->operator=<>(other);
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator+(const OtherType & other)
        {
            return stackVirtualDatumCopy(*this) += other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator-(const OtherType & other)
        {
            return stackVirtualDatumCopy(*this) -= other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator*(const OtherType & other)
        {
            return stackVirtualDatumCopy(*this) *= other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator/(const OtherType & other)
        {
            return stackVirtualDatumCopy(*this) /= other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator%(const OtherType & other)
        {
            return stackVirtualDatumCopy(*this) %= other;
        }

        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(==, std::equal_to<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(!=, std::not_equal_to<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(<, std::less<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(<=, std::less_equal<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(>, std::greater<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(>=, std::greater_equal<>)
    };

    /** Central LLAMA class holding memory and giving access to it defined by a
     *  mapping. Should not be instantiated "by hand" but with a \ref Factory.
     * \tparam T_Mapping the mapping of the view
     * \tparam T_BlobType the background data type of the raw data, at the
     * moment always an 8 bit type like "unsigned char"
     */
    template<typename T_Mapping, typename BlobType>
    struct View
    {
        using Mapping = T_Mapping; ///< used mapping
        using UserDomain = typename Mapping::UserDomain;
        using DatumDomain = typename Mapping::DatumDomain;
        using VirtualDatumType = VirtualDatum<View<Mapping, BlobType>>;
        using VirtualDatumTypeConst
            = VirtualDatum<const View<Mapping, BlobType>>;

        View() = default;
        // View(View &&) noexcept = default;
        // auto operator=(View &&) noexcept -> View & = default;

        LLAMA_NO_HOST_ACC_WARNING
        LLAMA_FN_HOST_ACC_INLINE
        View(Mapping mapping, Array<BlobType, Mapping::blobCount> blob) :
                mapping(mapping), blob(blob)
        {}

        /** Operator overloading to reverse the order of compile time (datum
         * domain) and run time (user domain) parameter with a helper object
         *  (\ref llama::VirtualDatum). Should be favoured to access data
         * because of the more array of struct like interface and the handy
         * intermediate \ref llama::VirtualDatum object. \param userDomain user
         * domain as \ref UserDomain \return \ref llama::VirtualDatum with bound
         * user domain, which can be used to access the datum domain
         */
        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {userDomain, *this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {userDomain, *this};
        }

        /** Operator overloading to reverse the order of compile time (datum
         * domain) and run time (user domain) parameter with a helper object
         *  (\ref llama::VirtualDatum). Should be favoured to access data
         * because of the more array of struct like interface and the handy
         * intermediate \ref llama::VirtualDatum object. \tparam Coord...
         * types of user domain coordinates \param coord user domain as list of
         * numbers \return \ref llama::VirtualDatum with bound user domain,
         * which can be used to access the datum domain
         */
        template<typename... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Coord... coord) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {UserDomain{coord...}, *this};
        }

        template<typename... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Coord... coord)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {UserDomain{coord...}, *this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(std::size_t coord = 0) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {UserDomain{coord}, *this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(std::size_t coord = 0)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {UserDomain{coord}, *this};
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto
        operator()(DatumCoord<Coord...> dc = {}) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(UserDomain{});
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoord<Coord...> dc = {})
            -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(UserDomain{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](UserDomain userDomain) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(userDomain);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](UserDomain userDomain)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(userDomain);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t coord) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(coord);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t coord)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(coord);
        }

        Mapping mapping; ///< mapping of the view
        Array<BlobType, Mapping::blobCount> blob; ///< memory of the view

    private:
        template<typename T_View, typename T_BoundDatumDomain, bool OwnView>
        friend struct VirtualDatum;

        LLAMA_NO_HOST_ACC_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto
        accessor(UserDomain userDomain, DatumCoord<Coords...> = {}) const
            -> const auto &
        {
            const auto [nr, offset]
                = mapping.template getBlobNrAndOffset<Coords...>(userDomain);
            using Type = GetType<DatumDomain, DatumCoord<Coords...>>;
            return reinterpret_cast<const Type &>(blob[nr][offset]);
        }

        LLAMA_NO_HOST_ACC_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto
        accessor(UserDomain userDomain, DatumCoord<Coords...> coord = {})
            -> auto &
        {
            const auto [nr, offset]
                = mapping.template getBlobNrAndOffset<Coords...>(userDomain);
            using Type = GetType<DatumDomain, DatumCoord<Coords...>>;
            return reinterpret_cast<Type &>(blob[nr][offset]);
        }
    };

    /** Struct which acts like a \ref View, but shows only a smaller and/or
     * shifted part of a parental (virtual) view. A virtual view does not hold
     * any memory itself but has a reference to the parent view (and its
     * memory). \tparam T_ParentViewType type of the parent, can also be another
     * VirtualView
     */
    template<typename T_ParentViewType>
    struct VirtualView
    {
        using ParentView = T_ParentViewType; ///< type of parent view
        using Mapping = typename ParentView::Mapping; ///< mapping type, gotten
                                                      ///< from parent view
        using UserDomain = typename Mapping::UserDomain;
        using VirtualDatumType =
            typename ParentView::VirtualDatumType; ///< VirtualDatum type,
                                                   ///< gotten from parent view

        /** Unlike a \ref View, a VirtualView can be created without a factory
         *  directly from a parent view.
         * \param parentView a reference to the parental view. Meaning, the
         * parental view should not get out of scope before the virtual view.
         * \param position shifted position relative to the parental view
         * \param size size of the virtual view
         */
        LLAMA_NO_HOST_ACC_WARNING
        LLAMA_FN_HOST_ACC_INLINE
        VirtualView(
            ParentView & parentView,
            UserDomain position,
            UserDomain size) :
                parentView(parentView), position(position), size(size)
        {}

        /** Explicit access function taking the datum domain as tree index
         *  coordinate template arguments and the user domain as runtime
         * parameter. The operator() overloadings should be preferred as they
         * show a more array of struct like interface using \ref VirtualDatum.
         * \tparam Coords... tree index coordinate
         * \param userDomain user domain as \ref UserDomain
         * \return reference to element
         */
        LLAMA_NO_HOST_ACC_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(UserDomain userDomain) const
            -> const auto &
        {
            return parentView.template accessor<Coords...>(
                userDomain + position);
        }

        LLAMA_NO_HOST_ACC_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(UserDomain userDomain) -> auto &
        {
            return parentView.template accessor<Coords...>(
                userDomain + position);
        }

        /** Explicit access function taking the datum domain as UID type list
         *  template arguments and the user domain as runtime parameter.
         *  The operator() overloadings should be preferred as they show a more
         *  array of struct like interface using \ref VirtualDatum.
         * \tparam UIDs... UID type list
         * \param userDomain user domain as \ref UserDomain
         * \return reference to element
         */
        LLAMA_NO_HOST_ACC_WARNING
        template<typename... UIDs>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(UserDomain userDomain) const
            -> const auto &
        {
            return parentView.template accessor<UIDs...>(
                userDomain + position);
        }

        LLAMA_NO_HOST_ACC_WARNING
        template<typename... UIDs>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(UserDomain userDomain) -> auto &
        {
            return parentView.template accessor<UIDs...>(
                userDomain + position);
        }

        /** Operator overloading to reverse the order of compile time (datum
         * domain) and run time (user domain) parameter with a helper object
         *  (\ref VirtualDatum). Should be favoured to access data because of
         * the more array of struct like interface and the handy intermediate
         *  \ref VirtualDatum object.
         * \param userDomain user domain as \ref UserDomain
         * \return \ref VirtualDatum with bound user domain, which can be used
         * to access the datum domain
         */
        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain) const
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(userDomain + position);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(userDomain + position);
        }

        /** Operator overloading to reverse the order of compile time (datum
         * domain) and run time (user domain) parameter with a helper object
         *  (\ref VirtualDatum). Should be favoured to access data because of
         * the more array of struct like interface and the handy intermediate
         *  \ref VirtualDatum object.
         * \tparam Coord... types of user domain coordinates
         * \param coord user domain as list of numbers
         * \return \ref VirtualDatum with bound user domain, which can be used
         * to access the datum domain
         */
        template<typename... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Coord... coord) const
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(UserDomain{coord...} + position);
        }

        template<typename... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Coord... coord)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(UserDomain{coord...} + position);
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(std::size_t coord = 0) const -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(UserDomain{coord} + position);
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(std::size_t coord = 0) -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(UserDomain{coord} + position);
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto
        operator()(DatumCoord<Coord...> && dc = {}) const -> const auto &
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(UserDomain{});
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto
        operator()(DatumCoord<Coord...> && dc = {}) -> auto &
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(UserDomain{});
        }

        ParentView & parentView; ///< reference to parental view
        const UserDomain position; ///< shifted position in parental view
        const UserDomain size; ///< shown size
    };
} // namespace llama
