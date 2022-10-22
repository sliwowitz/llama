// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <boost/mp11.hpp>

#if BOOST_MP11_VERSION < 107300
//  Copyright 2015 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
//
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

namespace boost::mp11
{
    namespace detail
    {
        template<class L2>
        struct mp_flatten_impl
        {
            template<class T>
            using fn = mp_if<mp_similar<L2, T>, T, mp_list<T>>;
        };
    } // namespace detail

    template<class L, class L2 = mp_clear<L>>
    using mp_flatten = mp_apply<mp_append, mp_push_front<mp_transform_q<detail::mp_flatten_impl<L2>, L>, mp_clear<L>>>;
} // namespace boost::mp11
#endif

namespace llama
{
    namespace internal
    {
        template<typename FromList, template<auto...> class ToList>
        struct mp_unwrap_values_into_impl;

        template<template<class...> class FromList, typename... Values, template<auto...> class ToList>
        struct mp_unwrap_values_into_impl<FromList<Values...>, ToList>
        {
            using type = ToList<Values::value...>;
        };

        template<typename FromList, template<auto...> class ToList>
        using mp_unwrap_values_into = typename mp_unwrap_values_into_impl<FromList, ToList>::type;

        template<typename E, typename... Args>
        struct ReplacePlaceholdersImpl
        {
            using type = E;
        };
        template<std::size_t I, typename... Args>
        struct ReplacePlaceholdersImpl<boost::mp11::mp_arg<I>, Args...>
        {
            using type = boost::mp11::mp_at_c<boost::mp11::mp_list<Args...>, I>;
        };

        template<template<typename...> typename E, typename... Ts, typename... Args>
        struct ReplacePlaceholdersImpl<E<Ts...>, Args...>
        {
            using type = E<typename ReplacePlaceholdersImpl<Ts, Args...>::type...>;
        };
    } // namespace internal

    template<typename Expression, typename... Args>
    using ReplacePlaceholders = typename internal::ReplacePlaceholdersImpl<Expression, Args...>::type;
} // namespace llama