// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "Core.hpp"

#include <stdexcept>
#include <string_view>

namespace llama
{
    namespace internal
    {
        // TODO(bgruber): just use std::copy which became constexpr in C++20
        template<typename In, typename Out>
        constexpr auto constexprCopy(In f, In l, Out d) -> Out
        {
            while(f != l)
                *d++ = *f++;
            return d;
        }

        // TODO(bgruber): just use std::search which became constexpr in C++20
        // from: https://en.cppreference.com/w/cpp/algorithm/search
        template<class ForwardIt1, class ForwardIt2>
        constexpr auto constexprSearch(ForwardIt1 first, ForwardIt1 last, ForwardIt2 sFirst, ForwardIt2 sLast)
            -> ForwardIt1
        {
            while(true)
            {
                ForwardIt1 it = first;
                for(ForwardIt2 sIt = sFirst;; ++it, ++sIt)
                {
                    if(sIt == sLast)
                        return first;
                    if(it == last)
                        return last;
                    if(!(*it == *sIt))
                        break;
                }
                ++first;
            }
        }

        // TODO(bgruber): just use std::remove_copy which became constexpr in C++20
        // from: https://en.cppreference.com/w/cpp/algorithm/remove_copy
        template<class InputIt, class OutputIt, class T>
        constexpr auto constexprRemoveCopy(InputIt first, InputIt last, OutputIt d_first, const T& value) -> OutputIt
        {
            for(; first != last; ++first)
            {
                if(!(*first == value))
                {
                    *d_first++ = *first;
                }
            }
            return d_first;
        }

        // TODO(bgruber): just use std::count which became constexpr in C++20
        // from: https://en.cppreference.com/w/cpp/algorithm/count
        template<class InputIt, class T>
        auto constexprCount(InputIt first, InputIt last, const T& value) ->
            typename std::iterator_traits<InputIt>::difference_type
        {
            typename std::iterator_traits<InputIt>::difference_type ret = 0;
            for(; first != last; ++first)
            {
                if(*first == value)
                {
                    ret++;
                }
            }
            return ret;
        }

        template<std::size_t NewSize, typename T, std::size_t N>
        constexpr auto resizeArray(Array<T, N> a)
        {
            Array<char, NewSize> r{};
            constexprCopy(a.begin(), a.begin() + NewSize, r.begin());
            return r;
        }

        template<typename T>
        constexpr auto typeNameAsArray()
        {
            // adapted from Matthew Rodusek:
            // https://bitwizeshift.github.io/posts/2021/03/09/getting-an-unmangled-type-name-at-compile-time/
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

#if defined(__clang__)
            constexpr auto prefix = std::string_view{"[T = "};
            constexpr auto suffix = std::string_view{"]"};
            constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
            constexpr auto prefix = std::string_view{"with T = "};
            constexpr auto suffix = std::string_view{"]"};
            constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
            constexpr auto prefix = std::string_view{"typeNameAsArray<"};
            constexpr auto suffix = std::string_view{">(void)"};
            constexpr auto function = std::string_view{__FUNCSIG__};
#else
#    warning Unsupported compiler
            constexpr auto prefix = std::string_view{};
            constexpr auto suffix = std::string_view{};
            constexpr auto function = std::string_view{};
#endif

            constexpr auto start = function.find(prefix) + prefix.size();
            constexpr auto end = function.rfind(suffix);
            static_assert(start <= end);

            constexpr auto name = function.substr(start, (end - start));

            constexpr auto arrAndSize = [&]() constexpr
            {
                Array<char, name.size()> nameArray{};
                constexprCopy(name.begin(), name.end(), nameArray.begin());

#ifdef _MSC_VER
                // MSVC 19.32 runs into a syntax error if we just capture nameArray. Passing it as argument is a
                // workaround. Applies to the following lambdas.

                // strip "struct " and "class ".
                auto removeAllOccurences = [](auto& nameArray, std::size_t size, std::string_view str) constexpr
                {
                    auto e = nameArray.begin() + size;
                    while(true)
                    {
                        auto it = constexprSearch(nameArray.begin(), e, str.begin(), str.end());
                        if(it == e)
                            break;
                        constexprCopy(it + str.size(), e, it);
                        e -= str.size();
                    }
                    return e - nameArray.begin();
                };

                auto size1 = removeAllOccurences(nameArray, nameArray.size(), std::string_view{"struct "});
                auto size2 = removeAllOccurences(nameArray, size1, std::string_view{"class "});
#else
                auto size2 = nameArray.size();
#endif

                auto size3Func = [&](auto& nameArray) constexpr
                {
                    // remove spaces between closing template angle brackets and after commas
                    auto e = nameArray.begin() + size2;
                    for(auto b = nameArray.begin(); b < e - 2; b++)
                    {
                        if((b[0] == '>' && b[1] == ' ' && b[2] == '>') || (b[0] == ',' && b[1] == ' '))
                        {
                            constexprCopy(b + 2, e, b + 1);
                            e--;
                        }
                    }
                    return e - nameArray.begin();
                };
                auto size3 = size3Func(nameArray);

                return std::pair{nameArray, size3};
            }();

            return resizeArray<arrAndSize.second>(arrAndSize.first);
        }

        template<typename T>
        inline constexpr auto typeNameStorage = typeNameAsArray<T>();
    } // namespace internal

    template<typename T>
    inline constexpr auto qualifiedTypeName = []
    {
        constexpr auto& value = internal::typeNameStorage<T>;
        return std::string_view{value.data(), value.size()};
    }();

    namespace internal
    {
        constexpr auto isIdentChar(char c) -> bool
        {
            if(c >= 'A' && c <= 'Z')
                return true;
            if(c >= 'a' && c <= 'z')
                return true;
            if(c >= '0' && c <= '9')
                return true;
            if(c == '_')
                return true;
            return false;
        }

        template<typename T>
        inline constexpr auto structNameStorage = []() constexpr
        {
            // strip namespace qualifiers before type names
            constexpr auto arrAndSize = []() constexpr
            {
                auto s = internal::typeNameStorage<T>;
                auto b = s.begin();
                auto e = s.end();

#if defined(__clang__)
                constexpr auto anonNs = std::string_view{"(anonymous namespace)::"};
#elif defined(__NVCOMPILER)
                constexpr auto anonNs = std::string_view{"<unnamed>::"};
#elif defined(__GNUG__)
                constexpr auto anonNs = std::string_view{"{anonymous}::"};
#elif defined(_MSC_VER)
                constexpr auto anonNs = std::string_view{"`anonymous-namespace'::"};
#else
                constexpr auto anonNs = std::string_view{"@"}; // just anything we won't find
#endif
                std::size_t pos = 0;
                while((pos = std::string_view(b, e - b).find(anonNs)) != std::string::npos)
                {
                    constexprCopy(b + pos + anonNs.size(), e, b + pos);
                    e -= anonNs.size();
                }

                while(true)
                {
                    // find iterator to after "::"
                    auto l = b;
                    while(l + 1 < e && !(l[0] == ':' && l[1] == ':'))
                        l++;
                    if(l + 1 == e)
                        break;
                    l += 2;

                    // find iterator to first identifier char before "::"
                    auto f = l - 3; // start at first char before "::"
                    while(s.begin() < f && isIdentChar(f[-1]))
                        f--;

                    // cut out [f:l[
                    constexprCopy(l, e, f);
                    e -= (l - f);
                    b = f;
                }

                return std::pair{s, e - s.begin()};
            }();

            return resizeArray<arrAndSize.second>(arrAndSize.first);
        }();
    } // namespace internal

    template<typename T>
    constexpr auto structName(T = {}) -> std::string_view
    {
        constexpr auto& value = internal::structNameStorage<T>;
        return std::string_view{&value[0], value.size()};
    }

    namespace internal
    {
        constexpr auto intToStrSize(std::size_t s)
        {
            std::size_t len = 1;
            while(s >= 10)
            {
                len++;
                s /= 10;
            }
            return len;
        }

        template<typename RecordDim, std::size_t... Coords>
        LLAMA_ACC inline constexpr auto recordCoordTagsStorage = []() constexpr
        {
            using Tags = GetTags<RecordDim, RecordCoord<Coords...>>;

            // precompute char array size
            constexpr auto size = [&]() constexpr
            {
                std::size_t s = 0;
                mp_for_each<Tags>(
                    [&](auto tag)
                    {
                        using Tag = decltype(tag);
                        if constexpr(isRecordCoord<Tag>)
                        {
                            // handle array indices
                            static_assert(Tag::size == 1);
                            s += 2; // for the '[' and ']'
                            s += intToStrSize(Tag::front);
                        }
                        else
                        {
                            if(s != 0)
                                s++; // for the '.'s
                            s += structName(tag).size();
                        }
                    });
                return s;
            }();
            llama::Array<char, size> a{};
            auto it = a.begin();

            mp_for_each<Tags>(
                [&](auto tag) constexpr
                {
                    using Tag = decltype(tag);
                    if constexpr(isRecordCoord<Tag>)
                    {
                        auto n = Tag::front;
                        *it = '[';
                        it++;
                        it += intToStrSize(n);
                        auto it2 = it; // take copy because we write number backward
                        do // NOLINT(cppcoreguidelines-avoid-do-while)
                        {
                            it2--;
                            *it2 = '0' + n % 10;
                            n /= 10;
                        } while(n != 0);
                        *it = ']';
                        it++;
                    }
                    else
                    {
                        if(it != a.begin())
                        {
                            *it = '.';
                            it++;
                        }
                        constexpr auto sn = structName(tag);
                        constexprCopy(sn.begin(), sn.end(), it);
                        it += sn.size();
                    }
                });

            if(!a.empty() && a.back() == 0)
                throw std::logic_error{"Implementation error: Array should have been completely overwritten."};

            return a;
        }();
    } // namespace internal

    /// Returns a pretty representation of the record coordinate inside the given record dimension. Tags are
    /// interspersed by '.' and arrays are represented using subscript notation ("[123]").
    template<typename RecordDim, std::size_t... Coords>
    constexpr auto prettyRecordCoord(RecordCoord<Coords...> = {}) -> std::string_view
    {
        constexpr auto& value = internal::recordCoordTagsStorage<RecordDim, Coords...>;
        return std::string_view{value.data(), value.size()};
    }

    template<typename RecordDim>
    constexpr auto prettyRecordCoord(RecordCoord<>) -> std::string_view
    {
        return {};
    }
} // namespace llama
