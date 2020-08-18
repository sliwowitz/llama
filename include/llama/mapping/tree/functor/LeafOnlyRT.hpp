/* Copyright 2018 Alexander Matthes
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

#include "../TreeElement.hpp"
#include "../operations/GetNode.hpp"

namespace llama::mapping::tree::functor
{
    /// Functor for \ref tree::Mapping. Moves all run time parts to the leaves,
    /// so in fact another struct of array implementation -- but with the
    /// possibility to add further finetuning of the mapping in the future. \see
    /// tree::Mapping
    struct LeafOnlyRT
    {
        template<typename T_Tree, typename T_SFINAE = void>
        struct BasicToResultImpl
        {
            auto LLAMA_FN_HOST_ACC_INLINE
            operator()(T_Tree const tree, std::size_t const runtime = 1) const
                -> TreeElement<
                    typename T_Tree::Identifier,
                    typename T_Tree::Type>
            {
                return {LLAMA_DEREFERENCE(tree.count) * runtime};
            }
        };

        template<typename T_Tree>
        struct BasicToResultImpl<
            T_Tree,
            typename T_Tree::IsTreeElementWithChilds>
        {
            struct ChildFunctor
            {
                const std::size_t runtime;

                template<typename T_Element>
                LLAMA_FN_HOST_ACC_INLINE auto
                operator()(T_Element const element) const -> decltype(auto)
                {
                    return BasicToResultImpl<T_Element>()(element, runtime);
                }
            };

            LLAMA_FN_HOST_ACC_INLINE
            auto
            operator()(T_Tree const tree, std::size_t const runtime = 1) const
            {
                auto children = tupleTransform(
                    tree.childs,
                    ChildFunctor{runtime * LLAMA_DEREFERENCE(tree.count)});
                return TreeElementConst<
                    typename T_Tree::Identifier,
                    decltype(children),
                    1>{children};
            }
        };

        template<typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(T_Tree const tree) const
            -> decltype(auto)
        {
            return BasicToResultImpl<T_Tree>()(tree);
        }

        template<typename T_Tree>
        using Result = decltype(BasicToResultImpl<T_Tree>()(T_Tree()));

        template<typename T_Tree, typename T_BasicCoord>
        struct BasicCoordToResultCoordImpl
        {
            using FirstResultCoord = TreeCoordElementConst<
                decltype(T_BasicCoord::FirstElement::compiletime)::value,
                0>;
            using ResultCoord = TupleCatType<
                Tuple<FirstResultCoord>,
                decltype(BasicCoordToResultCoordImpl<
                         GetTupleType<
                             typename T_Tree::Type,
                             decltype(T_BasicCoord::FirstElement::compiletime)::
                                 value>,
                         typename T_BasicCoord::RestTuple>()(
                    typename T_BasicCoord::RestTuple(),
                    getTupleElementRef<T_BasicCoord::FirstElement::compiletime>(
                        T_Tree().childs),
                    0))>;

            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(
                T_BasicCoord const & basicCoord,
                T_Tree const & tree,
                std::size_t const runtime = 0) const -> ResultCoord
            {
                auto const & branch = getTupleElementRef<
                    T_BasicCoord::FirstElement::compiletime>(tree.childs);
                return ResultCoord(
                    FirstResultCoord(),
                    BasicCoordToResultCoordImpl<
                        GetTupleType<
                            typename T_Tree::Type,
                            decltype(T_BasicCoord::FirstElement::compiletime)::
                                value>,
                        typename T_BasicCoord::RestTuple>()(
                        basicCoord.rest,
                        branch,
                        (runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))
                            * LLAMA_DEREFERENCE(branch.count)));
            }
        };

        template<typename T_Tree, typename T_LastCoord>
        struct BasicCoordToResultCoordImpl<T_Tree, Tuple<T_LastCoord>>
        {
            using BasicCoord = Tuple<T_LastCoord>;
            using ResultCoordElement
                = TreeCoordElement<decltype(T_LastCoord::compiletime)::value>;

            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(
                BasicCoord const & basicCoord,
                T_Tree const & tree,
                std::size_t const runtime = 0) const
                -> Tuple<ResultCoordElement>
            {
                return {ResultCoordElement(
                    runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))};
            }
        };

        template<typename T_Tree, typename T_BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            T_BasicCoord const & basicCoord,
            T_Tree const & tree) const -> decltype(auto)
        {
            return BasicCoordToResultCoordImpl<T_Tree, T_BasicCoord>()(
                basicCoord, tree);
        }

        template<typename T_Tree, typename T_ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            T_ResultCoord const & resultCoord,
            T_Tree const & tree) const -> T_ResultCoord
        {
            return resultCoord;
        }
    };
}
