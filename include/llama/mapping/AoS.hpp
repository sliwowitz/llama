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

#include "../Types.hpp"
#include "../UserDomain.hpp"

namespace llama::mapping
{
    /** Array of struct mapping which can be used for creating a \ref View with
     * a \ref Factory. For the interface details see \ref Factory. \tparam
     * T_UserDomain type of the user domain \tparam T_DatumDomain type of the
     * datum domain \tparam LinearizeUserDomainAdressFunctor Defines how the
     * user domain should be linearized, e.g. C like with the last dimension
     * being the "fast" one
     *  (\ref LinearizeUserDomainAdress, default) or Fortran like with the first
     *  dimension being the "fast" one (\ref
     * LinearizeUserDomainAdressLikeFortran). \tparam
     * ExtentUserDomainAdressFunctor Defines how the size of the view shall be
     * created. Should fit for `T_LinearizeUserDomainAdressFunctor`. Only right
     * now implemented and default value is \ref ExtentUserDomainAdress. \see
     * SoA
     */
    template<
        typename T_UserDomain,
        typename T_DatumDomain,
        typename LinearizeUserDomainAdressFunctor = LinearizeUserDomainAdress,
        typename ExtentUserDomainAdressFunctor = ExtentUserDomainAdress>
    struct AoS
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        AoS(UserDomain size) : userDomainSize(size) {}

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const
            -> std::size_t
        {
            return ExtentUserDomainAdressFunctor{}(
                userDomainSize)*SizeOf<DatumDomain>;
        }

        template<std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain coord) const
            -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto offset
                = LinearizeUserDomainAdressFunctor{}(coord, userDomainSize)
                    * SizeOf<DatumDomain>
                + linearBytePos<DatumDomain, DatumDomainCoord...>();
            return {0, offset};
        }

        UserDomain userDomainSize;
    };
}
