/*
 * Copyright (C) 2019-2023 Matthias W. Klein
 *
 * This file is part of OCL - a GNU Octave package providing OpenCL support.
 *
 * OCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OCL.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __OCL_OCTAVE_VERSIONS_H
#define __OCL_OCTAVE_VERSIONS_H

#include <octave/oct.h>

// definition of the following macros is provided for octave versions >= 4.0.0
#define OCL_OCTAVE_VERSION ((OCTAVE_MAJOR_VERSION*100+OCTAVE_MINOR_VERSION)*100+OCTAVE_PATCH_VERSION)

#if OCL_OCTAVE_VERSION >= 40400
#define OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER
#endif

#if OCL_OCTAVE_VERSION >= 50100
#define OCL_OCTAVE_VERSION_5_1_0_AND_HIGHER
#endif

#if OCL_OCTAVE_VERSION >= 50200
#define OCL_OCTAVE_VERSION_5_2_0_AND_HIGHER
#endif

#if OCL_OCTAVE_VERSION >= 60100
#define OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER
#endif

#if OCL_OCTAVE_VERSION >= 60200
#define OCL_OCTAVE_VERSION_6_2_0_AND_HIGHER
#endif

#endif  /* __OCL_OCTAVE_VERSIONS_H */
