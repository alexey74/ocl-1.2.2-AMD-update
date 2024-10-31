## Copyright (C) 2019-2023 Matthias W. Klein
##
## This file is part of OCL - a GNU Octave package providing OpenCL support.
##
## OCL is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## OCL is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with OCL.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
##
## @deftypefn{Function File} {@var{ocl_mat} =} oclArray (@var{octave_mat})
##
## Construct an OCL matrix from an octave matrix, preserving the elements'
## data type.
##
## @code{oclArray} takes as input a conventional numeric octave matrix
## @var{octave_mat} (actually an N-dimensional array) of any numeric data type
## (float, int32, etc.).
## The function creates a new OCL matrix @var{ocl_mat} (as an N-dimensional
## array) of data type corresponding to @var{octave_mat}, by calling
## the corresponding OCL matrix constructor function (e.g., @code{ocl_double}).
## In effect, the function allocates storage space on the OpenCL
## device hardware and copies the octave matrix data into the OpenCL
## device memory.  The data then remains in device memory until the OCL
## matrix is cleared from the octave workspace (or as long as the
## OpenCL context exists).
##
## The reverse operation can be performed with the @code{ocl_to_octave}
## function.
##
## Example:
##
## @example
## @group
## mat = magic (4);
##
## # transfer mat to OpenCL memory
## ocl_mat = oclArray (mat);
##
## # perform computations with ocl_mat on OpenCL device
## ocl_mat2 = ocl_mat + 1;
## ocl_mat3 = ocl_mat2 .^ 3;
## ocl_mat3(:,3) = 5;
## ocl_mat2 = mean (floor (ocl_mat3 / 2), 2);
##
## # transfer ocl_mat2 to octave memory
## mat2 = ocl_to_octave (ocl_mat2);
##
## disp (mat2)
## @end group
## @end example
##
## For compatibility with MATLAB, the @code{gpuArray} function is an alias
## to @code{oclArray}.
##
## Two kinds of operation are possible with OCL matrices to perform numeric
## computations:
##
## First, many (but not all) built-in operations known from octave matrices are possible
## (e.g., multiplication by @code{*}, indexing by ranges; standard functions like
## @code{reshape}, @code{repmat}, @code{ndgrid}; numeric functions like @code{cos},
## @code{sumsq}; searching functions like @code{max} and OCL's special @code{findfirst} /
## @code{findlast}).  All of these operations are performed via small OCL-internal
## OpenCL C subprograms (kernels) which are restricted to the SIMD principle (Single
## Instruction Multiple Data).  Because of this, there are various restrictions on
## built-in operations with OCL matrices (e.g., indexing by ranges must result in data
## which is contiguous in OpenCL memory; no broadcasting).  In particular, math functions
## which are expected to give complex-valued results require complex input matrices.
## See the ocl_tests.m file for details of the implemented functionality.
##
## Using the built-in operations on OCL matrices will help in the transition
## from a CPU-based computation to an OpenCL-based computation, since little
## octave code needs changes (mostly the beginning and final parts).  However, be
## aware that the internal effort of both octave and the OpenCL driver for
## handling the built-in operations may cause a significant overhead.
## Also be aware that all OCL matrix operations are computed asynchronously,
## and that any intermediate copying of data to or from the OpenCL device
## interrupts and potentially delays this asynchronous workflow.
##
## Keep in mind that OCL data virtually lives in another world, in space and time;
## space, because it is generally stored in a memory which is physically separate
## from octave CPU memory;  time, because the data resulting from an operation
## will generally exist not directly after the scheduling octave command returns,
## but only later, due to the asynchronous workflow.
##
## The second kind of operation is to use an OCL matrix as an argument when
## calling a user-written OpenCL C program (i.e., calling a kernel in an OCL
## program, see @code{ocl_program}).  User-written OpenCL C programs make
## the OCL functionality easily extendible.
##
## @code{oclArray} automatically assures that the OpenCL library is
## loaded (see @code{ocl_lib}) and that an OpenCL context is created with an
## OpenCL device (see @code{ocl_context}).
##
## @seealso{ocl_to_octave, gpuArray, gather,
## ocl_tests, ocl_program, ocl_context, ocl_lib, ocl_constant,
## ocl_ones, ocl_zeros, ocl_eye, ocl_cat, ocl_linspace, ocl_logspace,
## ocl_double, ocl_single,
## ocl_int8, ocl_int16, ocl_int32, ocl_int64,
## ocl_uint8, ocl_uint16, ocl_uint32, ocl_uint64}
## @end deftypefn

function ret = oclArray (octave_array)

  if ((nargin != 1) || (! isnumeric (octave_array)))
    error ("single argument must be a numeric octave object");
  endif

  switch typeinfo (octave_array)
    case {"matrix", "scalar", "range", "double_range", "complex matrix", "complex scalar"}
      ret = ocl_double (full (octave_array));
    case {"float matrix", "float scalar", "float complex matrix", "float complex scalar"}
      ret = ocl_single (octave_array);
    case {"int8 matrix", "int8 scalar"}
      ret = ocl_int8 (octave_array);
    case {"int16 matrix", "int16 scalar"}
      ret = ocl_int16 (octave_array);
    case {"int32 matrix", "int32 scalar"}
      ret = ocl_int32 (octave_array);
    case {"int64 matrix", "int64 scalar"}
      ret = ocl_int64 (octave_array);
    case {"uint8 matrix", "uint8 scalar"}
      ret = ocl_uint8 (octave_array);
    case {"uint16 matrix", "uint16 scalar"}
      ret = ocl_uint16 (octave_array);
    case {"uint32 matrix", "uint32 scalar"}
      ret = ocl_uint32 (octave_array);
    case {"uint64 matrix", "uint64 scalar"}
      ret = ocl_uint64 (octave_array);
    otherwise
      error ("invalid argument type");
  endswitch

endfunction

%!assert (typeinfo (oclArray (0)), "ocl matrix");
%!assert (typeinfo (oclArray ([0 0])), "ocl matrix");
%!assert (typeinfo (oclArray (0:2)), "ocl matrix");
%!assert (typeinfo (oclArray (single (0))), "ocl float matrix");
%!assert (typeinfo (oclArray (single ([0 0]))), "ocl float matrix");
%!assert (typeinfo (oclArray (i)), "ocl complex matrix");
%!assert (typeinfo (oclArray ([0 i])), "ocl complex matrix");
%!assert (typeinfo (oclArray (single (i))), "ocl float complex matrix");
%!assert (typeinfo (oclArray (single ([0 i]))), "ocl float complex matrix");
%!assert (typeinfo (oclArray (int8 (0))), "ocl int8 matrix");
%!assert (typeinfo (oclArray (int8 ([0 0]))), "ocl int8 matrix");
%!assert (typeinfo (oclArray (int16 (0))), "ocl int16 matrix");
%!assert (typeinfo (oclArray (int16 ([0 0]))), "ocl int16 matrix");
%!assert (typeinfo (oclArray (int32 (0))), "ocl int32 matrix");
%!assert (typeinfo (oclArray (int32 ([0 0]))), "ocl int32 matrix");
%!assert (typeinfo (oclArray (int64 (0))), "ocl int64 matrix");
%!assert (typeinfo (oclArray (int64 ([0 0]))), "ocl int64 matrix");
%!assert (typeinfo (oclArray (uint8 (0))), "ocl uint8 matrix");
%!assert (typeinfo (oclArray (uint8 ([0 0]))), "ocl uint8 matrix");
%!assert (typeinfo (oclArray (uint16 (0))), "ocl uint16 matrix");
%!assert (typeinfo (oclArray (uint16 ([0 0]))), "ocl uint16 matrix");
%!assert (typeinfo (oclArray (uint32 (0))), "ocl uint32 matrix");
%!assert (typeinfo (oclArray (uint32 ([0 0]))), "ocl uint32 matrix");
%!assert (typeinfo (oclArray (uint64 (0))), "ocl uint64 matrix");
%!assert (typeinfo (oclArray (uint64 ([0 0]))), "ocl uint64 matrix");
