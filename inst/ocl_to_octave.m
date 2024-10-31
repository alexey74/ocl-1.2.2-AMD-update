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
## @deftypefn{Function File} {@var{octave_mat} =} ocl_to_octave (@var{ocl_mat})
##
## Transfer the data from an OCL matrix into an octave matrix of corresponding type.
##
## @code{ocl_to_octave} takes as input any OCL matrix @var{ocl_mat}.  The OCL
## data contained is copied into a conventional octave matrix @var{octave_mat},
## preserving the underlying data type (e.g., @code{int32}).  This data
## transfer thus copies the data from the OpenCL device to conventional memory.
## It is an operation which destroys the asynchronous workflow of OCL
## calculations and should be avoided during time-critical tasks.
## It is suitable for the inspection of the results of OCL calculations.
##
## The reverse operation can be performed with the @code{oclArray}
## function (see example there).
##
## For an explanation of OCL matrices, see @code{ocl_double}.
##
## For compatibility with MATLAB, the @code{gather} function is an alias
## to @code{ocl_to_octave}.
##
## @seealso{oclArray, ocl_double, ocl_single,
## ocl_int8, ocl_int16, ocl_int32, ocl_int64,
## ocl_uint8, ocl_uint16, ocl_uint32, ocl_uint64,
## gpuArray, gather}
## @end deftypefn

function ret = ocl_to_octave (ocl_array)

  if (nargin != 1)
    error ("single argument must be an OCL matrix");
  endif

  switch typeinfo (ocl_array)
    case {"ocl matrix", "ocl complex matrix"}
      ret = double (ocl_array);
    case {"ocl float matrix", "ocl float complex matrix"}
      ret = single (ocl_array);
    case {"ocl int8 matrix"}
      ret = int8 (ocl_array);
    case {"ocl int16 matrix"}
      ret = int16 (ocl_array);
    case {"ocl int32 matrix"}
      ret = int32 (ocl_array);
    case {"ocl int64 matrix"}
      ret = int64 (ocl_array);
    case {"ocl uint8 matrix"}
      ret = uint8 (ocl_array);
    case {"ocl uint16 matrix"}
      ret = uint16 (ocl_array);
    case {"ocl uint32 matrix"}
      ret = uint32 (ocl_array);
    case {"ocl uint64 matrix"}
      ret = uint64 (ocl_array);
    otherwise
      error ("invalid argument type");
  endswitch

endfunction

%!assert (typeinfo (ocl_to_octave (ocl_double ([0 0]))), "matrix");
%!assert (typeinfo (ocl_to_octave (ocl_single ([0 0]))), "float matrix");
%!assert (typeinfo (ocl_to_octave (ocl_double ([0 i]))), "complex matrix");
%!assert (typeinfo (ocl_to_octave (ocl_single ([0 i]))), "float complex matrix");
%!assert (typeinfo (ocl_to_octave (ocl_int8 ([0 0]))), "int8 matrix");
%!assert (typeinfo (ocl_to_octave (ocl_int16 ([0 0]))), "int16 matrix");
%!assert (typeinfo (ocl_to_octave (ocl_int32 ([0 0]))), "int32 matrix");
%!assert (typeinfo (ocl_to_octave (ocl_int64 ([0 0]))), "int64 matrix");
%!assert (typeinfo (ocl_to_octave (ocl_uint8 ([0 0]))), "uint8 matrix");
%!assert (typeinfo (ocl_to_octave (ocl_uint16 ([0 0]))), "uint16 matrix");
%!assert (typeinfo (ocl_to_octave (ocl_uint32 ([0 0]))), "uint32 matrix");
%!assert (typeinfo (ocl_to_octave (ocl_uint64 ([0 0]))), "uint64 matrix");
