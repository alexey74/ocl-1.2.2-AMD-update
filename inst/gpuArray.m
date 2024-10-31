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
## @deftypefn{Function File} {@var{ocl_mat} =} gpuArray (@var{octave_mat})
##
## Construct an OCL matrix from an octave matrix, preserving the elements'
## data type.
##
## For compatibility with MATLAB, the @code{gpuArray} function is an alias
## to @code{oclArray}.
##
## @seealso{oclArray, ocl_to_octave, gather}
## @end deftypefn

function ret = gpuArray (octave_array)
  if ! isgpuarray(octave_array)
    ret = oclArray (octave_array);
  else
    ret = octave_array;
  endif
endfunction

%!assert (test ("oclArray"));
%!assert (gather(gpuArray(gpuArray([1 3 4]))), gather(gpuArray([1 3 4])))
