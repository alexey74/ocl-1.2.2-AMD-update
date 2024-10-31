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
## @deftypefn{Function File} {@var{octave_mat} =} gather (@var{ocl_mat})
##
## Transfer the data from an OCL matrix into an octave matrix of corresponding type.
##
## For compatibility with MATLAB, the @code{gather} function is an alias
## to @code{ocl_to_octave}.
##
## @seealso{oclArray, ocl_to_octave, gpuArray}
## @end deftypefn

function ret = gather (ocl_array)

  ret = ocl_to_octave (ocl_array);

endfunction

%!assert (test ("ocl_to_octave"));
