## Copyright (C) 2022-2023 Matthias W. Klein
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
##
##
## The content of this file is mainly a copy of the file of the same name,
## originally published with GNU Octave 4.2.2, distributed under the same
## license (as OCL, see above), with the following Copyright notices:
## Copyright (C) 2006-2017 David Bateman

## -*- texinfo -*-
## @deftypefn {} {} sind (@var{x})
## Compute the sine for each element of @var{x} in degrees.
##
## Returns zero for elements where @code{@var{x}/180} is an integer.
## @seealso{asind, sin}
## @end deftypefn

## Author: David Bateman <dbateman@free.fr>

function y = sind (x)

  if (nargin != 1)
    print_usage ();
  endif

  I = x / 180;
  y = sin (I .* pi);
  y(I == fix (I) & isfinite (I)) = 0;

endfunction
