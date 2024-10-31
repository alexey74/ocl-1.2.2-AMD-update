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
## @deftypefn{Function File} {@var{ocl_prog} =} ocl_program_file (@var{fname})
## @deftypefnx{Function File} {@var{ocl_prog} =} ocl_program_file (@var{fname}, @var{build_opts_str})
##
## Construct and compile an OCL program from an OpenCL C source code file.
##
## @code{ocl_program_file} loads an OpenCL C source code file with filename
## @var{fname} (which is passed to @code{fopen}) and compiles this code
## using the OpenCL online compiler (via the function @code{ocl_program}).
## If given, the build options specified in the string @var{build_opts_str} are
## applied during compilation.  Upon success, an OCL program @var{ocl_prog} is returned.
##
## For all OCL program properties, see @code{ocl_program}.
##
## @seealso{oclArray, ocl_program}
## @end deftypefn

function ocl_prog = ocl_program_file (fname, build_opts_str)

  if ((nargin < 1) || (! ischar (fname)))
    error ("First argument must be a string (the OpenCL program filename).");
  elseif ((nargin > 1) && (! ischar (build_opts_str)))
    error ("Second argument must be a string (the build options), if given.");
  elseif (nargin > 2)
    error ("Too many arguments.");
  end

  if nargin < 2, build_opts_str = ""; end

  [fid, msg] = fopen (fname, "r");
  if (fid < 0)
    error (["Unable to open file '" fname(:)' "': " msg]);
  end

  [program_src, count] = fread (fid, Inf, "*char");
  fclose (fid);

  ocl_prog = ocl_program (program_src(:)', build_opts_str);

endfunction
