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

function varargout = atan2 (varargin)

varargout = cell (1, max (1, nargout));
[varargout{:}] = __ocl_mat_atan2__ (varargin{:});

endfunction
