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
## @deftypefn{Function File} {@var{RET}} = ocl_tests ()
##
## Perform tests of OCL functions and with all OCL data types.
##
## If you have more than one OpenCL device available, you can use
## @code{ocl_context} to select the device to perform tests with
## prior to calling @code{ocl_tests}.
##
## @seealso{oclArray, ocl_context}
## @end deftypefn

function ret = ocl_tests ()

ret = 0;

## --------- ocl_constant tests ---------

disp (["Testing ocl_constant function..."]); fflush (stdout);

assert (class (help ("ocl_constant")), "char") # test for help string

assert (ocl_constant ("CL_DEVICE_TYPE_GPU"), 4)
assert (ocl_constant (-5), "CL_OUT_OF_RESOURCES")


## --------- ocl_lib tests ---------

disp (["Testing ocl_lib function..."]); fflush (stdout);

assert (class (help ("ocl_lib")), "char") # test for help string

loaded = ocl_lib ("loaded");
assert (class (loaded), "double")

ocl_lib ("unload");
assert (ocl_lib ("loaded"), 0)

ocl_lib ("assure");
assert (ocl_lib ("loaded"), 1)

ocl_lib ("unload");
assert (ocl_lib ("loaded"), 0)

[oldpath, oldfname] = ocl_lib ("lib_path_filename");
assert (class (oldpath), "char")
assert (class (oldfname), "char")


## --------- ocl_context tests ---------

disp (["Testing ocl_context function..."]); fflush (stdout);

assert (class (help ("ocl_context")), "char") # test for help string

[active, fp64] = ocl_context ("active");
assert (class (active), "double")
assert (class (fp64), "double")

[activeID, fp64] = ocl_context ("active_id");
assert (class (activeID), "double")
assert (class (fp64), "double")

selection = ocl_context ("device_selection");
assert (class (selection), "char")

resources = ocl_context ("get_resources");
assert (class (resources), "struct")
assert (fieldnames (resources), { "platforms"; "devices"; "summary"})

selection = ocl_context ("device_selection", "selected");
assert (class (selection), "double")
assert (size (selection), [2 1])

ocl_lib ("unload");
assert (ocl_context ("active"), 0)

ocl_context ("assure");
assert (ocl_context ("active"), 1)
[~, fp64] = ocl_context ("active");

ocl_context ("destroy");
assert (ocl_context ("active"), 0)

ocl_context ("assure");
assert (ocl_context ("active"), 1)

ocl_lib ("unload");
assert (ocl_context ("active"), 0)


## --------- ocl matrix data type tests ---------

for ocltype = 1:10

switch ocltype
  case 1,  to_octave_type = @(a) double (a); to_ocl_type = @(a) ocl_double (full (a)); typestr = "ocl_double";
  case 2,  to_octave_type = @(a) single (a); to_ocl_type = @(a) ocl_single (full (a)); typestr = "ocl_single";
  case 3,  to_octave_type = @(a) int8 (a);   to_ocl_type = @(a) ocl_int8 (full (a));   typestr = "ocl_int8";
  case 4,  to_octave_type = @(a) int16 (a);  to_ocl_type = @(a) ocl_int16 (full (a));  typestr = "ocl_int16";
  case 5,  to_octave_type = @(a) int32 (a);  to_ocl_type = @(a) ocl_int32 (full (a));  typestr = "ocl_int32";
  case 6,  to_octave_type = @(a) int64 (a);  to_ocl_type = @(a) ocl_int64 (full (a));  typestr = "ocl_int64";
  case 7,  to_octave_type = @(a) uint8 (a);  to_ocl_type = @(a) ocl_uint8 (full (a));  typestr = "ocl_uint8";
  case 8,  to_octave_type = @(a) uint16 (a); to_ocl_type = @(a) ocl_uint16 (full (a)); typestr = "ocl_uint16";
  case 9,  to_octave_type = @(a) uint32 (a); to_ocl_type = @(a) ocl_uint32 (full (a)); typestr = "ocl_uint32";
  case 10, to_octave_type = @(a) uint64 (a); to_ocl_type = @(a) ocl_uint64 (full (a)); typestr = "ocl_uint64";
  otherwise, error ("unknown data type");
endswitch
if ocltype <= 2, typefloat = 1; else typefloat = 0; endif
if ocltype >= 7, typeuint = 1; else typeuint = 0; endif

disp (["Testing ocl matrix of data type " strrep(typestr," ","_") "..."]); fflush (stdout);

if (ocltype == 1) && (fp64 == 0)
disp (["  ...skipped because the current OpenCL context does not support it"]); fflush (stdout);
continue;
endif

## test for constructor help string
assert (class (help (strrep(typestr," ","_"))), "char")

## --------- constructor tests ---------

for complex_iter = 0:typefloat

if complex_iter > 0, j = sqrt (-1); else j = 0; endif

a = to_ocl_type ((1:36) + j);

## test for variable printing to terminal
assert (class (disp (a)), "char")

## test for class string
assert (class (a), typestr)

## test for back conversion and class
assert (class (to_octave_type (a)), typestr(5:end))

## test as_index conversion
assert (class (as_index (a)), "ocl_int64")

endfor # complex_iter

for complex_iter = 0:typefloat

if complex_iter > 0, j = sqrt (-1); else j = 0; endif

a = to_ocl_type (zeros ([4 5 6]) + j);

## dimension tests
assert (size (a), [4 5 6])
assert (ndims (a), 3)
assert (rows (a), 4)
assert (columns (a), 5)
assert (length (a), 6)
assert (numel (a), 4*5*6)

## property checks
assert (! isempty (a))
assert (! isnumeric (a))   # !!!!! probably neccessary to distinguish ocl matrix from standard matrix
assert (! isreal (a))   # !!!!! probably neccessary to distinguish ocl matrix from standard matrix
assert (iscomplex (a), logical (complex_iter))
assert (isfloat (a), logical (typefloat))
assert (! isinteger (a), logical (typefloat))
assert (! isscalar (a))
assert (! ismatrix (a))   # !!!!! probably neccessary to distinguish ocl matrix from standard matrix
assert (iscolumn (to_ocl_type (zeros ([4 1]))))
assert (isrow (to_ocl_type (zeros ([1 5]))))
assert (isvector (to_ocl_type (zeros ([4 1]))))
assert (isvector (to_ocl_type (zeros ([1 5]))))
assert (! isvector (to_ocl_type (zeros ([4 5]))))
assert (! islogical (a))
assert (islogical (a < 18))

## other constructors
assert (to_octave_type (ocl_cat (1, a, a, a)), repmat (to_octave_type (a), [3 1 1]))
assert (to_octave_type (ocl_cat (2, a, a, a)), repmat (to_octave_type (a), [1 3 1]))
assert (to_octave_type (ocl_cat (3, a, a, a)), repmat (to_octave_type (a), [1 1 3]))

endfor # complex_iter

assert (to_octave_type (ocl_ones (4, typestr(5:end))), ones (4, typestr(5:end)))
assert (to_octave_type (ocl_ones (4, 5, 6, typestr(5:end))), ones (4, 5, 6, typestr(5:end)))
assert (to_octave_type (ocl_ones ([4 5 6], typestr(5:end))), ones ([4 5 6], typestr(5:end)))
assert (to_octave_type (ocl_zeros (4, typestr(5:end))), zeros (4, typestr(5:end)))
assert (to_octave_type (ocl_zeros (4, 5, 6, typestr(5:end))), zeros (4, 5, 6, typestr(5:end)))
assert (to_octave_type (ocl_zeros ([4 5 6], typestr(5:end))), zeros ([4 5 6], typestr(5:end)))
assert (to_octave_type (ocl_eye ([4 5], typestr(5:end))), eye ([4 5], typestr(5:end)))
assert (to_octave_type (ocl_eye (4, typestr(5:end))), eye (4, typestr(5:end)))
if ocltype == 1 # ocl double
assert (to_octave_type (ocl_linspace (0, 1, 6)), linspace (0, 1, 6), 1e-14)
assert (to_octave_type (ocl_logspace (0, 1, 6)), logspace (0, 1, 6), 1e-14)
endif

if typefloat
a = to_ocl_type (zeros ([4 5]));
b = to_ocl_type (ones ([4 5]));
assert (iscomplex (complex (a)))
assert (iscomplex (complex (a, b)))
assert (real (to_octave_type (complex (a))), real (complex (to_octave_type (a))))
assert (to_octave_type (complex (a, b)), to_octave_type (complex (to_octave_type (a), to_octave_type (b))))
endif

assert (class (help ("ocl_cat")), "char") # test for help string
assert (class (help ("ocl_ones")), "char") # test for help string
assert (class (help ("ocl_zeros")), "char") # test for help string
assert (class (help ("ocl_eye")), "char") # test for help string
assert (class (help ("ocl_linspace")), "char") # test for help string
assert (class (help ("ocl_logspace")), "char") # test for help string

## --------- utility function tests ---------

for complex_iter = 0:typefloat

if complex_iter > 0, j = sqrt (-1); else j = 0; endif

a = to_ocl_type (zeros ([4 5 6]) + j);
b = reshape (a, [6 5 1 4]);
assert (size (b), [6 5 1 4])

assert (size (squeeze (b)), [6 5 4])

assert (to_octave_type (repmat (a, [3 1 1])), repmat (to_octave_type (a), [3 1 1]))
assert (to_octave_type (repmat (a, [1 3 1])), repmat (to_octave_type (a), [1 3 1]))
assert (to_octave_type (repmat (a, [1 1 3])), repmat (to_octave_type (a), [1 1 3]))

c = to_ocl_type ((1:2:5) + j);
[c1, c2] = ndgrid (c);
[co1, co2] = ndgrid (to_octave_type (c));
assert (to_octave_type (c1), co1)
assert (to_octave_type (c2), co2)
d = to_ocl_type ((4:2:8) + j);
e = to_ocl_type ((0:3:9) + j);
[c1, d1, e1] = ndgrid (c, d, e);
[co1, do1, eo1] = ndgrid (to_octave_type (c), to_octave_type (d), to_octave_type (e));
assert (to_octave_type (c1), co1)
assert (to_octave_type (d1), do1)
assert (to_octave_type (e1), eo1)

c = to_ocl_type ((1:2:5) + j);
[c1, c2] = meshgrid (c);
[co1, co2] = meshgrid (to_octave_type (c));
assert (to_octave_type (c1), co1)
assert (to_octave_type (c2), co2)
d = to_ocl_type ((4:2:8) + j);
e = to_ocl_type ((0:3:9) + j);
[c1, d1, e1] = meshgrid (c, d, e);
[co1, do1, eo1] = meshgrid (to_octave_type (c), to_octave_type (d), to_octave_type (e));
assert (to_octave_type (c1), co1)
assert (to_octave_type (d1), do1)
assert (to_octave_type (e1), eo1)

clear a b c d e c1 c2 co1 co2 c1 d1 e1 co1 do1 eo1

endfor # complex_iter

## --------- indexing tests ---------

for complex_iter = 0:typefloat

if complex_iter > 0, j = sqrt (-1); else j = 0; endif

## perform a first slice indexing here, so indexing tests below also test for double slice indexing
a = reshape (to_ocl_type ((1:36) + j),3,3,4) (:,:,2:end);
b = reshape (to_ocl_type ((1:30) + j),3,10) (:,2:end);
r = to_ocl_type ((1:27) + j) (3:end-1);
c = r';
i = to_ocl_type ([5 15 4 12]) (:,2:end);

assert (to_octave_type (a(:)), to_octave_type (a)(:))
assert (to_octave_type (a(12:23)), to_octave_type (a)(12:23))
assert (to_octave_type (a(23)), to_octave_type (a)(23))
assert (to_octave_type (a(i)), to_octave_type (a)(to_octave_type (i)))

assert (to_octave_type (a(:,:)), to_octave_type (a)(:,:))
assert (to_octave_type (a(:,2:7)), to_octave_type (a)(:,2:7))
assert (to_octave_type (a(:,7)), to_octave_type (a)(:,7))
assert (to_octave_type (a(2:3,3)), to_octave_type (a)(2:3,3))
assert (to_octave_type (a(2,3)), to_octave_type (a)(2,3))

assert (to_octave_type (a(:,:,:)), to_octave_type (a)(:,:,:))
assert (to_octave_type (a(:,:,2:3)), to_octave_type (a)(:,:,2:3))
assert (to_octave_type (a(:,:,3)), to_octave_type (a)(:,:,3))
assert (to_octave_type (a(:,2:3,3)), to_octave_type (a)(:,2:3,3))
assert (to_octave_type (a(:,2,3)), to_octave_type (a)(:,2,3))
assert (to_octave_type (a(2:3,2,2)), to_octave_type (a)(2:3,2,2))
assert (to_octave_type (a(2,2,2)), to_octave_type (a)(2,2,2))

assert (to_octave_type (b(:)), to_octave_type (b)(:))
assert (to_octave_type (b(12:23)), to_octave_type (b)(12:23))
assert (to_octave_type (b(23)), to_octave_type (b)(23))
assert (to_octave_type (b(i)), to_octave_type (b)(to_octave_type (i)))

assert (to_octave_type (b(:,:)), to_octave_type (b)(:,:))
assert (to_octave_type (b(:,2:7)), to_octave_type (b)(:,2:7))
assert (to_octave_type (b(:,7)), to_octave_type (b)(:,7))
assert (to_octave_type (b(2:3,3)), to_octave_type (b)(2:3,3))
assert (to_octave_type (b(2,3)), to_octave_type (b)(2,3))

assert (to_octave_type (r(:)), to_octave_type (r)(:))
assert (to_octave_type (r(12:23)), to_octave_type (r)(12:23))
assert (to_octave_type (r(23)), to_octave_type (r)(23))
assert (to_octave_type (r(i)), to_octave_type (r)(to_octave_type (i)))

assert (to_octave_type (c(:)), to_octave_type (c)(:))
assert (to_octave_type (c(12:23)), to_octave_type (c)(12:23))
assert (to_octave_type (c(23)), to_octave_type (c)(23))
assert (to_octave_type (c(i)), to_octave_type (c)(to_octave_type (i)))

## --------- indexed assignment tests ---------

a2 = to_octave_type (a);
b2 = to_octave_type (b);
r2 = to_octave_type (r);
c2 = to_octave_type (c);
i2 = to_octave_type (i);

a0 = a; a20 = a2; b0 = b; b20 = b2; r0 = r; r20 = r2; c0 = c; c20 = c2;

a = a0; a2 = a20;

a(23) = a(1); a2(23) = a2(1);
assert (to_octave_type (a), a2)
a(23) = to_octave_type (0); a2(23) = 0;
assert (to_octave_type (a), a2)
a(i) = a(i); a2(i2) = a2(i2);
assert (to_octave_type (a), a2)
a(i) = to_octave_type (0); a2(i2) = 0;
assert (to_octave_type (a), a2)
a(12:23) = a(2:13); a2(12:23) = a2(2:13);
assert (to_octave_type (a), a2)
a(12:23) = to_octave_type (0); a2(12:23) = 0;
assert (to_octave_type (a), a2)
a(:) = a(:);
assert (to_octave_type (a), a2)
a(:) = to_octave_type (j); a2(:) = j;
assert (to_octave_type (a), a2)

a = a0; a2 = a20;

a(2,3) = a(1,1); a2(2,3) = a2(1,1);
assert (to_octave_type (a), a2)
a(2,3) = to_octave_type (0); a2(2,3) = 0;
assert (to_octave_type (a), a2)
a(2:3,3) = a(2:3,2); a2(2:3,3) = a2(2:3,2);
assert (to_octave_type (a), a2)
a(2:3,3) = to_octave_type (0); a2(2:3,3) = 0;
assert (to_octave_type (a), a2)
a(:,7) = a(:,7);
assert (to_octave_type (a), a2)
a(:,7) = to_octave_type (0); a2(:,7) = 0;
assert (to_octave_type (a), a2)
a(:,2:7) = a(:,2:7);
assert (to_octave_type (a), a2)
a(:,2:7) = to_octave_type (0); a2(:,2:7) = 0;
assert (to_octave_type (a), a2)
a(:,:) = a(:,:);
assert (to_octave_type (a), a2)
a(:,:) = to_octave_type (j); a2(:,:) = j;
assert (to_octave_type (a), a2)

a = a0; a2 = a20;

a(2,2,2) = a(1,1,1); a2(2,2,2) = a2(1,1,1);
assert (to_octave_type (a), a2)
a(2,2,2) = to_octave_type (0); a2(2,2,2) = 0;
assert (to_octave_type (a), a2)
a(2:3,2,3) = a(2:3,2,2); a2(2:3,2,3) = a2(2:3,2,2);
assert (to_octave_type (a), a2)
a(2:3,2,3) = to_octave_type (0); a2(2:3,2,3) = 0;
assert (to_octave_type (a), a2)
a(:,2,3) = a(:,2,3);
assert (to_octave_type (a), a2)
a(:,2,3) = to_octave_type (0); a2(:,2,3) = 0;
assert (to_octave_type (a), a2)
a(:,2:3,2) = a(:,2:3,2);
assert (to_octave_type (a), a2)
a(:,2:3,2) = to_octave_type (0); a2(:,2:3,2) = 0;
assert (to_octave_type (a), a2)
a(:,:,2) = a(:,:,2);
assert (to_octave_type (a), a2)
a(:,:,2) = to_octave_type (0); a2(:,:,2) = 0;
assert (to_octave_type (a), a2)
a(:,:,2:3) = a(:,:,2:3);
assert (to_octave_type (a), a2)
a(:,:,2:3) = to_octave_type (0); a2(:,:,2:3) = 0;
assert (to_octave_type (a), a2)
a(:,:,:) = a(:,:,:);
assert (to_octave_type (a), a2)
a(:,:,:) = to_octave_type (j); a2(:,:,:) = j;
assert (to_octave_type (a), a2)

b = b0; b2 = b20;

b(23) = b(1); b2(23) = b2(1);
assert (to_octave_type (b), b2)
b(23) = to_octave_type (0); b2(23) = 0;
assert (to_octave_type (b), b2)
b(i) = b(i); b2(i2) = b2(i2);
assert (to_octave_type (b), b2)
b(i) = to_octave_type (0); b2(i2) = 0;
assert (to_octave_type (b), b2)
b(12:23) = b(2:13); b2(12:23) = b2(2:13);
assert (to_octave_type (b), b2)
b(12:23) = to_octave_type (0); b2(12:23) = 0;
assert (to_octave_type (b), b2)
b(:) = b(:);
assert (to_octave_type (b), b2)
b(:) = to_octave_type (j); b2(:) = j;
assert (to_octave_type (b), b2)

b = b0; b2 = b20;

b(2,3) = b(1,1); b2(2,3) = b2(1,1);
assert (to_octave_type (b), b2)
b(2,3) = to_octave_type (0); b2(2,3) = 0;
assert (to_octave_type (b), b2)
b(2:3,3) = b(2:3,2); b2(2:3,3) = b2(2:3,2);
assert (to_octave_type (b), b2)
b(2:3,3) = to_octave_type (0); b2(2:3,3) = 0;
assert (to_octave_type (b), b2)
b(:,7) = b(:,7);
assert (to_octave_type (b), b2)
b(:,7) = to_octave_type (0); b2(:,7) = 0;
assert (to_octave_type (b), b2)
b(:,2:7) = b(:,2:7);
assert (to_octave_type (b), b2)
b(:,2:7) = to_octave_type (0); b2(:,2:7) = 0;
assert (to_octave_type (b), b2)
b(:,:) = b(:,:);
assert (to_octave_type (b), b2)
b(:,:) = to_octave_type (j); b2(:,:) = j;
assert (to_octave_type (b), b2)

r = r0; r2 = r20;

r(23) = r(1); r2(23) = r2(1);
assert (to_octave_type (r), r2)
r(23) = to_octave_type (0); r2(23) = 0;
assert (to_octave_type (r), r2)
r(i) = r(i); r2(i2) = r2(i2);
assert (to_octave_type (r), r2)
r(i) = to_octave_type (0); r2(i2) = 0;
assert (to_octave_type (r), r2)
r(12:23) = r(2:13); r2(12:23) = r2(2:13);
assert (to_octave_type (r), r2)
r(12:23) = to_octave_type (0); r2(12:23) = 0;
assert (to_octave_type (r), r2)
r(:) = r(:);
assert (to_octave_type (r), r2)
r(:) = to_octave_type (j); r2(:) = j;
assert (to_octave_type (r), r2)

c = c0; c2 = c20;

c(23) = c(1); c2(23) = c2(1);
assert (to_octave_type (c), c2)
c(23) = to_octave_type (0); c2(23) = 0;
assert (to_octave_type (c), c2)
c(i) = c(i); c2(i2) = c2(i2);
assert (to_octave_type (c), c2)
c(i) = to_octave_type (0); c2(i2) = 0;
assert (to_octave_type (c), c2)
c(12:23) = c(2:13); c2(12:23) = c2(2:13);
assert (to_octave_type (c), c2)
c(12:23) = to_octave_type (0); c2(12:23) = 0;
assert (to_octave_type (c), c2)
c(:) = c(:);
assert (to_octave_type (c), c2)
c(:) = to_octave_type (j); c2(:) = j;
assert (to_octave_type (c), c2)

a = a0; a2 = a20;

a(a < 15) = 0; a2(a2 < 15) = 0;  # only logically indexed assignment with scalar allowed, no other logical indexing
assert (to_octave_type (a), a2)

endfor # complex_iter

## --------- operator tests ---------

for complex_iter = 0:typefloat

if complex_iter > 0, j = sqrt (-1); else j = 0; endif

d = (round(rem(reshape((0:24)/6,5,5),1)*6))*1;
if complex_iter > 0, d = d + rot90 (d) * sqrt (-1); endif
d = to_ocl_type (d);
d0 = d;
e = d.';
s0 = 5 + j;
s = to_octave_type (5 + j);

assert (real (to_octave_type (! d)), to_octave_type (! to_octave_type (d)))
assert (to_octave_type (+d), +to_octave_type (d))
if ! typeuint
assert (to_octave_type (-d), -to_octave_type (d))
endif
assert (to_octave_type (d.'), to_octave_type (d).')
assert (to_octave_type (d'), to_octave_type (d)')
d++;
assert (to_octave_type (d), to_octave_type (d0)+1)
d--;
assert (to_octave_type (d), to_octave_type (d0))
assert (to_octave_type (-(-d)), to_octave_type (d)) # uses op_uminus_nonconst

endfor # complex_iter

if typefloat
tol = -50 * eps (typestr(5:end));
else
tol = 0;
endif

for complex_iter = 0:(3*typefloat)

d = (round(rem(reshape((0:24)/6,5,5),1)*6))*1;
e = rot90 (d);
if rem (complex_iter, 2) > 0, j = sqrt (-1); e = e + rot90 (e) * j; else j = 0; endif
if complex_iter > 1, d = d + rot90 (d) * sqrt (-1); endif
d = to_ocl_type (d);
e = to_ocl_type (e);
d0 = d;
s0 = 5 + j;
s = to_octave_type (5 + j);

assert (to_octave_type (d + e), to_octave_type (d) + to_octave_type (e))
assert (to_octave_type (d + s), to_octave_type (d) + s)
assert (to_octave_type (s + d), s + to_octave_type (d))
assert (to_octave_type (d + s0), to_octave_type (d) + s0)
assert (to_octave_type (s0 + d), s0 + to_octave_type (d))
if ! typeuint
assert (to_octave_type (d - e), to_octave_type (d) - to_octave_type (e))
assert (to_octave_type (d - s), to_octave_type (d) - s)
assert (to_octave_type (s - d), s - to_octave_type (d))
assert (to_octave_type (d - s0), to_octave_type (d) - s0)
assert (to_octave_type (s0 - d), s0 - to_octave_type (d))
else
assert (to_octave_type ((d+s) - e), to_octave_type (d+s) - to_octave_type (e))
assert (to_octave_type ((d+s) - s), to_octave_type (d+s) - s)
assert (to_octave_type (s - d), s - to_octave_type (d))
assert (to_octave_type ((d+s0) - s0), to_octave_type (d+s0) - s0)
assert (to_octave_type (s0 - d), s0 - to_octave_type (d))
endif
assert (to_octave_type (d .* e), to_octave_type (d) .* to_octave_type (e))
assert (to_octave_type (d .* s), to_octave_type (d) .* s)
assert (to_octave_type (s .* d), s .* to_octave_type (d))
assert (to_octave_type (d .* s0), to_octave_type (d) .* s0)
assert (to_octave_type (s0 .* d), s0 .* to_octave_type (d))
if typefloat # due to a bug in 'assert' for complex single values, we need the 'double' hack
assert (double (to_octave_type (d ./ e)), double (to_octave_type (d) ./ to_octave_type (e)), tol)
assert (double (to_octave_type (d ./ s)), double (to_octave_type (d) ./ s), tol)
assert (double (to_octave_type (s ./ d)), double (s ./ to_octave_type (d)), tol)
assert (double (to_octave_type (d ./ s0)), double (to_octave_type (d) ./ s0), tol)
assert (double (to_octave_type (s0 ./ d)), double (s0 ./ to_octave_type (d)), tol)
else # handling of integer division by zero differs
#assert (to_octave_type ((d.*s) ./ e), to_octave_type (d.*s) ./ to_octave_type (e))
assert (to_octave_type ((d.*s) ./ s), to_octave_type (d.*s) ./ s)
assert (to_octave_type ((d.*s0) ./ s0), to_octave_type (d.*s0) ./ s0)
#assert (to_octave_type (s ./ d), s ./ to_octave_type (d))
endif
if typefloat
assert (to_octave_type (d * e), to_octave_type (d) * to_octave_type (e))
else # binary operator '*' not implemented for octave '[u]int* matrix' by '[u]int* matrix' operations
assert (to_octave_type (d * e), to_octave_type (double (to_octave_type (d)) * double (to_octave_type (e))))
endif
assert (to_octave_type (d * s), to_octave_type (d) * s)
assert (to_octave_type (s * d), s * to_octave_type (d))
assert (to_octave_type (d * s0), to_octave_type (d) * s0)
assert (to_octave_type (s0 * d), s0 * to_octave_type (d))
if typefloat
assert (double (to_octave_type (d / s)), double (to_octave_type (d) / s), tol)
assert (double (to_octave_type (d / s0)), double (to_octave_type (d) / s0), tol)
else
assert (to_octave_type ((d*s) / s), to_octave_type (d*s) / s)
assert (to_octave_type ((d*s0) / s0), to_octave_type (d*s0) / s0)
endif
if typefloat
if ! iscomplex (e)
assert (double (to_octave_type (d .^ e)), double (to_octave_type (d) .^ to_octave_type (e)), tol)
else # unfortunately, octave computes complex(0)^complex(0) = (0)^complex(0) = NaN + NaNi instead of 1
assert (double (to_octave_type ((d+eps) .^ e)), double ((to_octave_type (d) + eps) .^ to_octave_type (e)), tol)
endif
assert (double (to_octave_type (d .^ s)), double (to_octave_type (d) .^ s), tol)
assert (double (to_octave_type (s .^ d)), double (s .^ to_octave_type (d)), tol)
assert (double (to_octave_type (d .^ s0)), double (to_octave_type (d) .^ s0), tol)
assert (double (to_octave_type (s0 .^ d)), double (s0 .^ to_octave_type (d)), tol)
endif

assert (real (to_octave_type (d & e)), to_octave_type (to_octave_type (d) & to_octave_type (e)))
assert (real (to_octave_type (d & s)), to_octave_type (to_octave_type (d) & s))
assert (real (to_octave_type (s & d)), to_octave_type (s & to_octave_type (d)))
assert (real (to_octave_type (d & s0)), to_octave_type (to_octave_type (d) & s0))
assert (real (to_octave_type (s0 & d)), to_octave_type (s0 & to_octave_type (d)))
assert (real (to_octave_type (d | e)), to_octave_type (to_octave_type (d) | to_octave_type (e)))
assert (real (to_octave_type (d | s)), to_octave_type (to_octave_type (d) | s))
assert (real (to_octave_type (s | d)), to_octave_type (s | to_octave_type (d)))
assert (real (to_octave_type (d | s0)), to_octave_type (to_octave_type (d) | s0))
assert (real (to_octave_type (s0 | d)), to_octave_type (s0 | to_octave_type (d)))

s0 = 2;
s = to_octave_type (2);
assert (real (to_octave_type (d < e)), to_octave_type (to_octave_type (d) < to_octave_type (e)))
assert (real (to_octave_type (d < s)), to_octave_type (to_octave_type (d) < s))
assert (real (to_octave_type (s < d)), to_octave_type (s < to_octave_type (d)))
assert (real (to_octave_type (d < s0)), to_octave_type (to_octave_type (d) < s0))
assert (real (to_octave_type (s0 < d)), to_octave_type (s0 < to_octave_type (d)))
assert (real (to_octave_type (d <= e)), to_octave_type (to_octave_type (d) <= to_octave_type (e)))
assert (real (to_octave_type (d <= s)), to_octave_type (to_octave_type (d) <= s))
assert (real (to_octave_type (s <= d)), to_octave_type (s <= to_octave_type (d)))
assert (real (to_octave_type (d <= s0)), to_octave_type (to_octave_type (d) <= s0))
assert (real (to_octave_type (s0 <= d)), to_octave_type (s0 <= to_octave_type (d)))
assert (real (to_octave_type (d > e)), to_octave_type (to_octave_type (d) > to_octave_type (e)))
assert (real (to_octave_type (d > s)), to_octave_type (to_octave_type (d) > s))
assert (real (to_octave_type (s > d)), to_octave_type (s > to_octave_type (d)))
assert (real (to_octave_type (d > s0)), to_octave_type (to_octave_type (d) > s0))
assert (real (to_octave_type (s0 > d)), to_octave_type (s0 > to_octave_type (d)))
assert (real (to_octave_type (d >= e)), to_octave_type (to_octave_type (d) >= to_octave_type (e)))
assert (real (to_octave_type (d >= s)), to_octave_type (to_octave_type (d) >= s))
assert (real (to_octave_type (s >= d)), to_octave_type (s >= to_octave_type (d)))
assert (real (to_octave_type (d >= s0)), to_octave_type (to_octave_type (d) >= s0))
assert (real (to_octave_type (s0 >= d)), to_octave_type (s0 >= to_octave_type (d)))
assert (real (to_octave_type (d == e)), to_octave_type (to_octave_type (d) == to_octave_type (e)))
assert (real (to_octave_type (d == s)), to_octave_type (to_octave_type (d) == s))
assert (real (to_octave_type (s == d)), to_octave_type (s == to_octave_type (d)))
assert (real (to_octave_type (d == s0)), to_octave_type (to_octave_type (d) == s0))
assert (real (to_octave_type (s0 == d)), to_octave_type (s0 == to_octave_type (d)))
assert (real (to_octave_type (d != e)), to_octave_type (to_octave_type (d) != to_octave_type (e)))
assert (real (to_octave_type (d != s)), to_octave_type (to_octave_type (d) != s))
assert (real (to_octave_type (s != d)), to_octave_type (s != to_octave_type (d)))
assert (real (to_octave_type (d != s0)), to_octave_type (to_octave_type (d) != s0))
assert (real (to_octave_type (s0 != d)), to_octave_type (s0 != to_octave_type (d)))

d = d0;
d += e;
assert (to_octave_type (d), to_octave_type (d0) + to_octave_type (e))
d = d0 + 2 * e;
d -= e;
assert (to_octave_type (d), to_octave_type (d0) + to_octave_type (e))
d = d0;
d += s;
assert (to_octave_type (d), to_octave_type (d0) + s)
d = d0 + 2 * s;
d -= s;
assert (to_octave_type (d), to_octave_type (d0) + s)
d = d0;
d += s0;
assert (to_octave_type (d), to_octave_type (d0) + s0)
d = d0 + 2 * s0;
d -= s0;
assert (to_octave_type (d), to_octave_type (d0) + s0)
d = d0;
d .*= e;
assert (to_octave_type (d), to_octave_type (d0) .* to_octave_type (e))
d ./= e;
if typefloat
assert (double (to_octave_type (d)), double ((to_octave_type (d0) .* to_octave_type (e)) ./ to_octave_type (e)), tol)
else
inds = to_octave_type (e) != to_octave_type (0);
assert (to_octave_type (d)(inds), to_octave_type (d0)(inds))
endif
d = d0;
d .*= s;
assert (to_octave_type (d), to_octave_type (d0) .* s)
d = d0 .* (s*s);
d ./= s;
assert (to_octave_type (d), to_octave_type (d0) .* s, tol)
d = d0;
d *= s;
assert (to_octave_type (d), to_octave_type (d0) * s)
d = d0 * (s*s);
d /= s;
assert (to_octave_type (d), to_octave_type (d0) * s, tol)
d = d0;
d .*= s0;
assert (to_octave_type (d), to_octave_type (d0) .* s0)
d = d0 .* (s0*s0);
d ./= s0;
assert (to_octave_type (d), to_octave_type (d0) .* s0, tol)
d = d0;
d *= s0;
assert (to_octave_type (d), to_octave_type (d0) * s0)
d = d0 * (s0*s0);
d /= s0;
assert (to_octave_type (d), to_octave_type (d0) * s0, tol)

endfor # complex_iter

## --------- dimension-wise (math) tests ---------

for complex_iter = 0:typefloat

if complex_iter > 0, j = sqrt (-1); else j = 0; endif

d = to_ocl_type ((round(rem(reshape((1:16)/5,4,4),1)*5))*1 + 3*j);
e = d.';
s0 = 2 + j;
s = to_octave_type (2 + j);
c = d(:,2);
r = c';

assert (real (to_octave_type (all (r))), to_octave_type (all (to_octave_type (r))))
assert (real (to_octave_type (all (c))), to_octave_type (all (to_octave_type (c))))
assert (real (to_octave_type (all (d))), to_octave_type (all (to_octave_type (d))))
assert (real (to_octave_type (all (d, 1))), to_octave_type (all (to_octave_type (d), 1)))
assert (real (to_octave_type (all (d, 2))), to_octave_type (all (to_octave_type (d), 2)))

assert (real (to_octave_type (any (r))), to_octave_type (any (to_octave_type (r))))
assert (real (to_octave_type (any (c))), to_octave_type (any (to_octave_type (c))))
assert (real (to_octave_type (any (d))), to_octave_type (any (to_octave_type (d))))
assert (real (to_octave_type (any (d, 1))), to_octave_type (any (to_octave_type (d), 1)))
assert (real (to_octave_type (any (d, 2))), to_octave_type (any (to_octave_type (d), 2)))

assert (to_octave_type (sum (r)), to_octave_type (sum (to_octave_type (r))))
assert (to_octave_type (sum (c)), to_octave_type (sum (to_octave_type (c))))
assert (to_octave_type (sum (d)), to_octave_type (sum (to_octave_type (d))))
assert (to_octave_type (sum (d, 1)), to_octave_type (sum (to_octave_type (d), 1)))
assert (to_octave_type (sum (d, 2)), to_octave_type (sum (to_octave_type (d), 2)))

assert (real (to_octave_type (sumsq (r))), to_octave_type (sumsq (to_octave_type (r))))
assert (real (to_octave_type (sumsq (c))), to_octave_type (sumsq (to_octave_type (c))))
assert (real (to_octave_type (sumsq (d))), to_octave_type (sumsq (to_octave_type (d))))
assert (real (to_octave_type (sumsq (d, 1))), to_octave_type (sumsq (to_octave_type (d), 1)))
assert (real (to_octave_type (sumsq (d, 2))), to_octave_type (sumsq (to_octave_type (d), 2)))

assert (to_octave_type (prod (r)), to_octave_type (prod (to_octave_type (r))))
assert (to_octave_type (prod (c)), to_octave_type (prod (to_octave_type (c))))
assert (to_octave_type (prod (d)), to_octave_type (prod (to_octave_type (d))))
assert (to_octave_type (prod (d, 1)), to_octave_type (prod (to_octave_type (d), 1)))
assert (to_octave_type (prod (d, 2)), to_octave_type (prod (to_octave_type (d), 2)))

if typefloat
assert (to_octave_type (mean (r)), to_octave_type (mean (to_octave_type (r))))
assert (to_octave_type (mean (c)), to_octave_type (mean (to_octave_type (c))))
assert (to_octave_type (mean (d)), to_octave_type (mean (to_octave_type (d))))
assert (to_octave_type (mean (d, 1)), to_octave_type (mean (to_octave_type (d), 1)))
assert (to_octave_type (mean (d, 2)), to_octave_type (mean (to_octave_type (d), 2)))

assert (real (to_octave_type (meansq (r))), to_octave_type (meansq (to_octave_type (r))))
assert (real (to_octave_type (meansq (c))), to_octave_type (meansq (to_octave_type (c))))
assert (real (to_octave_type (meansq (d))), to_octave_type (meansq (to_octave_type (d))))
assert (real (to_octave_type (meansq (d, 1))), to_octave_type (meansq (to_octave_type (d), 1)))
assert (real (to_octave_type (meansq (d, 2))), to_octave_type (meansq (to_octave_type (d), 2)))

tol = 5;

assert (to_octave_type (std (r)), to_octave_type (std (to_octave_type (r))), tol * eps (typestr(5:end)))
assert (to_octave_type (std (c)), to_octave_type (std (to_octave_type (c))), tol * eps (typestr(5:end)))
assert (to_octave_type (std (d)), to_octave_type (std (to_octave_type (d))), tol * eps (typestr(5:end)))
assert (to_octave_type (std (d, 0, 1)), to_octave_type (std (to_octave_type (d), 0, 1)), tol * eps (typestr(5:end)))
assert (to_octave_type (std (d, 0, 2)), to_octave_type (std (to_octave_type (d), 0, 2)), tol * eps (typestr(5:end)))
assert (to_octave_type (std (d, 1, 1)), to_octave_type (std (to_octave_type (d), 1, 1)), tol * eps (typestr(5:end)))
assert (to_octave_type (std (d, 1, 2)), to_octave_type (std (to_octave_type (d), 1, 2)), tol * eps (typestr(5:end)))
endif

assert (to_octave_type (cumsum (r)), to_octave_type (cumsum (to_octave_type (r))))
assert (to_octave_type (cumsum (c)), to_octave_type (cumsum (to_octave_type (c))))
assert (to_octave_type (cumsum (d)), to_octave_type (cumsum (to_octave_type (d))))
assert (to_octave_type (cumsum (d, 1)), to_octave_type (cumsum (to_octave_type (d), 1)))
assert (to_octave_type (cumsum (d, 2)), to_octave_type (cumsum (to_octave_type (d), 2)))

assert (to_octave_type (cumprod (r)), to_octave_type (cumprod (to_octave_type (r))))
assert (to_octave_type (cumprod (c)), to_octave_type (cumprod (to_octave_type (c))))
assert (to_octave_type (cumprod (d)), to_octave_type (cumprod (to_octave_type (d))))
assert (to_octave_type (cumprod (d, 1)), to_octave_type (cumprod (to_octave_type (d), 1)))
assert (to_octave_type (cumprod (d, 2)), to_octave_type (cumprod (to_octave_type (d), 2)))

## OCL offers dimension-wise (first/last) find functions for nD arrays, octave does not, i.e. only for vectors
assert (to_octave_type (int64 (findfirst (r))), to_octave_type (find (to_octave_type (r), 1, 'first')))
assert (to_octave_type (int64 (findfirst (c))), to_octave_type (find (to_octave_type (c), 1, 'first')))
  # assert (to_octave_type (int64 (findfirst (d))), to_octave_type (findfirst (to_octave_type (d)))) # see above
  # assert (to_octave_type (int64 (findfirst (d, 1))), to_octave_type (findfirst (to_octave_type (d), 1))) # see above
  # assert (to_octave_type (int64 (findfirst (d, 2))), to_octave_type (findfirst (to_octave_type (d), 2))) # see above
if complex_iter == 0
assert (to_octave_type (int64 (findfirst (d))), to_octave_type ([1 2 1 1]))
assert (to_octave_type (int64 (findfirst (d, 1))), to_octave_type ([1 2 1 1]))
assert (to_octave_type (int64 (findfirst (d, 2))), to_octave_type ([1 1 1 1]'))
else
assert (to_octave_type (int64 (findfirst (d))), to_octave_type ([1 1 1 1]))
assert (to_octave_type (int64 (findfirst (d, 1))), to_octave_type ([1 1 1 1]))
assert (to_octave_type (int64 (findfirst (d, 2))), to_octave_type ([1 1 1 1]'))
endif

assert (to_octave_type (int64 (findlast (r))), to_octave_type (find (to_octave_type (r), 1, 'last')))
assert (to_octave_type (int64 (findlast (c))), to_octave_type (find (to_octave_type (c), 1, 'last')))
  # assert (to_octave_type (int64 (findlast (d))), to_octave_type (findlast (to_octave_type (d)))) # see above
  # assert (to_octave_type (int64 (findlast (d, 1))), to_octave_type (findlast (to_octave_type (d), 1))) # see above
  # assert (to_octave_type (int64 (findlast (d, 2))), to_octave_type (findlast (to_octave_type (d), 2))) # see above
if complex_iter == 0
assert (to_octave_type (int64 (findlast (d))), to_octave_type ([4 4 4 4]))
assert (to_octave_type (int64 (findlast (d, 1))), to_octave_type ([4 4 4 4]))
assert (to_octave_type (int64 (findlast (d, 2))), to_octave_type ([4 4 3 4]'))
else
assert (to_octave_type (int64 (findlast (d))), to_octave_type ([4 4 4 4]))
assert (to_octave_type (int64 (findlast (d, 1))), to_octave_type ([4 4 4 4]))
assert (to_octave_type (int64 (findlast (d, 2))), to_octave_type ([4 4 4 4]'))
endif

assert (to_octave_type (max (r)), max (to_octave_type (r)))
assert (to_octave_type (max (c)), max (to_octave_type (c)))
assert (to_octave_type (max (d)), max (to_octave_type (d)))
assert (to_octave_type (max (d, [], 1)), max (to_octave_type (d), [], 1))
assert (to_octave_type (max (d, [], 2)), max (to_octave_type (d), [], 2))
[v, i] = max (r); [vo, io] = max (to_octave_type (r));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = max (c); [vo, io] = max (to_octave_type (c));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = max (d); [vo, io] = max (to_octave_type (d));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = max (d, [], 1); [vo, io] = max (to_octave_type (d), [], 1);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = max (d, [], 2); [vo, io] = max (to_octave_type (d), [], 2);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
assert (to_octave_type (max (d, e)), max (to_octave_type (d), to_octave_type (e)))
assert (to_octave_type (max (d, s)), max (to_octave_type (d), s))
assert (to_octave_type (max (s, d)), max (s, to_octave_type (d)))

assert (to_octave_type (min (r)), min (to_octave_type (r)))
assert (to_octave_type (min (c)), min (to_octave_type (c)))
assert (to_octave_type (min (d)), min (to_octave_type (d)))
assert (to_octave_type (min (d, [], 1)), min (to_octave_type (d), [], 1))
assert (to_octave_type (min (d, [], 2)), min (to_octave_type (d), [], 2))
[v, i] = min (r); [vo, io] = min (to_octave_type (r));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = min (c); [vo, io] = min (to_octave_type (c));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = min (d); [vo, io] = min (to_octave_type (d));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = min (d, [], 1); [vo, io] = min (to_octave_type (d), [], 1);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = min (d, [], 2); [vo, io] = min (to_octave_type (d), [], 2);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
assert (to_octave_type (min (d, e)), min (to_octave_type (d), to_octave_type (e)))
assert (to_octave_type (min (d, s)), min (to_octave_type (d), s))
assert (to_octave_type (min (s, d)), min (s, to_octave_type (d)))

assert (to_octave_type (cummax (r)), cummax (to_octave_type (r)))
assert (to_octave_type (cummax (c)), cummax (to_octave_type (c)))
assert (to_octave_type (cummax (d)), cummax (to_octave_type (d)))
assert (to_octave_type (cummax (d, 1)), cummax (to_octave_type (d), 1))
assert (to_octave_type (cummax (d, 2)), cummax (to_octave_type (d), 2))
[v, i] = cummax (r); [vo, io] = cummax (to_octave_type (r));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummax (c); [vo, io] = cummax (to_octave_type (c));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummax (d); [vo, io] = cummax (to_octave_type (d));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummax (d, 1); [vo, io] = cummax (to_octave_type (d), 1);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummax (d, 2); [vo, io] = cummax (to_octave_type (d), 2);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))

assert (to_octave_type (cummin (r)), cummin (to_octave_type (r)))
assert (to_octave_type (cummin (c)), cummin (to_octave_type (c)))
assert (to_octave_type (cummin (d)), cummin (to_octave_type (d)))
assert (to_octave_type (cummin (d, 1)), cummin (to_octave_type (d), 1))
assert (to_octave_type (cummin (d, 2)), cummin (to_octave_type (d), 2))
[v, i] = cummin (r); [vo, io] = cummin (to_octave_type (r));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummin (c); [vo, io] = cummin (to_octave_type (c));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummin (d); [vo, io] = cummin (to_octave_type (d));
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummin (d, 1); [vo, io] = cummin (to_octave_type (d), 1);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))
[v, i] = cummin (d, 2); [vo, io] = cummin (to_octave_type (d), 2);
assert (to_octave_type (v), vo)
assert (int64 (i), int64 (io))

endfor # complex_iter

## --------- mapping (math) function tests ---------

for complex_iter = 0:typefloat

if typefloat
tol = -50 * eps (typestr(5:end));
else
tol = 0;
endif

if (! typefloat) || (complex_iter == 0)
d = to_ocl_type ((round(rem(reshape((1:16)/5,4,4),1)*5))*1);
f = d(:,2:3);
else
d = repmat (-4:4, 9, 1);
d = d + rot90 (d) * sqrt (-1);
f = to_ocl_type (d);
endif

assert (to_octave_type (abs (f)), abs (to_octave_type (f)), tol)
assert (to_octave_type (ceil (f)), ceil (to_octave_type (f)))
assert (to_octave_type (fix (f)), fix (to_octave_type (f)))
assert (to_octave_type (floor (f)), floor (to_octave_type (f)))
assert (to_octave_type (round (f)), round (to_octave_type (f)))
assert (to_octave_type (real (f)), real (to_octave_type (f)))
assert (to_octave_type (imag (f)), imag (to_octave_type (f)))
assert (to_octave_type (conj (f)), conj (to_octave_type (f)))
assert (to_octave_type (isinf (f)), to_octave_type (isinf (to_octave_type (f))))
assert (to_octave_type (isnan (f)), to_octave_type (isnan (to_octave_type (f))))
if compare_versions (version (), "4.0.0", "<")
assert (to_octave_type (finite (f)), to_octave_type (finite (to_octave_type (f))))
else
assert (to_octave_type (isfinite (f)), to_octave_type (isfinite (to_octave_type (f))))
endif

if typefloat
tol = 5e4 * eps (typestr(5:end)); # let's be tolerant considering the possible range of drivers
tol_1 = max (tol, 1e-6);
if complex_iter == 0
assert (to_octave_type (acos (f/4)), acos (to_octave_type (f/4)), tol)
assert (to_octave_type (acosh (f+1)), acosh (to_octave_type (f+1)), tol)
assert (to_octave_type (asin (f/4)), asin (to_octave_type (f/4)), tol)
assert (to_octave_type (asinh (f)), asinh (to_octave_type (f)), tol)
assert (to_octave_type (atan (f)), atan (to_octave_type (f)), tol)
assert (to_octave_type (atanh (f/4)), atanh (to_octave_type (f/4)), tol)
assert (to_octave_type (cbrt (f)), cbrt (to_octave_type (f)), tol)
assert (to_octave_type (cos (f)), cos (to_octave_type (f)), tol)
assert (to_octave_type (cosh (f)), cosh (to_octave_type (f)), tol)
assert (to_octave_type (erf (f)), erf (to_octave_type (f)), tol_1)
assert (to_octave_type (erfc (f)), erfc (to_octave_type (f)), tol_1)
assert (to_octave_type (exp (f)), exp (to_octave_type (f)), tol)
assert (to_octave_type (expm1 (f)), expm1 (to_octave_type (f)), tol)
assert (to_octave_type (gamma (f)), gamma (to_octave_type (f)), tol_1)
assert (to_octave_type (lgamma (f)), lgamma (to_octave_type (f)), tol_1)
assert (to_octave_type (log (f+1)), log (to_octave_type (f+1)), tol)
assert (to_octave_type (log2 (f+1)), log2 (to_octave_type (f+1)), tol)
assert (to_octave_type (log10 (f+1)), log10 (to_octave_type (f+1)), tol)
assert (to_octave_type (log1p (f)), log1p (to_octave_type (f)), tol)
assert (to_octave_type (sign (f)), sign (to_octave_type (f)), tol)
assert (to_octave_type (sin (f)), sin (to_octave_type (f)), tol)
assert (to_octave_type (sinh (f)), sinh (to_octave_type (f)), tol)
assert (to_octave_type (sqrt (f)), sqrt (to_octave_type (f)), tol)
assert (to_octave_type (tan (f)), tan (to_octave_type (f)), tol)
assert (to_octave_type (tanh (f)), tanh (to_octave_type (f)), tol)
assert (to_octave_type (atan2 (f, f-2)), atan2 (to_octave_type (f), to_octave_type (f-2)), tol)
assert (to_octave_type (cosd (f*90)), cosd (to_octave_type (f*90)), tol)
assert (to_octave_type (sind (f*90)), sind (to_octave_type (f*90)), tol)
assert (to_octave_type (tand (f*90)), tand (to_octave_type (f*90)), tol)
assert (to_octave_type (cotd (f*90)), cotd (to_octave_type (f*90)), tol)
else # if complex_iter
assert (to_octave_type (acos (f)), acos (to_octave_type (f)), tol)
assert (to_octave_type (acosh (f)), acosh (to_octave_type (f)), tol)
assert (to_octave_type (asin (f)), asin (to_octave_type (f)), tol)
assert (to_octave_type (asinh (f)), asinh (to_octave_type (f)), tol)
assert (to_octave_type (atan (f)), atan (to_octave_type (f)), tol)
assert (to_octave_type (atanh (f)), atanh (to_octave_type (f)), tol)
assert (to_octave_type (cos (f)), cos (to_octave_type (f)), tol)
assert (to_octave_type (cosh (f)), cosh (to_octave_type (f)), tol)
assert (to_octave_type (exp (f)), exp (to_octave_type (f)), tol)
assert (to_octave_type (log (f)), log (to_octave_type (f)), tol)
assert (to_octave_type (log2 (f)), log2 (to_octave_type (f)), tol)
assert (to_octave_type (log10 (f)), log10 (to_octave_type (f)), tol)
assert (to_octave_type (sign (f)), sign (to_octave_type (f)), tol)
assert (to_octave_type (sin (f)), sin (to_octave_type (f)), tol)
assert (to_octave_type (sinh (f)), sinh (to_octave_type (f)), tol)
assert (to_octave_type (sqrt (f)), sqrt (to_octave_type (f)), tol)
assert (to_octave_type (tan (f)), tan (to_octave_type (f)), tol)
assert (to_octave_type (tanh (f)), tanh (to_octave_type (f)), tol)
endif # if complex_iter
endif

endfor # complex_iter

endfor # for ocltype


## --------- ocl program data type tests ---------

disp (["Testing ocl program data type..."]); fflush (stdout);

assert (class (help ("ocl_program")), "char") # test for help string

src = [...
"__kernel void" "\n" ...
"myprog" "\n" ...
"  (__global float *dst, const __global float *src)" "\n" ...
"{" "\n" ...
"  size_t i = get_global_id (0);" "\n" ...
"  dst[i] = src[i];" "\n" ...
"}" "\n" ...
"\n" ...
"__kernel void" "\n" ...
"linspace_types" "\n" ...
"  (__global float *data_dst_f," "\n" ...
"   __global uchar *data_dst_u," "\n" ...
"   __global int *data_dst_i," "\n" ...
"   const float start_val," "\n" ...
"   const float end_val," "\n" ...
"   const ulong n)" "\n" ...
"{" "\n" ...
"  size_t i = get_global_id (0);" "\n" ...
"  float f = start_val + ((end_val-start_val)*i)/(n-1);" "\n" ...
"  data_dst_f [i] = f;" "\n" ...
"  data_dst_u [i] = f;" "\n" ...
"  data_dst_i [i] = f;" "\n" ...
"}" "\n" ...
];

prog = ocl_program (src);

assert (class (disp (prog)), "char") # test for variable printing to terminal

assert (prog.valid, logical (1))
assert (prog.num_kernels, 2)

## the order of kernels, or the kernel indices, need not follow the same order as in the source string
kernel_names = prog.kernel_names;
assert (class (kernel_names), "cell")
kernelnr_myprog = prog("myprog");
kernelnr_linspace_types = prog("linspace_types");
assert (prog(0), 0)
assert (prog(1), 1)
assert (kernel_names{1+kernelnr_myprog}, "myprog")
assert (kernel_names{1+kernelnr_linspace_types}, "linspace_types")

a = ocl_single (reshape (1:16, 4, 4));
ac = a(:,3:4);

b = prog (kernelnr_myprog, numel (ac), { size(ac); "single" }, ac);

assert (class (b), class (ac))
assert (single (b), single (ac))

b = prog (kernelnr_myprog, numel (ac), { size(ac); "single" }, ac, "make_unique");
assert (single (b), single (ac))

n = 11;
[c1,c2,c3] = prog ("linspace_types", n, ...
                   { [n], "single"; [n], "uint8"; [n], "int32" }, ...
                   single (100), single (200), uint64 (n));

assert (class (c1), "ocl_single")
assert (class (c2), "ocl_uint8")
assert (class (c3), "ocl_int32")
assert (single (c1), single (100:10:200)')
assert (uint8 (c2), uint8 (100:10:200)')
assert (int32 (c3), int32 (100:10:200)')


# program example using complex arguments
src = [...
"__kernel void" "\n" ...
"myswap" "\n" ...
"  (__global float2 *dst, const __global float2 *src)" "\n" ...
"{" "\n" ...
"  size_t i = get_global_id (0);" "\n" ...
"  float2 tmp = src[i];" "\n" ...
"  dst[i] = (float2) (tmp.y, tmp.x);" "\n" ...
"}" "\n" ...
];

prog = ocl_program (src);

b = prog (0, 4, { [4 1]; "single_complex" }, complex (a(:,3), a(:,4)));

assert (single (b), single (complex (a(:,4), a(:,3))))


# program example using options
src = [...
"__kernel void" "\n" ...
"mysum" "\n" ...
"  (__global float *dst, " "\n" ...
"   const __global float *src1, " "\n" ...
"   const ulong ofs_src1, " "\n" ...
"   const __global float *src2, " "\n" ...
"   const ulong ofs_src2)" "\n" ...
"{" "\n" ...
"  src1 += ofs_src1;" "\n" ...
"  src2 += ofs_src2;" "\n" ...
"  size_t i = get_global_id (0);" "\n" ...
"  dst[i] = src1[i] + src2[i];" "\n" ...
"}" "\n" ...
];

prog = ocl_program (src);

## the following line works using slice_ofs
b = prog (0, 4, { [4 1]; "single" }, a(:,3), a(:,4), "slice_ofs");

assert (single (b), single (a(:,3)) + single (a(:,4)))

## the following line also works, with zero offsets, because of make_unique (default)
b = prog (0, 4, { [4 1]; "single" }, a(:,3), uint64 (0), a(:,4), uint64 (0));

assert (single (b), single (a(:,3)) + single (a(:,4)))


# dummy saving and loading tests
# The OCL saving and loading dummy functionality is there ONLY in order not break
# when saving / loading workspaces which also contain OCL variables (including core dumps).
# Contained OCL data is not saved, however, since it is always context dependent.
# OCL matrix data should be transfered to octave by 'ocl_to_octave' before saving.
# Before loading, the OCL package must be loaded (by 'pkg load ocl') nevertheless.

warning ("off", "Ocl:matrix_save", "local"); # for testing here, switch off the warnings
tmpfile = tempname ();
save ("-text", tmpfile);
load ("-text", tmpfile);
save ("-binary", tmpfile);
load ("-binary", tmpfile);
hdf = 1; try; save ("-hdf5", tmpfile, "hdf"); catch; hdf = 0; end
if hdf
save ("-hdf5", tmpfile);
load ("-hdf5", tmpfile);
endif # if hdf
try; unlink (tmpfile); catch; end


## --------- end of tests ---------

disp (["Testing DONE"]); fflush (stdout);

ret = 1;

endfunction

%!assert (ocl_tests ())
