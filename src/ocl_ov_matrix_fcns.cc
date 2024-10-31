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

/*
 * A minor part of this file is based on content from the following files,
 * originally published with GNU Octave 3.8.0, distributed under the same
 * license (as OCL, see above), with the following Copyright notices:
 *
 * data.cc (some help texts are derived from the file):
 *   Copyright (C) 1994-2013 John W. Eaton
 *   Copyright (C) 2009 Jaroslav Hajek
 *   Copyright (C) 2009-2010 VZLU Prague
 *   Copyright (C) 2012 Carlo de Falco
 */

#include <octave/oct.h>

#include "ocl_octave_versions.h"
#include "ocl_lib.h"
#include "ocl_ov_matrix.h"
#include "ocl_ov_types.h"


// ---------- the octave entry point to the 'ocl_cat' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_cat", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_cat", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_cat, args, nargout,
"-*- texinfo -*-\n\
@deftypefn {Loadable Function} {} ocl_cat (@var{dim}, @var{ocl_array1}, @var{ocl_array2}, @dots{}, @var{ocl_arrayN}) \n\
\n\
Return the concatenation of the N-dimensional OCL array objects, @var{ocl_array1}, \n\
@var{ocl_array2}, @dots{}, @var{ocl_arrayN} along dimension @var{dim}.  \n\
\n\
For details, see help for @code{cat}.  \n\
\n\
@seealso{cat, oclArray} \n\
@end deftypefn")
{
  octave_value_list retval;
  int nargin = args.length ();

  if (nargin < 2)
    ocl_error ("ocl_cat: too few arguments");
  if (!args(0).is_real_scalar ())
    ocl_error ("ocl_cat: first argument must be a real scalar");

  int dim = args(0).scalar_value () - 1;
  octave_value a0 = args(1);
  int type_id = a0.type_id ();

  for (int i = 2; i < nargin; i++)
    if (args(i).type_id () != type_id)
      ocl_error ("ocl_cat: all arguments to concatenate must have the same type");

  assure_installed_ocl_types ();

#define OCL_CAT_TYPE(T) \
  if (type_id == T::static_type_id ()) { \
    T::array_type array_list [nargin-1]; \
    T::array_type retval_array; \
    for (int i = 0; i < nargin-1; i++) { \
      T *mat = dynamic_cast<T *> (args(1+i).internal_rep ()); \
      if (!mat) \
        ocl_error ("ocl_cat: invalid argument"); \
      array_list [i] = mat->ocl_array_value (); \
    } \
    retval_array = T::array_type::cat (dim, nargin-1, array_list); \
    retval (0) = octave_value (new T (retval_array)); \
  } else

  OCL_CAT_TYPE( octave_ocl_matrix )
  OCL_CAT_TYPE( octave_ocl_float_matrix )
  OCL_CAT_TYPE( octave_ocl_complex_matrix )
  OCL_CAT_TYPE( octave_ocl_float_complex_matrix )
  OCL_CAT_TYPE( octave_ocl_int8_matrix )
  OCL_CAT_TYPE( octave_ocl_int16_matrix )
  OCL_CAT_TYPE( octave_ocl_int32_matrix )
  OCL_CAT_TYPE( octave_ocl_int64_matrix )
  OCL_CAT_TYPE( octave_ocl_uint8_matrix )
  OCL_CAT_TYPE( octave_ocl_uint16_matrix )
  OCL_CAT_TYPE( octave_ocl_uint32_matrix )
  OCL_CAT_TYPE( octave_ocl_uint64_matrix )
    ocl_error ("ocl_cat: arguments to concatenate must be ocl matrices, of same type"); // default case after last "else"

#undef OCL_CAT_TYPE

  return retval;
}


// ---------- the octave entry point to the 'ocl_ones' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_ones", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_ones", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_ones, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} ocl_ones (@var{n}) \n\
@deftypefnx {Loadable Function} {} ocl_ones (@var{m}, @var{n}) \n\
@deftypefnx {Loadable Function} {} ocl_ones (@var{m}, @var{n}, @var{k}, @dots{}) \n\
@deftypefnx {Loadable Function} {} ocl_ones ([@var{m} @var{n} @dots{}]) \n\
@deftypefnx {Loadable Function} {} ocl_ones (@dots{}, @var{class}) \n\
\n\
Return an OCL matrix or N-dimensional OCL array whose elements are all 1.  \n\
\n\
For details, see help for @code{ones}.  \n\
\n\
The OCL matrix is assembled on the OpenCL device.  \n\
\n\
@seealso{ones, ocl_zeros, oclArray} \n\
@end deftypefn")
{
  int value = 1;
  octave_value_list retval;
  int nargin = args.length ();
  std::string val_class = "double";
  dim_vector dv (1,1);

  if ((nargin > 0) && (args (nargin-1).is_string ())) {
    val_class = args (nargin-1).string_value ();
    nargin--;
  }
  if (nargin < 1)
    ocl_error ("ocl_ones: too few arguments");

  if (args (0).is_real_matrix ()) {
    if (nargin > 1)
      ocl_error ("ocl_ones: too many arguments");
    Matrix m = args (0).matrix_value ();
    int ndim = m.numel ();
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++)
      dv (i) = m (i);
  } else {
    int ndim = nargin;
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++) {
      if (!args (i).is_real_scalar ())
        ocl_error ("ocl_ones: wrong argument type");
      dv (i) = args (i).scalar_value ();
    }
    if (nargin == 1)
      dv (1) = dv (0);
  }

  assure_installed_ocl_types ();

#define OCL_ONES_TYPE(C, T) \
  if (val_class == #C) { \
    T::array_type retval_array (dv, T::element_type (value)); \
    retval (0) = octave_value (new T (retval_array)); \
  } else

  OCL_ONES_TYPE( double, octave_ocl_matrix )
  OCL_ONES_TYPE( single, octave_ocl_float_matrix )
  OCL_ONES_TYPE( int8,   octave_ocl_int8_matrix )
  OCL_ONES_TYPE( int16,  octave_ocl_int16_matrix )
  OCL_ONES_TYPE( int32,  octave_ocl_int32_matrix )
  OCL_ONES_TYPE( int64,  octave_ocl_int64_matrix )
  OCL_ONES_TYPE( uint8,  octave_ocl_uint8_matrix )
  OCL_ONES_TYPE( uint16, octave_ocl_uint16_matrix )
  OCL_ONES_TYPE( uint32, octave_ocl_uint32_matrix )
  OCL_ONES_TYPE( uint64, octave_ocl_uint64_matrix )
    ocl_error ("ocl_ones: 'class' must be a data type as for 'ones'"); // default case after last "else"

#undef OCL_ONES_TYPE

  return retval;
}


// ---------- the octave entry point to the 'ocl_zeros' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_zeros", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_zeros", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_zeros, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} ocl_zeros (@var{n}) \n\
@deftypefnx {Loadable Function} {} ocl_zeros (@var{m}, @var{n}) \n\
@deftypefnx {Loadable Function} {} ocl_zeros (@var{m}, @var{n}, @var{k}, @dots{}) \n\
@deftypefnx {Loadable Function} {} ocl_zeros ([@var{m} @var{n} @dots{}]) \n\
@deftypefnx {Loadable Function} {} ocl_zeros (@dots{}, @var{class}) \n\
\n\
Return an OCL matrix or N-dimensional OCL array whose elements are all 0.  \n\
\n\
For details, see help for @code{zeros}.  \n\
\n\
The OCL matrix is assembled on the OpenCL device.  \n\
\n\
@seealso{zeros, ocl_ones, oclArray} \n\
@end deftypefn")
{
  int value = 0;
  octave_value_list retval;
  int nargin = args.length ();
  std::string val_class = "double";
  dim_vector dv (1,1);

  if ((nargin > 0) && (args (nargin-1).is_string ())) {
    val_class = args (nargin-1).string_value ();
    nargin--;
  }
  if (nargin < 1)
    ocl_error ("ocl_zeros: too few arguments");

  if (args (0).is_real_matrix ()) {
    if (nargin > 1)
      ocl_error ("ocl_zeros: too many arguments");
    Matrix m = args (0).matrix_value ();
    int ndim = m.numel ();
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++)
      dv (i) = m (i);
  } else {
    int ndim = nargin;
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++) {
      if (!args (i).is_real_scalar ())
        ocl_error ("ocl_zeros: wrong argument type");
      dv (i) = args (i).scalar_value ();
    }
    if (nargin == 1)
      dv (1) = dv (0);
  }

  assure_installed_ocl_types ();

#define OCL_ZEROS_TYPE(C, T) \
  if (val_class == #C) { \
    T::array_type retval_array (dv, T::element_type (value)); \
    retval (0) = octave_value (new T (retval_array)); \
  } else

  OCL_ZEROS_TYPE( double, octave_ocl_matrix )
  OCL_ZEROS_TYPE( single, octave_ocl_float_matrix )
  OCL_ZEROS_TYPE( int8,   octave_ocl_int8_matrix )
  OCL_ZEROS_TYPE( int16,  octave_ocl_int16_matrix )
  OCL_ZEROS_TYPE( int32,  octave_ocl_int32_matrix )
  OCL_ZEROS_TYPE( int64,  octave_ocl_int64_matrix )
  OCL_ZEROS_TYPE( uint8,  octave_ocl_uint8_matrix )
  OCL_ZEROS_TYPE( uint16, octave_ocl_uint16_matrix )
  OCL_ZEROS_TYPE( uint32, octave_ocl_uint32_matrix )
  OCL_ZEROS_TYPE( uint64, octave_ocl_uint64_matrix )
    ocl_error ("ocl_zeros: 'class' must be a data type as for 'zeros'"); // default case after last "else"

#undef OCL_ZEROS_TYPE

  return retval;
}


// ---------- the octave entry point to the 'ocl_eye' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_eye", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_eye", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_eye, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} ocl_eye (@var{n}) \n\
@deftypefnx {Loadable Function} {} ocl_eye (@var{m}, @var{n}) \n\
@deftypefnx {Loadable Function} {} ocl_eye ([@var{m} @var{n}]) \n\
@deftypefnx {Loadable Function} {} ocl_eye (@dots{}, @var{class}) \n\
\n\
Return an identity matrix as OCL matrix.  \n\
\n\
For details, see help for @code{eye}.  \n\
\n\
The OCL matrix is assembled on the OpenCL device.  \n\
\n\
@seealso{eye, ocl_ones, ocl_zeros, oclArray} \n\
@end deftypefn")
{
  octave_value_list retval;
  int nargin = args.length ();
  std::string val_class = "double";
  dim_vector dv (1,1);

  if ((nargin > 0) && (args (nargin-1).is_string ())) {
    val_class = args (nargin-1).string_value ();
    nargin--;
  }
  if (nargin < 1)
    ocl_error ("ocl_eye: too few arguments");

  if (args (0).is_real_matrix ()) {
    if (nargin > 1)
      ocl_error ("ocl_eye: too many arguments");
    Matrix m = args (0).matrix_value ();
    int ndim = m.numel ();
    if (ndim > 2)
      ocl_error ("ocl_eye: too many dimensions");
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++)
      dv (i) = m (i);
  } else {
    int ndim = nargin;
    if (ndim > 2)
      ocl_error ("ocl_eye: too many dimensions");
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++) {
      if (!args (i).is_real_scalar ())
        ocl_error ("ocl_eye: wrong argument type");
      dv (i) = args (i).scalar_value ();
    }
    if (nargin == 1)
      dv (1) = dv (0);
  }

  assure_installed_ocl_types ();

#define OCL_EYE_TYPE(C, T) \
  if (val_class == #C) { \
    T::array_type retval_array; \
    retval_array = T::array_type::eye (dv (0), dv (1)); \
    retval (0) = octave_value (new T (retval_array)); \
  } else

  OCL_EYE_TYPE( double, octave_ocl_matrix )
  OCL_EYE_TYPE( single, octave_ocl_float_matrix )
  OCL_EYE_TYPE( int8,   octave_ocl_int8_matrix )
  OCL_EYE_TYPE( int16,  octave_ocl_int16_matrix )
  OCL_EYE_TYPE( int32,  octave_ocl_int32_matrix )
  OCL_EYE_TYPE( int64,  octave_ocl_int64_matrix )
  OCL_EYE_TYPE( uint8,  octave_ocl_uint8_matrix )
  OCL_EYE_TYPE( uint16, octave_ocl_uint16_matrix )
  OCL_EYE_TYPE( uint32, octave_ocl_uint32_matrix )
  OCL_EYE_TYPE( uint64, octave_ocl_uint64_matrix )
    ocl_error ("ocl_eye: 'class' must be a data type as for 'eye'"); // default case after last "else"

#undef OCL_EYE_TYPE

  return retval;
}


// ---------- the octave entry point to the 'ocl_linspace' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_linspace", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_linspace", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_linspace, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} ocl_linspace (@var{base}, @var{limit}) \n\
@deftypefnx {Loadable Function} {} ocl_linspace (@var{base}, @var{limit}, @var{n}) \n\
\n\
Return an OCL row vector with @var{n} linearly spaced elements between \n\
@var{base} and @var{limit}.  \n\
\n\
For details, see help for @code{linspace}.  \n\
\n\
The OCL matrix is assembled on the OpenCL device.  \n\
\n\
@seealso{linspace, ocl_logspace, oclArray} \n\
@end deftypefn")
{
  octave_value_list retval;
  int nargin = args.length ();

  if (nargin < 2)
    ocl_error ("ocl_linspace: too few arguments");
  if (nargin > 3)
    ocl_error ("ocl_linspace: too many arguments");
  for (octave_idx_type i = 0; i < nargin; i++)
    if (!args (i).is_real_scalar ())
      ocl_error ("ocl_linspace: wrong argument type");

  assure_installed_ocl_types ();

  double base = args (0).scalar_value (), limit = args (1).scalar_value ();
  octave_ocl_matrix::array_type retval_array;

  if (nargin == 2)
    retval_array = octave_ocl_matrix::array_type::linspace (base, limit);
  else
    retval_array = octave_ocl_matrix::array_type::linspace (base, limit, args (2).scalar_value ());

  retval (0) = octave_value (new octave_ocl_matrix (retval_array));

  return retval;
}


// ---------- the octave entry point to the 'ocl_logspace' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_logspace", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_logspace", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_logspace, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} ocl_logspace (@var{a}, @var{b}) \n\
@deftypefnx {Loadable Function} {} ocl_logspace (@var{a}, @var{b}, @var{n}) \n\
\n\
Return an OCL row vector with @var{n} elements logarithmically spaced from \n\
10^@var{a} to 10^@var{b}.  \n\
\n\
For details, see help for @code{logspace}.  \n\
\n\
The OCL matrix is assembled on the OpenCL device.  \n\
\n\
@seealso{logspace, ocl_linspace, oclArray} \n\
@end deftypefn")
{
  octave_value_list retval;
  int nargin = args.length ();

  if (nargin < 2)
    ocl_error ("ocl_logspace: too few arguments");
  if (nargin > 3)
    ocl_error ("ocl_logspace: too many arguments");
  for (octave_idx_type i = 0; i < nargin; i++)
    if (!args (i).is_real_scalar ())
      ocl_error ("ocl_logspace: wrong argument type");

  assure_installed_ocl_types ();

  double a = args (0).scalar_value (), b = args (1).scalar_value ();
  octave_ocl_matrix::array_type retval_array;

  if (nargin == 2)
    retval_array = octave_ocl_matrix::array_type::logspace (a, b);
  else
    retval_array = octave_ocl_matrix::array_type::logspace (a, b, args (2).scalar_value ());

  retval (0) = octave_value (new octave_ocl_matrix (retval_array));

  return retval;
}
