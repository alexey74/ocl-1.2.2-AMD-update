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
 * A major part of this file is based on content from the following files,
 * originally published with GNU Octave 3.8.0, distributed under the same
 * license (as OCL, see above), with the following Copyright notices:
 *
 * ov-base-mat.cc:
 *   Copyright (C) 1996-2013 John W. Eaton
 *   Copyright (C) 2009-2010 VZLU Prague
 */

#include "ocl_ov_types.h"
#include "ocl_ov_matrix.h"
#include "ocl_array.h"
#include "ocl_lib.h"
#include <ops.h>


#if ! defined (DEFINE_TEMPLATE_OV_TYPEID_FUNCTIONS_AND_DATA)
// macro for template classes to be registered as a new octave type
// as replacement for DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA for non-template classes
#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
#define DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC(t, n, c)                 \
  template <> int t::t_id (-1);                                         \
  template <> const std::string t::t_name (n);                          \
  template <> const std::string t::c_name (c);                          \
  template <> void t::register_type (void)                              \
  {                                                                     \
    static t exemplar;                                                  \
    octave_value v (&exemplar, true);                                   \
    t_id = octave_value_typeinfo::register_type (t::t_name, t::c_name, v); \
  }
#else // for octave versions >= 4.4.0
#define DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC(t, n, c)         \
  template <> int t::t_id (-1);                                 \
  template <> const std::string t::t_name (n);                  \
  template <> const std::string t::c_name (c);                  \
  template <> void t::register_type (octave::type_info& ti)     \
  {                                                             \
    octave_value v (new t ());                                  \
    t_id = ti.register_type (t::t_name, t::c_name, v);          \
  }                                                             \
  template <> void t::register_type (void)                      \
  {                                                             \
    octave::type_info& type_info                                \
      = octave::__get_type_info__ (#t "::register_type");       \
                                                                \
    register_type (type_info);                                  \
  }
#endif
#else // DEFINE_TEMPLATE_OV_TYPEID_FUNCTIONS_AND_DATA is defined
#define DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC(t, n, c)         \
  DEFINE_TEMPLATE_OV_TYPEID_FUNCTIONS_AND_DATA(t, n, c)
#endif


static
bool
warning_save_oclmat (void)
{
  const char warning_id_oclmat_save[] = "Ocl:matrix_save";
  const char warn_str[] =
"saving context-dependent ocl matrix is ignored (saved as if empty). \n\
Use 'ocl_to_octave' to convert data of interest to octave matrix before saving.";

//  warning (warn_str);
  (*current_liboctave_warning_with_id_handler)
    (warning_id_oclmat_save, warn_str);
  return true;
}


static
bool
warning_load_oclmat (void)
{
//  const char warning_id_oclmat_load[] = "Ocl:matrix_load";
//  const char warn_str[] =
//"loading context-dependent ocl matrix is ignored, returning an empty ocl matrix.";
//  warning (warn_str);
//  (*current_liboctave_warning_with_id_handler)
//    (warning_id_oclmat_load, warn_str);
// nothing to do, since data is never saved (no need to skip)
  return true;
}


static
bool
oclmat_to_oclidxarray (const octave_value& ov, OclArray<ocl_idx_type>& i)
{
  int type_id = ov.type_id ();

#define OCL_CONV2IDX_TYPE(T) \
  if (type_id == T::static_type_id ()) { \
    T *mat = dynamic_cast<T *> (ov.internal_rep ()); \
    if (!mat) \
      return false; \
    i = mat->ocl_array_value ().as_index (); \
    return true; \
  } else

  if (type_id == octave_base_ocl_matrix< OclArray< ocl_idx_type > >::static_type_id ()) {
    octave_base_ocl_matrix< OclArray< ocl_idx_type > > *mat =
        dynamic_cast<octave_base_ocl_matrix< OclArray< ocl_idx_type > > *> (ov.internal_rep ());
    if (!mat)
      return false;
    i = mat->ocl_array_value ();
    return true;
  } else
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< double > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< float > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< Complex > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< FloatComplex > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_int8 > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_int16 > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_int32 > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_int64 > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_uint8 > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_uint16 > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_uint32 > > )
  OCL_CONV2IDX_TYPE( octave_base_ocl_matrix< OclArray< octave_uint64 > > )
    return false; // default case after last "else"

#undef OCL_CONV2IDX_TYPE
}


template <typename element_type>
static element_type
extract_scalar_value (const octave_value& ov)
{ return element_type (ov.scalar_value ()); }


template <>
Complex
extract_scalar_value (const octave_value& ov)
{ return ov.complex_value (); }


template <>
FloatComplex
extract_scalar_value (const octave_value& ov)
{ return ov.float_complex_value (); }


// ---------- octave_base_ocl_matrix<AT> members


template <typename AT>
int8NDArray
octave_base_ocl_matrix<AT>::int8_array_value (void) const
{ return int8NDArray (matrix.as_array ()); }


template <typename AT>
int16NDArray
octave_base_ocl_matrix<AT>::int16_array_value (void) const
{ return int16NDArray (matrix.as_array ()); }


template <typename AT>
int32NDArray
octave_base_ocl_matrix<AT>::int32_array_value (void) const
{ return int32NDArray (matrix.as_array ()); }


template <typename AT>
int64NDArray
octave_base_ocl_matrix<AT>::int64_array_value (void) const
{ return int64NDArray (matrix.as_array ()); }


template <typename AT>
uint8NDArray
octave_base_ocl_matrix<AT>::uint8_array_value (void) const
{ return uint8NDArray (matrix.as_array ()); }


template <typename AT>
uint16NDArray
octave_base_ocl_matrix<AT>::uint16_array_value (void) const
{ return uint16NDArray (matrix.as_array ()); }


template <typename AT>
uint32NDArray
octave_base_ocl_matrix<AT>::uint32_array_value (void) const
{ return uint32NDArray (matrix.as_array ()); }


template <typename AT>
uint64NDArray
octave_base_ocl_matrix<AT>::uint64_array_value (void) const
{ return uint64NDArray (matrix.as_array ()); }


template <typename AT>
NDArray
octave_base_ocl_matrix<AT>::array_value (bool) const
{ return NDArray (matrix.as_array ()); }


template <typename AT>
FloatNDArray
octave_base_ocl_matrix<AT>::float_array_value (bool) const
{ return FloatNDArray (matrix.as_array ()); }


template <typename AT>
Matrix
octave_base_ocl_matrix<AT>::matrix_value (bool) const
{ return Matrix (matrix.as_array ()); }


template <typename AT>
FloatMatrix
octave_base_ocl_matrix<AT>::float_matrix_value (bool) const
{ return FloatMatrix (matrix.as_array ()); }


template <typename AT>
ComplexNDArray
octave_base_ocl_matrix<AT>::complex_array_value (bool) const
{ return ComplexNDArray (matrix.as_array ()); }


template <typename AT>
FloatComplexNDArray
octave_base_ocl_matrix<AT>::float_complex_array_value (bool) const
{ return FloatComplexNDArray (matrix.as_array ()); }


template <typename AT>
ComplexMatrix
octave_base_ocl_matrix<AT>::complex_matrix_value (bool) const
{ return ComplexMatrix (matrix.as_array ()); }


template <typename AT>
FloatComplexMatrix
octave_base_ocl_matrix<AT>::float_complex_matrix_value (bool) const
{ return FloatComplexMatrix (matrix.as_array ()); }

// type coercion

template <typename AT>
static octave_base_value *
default_numeric_conversion_function (const octave_base_value& a)
{
  const octave_base_ocl_matrix<AT>& v = dynamic_cast<const octave_base_ocl_matrix<AT>&> (a);

  return new octave_matrix (v.array_value ());
}

template <typename AT>
octave_base_value::type_conv_info
octave_base_ocl_matrix<AT>::numeric_conversion_function () const
{
  return octave_base_value::type_conv_info (default_numeric_conversion_function<AT>,
                                            octave_matrix::static_type_id ());
}

//

template <typename AT>
octave_value
octave_base_ocl_matrix<AT>::subsref (const std::string& type,
                                    const std::list<octave_value_list>& idx)
{
  octave_value retval;

  if (type[0] == '(')
    retval = do_index_op (idx.front ());
  else {
    octave_stdout << type_name ().c_str () << " cannot be indexed with " << type[0] << "\n";
    ocl_error ("indexing error");
  }

  return retval.next_subsref (type, idx);
}


template <typename AT>
octave_value
octave_base_ocl_matrix<AT>::subsasgn (const std::string& type,
                                     const std::list<octave_value_list>& idx,
                                     const octave_value& rhs)
{
  octave_value retval;

  if (type[0] == '(') {
    if (type.length () == 1)
      retval = numeric_assign (type, idx, rhs);
    else {
      octave_stdout << "in indexed assignment of " << type_name ().c_str () << ", last lhs index must be ()\n";
      ocl_error ("indexing error");
    }
  } else {
    octave_stdout << type_name ().c_str () << " cannot be indexed with " << type[0] << "\n";
    ocl_error ("indexing error");
  }

  return retval;
}


template <typename AT>
octave_base_ocl_matrix<AT> *
octave_base_ocl_matrix<AT>::ocl_index_op (const octave_value_list& idx,
                                         bool resize_ok)
{
  octave_base_ocl_matrix<AT> *retval = 0;

  octave_idx_type n_idx = idx.length ();

  bool is_ocllogicidx = false;
  bool is_oclidx = false;
  AT ocllogicidx;
  OclArray<ocl_idx_type> oclidx;
  if (n_idx == 1) {
    octave_base_value *arg0_rep = idx (0).internal_rep ();
    if (arg0_rep->type_id () == static_type_id ()) {
      octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
      if (ovom != 0) {
        ocllogicidx = ovom->matrix;
        is_ocllogicidx = ocllogicidx.islogical ();
      }
    }
    if (! is_ocllogicidx)
      is_oclidx = oclmat_to_oclidxarray (idx (0), oclidx);
  }

  switch (n_idx) {
    case 0:
      retval = new octave_base_ocl_matrix<AT> (matrix);
      break;

    case 1:
    {
      if (is_ocllogicidx) {
        octave_stdout << "logical indexing is not possible\n";
        ocl_error ("indexing error");
      } else if (is_oclidx) {
        oclidx -= 1; // this is where the conversion one-based to zero-based takes place
        retval = new octave_base_ocl_matrix<AT> (matrix.index (oclidx));
      } else {
        idx_vector i = idx (0).index_vector (); // this is where the conversion one-based to zero-based takes place
        retval = new octave_base_ocl_matrix<AT> (matrix.index (i));
      }
      break;
    }

    case 2:
    {
      idx_vector i = idx (0).index_vector ();
      idx_vector j = idx (1).index_vector ();
      retval = new octave_base_ocl_matrix<AT> (matrix.index (i, j));
      break;
    }

    default:
    {
      Array<idx_vector> idx_vec (dim_vector (n_idx, 1));
      for (octave_idx_type i = 0; i < n_idx; i++) {
        idx_vec(i) = idx(i).index_vector ();
      }
      retval = new octave_base_ocl_matrix<AT> (matrix.index (idx_vec));
      break;
    }
  }

  if (retval == 0)
    retval = new octave_base_ocl_matrix<AT> ();

  return retval;
}


template <typename AT>
void
octave_base_ocl_matrix<AT>::assign (const octave_value_list& idx, typename AT::element_type rhs)
{
  octave_idx_type n_idx = idx.length ();

  bool is_ocllogicidx = false;
  bool is_oclidx = false;
  AT ocllogicidx;
  OclArray<ocl_idx_type> oclidx;
  if (n_idx == 1) {
    octave_base_value *arg0_rep = idx (0).internal_rep ();
    if (arg0_rep->type_id () == static_type_id ()) {
      octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
      if (ovom != 0) {
        ocllogicidx = ovom->matrix;
        is_ocllogicidx = ocllogicidx.islogical ();
      }
    }
    if (! is_ocllogicidx)
      is_oclidx = oclmat_to_oclidxarray (idx (0), oclidx);
  }

  switch (n_idx) {
    case 0:
      panic_impossible ();
      break;

    case 1:
    {
      if (is_ocllogicidx) {
        matrix.assign_logical (ocllogicidx, rhs);
      } else if (is_oclidx) {
        oclidx -= 1; // this is where the conversion one-based to zero-based takes place
        matrix.assign (oclidx, rhs);
      } else {
        idx_vector i = idx (0).index_vector (); // this is where the conversion one-based to zero-based takes place
        matrix.assign (i, rhs);
      }
      break;
    }

    case 2:
    {
      idx_vector i = idx (0).index_vector ();
      idx_vector j = idx (1).index_vector ();
      matrix.assign (i, j, rhs);
      break;
    }

    default:
    {
      Array<idx_vector> idx_vec (dim_vector (n_idx, 1));
      for (octave_idx_type i = 0; i < n_idx; i++) {
        idx_vec(i) = idx(i).index_vector ();
      }
      matrix.assign (idx_vec, rhs);
      break;
    }
  }
}


template <typename AT>
void
octave_base_ocl_matrix<AT>::assign (const octave_value_list& idx, const AT& rhs)
{
  octave_idx_type n_idx = idx.length ();

  bool is_ocllogicidx = false;
  bool is_oclidx = false;
  AT ocllogicidx;
  OclArray<ocl_idx_type> oclidx;
  if (n_idx == 1) {
    octave_base_value *arg0_rep = idx (0).internal_rep ();
    if (arg0_rep->type_id () == static_type_id ()) {
      octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
      if (ovom != 0) {
        ocllogicidx = ovom->matrix;
        is_ocllogicidx = ocllogicidx.islogical ();
      }
    }
    if (! is_ocllogicidx)
      is_oclidx = oclmat_to_oclidxarray (idx (0), oclidx);
  }

  switch (n_idx) {
    case 0:
      panic_impossible ();
      break;

    case 1:
    {
      if (is_ocllogicidx) {
        octave_stdout << "logically indexed assignment is only possible with a scalar value\n";
        ocl_error ("indexing error");
      } else if (is_oclidx) {
        oclidx -= 1; // this is where the conversion one-based to zero-based takes place
        matrix.assign (oclidx, rhs);
      } else {
        idx_vector i = idx (0).index_vector (); // this is where the conversion one-based to zero-based takes place
        matrix.assign (i, rhs);
      }
      break;
    }

    case 2:
    {
      idx_vector i = idx (0).index_vector ();
      idx_vector j = idx (1).index_vector ();
      matrix.assign (i, j, rhs);
      break;
    }

    default:
    {
      Array<idx_vector> idx_vec (dim_vector (n_idx, 1));
      for (octave_idx_type i = 0; i < n_idx; i++) {
        idx_vec(i) = idx(i).index_vector ();
      }
      matrix.assign (idx_vec, rhs);
      break;
    }
  }
}


template <typename AT>
octave_value
octave_base_ocl_matrix<AT>::map (octave_base_value::unary_mapper_t umap) const
{
  return new octave_base_ocl_matrix<AT> (matrix.map (umap));
}


// saving ocl matrix data does not really make sense, because:
// - the data's existence or reachability is (OpenCL) context-dependent
// - the data cannot easily be re-loaded without assumptions about (OpenCL) resources
// hence, the 'save_...' functions only exist to not abort while saving the whole workspace etc.

// similarly, for 'load_...' functions; there, the additional complication
// is that the user-types have to be known to octave (e.g. at least 'pkg load ocl' is needed)
// prior to loading, or else the error "wrong type argument '<unknown type>'" is reported

template <typename AT>
bool
octave_base_ocl_matrix<AT>::save_ascii (std::ostream& os)
{
  return warning_save_oclmat ();
}


template <typename AT>
bool
octave_base_ocl_matrix<AT>::load_ascii (std::istream& is)
{
  return warning_load_oclmat ();
}


template <typename AT>
bool
#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
octave_base_ocl_matrix<AT>::save_binary (std::ostream& os, bool& save_as_floats)
#else // for octave versions >= 6.1.0
octave_base_ocl_matrix<AT>::save_binary (std::ostream& os, bool save_as_floats)
#endif
{
  return warning_save_oclmat ();
}


template <typename AT>
bool
octave_base_ocl_matrix<AT>::load_binary (std::istream& is, bool swap, octave::mach_info::float_format fmt)
{
  return warning_load_oclmat ();
}


template <typename AT>
bool
octave_base_ocl_matrix<AT>::save_hdf5 (octave_hdf5_id loc_id, const char *name, bool save_as_floats)
{
  return warning_save_oclmat ();
}


template <typename AT>
bool
octave_base_ocl_matrix<AT>::load_hdf5 (octave_hdf5_id loc_id, const char *name)
{
  return warning_load_oclmat ();
}


// ---------- octave_base_ocl_matrix<AT> static members


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::as_index (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 1) || (nargin != 1))
    ocl_error ("wrong number or type of arguments");

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom == 0)
    return octave_value ();

  return ovom->as_index ();
}


#define DEFINE_METHOD(METHOD) \
  template <typename AT> \
  octave_value_list \
  octave_base_ocl_matrix<AT>::METHOD (const octave_value_list& args, int nargout) \
  { \
    int nargin = args.length (); \
    if ((nargout > 1) || (nargin > 2) || ((nargin > 1) && (!args(1).is_real_scalar ()))) \
      ocl_error ("wrong number or type of arguments"); \
      \
    int dim = -1; \
    if (nargin > 1) \
      dim = args(1).scalar_value () - 1; \
      \
    octave_base_value *arg0_rep = args(0).internal_rep (); \
    if (arg0_rep->type_id () != static_type_id ()) \
      ocl_error ("wrong argument type"); \
    octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep); \
    if (ovom == 0) \
      return octave_value (); \
      \
    return ovom->METHOD (dim); \
  }

DEFINE_METHOD(sum)
DEFINE_METHOD(sumsq)
DEFINE_METHOD(prod)
DEFINE_METHOD(mean)
DEFINE_METHOD(meansq)
DEFINE_METHOD(cumsum)
DEFINE_METHOD(cumprod)
DEFINE_METHOD(findfirst)
DEFINE_METHOD(findlast)

#undef DEFINE_METHOD


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::std (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 1) || (nargin > 3) ||
      ((nargin > 1) && (!args(1).is_real_scalar ())) ||
      ((nargin > 2) && (!args(2).is_real_scalar ())))
    ocl_error ("wrong number or type of arguments");

  int opt = 0;
  if (nargin > 1)
    opt = args(1).scalar_value ();

  int dim = -1;
  if (nargin > 2)
    dim = args(2).scalar_value () - 1;

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom == 0)
    return octave_value ();

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  if (ovom->is_integer_type ())
#else // for octave versions >= 4.4.0
  if (ovom->isinteger ())
#endif
    ocl_error ("wrong argument type");

  return ovom->std (opt, dim);
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::max (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 2) || (nargin > 3) ||
      ((nargin > 2) && (!args(2).is_real_scalar ())) ||
      ((nargin == 2) && (nargout == 2)))
    ocl_error ("wrong number or type of arguments");

  int dim = -1;
  if (nargin > 2)
    dim = args(2).scalar_value () - 1;

  int i0 = 0, i1 = 1;
  if ((nargin == 2) && (args(0).is_scalar_type ())) { // max (scalar, oclmat)
    i0 = 1;
    i1 = 0;
  }

  octave_base_value *arg0_rep = args(i0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom0 = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom0 == 0)
    return octave_value ();

  if (nargin != 2) { // search along a dimension of arg(0), possibly ignoring arg(1)
    if (nargout < 2)
      return ovom0->max (dim);
    else {
      octave_value result, indices;
      result = ovom0->max (indices, dim);
      octave_value_list retval;
      retval (0) = result;
      retval (1) = indices;
      return retval;
    }
  } else { // compare arg(0) and arg(1)
    if (args(i1).is_scalar_type ()) {
      element_type v = extract_scalar_value<element_type> (args(i1));
      return ovom0->max2 (v);
    } else if (args(i1).type_id () == args(i0).type_id ()) {
      octave_base_value *arg1_rep = args(i1).internal_rep ();
      if (arg1_rep->type_id () != static_type_id ())
        ocl_error ("wrong argument type");
      octave_base_ocl_matrix<AT> *ovom1 = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg1_rep);
      if (ovom1 == 0)
        return octave_value ();
      return ovom0->max2 (*ovom1);
    } else
      ocl_error ("wrong argument type");
  }
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::min (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 2) || (nargin > 3) ||
      ((nargin > 2) && (!args(2).is_real_scalar ())) ||
      ((nargin == 2) && (nargout == 2)))
    ocl_error ("wrong number or type of arguments");

  int dim = -1;
  if (nargin > 2)
    dim = args(2).scalar_value () - 1;

  int i0 = 0, i1 = 1;
  if ((nargin == 2) && (args(0).is_scalar_type ())) { // min (scalar, oclmat)
    i0 = 1;
    i1 = 0;
  }

  octave_base_value *arg0_rep = args(i0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom0 = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom0 == 0)
    return octave_value ();

  if (nargin != 2) { // search along a dimension of arg(0), possibly ignoring arg(1)
    if (nargout < 2)
      return ovom0->min (dim);
    else {
      octave_value result, indices;
      result = ovom0->min (indices, dim);
      octave_value_list retval;
      retval (0) = result;
      retval (1) = indices;
      return retval;
    }
  } else { // compare arg(0) and arg(1)
    if (args(i1).is_scalar_type ()) {
      element_type v = extract_scalar_value<element_type> (args(i1));
      return ovom0->min2 (v);
    } else if (args(i1).type_id () == args(i0).type_id ()) {
      octave_base_value *arg1_rep = args(i1).internal_rep ();
      if (arg1_rep->type_id () != static_type_id ())
        ocl_error ("wrong argument type");
      octave_base_ocl_matrix<AT> *ovom1 = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg1_rep);
      if (ovom1 == 0)
        return octave_value ();
      return ovom0->min2 (*ovom1);
    } else
      ocl_error ("wrong argument type");
  }
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::cummax (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 2) || (nargin > 2) || ((nargin > 1) && (!args(1).is_real_scalar ())))
    ocl_error ("wrong number or type of arguments");

  int dim = -1;
  if (nargin > 1)
    dim = args(1).scalar_value () - 1;

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom == 0)
    return octave_value ();

  if (nargout < 2)
    return ovom->cummax (dim);
  else {
    octave_value result, indices;
    result = ovom->cummax (indices, dim);
    octave_value_list retval;
    retval (0) = result;
    retval (1) = indices;
    return retval;
  }
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::cummin (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 2) || (nargin > 2) || ((nargin > 1) && (!args(1).is_real_scalar ())))
    ocl_error ("wrong number or type of arguments");

  int dim = -1;
  if (nargin > 1)
    dim = args(1).scalar_value () - 1;

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom == 0)
    return octave_value ();

  if (nargout < 2)
    return ovom->cummin (dim);
  else {
    octave_value result, indices;
    result = ovom->cummin (indices, dim);
    octave_value_list retval;
    retval (0) = result;
    retval (1) = indices;
    return retval;
  }
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::atan2 (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 1) || (nargin != 2) ||
      (args(1).type_id () != args(0).type_id ()))
    ocl_error ("wrong number or type of arguments");

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom0 = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom0 == 0)
    return octave_value ();

  octave_base_value *arg1_rep = args(1).internal_rep ();
  if (arg1_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom1 = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg1_rep);
  if (ovom1 == 0)
    return octave_value ();

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  if (ovom0->is_integer_type ())
#else // for octave versions >= 4.4.0
  if (ovom0->isinteger ())
#endif
    ocl_error ("wrong argument type");

  return ovom0->atan2 (*ovom1);
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::ndgrid (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if (((nargin == 1) && (nargout != 2)) ||
      ((nargin != 1) && (nargout != nargin)))
    ocl_error ("wrong number of arguments");

  int type_id = args(0).type_id ();
  if (type_id != static_type_id ())
    ocl_error ("wrong argument type");
  for (int i = 0; i < nargin; i++) {
    octave_value ai = args(i);
    if ((ai.type_id () != type_id) || (ai.ndims () > 2) || ((ai.rows () != 1) && (ai.columns () != 1)))
      ocl_error ("all input arguments must be vectors of the same ocl matrix type");
  }

  std::vector< array_type > array_list (nargin);
  for (int i = 0; i < nargin; i++) {
    octave_base_value *argi_rep = args(i).internal_rep ();
    octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (argi_rep);
    if (ovom == 0)
      return octave_value ();
    array_list [i] = ovom->ocl_array_value ();
  }

  std::vector< array_type > result_list;
  result_list = array_type::ndgrid (array_list);

  octave_value_list retval;
  for (int i = 0; i < nargout; i++)
    retval (i) = new octave_base_ocl_matrix<AT> (result_list [i]);

  return retval;
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::meshgrid (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if (((nargin == 1) && (nargout != 2)) ||
      ((nargin != 1) && (nargout != nargin)))
    ocl_error ("wrong number of arguments");

  int type_id = args(0).type_id ();
  if (type_id != static_type_id ())
    ocl_error ("wrong argument type");
  for (int i = 0; i < nargin; i++) {
    octave_value ai = args(i);
    if ((ai.type_id () != type_id) || (ai.ndims () > 2) || ((ai.rows () != 1) && (ai.columns () != 1)))
      ocl_error ("all input arguments must be vectors of the same ocl matrix type");
  }

  std::vector< array_type > array_list (nargin);
  for (int i = 0; i < nargin; i++) {
    octave_base_value *argi_rep = args(i).internal_rep ();
    octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (argi_rep);
    if (ovom == 0)
      return octave_value ();
    array_list [i] = ovom->ocl_array_value ();
  }

  std::vector< array_type > result_list;
  result_list = array_type::meshgrid (array_list);

  octave_value_list retval;
  for (int i = 0; i < nargout; i++)
    retval (i) = new octave_base_ocl_matrix<AT> (result_list [i]);

  return retval;
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::repmat (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if (nargin < 2)
    ocl_error ("too few arguments");

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_base_ocl_matrix<AT> *ovom = dynamic_cast< octave_base_ocl_matrix<AT> *> (arg0_rep);
  if (ovom == 0)
    return octave_value ();

  dim_vector dv (1,1);
  if (args (1).is_real_matrix ()) {
    if (nargin > 2)
      ocl_error ("too many arguments");
    Matrix m = args (1).matrix_value ();
    int ndim = m.numel ();
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++)
      dv (i) = m (i);
  } else {
    int ndim = nargin-1;
    dv = dv.redim (ndim);
    for (octave_idx_type i = 0; i < ndim; i++) {
      if (!args (1+i).is_real_scalar ())
        ocl_error ("wrong argument type");
      dv (i) = args (1+i).scalar_value ();
    }
    if (nargin == 2)
      dv (1) = dv (0);
  }

  return octave_value (new octave_base_ocl_matrix<AT> (ovom->ocl_array_value ().repmat (dv)));
}


template <typename AT>
octave_value_list
octave_base_ocl_matrix<AT>::complex (const octave_value_list& args, int nargout)
{
  ocl_error ("complex: invalid conversion");
}


// ---------- octave_base_ocl_matrix<AT> specializations


#ifdef DEFINE_OCTAVE_ALLOCATOR
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< double > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< float > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< Complex > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< FloatComplex > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_int8 > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_int16 > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_int32 > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_int64 > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_uint8 > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_uint16 > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_uint32 > >);
DEFINE_OCTAVE_ALLOCATOR (octave_base_ocl_matrix< OclArray< octave_uint64 > >);
#endif


DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< double > >, "ocl matrix", "ocl_double");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< float > >, "ocl float matrix", "ocl_single");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< Complex > >, "ocl complex matrix", "ocl_double");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< FloatComplex > >, "ocl float complex matrix", "ocl_single");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_int8 > >, "ocl int8 matrix", "ocl_int8");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_int16 > >, "ocl int16 matrix", "ocl_int16");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_int32 > >, "ocl int32 matrix", "ocl_int32");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_int64 > >, "ocl int64 matrix", "ocl_int64");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_uint8 > >, "ocl uint8 matrix", "ocl_uint8");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_uint16 > >, "ocl uint16 matrix", "ocl_uint16");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_uint32 > >, "ocl uint32 matrix", "ocl_uint32");
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA_TC (octave_base_ocl_matrix< OclArray< octave_uint64 > >, "ocl uint64 matrix", "ocl_uint64");


template <>
int8NDArray
octave_base_ocl_matrix<OclArray< Complex > >::int8_array_value (void) const
{ ocl_error ("invalid conversion"); return int8NDArray (); }


template <>
int16NDArray
octave_base_ocl_matrix<OclArray< Complex > >::int16_array_value (void) const
{ ocl_error ("invalid conversion"); return int16NDArray (); }


template <>
int32NDArray
octave_base_ocl_matrix<OclArray< Complex > >::int32_array_value (void) const
{ ocl_error ("invalid conversion"); return int32NDArray (); }


template <>
int64NDArray
octave_base_ocl_matrix<OclArray< Complex > >::int64_array_value (void) const
{ ocl_error ("invalid conversion"); return int64NDArray (); }


template <>
uint8NDArray
octave_base_ocl_matrix<OclArray< Complex > >::uint8_array_value (void) const
{ ocl_error ("invalid conversion"); return uint8NDArray (); }


template <>
uint16NDArray
octave_base_ocl_matrix<OclArray< Complex > >::uint16_array_value (void) const
{ ocl_error ("invalid conversion"); return uint16NDArray (); }


template <>
uint32NDArray
octave_base_ocl_matrix<OclArray< Complex > >::uint32_array_value (void) const
{ ocl_error ("invalid conversion"); return uint32NDArray (); }


template <>
uint64NDArray
octave_base_ocl_matrix<OclArray< Complex > >::uint64_array_value (void) const
{ ocl_error ("invalid conversion"); return uint64NDArray (); }


template <>
NDArray
octave_base_ocl_matrix<OclArray< Complex > >::array_value (bool) const
{ ocl_error ("invalid conversion"); return NDArray (); }


template <>
FloatNDArray
octave_base_ocl_matrix<OclArray< Complex > >::float_array_value (bool) const
{ ocl_error ("invalid conversion"); return FloatNDArray (); }


template <>
Matrix
octave_base_ocl_matrix<OclArray< Complex > >::matrix_value (bool) const
{ ocl_error ("invalid conversion"); return Matrix (); }


template <>
FloatMatrix
octave_base_ocl_matrix<OclArray< Complex > >::float_matrix_value (bool) const
{ ocl_error ("invalid conversion"); return FloatMatrix (); }


template <>
int8NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::int8_array_value (void) const
{ ocl_error ("invalid conversion"); return int8NDArray (); }


template <>
int16NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::int16_array_value (void) const
{ ocl_error ("invalid conversion"); return int16NDArray (); }


template <>
int32NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::int32_array_value (void) const
{ ocl_error ("invalid conversion"); return int32NDArray (); }


template <>
int64NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::int64_array_value (void) const
{ ocl_error ("invalid conversion"); return int64NDArray (); }


template <>
uint8NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::uint8_array_value (void) const
{ ocl_error ("invalid conversion"); return uint8NDArray (); }


template <>
uint16NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::uint16_array_value (void) const
{ ocl_error ("invalid conversion"); return uint16NDArray (); }


template <>
uint32NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::uint32_array_value (void) const
{ ocl_error ("invalid conversion"); return uint32NDArray (); }


template <>
uint64NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::uint64_array_value (void) const
{ ocl_error ("invalid conversion"); return uint64NDArray (); }


template <>
NDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::array_value (bool) const
{ ocl_error ("invalid conversion"); return NDArray (); }


template <>
FloatNDArray
octave_base_ocl_matrix<OclArray< FloatComplex > >::float_array_value (bool) const
{ ocl_error ("invalid conversion"); return FloatNDArray (); }


template <>
Matrix
octave_base_ocl_matrix<OclArray< FloatComplex > >::matrix_value (bool) const
{ ocl_error ("invalid conversion"); return Matrix (); }


template <>
FloatMatrix
octave_base_ocl_matrix<OclArray< FloatComplex > >::float_matrix_value (bool) const
{ ocl_error ("invalid conversion"); return FloatMatrix (); }


template <>
octave_value
octave_base_ocl_matrix<OclArray<Complex> >::map (octave_base_value::unary_mapper_t umap) const
{
  switch (umap)
  {
    case octave_base_value::umap_real:
    case octave_base_value::umap_imag:
    case octave_base_value::umap_abs:
    case octave_base_value::umap_angle:
    case octave_base_value::umap_arg:
    case octave_base_value::umap_isfinite:
    case octave_base_value::umap_isinf:
    case octave_base_value::umap_isnan:
      return new octave_base_ocl_matrix<OclArray<double> > (matrix.map_c2r<double> (umap));

    default:
      return new octave_base_ocl_matrix<OclArray<Complex> > (matrix.map (umap));
  }
}


template <>
octave_value
octave_base_ocl_matrix<OclArray<FloatComplex> >::map (octave_base_value::unary_mapper_t umap) const
{
  switch (umap)
  {
    case octave_base_value::umap_real:
    case octave_base_value::umap_imag:
    case octave_base_value::umap_abs:
    case octave_base_value::umap_angle:
    case octave_base_value::umap_arg:
    case octave_base_value::umap_isfinite:
    case octave_base_value::umap_isinf:
    case octave_base_value::umap_isnan:
      return new octave_base_ocl_matrix<OclArray<float> > (matrix.map_c2r<float> (umap));

    default:
      return new octave_base_ocl_matrix<OclArray<FloatComplex> > (matrix.map (umap));
  }
}


template <>
octave_value_list
octave_base_ocl_matrix<OclArray<double> >::complex (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 1) || (nargin < 1) || (nargin > 2) ||
      (args(0).type_id () != octave_ocl_matrix::static_type_id ()))
    ocl_error ("wrong number or type of arguments");
  if ((nargin == 2) &&
      (args(1).type_id () != octave_ocl_matrix::static_type_id ()))
    ocl_error ("wrong number or type of arguments");

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_ocl_matrix *ovom0 = dynamic_cast< octave_ocl_matrix *> (arg0_rep);
  if (ovom0 == 0)
    return octave_value ();

  if (nargin == 1)
    return octave_value (new octave_ocl_complex_matrix (OclComplexNDArray (ovom0->matrix_ref ())));

  octave_base_value *arg1_rep = args(1).internal_rep ();
  if (arg1_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_ocl_matrix *ovom1 = dynamic_cast< octave_ocl_matrix *> (arg1_rep);
  if (ovom1 == 0)
    return octave_value ();

  return octave_value (new octave_ocl_complex_matrix (OclComplexNDArray (ovom0->matrix_ref (), ovom1->matrix_ref ())));
}


template <>
octave_value_list
octave_base_ocl_matrix<OclArray<float> >::complex (const octave_value_list& args, int nargout)
{
  int nargin = args.length ();
  if ((nargout > 1) || (nargin < 1) || (nargin > 2) ||
      (args(0).type_id () != octave_ocl_float_matrix::static_type_id ()))
    ocl_error ("wrong number or type of arguments");
  if ((nargin == 2) &&
      (args(1).type_id () != octave_ocl_float_matrix::static_type_id ()))
    ocl_error ("wrong number or type of arguments");

  octave_base_value *arg0_rep = args(0).internal_rep ();
  if (arg0_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_ocl_float_matrix *ovom0 = dynamic_cast< octave_ocl_float_matrix *> (arg0_rep);
  if (ovom0 == 0)
    return octave_value ();

  if (nargin == 1)
    return octave_value (new octave_ocl_float_complex_matrix (OclFloatComplexNDArray (ovom0->matrix_ref ())));

  octave_base_value *arg1_rep = args(1).internal_rep ();
  if (arg1_rep->type_id () != static_type_id ())
    ocl_error ("wrong argument type");
  octave_ocl_float_matrix *ovom1 = dynamic_cast< octave_ocl_float_matrix *> (arg1_rep);
  if (ovom1 == 0)
    return octave_value ();

  return octave_value (new octave_ocl_float_complex_matrix (OclFloatComplexNDArray (ovom0->matrix_ref (), ovom1->matrix_ref ())));
}


// ---------- octave_base_ocl_matrix<AT> instantiations


#define INSTANTIATE_OCLBASEMATRIX( AT ) \
  template class octave_base_ocl_matrix< OclArray<AT> >;


INSTANTIATE_OCLBASEMATRIX (octave_int8  );
INSTANTIATE_OCLBASEMATRIX (octave_int16 );
INSTANTIATE_OCLBASEMATRIX (octave_int32 );
INSTANTIATE_OCLBASEMATRIX (octave_int64 );
INSTANTIATE_OCLBASEMATRIX (octave_uint8 );
INSTANTIATE_OCLBASEMATRIX (octave_uint16);
INSTANTIATE_OCLBASEMATRIX (octave_uint32);
INSTANTIATE_OCLBASEMATRIX (octave_uint64);
INSTANTIATE_OCLBASEMATRIX (float        );
INSTANTIATE_OCLBASEMATRIX (double       );
INSTANTIATE_OCLBASEMATRIX (FloatComplex );
INSTANTIATE_OCLBASEMATRIX (Complex      );


// ---------- ocl matrix constructor functions


static std::string ocl_mat_help_text =
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {@var{ocl_mat} =} ocl_double (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_single (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_int8 (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_int16 (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_int32 (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_int64 (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_uint8 (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_uint16 (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_uint32 (@var{octave_mat}) \n\
@deftypefnx {Loadable Function} {@var{ocl_mat} =} ocl_uint64 (@var{octave_mat}) \n\
\n\
Construct an OCL matrix of specific type from an octave matrix.  \n\
\n\
All the above constructor functions take as input a conventional numeric \n\
octave matrix @var{octave_mat} (actually an N-dimensional array) of any \n\
numeric data type.  The constructors create a new OCL matrix @var{ocl_mat} \n\
(as an N-dimensional array) of the specified numeric data type, allocate \n\
storage space on the OpenCL device hardware, and copy the octave data \n\
into the OpenCL device memory.  The data then remains in device memory \n\
until the OCL matrix is cleared from the octave workspace (or as long as the \n\
OpenCL context exists).  \n\
@code{ocl_double} and @code{ocl_single} allow operation on real and complex data.  \n\
\n\
Copying data \n\
from an OCL matrix back to an octave matrix is possible via the corresponding \n\
standard type casting function (e.g., @code{double}, @code{single}, @code{int16}).  \n\
\n\
For further explanation on using OCL matrices and example code, see @code{oclArray}. \n\
\n\
The constructor functions automatically assure that the OpenCL library is \n\
loaded (see @code{ocl_lib}) and that an OpenCL context is created with an \n\
OpenCL device (see @code{ocl_context}).  \n\
\n\
@seealso{oclArray, ocl_tests, ocl_program, ocl_context, ocl_lib} \n\
@end deftypefn";


#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
#define ISREAL is_real_type
#else // for octave versions >= 4.4.0
#define ISREAL isreal
#endif


#define DEFINE_OCL_MAT_CONSTRUCTOR(CLASS, OCTAVE_VALUE_OCL_MATRIX_TYPE, OCTAVE_ARRAY_TYPE, OCTAVE_ARRAY_FCN) \
  DEFUN_DLD (CLASS, args, nargout, ocl_mat_help_text) \
  { \
    octave_value_list retval; \
    int nargin = args.length (); \
   \
    if ((nargin != 1) || (!args (0).ISREAL ())) { \
      print_usage (); \
      return retval; \
    } \
   \
    assure_installed_ocl_types (); \
   \
    OCTAVE_ARRAY_TYPE a = args (0).OCTAVE_ARRAY_FCN (); \
    retval = octave_value (new OCTAVE_VALUE_OCL_MATRIX_TYPE (a)); \
   \
    return retval; \
  }


DEFINE_OCL_MAT_CONSTRUCTOR( ocl_int8,   octave_ocl_int8_matrix,   int8NDArray,   int8_array_value   )
DEFINE_OCL_MAT_CONSTRUCTOR( ocl_int16,  octave_ocl_int16_matrix,  int16NDArray,  int16_array_value  )
DEFINE_OCL_MAT_CONSTRUCTOR( ocl_int32,  octave_ocl_int32_matrix,  int32NDArray,  int32_array_value  )
DEFINE_OCL_MAT_CONSTRUCTOR( ocl_int64,  octave_ocl_int64_matrix,  int64NDArray,  int64_array_value  )
DEFINE_OCL_MAT_CONSTRUCTOR( ocl_uint8,  octave_ocl_uint8_matrix,  uint8NDArray,  uint8_array_value  )
DEFINE_OCL_MAT_CONSTRUCTOR( ocl_uint16, octave_ocl_uint16_matrix, uint16NDArray, uint16_array_value )
DEFINE_OCL_MAT_CONSTRUCTOR( ocl_uint32, octave_ocl_uint32_matrix, uint32NDArray, uint32_array_value )
DEFINE_OCL_MAT_CONSTRUCTOR( ocl_uint64, octave_ocl_uint64_matrix, uint64NDArray, uint64_array_value )


DEFUN_DLD (ocl_double, args, nargout, ocl_mat_help_text)
{
  octave_value_list retval;
  int nargin = args.length ();

  if (nargin != 1) {
    print_usage ();
    return retval;
  }

  assure_installed_ocl_types ();

  if (args (0).ISREAL ()) {
    NDArray a = args (0).array_value ();
    retval = octave_value (new octave_ocl_matrix (a));
  } else {
    ComplexNDArray a = args (0).complex_array_value ();
    retval = octave_value (new octave_ocl_complex_matrix (a));
  }

  return retval;
}


DEFUN_DLD (ocl_single, args, nargout, ocl_mat_help_text)
{
  octave_value_list retval;
  int nargin = args.length ();

  if (nargin != 1) {
    print_usage ();
    return retval;
  }

  assure_installed_ocl_types ();

  if (args (0).ISREAL ()) {
    FloatNDArray a = args (0).float_array_value ();
    retval = octave_value (new octave_ocl_float_matrix (a));
  } else {
    FloatComplexNDArray a = args (0).float_complex_array_value ();
    retval = octave_value (new octave_ocl_float_complex_matrix (a));
  }

  return retval;
}


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_double", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_double", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_single", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_single", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_int8", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_int8", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_int16", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_int16", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_int32", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_int32", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_int64", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_int64", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_uint8", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_uint8", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_uint16", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_uint16", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_uint32", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_uint32", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_uint64", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_uint64", "ocl_bin.oct", "remove");


// ---------- ocl matrix member functions


#define OCL_MAT_METHOD(METHOD, OCLMAT_TYPE) \
  if (type_id == OCLMAT_TYPE::static_type_id ()) { \
    return OCLMAT_TYPE::METHOD (args, nargout); \
  } else


#define DEFINE_OCL_MAT_METHOD(METHOD) \
  DEFUN_DLD (CONCAT3(__ocl_mat_, METHOD, __), args, nargout, "internal OCL function") \
  { \
    octave_value_list retval; \
    int nargin = args.length (); \
    if (nargin < 1) \
      ocl_error ("too few arguments"); \
    int type_id = args(0).type_id (); \
    if (nargin > 1) { \
      int type_id2 = args(1).type_id (); \
      if (type_id2 > type_id) /* second argument has the more complex type */ \
        type_id = type_id2; \
    } \
    if (type_id == -1) \
      ocl_error ("unknown argument type"); \
     \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< double > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< float > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< Complex > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< FloatComplex > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_int8 > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_int16 > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_int32 > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_int64 > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_uint8 > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_uint16 > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_uint32 > > ) \
    OCL_MAT_METHOD( METHOD, octave_base_ocl_matrix< OclArray< octave_uint64 > > ) \
      ocl_error ("method arguments must contain ocl matrices consistently"); /* default case after last "else" */ \
     \
    return octave_value_list (); \
  }


DEFINE_OCL_MAT_METHOD(as_index)
DEFINE_OCL_MAT_METHOD(sum)
DEFINE_OCL_MAT_METHOD(sumsq)
DEFINE_OCL_MAT_METHOD(prod)
DEFINE_OCL_MAT_METHOD(mean)
DEFINE_OCL_MAT_METHOD(meansq)
DEFINE_OCL_MAT_METHOD(cumsum)
DEFINE_OCL_MAT_METHOD(cumprod)
DEFINE_OCL_MAT_METHOD(findfirst)
DEFINE_OCL_MAT_METHOD(findlast)
DEFINE_OCL_MAT_METHOD(std)
DEFINE_OCL_MAT_METHOD(max)
DEFINE_OCL_MAT_METHOD(min)
DEFINE_OCL_MAT_METHOD(cummax)
DEFINE_OCL_MAT_METHOD(cummin)
DEFINE_OCL_MAT_METHOD(atan2)
DEFINE_OCL_MAT_METHOD(ndgrid)
DEFINE_OCL_MAT_METHOD(meshgrid)
DEFINE_OCL_MAT_METHOD(repmat)
DEFINE_OCL_MAT_METHOD(complex)


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_as_index__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_as_index__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_sum__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_sum__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_sumsq__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_sumsq__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_prod__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_prod__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_mean__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_mean__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_meansq__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_meansq__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_cumsum__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_cumsum__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_cumprod__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_cumprod__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_findfirst__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_findfirst__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_findlast__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_findlast__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_std__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_std__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_max__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_max__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_min__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_min__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_cummax__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_cummax__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_cummin__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_cummin__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_atan2__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_atan2__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_ndgrid__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_ndgrid__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_meshgrid__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_meshgrid__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_repmat__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_repmat__", "ocl_bin.oct", "remove");

// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_mat_complex__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_mat_complex__", "ocl_bin.oct", "remove");
