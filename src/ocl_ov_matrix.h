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
 * ov-base-mat.h:
 *   Copyright (C) 1998-2013 John W. Eaton
 *   Copyright (C) 2009-2010 VZLU Prague
 * ov-re-mat.h:
 *   Copyright (C) 1996-2013 John W. Eaton
 *   Copyright (C) 2009-2010 VZLU Prague
 */

#ifndef __OCL_OV_MATRIX_H
#define __OCL_OV_MATRIX_H

#include "ocl_octave_versions.h"
#include "ocl_array.h"

#include <ov-int8.h>
#include <ov-int16.h>
#include <ov-int32.h>
#include <ov-int64.h>
#include <ov-uint8.h>
#include <ov-uint16.h>
#include <ov-uint32.h>
#include <ov-uint64.h>
#include <ov-flt-re-mat.h>
#include <ov-re-mat.h>
#include <ov-flt-cx-mat.h>
#include <ov-cx-mat.h>
#include <ov-float.h>
#include <ov-complex.h>
#include <ov-flt-complex.h>

template <typename AT>
class
octave_base_ocl_matrix : public octave_base_value
{
public:

  typedef typename AT::element_type element_type;
  typedef AT array_type;

  octave_base_ocl_matrix (void)
    : octave_base_value (), matrix () { }

  octave_base_ocl_matrix (const octave_base_ocl_matrix& m)
    : octave_base_value (), matrix (m.matrix) { }

  octave_base_ocl_matrix (const AT& m)
    : octave_base_value (), matrix (m) { }

  ~octave_base_ocl_matrix (void) { }

  octave_base_value *clone (void) const { return new octave_base_ocl_matrix<AT> (*this); }
  octave_base_value *empty_clone (void) const { return new octave_base_ocl_matrix<AT> (); }

  const AT& ocl_array_value (void) const { return matrix; }

  type_conv_info numeric_conversion_function () const;

  int8NDArray
  int8_array_value (void) const;

  int16NDArray
  int16_array_value (void) const;

  int32NDArray
  int32_array_value (void) const;

  int64NDArray
  int64_array_value (void) const;

  uint8NDArray
  uint8_array_value (void) const;

  uint16NDArray
  uint16_array_value (void) const;

  uint32NDArray
  uint32_array_value (void) const;

  uint64NDArray
  uint64_array_value (void) const;

  NDArray
  array_value (bool = false) const;

  FloatNDArray
  float_array_value (bool = false) const;

  Matrix
  matrix_value (bool = false) const;

  FloatMatrix
  float_matrix_value (bool = false) const;

  ComplexNDArray
  complex_array_value (bool = false) const;

  FloatComplexNDArray
  float_complex_array_value (bool = false) const;

  ComplexMatrix
  complex_matrix_value (bool = false) const;

  FloatComplexMatrix
  float_complex_matrix_value (bool = false) const;

  octave_value as_int8 (void) const // for octave versions >= 4.2.0
  { return new octave_int8_matrix (int8_array_value ()); }

  octave_value as_int16 (void) const
  { return new octave_int16_matrix (int16_array_value ()); }

  octave_value as_int32 (void) const
  { return new octave_int32_matrix (int32_array_value ()); }

  octave_value as_int64 (void) const
  { return new octave_int64_matrix (int64_array_value ()); }

  octave_value as_uint8 (void) const
  { return new octave_uint8_matrix (uint8_array_value ()); }

  octave_value as_uint16 (void) const
  { return new octave_uint16_matrix (uint16_array_value ()); }

  octave_value as_uint32 (void) const
  { return new octave_uint32_matrix (uint32_array_value ()); }

  octave_value as_uint64 (void) const
  { return new octave_uint64_matrix (uint64_array_value ()); }

  octave_value as_double (void) const
  {
    if (matrix.is_complex_type ())
      return new octave_complex_matrix (complex_array_value ());
    else
      return new octave_matrix (array_value ());
  }

  octave_value as_single (void) const
  {
    if (matrix.is_complex_type ())
      return new octave_float_complex_matrix (float_complex_array_value ());
    else
      return new octave_float_matrix (float_array_value ());
  }

  size_t byte_size (void) const { return matrix.byte_size (); }

  void maybe_economize (void) { matrix.maybe_economize (); }

  octave_value subsref (const std::string& type,
                        const std::list<octave_value_list>& idx);

  octave_value_list subsref (const std::string& type,
                             const std::list<octave_value_list>& idx, int)
  { return subsref (type, idx); }

  octave_value subsasgn (const std::string& type,
                         const std::list<octave_value_list>& idx,
                         const octave_value& rhs);

  octave_base_ocl_matrix<AT> *ocl_index_op (const octave_value_list& idx,
                                            bool resize_ok = false);

  octave_value_list do_multi_index_op (int, const octave_value_list& idx)
  { return do_index_op (idx); }

  void assign (const octave_value_list& idx, typename AT::element_type rhs);

  void assign (const octave_value_list& idx, const AT& rhs);

  // octave_base_ocl_matrix: no delete_elements!

  dim_vector dims (void) const { return matrix.dims (); }

  octave_idx_type numel (void) const { return matrix.numel (); }

  int ndims (void) const { return matrix.ndims (); }

  // octave_base_ocl_matrix: no nnz!

  // octave_base_ocl_matrix: no permute, no resize!

  // octave_base_ocl_matrix: no diag, no sorting!

  bool is_matrix_type (void) const { return true; }

  bool is_numeric_type (void) const { return true; }

  bool is_defined (void) const { return true; }

  bool is_constant (void) const { return true; }

  bool is_real_type (void) const { return true; }

  bool is_real_matrix (void) const { return true; }

  bool is_single_type(void) const {return true;}

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  bool is_complex_type (void) const
#else // for octave versions >= 4.4.0
  bool iscomplex (void) const
#endif
  { return matrix.is_complex_type (); }

  bool is_complex_matrix (void) const
  { return matrix.is_complex_type (); }

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  bool is_integer_type (void) const
#else // for octave versions >= 4.4.0
  bool isinteger (void) const
#endif
  { return matrix.is_integer_type (); }

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  bool is_float_type (void) const
#else // for octave versions >= 4.4.0
  bool isfloat (void) const
#endif
  { return ! matrix.is_integer_type (); }

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  bool is_bool_type (void) const
#else // for octave versions >= 4.4.0
  bool islogical (void) const
#endif
  { return matrix.islogical (); }

  bool print_as_scalar (void) const { return true; }

  void print (std::ostream& os, bool pr_as_read_syntax = false)
  { print_raw (os, pr_as_read_syntax); }

  void print_raw (std::ostream& os, bool pr_as_read_syntax = false) const
  { os << matrix; }

  void print_info (std::ostream& os, const std::string& prefix) const
  { matrix.print_info (os, prefix); }

  // octave_base_ocl_matrix: no short_disp!

  AT& matrix_ref (void)
  { return matrix; }

  const AT& matrix_ref (void) const
  { return matrix; }

  // octave_base_ocl_matrix: no diag, no sorting!

  octave_value transpose (void) const
  { return new octave_base_ocl_matrix<AT> (matrix.transpose ()); }

  octave_value hermitian (void) const
  { return new octave_base_ocl_matrix<AT> (matrix.hermitian ()); }

  octave_value reshape (const dim_vector& new_dims) const
  { return new octave_base_ocl_matrix<AT> (matrix.reshape (new_dims)); }

  octave_value squeeze (void) const
  { return new octave_base_ocl_matrix<AT> (matrix.squeeze ()); }

  octave_value all (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.all (dim)); }

  octave_value any (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.any (dim)); }

  octave_value sum (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.sum (dim)); }

  octave_value sumsq (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.sumsq (dim)); }

  octave_value prod (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.prod (dim)); }

  octave_value mean (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.mean (dim)); }

  octave_value meansq (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.meansq (dim)); }

  octave_value std (int opt = 0, int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.std (opt, dim)); }

  octave_value cumsum (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.cumsum (dim)); }

  octave_value cumprod (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.cumprod (dim)); }

  octave_value as_index (void) const
  {
    OclArray<ocl_idx_type> inds = matrix.as_index ();
    return new octave_base_ocl_matrix< OclArray<ocl_idx_type> > (inds); // NOT: += 1
  }

  octave_value findfirst (int dim = 0) const
  {
    OclArray<ocl_idx_type> inds = matrix.findfirst (dim);
    return new octave_base_ocl_matrix< OclArray<ocl_idx_type> > (inds += 1);
  }

  octave_value findlast (int dim = 0) const
  {
    OclArray<ocl_idx_type> inds = matrix.findlast (dim);
    return new octave_base_ocl_matrix< OclArray<ocl_idx_type> > (inds += 1);
  }

  octave_value max2 (const element_type& v) const
  { return new octave_base_ocl_matrix<AT> (matrix.max2 (v)); }

  octave_value max2 (const octave_base_ocl_matrix<AT>& s2) const
  { return new octave_base_ocl_matrix<AT> (matrix.max2 (s2.matrix)); }

  octave_value min2 (const element_type& v) const
  { return new octave_base_ocl_matrix<AT> (matrix.min2 (v)); }

  octave_value min2 (const octave_base_ocl_matrix<AT>& s2) const
  { return new octave_base_ocl_matrix<AT> (matrix.min2 (s2.matrix)); }

  octave_value max (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.max (dim)); }

  octave_value max (octave_value& indices, int dim = 0) const
  {
    octave_base_ocl_matrix<AT> *newmat;
    OclArray<ocl_idx_type> inds;
    newmat = new octave_base_ocl_matrix<AT> (matrix.max (inds, dim));
    indices = octave_value (new octave_base_ocl_matrix< OclArray<ocl_idx_type> > (inds += 1));
    return newmat;
  }

  octave_value min (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.min (dim)); }

  octave_value min (octave_value& indices, int dim = 0) const
  {
    octave_base_ocl_matrix<AT> *newmat;
    OclArray<ocl_idx_type> inds;
    newmat = new octave_base_ocl_matrix<AT> (matrix.min (inds, dim));
    indices = octave_value (new octave_base_ocl_matrix< OclArray<ocl_idx_type> > (inds += 1));
    return newmat;
  }

  octave_value cummax (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.cummax (dim)); }

  octave_value cummax (octave_value& indices, int dim = 0) const
  {
    octave_base_ocl_matrix<AT> *newmat;
    OclArray<ocl_idx_type> inds;
    newmat = new octave_base_ocl_matrix<AT> (matrix.cummax (inds, dim));
    indices = octave_value (new octave_base_ocl_matrix< OclArray<ocl_idx_type> > (inds += 1));
    return newmat;
  }

  octave_value cummin (int dim = 0) const
  { return new octave_base_ocl_matrix<AT> (matrix.cummin (dim)); }

  octave_value cummin (octave_value& indices, int dim = 0) const
  {
    octave_base_ocl_matrix<AT> *newmat;
    OclArray<ocl_idx_type> inds;
    newmat = new octave_base_ocl_matrix<AT> (matrix.cummin (inds, dim));
    indices = octave_value (new octave_base_ocl_matrix< OclArray<ocl_idx_type> > (inds += 1));
    return newmat;
  }

  octave_value atan2 (const octave_base_ocl_matrix<AT>& s2) const
  { return new octave_base_ocl_matrix<AT> (matrix.atan2 (s2.matrix)); }

  octave_value map (octave_base_value::unary_mapper_t umap) const;

  octave_value do_index_op (const octave_value_list& idx,
                            bool resize_ok = false)
  {
    octave_base_ocl_matrix<AT> *oldmat;
    octave_base_ocl_matrix<AT> *newmat;
    oldmat = octave_base_ocl_matrix<AT>::ocl_index_op (idx);
    newmat = new octave_base_ocl_matrix<AT> (*oldmat);
    delete oldmat;
    return newmat;
  }

  void increment (void) { matrix += element_type (1); }

  void decrement (void) { matrix -= element_type (1); }

  void changesign (void) { matrix.changesign (); }

  // octave_base_ocl_matrix: no convert_to_str_internal!

  bool save_ascii (std::ostream& os);

  bool load_ascii (std::istream& is);

#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
  bool save_binary (std::ostream& os, bool& save_as_floats);
#else // for octave versions >= 6.1.0
  bool save_binary (std::ostream& os, bool save_as_floats);
#endif

  bool load_binary (std::istream& is, bool swap, octave::mach_info::float_format fmt);

  bool save_hdf5 (octave_hdf5_id loc_id, const char *name, bool save_as_floats);

  bool load_hdf5 (octave_hdf5_id loc_id, const char *name);

  // octave_base_ocl_matrix: no write!

  static octave_value_list
  as_index (const octave_value_list& args, int nargout);

  static octave_value_list
  sum (const octave_value_list& args, int nargout);

  static octave_value_list
  sumsq (const octave_value_list& args, int nargout);

  static octave_value_list
  prod (const octave_value_list& args, int nargout);

  static octave_value_list
  mean (const octave_value_list& args, int nargout);

  static octave_value_list
  meansq (const octave_value_list& args, int nargout);

  static octave_value_list
  cumsum (const octave_value_list& args, int nargout);

  static octave_value_list
  cumprod (const octave_value_list& args, int nargout);

  static octave_value_list
  findfirst (const octave_value_list& args, int nargout);

  static octave_value_list
  findlast (const octave_value_list& args, int nargout);

  static octave_value_list
  std (const octave_value_list& args, int nargout);

  static octave_value_list
  max (const octave_value_list& args, int nargout);

  static octave_value_list
  min (const octave_value_list& args, int nargout);

  static octave_value_list
  cummax (const octave_value_list& args, int nargout);

  static octave_value_list
  cummin (const octave_value_list& args, int nargout);

  static octave_value_list
  atan2 (const octave_value_list& args, int nargout);

  static octave_value_list
  ndgrid (const octave_value_list& args, int nargout);

  static octave_value_list
  meshgrid (const octave_value_list& args, int nargout);

  static octave_value_list
  repmat (const octave_value_list& args, int nargout);

  static octave_value_list
  complex (const octave_value_list& args, int nargout);

protected:

  AT matrix;

private:

  octave_base_ocl_matrix& operator = (const octave_base_ocl_matrix&); // No assignment.

#ifdef DECLARE_OCTAVE_ALLOCATOR
  DECLARE_OCTAVE_ALLOCATOR
#endif

  DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

#if defined (DEFINE_TEMPLATE_OV_TYPEID_FUNCTIONS_AND_DATA)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< double        >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< float         >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< Complex       >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< FloatComplex  >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_int8   >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_int16  >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_int32  >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_int64  >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_uint8  >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_uint16 >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_uint32 >)
DECLARE_TEMPLATE_OV_TYPEID_SPECIALIZATIONS (octave_base_ocl_matrix, OclArray< octave_uint64 >)
#endif

typedef octave_base_ocl_matrix< OclArray< double        > > octave_ocl_matrix;
typedef octave_base_ocl_matrix< OclArray< float         > > octave_ocl_float_matrix;
typedef octave_base_ocl_matrix< OclArray< Complex       > > octave_ocl_complex_matrix;
typedef octave_base_ocl_matrix< OclArray< FloatComplex  > > octave_ocl_float_complex_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_int8   > > octave_ocl_int8_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_int16  > > octave_ocl_int16_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_int32  > > octave_ocl_int32_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_int64  > > octave_ocl_int64_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_uint8  > > octave_ocl_uint8_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_uint16 > > octave_ocl_uint16_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_uint32 > > octave_ocl_uint32_matrix;
typedef octave_base_ocl_matrix< OclArray< octave_uint64 > > octave_ocl_uint64_matrix;

#endif  /* __OCL_OV_MATRIX_H */
