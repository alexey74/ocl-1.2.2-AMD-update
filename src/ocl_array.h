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
 * Array.h:
 *   Copyright (C) 1993-2013 John W. Eaton
 *   Copyright (C) 2008-2009 Jaroslav Hajek
 *   Copyright (C) 2010 VZLU Prague
 * MArray.h:
 *   Copyright (C) 1993-2013 John W. Eaton
 *   Copyright (C) 2010 VZLU Prague
 * dNDArray.h:
 *   Copyright (C) 1996-2013 John W. Eaton
 * intNDArray.h:
 *   Copyright (C) 2004-2013 John W. Eaton
 */

#ifndef __OCL_ARRAY_H
#define __OCL_ARRAY_H

#include "ocl_array_prog.h"
#include <stdint.h>
#include <vector>

#include "ocl_octave_versions.h"

typedef octave_int64 ocl_idx_type;

class OclMemoryObject;
class OclProgram;


// OCL array class allowing shallow copies
template <typename T>
class
OclArray
{
protected:

  // class holding the OpenCL memory object
  class
  OclArrayRep
  {
  public:

    // empty, inoperable array
    OclArrayRep ()
      : memobj (0), len (0), count (1) {}

    // array with length; needs/activates an OpenCL context if non-empty
    OclArrayRep (octave_idx_type n)
      : memobj (0), len (n), count (1)
    {
      if (len > 0) {
        allocate ();
      }
    }

    // OclArrayRep: no constructor with length and constant fill!

    // array as copy of OpenCL array; needs/activates an OpenCL context if non-empty
    OclArrayRep (const OclArrayRep& a)
      : memobj (0), len (a.len), count (1)
    {
      if (len > 0) {
        assure_valid (a);
        allocate ();
        copy_from_oclbuffer (a, 0, 0, len);
      }
    }

    // array as partial copy of OpenCL array; needs/activates an OpenCL context if non-empty
    OclArrayRep (const OclArrayRep& a,
                 octave_idx_type slice_ofs_src,
                 octave_idx_type slice_len)
      : memobj (0), len (slice_len), count (1)
    {
      if (len > 0) {
        assure_valid (a);
        allocate ();
        copy_from_oclbuffer (a, slice_ofs_src, 0, len);
      }
    }

    // array as copy of octave memory array; needs/activates an OpenCL context if non-empty
    OclArrayRep (const T *d, octave_idx_type l)
      : memobj (0), len (l), count (1)
    {
      if (len > 0) {
        allocate ();
        copy_from_host (d, 0, len);
      }
    }

    ~OclArrayRep () { deallocate (); }

    void copy_from_host (const T *d_src,
                         octave_idx_type slice_ofs,
                         octave_idx_type slice_len);

    void copy_from_oclbuffer (const OclArrayRep& a,
                              octave_idx_type slice_ofs_src,
                              octave_idx_type slice_ofs_dst,
                              octave_idx_type slice_len);

    void copy_to_host (T *d_dst,
                       octave_idx_type slice_ofs,
                       octave_idx_type slice_len);

    bool is_valid (void) const;

    octave_idx_type length (void) const { return len; }

    void assure_valid (void) const;

    void assure_valid (const OclArrayRep& a) const;

    void *get_ocl_buffer (void) const;

    OclMemoryObject *memobj;
    octave_idx_type len;
    int count;

  private:

    void allocate (void);
    void deallocate (void);

    OclArrayRep& operator = (const OclArrayRep& a); // no assignment
  };

public:

  void make_unique (void)
  {
    if ((rep->count) > 1) {
      (rep->count)--;
      rep = new OclArrayRep (*rep, slice_ofs, slice_len);
      slice_ofs = 0;
    }
  }

  typedef T element_type;

protected:

  dim_vector dimensions;
  typename OclArray<T>::OclArrayRep *rep;
  octave_idx_type slice_ofs; // or should we use OpenCL sub-buffers ?! NO, MAJOR issues with CL_MISALIGNED_SUB_BUFFER_OFFSET
  octave_idx_type slice_len;
  bool is_logical;

  // Rationale:
  // slice_ofs is an offset into the OCL buffer, denoting together with slice_len the
  // actual portion of the data referenced by this OclArray<T> object. This allows
  // to make shallow copies not only of a whole array, but also of contiguous
  // subranges. Every time rep is directly manipulated, slice_ofs and slice_len
  // need to be properly updated.

  // slice constructor
  // octave's Array<T> uses the slice constructor in its members:
  //   column (...), page (...), linear_slice (...),
  //   index (const idx_vector& i),
  //   index (const idx_vector& i, const idx_vector& j),
  //   index (const Array<idx_vector>& ia)
  explicit
  OclArray (const OclArray<T>& a,
            const dim_vector& dv,
            octave_idx_type l,
            octave_idx_type u)
    : dimensions (dv), rep(a.rep), slice_ofs (a.slice_ofs+l), slice_len (u-l), is_logical(a.is_logical)
  {
    a.rep->assure_valid ();
    (rep->count)++;
    dimensions.chop_trailing_singletons ();
  }

  void fill (octave_idx_type fill_ofs, octave_idx_type fill_len, const T& val);

  void fill0 (octave_idx_type fill_ofs, octave_idx_type fill_len, const OclArray<T>& a);

private:

  typename OclArray<T>::OclArrayRep *
  nil_rep (void) const
  {/*fixed by Jinchuan Tang*/
    static typename OclArray<T>::OclArrayRep *nr =NULL;
    if (nr == NULL)
       nr = new OclArrayRep ();
    return nr;
  }

public:

  // Empty constructor (0x0).
  OclArray (void)
    : dimensions (), rep (nil_rep ()),
      slice_ofs (0), slice_len (rep->len), is_logical(false)
  {
    (rep->count)++;
  }

  // Copy constructor.
  OclArray (const OclArray<T>& a)
    : dimensions (a.dimensions), rep (a.rep),
      slice_ofs (a.slice_ofs), slice_len (a.slice_len), is_logical(a.is_logical)
  {
    (rep->count)++;
  }

  // nD uninitialized constructor.
  explicit OclArray (const dim_vector& dv)
    : dimensions (dv),
      rep (new typename OclArray<T>::OclArrayRep (dv.safe_numel ())),
      slice_ofs (0), slice_len (rep->len), is_logical(false)
  {
    dimensions.chop_trailing_singletons ();
  }

  // nD initialized constructor.
  explicit OclArray (const dim_vector& dv, const T& val)
    : dimensions (dv),
      rep (new typename OclArray<T>::OclArrayRep (dv.safe_numel ())),
      slice_ofs (0), slice_len (rep->len), is_logical(false)
  {
    dimensions.chop_trailing_singletons ();
    fill (slice_ofs, slice_len, val);
  }

  // Reshape constructor.
  explicit OclArray (const OclArray<T>& a, const dim_vector& dv);

  // Copy from host memory constructor.
  OclArray (const Array<T>& a)
    : dimensions (a.dims ()),
      rep (new typename OclArray<T>::OclArrayRep (a.data (), dimensions.safe_numel ())),
      slice_ofs (0), slice_len (a.numel ()), is_logical(false)
  {}

  // Type conversion constructor. OclArray: only for real->complex conversion.
  template <typename U>
  OclArray (const OclArray<U>& a);

  // Type conversion constructor. OclArray: only for (real,imag)->complex conversion.
  template <typename U>
  OclArray (const OclArray<U>& r, const OclArray<U>& i);

  virtual ~OclArray (void)
  {
    if ((--(rep->count)) == 0)
      delete rep;
  }

  OclArray<T>& operator = (const OclArray<T>& a)
  {
    if (this != & a) {
      if ((--(rep->count)) == 0)
        delete rep;

      rep = a.rep;
      (rep->count)++;

      dimensions = a.dimensions;
      slice_ofs = a.slice_ofs;
      slice_len = a.slice_len;
      is_logical = a.is_logical;
    }
    return *this;
  }

  bool is_valid (void) const { return rep->is_valid (); }

  bool islogical (void) const { return is_logical; }

  static bool is_integer_type (void);

  static bool is_uint_type (void);

  static bool is_complex_type (void);

  void fill (const T& val)
  {
    if ((rep->count) > 1) {
      (rep->count)--;
      rep = new OclArrayRep (slice_len);
      slice_ofs = 0;
    }
    fill (slice_ofs, slice_len, val);
  }

  void clear (void)
  {
    if ((--(rep->count)) == 0)
      delete rep;

    dimensions = dim_vector ();
    rep = nil_rep ();
    (rep->count)++;
    slice_ofs = 0;
    slice_len = rep->len;
    is_logical = false;
  }

  void clear (const dim_vector& dv)
  {
    if ((--(rep->count)) == 0)
      delete rep;

    dimensions = dv;
    rep = new OclArrayRep (dv.safe_numel ());
    slice_ofs = 0;
    slice_len = rep->len;
    is_logical = false;
    dimensions.chop_trailing_singletons ();
  }

  void clear (octave_idx_type r, octave_idx_type c)
  {
    clear (dim_vector (r, c));
  }

  octave_idx_type capacity (void) const { return slice_len; }
  octave_idx_type length (void) const { return slice_len; }
  octave_idx_type nelem (void) const { return slice_len; }
  octave_idx_type numel (void) const { return slice_len; }

  octave_idx_type dim1 (void) const { return dimensions (0); }
  octave_idx_type dim2 (void) const { return dimensions (1); }
  octave_idx_type dim3 (void) const { return dimensions (2); }

  // Return the OCL array as an OCL column vector.
  OclArray<T> as_column (void) const
  {
    rep->assure_valid ();
    OclArray<T> retval (*this);
    if (dimensions.length () != 2 || dimensions(1) != 1)
      retval.dimensions = dim_vector (numel (), 1);

    return retval;
  }

  // Return the OCL array as an OCL row vector.
  OclArray<T> as_row (void) const
  {
    rep->assure_valid ();
    OclArray<T> retval (*this);
    if (dimensions.length () != 2 || dimensions(0) != 1)
      retval.dimensions = dim_vector (1, numel ());

    return retval;
  }

  // Return the OCL array as an OCL matrix.
  OclArray<T> as_matrix (void) const
  {
    rep->assure_valid ();
    OclArray<T> retval (*this);
    if (dimensions.length () != 2)
      retval.dimensions = dimensions.redim (2);

    return retval;
  }

  octave_idx_type rows (void) const { return dim1 (); }
  octave_idx_type cols (void) const { return dim2 (); }
  octave_idx_type columns (void) const { return dim2 (); }
  octave_idx_type pages (void) const { return dim3 (); }

  size_t byte_size (void) const
  { return static_cast<size_t> (numel ()) * sizeof (T); }

  // Return a const-reference so that dims ()(i) works efficiently.
  const dim_vector& dims (void) const { return dimensions; }

  OclArray<T> squeeze (void) const;

  octave_idx_type compute_index (octave_idx_type i, octave_idx_type j) const;
  octave_idx_type compute_index (octave_idx_type i, octave_idx_type j,
                                 octave_idx_type k) const;
  octave_idx_type compute_index (const Array<octave_idx_type>& ra_idx) const;

  octave_idx_type compute_index_unchecked (const Array<octave_idx_type>& ra_idx)
  const
  { return dimensions.compute_index (ra_idx.data (), ra_idx.numel ()); }

  // OclArray: no elem(), xelem(), checkelem() members, no operator () possible!

  // Fast extractors. All of these produce shallow copies.
  // Extract column: A(:,k+1).
  OclArray<T> column (octave_idx_type k) const;
  // Extract page: A(:,:,k+1).
  OclArray<T> page (octave_idx_type k) const;
  // Extract a slice from this array as a column vector: A(:)(lo+1:up).
  // Must be 0 <= lo && up <= numel. May be up < lo.
  OclArray<T> linear_slice (octave_idx_type lo, octave_idx_type up) const;

  OclArray<T> reshape (octave_idx_type nr, octave_idx_type nc) const
  { return OclArray<T> (*this, dim_vector (nr, nc)); }

  OclArray<T> reshape (const dim_vector& new_dims) const
  { return OclArray<T> (*this, new_dims); }

  // OclArray: no permute, ipermute!

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  bool is_square (void) const { return (dim1 () == dim2 ()); }
  bool is_empty (void) const { return numel () == 0; }
  bool is_vector (void) const { return dimensions.is_vector (); }
#else // for octave versions >= 4.4.0
  bool issquare (void) const { return (dim1 () == dim2 ()); }
  bool isempty (void) const { return numel () == 0; }
  bool isvector (void) const { return dimensions.isvector (); }
#endif

  OclArray<T> transpose (void) const;
  OclArray<T> hermitian (void) const;

  // OclArray: no data() or fortran_vec()!

  // OclArray: copy to host memory
  Array<T> as_array (void) const;

  bool is_shared (void) { return ((rep->count) > 1); }

  int ndims (void) const { return dimensions.length (); }
  // we should use dimensions.ndims() here, but this member is still private in octave 3.8.0

  // Return the OCL array as an OCL index array.
  OclArray<ocl_idx_type> as_index (void) const;

  // Indexing (OclArray: never with resize).

  OclArray<T> index (const OclArray<ocl_idx_type>& i) const;
  OclArray<T> index (const idx_vector& i) const;
  OclArray<T> index (const idx_vector& i, const idx_vector& j) const;
  OclArray<T> index (const Array<idx_vector>& ia) const;

  // OclArray: no resize!

  // Indexed assignment (OclArray: never with resize).

  void assign (const OclArray<ocl_idx_type>& i, const T& rhs);
  void assign (const idx_vector& i, const T& rhs);
  void assign (const idx_vector& i, const idx_vector& j, const T& rhs);
  void assign (const Array<idx_vector>& ia, const T& rhs);

  void assign (const OclArray<ocl_idx_type>& i, const OclArray<T>& rhs);
  void assign (const idx_vector& i, const OclArray<T>& rhs);
  void assign (const idx_vector& i, const idx_vector& j, const OclArray<T>& rhs);
  void assign (const Array<idx_vector>& ia, const OclArray<T>& rhs);

  void assign_logical (const OclArray<T>& i, const T& rhs);

  // OclArray: no inserting or deleting of elements!

  void maybe_economize (void)
  {
    if (((rep->count) == 1) && (slice_len != rep->len)) {
      OclArrayRep *new_rep = new OclArrayRep (*rep, slice_ofs, slice_len);
      delete rep;
      rep = new_rep;
      slice_ofs = 0;
    }
  }

  // OclArray: no sorting or finding, no diag()!

  // Concatenation along a specified (0-based) dimension, equivalent to cat().
  // dim = -1 corresponds to dim = 0 and dim = -2 corresponds to dim = 1,
  // but apply the looser matching rules of vertcat/horzcat.
  static OclArray<T> cat (int dim, octave_idx_type n, const OclArray<T> *array_list);

  // eye constructor
  static OclArray<T> eye (octave_idx_type r, octave_idx_type c = -1);

  // linspace constructor
  static OclArray<T> linspace (T base, T limit, octave_idx_type n = 100);

  // logspace constructor
  static OclArray<T> logspace (T a, T b, octave_idx_type n = 50);

  // ndgrid utility function
  static std::vector< OclArray<T> > ndgrid (const std::vector< OclArray<T> > array_list);

  // meshgrid utility function
  static std::vector< OclArray<T> > meshgrid (const std::vector< OclArray<T> > array_list);

  // repmat utility function
  OclArray<T> repmat (const dim_vector& dv) const;

  // math functions

  OclArray<T> map (octave_base_value::unary_mapper_t umap) const;

  template <typename U>
  OclArray<U> map_c2r (octave_base_value::unary_mapper_t umap) const;

  OclArray<T> all (int dim = -1) const { return map1r (OclArrayKernels::all, dim); }
  OclArray<T> any (int dim = -1) const { return map1r (OclArrayKernels::any, dim); }
  OclArray<T> sum (int dim = -1) const { return map1r (OclArrayKernels::sum, dim); }
  OclArray<T> sumsq (int dim = -1) const { return map1r (OclArrayKernels::sumsq, dim); }
  OclArray<T> prod (int dim = -1) const { return map1r (OclArrayKernels::prod, dim); }
  OclArray<T> mean (int dim = -1) const { return map1r (OclArrayKernels::mean, dim); }
  OclArray<T> meansq (int dim = -1) const { return map1r (OclArrayKernels::meansq, dim); }
  OclArray<T> std (int opt = 0, int dim = -1) const;
  OclArray<T> max (int dim = -1) const
    { return map1ri (OclArrayKernels::max, dim, 0); }
  OclArray<T> max (OclArray<ocl_idx_type>& indices, int dim = -1) const
    { return map1ri (OclArrayKernels::max, dim, & indices); }
  OclArray<T> min (int dim = -1) const
    { return map1ri (OclArrayKernels::min, dim, 0); }
  OclArray<T> min (OclArray<ocl_idx_type>& indices, int dim = -1) const
    { return map1ri (OclArrayKernels::min, dim, & indices); }

  OclArray<T> cumsum (int dim = -1) const { return map1re (OclArrayKernels::cumsum, dim); }
  OclArray<T> cumprod (int dim = -1) const { return map1re (OclArrayKernels::cumprod, dim); }
  OclArray<T> cummax (int dim = -1) const
    { return map1rie (OclArrayKernels::cummax, dim, 0); }
  OclArray<T> cummax (OclArray<ocl_idx_type>& indices, int dim = -1) const
    { return map1rie (OclArrayKernels::cummax, dim, & indices); }
  OclArray<T> cummin (int dim = -1) const
    { return map1rie (OclArrayKernels::cummin, dim, 0); }
  OclArray<T> cummin (OclArray<ocl_idx_type>& indices, int dim = -1) const
    { return map1rie (OclArrayKernels::cummin, dim, & indices); }

  OclArray<ocl_idx_type> findfirst (int dim = -1) const { return map1rf (OclArrayKernels::findfirst, dim); }
  OclArray<ocl_idx_type> findlast (int dim = -1) const { return map1rf (OclArrayKernels::findlast, dim); }

  OclArray<T> max2 (const T& v) const { return map1 (OclArrayKernels::max1, v); }
  OclArray<T> max2 (const OclArray<T>& s2) const { return map2s (OclArrayKernels::max2, s2); }
  OclArray<T> min2 (const T& v) const { return map1 (OclArrayKernels::min1, v); }
  OclArray<T> min2 (const OclArray<T>& s2) const { return map2s (OclArrayKernels::min2, s2); }

  OclArray<T> atan2 (const OclArray<T>& s2) const { return map2s (OclArrayKernels::atan2, s2); }

  // math arithmetic

  OclArray<T> uminus (void) const { return map (OclArrayKernels::uminus); }

  OclArray<T> add (const T& summand) const { return map1 (OclArrayKernels::add1, summand); }
  OclArray<T> add (const OclArray<T>& s2) const { return map2s (OclArrayKernels::add2, s2); }

  OclArray<T> sub_constmin (const T& minuend) const { return map1 (OclArrayKernels::sub1m, minuend); }
  OclArray<T> sub_constsub (const T& subtrahend) const { return map1 (OclArrayKernels::sub1s, subtrahend); }
  OclArray<T> sub (const OclArray<T>& s2) const { return map2s (OclArrayKernels::sub2, s2); }

  OclArray<T> times (const T& factor) const { return map1 (OclArrayKernels::mul1, factor); }
  OclArray<T> times (const OclArray<T>& s2) const { return map2s (OclArrayKernels::mul2, s2); }

  OclArray<T> mtimes (const OclArray<T>& s2) const;

  OclArray<T> divide_constnum (const T& numerator) const { return map1 (OclArrayKernels::div1n, numerator); }
  OclArray<T> divide_constdenom (const T& denominator) const { return map1 (OclArrayKernels::div1d, denominator); }
  OclArray<T> divide (const OclArray<T>& s2) const { return map2s (OclArrayKernels::div2, s2); }

  OclArray<T> power_constbase (const T& base) const { return map1 (OclArrayKernels::power1b, base); }
  OclArray<T> power_constexp (const T& exponent) const { return map1 (OclArrayKernels::power1e, exponent); }
  OclArray<T> power (const OclArray<T>& s2) const { return map2s (OclArrayKernels::power2, s2); }

  void changesign (void)
  { map_inplace (OclArrayKernels::uminus); }

  // OclArray: no diff, no fourier!

  void print_info (std::ostream& os, const std::string& prefix = "") const;

  static std::string get_type_str_oct (void) { return type_str_oct; }
  static std::string get_type_str_oclc (void) { return type_str_oclc; }

public:

  // math operators

#define OCLARRAY_CMP_OP(OP, F) \
  friend OclArray<T> operator OP (const OclArray<T>& s1, const T& s2) \
    { return OclArray<T>::map2sf (OclArrayKernels::compare, s1, s1, s2, 16 * F + 0); } \
  friend OclArray<T> operator OP (const T& s1, const OclArray<T>& s2) \
    { return OclArray<T>::map2sf (OclArrayKernels::compare, s2, s2, s1, 16 * F + 1); } \
  friend OclArray<T> operator OP (const OclArray<T>& s1, const OclArray<T>& s2) \
    { return OclArray<T>::map2sf (OclArrayKernels::compare, s1, s2, T (0), 16 * F + 2); }

  OCLARRAY_CMP_OP(<,  0)
  OCLARRAY_CMP_OP(<=, 1)
  OCLARRAY_CMP_OP(>,  2)
  OCLARRAY_CMP_OP(>=, 3)
  OCLARRAY_CMP_OP(==, 4)
  OCLARRAY_CMP_OP(!=, 5)

#undef OCLARRAY_CMP_OP

  friend OclArray<T> operator && (const OclArray<T>& s1, const T& s2)
    { return OclArray<T>::map2sf (OclArrayKernels::logic, s1, s1, s2, 16 * 0 + 0); }
  friend OclArray<T> operator && (const T& s1, const OclArray<T>& s2)
    { return OclArray<T>::map2sf (OclArrayKernels::logic, s2, s2, s1, 16 * 0 + 1); }
  friend OclArray<T> operator && (const OclArray<T>& s1, const OclArray<T>& s2)
    { return OclArray<T>::map2sf (OclArrayKernels::logic, s1, s2, T (0), 16 * 0 + 2); }

  friend OclArray<T> operator || (const OclArray<T>& s1, const T& s2)
    { return OclArray<T>::map2sf (OclArrayKernels::logic, s1, s1, s2, 16 * 1 + 0); }
  friend OclArray<T> operator || (const T& s1, const OclArray<T>& s2)
    { return OclArray<T>::map2sf (OclArrayKernels::logic, s2, s2, s1, 16 * 1 + 1); }
  friend OclArray<T> operator || (const OclArray<T>& s1, const OclArray<T>& s2)
    { return OclArray<T>::map2sf (OclArrayKernels::logic, s1, s2, T (0), 16 * 1 + 2); }

  friend OclArray<T> operator ! (const OclArray<T>& s1)
    { return OclArray<T>::map2sf (OclArrayKernels::logic, s1, s1, T (0), 16 * 2 + 0); }

#define OCLARRAY_ASN_OPS(OP2, OP1, K2, K1) \
  friend OclArray<T> OP2 (OclArray<T>& s1, const OclArray<T>& s2) \
    { return s1.map2s_inplace (OclArrayKernels::K2, s2); } \
  friend OclArray<T> OP1 (OclArray<T>& s1, const T& s2) \
    { return s1.map1_inplace (OclArrayKernels::K1, s2); }

  OCLARRAY_ASN_OPS(operator +=, operator +=, add2, add1)
  OCLARRAY_ASN_OPS(operator -=, operator -=, sub2, sub1s)
  OCLARRAY_ASN_OPS(product_eq,  operator *=, mul2, mul1)
  OCLARRAY_ASN_OPS(quotient_eq, operator /=, div2, div1d)

#undef OCLARRAY_ASN_OPS

#define OCLARRAY_BINOPS(OP2, OP1, K2, K1O, K1R) \
  friend OclArray<T> OP2 (const OclArray<T>& s1, const OclArray<T>& s2) \
    { return s1.map2s (OclArrayKernels::K2, s2); } \
  friend OclArray<T> OP1 (const OclArray<T>& s1, const T& s2) \
    { return s1.map1 (OclArrayKernels::K1O, s2); } \
  friend OclArray<T> OP1 (const T& s1, const OclArray<T>& s2) \
    { return s2.map1 (OclArrayKernels::K1R, s1); }

  OCLARRAY_BINOPS(operator +, operator +, add2,   add1,    add1   )
  OCLARRAY_BINOPS(operator -, operator -, sub2,   sub1s,   sub1m  )
  OCLARRAY_BINOPS(product,    operator *, mul2,   mul1,    mul1   )
  OCLARRAY_BINOPS(quotient,   operator /, div2,   div1d,   div1n  )
  OCLARRAY_BINOPS(pow,        pow,        power2, power1e, power1b)

#undef OCLARRAY_BINOPS

  friend OclArray<T> operator + (const OclArray<T>& s)
    { return s; }
  friend OclArray<T> operator - (const OclArray<T>& s)
    { return s.uminus(); }

protected:

  void dim_wise_op_newdims (int dim,
                            dim_vector& new_dimensions,
                            octave_idx_type& len,
                            octave_idx_type& fac) const;

  OclArray<T> map (OclArrayKernels::Kernel kernel) const;
  template <typename U> OclArray<U> map_c2r (OclArrayKernels::Kernel kernel) const;
  OclArray<T> map1 (OclArrayKernels::Kernel kernel, const T& par) const;
  OclArray<T> map1r (OclArrayKernels::Kernel kernel, int dim = -1) const;
  OclArray<T> map1re (OclArrayKernels::Kernel kernel, int dim = -1) const;
  OclArray<ocl_idx_type> map1rf (OclArrayKernels::Kernel kernel, int dim = -1) const;
  OclArray<T> map1ri (OclArrayKernels::Kernel kernel, int dim = -1, OclArray<ocl_idx_type> *indices = 0) const;
  OclArray<T> map1rie (OclArrayKernels::Kernel kernel, int dim = -1, OclArray<ocl_idx_type> *indices = 0) const;
  OclArray<T> map2s (OclArrayKernels::Kernel kernel, const OclArray<T>& s2) const;

  void map_inplace (OclArrayKernels::Kernel kernel);
  OclArray<T> map1_inplace (OclArrayKernels::Kernel kernel, const T& par);
  OclArray<T> map2s_inplace (OclArrayKernels::Kernel kernel, const OclArray<T>& s2);

  static OclArray<T> map2sf (OclArrayKernels::Kernel kernel,
                             const OclArray<T>& s1,
                             const OclArray<T>& s2,
                             const T& par,
                             unsigned long fcn);

  OclArray<T> repmat1 (int dim, octave_idx_type rep) const;
  void index_helper (const Array<idx_vector>& ia,
                     dim_vector& dv,
                     dim_vector& rdv,
                     bool& all_colons,
                     octave_idx_type& l,
                     octave_idx_type& u) const;

  static OclProgram array_prog;
  static std::vector<int> kernel_indices;

  static void assure_valid_array_prog (void);

private:

  static std::string type_str_oct;
  static std::string type_str_oclc;

  static void instantiation_guard ();

  template <class U> friend class OclArray;
  friend class OclProgram;
  friend class octave_ocl_program;
};


template <typename T>
std::ostream&
operator << (std::ostream& os, const OclArray<T>& a);


typedef OclArray<octave_int8  > OclInt8NDArray;
typedef OclArray<octave_int16 > OclInt16NDArray;
typedef OclArray<octave_int32 > OclInt32NDArray;
typedef OclArray<octave_int64 > OclInt64NDArray;
typedef OclArray<octave_uint8 > OclUint8NDArray;
typedef OclArray<octave_uint16> OclUint16NDArray;
typedef OclArray<octave_uint32> OclUint32NDArray;
typedef OclArray<octave_uint64> OclUint64NDArray;
typedef OclArray<float        > OclFloatNDArray;
typedef OclArray<double       > OclNDArray;
typedef OclArray<FloatComplex > OclFloatComplexNDArray;
typedef OclArray<Complex      > OclComplexNDArray;


#define OCLARRAY_BINOPS_C(OP2, OP1) \
  template <typename TR> \
  OclArray< std::complex<TR> > OP2 (const OclArray<TR>& s1, const OclArray< std::complex<TR> >& s2) \
    { return OP2 (OclArray< std::complex<TR> >(s1), s2); } \
  template <typename TR> \
  OclArray< std::complex<TR> > OP2 (const OclArray< std::complex<TR> >& s1, const OclArray<TR>& s2) \
    { return OP2 (s1, OclArray< std::complex<TR> >(s2)); } \
  template <typename TR> \
  OclArray< std::complex<TR> > OP1 (const OclArray<TR>& s1, const  std::complex<TR> & s2) \
    { return OP1 (OclArray< std::complex<TR> >(s1), s2); } \
  template <typename TR> \
  OclArray< std::complex<TR> > OP1 (const  std::complex<TR> & s1, const OclArray<TR>& s2) \
    { return OP1 (s1, OclArray< std::complex<TR> >(s2)); } \

  OCLARRAY_BINOPS_C(operator <,  operator < )
  OCLARRAY_BINOPS_C(operator <=, operator <=)
  OCLARRAY_BINOPS_C(operator >,  operator > )
  OCLARRAY_BINOPS_C(operator >=, operator >=)
  OCLARRAY_BINOPS_C(operator ==, operator ==)
  OCLARRAY_BINOPS_C(operator !=, operator !=)

  OCLARRAY_BINOPS_C(operator &&, operator &&)
  OCLARRAY_BINOPS_C(operator ||, operator ||)

  OCLARRAY_BINOPS_C(operator +, operator +)
  OCLARRAY_BINOPS_C(operator -, operator -)
  OCLARRAY_BINOPS_C(product,    operator *)
  OCLARRAY_BINOPS_C(quotient,   operator /)
  OCLARRAY_BINOPS_C(pow,        pow       )

#undef OCLARRAY_BINOPS_C

#endif  /* __OCL_ARRAY_H */
