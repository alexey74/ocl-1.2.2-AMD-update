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
 * Array.cc:
 *   Copyright (C) 1993-2013 John W. Eaton
 *   Copyright (C) 2008-2009 Jaroslav Hajek
 *   Copyright (C) 2009 VZLU Prague
 */

#include "ocl_array.h"
#include "ocl_program.h"
#include "ocl_lib.h"
#include "ocl_array_prog.h"
#include "ocl_memobj.h"
#include <Array-util.h>



// ---------- OclArray<T> specializations


#define SPECIALIZE_OCLARRAY( T, OCT_STR, OCLC_STR, IS_INTEGER, IS_UINT, IS_COMPLEX ) \
  template <> std::string OclArray<T>::type_str_oct = #OCT_STR; \
  template <> std::string OclArray<T>::type_str_oclc = #OCLC_STR; \
  template <> OclProgram OclArray<T>::array_prog = OclProgram (); \
  template <> std::vector<int> OclArray<T>::kernel_indices  = std::vector<int> (); \
  template <> bool OclArray<T>::is_integer_type (void) { return IS_INTEGER; } \
  template <> bool OclArray<T>::is_uint_type (void) { return IS_UINT; } \
  template <> bool OclArray<T>::is_complex_type (void) { return IS_COMPLEX; }


SPECIALIZE_OCLARRAY (octave_int8,    int8,    char,     true,   false,  false );
SPECIALIZE_OCLARRAY (octave_int16,   int16,   short,    true,   false,  false );
SPECIALIZE_OCLARRAY (octave_int32,   int32,   int,      true,   false,  false );
SPECIALIZE_OCLARRAY (octave_int64,   int64,   long,     true,   false,  false );
SPECIALIZE_OCLARRAY (octave_uint8,   uint8,   uchar,    true,   true,   false );
SPECIALIZE_OCLARRAY (octave_uint16,  uint16,  ushort,   true,   true,   false );
SPECIALIZE_OCLARRAY (octave_uint32,  uint32,  uint,     true,   true,   false );
SPECIALIZE_OCLARRAY (octave_uint64,  uint64,  ulong,    true,   true,   false );
SPECIALIZE_OCLARRAY (float,          single,  float,    false,  false,  false );
SPECIALIZE_OCLARRAY (double,         double,  double,   false,  false,  false );
SPECIALIZE_OCLARRAY (FloatComplex,   single,  float2,   false,  false,  true  );
SPECIALIZE_OCLARRAY (Complex,        double,  double2,  false,  false,  true  );
// C type, octave interpreter class, OpenCL C type



// when expanding the OclArray members:
// assure "rep->assure_valid ();" and "assure_valid_array_prog ();" with all modifying operations
// assure all calculations include slice_ofs
// assure all operations handle empty matrices correctly


// TODO:
// - handle empty ocl matrices everywhere, mostly without error (currently, empty matrices are inoperable)


// ---------- static helper functions


static
void
ocl_array_inop_error (void)
{
  ocl_error ("OclArray: operating on an inoperable array object (e.g., context destroyed or empty object)");
}


// ---------- OclArray<T>::OclArrayRep members


template <typename T>
void *
OclArray<T>::OclArrayRep::get_ocl_buffer (void) const
{
  return is_valid () ? memobj->get_ocl_buffer () : 0;
}


template <typename T>
bool
OclArray<T>::OclArrayRep::is_valid (void) const
{
  return (memobj != 0) && (memobj->object_context_still_valid ());
}


template <typename T>
void
OclArray<T>::OclArrayRep::assure_valid (void) const
{
  if (! is_valid ())
    ocl_array_inop_error ();
}


template <typename T>
void
OclArray<T>::OclArrayRep::assure_valid (const OclArrayRep& a) const
{
  if (! a.is_valid ())
    ocl_array_inop_error ();
}


template <typename T>
void
OclArray<T>::OclArrayRep::allocate (void)
{
  // only called from a constructor
  // we know: len > 0
  size_t size = len * sizeof (T);
  memobj = new OclMemoryObject (size);
}


template <typename T>
void
OclArray<T>::OclArrayRep::deallocate (void)
{
  // only called from destructor
  delete memobj;
  memobj = 0;
}


template <typename T>
void
OclArray<T>::OclArrayRep::copy_from_oclbuffer (const OclArrayRep& a,
                                               octave_idx_type slice_ofs_src,
                                               octave_idx_type slice_ofs_dst,
                                               octave_idx_type slice_len)
{
  assure_valid ();
  assure_valid (a);

  void *ocl_buffer_src = a.get_ocl_buffer ();
  size_t offset_src = static_cast<size_t> (slice_ofs_src) * sizeof (T);
  size_t offset_dst = static_cast<size_t> (slice_ofs_dst) * sizeof (T);
  size_t size = static_cast<size_t> (slice_len) * sizeof (T);

  last_error = clEnqueueCopyBuffer (get_command_queue (),
                                    (cl_mem) ocl_buffer_src,
                                    (cl_mem) get_ocl_buffer (),
                                    offset_src,
                                    offset_dst,
                                    size,
                                    0, 0, 0);
  ocl_check_error ("clEnqueueCopyBuffer");
}


template <typename T>
void
OclArray<T>::OclArrayRep::copy_from_host (const T *d_src,
                                          octave_idx_type slice_ofs,
                                          octave_idx_type slice_len)
{
  assure_valid ();

  size_t offset = static_cast<size_t> (slice_ofs) * sizeof (T);
  size_t size = static_cast<size_t> (slice_len) * sizeof (T);

  last_error = clEnqueueWriteBuffer (get_command_queue (),
                                     (cl_mem) get_ocl_buffer (),
                                     CL_TRUE,
                                     offset,
                                     size,
                                     d_src,
                                     0, 0, 0);
  ocl_check_error ("clEnqueueWriteBuffer");
}


template <typename T>
void
OclArray<T>::OclArrayRep::copy_to_host (T *d_dst,
                                        octave_idx_type slice_ofs,
                                        octave_idx_type slice_len)
{
  assure_valid ();

  size_t offset = static_cast<size_t> (slice_ofs) * sizeof (T);
  size_t size = static_cast<size_t> (slice_len) * sizeof (T);

  last_error = clEnqueueReadBuffer (get_command_queue (),
                                    (cl_mem) get_ocl_buffer (),
                                    CL_TRUE,
                                    offset,
                                    size,
                                    d_dst,
                                    0, 0, 0);
  ocl_check_error ("clEnqueueReadBuffer");
}


// ---------- OclArray<T> members


// OclArray reshape constructor.
template <typename T>
OclArray<T>::OclArray (const OclArray<T>& a, const dim_vector& dv)
  : dimensions (dv), rep (a.rep),
    slice_ofs (a.slice_ofs), slice_len (a.slice_len), is_logical(a.is_logical)
{
  a.rep->assure_valid ();
  if (dimensions.safe_numel () != a.numel ()) {
    std::string dimensions_str = a.dimensions.str ();
    std::string new_dims_str = dimensions.str ();

    (*current_liboctave_error_handler)
      ("reshape: can't reshape %s ocl_array to %s ocl_array",
        dimensions_str.c_str (), new_dims_str.c_str ());
  }
  (rep->count)++;
  dimensions.chop_trailing_singletons ();
}


// Type conversion constructor. OclArray: only for real->complex conversion.
template <typename T> template <typename U>
OclArray<T>::OclArray (const OclArray<U>& a)
  : dimensions (), rep (nil_rep ()), slice_ofs (0), slice_len (rep->len), is_logical(false)
{
  (*current_liboctave_error_handler)
    ("can't convert ocl array types generally");
}


// Type conversion constructor. OclArray: only for (real,imag)->complex conversion.
template <typename T> template <typename U>
OclArray<T>::OclArray (const OclArray<U>& r, const OclArray<U>& i)
  : dimensions (), rep (nil_rep ()), slice_ofs (0), slice_len (rep->len), is_logical(false)
{
  (*current_liboctave_error_handler)
    ("can't convert ocl array types generally");
}


template <> template <>
OclArray<Complex>::OclArray (const OclArray<double>& a)
  : dimensions (a.dims ()),
    rep (new OclArray<Complex>::OclArrayRep (dimensions.safe_numel ())),
    slice_ofs (0), slice_len (a.numel ()), is_logical(false)
{
  a.rep->assure_valid ();
  assure_valid_array_prog ();

  int kernel_index = kernel_indices [OclArrayKernels::real2complex_r];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, a);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (a.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, double (0.0));

  array_prog.enqueue_kernel (kernel_index, a.slice_len);
}


template <> template <>
OclArray<FloatComplex>::OclArray (const OclArray<float>& a)
  : dimensions (a.dims ()),
    rep (new OclArray<FloatComplex>::OclArrayRep (dimensions.safe_numel ())),
    slice_ofs (0), slice_len (a.numel ()), is_logical(false)
{
  a.rep->assure_valid ();
  assure_valid_array_prog ();

  int kernel_index = kernel_indices [OclArrayKernels::real2complex_r];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, a);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (a.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, float (0.0));

  array_prog.enqueue_kernel (kernel_index, a.slice_len);
}


template <> template <>
OclArray<Complex>::OclArray (const OclArray<double>& r, const OclArray<double>& i)
  : dimensions (r.dims ()),
    rep (new OclArray<Complex>::OclArrayRep (dimensions.safe_numel ())),
    slice_ofs (0), slice_len (r.numel ()), is_logical(false)
{
  if (r.dimensions != i.dimensions)
    ocl_error ("OclArray: dimensions of both arrays must match exactly");

  rep->assure_valid ();
  r.rep->assure_valid ();
  i.rep->assure_valid ();
  assure_valid_array_prog ();

  int kernel_index = kernel_indices [OclArrayKernels::real2complex_ri];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, r);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (r.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, i);
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (i.slice_ofs));

  array_prog.enqueue_kernel (kernel_index, r.slice_len);
}


template <> template <>
OclArray<FloatComplex>::OclArray (const OclArray<float>& r, const OclArray<float>& i)
  : dimensions (r.dims ()),
    rep (new OclArray<FloatComplex>::OclArrayRep (dimensions.safe_numel ())),
    slice_ofs (0), slice_len (r.numel ()), is_logical(false)
{
  if (r.dimensions != i.dimensions)
    ocl_error ("OclArray: dimensions of both arrays must match exactly");

  rep->assure_valid ();
  r.rep->assure_valid ();
  i.rep->assure_valid ();
  assure_valid_array_prog ();

  int kernel_index = kernel_indices [OclArrayKernels::real2complex_ri];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, r);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (r.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, i);
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (i.slice_ofs));

  array_prog.enqueue_kernel (kernel_index, r.slice_len);
}


template <typename T>
void
OclArray<T>::fill (octave_idx_type fill_ofs,
                   octave_idx_type fill_len,
                   const T& val)
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  int kernel_index = kernel_indices [OclArrayKernels::fill];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, val); // val already has correct type

  array_prog.enqueue_kernel (kernel_index, fill_len, fill_ofs);
}


template <typename T>
void
OclArray<T>::fill0 (octave_idx_type fill_ofs,
                    octave_idx_type fill_len,
                    const OclArray<T>& a)
{
  rep->assure_valid ();
  a.rep->assure_valid ();
  assure_valid_array_prog ();

  int kernel_index = kernel_indices [OclArrayKernels::fill0];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, a);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (a.slice_ofs));

  array_prog.enqueue_kernel (kernel_index, fill_len, fill_ofs);
}


template <typename T>
OclArray<T>
OclArray<T>::squeeze (void) const
{
  rep->assure_valid ();

  if (ndims () <= 2)
    return *this;

  dim_vector new_dimensions = dimensions;
  int k = 0;

  for (int i = 0; i < ndims (); i++)
    if (dimensions (i) != 1)
      new_dimensions (k++) = dimensions (i);

  if (k == ndims ())
    return *this;

  if (k == 0)
    new_dimensions = dim_vector (1, 1);
   else if (k == 1)
    new_dimensions = dim_vector (new_dimensions (0), 1);
   else
    new_dimensions.resize (k);

  return OclArray<T> (*this, new_dimensions);
}


template <typename T>
octave_idx_type
OclArray<T>::compute_index (octave_idx_type i, octave_idx_type j) const
{
  return ::compute_index (i, j, dimensions);
}


template <typename T>
octave_idx_type
OclArray<T>::compute_index (octave_idx_type i, octave_idx_type j,
                         octave_idx_type k) const
{
  return ::compute_index (i, j, k, dimensions);
}


template <typename T>
octave_idx_type
OclArray<T>::compute_index (const Array<octave_idx_type>& ra_idx) const
{
  return ::compute_index (ra_idx, dimensions);
}


template <typename T>
OclArray<T>
OclArray<T>::column (octave_idx_type k) const
{
  rep->assure_valid ();

  octave_idx_type r = dimensions (0);
  if (k < 0 || k > dimensions.numel (1))
#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
    octave::err_index_out_of_range (2, 2, k+1, dimensions.numel (1));
#else // for octave versions >= 6.1.0
    octave::err_index_out_of_range (2, 2, k+1, dimensions.numel (1), dimensions);
#endif
  // shallow copy
  return OclArray<T> (*this, dim_vector (r, 1), k*r, k*r + r);
}


template <typename T>
OclArray<T>
OclArray<T>::page (octave_idx_type k) const
{
  rep->assure_valid ();

  octave_idx_type r = dimensions (0), c = dimensions (1), p = r*c;
  if (k < 0 || k > dimensions.numel (2))
#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
    octave::err_index_out_of_range (3, 3, k+1, dimensions.numel (2));
#else // for octave versions >= 6.1.0
    octave::err_index_out_of_range (3, 3, k+1, dimensions.numel (2), dimensions);
#endif
  // shallow copy
  return OclArray<T> (*this, dim_vector (r, c), k*p, k*p + p);
}


template <typename T>
OclArray<T>
OclArray<T>::linear_slice (octave_idx_type lo, octave_idx_type up) const
{
  rep->assure_valid ();

  if (lo < 0)
#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
    octave::err_index_out_of_range (1, 1, lo+1, numel ());
#else // for octave versions >= 6.1.0
    octave::err_index_out_of_range (1, 1, lo+1, numel (), dimensions);
#endif
  if (up > numel ())
#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
    octave::err_index_out_of_range (1, 1, up, numel ());
#else // for octave versions >= 6.1.0
    octave::err_index_out_of_range (1, 1, up, numel (), dimensions);
#endif
  if (up < lo) up = lo;
  // shallow copy
  return OclArray<T> (*this, dim_vector (up - lo, 1), lo, up);
}


template <typename T>
OclArray<T>
OclArray<T>::transpose (void) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  if (ndims () != 2)
    ocl_error ("OclArray::transpose: array has > 2 dimensions");

  octave_idx_type nr = dim1 ();
  octave_idx_type nc = dim2 ();

  if ((nr > 1) && (nc > 1)) {
    OclArray<T> result (dim_vector (nc, nr));

    int kernel_index = kernel_indices [OclArrayKernels::transpose];

    array_prog.set_kernel_arg (kernel_index, 0, result);
    array_prog.set_kernel_arg (kernel_index, 1, *this);
    array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
    array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (nr));
    array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (nc));

    array_prog.enqueue_kernel (kernel_index, slice_len);

    return result;
  } else {
    // Fast transpose for vectors and empty matrices.
    return OclArray<T> (*this, dim_vector (nc, nr));
  }
}


template <typename T>
OclArray<T>
OclArray<T>::hermitian (void) const
{
  if (! is_complex_type ())
    return transpose ();

  rep->assure_valid ();
  assure_valid_array_prog ();

  if (ndims () != 2)
    ocl_error ("OclArray::hermitian: array has > 2 dimensions");

  octave_idx_type nr = dim1 ();
  octave_idx_type nc = dim2 ();

  OclArray<T> result (dim_vector (nc, nr));

  int kernel_index = kernel_indices [OclArrayKernels::hermitian];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (nr));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (nc));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  return result;
}


template <typename T>
Array<T>
OclArray<T>::as_array (void) const
{
  rep->assure_valid ();

  Array<T> result (dimensions);
  rep->copy_to_host (result.fortran_vec(), slice_ofs, slice_len);
  return result;
}


template <typename T>
OclArray<ocl_idx_type>
OclArray<T>::as_index (void) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<ocl_idx_type> result (dimensions);

  int kernel_index = kernel_indices [OclArrayKernels::as_index];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::index (const OclArray<ocl_idx_type>& i) const
{
  rep->assure_valid ();
  i.rep->assure_valid ();
  assure_valid_array_prog ();

  dim_vector rdv = i.dimensions;
  octave_idx_type il = rdv.numel ();
#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  if ((ndims () == 2) && (numel () != 1) && rdv.is_vector ()) {
#else // for octave versions >= 4.4.0
  if ((ndims () == 2) && (numel () != 1) && rdv.isvector ()) {
#endif
    if (columns () == 1)
      rdv = dim_vector (il, 1);
    else if (rows () == 1)
      rdv = dim_vector (1, il);
  }

  OclArray<T> result (rdv);

  int kernel_index = kernel_indices [OclArrayKernels::index];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (slice_len));
  array_prog.set_kernel_arg (kernel_index, 4, i);
  array_prog.set_kernel_arg (kernel_index, 5, octave_uint64 (i.slice_ofs));

  array_prog.enqueue_kernel (kernel_index, i.slice_len);

  return result;
}


template <typename T>
void
OclArray<T>::assign (const OclArray<ocl_idx_type>& i,
                     const T& rhs)
{
  rep->assure_valid ();
  i.rep->assure_valid ();
  assure_valid_array_prog ();

  make_unique ();

  int kernel_index = kernel_indices [OclArrayKernels::assign_el];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_len));
  array_prog.set_kernel_arg (kernel_index, 3, i);
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (i.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 5, rhs);

  array_prog.enqueue_kernel (kernel_index, i.slice_len);
}


template <typename T>
void
OclArray<T>::assign (const OclArray<ocl_idx_type>& i,
                     const OclArray<T>& rhs)
{
  rep->assure_valid ();
  i.rep->assure_valid ();
  rhs.rep->assure_valid ();
  assure_valid_array_prog ();

  make_unique ();

  if (rhs.numel () == 1) {

    int kernel_index = kernel_indices [OclArrayKernels::assign0];

    array_prog.set_kernel_arg (kernel_index, 0, *this);
    array_prog.set_kernel_arg (kernel_index, 1, octave_uint64 (slice_ofs));
    array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_len));
    array_prog.set_kernel_arg (kernel_index, 3, i);
    array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (i.slice_ofs));
    array_prog.set_kernel_arg (kernel_index, 5, rhs);
    array_prog.set_kernel_arg (kernel_index, 6, octave_uint64 (rhs.slice_ofs));

    array_prog.enqueue_kernel (kernel_index, i.slice_len);

  } else if (rhs.numel () != i.numel ()) {

    octave::err_nonconformant ("=", i.numel (), rhs.numel ());

  } else {

    int kernel_index = kernel_indices [OclArrayKernels::assign];

    array_prog.set_kernel_arg (kernel_index, 0, *this);
    array_prog.set_kernel_arg (kernel_index, 1, octave_uint64 (slice_ofs));
    array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_len));
    array_prog.set_kernel_arg (kernel_index, 3, i);
    array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (i.slice_ofs));
    array_prog.set_kernel_arg (kernel_index, 5, rhs);
    array_prog.set_kernel_arg (kernel_index, 6, octave_uint64 (rhs.slice_ofs));

    array_prog.enqueue_kernel (kernel_index, i.slice_len); // i.slice_len == rhs.slice_len

  }
}


template <typename T>
void
OclArray<T>::assign_logical (const OclArray<T>& i,
                             const T& rhs)
{
  rep->assure_valid ();
  i.rep->assure_valid ();
  assure_valid_array_prog ();

  make_unique ();

  if (numel () != i.numel ()) {
    octave::err_nonconformant ("=", i.numel (), numel ());
    return;
  }

  int kernel_index = kernel_indices [OclArrayKernels::assign_el_logind];

  array_prog.set_kernel_arg (kernel_index, 0, *this);
  array_prog.set_kernel_arg (kernel_index, 1, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 2, i);
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (i.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 4, rhs);

  array_prog.enqueue_kernel (kernel_index, i.slice_len);
}


template <typename T>
OclArray<T>
OclArray<T>::index (const idx_vector& i) const
{
  Array<idx_vector> ia (dim_vector (1,1));
  ia (0) = i;
  return index (ia);
}


template <typename T>
OclArray<T>
OclArray<T>::index (const idx_vector& i, const idx_vector& j) const
{
  Array<idx_vector> ia (dim_vector (2,1));
  ia (0) = i;
  ia (1) = j;
  return index (ia);
}


template <typename T>
void
OclArray<T>::assign (const idx_vector& i,
                     const T& rhs)
{
  Array<idx_vector> ia (dim_vector (1,1));
  ia (0) = i;
  assign (ia, rhs);
}


template <typename T>
void
OclArray<T>::assign (const idx_vector& i, const idx_vector& j,
                     const T& rhs)
{
  Array<idx_vector> ia (dim_vector (2,1));
  ia (0) = i;
  ia (1) = j;
  assign (ia, rhs);
}


template <typename T>
void
OclArray<T>::assign (const idx_vector& i,
                     const OclArray<T>& rhs)
{
  Array<idx_vector> ia (dim_vector (1,1));
  ia (0) = i;
  assign (ia, rhs);
}


template <typename T>
void
OclArray<T>::assign (const idx_vector& i, const idx_vector& j,
                     const OclArray<T>& rhs)
{
  Array<idx_vector> ia (dim_vector (2,1));
  ia (0) = i;
  ia (1) = j;
  assign (ia, rhs);
}


template <typename T>
void
OclArray<T>::index_helper (const Array<idx_vector>& ia,
                           dim_vector& dv,
                           dim_vector& rdv,
                           bool& all_colons,
                           octave_idx_type& l,
                           octave_idx_type& u) const
{
  int ial = ia.numel ();

  dv = dimensions.redim (ial);
  if (ial == 1)
    dv = dv.redim (2);

  if (ial == 1)
    rdv = dim_vector (1,1);
  else
    rdv = dim_vector::alloc (ial);

  all_colons = true;
  int first_range = -1, first_scalar = -1;
  octave_idx_type s = 1;
  l = 0;
  u = 1;

  for (int i = 0; i < ial; i++) {

    if (ia (i).extent (dv (i)) != dv (i))
#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
      octave::err_index_out_of_range (ial, i+1, ia (i).extent (dv (i)), dv (i)); // throws
#else // for octave versions >= 6.1.0
      octave::err_index_out_of_range (ial, i+1, ia (i).extent (dv (i)), dv (i), dimensions); // throws
#endif

    rdv (i) = ia (i).length (dv (i));

    idx_vector::idx_class_type idx_class = ia (i).idx_class ();

    // only allowed indexing: A(:,:,...,r,...,s,s)
    if ((idx_class == idx_vector::class_colon) || (ia (i).is_colon_equiv (dv (i)))) {
      if (first_range >= 0) // A(...,r,...,:,...)
        ocl_error ("OclArray: octave indexing must result in a contiguous memory range");
      u = s * dv (i);
    } else if (idx_class == idx_vector::class_range) {
      if ((first_range >= 0) // A(...,r,...,r,...)
          || (first_scalar >= 0) // A(...,s,...,r,...)
          || (ia (i).xelem (1) - ia (i).xelem (0) != 1)) // step != 1
        ocl_error ("OclArray: octave indexing must result in a contiguous memory range");
      if (first_range < 0)
        first_range = i;
      all_colons = false;
      // s=length of block of colons
      l = s * ia (i).xelem (0);
      u = s * ia (i).xelem (rdv (i));
    } else if (idx_class == idx_vector::class_scalar) {
      if (first_scalar < 0)
        first_scalar = i;
      if (first_range < 0)
        first_range = i;
      all_colons = false;
      l += s * ia (i).xelem (0);
      u += s * ia (i).xelem (0);
    } else
      ocl_error ("OclArray: octave indexing is only possible with colon, scalar, or range");

    s *= dv (i);
  }

  // correct dimensions for single index and special cases
  if ((ial == 1) && ((ndims () != 2) || (columns () != 1)))
    rdv = dim_vector (1, rdv (0));

  rdv.chop_trailing_singletons ();
}


template <typename T>
OclArray<T>
OclArray<T>::index (const Array<idx_vector>& ia) const
{
  rep->assure_valid ();

  int ial = ia.numel ();

  if (ial == 0)
    return OclArray<T> ();

  dim_vector dv, rdv;
  bool all_colons;
  octave_idx_type l, u;

  index_helper (ia, dv, rdv, all_colons, l, u);

  if (all_colons) {
    dv.chop_trailing_singletons ();
    return OclArray<T> (*this, dv); // A(:,:,...,:) produces a shallow copy.
  } else
    return OclArray<T> (*this, rdv, l, u); // produce a shallow copy.
}


template <typename T>
void
OclArray<T>::assign (const Array<idx_vector>& ia,
                     const T& rhs)
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  int ial = ia.numel ();

  if (ial == 0)
    return;

  dim_vector dv, rdv;
  bool all_colons;
  octave_idx_type l, u;

  index_helper (ia, dv, rdv, all_colons, l, u);

  if (all_colons && ((rep->count) > 1))
    *this = OclArray<T> (dimensions, rhs);
  else {
    make_unique ();
    fill (l, u-l, rhs); // fill LHS' memory range from l to u with rhs.
  }
}


template <typename T>
void
OclArray<T>::assign (const Array<idx_vector>& ia,
                     const OclArray<T>& rhs)
{
  // OclArray: No special case when all dimensions are initially zero, since resizing is not allowed.

  rep->assure_valid ();
  rhs.rep->assure_valid ();
  assure_valid_array_prog ();

  int ial = ia.numel ();

  if (ial == 0)
    return;

  dim_vector dv, rdv; // rdv is used here for indexed part of *this (NOT the fixed extent of the assignment result)
  bool all_colons;
  octave_idx_type l, u;

  index_helper (ia, dv, rdv, all_colons, l, u);

  rdv.chop_all_singletons ();

  dim_vector rhdv = rhs.dims ();
  rhdv.chop_all_singletons ();

  bool isfill = (rhs.numel () == 1);
  bool match;

  // Check whether LHS and RHS match, disregarding singleton dims.
  if (ial == 1)
    match = (rhdv.numel () == rdv.numel ());
  else
    match = (rhdv == rdv);
  match = match || isfill;

  if (!match) {
    if (ial == 1)
      octave::err_nonconformant ("=", rdv.numel (), rhdv.numel ());
    else
      octave::err_nonconformant ("=", rdv, rhdv);
  }

  if (isfill) {
    if (all_colons && ((rep->count) > 1))
      *this = OclArray<T> (dimensions);
    else
      make_unique ();
    fill0 (l, u-l, rhs); // fill LHS' memory range from l to u with rhs(0).
  } else if (all_colons) {
    *this = rhs.reshape (dimensions); // A(:,:,...,:) = X makes a shallow copy.
  } else {
    make_unique ();
    rep->copy_from_oclbuffer (*(rhs.rep), rhs.slice_ofs, l, u-l); // copy RHS to LHS' memory range from l to u.
  }
}


template <typename T>
OclArray<T>
OclArray<T>::cat (int dim, octave_idx_type n, const OclArray<T> *array_list)
{
  assure_valid_array_prog ();

  if (dim < 0)
    ocl_error ("OclArray::cat: invalid dimension");

  if (n <= 0)
    return OclArray<T> ();

  array_list [0].rep->assure_valid ();

  if (n == 1)
    return array_list [0];

  dim_vector dv = array_list [0].dims ();
  if (dim >= dv.length ())
    dv = dv.redim (dim+1);
  int ndim = dv.length ();
  dim_vector dvc = dv;
  dvc (dim) = 1;

  for (octave_idx_type i = 1; i < n; i++) {
    array_list [i].rep->assure_valid ();

    dim_vector dvi = array_list [i].dims ();
    if (ndim >= dvi.length ())
      dvi = dvi.redim (ndim);
    dv (dim) += dvi (dim);
    dvi (dim) = 1;
    if (dvc.numel () == 0)
      dvc = dvi; // if previous matrices were empty, try this for size comparisons
    if (dvc != dvi)
      ocl_error ("OclArray::cat: dimension mismatch");
  }

  OclArray<T> result (dv);

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  if (result.is_empty ())
#else // for octave versions >= 4.4.0
  if (result.isempty ())
#endif
    return result;

  octave_idx_type spdim = 1;
  for (octave_idx_type i = 0; i < dim; i++)
    spdim *= dv (i);

  octave_idx_type offset = 0;

  int kernel_index = kernel_indices [OclArrayKernels::cat];

  for (octave_idx_type i = 0; i < n; i++) {
    octave_quit ();

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
    if (array_list [i].is_empty ())
#else // for octave versions >= 4.4.0
    if (array_list [i].isempty ())
#endif
      continue;
    dim_vector dvi = array_list [i].dims ();
    if (ndim >= dvi.length ())
      dvi = dvi.redim (ndim);

    octave_idx_type fac1, fac2;
    fac1 = spdim * dvi (dim);
    fac2 = spdim * dv (dim);

    array_prog.set_kernel_arg (kernel_index, 0, result);
    array_prog.set_kernel_arg (kernel_index, 1, array_list [i]);
    array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (array_list [i].slice_ofs));
    array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (offset * spdim));
    array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (fac1));
    array_prog.set_kernel_arg (kernel_index, 5, octave_uint64 (fac2));

    array_prog.enqueue_kernel (kernel_index, dvi.numel ());

    offset += dvi (dim);
  }

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::eye (octave_idx_type r, octave_idx_type c)
{
  assure_valid_array_prog ();

  if (r < 0)
    ocl_error ("OclArray::eye: invalid size");
  if (c < 0)
    c = r;

  dim_vector dv (r, c);

  OclArray<T> result (dv);

  int kernel_index = kernel_indices [OclArrayKernels::eye];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, octave_uint64 (r+1));
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (r * r));

  array_prog.enqueue_kernel (kernel_index, dv.numel ());

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::linspace (T base, T limit, octave_idx_type n)
{
  assure_valid_array_prog ();

  if (n < 2)
    return OclArray<T> (dim_vector (1, 1), limit);

  dim_vector dv (1, n);

  OclArray<T> result (dv);

  int kernel_index = kernel_indices [OclArrayKernels::linspace];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, base);
  array_prog.set_kernel_arg (kernel_index, 2, limit);
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (n));

  array_prog.enqueue_kernel (kernel_index, dv.numel ());

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::logspace (T a, T b, octave_idx_type n)
{
  assure_valid_array_prog ();

  if (is_integer_type () || is_complex_type ())
    ocl_error ("OclArray::logspace: not possible with this type");

  if (n < 2)
    return OclArray<T> (dim_vector (1, 1), b);

  dim_vector dv (1, n);

  OclArray<T> result (dv);

  int kernel_index = kernel_indices [OclArrayKernels::logspace];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, a);
  array_prog.set_kernel_arg (kernel_index, 2, b);
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (n));

  array_prog.enqueue_kernel (kernel_index, dv.numel ());

  return result;
}


template <typename T>
std::vector< OclArray<T> >
OclArray<T>::ndgrid (const std::vector< OclArray<T> > array_list)
{
  assure_valid_array_prog ();

  if (array_list.empty ())
    return std::vector< OclArray<T> > ();

  std::vector< OclArray<T> > args = array_list;
  if (args.size () == 1) {
    args.resize (2);
    args [1] = args [0];
  }

  int ndim = args.size ();
  for (int i = 0; i < ndim; i++) {
    OclArray<T> ai = args [i];
    if ((ai.ndims () > 2) || ((ai.rows () != 1) && (ai.columns () != 1)))
      ocl_error ("OclArray::ndgrid: all input arguments must be vectors");
    ai.rep->assure_valid ();
  }

  dim_vector dv;
  dv = dv.redim (ndim);
  for (int i = 0; i < ndim; i++)
    dv (i) = args [i].numel ();

  int kernel_index = kernel_indices [OclArrayKernels::ndgrid1];

  std::vector< OclArray<T> > result (ndim);
  octave_idx_type div1 = 1;

  for (int i = 0; i < ndim; i++) {
    octave_quit ();

    result [i] = OclArray<T> (dv);
    octave_idx_type div2 = dv (i);

    array_prog.set_kernel_arg (kernel_index, 0, result [i]);
    array_prog.set_kernel_arg (kernel_index, 1, args [i]);
    array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (args [i].slice_ofs));
    array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (div1));
    array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (div2));

    array_prog.enqueue_kernel (kernel_index, dv.numel ());

    div1 *= div2;
  }

  return result;
}


template <typename T>
std::vector< OclArray<T> >
OclArray<T>::meshgrid (const std::vector< OclArray<T> > array_list)
{
  OclArray<T> tmp;
  std::vector< OclArray<T> > args = array_list;
  if (args.size () >= 2) {
    tmp = args [1];
    args [1] = args [0];
    args [0] = tmp;
  }

  std::vector< OclArray<T> > result;
  result = ndgrid (args);

  tmp = result [1];
  result [1] = result [0];
  result [0] = tmp;
  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::repmat1 (int dim, octave_idx_type n) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  if (dim < 0)
    ocl_error ("OclArray::repmat: invalid dimension");
  if (n <= 0)
    return OclArray<T> ();
  if (n == 1)
    return *this;

  octave_idx_type fac1 = 1, fac2, fac3;

  dim_vector dv = dims ();
  if (dim >= dv.length ())
    dv = dv.redim (dim+1);
  for (int i = 0; i < dim; i++)
    fac1 *= dv (i);
  fac2 = dv (dim);
  dv (dim) *= n;
  fac3 = dv (dim);

  OclArray<T> result (dv);

  int kernel_index = kernel_indices [OclArrayKernels::repmat1];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (fac1));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (fac2));
  array_prog.set_kernel_arg (kernel_index, 5, octave_uint64 (fac3));

  array_prog.enqueue_kernel (kernel_index, dv.numel ());

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::repmat (const dim_vector& dv) const
{
  if (dv.any_zero () || dv.any_neg ())
    return OclArray<T> ();

  OclArray<T> result = *this;

  for (int i = 0; i < dv.length (); i++) {
    octave_quit ();

    if (dv (i) > 1)
      result = result.repmat1 (i, dv (i));
  }

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::std (int opt, int dim) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  dim_vector new_dimensions;
  octave_idx_type len, fac, n;

  dim_wise_op_newdims (dim, new_dimensions, len, fac);
  n = (opt == 0) ? len-1 : len;

  OclArray<T> result (new_dimensions);

  int kernel_index = kernel_indices [OclArrayKernels::std];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (len));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (fac));
  array_prog.set_kernel_arg (kernel_index, 5, octave_uint64 (n));

  array_prog.enqueue_kernel (kernel_index, slice_len / len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map (octave_base_value::unary_mapper_t umap) const
{
  if (! is_complex_type ()) {

    switch (umap)
    {
      case octave_base_value::umap_imag:
        return OclArray<T> (dimensions, 0);

      case octave_base_value::umap_real:
      case octave_base_value::umap_conj:
        return *this;

      default: ; // handled below (this line only prevents compiler warnings)
    }

  } else {

    switch (umap)
    {
      case octave_base_value::umap_conj:
        return map (OclArrayKernels::conj);

      case octave_base_value::umap_real:
      case octave_base_value::umap_imag:
      case octave_base_value::umap_abs:
      case octave_base_value::umap_angle:
      case octave_base_value::umap_arg:
      case octave_base_value::umap_isfinite:
      case octave_base_value::umap_isinf:
      case octave_base_value::umap_isnan:
        ocl_error ("not applicable to type OclArray of this class"); // use function map_c2r instead

      case octave_base_value::umap_cbrt:
      case octave_base_value::umap_erf:
      case octave_base_value::umap_erfc:
      case octave_base_value::umap_expm1:
      case octave_base_value::umap_gamma:
      case octave_base_value::umap_lgamma:
      case octave_base_value::umap_log1p:
        ocl_error ("not applicable to type OclArray of this class"); // not implemented

      default: ; // handled below (this line only prevents compiler warnings)
    }

  }

  if (is_integer_type ()) {

    switch (umap)
    {
      case octave_base_value::umap_abs:
        if (is_uint_type ())
          return *this;
        else
          return map (OclArrayKernels::abs);

      case octave_base_value::umap_ceil:
      case octave_base_value::umap_fix:
      case octave_base_value::umap_floor:
      case octave_base_value::umap_round:
        return *this;

      case octave_base_value::umap_isinf:
      case octave_base_value::umap_isnan:
        return OclArray<T> (dimensions, 0);

      case octave_base_value::umap_isfinite:
        return OclArray<T> (dimensions, 1);

      default:
        ocl_error ("not applicable to type OclArray of this class");
    }

  }

  // float or double type
#define MAP_ENTRY(UMAP, KERNEL) \
  case octave_base_value::UMAP: return map (OclArrayKernels::KERNEL);

  switch (umap)
  {
    MAP_ENTRY(umap_abs, fabs)
    MAP_ENTRY(umap_acos, acos)
    MAP_ENTRY(umap_acosh, acosh)
    MAP_ENTRY(umap_asin, asin)
    MAP_ENTRY(umap_asinh, asinh)
    MAP_ENTRY(umap_atan, atan)
    MAP_ENTRY(umap_atanh, atanh)
    MAP_ENTRY(umap_cbrt, cbrt)
    MAP_ENTRY(umap_ceil, ceil)
    MAP_ENTRY(umap_cos, cos)
    MAP_ENTRY(umap_cosh, cosh)
    MAP_ENTRY(umap_erf, erf)
    MAP_ENTRY(umap_erfc, erfc)
    MAP_ENTRY(umap_exp, exp)
    MAP_ENTRY(umap_expm1, expm1)
    MAP_ENTRY(umap_isfinite, isfinite)
    MAP_ENTRY(umap_fix, fix)
    MAP_ENTRY(umap_floor, floor)
    MAP_ENTRY(umap_isinf, isinf)
    MAP_ENTRY(umap_isnan, isnan)
    MAP_ENTRY(umap_gamma, tgamma)
    MAP_ENTRY(umap_lgamma, lgamma)
    MAP_ENTRY(umap_log, log)
    MAP_ENTRY(umap_log2, log2)
    MAP_ENTRY(umap_log10, log10)
    MAP_ENTRY(umap_log1p, log1p)
    MAP_ENTRY(umap_round, round)
    MAP_ENTRY(umap_signum, sign)
    MAP_ENTRY(umap_sin, sin)
    MAP_ENTRY(umap_sinh, sinh)
    MAP_ENTRY(umap_sqrt, sqrt)
    MAP_ENTRY(umap_tan, tan)
    MAP_ENTRY(umap_tanh, tanh)

    default:
      ocl_error ("not applicable to type OclArray of this class");
  }

/*
no_opencl_support:
    umap_erfinv,
    umap_erfcinv,
    umap_erfcx,
    umap_erfi,
    umap_dawson,
    umap_isna,
    umap_roundb
    umap_x...
*/
#undef MAP_ENTRY
}


template <> template <>
OclArray<double>
OclArray<Complex>::map_c2r (OclArrayKernels::Kernel kernel) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<double> result (dimensions);

  int kernel_index = kernel_indices [kernel];

  if (kernel_index < 0)
    ocl_error ("not applicable to type OclArray of this class");

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  return result;
}


template <> template <>
OclArray<float>
OclArray<FloatComplex>::map_c2r (OclArrayKernels::Kernel kernel) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<float> result (dimensions);

  int kernel_index = kernel_indices [kernel];

  if (kernel_index < 0)
    ocl_error ("not applicable to type OclArray of this class");

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  return result;
}


template <> template <>
OclArray<double>
OclArray<Complex>::map_c2r (octave_base_value::unary_mapper_t umap) const
{
#define MAP_ENTRY(UMAP, KERNEL) \
  case octave_base_value::UMAP: return map_c2r<double> (OclArrayKernels::KERNEL);

  switch (umap)
  {
    MAP_ENTRY(umap_real, real)
    MAP_ENTRY(umap_imag, imag)
    MAP_ENTRY(umap_abs, fabs)
    MAP_ENTRY(umap_angle, arg)
    MAP_ENTRY(umap_arg, arg)
    MAP_ENTRY(umap_isfinite, isfinite)
    MAP_ENTRY(umap_isinf, isinf)
    MAP_ENTRY(umap_isnan, isnan)

    default:
      ocl_error ("not applicable to type OclArray of this class");
  }
#undef MAP_ENTRY
}


template <> template <>
OclArray<float>
OclArray<FloatComplex>::map_c2r (octave_base_value::unary_mapper_t umap) const
{
#define MAP_ENTRY(UMAP, KERNEL) \
  case octave_base_value::UMAP: return map_c2r<float> (OclArrayKernels::KERNEL);

  switch (umap)
  {
    MAP_ENTRY(umap_real, real)
    MAP_ENTRY(umap_imag, imag)
    MAP_ENTRY(umap_abs, fabs)
    MAP_ENTRY(umap_angle, arg)
    MAP_ENTRY(umap_arg, arg)
    MAP_ENTRY(umap_isfinite, isfinite)
    MAP_ENTRY(umap_isinf, isinf)
    MAP_ENTRY(umap_isnan, isnan)

    default:
      ocl_error ("not applicable to type OclArray of this class");
  }
#undef MAP_ENTRY
}


template <typename T>
void
OclArray<T>::dim_wise_op_newdims (int dim,
                                  dim_vector& new_dimensions,
                                  octave_idx_type& len,
                                  octave_idx_type& fac) const
{
  new_dimensions = dimensions;

  if (dim < 0)
    dim = dimensions.first_non_singleton ();

  if (dim < ndims ()) {
    len = dimensions (dim);
    fac = 1;
    for (int i=0; i<dim; i++)
      fac *= dimensions (i);
    new_dimensions (dim) = 1;
  } else {
    len = 1;
    fac = slice_len;
  }

}


template <typename T>
OclArray<T>
OclArray<T>::map (OclArrayKernels::Kernel kernel) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result (dimensions);

  int kernel_index = kernel_indices [kernel];

  if (kernel_index < 0)
    ocl_error ("not applicable to type OclArray of this class");

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map1 (OclArrayKernels::Kernel kernel, const T& par) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result (dimensions);

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, par);

  array_prog.enqueue_kernel (kernel_index, slice_len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map1_inplace (OclArrayKernels::Kernel kernel, const T& par)
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result;

  if (is_shared () || (slice_len != rep->len)) {
    result = OclArray<T> (dimensions);
  } else {
    // slice_ofs == 0 assured
    result = *this;
  }

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, par);

  array_prog.enqueue_kernel (kernel_index, slice_len);

  *this = result;

  return *this;
}


template <typename T>
OclArray<T>
OclArray<T>::map1r (OclArrayKernels::Kernel kernel, int dim) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  dim_vector new_dimensions;
  octave_idx_type len, fac;

  dim_wise_op_newdims (dim, new_dimensions, len, fac);

  OclArray<T> result (new_dimensions);

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (len));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (fac));

  array_prog.enqueue_kernel (kernel_index, slice_len / len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map1re (OclArrayKernels::Kernel kernel, int dim) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  dim_vector new_dimensions;
  octave_idx_type len, fac;

  dim_wise_op_newdims (dim, new_dimensions, len, fac);

  OclArray<T> result (dimensions);

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (len));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (fac));

  array_prog.enqueue_kernel (kernel_index, slice_len / len);

  return result;
}


template <typename T>
OclArray<ocl_idx_type>
OclArray<T>::map1rf (OclArrayKernels::Kernel kernel, int dim) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  dim_vector new_dimensions;
  octave_idx_type len, fac;

  dim_wise_op_newdims (dim, new_dimensions, len, fac);

  OclArray<ocl_idx_type> result (new_dimensions);

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (len));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (fac));

  array_prog.enqueue_kernel (kernel_index, slice_len / len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map1ri (OclArrayKernels::Kernel kernel, int dim, OclArray<ocl_idx_type> *indices) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  dim_vector new_dimensions;
  octave_idx_type len, fac;

  dim_wise_op_newdims (dim, new_dimensions, len, fac);

  OclArray<T> result (new_dimensions);
  OclArray<ocl_idx_type> result_indices;
  if (indices)
    result_indices = OclArray<ocl_idx_type> (new_dimensions);

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  if (indices)
    array_prog.set_kernel_arg (kernel_index, 1, result_indices);
  else
    array_prog.set_kernel_arg (kernel_index, 1, result); // as indicator for unused indices
  array_prog.set_kernel_arg (kernel_index, 2, *this);
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (len));
  array_prog.set_kernel_arg (kernel_index, 5, octave_uint64 (fac));

  array_prog.enqueue_kernel (kernel_index, slice_len / len);

  if (indices)
    *indices = result_indices;

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map1rie (OclArrayKernels::Kernel kernel, int dim, OclArray<ocl_idx_type> *indices) const
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  dim_vector new_dimensions;
  octave_idx_type len, fac;

  dim_wise_op_newdims (dim, new_dimensions, len, fac);

  OclArray<T> result (dimensions);
  OclArray<ocl_idx_type> result_indices;
  if (indices)
    result_indices = OclArray<ocl_idx_type> (dimensions);

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  if (indices)
    array_prog.set_kernel_arg (kernel_index, 1, result_indices);
  else
    array_prog.set_kernel_arg (kernel_index, 1, result); // as indicator for unused indices
  array_prog.set_kernel_arg (kernel_index, 2, *this);
  array_prog.set_kernel_arg (kernel_index, 3, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (len));
  array_prog.set_kernel_arg (kernel_index, 5, octave_uint64 (fac));

  array_prog.enqueue_kernel (kernel_index, slice_len / len);

  if (indices)
    *indices = result_indices;

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map2s (OclArrayKernels::Kernel kernel, const OclArray<T>& s2) const
{
  if (s2.dimensions != dimensions)
    ocl_error ("OclArray: dimensions of both arrays must match exactly");

  if (is_complex_type () && (kernel == OclArrayKernels::atan2))
    ocl_error ("not applicable to type OclArray of this complex class");

  rep->assure_valid ();
  s2.rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result (dimensions);

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, s2);
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (s2.slice_ofs));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::map2s_inplace (OclArrayKernels::Kernel kernel, const OclArray<T>& s2)
{
  if (s2.dimensions != dimensions)
    ocl_error ("OclArray: dimensions of both arrays must match exactly");

  rep->assure_valid ();
  s2.rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result;

  if (is_shared () || (slice_len != rep->len)) {
    result = OclArray<T> (dimensions);
  } else {
    // slice_ofs == 0 assured
    result = *this;
  }

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, s2);
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (s2.slice_ofs));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  *this = result;

  return *this;
}


template <typename T>
OclArray<T>
OclArray<T>::map2sf (OclArrayKernels::Kernel kernel,
                     const OclArray<T>& s1,
                     const OclArray<T>& s2,
                     const T& par,
                     unsigned long fcn)
{
  if (s1.dimensions != s2.dimensions)
    ocl_error ("OclArray: dimensions of both arrays must match exactly");

  s1.rep->assure_valid ();
  s2.rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result (s1.dimensions);

  if ((kernel == OclArrayKernels::compare) || (kernel == OclArrayKernels::logic))
    result.is_logical = true;

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, s1);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (s1.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, s2);
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (s2.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 5, par);
  array_prog.set_kernel_arg (kernel_index, 6, octave_uint64 (fcn));

  array_prog.enqueue_kernel (kernel_index, s1.slice_len);

  return result;
}


template <typename T>
OclArray<T>
OclArray<T>::mtimes (const OclArray<T>& s2) const
{
  if ((ndims () != 2) || (s2.ndims () != 2))
    ocl_error ("OclArray: operands must both be 2-dim arrays, or vectors, for matrix multiplication");
  if (dim2 () != s2.dim1 ())
    ocl_error ("OclArray: mismatch in operands' sizes for matrix multiplication");

  dim_vector new_dimensions (dim1 (), s2.dim2 ());

  rep->assure_valid ();
  s2.rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result (new_dimensions);

  int kernel_index = kernel_indices [OclArrayKernels::mtimes];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 3, s2);
  array_prog.set_kernel_arg (kernel_index, 4, octave_uint64 (s2.slice_ofs));
  array_prog.set_kernel_arg (kernel_index, 5, octave_uint64 (dim1 ()));
  array_prog.set_kernel_arg (kernel_index, 6, octave_uint64 (dim2 ()));

  array_prog.enqueue_kernel (kernel_index, new_dimensions.numel ());

  return result;
}


template <typename T>
void
OclArray<T>::map_inplace (OclArrayKernels::Kernel kernel)
{
  rep->assure_valid ();
  assure_valid_array_prog ();

  OclArray<T> result;

  if (is_shared () || (slice_len != rep->len)) {
    result = OclArray<T> (dimensions);
  } else {
    // slice_ofs == 0 assured
    result = *this;
  }

  int kernel_index = kernel_indices [kernel];

  array_prog.set_kernel_arg (kernel_index, 0, result);
  array_prog.set_kernel_arg (kernel_index, 1, *this);
  array_prog.set_kernel_arg (kernel_index, 2, octave_uint64 (slice_ofs));

  array_prog.enqueue_kernel (kernel_index, slice_len);

  *this = result;
}


template <typename T>
void
OclArray<T>::print_info (std::ostream& os, const std::string& prefix) const
{
  os << prefix << "rep address: " << rep << '\n'
     << prefix << "rep->len:    " << rep->len << '\n'
     << prefix << "rep->buffer: " << rep->get_ocl_buffer () << '\n'
     << prefix << "rep->count:  " << rep->count << '\n'
     << prefix << "slice_ofs:   " << slice_ofs << '\n'
     << prefix << "slice_len:   " << slice_len << '\n';
}


template <typename T>
void
OclArray<T>::assure_valid_array_prog (void)
{
  if (array_prog.is_valid ())
    return;

  kernel_indices.resize (OclArrayKernels::max_array_prog_kernels);
  for (int i = 0; i < OclArrayKernels::max_array_prog_kernels; i++)
    kernel_indices [i] = -1;

  std::string build_options, oclc_type;
  oclc_type = get_type_str_oclc ();
  build_options += "-DTYPE=" + oclc_type + " "; // build for specific array type
  if ((oclc_type == "double") || (oclc_type == "float") || (is_complex_type ()))
    build_options += "-DFLOATINGPOINT ";
  else
    build_options += "-DINTEGER ";
  if (is_complex_type ()) {
    std::string oclc_type1 = oclc_type.substr (0, oclc_type.length () - 1);
    build_options += "-DTYPE1=" + oclc_type1 + " "; // also define non-complex array type
    build_options += "-DCOMPLEX ";
  }

  assure_opencl_context ();
  if ((! opencl_context_is_fp64 ()) && (oclc_type == "double"))
    ocl_error ("OclArray: currently selected OpenCL context is not capable of operating on OCL arrays of 'double' type");

  array_prog = OclProgram (ocl_array_prog_source, build_options);

  for (int i = 0; i < OclArrayKernels::max_array_prog_kernels; i++)
    kernel_indices [i] =
      array_prog.get_kernel_index (
        get_array_prog_kernel_name (OclArrayKernels::Kernel (i)),
        false // non-strict kernel index lookup
      );

  // now it is possible to obtain a kernel index (or -1) by, e.g.
  //   kernel_indices [OclArrayKernels::fmad1]
}


// ---------- OclArray<T> non-members


template <typename T>
std::ostream&
operator << (std::ostream& os, const OclArray<T>& a)
{
  dim_vector a_dims = a.dims ();
  int n_dims = a_dims.length ();

  os << "  " << n_dims << "-dimensional OCL array";
  if (n_dims)
    os << " (" << a_dims.str () << ")";
  os << " of class " << a.get_type_str_oct () << " (" << a.get_type_str_oclc () << ")\n";

  return os;
}


// ---------- OclArray<T> instantiations


template <typename T>
void OclArray<T>::instantiation_guard ()
{
  // This guards against accidental implicit instantiations.
  // OclArray<T> instances should always be explicit and use INSTANTIATE_OCLARRAY.
  T::__xXxXx__ ();
}


#define INSTANTIATE_OCLARRAY( T ) \
  template std::ostream& operator << (std::ostream& os, const OclArray<T>& a); \
  template <> void OclArray<T>::instantiation_guard () {} \
  template class OclArray<T>;


INSTANTIATE_OCLARRAY (octave_int8  );
INSTANTIATE_OCLARRAY (octave_int16 );
INSTANTIATE_OCLARRAY (octave_int32 );
INSTANTIATE_OCLARRAY (octave_int64 );
INSTANTIATE_OCLARRAY (octave_uint8 );
INSTANTIATE_OCLARRAY (octave_uint16);
INSTANTIATE_OCLARRAY (octave_uint32);
INSTANTIATE_OCLARRAY (octave_uint64);
INSTANTIATE_OCLARRAY (float        );
INSTANTIATE_OCLARRAY (double       );
INSTANTIATE_OCLARRAY (FloatComplex );
INSTANTIATE_OCLARRAY (Complex      );
