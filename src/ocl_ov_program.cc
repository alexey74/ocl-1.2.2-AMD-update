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

#include <octave/oct.h>

#include "ocl_octave_versions.h"
#include "ocl_lib.h"
#include "ocl_program.h"
#include "ocl_ov_matrix.h"
#include "ocl_ov_types.h"


#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
#define ISREAL is_real_type
#else // for octave versions >= 4.4.0
#define ISREAL isreal
#endif


// ---------- declaration and definition of octave_ocl_program class


class
octave_ocl_program : public octave_base_value
{
protected:

  enum InArgOpt { OptMakeUnique, OptSliceOfs, OptSubBuffer };

  class SubBuffer { // helper class for exception-safe destruction of OpenCL sub-buffers
    public:
      void *subbuf;

      SubBuffer () : subbuf (0) {}

      template <typename T>
      SubBuffer (const OclArray<T>& array) : subbuf (0) {
        cl_mem_flags mem_flags = CL_MEM_READ_ONLY; // we use sub-buffers only for kernel input arrays
        cl_buffer_region buffer_create_info;
        buffer_create_info.origin = array.slice_ofs * sizeof (T); // MAJOR issues with CL_MISALIGNED_SUB_BUFFER_OFFSET with many drivers / hardware
        buffer_create_info.size = array.slice_len * sizeof (T);
        cl_mem mem_obj = clCreateSubBuffer ((cl_mem) array.rep->get_ocl_buffer (),
                                            mem_flags,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            & buffer_create_info,
                                            & last_error);
        ocl_check_error ("clCreateSubBuffer");
        subbuf = (void *) mem_obj;
      }

      ~SubBuffer () { if (subbuf) clReleaseMemObject ((cl_mem) subbuf); }
  };

public:

  octave_ocl_program (void)
    : octave_base_value (), program () { }

  octave_ocl_program (std::string source, std::string build_options = "")
    : octave_base_value (), program (source, build_options) { }

  ~octave_ocl_program (void) { }

  octave_base_value *clone (void) const { return new octave_ocl_program (*this); }
  octave_base_value *empty_clone (void) const { return new octave_ocl_program (); }

  // use do_multi_index_op for calling kernel

  octave_value_list do_multi_index_op (int nargout, const octave_value_list& idx)
  {
    program.rep->assure_valid ();

    int kernel_index = -1;
    Matrix work_size;
    Cell out_descr;

    int nargin = idx.length ();
    if (nargin < 1)
      ocl_error ("ocl program: no kernel specified");

    octave_value kernel_ov = idx (0);
    if (kernel_ov.is_real_scalar ())
      kernel_index = kernel_ov.int_value ();
    else if (kernel_ov.is_string ()) {
      kernel_index = program.get_kernel_index (kernel_ov.string_value ());
    }
    if ((kernel_index < 0) || (kernel_index >= (int) program.num_kernels ()))
      ocl_error ("ocl program: invalid kernel specifier");

    if (nargin < 2)
      return octave_value (kernel_index);

    work_size = idx (1).matrix_value ();
    if (work_size.numel () == 0)
      ocl_error ("ocl program: invalid work size specified");

    int idx_arg = 0;
    octave_value_list out_args;
    octave_value_list tmp_args;
    int num_tmp_args = 0;

    if (nargin < 3)
      ocl_error ("ocl program: no output argument descriptor specified");
    out_descr = idx (2).cell_value ();
    if ((out_descr.ndims () > 2) ||
        (out_descr.numel () == 0) || ((out_descr.rows () > 2) && (out_descr.columns () > 2)))
      ocl_error ("ocl program: invalid output argument descriptor");
    if (((out_descr.rows () >= 2) && out_descr(1,0).is_string ()) || (out_descr.rows () == 1))
      out_descr = out_descr.transpose ();
    if (out_descr.columns () > 2)
      ocl_error ("ocl program: invalid output argument descriptor");
    if (nargout > out_descr.rows ())
      ocl_error ("ocl program: more output arguments than specified in descriptor");

    bool out_def_type = (out_descr.columns () == 1);
    std::string out_type_str ("double");
    dim_vector dv (1,1);

    nargout = out_descr.rows ();

    nargin -= 3;

    InArgOpt in_arg_opt = OptMakeUnique;

    while (nargin > 0) {
      octave_value arg = idx (3+nargin-1);
      if (!arg.is_string ())
        break;
      std::string optstr = arg.string_value ();
      nargin--;

      if (optstr == "make_unique")
        in_arg_opt = OptMakeUnique;
      else if (optstr == "slice_ofs")
        in_arg_opt = OptSliceOfs;
#if 0 // currently inactive because of MAJOR issues with CL_MISALIGNED_SUB_BUFFER_OFFSET
      else if (optstr == "sub_buffer")
        in_arg_opt = OptSubBuffer;
#endif
      else
        ocl_error ("ocl program: invalid option");
    }

    SubBuffer subbuffers[nargin];

    for (int i = 0; i < nargout; i++) {
      Matrix out_size = out_descr(i,0).matrix_value ();
#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
      if (!out_size.dims ().is_vector ())
#else // for octave versions >= 4.4.0
      if (!out_size.dims ().isvector ())
#endif
        ocl_error ("ocl program: invalid output argument descriptor");
      if (!out_def_type) {
        out_type_str = out_descr(i,1).string_value ();
      }

      int ndim = out_size.numel ();
      dv = dv.redim (ndim);
      for (octave_idx_type j = 0; j < ndim; j++)
        dv (j) = out_size (j);

#define SET_KERNEL_OUTARG_OCL_TYPE(C, T) \
      if (out_type_str == #C) { \
        T::array_type array (dv); \
        out_args (i) = octave_value (new T (array)); \
        program.set_kernel_arg (kernel_index, idx_arg++, array); \
      } else

      SET_KERNEL_OUTARG_OCL_TYPE( double, octave_ocl_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( single, octave_ocl_float_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( double_complex, octave_ocl_complex_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( single_complex, octave_ocl_float_complex_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( int8,   octave_ocl_int8_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( int16,  octave_ocl_int16_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( int32,  octave_ocl_int32_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( int64,  octave_ocl_int64_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( uint8,  octave_ocl_uint8_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( uint16, octave_ocl_uint16_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( uint32, octave_ocl_uint32_matrix )
      SET_KERNEL_OUTARG_OCL_TYPE( uint64, octave_ocl_uint64_matrix )
        ocl_error ("ocl program: invalid output argument descriptor data type"); // default case after last "else"

#undef SET_KERNEL_OUTARG_OCL_TYPE

    }

    for (int i = 0; i < nargin; i++) {
      octave_value arg = idx (3+i);
      int type_id = arg.type_id ();

#define SET_KERNEL_ARG_OCL_TYPE(T) \
      if (type_id == T::static_type_id ()) { \
        T *mat = dynamic_cast<T *> (arg.internal_rep ()); \
        if (mat) { \
          T::array_type array (mat->ocl_array_value ()); \
          array.rep->assure_valid (); \
          switch (in_arg_opt) { \
            case OptMakeUnique: \
              array.make_unique (); \
              tmp_args (num_tmp_args++) = octave_value (new T (array)); \
              program.set_kernel_arg (kernel_index, idx_arg++, array); \
              break; \
            case OptSliceOfs: \
              program.set_kernel_arg (kernel_index, idx_arg++, array); \
              program.set_kernel_arg (kernel_index, idx_arg++, octave_uint64 (array.slice_ofs)); \
              break; \
            case OptSubBuffer: \
              subbuffers[i] = SubBuffer (array); \
              program.rep->set_kernel_arg (kernel_index, idx_arg++, \
                                           &(subbuffers[i].subbuf), sizeof (cl_mem)); \
              break; \
          } \
        } else \
          ocl_error ("ocl program: invalid argument"); \
      } else

#define SET_KERNEL_ARG_OCTAVE_TYPE(QUERY, T, EXTRACTOR_FCN) \
      if (arg.QUERY()) { \
        T values = arg.EXTRACTOR_FCN(); \
        program.set_kernel_arg (kernel_index, idx_arg++, values); \
      } else

      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_float_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_complex_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_float_complex_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_int8_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_int16_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_int32_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_int64_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_uint8_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_uint16_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_uint32_matrix )
      SET_KERNEL_ARG_OCL_TYPE( octave_ocl_uint64_matrix )

      SET_KERNEL_ARG_OCTAVE_TYPE(is_double_type () &&   arg.ISREAL, NDArray      , array_value       )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_single_type () &&   arg.ISREAL, FloatNDArray , float_array_value )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_double_type () && ! arg.ISREAL, ComplexNDArray      , complex_array_value       )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_single_type () && ! arg.ISREAL, FloatComplexNDArray , float_complex_array_value )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_int8_type  , int8NDArray  , int8_array_value  )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_int16_type , int16NDArray , int16_array_value )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_int32_type , int32NDArray , int32_array_value )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_int64_type , int64NDArray , int64_array_value )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_uint8_type , uint8NDArray , uint8_array_value )
      SET_KERNEL_ARG_OCTAVE_TYPE(is_uint16_type, uint16NDArray, uint16_array_value)
      SET_KERNEL_ARG_OCTAVE_TYPE(is_uint32_type, uint32NDArray, uint32_array_value)
      SET_KERNEL_ARG_OCTAVE_TYPE(is_uint64_type, uint64NDArray, uint64_array_value)

        ocl_error ("ocl program: invalid argument type"); // default case after last "else"

#undef SET_KERNEL_ARG_OCL_TYPE

#undef SET_KERNEL_ARG_OCTAVE_TYPE
    }

    program.enqueue_kernel (kernel_index, work_size);

    return out_args;
  }

  octave_value subsref (const std::string& type,
                        const std::list<octave_value_list>& idx)
  {
    octave_value_list retvals = subsref (type, idx, 1);

    if (retvals.length () >= 1)
      return retvals(0);
    else
      return octave_value ();
  }

  // use subsref to obtain information on program object

  octave_value_list subsref (const std::string& type,
                             const std::list<octave_value_list>& idx, int nargout)
  {
    octave_value_list retvals;
    std::string indstr;

    switch (type[0]) {
      case '(':
        retvals = do_multi_index_op (nargout, idx.front ());
        break;
      case '.':
        indstr = idx.front () (0).string_value ();
        if (indstr == "valid")
          retvals(0) = octave_value (program.is_valid ());
        else if (indstr == "num_kernels")
          retvals(0) = octave_value (program.num_kernels ());
        else if (indstr == "kernel_names")
        {
          std::vector<std::string> kernel_names = program.get_kernel_names ();
          unsigned int num_kernels = kernel_names.size ();
          Cell c (num_kernels, 1);
          for (unsigned int i = 0; i < num_kernels; i++)
            c (i) = octave_value (kernel_names [i]);
          retvals(0) = octave_value (c);
        }
        else if (indstr == "clEnqueueBarrier")
          program.clEnqueueBarrier ();
        else if (indstr == "clFlush")
          program.clFlush ();
        else if (indstr == "clFinish")
          program.clFinish ();
        else {
          octave_stdout << "ocl program: unknown index '" << indstr.c_str () << "'\n";
          ocl_error ("ocl program: indexing error");
        }
        break;
      default:
        octave_stdout << type_name ().c_str () << " cannot be indexed with " << type[0] << "\n";
        ocl_error ("ocl program: indexing error");
    }

    if (retvals.length () == 1)
      return retvals(0).next_subsref (type, idx);
    else
      return retvals;
  }

  octave_idx_type numel (void) const { return program.num_kernels (); }

  bool is_defined (void) const { return true; }

  bool is_constant (void) const { return true; }

  void print (std::ostream& os, bool pr_as_read_syntax = false)
  { print_raw (os, pr_as_read_syntax); }

  void print_raw (std::ostream& os, bool pr_as_read_syntax = false) const
  { os << program; }

// saving or loading context-dependent ocl programs does not really make sense;
// see also the comments in ocl_ov_matrix.cc
// the stub functions only exist to avoid errors when also saving/loading other variables

  bool save_ascii (std::ostream& os)
  { return true; }

  bool load_ascii (std::istream& is)
  { return true; }

#if ! defined (OCL_OCTAVE_VERSION_6_1_0_AND_HIGHER) // for octave versions < 6.1.0
  bool save_binary (std::ostream& os, bool& save_as_floats)
#else // for octave versions >= 6.1.0
  bool save_binary (std::ostream& os, bool save_as_floats)
#endif
  { return true; }

  bool load_binary (std::istream& is, bool swap, octave::mach_info::float_format fmt)
  { return true; }

  bool save_hdf5 (octave_hdf5_id loc_id, const char *name, bool save_as_floats)
  { return true; }

  bool load_hdf5 (octave_hdf5_id loc_id, const char *name)
  { return true; }

protected:

  OclProgram program;

private:

  octave_ocl_program& operator = (const octave_ocl_program&); // No assignment.

private:
#ifdef DECLARE_OCTAVE_ALLOCATOR
  DECLARE_OCTAVE_ALLOCATOR
#endif

  DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};


#ifdef DEFINE_OCTAVE_ALLOCATOR
DEFINE_OCTAVE_ALLOCATOR (octave_ocl_program);
#endif
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_ocl_program, "ocl program", "ocl program");


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_program", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_program", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_program, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {@var{ocl_prog} =} ocl_program (@var{src_str}) \n\
@deftypefnx {Loadable Function} {@var{ocl_prog} =} ocl_program (@var{src_str}, @var{build_opts_str}) \n\
\n\
Construct and compile an OCL program from an OpenCL C source code string.  \n\
\n\
@code{ocl_program} ingests an OpenCL C source code string @var{src_str} and \n\
proceeds to compile this code using the OpenCL online compiler.  \n\
If given, the build options specified in the string @var{build_opts_str} are \n\
applied during compilation.  If a compilation error occurs, the function \n\
prints the compiler build log with its error messages and aborts.  Otherwise, an \n\
OCL program @var{ocl_prog} is returned.  \n\
For the OpenCL C language, consult the OpenCL specification.  We recommend to \n\
use the language in Version 1.1.  \n\
\n\
@code{ocl_program} prepends one line to the provided source code, possibly enabling \n\
64-bit floating point (double precision), depending on the ability of the current \n\
OpenCL context; the provided source code must allow addition of this line.  \n\
\n\
An OCL program can contain multiple sub-programs, so-called kernels, \n\
which are referenced either by their names (taken from the source code) \n\
or by their indices in a list of all kernels.  \n\
\n\
Access to the OCL program is provided by ways of indexing.  \n\
Information on the OCL program can be read from the following fields:  \n\
\n\
@table @asis \n\
@item @code{.valid} \n\
An integer value, with non-zero meaning that the OCL program is valid \n\
(compiled successfully and the corresponding OpenCL context is still active).  \n\
\n\
@item @code{.num_kernels} \n\
The number of kernels (sub-programs) in the program.  \n\
\n\
@item @code{.kernel_names} \n\
A cell array of strings holding the names of all kernels.  \n\
@end table \n\
\n\
@noindent \n\
Furthermore, the user is able to enqueue specific OpenCL commands controlling \n\
the command queue workflow by issuing statements with the following fields \n\
(see the OpenCL specification for details):  \n\
\n\
@table @asis \n\
@item @code{.clEnqueueBarrier} \n\
\n\
@item @code{.clFlush} \n\
\n\
@item @code{.clFinish} \n\
@end table \n\
\n\
Executing a kernel is performed in OpenCL by setting the kernel's arguments and \n\
enqueueing the kernel into the (asynchronous) command queue.  \n\
Using an OCL program in octave, both steps are performed using a single \n\
indexing statement with parentheses:  \n\
\n\
@example \n\
@group \n\
[argout1, argout2, ...] = ocl_prog (kernel_index, work_size, cellout, argin1, argin2, ..., opt) \n\
@end group \n\
@end example \n\
\n\
@noindent \n\
The parameters have the following meaning:  \n\
\n\
@table @asis \n\
@item @var{kernel_index} \n\
Either the kernel index (0 <= kernel_index < num_kernels), \n\
or a kernel name string (which is slightly slower).  \n\
\n\
@item @var{work_size} \n\
Either a single positive integer specifying the total number of work-items \n\
for parallel execution (SIMD principle, i.e., Single Instruction Multiple Data), \n\
or a matrix with at most three rows.  \n\
The number of columns of the matrix is the number of dimensions for specifying \n\
work-items.  \n\
The first row of the matrix specifies the number of work-items per dimension; \n\
their overall product corresponds to the single integer mentioned earlier.  \n\
The second row of the matrix, if given, specifies an offest, per dimension, \n\
for work-item indices.  \n\
The third row of the matrix, if given, specifies the number of work-items, \n\
per dimension, that make up a work-group.  \n\
For details, consult the OpenCL specification.  \n\
\n\
@item @var{cellout} \n\
A cell array describing the output arguments.  Output arguments are OCL matrices \n\
of which the number, sizes (and types) must be pre-specified in order to be allocated \n\
automatically before the actual kernel call.  To specify N output arguments, \n\
the size of the cell array must be either 1xN, Nx1, 2xN, or Nx2.  The cell \n\
array must contain either only the matrices' sizes (each as an octave row vector), \n\
in which case the default type 'double' is assumed, or contain in a second row / \n\
column also the matrices' data types (e.g., 'single') as strings.  For complex-valued \n\
output arguments, the type must indicate this explicitly (e.g. 'double_complex').  \n\
In the kernel's \n\
OpenCL C declaration, these output arguments must be the first arguments, \n\
preceeding the input parameters.  \n\
Complex-valued (output and input) arguments to OpenCL C kernels must be declared \n\
as global pointers to 'double2' or 'float2' (e.g., @code{__global float2 *arg}).  \n\
\n\
@item @var{argin1, argin2, ...} \n\
A list of input arguments to the kernel. These can be: an OCL matrix, or a single \n\
octave scalar, or a (small) octave matrix.  Note that in the first case, \n\
no type checking is possible, so it is the user's responsibility to match \n\
the matrix data types in octave and in the kernel code.  Note also that in the \n\
later cases, type matching is also essential; often, one will want to convert \n\
parameters explicitly before using as an argument (e.g., @code{uint64(n)} to \n\
convert an octave double scalar to a kernel source argument of type @code{ulong}).  \n\
Note finally that passing an octave matrix has tight data size limitations, \n\
whereas passing an OCL matrix has not.  \n\
\n\
@item @var{opt} \n\
(Optional) An option string specifying input OCL matrix handling.  \n\
\"make_unique\" (the default) is the safest and easiest, but may, in some cases, \n\
involve deep data copying before the kernel call.  It is recommended for kernel \n\
prototyping and simple calls (e.g., with OCL matrices created just before the call).  \n\
\"slice_ofs\" is the elaborate and efficient alternative, which needs small \n\
modifications to the kernel declaration and code (for an example, see ocl_tests.m).  \n\
This option is recommended for any new function accepting OCL matrices \n\
to be passed to kernels (e.g. library functions working on OCL data).  \n\
@end table \n\
\n\
For convenience, a call with only the kernel name string specified does not \n\
execute a kernel but returns its kernel index (which might be stored in a \n\
persistent variable for all future kernel calls): \n\
\n\
@example \n\
@group \n\
@var{kernel_index} = ocl_prog (@var{kernel_name}) \n\
@end group \n\
@end example \n\
\n\
@code{ocl_program} automatically assures that the OpenCL library is \n\
loaded (see @code{ocl_lib}) and that an OpenCL context is created with an \n\
OpenCL device (see @code{ocl_context}).  \n\
\n\
Be aware that running your own OpenCL C code comes with a certain risk.  If your code \n\
contains an infinite loop, there is no way of stopping the code; similarly, \n\
in case of a memory access bug, the octave interpreter may crash or stall, \n\
needing to be stopped by means of the operating system, losing all data \n\
that was unique in octave's workspace.  \n\
\n\
@seealso{oclArray, ocl_tests, ocl_context, ocl_lib, \
ocl_double, ocl_single, \
ocl_int8, ocl_int16, ocl_int32, ocl_int64, \
ocl_uint8, ocl_uint16, ocl_uint32, ocl_uint64} \n\
@end deftypefn")
{
  octave_value_list retval;
  int nargin = args.length ();
  std::string source, build_options;

  if ((nargin > 2) ||
      ((nargin > 0) && (!args (0).is_string ())) ||
      ((nargin > 1) && (!args (1).is_string ()))) {
    print_usage ();
    return retval;
  }

  assure_installed_ocl_types ();

  if (nargin > 0)
    source = args (0).string_value ();

  if (nargin > 1)
    build_options = args (1).string_value ();

  if (source != "")
    retval = octave_value (new octave_ocl_program (source, build_options));
  else
    retval = octave_value (new octave_ocl_program ());

  return retval;
}


// ---------- public functions


void
install_ocl_program_type (void)
{
  octave_ocl_program::register_type ();
}
