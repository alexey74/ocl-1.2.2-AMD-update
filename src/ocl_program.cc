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

#include "ocl_program.h"
#include "ocl_array.h"
#include "ocl_lib.h"
#include <octave/oct.h>


// ---------- static helper functions


static
void
ocl_program_inop_error (void)
{
  ocl_error ("OclProgram: operating on an inoperable program object (e.g., context destroyed or empty object)");
}


// ---------- OclProgram::OclProgramRep members


void
OclProgram::OclProgramRep::assure_valid (void) const
{
  if (! is_valid ())
    ocl_program_inop_error ();
}


void
OclProgram::OclProgramRep::compile (const std::string& source, const std::string& build_options)
{
  std::string source_ext (source);
  if (opencl_context_is_fp64 ()) // always prepend one line, possibly enabling double support
    source_ext = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" + source_ext;
  else
    source_ext = "\n" + source_ext;

  const char *source_ptr = source_ext.c_str ();
  cl_program program = clCreateProgramWithSource (get_context (), 1, & source_ptr, 0, & last_error);
  ocl_check_error ("clCreateProgramWithSource");

  cl_device_id device_id = get_device_id ();
  last_error = clBuildProgram (program, 1, & device_id, build_options.c_str (), 0, 0);

  if (last_error == CL_BUILD_PROGRAM_FAILURE) { // build error from clBuildProgram
    cl_int err = CL_SUCCESS;
    size_t len = 0;
    err = clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, 0, 0, & len);
    if ((err == CL_SUCCESS) && (len > 0)) {
      char build_log_ca [len];
      err = clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, len, build_log_ca, 0);
      if (err == CL_SUCCESS) {
        build_log = build_log_ca;
        octave_stdout << "OclProgram: building OpenCL program returned with error. Build log:\n\n";
        octave_stdout << build_log << "\n\n";
      }
    }
    clReleaseProgram (program);
    ocl_check_error ("clBuildProgram"); // or, possibly, return
  }

  if (last_error != CL_SUCCESS) { // other error from clBuildProgram
    clReleaseProgram (program);
    ocl_check_error ("clBuildProgram"); // or, possibly, return
  }

  // successfully built the program
  ocl_program = (void *) program; // indicator for valid program

  unsigned int num_kernels = 0;
  last_error = clCreateKernelsInProgram (program, 0, 0, & num_kernels);
  if ((last_error != CL_SUCCESS) || (num_kernels == 0))
    return;

  cl_kernel kernel_objs [num_kernels];
  last_error = clCreateKernelsInProgram (program, num_kernels, kernel_objs, 0);
  if (last_error != CL_SUCCESS)
    return;

  ocl_kernels.resize (num_kernels);
  kernel_names.resize (num_kernels);

  for (unsigned int i = 0; i < num_kernels; i++) {
    ocl_kernels [i] = (void*) kernel_objs [i];

    size_t name_length = 0;
    last_error = clGetKernelInfo (kernel_objs [i], CL_KERNEL_FUNCTION_NAME, 0, 0, & name_length);
    if ((last_error != CL_SUCCESS) || (name_length == 0))
      continue;

    char kernel_name [name_length];
    last_error = clGetKernelInfo (kernel_objs [i], CL_KERNEL_FUNCTION_NAME, name_length, kernel_name, 0);
    if (last_error != CL_SUCCESS)
      continue;

    kernel_names [i] = kernel_name;
    kernel_dictionary [kernel_name] = i;
  }
}


void
OclProgram::OclProgramRep::destroy (void)
{
  if (object_context_still_valid ()) {
    // never check for errors when deleting objects now
    for (unsigned int i = 0; i < ocl_kernels.size (); i++)
      clReleaseKernel ((cl_kernel) ocl_kernels [i]);
    clReleaseProgram ((cl_program) ocl_program);
  }
  // do not complain about inoperable programs
  ocl_program = (void *) 0; // indicator for invalid program
}


int
OclProgram::OclProgramRep::get_kernel_index (const std::string& str, bool strict) const
{
  std::map<std::string, int>::const_iterator
    it = kernel_dictionary.find(str),
    end = kernel_dictionary.end();

  if (it == end) {
    if (strict)
      ocl_error ("OclProgram::get_kernel_index(): kernel name not found");
    else
      return -1;
  }

  return it->second;
}


void
OclProgram::OclProgramRep::set_kernel_arg
  (int kernel_index,
   unsigned int arg_index,
   const void *arg_ptr,
   size_t byte_size)
{
  assure_valid ();
  if ((kernel_index < 0) || (kernel_index >= (int) num_kernels ()))
    ocl_error ("OclProgram::set_kernel_arg(): kernel index not found");

  last_error = clSetKernelArg ((cl_kernel) ocl_kernels [kernel_index],
                               arg_index,
                               byte_size,
                               arg_ptr);
  ocl_check_error ("clSetKernelArg");
}


void
OclProgram::OclProgramRep::enqueue_kernel
  (int kernel_index,
   const Matrix& work_size)
{
  assure_valid ();
  if ((kernel_index < 0) || (kernel_index >= (int) num_kernels ()))
    ocl_error ("OclProgram::enqueue_kernel(): kernel index not found");

  unsigned int work_dim = work_size.columns ();
  unsigned int rows = work_size.rows ();

  if (work_dim < 1)
    ocl_error ("OclProgram::enqueue_kernel(): work_dim too small");
  if (work_dim > 6)
    ocl_error ("OclProgram::enqueue_kernel(): work_dim too large");
  if (rows > 3)
    ocl_error ("OclProgram::enqueue_kernel(): work_size must have at most 3 rows");

  double d;
  size_t global_work_size [work_dim];
  size_t global_work_offset [work_dim];
  size_t *global_work_offset_pointer = 0;
  size_t local_work_size [work_dim];
  size_t *local_work_size_pointer = 0;

  Matrix global_work_size_d = work_size.row (0);
  for (unsigned int i = 0; i < work_dim; i++) {
    d = global_work_size_d.elem (i);
    if (d < 1.0)
      ocl_error ("OclProgram::enqueue_kernel(): invalid global work size");
    global_work_size [i] = d;
  }

  if (rows >= 2) {
    Matrix global_work_offset_d = work_size.row (1);
    for (unsigned int i = 0; i < work_dim; i++) {
      d = global_work_offset_d.elem (i);
      if (d < 0.0)
        ocl_error ("OclProgram::enqueue_kernel(): invalid global work offset");
      global_work_offset [i] = d;
    }
    global_work_offset_pointer = global_work_offset;
  }

  if (rows >= 3) {
    Matrix local_work_size_d = work_size.row (2);
    for (unsigned int i = 0; i < work_dim; i++) {
      d = local_work_size_d.elem (i);
      if (d < 1.0)
        ocl_error ("OclProgram::enqueue_kernel(): invalid local work size");
      local_work_size [i] = d;
    }
    local_work_size_pointer = local_work_size;
  }

  last_error = clEnqueueNDRangeKernel (get_command_queue (),
                                       (cl_kernel) ocl_kernels [kernel_index],
                                       work_dim,
                                       global_work_offset_pointer,
                                       global_work_size,
                                       local_work_size_pointer,
                                       0, 0, 0);
  ocl_check_error ("clEnqueueNDRangeKernel");
}


// ---------- OclProgram members


template <typename T>
void
OclProgram::set_kernel_arg (int kernel_index,
                            unsigned int arg_index,
                            const OclArray<T>& arg)
{
  if (! (arg.rep->is_valid ()))
    ocl_error ("OclProgram::set_kernel_arg(): invalid / inoperable OclArray as argument");

  void *buffer = arg.rep->get_ocl_buffer ();

  rep->set_kernel_arg (kernel_index,
                       arg_index,
                       &buffer,
                       sizeof (cl_mem));
}
// instantiate right away
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_int8  >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_int16 >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_int32 >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_int64 >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_uint8 >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_uint16>&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_uint32>&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<octave_uint64>&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<float        >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<double       >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<FloatComplex >&);
template void OclProgram::set_kernel_arg (int, unsigned int, const OclArray<Complex      >&);


void
OclProgram::clEnqueueBarrier (void)
{
  if (opencl_context_active ()) {
    last_error = ::clEnqueueBarrier (get_command_queue ());
    ocl_check_error ("clEnqueueBarrier");
  } else
    ocl_error ("OclProgram::clEnqueueBarrier: no valid OpenCL context");
}


void
OclProgram::clFlush (void)
{
  if (opencl_context_active ()) {
    last_error = ::clFlush (get_command_queue ());
    ocl_check_error ("clFlush");
  } else
    ocl_error ("OclProgram::clFlush: no valid OpenCL context");
}


void
OclProgram::clFinish (void)
{
  if (opencl_context_active ()) {
    last_error = ::clFinish (get_command_queue ());
    ocl_check_error ("clFinish");
  } else
    ocl_error ("OclProgram::clFinish: no valid OpenCL context");
}


// ---------- OclProgram non-members


std::ostream&
operator << (std::ostream& os, const OclProgram& p)
{
  os << "  OCL program (with " << p.num_kernels () << " kernels)\n";

  return os;
}
