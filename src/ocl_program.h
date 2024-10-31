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

#ifndef __OCL_PROGRAM_H
#define __OCL_PROGRAM_H

#include "ocl_context_obj.h"
#include <vector>
#include <string>
#include <map>

#include <octave/oct.h>

template <typename T> class OclArray;


// OCL program class allowing shallow copies
class
OclProgram
{
protected:

  // class holding the OpenCL program object and kernel objects
  class
  OclProgramRep : public OclContextObject
  {
  public:

    // empty, inoperable program
    OclProgramRep ()
      : OclContextObject (false), ocl_program (0), count (1) {}

    // program with source code, for immediate compilation; needs/activates an OpenCL context
    OclProgramRep (const std::string& source, const std::string& build_options = "")
      : OclContextObject (true), ocl_program (0), count (1)
    {
      compile (source, build_options);
    }

    ~OclProgramRep () { destroy (); }

    bool
    is_valid (void) const
    {
      return (ocl_program != 0) && (object_context_still_valid ());
    }

    unsigned int
    num_kernels (void) const { return ocl_kernels.size (); }

    std::string
    get_kernel_name (int index) const
    {
      return ((index >= 0) && (index < (int) num_kernels ())) ? kernel_names [index] : std::string ("");
    }

    int
    get_kernel_index (const std::string& str, bool strict = true) const;

    std::vector<std::string>
    get_kernel_names (void) const { return kernel_names; }

    void assure_valid (void) const;

    void set_kernel_arg (int kernel_index,
                         unsigned int arg_index,
                         const void *arg_ptr,
                         size_t byte_size);

    void enqueue_kernel (int kernel_index, const Matrix& work_size);

    void *ocl_program;
    std::string build_log;
    std::vector<void *> ocl_kernels;
    std::vector<std::string> kernel_names;
    std::map<std::string, int> kernel_dictionary;
    int count;

  private:

    void compile (const std::string& source, const std::string& build_options = "");
    void destroy (void);

    OclProgramRep (const OclProgramRep&); // no copying
    OclProgramRep& operator = (const OclProgramRep&); // no assignment
  };

protected:

  OclProgramRep *rep;

private:

  OclProgramRep *
  nil_rep (void) const
  {
    static OclProgramRep nr;
    return & nr;
  }

public:

  virtual ~OclProgram ()
  {
    if ((--(rep->count)) == 0)
      delete rep;
  }

  // empty constructor
  OclProgram (void) : rep (nil_rep ())
  {
    (rep->count)++;
  }

  // constructor with source code (and build options)
  OclProgram (std::string source, std::string build_options = "")
    : rep (new OclProgramRep (source, build_options))
  { }

  // copy constructor
  OclProgram (const OclProgram& a)
    : rep (a.rep)
  {
    (rep->count)++;
  }

  // assignment
  OclProgram& operator = (const OclProgram& a)
  {
    if (this != & a) {
      if ((--(rep->count)) == 0)
        delete rep;
      rep = a.rep;
      (rep->count)++;
    }
    return *this;
  }

  bool is_valid (void) const { return rep->is_valid (); }

  unsigned int num_kernels (void) const { return rep->num_kernels (); }

  std::string
  get_kernel_name (int i) const
  {
    return rep->get_kernel_name (i);
  }

  int
  get_kernel_index (const std::string& str, bool strict = true) const
  {
    return rep->get_kernel_index (str, strict);
  }

  std::vector<std::string>
  get_kernel_names (void) const
  {
    return rep->get_kernel_names ();
  }

  void
  clear (void)
  {
    if (rep != nil_rep ()) {
      if ((--(rep->count)) == 0)
        delete rep;
      rep = nil_rep ();
      (rep->count)++;
    }
  }

  // for OclArray type kernel arguments
  template <typename T>
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const OclArray<T>& arg);

  // for void* kernel arguments (setting an OpenCL buffer object pointer to zero)
  // CAUTION: on some older OpenCL drivers, this may not work and lead to crashes
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const void *arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }

  // for all scalar type kernel arguments
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_int8& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_int16& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_int32& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_int64& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_uint8& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_uint16& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_uint32& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const octave_uint64& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const char& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const short& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const int& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const long& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const unsigned char& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const unsigned short& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const unsigned int& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const unsigned long& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const float& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const double& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const FloatComplex& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const Complex& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, & arg, sizeof (arg)); }

  // for all numeric octave array type kernel arguments
  // (only small, fixed-size kernel arguments)
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const int8NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const int16NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const int32NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const int64NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const uint8NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const uint16NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const uint32NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const uint64NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const FloatNDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const NDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const FloatComplexNDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }
  void set_kernel_arg (int kernel_index, unsigned int arg_index, const ComplexNDArray& arg)
    { rep->set_kernel_arg (kernel_index, arg_index, arg.data (), arg.byte_size ()); }

  // enqueue kernel, short
  void enqueue_kernel (int kernel_index, size_t n, size_t ofs = 0)
  {
    Matrix work_size (2,1);
    work_size (0,0) = n;
    work_size (1,0) = ofs;
    rep->enqueue_kernel (kernel_index, work_size);
  }

  // enqueue kernel, full functionality
  void enqueue_kernel (int kernel_index, const Matrix& work_size)
  {
    rep->enqueue_kernel (kernel_index, work_size);
  }

  // for convenience
  static void clEnqueueBarrier (void);

  static void clFlush (void);

  static void clFinish (void);

  friend class octave_ocl_program;
};


std::ostream&
operator << (std::ostream& os, const OclProgram& p);

#endif  /* __OCL_PROGRAM_H */
