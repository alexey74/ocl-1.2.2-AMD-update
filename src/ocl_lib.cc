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
#include <string>

#define OCL_LIB_CPP
#include "ocl_lib.h"

#include "ocl_octave_versions.h"


/*

Notes on GPL3 licensing and linking to OpenCL:

The Mesa package (see https://mesa3d.org/) comprises an OpenCL 1.1 driver and
ICD loader since years.  Mesa is a free, open-source, available implementation
of the OpenCL Standard Interface.  Mesa and, as a consequence, any other OpenCL
library thus belong to the class of "System Libraries" as defined in GPL3.
These libraries thus may be linked to by works released under GPL3.  Moreover,
an OpenCL driver and ICD loader are not part of the "Corresponding Source" of
a work linking to them.
The OCL source code can thus be distributed under the GPL3, but may not contain
any OpenCL driver or library.


Technical motivation for dynamic linking to the driver or loader library:

- I require the OCL package to be also largely compatible with Octave under
  Windows (at least with the official MXE builds with MinGW, at least for
  octave versions >= 4.2.0).  This requires linking to an OpenCL library of
  which no static version or information is available, but only the
  (vendor-given) dll-file for dynamic linking.
- Dynamic linking also enables the interactive choosing of the best or desired
  ICD loader (or single driver) file, in the common case that multiple such
  files are present in the operating system.

*/


// ---------- operating system dependent shared library names


#if defined (_WIN32) // WINDOWS
  #define lib_path_default ""
  #define lib_name_default "OpenCL.dll"
#elif defined (__APPLE__) // macOS
  #define lib_path_default ""
  #define lib_name_default "libOpenCL.so"
#else // GNU/Linux and BSD
  #define lib_path_default ""
  #define lib_name_default "libOpenCL.so.1"
  // This should work for OpenCL 1.0 up to at least 2.2; the OpenCL
  // specifications state deprecations, but no removals, so the
  // binary interface remains compatible.
  // Using libOpenCL.so.1 (the SONAME) suits all OpenCL package
  // installations (while libOpenCL.so is the development library
  // name used for building, needing an additional symlink).
#endif


// ---------- shared library load/unload functions


static bool cl_lib_loaded = false;

static octave::dynamic_library opencl_lib;

static std::string lib_path (lib_path_default);
static std::string lib_name (lib_name_default);


//#define UNLOAD_ADDR ( sym ) sym = NULL;
#define LOAD_ADDR( sym ) \
  *(void **) (&sym) = opencl_lib.search (#sym); \
  if (sym == NULL) { \
    opencl_lib.close (); \
    cl_lib_loaded = false; \
    octave_stdout << "error loading OpenCL library symbol: " << #sym << "\n"; \
    ocl_error ("error loading OpenCL library symbol.\n"); \
  }


static
void
load_opencl_library_symbols (void)
{
  /* Platform API */
  LOAD_ADDR ( clGetPlatformIDs )
  LOAD_ADDR ( clGetPlatformInfo )

  /* Device APIs */
  LOAD_ADDR ( clGetDeviceIDs )
  LOAD_ADDR ( clGetDeviceInfo )

  /* Context APIs  */
  LOAD_ADDR ( clCreateContext )
  LOAD_ADDR ( clCreateContextFromType )
  LOAD_ADDR ( clRetainContext )
  LOAD_ADDR ( clReleaseContext )
  LOAD_ADDR ( clGetContextInfo )

  /* Command Queue APIs */
  LOAD_ADDR ( clCreateCommandQueue )
  LOAD_ADDR ( clRetainCommandQueue )
  LOAD_ADDR ( clReleaseCommandQueue )
  LOAD_ADDR ( clGetCommandQueueInfo )

  /* Memory Object APIs */
  LOAD_ADDR ( clCreateBuffer )
  LOAD_ADDR ( clCreateSubBuffer )
  LOAD_ADDR ( clRetainMemObject )
  LOAD_ADDR ( clReleaseMemObject )
  LOAD_ADDR ( clGetMemObjectInfo )
  LOAD_ADDR ( clSetMemObjectDestructorCallback )
#ifdef OCL_LIB_HAVE_IMAGES
  LOAD_ADDR ( clCreateImage2D )
  LOAD_ADDR ( clCreateImage3D )
  LOAD_ADDR ( clGetSupportedImageFormats )
  LOAD_ADDR ( clGetImageInfo )
#endif

  /* Sampler APIs  */
#ifdef OCL_LIB_HAVE_SAMPLERS
  LOAD_ADDR ( clCreateSampler )
  LOAD_ADDR ( clRetainSampler )
  LOAD_ADDR ( clReleaseSampler )
  LOAD_ADDR ( clGetSamplerInfo )
#endif

  /* Program Object APIs  */
  LOAD_ADDR ( clCreateProgramWithSource )
  LOAD_ADDR ( clCreateProgramWithBinary )
  LOAD_ADDR ( clRetainProgram )
  LOAD_ADDR ( clReleaseProgram )
  LOAD_ADDR ( clBuildProgram )
  LOAD_ADDR ( clUnloadCompiler )
  LOAD_ADDR ( clGetProgramInfo )
  LOAD_ADDR ( clGetProgramBuildInfo )

  /* Kernel Object APIs */
  LOAD_ADDR ( clCreateKernel )
  LOAD_ADDR ( clCreateKernelsInProgram )
  LOAD_ADDR ( clRetainKernel )
  LOAD_ADDR ( clReleaseKernel )
  LOAD_ADDR ( clSetKernelArg )
  LOAD_ADDR ( clGetKernelInfo )
  LOAD_ADDR ( clGetKernelWorkGroupInfo )

  /* Event Object APIs  */
  LOAD_ADDR ( clWaitForEvents )
  LOAD_ADDR ( clGetEventInfo )
  LOAD_ADDR ( clCreateUserEvent )
  LOAD_ADDR ( clRetainEvent )
  LOAD_ADDR ( clReleaseEvent )
  LOAD_ADDR ( clSetUserEventStatus )
  LOAD_ADDR ( clSetEventCallback )

  /* Profiling APIs  */
  LOAD_ADDR ( clGetEventProfilingInfo )

  /* Flush and Finish APIs */
  LOAD_ADDR ( clFlush )
  LOAD_ADDR ( clFinish )

  /* Enqueued Commands APIs */
  LOAD_ADDR ( clEnqueueReadBuffer )
  LOAD_ADDR ( clEnqueueWriteBuffer )
  LOAD_ADDR ( clEnqueueCopyBuffer )
#ifdef OCL_LIB_HAVE_BUFFER_RECT
  LOAD_ADDR ( clEnqueueReadBufferRect )
  LOAD_ADDR ( clEnqueueWriteBufferRect )
  LOAD_ADDR ( clEnqueueCopyBufferRect )
#endif
  LOAD_ADDR ( clEnqueueMapBuffer )
#ifdef OCL_LIB_HAVE_IMAGES
  LOAD_ADDR ( clEnqueueReadImage )
  LOAD_ADDR ( clEnqueueWriteImage )
  LOAD_ADDR ( clEnqueueCopyImage )
  LOAD_ADDR ( clEnqueueCopyImageToBuffer )
  LOAD_ADDR ( clEnqueueCopyBufferToImage )
  LOAD_ADDR ( clEnqueueMapImage )
#endif
  LOAD_ADDR ( clEnqueueUnmapMemObject )
  LOAD_ADDR ( clEnqueueNDRangeKernel )
  LOAD_ADDR ( clEnqueueTask )
  LOAD_ADDR ( clEnqueueNativeKernel )
  LOAD_ADDR ( clEnqueueMarker )
  LOAD_ADDR ( clEnqueueWaitForEvents )
  LOAD_ADDR ( clEnqueueBarrier )

  /* Extension function access */
  LOAD_ADDR ( clGetExtensionFunctionAddress )
}


void
assure_opencl_library (void)
{
  if (cl_lib_loaded)
    return;

  clear_resources ();

  std::string fullname (lib_path+lib_name);

  try {
    opencl_lib.open (fullname);
  }
  catch (...) {
  }
  if (! opencl_lib)
    ocl_error ("octave's dynamic library loader reported an error while dynamically loading the OpenCL library\n  consider manual inspection with 'ocl_lib' function.");

  load_opencl_library_symbols ();

  cl_lib_loaded = true;
}


void
unload_opencl_library (void)
{
  destroy_opencl_context ();
  clear_resources ();

  if (cl_lib_loaded)
    opencl_lib.close ();

  cl_lib_loaded = false;
}


bool
opencl_library_loaded (void)
{
  return cl_lib_loaded;
}


// ---------- the octave entry point to the 'ocl_lib' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_lib", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_lib", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_lib, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} ocl_lib (@qcode{\"assure\"}) \n\
@deftypefnx {Loadable Function} ocl_lib (@qcode{\"unload\"}) \n\
@deftypefnx {Loadable Function} {@var{loaded} =} ocl_lib (@qcode{\"loaded\"}) \n\
@deftypefnx {Loadable Function} {[[@var{oldpath}], [@var{oldfname}]] =} \
ocl_lib (@qcode{\"lib_path_filename\"}, [@var{newpath}], [@var{newfname}]) \n\
\n\
Manage dynamic loading/unloading of OpenCL Library. \n\
\n\
@code{ocl_lib (\"assure\")} loads the OpenCL library and \n\
dynamically links to it.  If any step is unsuccessful, @code{ocl_lib} aborts with an error.  \n\
If the OpenCL library was already loaded, @code{ocl_lib} has no effect.  The currently set \n\
path and filename (see below) are used for loading the library. \n\
\n\
@code{ocl_lib (\"unload\")} unloads the OpenCL library.  \n\
Further (internal or explicit) calls to the library are no longer possible.  \n\
If the OpenCL library was not loaded, @code{ocl_lib} has no effect.  \n\
The subfunction also destroys the OpenCL context.  \n\
\n\
@code{ocl_lib (\"loaded\")} simply returns whether the library is currently loaded.  \n\
A zero result means the library is currently not loaded.  \n\
\n\
Called with the @qcode{\"lib_path_filename\"} parameter, @code{ocl_lib} can be used \n\
to query, set, or reset the path and filename pointing to the OpenCL library.  \n\
System-dependent default settings for both are set when loading the OCL package.  \n\
The optional one or two output parameters return the current settings for the \n\
path @var{oldpath} and filename @var{oldfname}, respectively.  \n\
The optional one or two additional input parameters @var{newpath} and @var{newfname} \n\
overwrite the current settings for the path and filename, respectively.  \n\
If @var{newpath} is not an empty string, the concatenation of @var{newpath} and \n\
@var{newfname} must result in a correct full path to the file (i.e., @var{newpath} \n\
must then end with the system-dependent path separator, e.g., a slash or backslash).  \n\
If any of @var{newpath} or @var{newfname} is equal to @qcode{\"default\"}, then \n\
the corresponding setting is reset to the system-dependent default value instead.  \n\
\n\
The function @code{ocl_lib} only needs to be called explicitly in rare situations, \n\
since many other (\"higher\") OCL functions call it internally.  \n\
The function is provided mainly for testing and for troubleshooting regarding \n\
an OpenCL installation.  \n\
\n\
@seealso{oclArray} \n\
@end deftypefn")
{
  octave_value_list retval;
  int nargin = args.length ();

  std::string fcn;
  if ((nargin > 0) && (args (0).is_string ()))
    fcn = args (0).char_matrix_value ().row_as_string (0);

  if ((nargin == 0) || (!args (0).is_string ())) {

    ocl_error ("first argument must be a string");

  } else if (fcn == "assure") {

    if (nargin > 1)
      ocl_error ("assure: too many arguments");

    assure_opencl_library ();

  } else if (fcn == "unload") {

    if (nargin > 1)
      ocl_error ("unload: too many arguments");

    unload_opencl_library ();

  } else if (fcn == "loaded") {

    if (nargin > 1)
      ocl_error ("loaded: too many arguments");

    retval (0) = octave_value (double (cl_lib_loaded));

  } else if (fcn == "lib_path_filename") {

    if (nargin > 3)
      ocl_error ("lib_path_filename: too many arguments");
    if ((nargin > 1) && (!args (1).is_string ()))
      ocl_error ("lib_path_filename: second argument must be a string");
    if ((nargin > 2) && (!args (2).is_string ()))
      ocl_error ("lib_path_filename: third argument must be a string");

    if ((nargout > 0) || (nargin == 1))
      retval (0) = lib_path;
    if ((nargout > 1) || (nargin == 1))
      retval (1) = lib_name;

    if ((nargin > 1) && (cl_lib_loaded))
      ocl_error ("lib_path_filename: changing the library path or name is not permitted while the library is loaded");

    if (nargin > 1)
      lib_path = args (1).char_matrix_value ().row_as_string (0);
    if (nargin > 2)
      lib_name = args (2).char_matrix_value ().row_as_string (0);

    if (lib_path == "default")
      lib_path = lib_path_default;
    if (lib_name == "default")
      lib_name = lib_name_default;

  } else {

    ocl_error ("subfunction not recognized");

  }

  return retval;
}
