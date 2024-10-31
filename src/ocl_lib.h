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

#ifndef __OCL_LIB_H
#define __OCL_LIB_H

#include "cl_1_1_dl.h"
#include <octave/oct.h>

#ifdef __cplusplus
extern "C" {
#endif

#undef OCL_LIB_HAVE_IMAGES
#undef OCL_LIB_HAVE_SAMPLERS
#undef OCL_LIB_HAVE_BUFFER_RECT

#ifdef OCL_LIB_CPP
  #define FCNVAR_PREFIX
  #define FCNVAR_SUFFIX = NULL
#else
  #define FCNVAR_PREFIX extern
  #define FCNVAR_SUFFIX
#endif

#define OCL_REDEFINE( fcn )  FCNVAR_PREFIX  t_ ## fcn  fcn  FCNVAR_SUFFIX;

/*
 * Use function types declared in cl_1_1_dl.h to declare function pointers.
 * Function pointers are set in ocl_lib.cc after loading the OpenCL library.
 */

/* Platform API */
OCL_REDEFINE ( clGetPlatformIDs )
OCL_REDEFINE ( clGetPlatformInfo )

/* Device APIs */
OCL_REDEFINE ( clGetDeviceIDs )
OCL_REDEFINE ( clGetDeviceInfo )

/* Context APIs  */
OCL_REDEFINE ( clCreateContext )
OCL_REDEFINE ( clCreateContextFromType )
OCL_REDEFINE ( clRetainContext )
OCL_REDEFINE ( clReleaseContext )
OCL_REDEFINE ( clGetContextInfo )

/* Command Queue APIs */
OCL_REDEFINE ( clCreateCommandQueue )
OCL_REDEFINE ( clRetainCommandQueue )
OCL_REDEFINE ( clReleaseCommandQueue )
OCL_REDEFINE ( clGetCommandQueueInfo )

/* Memory Object APIs */
OCL_REDEFINE ( clCreateBuffer )
OCL_REDEFINE ( clCreateSubBuffer )
OCL_REDEFINE ( clRetainMemObject )
OCL_REDEFINE ( clReleaseMemObject )
OCL_REDEFINE ( clGetMemObjectInfo )
OCL_REDEFINE ( clSetMemObjectDestructorCallback )
#ifdef OCL_LIB_HAVE_IMAGES
OCL_REDEFINE ( clCreateImage2D )
OCL_REDEFINE ( clCreateImage3D )
OCL_REDEFINE ( clGetSupportedImageFormats )
OCL_REDEFINE ( clGetImageInfo )
#endif

/* Sampler APIs  */
#ifdef OCL_LIB_HAVE_SAMPLERS
OCL_REDEFINE ( clCreateSampler )
OCL_REDEFINE ( clRetainSampler )
OCL_REDEFINE ( clReleaseSampler )
OCL_REDEFINE ( clGetSamplerInfo )
#endif

/* Program Object APIs  */
OCL_REDEFINE ( clCreateProgramWithSource )
OCL_REDEFINE ( clCreateProgramWithBinary )
OCL_REDEFINE ( clRetainProgram )
OCL_REDEFINE ( clReleaseProgram )
OCL_REDEFINE ( clBuildProgram )
OCL_REDEFINE ( clUnloadCompiler )
OCL_REDEFINE ( clGetProgramInfo )
OCL_REDEFINE ( clGetProgramBuildInfo )

/* Kernel Object APIs */
OCL_REDEFINE ( clCreateKernel )
OCL_REDEFINE ( clCreateKernelsInProgram )
OCL_REDEFINE ( clRetainKernel )
OCL_REDEFINE ( clReleaseKernel )
OCL_REDEFINE ( clSetKernelArg )
OCL_REDEFINE ( clGetKernelInfo )
OCL_REDEFINE ( clGetKernelWorkGroupInfo )

/* Event Object APIs  */
OCL_REDEFINE ( clWaitForEvents )
OCL_REDEFINE ( clGetEventInfo )
OCL_REDEFINE ( clCreateUserEvent )
OCL_REDEFINE ( clRetainEvent )
OCL_REDEFINE ( clReleaseEvent )
OCL_REDEFINE ( clSetUserEventStatus )
OCL_REDEFINE ( clSetEventCallback )

/* Profiling APIs  */
OCL_REDEFINE ( clGetEventProfilingInfo )

/* Flush and Finish APIs */
OCL_REDEFINE ( clFlush )
OCL_REDEFINE ( clFinish )

/* Enqueued Commands APIs */
OCL_REDEFINE ( clEnqueueReadBuffer )
OCL_REDEFINE ( clEnqueueWriteBuffer )
OCL_REDEFINE ( clEnqueueCopyBuffer )
#ifdef OCL_LIB_HAVE_BUFFER_RECT
OCL_REDEFINE ( clEnqueueReadBufferRect )
OCL_REDEFINE ( clEnqueueWriteBufferRect )
OCL_REDEFINE ( clEnqueueCopyBufferRect )
#endif
OCL_REDEFINE ( clEnqueueMapBuffer )
#ifdef OCL_LIB_HAVE_IMAGES
OCL_REDEFINE ( clEnqueueReadImage )
OCL_REDEFINE ( clEnqueueWriteImage )
OCL_REDEFINE ( clEnqueueCopyImage )
OCL_REDEFINE ( clEnqueueCopyImageToBuffer )
OCL_REDEFINE ( clEnqueueCopyBufferToImage )
OCL_REDEFINE ( clEnqueueMapImage )
#endif
OCL_REDEFINE ( clEnqueueUnmapMemObject )
OCL_REDEFINE ( clEnqueueNDRangeKernel )
OCL_REDEFINE ( clEnqueueTask )
OCL_REDEFINE ( clEnqueueNativeKernel )
OCL_REDEFINE ( clEnqueueMarker )
OCL_REDEFINE ( clEnqueueWaitForEvents )
OCL_REDEFINE ( clEnqueueBarrier )

/* Extension function access */
OCL_REDEFINE ( clGetExtensionFunctionAddress )


/* ----------------------------- */

/* OCL C++ interface functions */

/* from ocl_constant.cc */
OCTAVE_NORETURN extern void ocl_error (const char *fmt, ...);
extern cl_int last_error;
extern bool ocl_check_error (const char *fun);

/* from ocl_lib.cc */
extern void assure_opencl_library (void);
extern void unload_opencl_library (void);
extern bool opencl_library_loaded (void);

/* from ocl_context.cc */
extern unsigned long assure_opencl_context (void);
extern void destroy_opencl_context (void);
extern cl_platform_id get_platform_id (void);
extern cl_device_id get_device_id (void);
extern cl_context get_context (void);
extern cl_command_queue get_command_queue (void);
extern unsigned long opencl_context_id (void);
extern bool opencl_context_active (void);
extern bool opencl_context_id_active (unsigned long);
extern void assure_opencl_context_id (unsigned long id);
extern bool opencl_context_is_fp64 (void);
extern void clear_resources (void);


#ifdef __cplusplus
}
#endif

#endif  /* __OCL_LIB_H */

