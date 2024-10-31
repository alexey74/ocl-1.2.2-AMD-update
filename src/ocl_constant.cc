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
#include <map>

#include "ocl_lib.h"


// ---------- utility functions


#ifndef fract
static
double
fract (double x)
{
  return x-int (x);
}
#endif


// ---------- dictionaries for translating between OpenCL constants and corresponding strings


typedef std::map<std::string, int32_t> DictStr2Int;
typedef std::map<int32_t, std::string> DictInt2Str;

static DictStr2Int oclc_str2int;
static DictInt2Str oclc_int2errstr;


#define REGISTER_STR2INT( xi, xs ) { oclc_str2int[ xs ] = xi; }
#define REGISTER_INT2STR( xi, xs ) { oclc_int2errstr[ xi ] = xs; }
#define REGISTER_BOTH( x ) { REGISTER_STR2INT ( x, #x ); REGISTER_INT2STR ( x, #x ); }
#define REGISTER_ONLY( x ) { REGISTER_STR2INT ( x, #x ); }


static
void
init_dictionaries (void)
{
  REGISTER_BOTH ( CL_SUCCESS )
  REGISTER_BOTH ( CL_DEVICE_NOT_FOUND )
  REGISTER_BOTH ( CL_DEVICE_NOT_AVAILABLE )
  REGISTER_BOTH ( CL_COMPILER_NOT_AVAILABLE )
  REGISTER_BOTH ( CL_MEM_OBJECT_ALLOCATION_FAILURE )
  REGISTER_BOTH ( CL_OUT_OF_RESOURCES )
  REGISTER_BOTH ( CL_OUT_OF_HOST_MEMORY )
  REGISTER_BOTH ( CL_PROFILING_INFO_NOT_AVAILABLE )
  REGISTER_BOTH ( CL_MEM_COPY_OVERLAP )
  REGISTER_BOTH ( CL_IMAGE_FORMAT_MISMATCH )
  REGISTER_BOTH ( CL_IMAGE_FORMAT_NOT_SUPPORTED )
  REGISTER_BOTH ( CL_BUILD_PROGRAM_FAILURE )
  REGISTER_BOTH ( CL_MAP_FAILURE )
  REGISTER_BOTH ( CL_MISALIGNED_SUB_BUFFER_OFFSET )
  REGISTER_BOTH ( CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST )
  REGISTER_BOTH ( CL_INVALID_VALUE )
  REGISTER_BOTH ( CL_INVALID_DEVICE_TYPE )
  REGISTER_BOTH ( CL_INVALID_PLATFORM )
  REGISTER_BOTH ( CL_INVALID_DEVICE )
  REGISTER_BOTH ( CL_INVALID_CONTEXT )
  REGISTER_BOTH ( CL_INVALID_QUEUE_PROPERTIES )
  REGISTER_BOTH ( CL_INVALID_COMMAND_QUEUE )
  REGISTER_BOTH ( CL_INVALID_HOST_PTR )
  REGISTER_BOTH ( CL_INVALID_MEM_OBJECT )
  REGISTER_BOTH ( CL_INVALID_IMAGE_FORMAT_DESCRIPTOR )
  REGISTER_BOTH ( CL_INVALID_IMAGE_SIZE )
  REGISTER_BOTH ( CL_INVALID_SAMPLER )
  REGISTER_BOTH ( CL_INVALID_BINARY )
  REGISTER_BOTH ( CL_INVALID_BUILD_OPTIONS )
  REGISTER_BOTH ( CL_INVALID_PROGRAM )
  REGISTER_BOTH ( CL_INVALID_PROGRAM_EXECUTABLE )
  REGISTER_BOTH ( CL_INVALID_KERNEL_NAME )
  REGISTER_BOTH ( CL_INVALID_KERNEL_DEFINITION )
  REGISTER_BOTH ( CL_INVALID_KERNEL )
  REGISTER_BOTH ( CL_INVALID_ARG_INDEX )
  REGISTER_BOTH ( CL_INVALID_ARG_VALUE )
  REGISTER_BOTH ( CL_INVALID_ARG_SIZE )
  REGISTER_BOTH ( CL_INVALID_KERNEL_ARGS )
  REGISTER_BOTH ( CL_INVALID_WORK_DIMENSION )
  REGISTER_BOTH ( CL_INVALID_WORK_GROUP_SIZE )
  REGISTER_BOTH ( CL_INVALID_WORK_ITEM_SIZE )
  REGISTER_BOTH ( CL_INVALID_GLOBAL_OFFSET )
  REGISTER_BOTH ( CL_INVALID_EVENT_WAIT_LIST )
  REGISTER_BOTH ( CL_INVALID_EVENT )
  REGISTER_BOTH ( CL_INVALID_OPERATION )
  REGISTER_BOTH ( CL_INVALID_GL_OBJECT )
  REGISTER_BOTH ( CL_INVALID_BUFFER_SIZE )
  REGISTER_BOTH ( CL_INVALID_MIP_LEVEL )
  REGISTER_BOTH ( CL_INVALID_GLOBAL_WORK_SIZE )
  REGISTER_BOTH ( CL_INVALID_PROPERTY )
  REGISTER_BOTH ( CL_PLATFORM_NOT_FOUND_KHR )

  REGISTER_ONLY ( CL_VERSION_1_0  )
  REGISTER_ONLY ( CL_VERSION_1_1  )
  REGISTER_ONLY ( CL_FALSE        )
  REGISTER_ONLY ( CL_TRUE )
  REGISTER_ONLY ( CL_PLATFORM_PROFILE     )
  REGISTER_ONLY ( CL_PLATFORM_VERSION     )
  REGISTER_ONLY ( CL_PLATFORM_NAME        )
  REGISTER_ONLY ( CL_PLATFORM_VENDOR      )
  REGISTER_ONLY ( CL_PLATFORM_EXTENSIONS  )
  REGISTER_ONLY ( CL_DEVICE_TYPE_DEFAULT  )
  REGISTER_ONLY ( CL_DEVICE_TYPE_CPU      )
  REGISTER_ONLY ( CL_DEVICE_TYPE_GPU      )
  REGISTER_ONLY ( CL_DEVICE_TYPE_ACCELERATOR      )
  REGISTER_ONLY ( CL_DEVICE_TYPE_ALL      )
  REGISTER_ONLY ( CL_DEVICE_TYPE  )
  REGISTER_ONLY ( CL_DEVICE_VENDOR_ID     )
  REGISTER_ONLY ( CL_DEVICE_MAX_COMPUTE_UNITS     )
  REGISTER_ONLY ( CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS      )
  REGISTER_ONLY ( CL_DEVICE_MAX_WORK_GROUP_SIZE   )
  REGISTER_ONLY ( CL_DEVICE_MAX_WORK_ITEM_SIZES   )
  REGISTER_ONLY ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR   )
  REGISTER_ONLY ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT  )
  REGISTER_ONLY ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT    )
  REGISTER_ONLY ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG   )
  REGISTER_ONLY ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT  )
  REGISTER_ONLY ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE )
  REGISTER_ONLY ( CL_DEVICE_MAX_CLOCK_FREQUENCY   )
  REGISTER_ONLY ( CL_DEVICE_ADDRESS_BITS  )
  REGISTER_ONLY ( CL_DEVICE_MAX_READ_IMAGE_ARGS   )
  REGISTER_ONLY ( CL_DEVICE_MAX_WRITE_IMAGE_ARGS  )
  REGISTER_ONLY ( CL_DEVICE_MAX_MEM_ALLOC_SIZE    )
  REGISTER_ONLY ( CL_DEVICE_IMAGE2D_MAX_WIDTH     )
  REGISTER_ONLY ( CL_DEVICE_IMAGE2D_MAX_HEIGHT    )
  REGISTER_ONLY ( CL_DEVICE_IMAGE3D_MAX_WIDTH     )
  REGISTER_ONLY ( CL_DEVICE_IMAGE3D_MAX_HEIGHT    )
  REGISTER_ONLY ( CL_DEVICE_IMAGE3D_MAX_DEPTH     )
  REGISTER_ONLY ( CL_DEVICE_IMAGE_SUPPORT )
  REGISTER_ONLY ( CL_DEVICE_MAX_PARAMETER_SIZE    )
  REGISTER_ONLY ( CL_DEVICE_MAX_SAMPLERS  )
  REGISTER_ONLY ( CL_DEVICE_MEM_BASE_ADDR_ALIGN   )
  REGISTER_ONLY ( CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE      )
  REGISTER_ONLY ( CL_DEVICE_SINGLE_FP_CONFIG      )
  REGISTER_ONLY ( CL_DEVICE_GLOBAL_MEM_CACHE_TYPE )
  REGISTER_ONLY ( CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE     )
  REGISTER_ONLY ( CL_DEVICE_GLOBAL_MEM_CACHE_SIZE )
  REGISTER_ONLY ( CL_DEVICE_GLOBAL_MEM_SIZE       )
  REGISTER_ONLY ( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE      )
  REGISTER_ONLY ( CL_DEVICE_MAX_CONSTANT_ARGS     )
  REGISTER_ONLY ( CL_DEVICE_LOCAL_MEM_TYPE        )
  REGISTER_ONLY ( CL_DEVICE_LOCAL_MEM_SIZE        )
  REGISTER_ONLY ( CL_DEVICE_ERROR_CORRECTION_SUPPORT      )
  REGISTER_ONLY ( CL_DEVICE_PROFILING_TIMER_RESOLUTION    )
  REGISTER_ONLY ( CL_DEVICE_ENDIAN_LITTLE )
  REGISTER_ONLY ( CL_DEVICE_AVAILABLE     )
  REGISTER_ONLY ( CL_DEVICE_COMPILER_AVAILABLE    )
  REGISTER_ONLY ( CL_DEVICE_EXECUTION_CAPABILITIES        )
  REGISTER_ONLY ( CL_DEVICE_QUEUE_PROPERTIES      )
  REGISTER_ONLY ( CL_DEVICE_NAME  )
  REGISTER_ONLY ( CL_DEVICE_VENDOR        )
  REGISTER_ONLY ( CL_DRIVER_VERSION       )
  REGISTER_ONLY ( CL_DEVICE_PROFILE       )
  REGISTER_ONLY ( CL_DEVICE_VERSION       )
  REGISTER_ONLY ( CL_DEVICE_EXTENSIONS    )
  REGISTER_ONLY ( CL_DEVICE_PLATFORM      )
  REGISTER_ONLY ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF   )
  REGISTER_ONLY ( CL_DEVICE_HOST_UNIFIED_MEMORY   )
  REGISTER_ONLY ( CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR      )
  REGISTER_ONLY ( CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT     )
  REGISTER_ONLY ( CL_DEVICE_NATIVE_VECTOR_WIDTH_INT       )
  REGISTER_ONLY ( CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG      )
  REGISTER_ONLY ( CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT     )
  REGISTER_ONLY ( CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE    )
  REGISTER_ONLY ( CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF      )
  REGISTER_ONLY ( CL_DEVICE_OPENCL_C_VERSION      )
  REGISTER_ONLY ( CL_FP_DENORM    )
  REGISTER_ONLY ( CL_FP_INF_NAN   )
  REGISTER_ONLY ( CL_FP_ROUND_TO_NEAREST  )
  REGISTER_ONLY ( CL_FP_ROUND_TO_ZERO     )
  REGISTER_ONLY ( CL_FP_ROUND_TO_INF      )
  REGISTER_ONLY ( CL_FP_FMA       )
  REGISTER_ONLY ( CL_FP_SOFT_FLOAT        )
  REGISTER_ONLY ( CL_NONE )
  REGISTER_ONLY ( CL_READ_ONLY_CACHE      )
  REGISTER_ONLY ( CL_READ_WRITE_CACHE     )
  REGISTER_ONLY ( CL_LOCAL        )
  REGISTER_ONLY ( CL_GLOBAL       )
  REGISTER_ONLY ( CL_EXEC_KERNEL  )
  REGISTER_ONLY ( CL_EXEC_NATIVE_KERNEL   )
  REGISTER_ONLY ( CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE  )
  REGISTER_ONLY ( CL_QUEUE_PROFILING_ENABLE       )
  REGISTER_ONLY ( CL_CONTEXT_REFERENCE_COUNT      )
  REGISTER_ONLY ( CL_CONTEXT_DEVICES      )
  REGISTER_ONLY ( CL_CONTEXT_PROPERTIES   )
  REGISTER_ONLY ( CL_CONTEXT_NUM_DEVICES  )
  REGISTER_ONLY ( CL_CONTEXT_PLATFORM     )
  REGISTER_ONLY ( CL_QUEUE_CONTEXT        )
  REGISTER_ONLY ( CL_QUEUE_DEVICE )
  REGISTER_ONLY ( CL_QUEUE_REFERENCE_COUNT        )
  REGISTER_ONLY ( CL_QUEUE_PROPERTIES     )
  REGISTER_ONLY ( CL_MEM_READ_WRITE       )
  REGISTER_ONLY ( CL_MEM_WRITE_ONLY       )
  REGISTER_ONLY ( CL_MEM_READ_ONLY        )
  REGISTER_ONLY ( CL_MEM_USE_HOST_PTR     )
  REGISTER_ONLY ( CL_MEM_ALLOC_HOST_PTR   )
  REGISTER_ONLY ( CL_MEM_COPY_HOST_PTR    )
  REGISTER_ONLY ( CL_R    )
  REGISTER_ONLY ( CL_A    )
  REGISTER_ONLY ( CL_RG   )
  REGISTER_ONLY ( CL_RA   )
  REGISTER_ONLY ( CL_RGB  )
  REGISTER_ONLY ( CL_RGBA )
  REGISTER_ONLY ( CL_BGRA )
  REGISTER_ONLY ( CL_ARGB )
  REGISTER_ONLY ( CL_INTENSITY    )
  REGISTER_ONLY ( CL_LUMINANCE    )
  REGISTER_ONLY ( CL_Rx   )
  REGISTER_ONLY ( CL_RGx  )
  REGISTER_ONLY ( CL_RGBx )
  REGISTER_ONLY ( CL_SNORM_INT8   )
  REGISTER_ONLY ( CL_SNORM_INT16  )
  REGISTER_ONLY ( CL_UNORM_INT8   )
  REGISTER_ONLY ( CL_UNORM_INT16  )
  REGISTER_ONLY ( CL_UNORM_SHORT_565      )
  REGISTER_ONLY ( CL_UNORM_SHORT_555      )
  REGISTER_ONLY ( CL_UNORM_INT_101010     )
  REGISTER_ONLY ( CL_SIGNED_INT8  )
  REGISTER_ONLY ( CL_SIGNED_INT16 )
  REGISTER_ONLY ( CL_SIGNED_INT32 )
  REGISTER_ONLY ( CL_UNSIGNED_INT8        )
  REGISTER_ONLY ( CL_UNSIGNED_INT16       )
  REGISTER_ONLY ( CL_UNSIGNED_INT32       )
  REGISTER_ONLY ( CL_HALF_FLOAT   )
  REGISTER_ONLY ( CL_FLOAT        )
  REGISTER_ONLY ( CL_MEM_OBJECT_BUFFER    )
  REGISTER_ONLY ( CL_MEM_OBJECT_IMAGE2D   )
  REGISTER_ONLY ( CL_MEM_OBJECT_IMAGE3D   )
  REGISTER_ONLY ( CL_MEM_TYPE     )
  REGISTER_ONLY ( CL_MEM_FLAGS    )
  REGISTER_ONLY ( CL_MEM_SIZE     )
  REGISTER_ONLY ( CL_MEM_HOST_PTR )
  REGISTER_ONLY ( CL_MEM_MAP_COUNT        )
  REGISTER_ONLY ( CL_MEM_REFERENCE_COUNT  )
  REGISTER_ONLY ( CL_MEM_CONTEXT  )
  REGISTER_ONLY ( CL_MEM_ASSOCIATED_MEMOBJECT     )
  REGISTER_ONLY ( CL_MEM_OFFSET   )
  REGISTER_ONLY ( CL_IMAGE_FORMAT )
  REGISTER_ONLY ( CL_IMAGE_ELEMENT_SIZE   )
  REGISTER_ONLY ( CL_IMAGE_ROW_PITCH      )
  REGISTER_ONLY ( CL_IMAGE_SLICE_PITCH    )
  REGISTER_ONLY ( CL_IMAGE_WIDTH  )
  REGISTER_ONLY ( CL_IMAGE_HEIGHT )
  REGISTER_ONLY ( CL_IMAGE_DEPTH  )
  REGISTER_ONLY ( CL_ADDRESS_NONE )
  REGISTER_ONLY ( CL_ADDRESS_CLAMP_TO_EDGE        )
  REGISTER_ONLY ( CL_ADDRESS_CLAMP        )
  REGISTER_ONLY ( CL_ADDRESS_REPEAT       )
  REGISTER_ONLY ( CL_ADDRESS_MIRRORED_REPEAT      )
  REGISTER_ONLY ( CL_FILTER_NEAREST       )
  REGISTER_ONLY ( CL_FILTER_LINEAR        )
  REGISTER_ONLY ( CL_SAMPLER_REFERENCE_COUNT      )
  REGISTER_ONLY ( CL_SAMPLER_CONTEXT      )
  REGISTER_ONLY ( CL_SAMPLER_NORMALIZED_COORDS    )
  REGISTER_ONLY ( CL_SAMPLER_ADDRESSING_MODE      )
  REGISTER_ONLY ( CL_SAMPLER_FILTER_MODE  )
  REGISTER_ONLY ( CL_MAP_READ     )
  REGISTER_ONLY ( CL_MAP_WRITE    )
  REGISTER_ONLY ( CL_PROGRAM_REFERENCE_COUNT      )
  REGISTER_ONLY ( CL_PROGRAM_CONTEXT      )
  REGISTER_ONLY ( CL_PROGRAM_NUM_DEVICES  )
  REGISTER_ONLY ( CL_PROGRAM_DEVICES      )
  REGISTER_ONLY ( CL_PROGRAM_SOURCE       )
  REGISTER_ONLY ( CL_PROGRAM_BINARY_SIZES )
  REGISTER_ONLY ( CL_PROGRAM_BINARIES     )
  REGISTER_ONLY ( CL_PROGRAM_BUILD_STATUS )
  REGISTER_ONLY ( CL_PROGRAM_BUILD_OPTIONS        )
  REGISTER_ONLY ( CL_PROGRAM_BUILD_LOG    )
  REGISTER_ONLY ( CL_BUILD_SUCCESS        )
  REGISTER_ONLY ( CL_BUILD_NONE   )
  REGISTER_ONLY ( CL_BUILD_ERROR  )
  REGISTER_ONLY ( CL_BUILD_IN_PROGRESS    )
  REGISTER_ONLY ( CL_KERNEL_FUNCTION_NAME )
  REGISTER_ONLY ( CL_KERNEL_NUM_ARGS      )
  REGISTER_ONLY ( CL_KERNEL_REFERENCE_COUNT       )
  REGISTER_ONLY ( CL_KERNEL_CONTEXT       )
  REGISTER_ONLY ( CL_KERNEL_PROGRAM       )
  REGISTER_ONLY ( CL_KERNEL_WORK_GROUP_SIZE       )
  REGISTER_ONLY ( CL_KERNEL_COMPILE_WORK_GROUP_SIZE       )
  REGISTER_ONLY ( CL_KERNEL_LOCAL_MEM_SIZE        )
  REGISTER_ONLY ( CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE    )
  REGISTER_ONLY ( CL_KERNEL_PRIVATE_MEM_SIZE      )
  REGISTER_ONLY ( CL_EVENT_COMMAND_QUEUE  )
  REGISTER_ONLY ( CL_EVENT_COMMAND_TYPE   )
  REGISTER_ONLY ( CL_EVENT_REFERENCE_COUNT        )
  REGISTER_ONLY ( CL_EVENT_COMMAND_EXECUTION_STATUS       )
  REGISTER_ONLY ( CL_EVENT_CONTEXT        )
  REGISTER_ONLY ( CL_COMMAND_NDRANGE_KERNEL       )
  REGISTER_ONLY ( CL_COMMAND_TASK )
  REGISTER_ONLY ( CL_COMMAND_NATIVE_KERNEL        )
  REGISTER_ONLY ( CL_COMMAND_READ_BUFFER  )
  REGISTER_ONLY ( CL_COMMAND_WRITE_BUFFER )
  REGISTER_ONLY ( CL_COMMAND_COPY_BUFFER  )
  REGISTER_ONLY ( CL_COMMAND_READ_IMAGE   )
  REGISTER_ONLY ( CL_COMMAND_WRITE_IMAGE  )
  REGISTER_ONLY ( CL_COMMAND_COPY_IMAGE   )
  REGISTER_ONLY ( CL_COMMAND_COPY_IMAGE_TO_BUFFER )
  REGISTER_ONLY ( CL_COMMAND_COPY_BUFFER_TO_IMAGE )
  REGISTER_ONLY ( CL_COMMAND_MAP_BUFFER   )
  REGISTER_ONLY ( CL_COMMAND_MAP_IMAGE    )
  REGISTER_ONLY ( CL_COMMAND_UNMAP_MEM_OBJECT     )
  REGISTER_ONLY ( CL_COMMAND_MARKER       )
  REGISTER_ONLY ( CL_COMMAND_ACQUIRE_GL_OBJECTS   )
  REGISTER_ONLY ( CL_COMMAND_RELEASE_GL_OBJECTS   )
  REGISTER_ONLY ( CL_COMMAND_READ_BUFFER_RECT     )
  REGISTER_ONLY ( CL_COMMAND_WRITE_BUFFER_RECT    )
  REGISTER_ONLY ( CL_COMMAND_COPY_BUFFER_RECT     )
  REGISTER_ONLY ( CL_COMMAND_USER )
  REGISTER_ONLY ( CL_COMPLETE     )
  REGISTER_ONLY ( CL_RUNNING      )
  REGISTER_ONLY ( CL_SUBMITTED    )
  REGISTER_ONLY ( CL_QUEUED       )
  REGISTER_ONLY ( CL_BUFFER_CREATE_TYPE_REGION    )
  REGISTER_ONLY ( CL_PROFILING_COMMAND_QUEUED     )
  REGISTER_ONLY ( CL_PROFILING_COMMAND_SUBMIT     )
  REGISTER_ONLY ( CL_PROFILING_COMMAND_START      )
  REGISTER_ONLY ( CL_PROFILING_COMMAND_END        )
  REGISTER_ONLY ( CL_DEVICE_DOUBLE_FP_CONFIG      )
  REGISTER_ONLY ( CL_DEVICE_HALF_FP_CONFIG        )
}


bool
translate_cl_string_to_int (const std::string& str_in, cl_int& value_out)
{
  if (oclc_str2int.empty ())
    init_dictionaries ();

  DictStr2Int::iterator it = oclc_str2int.find (str_in), end = oclc_str2int.end ();
  if (it == end)
    return false;
  value_out = it->second;
  return true;
}


bool
translate_cl_int_to_errstring (cl_int value, std::string& str_out)
{
  if (oclc_str2int.empty ())
    init_dictionaries ();

  DictInt2Str::iterator it = oclc_int2errstr.find (value), end = oclc_int2errstr.end ();
  if (it == end)
    return false;
  str_out = it->second;
  return true;
}


// ---------- error handling data and functions


cl_int last_error = CL_SUCCESS;


void
ocl_error (const char *fmt, ...)
{
  va_list args;
  va_start (args, fmt);
  (*current_liboctave_error_handler) (fmt, args); // this should never return; unless someone applies a hacks
  va_end (args);
  throw 0; // we should never get here; throw, if we do, to assure the function never returns
}


bool
ocl_check_error (const char *fun)
{
  if (last_error == CL_SUCCESS)
    return true;

  std::string error_str;
  if (!translate_cl_int_to_errstring (last_error, error_str))
    error_str = "<unknown error>";

  octave_stdout << "ocl: calling OpenCL function '" << fun << "'\n";
  octave_stdout << "  returned error '" << error_str << "' (" << last_error << ").\n";

  if (last_error == CL_PLATFORM_NOT_FOUND_KHR)
    octave_stdout << "  Please check your OpenCL installation.\n";

  ocl_error ("OpenCL function call error");

  return false;
}


// ---------- the octave entry point to the 'ocl_constant' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_constant", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_constant", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_constant, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {@var{x} =}   ocl_constant (@var{str}) \n\
@deftypefnx {Loadable Function} {@var{str} =} ocl_constant (@var{x}) \n\
\n\
Translate an OpenCL constant.  \n\
\n\
In the first form, translate the OpenCL constant given as string @var{str} \n\
to its numeric value @var{x}.  Example:  \n\
\n\
@example \n\
@group \n\
ocl_constant (\"CL_DEVICE_TYPE_GPU\") \n\
@result{} 4 \n\
@end group \n\
@end example \n\
\n\
In the second form, translate the OpenCL error code given as negative integer \n\
@var{x} to its human-readable string value @var{str}.  Example:  \n\
\n\
@example \n\
@group \n\
ocl_constant (-5) \n\
@result{} CL_OUT_OF_RESOURCES \n\
@end group \n\
@end example \n\
\n\
@seealso{oclArray} \n\
@end deftypefn")
{
  octave_value_list retval;
  int nargin = args.length ();

  std::string fcn;
  if ((nargin > 0) && (args (0).is_string ()))
    fcn = args (0).char_matrix_value ().row_as_string (0);

  if ((nargin != 1) ||
      (((!args (0).is_string ()) || (args (0).char_matrix_value ().rows () != 1)) &&
       ((!args (0).is_real_scalar ()) || (fract (args (0).row_vector_value ().elem (0)) != 0.0)))) {

    ocl_error ("the single argument must be an OpenCL constant string, or an integer as OpenCL error code");

  // OpenCL constant translation functions

  } else if (args (0).is_real_scalar ()) {

    cl_int value = args (0).row_vector_value ().elem (0);
    std::string str;
    if (translate_cl_int_to_errstring (value, str))
      retval (0) = octave_value (charMatrix (str));
    else
      ocl_error ("cannot translate unknown OpenCL error code");

  } else {

    charMatrix ch = args (0).char_matrix_value ();
    std::string str = ch.row_as_string (0);
    cl_int value;
    if (translate_cl_string_to_int (str, value))
      retval (0) = octave_value (value);
    else
      ocl_error ("cannot translate unknown OpenCL string");

  }

  return retval;
}
