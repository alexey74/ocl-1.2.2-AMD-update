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
#include <octave/ov-struct.h>
#include <string>
#include <set>
#include <vector>

#include "ocl_lib.h"
#include "ocl_memobj.h"


// ---------- platform and device (=resources) data and functions


typedef std::vector<cl_platform_id> Platforms_t;

static Platforms_t platforms;

typedef std::vector<cl_device_id> Devices_of_platform_t;
typedef std::vector<Devices_of_platform_t> Devices_t;

static Devices_t devices;

typedef std::set<int32_t> DevPropSet_t;

static DevPropSet_t dev_props_char;
static DevPropSet_t dev_props_ulong;

static octave_scalar_map ocl_resources;

static std::string selection ("auto");

static Matrix device (2,1);
static bool device_fp64;


static
void get_resources (void);


static
void
select_device (void)
{
  if (ocl_resources.nfields () == 0) {
    get_resources ();
    device (0) = -1;
  }

  if (!(device (0) < 0))
    return;

  Cell summary = ocl_resources.getfield ("summary").cell_value ();

  if (selection == "auto") {
    device (0) = summary (0).scalar_map_value ().getfield ("platform_index").double_value ();
    device (1) = summary (0).scalar_map_value ().getfield ("device_index").double_value ();
    device_fp64 = summary (0).scalar_map_value ().getfield ("fp64").double_value ();
    return;
  }

  std::string type = selection.substr (0, 3);

  std::string num = "0";
  if (selection.length () > 3)
    num = selection.substr (3, selection.length ()-3);
  long numi = atol (num.c_str ());

  long num_all_devices = summary.dim1 ();

  if (type == "dev") {
    if (numi >= num_all_devices)
      ocl_error ("device_selection: explicitly specified OpenCL device not found");

    device (0) = summary (numi).scalar_map_value ().getfield ("platform_index").double_value ();
    device (1) = summary (numi).scalar_map_value ().getfield ("device_index").double_value ();
    device_fp64 = summary (numi).scalar_map_value ().getfield ("fp64").double_value ();
    return;
  }

  long count_type = 0;
  for (long idev=0; idev<num_all_devices; idev++) {
    if (summary (idev).scalar_map_value ().getfield ("type").string_value () == type) {
      if (count_type == numi) {
        device (0) = summary (idev).scalar_map_value ().getfield ("platform_index").double_value ();
        device (1) = summary (idev).scalar_map_value ().getfield ("device_index").double_value ();
        device_fp64 = summary (idev).scalar_map_value ().getfield ("fp64").double_value ();
        return;
      }
      count_type++;
    }
  }

  if (count_type == 0)
    ocl_error ("device_selection: no OpenCL devices of requested type found");
  else
    ocl_error ("device_selection: explicitly specified OpenCL device not found");
}


static
void
init_props ()
{
  dev_props_char.insert ( CL_DEVICE_NAME );
  dev_props_char.insert ( CL_DEVICE_VENDOR );
  dev_props_char.insert ( CL_DRIVER_VERSION );
  dev_props_char.insert ( CL_DEVICE_PROFILE );
  dev_props_char.insert ( CL_DEVICE_VERSION );
  dev_props_char.insert ( CL_DEVICE_EXTENSIONS );
  dev_props_char.insert ( CL_DEVICE_OPENCL_C_VERSION );

  dev_props_ulong.insert ( CL_DEVICE_TYPE );
  dev_props_ulong.insert ( CL_DEVICE_VENDOR_ID );
  dev_props_ulong.insert ( CL_DEVICE_MAX_COMPUTE_UNITS );
  dev_props_ulong.insert ( CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS );
  dev_props_ulong.insert ( CL_DEVICE_MAX_WORK_GROUP_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR );
  dev_props_ulong.insert ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT );
  dev_props_ulong.insert ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT );
  dev_props_ulong.insert ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG );
  dev_props_ulong.insert ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT );
  dev_props_ulong.insert ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE );
  dev_props_ulong.insert ( CL_DEVICE_MAX_CLOCK_FREQUENCY );
  dev_props_ulong.insert ( CL_DEVICE_ADDRESS_BITS );
  dev_props_ulong.insert ( CL_DEVICE_MAX_READ_IMAGE_ARGS );
  dev_props_ulong.insert ( CL_DEVICE_MAX_WRITE_IMAGE_ARGS );
  dev_props_ulong.insert ( CL_DEVICE_MAX_MEM_ALLOC_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_IMAGE2D_MAX_WIDTH );
  dev_props_ulong.insert ( CL_DEVICE_IMAGE2D_MAX_HEIGHT );
  dev_props_ulong.insert ( CL_DEVICE_IMAGE3D_MAX_WIDTH );
  dev_props_ulong.insert ( CL_DEVICE_IMAGE3D_MAX_HEIGHT );
  dev_props_ulong.insert ( CL_DEVICE_IMAGE3D_MAX_DEPTH );
  dev_props_ulong.insert ( CL_DEVICE_IMAGE_SUPPORT );
  dev_props_ulong.insert ( CL_DEVICE_MAX_PARAMETER_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_MAX_SAMPLERS );
  dev_props_ulong.insert ( CL_DEVICE_MEM_BASE_ADDR_ALIGN );
  dev_props_ulong.insert ( CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_SINGLE_FP_CONFIG );
  dev_props_ulong.insert ( CL_DEVICE_GLOBAL_MEM_CACHE_TYPE );
  dev_props_ulong.insert ( CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_GLOBAL_MEM_CACHE_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_GLOBAL_MEM_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_MAX_CONSTANT_ARGS );
  dev_props_ulong.insert ( CL_DEVICE_LOCAL_MEM_TYPE );
  dev_props_ulong.insert ( CL_DEVICE_LOCAL_MEM_SIZE );
  dev_props_ulong.insert ( CL_DEVICE_ERROR_CORRECTION_SUPPORT );
  dev_props_ulong.insert ( CL_DEVICE_PROFILING_TIMER_RESOLUTION );
  dev_props_ulong.insert ( CL_DEVICE_ENDIAN_LITTLE );
  dev_props_ulong.insert ( CL_DEVICE_AVAILABLE );
  dev_props_ulong.insert ( CL_DEVICE_COMPILER_AVAILABLE );
  dev_props_ulong.insert ( CL_DEVICE_EXECUTION_CAPABILITIES );
  dev_props_ulong.insert ( CL_DEVICE_QUEUE_PROPERTIES );
  dev_props_ulong.insert ( CL_DEVICE_DOUBLE_FP_CONFIG );
  dev_props_ulong.insert ( CL_DEVICE_HALF_FP_CONFIG );
  dev_props_ulong.insert ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF );
  dev_props_ulong.insert ( CL_DEVICE_HOST_UNIFIED_MEMORY );
  dev_props_ulong.insert ( CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR );
  dev_props_ulong.insert ( CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT );
  dev_props_ulong.insert ( CL_DEVICE_NATIVE_VECTOR_WIDTH_INT );
  dev_props_ulong.insert ( CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG );
  dev_props_ulong.insert ( CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT );
  dev_props_ulong.insert ( CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE );
  dev_props_ulong.insert ( CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF );
}


static
bool
is_char_dev_prop (cl_int prop)
{
  if (dev_props_char.empty ())
    init_props ();

  return (dev_props_char.find (prop) != dev_props_char.end ());
}


static
bool
is_ulong_dev_prop (cl_int prop)
{
  if (dev_props_char.empty ())
    init_props ();

  return (dev_props_ulong.find (prop) != dev_props_ulong.end ());
}


static
octave_value
get_platform_prop (cl_platform_id platform, cl_platform_info property)
{
  octave_value retval;
  size_t property_length = 0;

  last_error = clGetPlatformInfo (platform, property, 0, 0, & property_length);
  ocl_check_error ("clGetPlatformInfo");
  char *property_value = new char[property_length];
  last_error = clGetPlatformInfo (platform, property, property_length, property_value, 0);
  ocl_check_error ("clGetPlatformInfo");
  retval = octave_value (charMatrix (property_value));
  delete[] property_value;

  return retval;
}


static
octave_value
get_device_prop (cl_device_id device, cl_device_info property)
{
  octave_value retval;
  size_t property_length = 0;

  if (is_char_dev_prop (property)) {

    last_error = clGetDeviceInfo (device, property, 0, 0, & property_length);
    ocl_check_error ("clGetDeviceInfo");
    char *property_value = new char[property_length];
    last_error = clGetDeviceInfo (device,property, property_length, property_value, 0);
    ocl_check_error ("clGetDeviceInfo");
    retval = octave_value (charMatrix (property_value));
    delete[] property_value;

  } else if (is_ulong_dev_prop (property)) {

    cl_ulong property_value = 0;
    last_error = clGetDeviceInfo (device, property, sizeof (cl_ulong), & property_value, 0);
    if ((last_error == CL_INVALID_VALUE) && ((property == CL_DEVICE_DOUBLE_FP_CONFIG) || (property == CL_DEVICE_HALF_FP_CONFIG))) {
      property_value = 0;
      last_error = CL_SUCCESS;
    }
    ocl_check_error ("clGetDeviceInfo");
    retval = octave_value (property_value);

  } else if (property == CL_DEVICE_MAX_WORK_ITEM_SIZES) {

    last_error = clGetDeviceInfo (device, property, 0, 0, & property_length);
    ocl_check_error ("clGetDeviceInfo");
    int dimensions =  (property_length-1)/sizeof (size_t)+1;
    size_t property_value[dimensions];
    last_error = clGetDeviceInfo (device, property, property_length, property_value, 0);
    ocl_check_error ("clGetDeviceInfo");

    Matrix ret (1, dimensions);
    for (int i=0; i<dimensions; i++)
      ret (i) = property_value[i];
    retval = octave_value (ret);

  } else {
    ocl_error ("unknown device property");
  }

  return retval;
}


void
clear_resources (void)
{
  ocl_resources.clear ();
  device (0) = -1;
  device (1) = -1;
}


static
void
get_resources (void)
{
  ocl_resources.clear ();
  platforms.clear ();
  devices.clear ();
  cl_uint num_platforms = 0;
  cl_uint num_all_devices = 0;

  assure_opencl_library ();

  last_error = clGetPlatformIDs (0, 0, & num_platforms);
  ocl_check_error ("clGetPlatformIDs");

  if (num_platforms == 0)
    ocl_error ("could not find any OpenCL platforms -- please check your OpenCL installation");

  cl_platform_id platform_ids[num_platforms];

  last_error = clGetPlatformIDs (num_platforms, platform_ids, 0);
  ocl_check_error ("clGetPlatformIDs");

  platforms.resize (num_platforms);
  Cell all_platforms_props (num_platforms,1);

  for (cl_uint platform_index=0; platform_index<num_platforms; platform_index++) {
    cl_platform_id platform = platforms[platform_index] = platform_ids[platform_index];

    octave_scalar_map platform_props;
    platform_props.setfield ("platform_index", octave_value (platform_index));
    platform_props.setfield ("name",           get_platform_prop (platform, CL_PLATFORM_NAME));
    platform_props.setfield ("version",        get_platform_prop (platform, CL_PLATFORM_VERSION));
    platform_props.setfield ("profile",        get_platform_prop (platform, CL_PLATFORM_PROFILE));
    platform_props.setfield ("vendor",         get_platform_prop (platform, CL_PLATFORM_VENDOR));
    platform_props.setfield ("extensions",     get_platform_prop (platform, CL_PLATFORM_EXTENSIONS));

    all_platforms_props (platform_index) = octave_value (platform_props);
  }

  ocl_resources.setfield ("platforms", octave_value (all_platforms_props));

  devices.resize (num_platforms);
  Cell all_devices_props (num_platforms,1);
  cl_device_type device_types = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR;
  cl_uint num_devices_of_platform;

  for (cl_uint platform_index=0; platform_index<num_platforms; platform_index++) {
    cl_platform_id platform = platforms[platform_index];
    last_error = clGetDeviceIDs (platform, device_types, 0, 0, & num_devices_of_platform);
    ocl_check_error ("clGetDeviceIDs");

    cl_device_id device_ids[num_devices_of_platform];

    last_error = clGetDeviceIDs (platform, device_types, num_devices_of_platform, device_ids, 0);
    ocl_check_error ("clGetDeviceIDs");

    devices[platform_index].resize (num_devices_of_platform);
    Cell all_platform_devices_props (num_devices_of_platform,1);

    for (cl_uint device_index=0; device_index<num_devices_of_platform; device_index++) {
      num_all_devices++;
      cl_device_id device = devices[platform_index][device_index] = device_ids[device_index];

      octave_value extensions =               get_device_prop (device, CL_DEVICE_EXTENSIONS);
      std::string ext = " "+extensions.string_value ()+" ";
      double half_supported = 0, single_supported = 1, double_supported = 0;
      if (ext.find (" cl_khr_fp16 ") < std::string::npos) half_supported = 1;
      if (ext.find (" cl_khr_fp64 ") < std::string::npos) double_supported = 1;

      octave_scalar_map device_props, t1, t2, t3;

      device_props.setfield ("platform_index", octave_value (platform_index));
      device_props.setfield ("device_index",   octave_value (device_index));

      device_props.setfield ("name",           get_device_prop (device, CL_DEVICE_NAME));
      device_props.setfield ("vendor",         get_device_prop (device, CL_DEVICE_VENDOR));
      device_props.setfield ("type",           get_device_prop (device, CL_DEVICE_TYPE));

      t1.clear (); // .version.
        t1.setfield ("driver",                 get_device_prop (device, CL_DRIVER_VERSION));
        t1.setfield ("device",                 get_device_prop (device, CL_DEVICE_VERSION));
        t1.setfield ("opencl_c",               get_device_prop (device, CL_DEVICE_OPENCL_C_VERSION));
        t1.setfield ("profile",                get_device_prop (device, CL_DEVICE_PROFILE));
        t1.setfield ("vendorid",               get_device_prop (device, CL_DEVICE_VENDOR_ID));
      device_props.setfield ("version",        octave_value (t1));

      t1.clear (); // .compute.
        t1.setfield ("units",                  get_device_prop (device, CL_DEVICE_MAX_COMPUTE_UNITS));
        t1.setfield ("max_dimension",          get_device_prop (device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
        t1.setfield ("max_workgroup_size",     get_device_prop (device, CL_DEVICE_MAX_WORK_GROUP_SIZE));
        t1.setfield ("max_workitems_size",     get_device_prop (device, CL_DEVICE_MAX_WORK_ITEM_SIZES));
        t1.setfield ("clock_frequency",        get_device_prop (device, CL_DEVICE_MAX_CLOCK_FREQUENCY));
      device_props.setfield ("compute",        octave_value (t1));

      t1.clear (); // .mem.
        t2.clear (); // .mem.global.
          t2.setfield ("size",                 get_device_prop (device, CL_DEVICE_GLOBAL_MEM_SIZE));
          t2.setfield ("max_alloc",            get_device_prop (device, CL_DEVICE_MAX_MEM_ALLOC_SIZE));
          t3.clear (); // .mem.global.cache.
            t3.setfield ("size",               get_device_prop (device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE));
            t3.setfield ("type",               get_device_prop (device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE));
            t3.setfield ("line_size",          get_device_prop (device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE));
          t2.setfield ("cache",                octave_value (t3));
        t1.setfield ("global",                 octave_value (t2));
        t2.clear (); // .mem.local.
          t2.setfield ("size",                 get_device_prop (device, CL_DEVICE_LOCAL_MEM_SIZE));
          t2.setfield ("type",                 get_device_prop (device, CL_DEVICE_LOCAL_MEM_TYPE));
        t1.setfield ("local",                  octave_value (t2));
        t2.clear (); // .mem.const.
          t2.setfield ("size",                 get_device_prop (device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE));
          t2.setfield ("args",                 get_device_prop (device, CL_DEVICE_MAX_CONSTANT_ARGS));
        t1.setfield ("const",                  octave_value (t2));
        t2.clear (); // .mem.param.
          t2.setfield ("arg_size",             get_device_prop (device, CL_DEVICE_MAX_PARAMETER_SIZE));
        t1.setfield ("param",                  octave_value (t2));
        t1.setfield ("address_bits",           get_device_prop (device, CL_DEVICE_ADDRESS_BITS));
        t2.clear (); // .mem.align.
          t2.setfield ("base_addr",            get_device_prop (device, CL_DEVICE_MEM_BASE_ADDR_ALIGN));
          t2.setfield ("data_type",            get_device_prop (device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE));
        t1.setfield ("align",                  octave_value (t2));
        t1.setfield ("little_endian",          get_device_prop (device, CL_DEVICE_ENDIAN_LITTLE));
        t1.setfield ("host_unified",           get_device_prop (device, CL_DEVICE_HOST_UNIFIED_MEMORY));
        t2.clear (); // .mem.vector_width.
          t3.clear (); // .mem.vector_width.native.
            t3.setfield ("char",               get_device_prop (device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR));
            t3.setfield ("short",              get_device_prop (device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT));
            t3.setfield ("int",                get_device_prop (device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT));
            t3.setfield ("long",               get_device_prop (device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG));
            t3.setfield ("half",               get_device_prop (device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF));
            t3.setfield ("float",              get_device_prop (device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT));
            t3.setfield ("double",             get_device_prop (device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE));
          t2.setfield ("native",               octave_value (t3));
          t3.clear (); // .mem.vector_width.preferred.
            t3.setfield ("char",               get_device_prop (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR));
            t3.setfield ("short",              get_device_prop (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT));
            t3.setfield ("int",                get_device_prop (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT));
            t3.setfield ("long",               get_device_prop (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG));
            t3.setfield ("half",               get_device_prop (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF));
            t3.setfield ("float",              get_device_prop (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT));
            t3.setfield ("double",             get_device_prop (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE));
          t2.setfield ("preferred",            octave_value (t3));
        t1.setfield ("vector_width",           octave_value (t2));
      device_props.setfield ("mem",            octave_value (t1));

      t1.clear (); // .caps.
        t1.setfield ("device_available",       get_device_prop (device, CL_DEVICE_AVAILABLE));
        t1.setfield ("compiler_available",     get_device_prop (device, CL_DEVICE_COMPILER_AVAILABLE));
        t1.setfield ("queue_props",            get_device_prop (device, CL_DEVICE_QUEUE_PROPERTIES));
        t1.setfield ("execution",              get_device_prop (device, CL_DEVICE_EXECUTION_CAPABILITIES));
        t1.setfield ("profile_timer_res",      get_device_prop (device, CL_DEVICE_PROFILING_TIMER_RESOLUTION));
        t1.setfield ("error_correction",       get_device_prop (device, CL_DEVICE_ERROR_CORRECTION_SUPPORT));
        t2.clear (); // .caps.half.
          t2.setfield ("supported",            octave_value (half_supported));
          t2.setfield ("fp_config",            get_device_prop (device, CL_DEVICE_HALF_FP_CONFIG));
        t1.setfield ("half",                   octave_value (t2));
        t2.clear (); // .caps.single.
          t2.setfield ("supported",            octave_value (single_supported));
          t2.setfield ("fp_config",            get_device_prop (device, CL_DEVICE_SINGLE_FP_CONFIG));
        t1.setfield ("single",                 octave_value (t2));
        t2.clear (); // .caps.double.
          t2.setfield ("supported",            octave_value (double_supported));
          t2.setfield ("fp_config",            get_device_prop (device, CL_DEVICE_DOUBLE_FP_CONFIG));
        t1.setfield ("double",                 octave_value (t2));

        t2.clear (); // .caps.images.
          t2.setfield ("supported",            get_device_prop (device, CL_DEVICE_IMAGE_SUPPORT));
          t2.setfield ("max_samplers",         get_device_prop (device, CL_DEVICE_MAX_SAMPLERS));
          t2.setfield ("max_read_args",        get_device_prop (device, CL_DEVICE_MAX_READ_IMAGE_ARGS));
          t2.setfield ("max_write_args",       get_device_prop (device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS));
          Matrix m2 (1, 2), m3 (1, 3);
            m2 (0) =                           get_device_prop (device, CL_DEVICE_IMAGE2D_MAX_WIDTH).double_value ();
            m2 (1) =                           get_device_prop (device, CL_DEVICE_IMAGE2D_MAX_HEIGHT).double_value ();
            m3 (0) =                           get_device_prop (device, CL_DEVICE_IMAGE3D_MAX_WIDTH).double_value ();
            m3 (1) =                           get_device_prop (device, CL_DEVICE_IMAGE3D_MAX_HEIGHT).double_value ();
            m3 (2) =                           get_device_prop (device, CL_DEVICE_IMAGE3D_MAX_DEPTH).double_value ();
          t2.setfield ("max_2d_dim",           octave_value (m2));
          t2.setfield ("max_3d_dim",           octave_value (m3));
        t1.setfield ("images",                 octave_value (t2));
        t1.setfield ("extensions",             extensions);
      device_props.setfield ("caps",           octave_value (t1));

      all_platform_devices_props (device_index) = octave_value (device_props);
    }

    all_devices_props (platform_index) = octave_value (all_platform_devices_props);
  }

  ocl_resources.setfield ("devices", octave_value (all_devices_props));

  if (num_all_devices == 0)
    ocl_error ("could not find any OpenCL devices -- please check your OpenCL installation");

  Cell summary (num_all_devices,1);
  Matrix prios (num_all_devices,1);
  int index=0;

  for (cl_uint platform_index=0; platform_index<num_platforms; platform_index++) {
    octave_scalar_map platform_props (all_platforms_props (platform_index).scalar_map_value ());
    num_devices_of_platform = devices[platform_index].size ();

    for (cl_uint device_index=0; device_index<num_devices_of_platform; device_index++) {
      octave_scalar_map device_props (all_devices_props (platform_index).cell_value ()(device_index).scalar_map_value ());

      int prio = index;

      uint64_t type = device_props.getfield ("type").ulong_value ();
      std::string type_str;
      if (type & CL_DEVICE_TYPE_GPU) {
        type_str = "GPU";
      } else if (type & CL_DEVICE_TYPE_ACCELERATOR) {
        type_str = "ACC";
        prio += 1 * num_all_devices;
      } else if (type & CL_DEVICE_TYPE_CPU) {
        type_str = "CPU";
        prio += 2 * num_all_devices;
      } else {
        type_str = "???";
        prio += 4 * num_all_devices;
      }

      double fp64 = device_props.getfield ("caps").scalar_map_value ()
                                .getfield ("double").scalar_map_value ()
                                .getfield ("supported").double_value ();
      if (fp64 != 1.0)
        prio += 8 * num_all_devices;

      double ver;
      std::string s;
      s = platform_props.getfield ("version").string_value ().substr (7,3);
      ver = atof (s.c_str ());
      s = device_props.getfield ("version").scalar_map_value ().getfield ("driver").string_value ().substr (0,3);
      ver = std::min (ver, atof (s.c_str ()));
      s = device_props.getfield ("version").scalar_map_value ().getfield ("device").string_value ().substr (7,3);
      ver = std::min (ver, atof (s.c_str ()));
      s = device_props.getfield ("version").scalar_map_value ().getfield ("opencl_c").string_value ().substr (9,3);
      ver = std::min (ver, atof (s.c_str ()));
      if (ver < 1.1)
        prio += 16 * num_all_devices;

      octave_scalar_map device_summary;
      device_summary.setfield ("type", octave_value (type_str));
      device_summary.setfield ("fp64", octave_value (fp64));
      device_summary.setfield ("version", octave_value (ver));
      device_summary.setfield ("platform_index", octave_value (platform_index));
      device_summary.setfield ("device_index", octave_value (device_index));
      device_summary.setfield ("name", octave_value (device_props.getfield ("name")));

      summary (index) = octave_value (device_summary);
      prios (index) = prio;
      index++;
    }
  }

  Array<octave_idx_type> indices;
  prios.sort (indices, 0, ASCENDING);
  summary = summary.index (idx_vector (indices));

  ocl_resources.setfield ("summary", octave_value (summary));
}


// ---------- context management functions


static cl_platform_id platform_id = 0;
static cl_device_id device_id = 0;
static bool active_opencl_context_is_fp64 = false;

static cl_context context = 0;
static cl_command_queue command_queue = 0;

static unsigned long active_opencl_context_id = 0;
static unsigned long next_opencl_context_id = 1;


unsigned long
assure_opencl_context (void)
{
  if (active_opencl_context_id)
    return active_opencl_context_id;

  select_device (); // will call get_resources () if needed, which will call assure_opencl_library ()

  long platform_index = device (0);
  long device_index   = device (1);

  if ((platform_index < 0) || (platform_index >= (long) platforms.size ()))
    ocl_error ("device_selection: invalid platform index");
  if ((device_index < 0) || (device_index >= (long) devices[platform_index].size ()))
    ocl_error ("device_selection: invalid device index");

  platform_id = platforms[platform_index];
  device_id = devices[platform_index][device_index];

  cl_context_properties context_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0 };

  context = clCreateContext (context_properties, 1, & device_id, 0, 0, & last_error);
  ocl_check_error ("clCreateContext");

  command_queue = clCreateCommandQueue (context, device_id, 0, & last_error);
  if (last_error != CL_SUCCESS) {
    clReleaseContext (context);
    platform_id = 0;
    device_id = 0;
    context = 0;
    command_queue = 0;
  }
  ocl_check_error ("clCreateCommandQueue");

  active_opencl_context_id = (next_opencl_context_id++);
  active_opencl_context_is_fp64 = device_fp64;

  return active_opencl_context_id;
}


void
destroy_opencl_context (void)
{
  if (opencl_library_loaded () && opencl_context_active ()) {
    last_error = clReleaseCommandQueue (command_queue);
    last_error = clReleaseContext (context);
    platform_id = 0;
    device_id = 0;
    context = 0;
    command_queue = 0;
    active_opencl_context_id = 0;
    active_opencl_context_is_fp64 = false;
    reset_memmgr ();
  }
}


cl_platform_id
get_platform_id (void)
{
  return platform_id;
}


cl_device_id
get_device_id (void)
{
  return device_id;
}


cl_context
get_context (void)
{
  return context;
}


cl_command_queue
get_command_queue (void)
{
  return command_queue;
}


unsigned long
opencl_context_id (void)
{
  return active_opencl_context_id;
}


bool
opencl_context_active (void)
{
  return active_opencl_context_id != 0;
}


bool
opencl_context_id_active (unsigned long id)
{
  return (active_opencl_context_id == id) && (id != 0);
}


void
assure_opencl_context_id (unsigned long id)
{
  if (id == 0)
    ocl_error ("ocl: internal error: null context requested");

  if (active_opencl_context_id == id)
    return;

  ocl_error ("OpenCL context no longer valid");
}


bool
opencl_context_is_fp64 (void)
{
  return active_opencl_context_is_fp64;
}


// ---------- the octave entry point to the 'ocl_context' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("ocl_context", "ocl_bin.oct");
// PKG_DEL: autoload ("ocl_context", "ocl_bin.oct", "remove");


DEFUN_DLD (ocl_context, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} ocl_context (@qcode{\"assure\"}) \n\
@deftypefnx {Loadable Function} ocl_context (@qcode{\"destroy\"}) \n\
@deftypefnx {Loadable Function} {[@var{active}, [@var{fp64}]] =} \
 ocl_context (@qcode{\"active\"}) \n\
@deftypefnx {Loadable Function} {[@var{activeid}, [@var{fp64}]] =} \
 ocl_context (@qcode{\"active_id\"}) \n\
@deftypefnx {Loadable Function} {@var{resources} =} \
 ocl_context (@qcode{\"get_resources\"}) \n\
@deftypefnx {Loadable Function} {[@var{selection}] =} \
 ocl_context (@qcode{\"device_selection\"}, [@var{str}]) \n\
\n\
Manage the OpenCL Context.  \n\
\n\
@code{ocl_context (\"assure\")} sets up the OpenCL context and makes it \n\
active and usable for operations with OpenCL memory objects and programs.  \n\
The single currently selected OpenCL device (see below) is determined and, \n\
if valid, is used for setting up the OpenCL context.  \n\
If the OpenCL context was already active, @code{ocl_context} has no effect.  \n\
If any step is unsuccessful, @code{ocl_context} aborts with an error.  \n\
\n\
@code{ocl_context (\"destroy\")} destroys the OpenCL context.  \n\
If no OpenCL context was active, @code{ocl_context} has no effect.  \n\
Destroying the OpenCL context has two distinct consequences:  First, the OpenCL \n\
memory and programs allocated within the context are immediately deleted and \n\
freed on the device.  \n\
Second, all OCL or octave objects which rely on these deleted OpenCL objects \n\
and which remain in octave \n\
memory are made inoperable and will produce an error when used afterwards.  \n\
\n\
@code{ocl_context (\"active\")} returns whether an OpenCL context is currently active.  \n\
A nonzero value @var{active} means that a context is currently active.  \n\
A nonzero value of the optional output variable @var{fp64} means that the active context \n\
is capable of computing with 64-bit floating-point (i.e., double precision).  \n\
\n\
@code{ocl_context (\"active_id\")} is similar to @code{ocl_context (\"active\")}, \n\
but returns the current context identifier @var{activeid} instead.  \n\
The context identifier is only nonzero when a context is active.  \n\
When using @code{ocl_context (\"destroy\")} in between, the context identifier value is \n\
distinct for each subsequent active OpenCL context \n\
(which means that each OpenCL memory object or program object is associated with a specific \n\
context identifier to be operable with).  \n\
\n\
@code{ocl_context (\"get_resources\")} returns comprehensive information on the available \n\
resources (hardware and software) which can potentially be used for OpenCL computations.  \n\
The return value @var{resources} is a hierarchical struct of which many leaf values have \n\
self-explanatory names; for detailled reference, see the OpenCL specification.  \n\
@var{resources} itself is assembled by @code{ocl_context} and contains the following fields:  \n\
\n\
@table @asis \n\
@item @code{.platforms} \n\
A struct array containing information on the available OpenCL platforms (i.e., vendors).  \n\
\n\
@item @code{.devices} \n\
A cell array containing, per platform, all OpenCL devices (i.e., hardware units with \n\
separate memory and processors), each with detailled information.  \n\
\n\
@item @code{.summary} \n\
A struct array containing a pre-ordered single list of all devices with \n\
only the most important information.  \n\
@end table \n\
\n\
@noindent \n\
This information, especially the @code{summary} field, should give \n\
the user enough guidance on which device \n\
to select for actual OCL computations (see below).  \n\
\n\
@code{ocl_context (\"device_selection\", ...)} can be used \n\
to query or set the device selection strategy, or to return the single device so selected.  \n\
To set the device selection strategy, @var{str} must be one of:  \n\
\n\
@table @asis \n\
@item @qcode{\"auto\"} \n\
The future selected device will be the first device from the (pre-ordered) resource summary list.  \n\
\n\
@item @qcode{\"GPU\"} \n\
The future selected device will be the first GPU device from the resource summary list.  \n\
\n\
@item @qcode{\"GPUn\"} \n\
The future selected device will be the (n+1)-th GPU device from the resource summary list, \n\
with n being a non-negative integer (i.e., @qcode{\"GPU0\"} is equivalent to @qcode{\"GPU\"}).  \n\
\n\
@item @qcode{\"ACC\"} \n\
The future selected device will be the first ACC device from the resource summary list.  \n\
\n\
@item @qcode{\"ACCn\"} \n\
The future selected device will be the (n+1)-th ACC device from the resource summary list.  \n\
\n\
@item @qcode{\"CPU\"} \n\
The future selected device will be the first CPU device from the resource summary list.  \n\
\n\
@item @qcode{\"CPUn\"} \n\
The future selected device will be the (n+1)-th CPU device from the resource summary list.  \n\
\n\
@item @qcode{\"devn\"} \n\
The future selected device will be the (n+1)-th device from the resource summary list.  \n\
@end table \n\
\n\
@noindent \n\
Without @var{str}, or when an output parameter @var{selection} is requested, \n\
the current or prior setting of the device selection strategy \n\
is returned as one of the above strings.  \n\
These calls have no immediate effect on the OpenCL library or context.  \n\
No checking of availability against present resources is performed \n\
(only syntax checking of @var{str}).  \n\
\n\
In contrast, @code{ocl_context (\"device_selection\", \"selected\")} applies the \n\
current device selection strategy onto the actually available resources and \n\
selects a single OpenCL device from the summary accordingly.  \n\
If the strategy fails to find a corresponding device, @code{ocl_context} aborts \n\
with an error at this point.  \n\
Otherwise, @var{selection} returns a 2x1 array containing the platform and \n\
device index (starting from zero; as counted in the @var{resources} fields).  \n\
\n\
The first four subfunctions of @code{ocl_context} only need to be called explicitly \n\
in rare situations, since many other (\"higher\") OCL functions call them internally.  \n\
These subfunctions are provided mainly for testing.  \n\
@code{ocl_context (\"get_resources\")} is of regular interest to the user, and \n\
@code{ocl_context (\"device_selection\", ...)} to choose the device selection strategy \n\
is likey to be called once or more per octave session \n\
(maybe even in your .octaverc file).  \n\
\n\
Note that @code{ocl_context (\"assure\")}, @code{ocl_context (\"get_resources\")}, and \n\
@code{ocl_context (\"device_selection\", \"selected\")} automatically load the OpenCL library.  \n\
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

  } else if (fcn == "get_resources") {

    if (nargin > 1)
      ocl_error ("get_resources: too many arguments");

    if (ocl_resources.nfields () == 0)
      get_resources ();

    retval (0) = octave_value (ocl_resources);

  } else if (fcn == "device_selection") {

    if (nargin > 2)
      ocl_error ("device_selection: too many arguments");

    if (nargin == 1)
      retval (0) = octave_value (selection);
    else if (!args (1).is_string ())
      ocl_error ("device_selection: second argument must be a string, if given");
    else {
      std::string arg = args (1).string_value (), arg3 = arg.substr (0, 3);

      if ((arg == "auto") || (arg3 == "GPU") || (arg3 == "ACC") || (arg3 == "CPU") || (arg3 == "dev")) {

        if ((!(arg == selection)) && opencl_context_active ())
          ocl_error ("device_selection: changing the device selection is not permitted while using an active OpenCL context");

        if (!(arg == "auto") && (arg.length () > 3)) {
          for (unsigned int i=3; i<arg.length (); i++) {
            char c = arg[i];
            if ((c < '0') || (c > '9'))
              ocl_error ("device_selection: invalid device specifier");
          }
        }

        if (nargout > 0)
          retval (0) = octave_value (selection);

        if (!(arg == selection)) {
          device (0) = -1;
          device (1) = -1;
        }

        selection = arg;

      } else if (arg == "selected") {

        select_device ();

        retval (0) = octave_value (device);

      } else {
        ocl_error ("device_selection: invalid argument");
      }
    }

  } else if (fcn == "assure") {

    if (nargin > 1)
      ocl_error ("assure: too many arguments");

    assure_opencl_context ();

  } else if (fcn == "destroy") {

    if (nargin > 1)
      ocl_error ("destroy: too many arguments");

    destroy_opencl_context ();

  } else if (fcn == "active") {

    if (nargin > 1)
      ocl_error ("active: too many arguments");

    retval (0) = octave_value (double (opencl_context_active ()));

    if (nargout > 1)
      retval (1) = octave_value (double (opencl_context_is_fp64 ()));

  } else if (fcn == "active_id") {

    if (nargin > 1)
      ocl_error ("active_id: too many arguments");

    retval (0) = octave_value (uint64_t (opencl_context_id ()));

    if (nargout > 1)
      retval (1) = octave_value (double (opencl_context_is_fp64 ()));

  } else {

    ocl_error ("subfunction not recognized");

  }

  return retval;
}
