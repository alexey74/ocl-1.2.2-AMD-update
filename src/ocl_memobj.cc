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

#include "ocl_memobj.h"
#include "ocl_lib.h"
#include <list>
#include <map>
#include <octave/oct.h>


// ---------- static variables


// a pool of retained OpenCL memory buffer handles,
// only for the currently active context
// (as opposed to OclMemoryObject objects, which may remain
// in octave memory but become inoperable)

typedef std::list<cl_mem> OclMemobjSizedPool_t;
typedef std::map<size_t, OclMemobjSizedPool_t> OclMemobjPool_t;

static OclMemobjPool_t memobj_pool;


static size_t max_sized_memobj_pool_objs = 3;


// a list of assigned (and not retained) memory objects, in order to remember their sizes

static std::map<cl_mem, size_t> assigned_ocl_memobjs;


// ---------- static functions


static
cl_mem
new_ocl_buffer (size_t size)
{
  // really allocate new OpenCL buffer
  cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
  cl_mem mem_obj = clCreateBuffer (get_context (), mem_flags, size, 0, & last_error);
  return mem_obj;
  // checking for errors deferred to calling function (obtain_ocl_buffer)
}


static
void
delete_ocl_buffer (cl_mem mem_obj)
{
  // really deallocate OpenCL buffer
  // never check for errors when deleting objects now
  clReleaseMemObject (mem_obj);
}


static
cl_mem
obtain_ocl_buffer (size_t size)
{
  // obtain a retained buffer or allocate new a buffer

  // TODO: possibly also return a retained buffer which is slightly larger than requested; conditions?

  cl_mem mem_obj;
  if (memobj_pool.count (size) == 0) { // allocate new buffer (since pool of buffers of this size is empty)
    while (1) {
      mem_obj = new_ocl_buffer (size);
      if ((last_error == CL_SUCCESS) || (memobj_pool.empty ()))
        break;

      // allocation of new buffer failed, release a large retained one and retry
      size_t sizemax = memobj_pool.rbegin ()->first;
      OclMemobjSizedPool_t sp = memobj_pool[sizemax];
      mem_obj = sp.back ();
      sp.pop_back ();
      if (sp.empty ())
        memobj_pool.erase (sizemax);
      else
        memobj_pool[sizemax] = sp;
      delete_ocl_buffer (mem_obj);
    }

    ocl_check_error ("clCreateBuffer");
    // successfully allocated the buffer
  } else { // a non-empty pool entry exists, reuse a retained memory object
    OclMemobjSizedPool_t sp = memobj_pool[size];
    mem_obj = sp.back ();
    sp.pop_back ();
    if (sp.empty ())
      memobj_pool.erase (size);
    else
      memobj_pool[size] = sp;
  }
  assigned_ocl_memobjs[mem_obj] = size;
  return mem_obj;
}


static
void
release_ocl_buffer (cl_mem mem_obj)
{
  // buffer no longer used: retain in pool or release permanently

  if (assigned_ocl_memobjs.count (mem_obj) == 0) { // for safety: if buffer is unknown, simply release
    delete_ocl_buffer (mem_obj);
    return;
  }
  size_t size  = assigned_ocl_memobjs[mem_obj];
  assigned_ocl_memobjs.erase (mem_obj);

  if (assigned_ocl_memobjs.empty ()) {
    // deletion of last assigned memory object: delete all retained buffers
    // i.e., the octave command "clear" also empties the pool of retained buffers
    // (OCL matrix objects should never be assigned to persistent variables)
    delete_ocl_buffer (mem_obj);
    for (OclMemobjPool_t::iterator it1 = memobj_pool.begin (); it1 != memobj_pool.end (); it1++)
      for (OclMemobjSizedPool_t::iterator it2 = it1->second.begin (); it2 != it1->second.end (); it2++)
        delete_ocl_buffer (*it2);
    memobj_pool.clear ();
    return;
  }

  size_t count;
  if (memobj_pool.count (size) == 0)
    count = 0;
  else
    count = memobj_pool[size].size ();

  if (count >= max_sized_memobj_pool_objs) { // already many buffers retained
    delete_ocl_buffer (mem_obj);
  } else { // retain buffer
    memobj_pool[size].push_back (mem_obj);
  }
}


// ---------- public functions


void reset_memmgr (void)
{
  // to be called by "destroy_opencl_context"
  // no OpenCL library calls needed, so no querying of "opencl_library_loaded" needed
  assigned_ocl_memobjs.clear ();
  memobj_pool.clear ();
}


// ---------- OclMemoryObject members


OclMemoryObject::OclMemoryObject (size_t size)
  : OclContextObject (true)
{
  if (size <= 0)
    ocl_error ("OclArray: requesting empty buffer");

  // we know: size > 0, and OpenCL context is active
  ocl_mem_buffer = (void *) obtain_ocl_buffer (size);
}


OclMemoryObject::~OclMemoryObject ()
{
  if (object_context_still_valid ())
    release_ocl_buffer ((cl_mem) ocl_mem_buffer);
}


// ---------- the octave entry point to the '__ocl_memmgr__' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_memmgr__", "ocl_bin.oct");
// PKG_DEL: autoload ("__ocl_memmgr__", "ocl_bin.oct", "remove");


DEFUN_DLD (__ocl_memmgr__, args, nargout,
"OCL internal function")
{
  octave_value_list retval;
  int nargin = args.length ();

  std::string fcn;
  if ((nargin > 0) && (args (0).is_string ()))
    fcn = args (0).char_matrix_value ().row_as_string (0);

  if (fcn == "maxobjs") {

    // handle maximum number of buffer objects per size

    if (nargout > 0)
      retval = octave_value (max_sized_memobj_pool_objs);
    if (nargin > 1)
      max_sized_memobj_pool_objs = args (1).int_value ();

  } else if (fcn == "numobjs") {

    // list number of retained memory objects per size

    Matrix m(memobj_pool.size (), 2);
    octave_idx_type i = 0;
    for (OclMemobjPool_t::iterator it1 = memobj_pool.begin (); it1 != memobj_pool.end (); it1++) {
      m(i  ,0) = it1->first;
      m(i++,1) = it1->second.size ();
    }
    retval = octave_value (m);

  } else {

    ocl_error ("unknown subfunction");

  }

  return retval;
}
