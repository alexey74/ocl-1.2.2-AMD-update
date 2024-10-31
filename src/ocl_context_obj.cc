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

#include "ocl_context_obj.h"
#include "ocl_lib.h"


OclContextObject::OclContextObject (bool need_context)
{
  if (need_context) {
    assctd_ctx_id = assure_opencl_context ();
  } else {
    assctd_ctx_id = 0;
  }
}


OclContextObject::~OclContextObject () // must define pure virtual destructor
{}


/*
void
OclContextObject::assure_object_context (void) const
{
  if (assctd_ctx_id == 0)
    ocl_error ("ocl: cannot use inoperable OCL object"); // too unspecific

  assure_opencl_context_id (assctd_ctx_id);
}
*/


bool
OclContextObject::object_context_still_valid (void) const
{
  return opencl_context_id_active (assctd_ctx_id);
}

