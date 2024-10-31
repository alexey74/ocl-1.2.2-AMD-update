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
#include "ocl_ov_matrix_ops.h"
#include "ocl_ov_program.h"

#if defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions >= 4.4.0
#include <octave/interpreter.h>
#endif


// ---------- public functions


void
assure_installed_ocl_types (void)
{
  static bool ocl_types_loaded = false;

  if (ocl_types_loaded)
    return;

#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0
  mlock ();
#else // for octave versions >= 4.4.0
  octave::interpreter::the_interpreter () -> mlock ();
#endif
  ocl_types_loaded = true;

  install_ocl_matrix_types ();
  install_ocl_program_type ();
}


// ---------- the octave entry point to the '__ocl_install_ocl_types__' function


// The following two comment lines are needed verbatim for the Octave package manager:
// PKG_ADD: autoload ("__ocl_install_ocl_types__", "ocl_bin.oct"); __ocl_install_ocl_types__ (); ## with install
// PKG_DEL: autoload ("__ocl_install_ocl_types__", "ocl_bin.oct", "remove");


DEFUN_DLD (__ocl_install_ocl_types__, args, nargout,
"OCL internal function")
{
  octave_value_list retval;

  assure_installed_ocl_types ();

  return retval;
}
