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

#ifndef __OCL_CTX_OBJ_H
#define __OCL_CTX_OBJ_H

class
OclContextObject
{
public:

  OclContextObject (bool need_context = false);

  virtual ~OclContextObject () = 0; // no instances

//  virtual void assure_object_context (void) const;

  virtual bool object_context_still_valid (void) const;

  virtual unsigned long get_context_id (void) const { return assctd_ctx_id; }

private:

  unsigned long assctd_ctx_id;
};

#endif  /* __OCL_CTX_OBJ_H */
