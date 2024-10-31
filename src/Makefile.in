##
##  Copyright (C) 2019-2023 Matthias W. Klein
##
##  This file is part of OCL - a GNU Octave package providing OpenCL support.
##
##  OCL is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  OCL is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with OCL.  If not, see <http://www.gnu.org/licenses/>.
##


# variables

TARGET = ocl_bin.oct

OBJ_FILES = \
  ocl_constant.o \
  ocl_lib.o \
  ocl_context.o \
  ocl_context_obj.o \
  ocl_program.o \
  ocl_memobj.o \
  ocl_array.o \
  ocl_array_prog.o \
  ocl_ov_matrix.o \
  ocl_ov_matrix_ops.o \
  ocl_ov_matrix_fcns.o \
  ocl_ov_program.o \
  ocl_ov_types.o

C_FILES = $(OBJ_FILES:.o=.cc)

CC = $(MKOCTFILE)
LD = $(MKOCTFILE)

# main target

$(TARGET): $(OBJ_FILES)
	$(LD) $(OBJ_FILES) -o $(TARGET)

.SUFFIXES=
.SUFFIXES= .cc .o

.cc.o:
	$(CC) -c $<

-include depends.mk

# helper targets for local development

GREP ?= grep
CUT ?= cut
SED ?= sed

# remove all generated files
clean:
	@rm -f *~ \#*
	@rm -f *.o
	@rm -f $(TARGET)
	@rm -f depends.mk

# generate dependencies file
deps:
	@gcc -MM $(C_FILES) > depends.mk

# (re)generate oclmat method files
oclmat_methods = as_index sum sumsq prod mean meansq cumsum cumprod findfirst findlast std max min cummax cummin atan2 ndgrid meshgrid repmat complex
oclmat_classes = double single int8 int16 int32 int64 uint8 uint16 uint32 uint64

outfiles = $(addsuffix .m,$(oclmat_methods))
outdirfiles = $(foreach dirname,$(oclmat_classes),$(addprefix ../inst/@ocl_$(dirname)/,$(outfiles)))

methods: $(outdirfiles)

$(outdirfiles): ocl_mat_method_template.m.in
	@mkdir -p "$(dir $@)"
	@sed 's/__METHOD__/$(basename $(notdir $@))/' ocl_mat_method_template.m.in > $@

.PHONY = clean deps methods
