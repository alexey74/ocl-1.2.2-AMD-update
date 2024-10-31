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

#ifndef __OCL_ARRAY_PROG_H
#define __OCL_ARRAY_PROG_H

#include <string>

namespace OclArrayKernels
{
  enum Kernel {
    fill,
    fill0,
    eye,
    linspace,
    logspace,
    ndgrid1,
    repmat1,
    cat,
    transpose,
    hermitian,
    as_index,
    index,
    assign_el,
    assign,
    assign0,
    assign_el_logind,
    findfirst,
    findlast,
    all,
    any,
    sum,
    sumsq,
    prod,
    cumsum,
    cumprod,
    mean,
    meansq,
    std,
    max,
    max2,
    max1,
    min,
    min2,
    min1,
    cummax,
    cummin,
    compare,
    logic,
    fmad1,
    fmad2,
    uminus,
    add1,
    add2,
    sub1m,
    sub1s,
    sub2,
    mul1,
    mul2,
    mtimes,
    div1n,
    div1d,
    div2,
    abs,
    fabs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cbrt,
    ceil,
    cos,
    cosh,
    erf,
    erfc,
    exp,
    expm1,
    fix,
    floor,
    isfinite,
    isinf,
    isnan,
    lgamma,
    log,
    log2,
    log10,
    log1p,
    round,
    sign,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    tgamma,
    power1e,
    power1b,
    power2,
    atan2,
    real2complex_r,
    real2complex_i,
    real2complex_ri,
    real,
    imag,
    arg,
    conj,
    max_array_prog_kernels // must be the last entry
  };

}

extern const std::string get_array_prog_kernel_name (OclArrayKernels::Kernel kernel);

extern const std::string ocl_array_prog_source;

#endif  /* __OCL_ARRAY_PROG_H */
