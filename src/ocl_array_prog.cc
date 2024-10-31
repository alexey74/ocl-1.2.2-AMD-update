/*
 * Copyright (C) 2019-2023 Matthias W. Klein
 * 2021 Add fix to support AMD devices by Prof. Jinchuan Tang (jctang@gzu.edu.cn) with
 * the kind (VIP speed) support from AMD's Dipak.
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

#include "ocl_array_prog.h"
#include "ocl_lib.h"


const std::string
ocl_array_prog_source = "\
\
#define IDX_T long                                           \n\
                                                             \n\
                                                             \n\
#if ! defined (COMPLEX) // non-COMPLEX                       \n\
                                                             \n\
#define ZERO (TYPE) (0)                                      \n\
#define ONE (TYPE) (1)                                       \n\
#define IS_NONZERO(z) (z != ZERO)                            \n\
#define IS_NE(a,b) (a != b)                                  \n\
#define IS_EQ(a,b) (a == b)                                  \n\
#define NORM(z)  (z*z)                                       \n\
#define MUL(a,b) (a*b)                                       \n\
#define DIV(a,b) (a/b)                                       \n\
                                                             \n\
#define DEFCMP(NAME, OP) \\                                  \n\
  int NAME (TYPE a, TYPE b) { \\                             \n\
    return (a OP b); \\                                      \n\
  }                                                          \n\
                                                             \n\
#else // COMPLEX (i.e., float2 or double2)                   \n\
                                                             \n\
#define ZERO (TYPE) (0)                                      \n\
#define ONE (TYPE) (1,0)                                     \n\
#define ZERO1 ((TYPE1) (0))                                  \n\
#define ONE1 ((TYPE1) (1))                                   \n\
#define IS_NONZERO(z) (any (z != ZERO))                      \n\
#define IS_NE(a,b) (any (a != b))                            \n\
#define IS_EQ(a,b) (all (a == b))                            \n\
#define NORM(z)   ((TYPE) (z.x*z.x + z.y*z.y, 0))            \n\
#define MUL(a,b)  ((TYPE) (a.x*b.x-a.y*b.y, a.y*b.x+a.x*b.y)) \n\
#define DIV(a,b) ((IS_NE (b, ZERO)) ? ((TYPE) (a.x*b.x+a.y*b.y, a.y*b.x-a.x*b.y)) / (b.x*b.x+b.y*b.y) : a / ZERO) \n\
                                                             \n\
#define DEFCMP(NAME, OP) \\                                  \n\
  int NAME (TYPE a, TYPE b) { \\                             \n\
    TYPE n = (TYPE) (a.x*a.x + a.y*a.y, b.x*b.x + b.y*b.y); \\ \n\
    if (n.x == n.y) { \\                                     \n\
      return (atan2 (a.y, a.x) OP atan2 (b.y, b.x)); \\      \n\
    } else { \\                                              \n\
      return (n.x OP n.y); \\                                \n\
    } \\                                                     \n\
  }                                                          \n\
                                                             \n\
#endif                                                       \n\
                                                             \n\
DEFCMP (IS_LT, <)                                            \n\
DEFCMP (IS_GT, >)                                            \n\
DEFCMP (IS_LE, <=)                                           \n\
DEFCMP (IS_GE, >=)                                           \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_fill                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const TYPE value)                                         \n\
{                                                            \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = value;                                      \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_fill0                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = data_src [0];                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_eye                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const ulong n_repeat,                                     \n\
   const ulong n_max)                                        \n\
{                                                            \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] =                                             \n\
    ((i % n_repeat) == 0) && (i < n_max) ? ONE : ZERO;       \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_linspace                                                 \n\
  (__global TYPE *data_dst,                                  \n\
   const TYPE start_val,                                     \n\
   const TYPE end_val,                                       \n\
   const ulong n)                                            \n\
{                                                            \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = start_val + ((end_val-start_val)*i)/(n-1);  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
#if defined (FLOATINGPOINT) && ! defined (COMPLEX)           \n\
__kernel void                                                \n\
ocl_logspace                                                 \n\
  (__global TYPE *data_dst,                                  \n\
   const TYPE start_val,                                     \n\
   const TYPE end_val,                                       \n\
   const ulong n)                                            \n\
{                                                            \n\
  size_t i = get_global_id (0);                              \n\
  TYPE exponent = start_val + ((end_val-start_val)*i)/(n-1); \n\
  data_dst [i] = exp (log ((TYPE) 10.0)*exponent);           \n\
}                                                            \n\
#endif                                                       \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_ndgrid1                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong div1, const ulong div2)                       \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j;                           \n\
  j = (i/div1) % div2;                                       \n\
  data_dst [i] = data_src [j];                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_repmat1                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong fac1, const ulong fac2, const ulong fac3)     \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j;                           \n\
  j = (i % fac1)                                             \n\
    + ((i / fac1) % fac2) * fac1                             \n\
    + (i / fac1 / fac3) * fac1 * fac2;                       \n\
  data_dst [i] = data_src [j];                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_cat                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong offs, const ulong fac1, const ulong fac2)     \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j;                           \n\
  j = offs + (i % fac1) + (i / fac1) * fac2;                 \n\
  data_dst [j] = data_src [i];                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_transpose                                                \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong s1,                                           \n\
   const ulong s2)                                           \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j;                           \n\
  j = (i / s2) + (i % s2) * s1;                              \n\
  data_dst [i] = data_src [j];                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_hermitian                                                \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong s1,                                           \n\
   const ulong s2)                                           \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j;                           \n\
  j = (i / s2) + (i % s2) * s1;                              \n\
  TYPE z;                                                    \n\
  z = data_src [j];                                          \n\
#if defined (COMPLEX)                                        \n\
  data_dst [i] = (TYPE) (z.x, -z.y);                         \n\
#else                                                        \n\
  data_dst [i] = z;                                          \n\
#endif                                                       \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_as_index                                                 \n\
  (__global IDX_T *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
#if defined (COMPLEX)                                        \n\
  data_dst [i] = (IDX_T) round (data_src [i].x);             \n\
#elif defined (FLOATINGPOINT)                                \n\
  data_dst [i] = (IDX_T) round (data_src [i]);               \n\
#else                                                        \n\
  data_dst [i] = (IDX_T) (data_src [i]);                     \n\
#endif                                                       \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_index                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len_src,                                      \n\
   const __global IDX_T *data_idx,                           \n\
   const ulong ofs_idx)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  data_idx += ofs_idx;                                       \n\
  size_t i = get_global_id (0);                              \n\
  IDX_T j = data_idx [i];                                    \n\
  TYPE val;                                                  \n\
  if ((j >= 0) && (j < len_src))                             \n\
    val = data_src [j];                                      \n\
  else                                                       \n\
    val = ZERO;                                              \n\
  data_dst [i] = val;                                        \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_assign_el                                                \n\
  (__global TYPE *data_dst,                                  \n\
   const ulong ofs_dst,                                      \n\
   const ulong len_dst,                                      \n\
   const __global IDX_T *data_idx,                           \n\
   const ulong ofs_idx,                                      \n\
   const TYPE value)                                         \n\
{                                                            \n\
  data_dst += ofs_dst;                                       \n\
  data_idx += ofs_idx;                                       \n\
  size_t i = get_global_id (0);                              \n\
  IDX_T j = data_idx [i];                                    \n\
  if ((j >= 0) && (j < len_dst))                             \n\
    data_dst [j] = value;                                    \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_assign0                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const ulong ofs_dst,                                      \n\
   const ulong len_dst,                                      \n\
   const __global IDX_T *data_idx,                           \n\
   const ulong ofs_idx,                                      \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_dst += ofs_dst;                                       \n\
  data_idx += ofs_idx;                                       \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  IDX_T j = data_idx [i];                                    \n\
  if ((j >= 0) && (j < len_dst))                             \n\
    data_dst [j] = data_src [0];                             \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_assign                                                   \n\
  (__global TYPE *data_dst,                                  \n\
   const ulong ofs_dst,                                      \n\
   const ulong len_dst,                                      \n\
   const __global IDX_T *data_idx,                           \n\
   const ulong ofs_idx,                                      \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_dst += ofs_dst;                                       \n\
  data_idx += ofs_idx;                                       \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  IDX_T j = data_idx [i];                                    \n\
  if ((j >= 0) && (j < len_dst))                             \n\
    data_dst [j] = data_src [i];                             \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_assign_el_logind                                         \n\
  (__global TYPE *data_dst,                                  \n\
   const ulong ofs_dst,                                      \n\
   const __global TYPE *data_log,                            \n\
   const ulong ofs_log,                                      \n\
   const TYPE value)                                         \n\
{                                                            \n\
  data_dst += ofs_dst;                                       \n\
  data_log += ofs_log;                                       \n\
  size_t i = get_global_id (0);                              \n\
  if (IS_NONZERO (data_log [i]))                             \n\
    data_dst [i] = value;                                    \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_findfirst                                                \n\
  (__global IDX_T *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    if (IS_NONZERO (data_src [j])) {                         \n\
      data_dst [i] = (IDX_T) (k);                            \n\
      return;                                                \n\
    }                                                        \n\
  }                                                          \n\
  data_dst [i] = -1;                                         \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_findlast                                                 \n\
  (__global IDX_T *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  for (k=len-1; k<len; k--) { // k is unsigned!              \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    if (IS_NONZERO (data_src [j])) {                         \n\
      data_dst [i] = (IDX_T) (k);                            \n\
      return;                                                \n\
    }                                                        \n\
  }                                                          \n\
  data_dst [i] = -1;                                         \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_all                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    if (! IS_NONZERO (data_src [j])) {                       \n\
      data_dst [i] = ZERO;                                   \n\
      return;                                                \n\
    }                                                        \n\
  }                                                          \n\
  data_dst [i] = ONE;                                        \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_any                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    if (IS_NONZERO (data_src [j])) {                         \n\
      data_dst [i] = ONE;                                    \n\
      return;                                                \n\
    }                                                        \n\
  }                                                          \n\
  data_dst [i] = ZERO;                                       \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sum                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE val = ZERO;                                           \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    val += data_src [j];                                     \n\
  }                                                          \n\
  data_dst [i] = val;                                        \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sumsq                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE val = ZERO;                                           \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    val += NORM (data_src [j]);                              \n\
  }                                                          \n\
  data_dst [i] = val;                                        \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_prod                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE val = ONE;                                            \n\
  TYPE v;                                                    \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    v = data_src [j];                                        \n\
    val = MUL (val, v);                                      \n\
  }                                                          \n\
  data_dst [i] = val;                                        \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_cumsum                                                   \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE val = ZERO;                                           \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    val += data_src [j];                                     \n\
    data_dst [j] = val;                                      \n\
  }                                                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_cumprod                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE val = ONE;                                            \n\
  TYPE v;                                                    \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    v = data_src [j];                                        \n\
    val = MUL (val, v);                                      \n\
    data_dst [j] = val;                                      \n\
  }                                                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_mean                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE val = ZERO;                                           \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    val += data_src [j];                                     \n\
  }                                                          \n\
  data_dst [i] = val/len;                                    \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_meansq                                                   \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE val = ZERO;                                           \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    val += NORM (data_src [j]);                              \n\
  }                                                          \n\
  data_dst [i] = val/len;                                    \n\
}                                                            \n\
                                                             \n\
                                                             \n\
#if defined (FLOATINGPOINT) || defined (COMPLEX)             \n\
__kernel void                                                \n\
ocl_std                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac,                                          \n\
   const ulong n)                                            \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k;                        \n\
  TYPE m1 = ZERO;                                            \n\
  TYPE m2 = ZERO;                                            \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    m1 += data_src [j];                                      \n\
    m2 += NORM (data_src [j]);                               \n\
  }                                                          \n\
  data_dst [i] = sqrt (max ((m2-NORM(m1)/len)/n, ZERO));     \n\
}                                                            \n\
#endif                                                       \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_max                                                      \n\
  (__global TYPE *data_dst1,                                 \n\
   __global IDX_T *data_dst2,                                \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k, km;                    \n\
  TYPE val, v;                                               \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    v = data_src [j];                                        \n\
    if ((k == 0) || (IS_GT (v, val))) {                      \n\
      val = v; km = k;                                       \n\
    }                                                        \n\
  }                                                          \n\
  data_dst1 [i] = val;                                       \n\
  if (data_dst2 != (__global IDX_T *)data_dst1)              \n\
    data_dst2 [i] = (IDX_T) (km);                            \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_max2                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v1, v2;                                               \n\
  v1 = data_src1 [i];                                        \n\
  v2 = data_src2 [i];                                        \n\
  data_dst [i] = IS_GT (v1, v2) ? v1 : v2;                   \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_max1                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE v2)                                            \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v1;                                                   \n\
  v1 = data_src [i];                                         \n\
  data_dst [i] = IS_GT (v1, v2) ? v1 : v2;                   \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_min                                                      \n\
  (__global TYPE *data_dst1,                                 \n\
   __global IDX_T *data_dst2,                                \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k, km;                    \n\
  TYPE val, v;                                               \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    v = data_src [j];                                        \n\
    if ((k == 0) || (IS_LT (v, val))) {                      \n\
      val = v; km = k;                                       \n\
    }                                                        \n\
  }                                                          \n\
  data_dst1 [i] = val;                                       \n\
  if (data_dst2 != (__global IDX_T *)data_dst1)              \n\
    data_dst2 [i] = (IDX_T) (km);                            \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_min2                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v1, v2;                                               \n\
  v1 = data_src1 [i];                                        \n\
  v2 = data_src2 [i];                                        \n\
  data_dst [i] = IS_LT (v1, v2) ? v1 : v2;                   \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_min1                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE v2)                                            \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v1;                                                   \n\
  v1 = data_src [i];                                         \n\
  data_dst [i] = IS_LT (v1, v2) ? v1 : v2;                   \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_cummax                                                   \n\
  (__global TYPE *data_dst1,                                 \n\
   __global IDX_T *data_dst2,                                \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k, km;                    \n\
  TYPE val, v;                                               \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    v = data_src [j];                                        \n\
    if ((k == 0) || (IS_GT (v, val))) {                      \n\
      val = v; km = k;                                       \n\
    }                                                        \n\
    data_dst1 [j] = val;                                     \n\
    if (data_dst2 != (__global IDX_T *)data_dst1)            \n\
      data_dst2 [j] = (IDX_T) (km);                          \n\
  }                                                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_cummin                                                   \n\
  (__global TYPE *data_dst1,                                 \n\
   __global IDX_T *data_dst2,                                \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const ulong len,                                          \n\
   const ulong fac)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0), j, k, km;                    \n\
  TYPE val, v;                                               \n\
  for (k=0; k<len; k++) {                                    \n\
    j = (i % fac) + k * fac + (i / fac) * fac * len;         \n\
    v = data_src [j];                                        \n\
    if ((k == 0) || (IS_LT (v, val))) {                      \n\
      val = v; km = k;                                       \n\
    }                                                        \n\
    data_dst1 [j] = val;                                     \n\
    if (data_dst2 != (__global IDX_T *)data_dst1)            \n\
      data_dst2 [j] = (IDX_T) (km);                          \n\
  }                                                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_compare                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2,                                     \n\
   const TYPE c,                                             \n\
   const ulong fcn)                                          \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  TYPE o1, o2;                                               \n\
  switch (fcn & 0xF) {                                       \n\
    case 0: o1 = data_src1 [i]; o2 = c; break;               \n\
    case 1: o1 = c; o2 = data_src1 [i]; break;               \n\
    case 2: o1 = data_src1 [i]; o2 = data_src2 [i]; break;   \n\
  }                                                          \n\
  int res;                                                   \n\
  switch (fcn >> 4) {                                        \n\
    case 0: res = IS_LT (o1, o2); break;                     \n\
    case 1: res = IS_LE (o1, o2); break;                     \n\
    case 2: res = IS_GT (o1, o2); break;                     \n\
    case 3: res = IS_GE (o1, o2); break;                     \n\
    case 4: res = IS_EQ (o1, o2); break;                     \n\
    case 5: res = IS_NE (o1, o2); break;                     \n\
  }                                                          \n\
  data_dst [i] = res ? ONE : ZERO;                           \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_logic                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2,                                     \n\
   const TYPE c,                                             \n\
   const ulong fcn)                                          \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  int o1, o2;                                                \n\
  switch (fcn & 0xF) {                                       \n\
    case 0:                                                  \n\
      o1 = IS_NONZERO (data_src1 [i]);                       \n\
      o2 = IS_NONZERO (c);                                   \n\
      break;                                                 \n\
    case 1:                                                  \n\
      o1 = IS_NONZERO (c);                                   \n\
      o2 = IS_NONZERO (data_src1 [i]);                       \n\
      break;                                                 \n\
    case 2:                                                  \n\
      o1 = IS_NONZERO (data_src1 [i]);                       \n\
      o2 = IS_NONZERO (data_src2 [i]);                       \n\
      break;                                                 \n\
  }                                                          \n\
  int res;                                                   \n\
  switch (fcn >> 4) {                                        \n\
    case 0: res = (o1 && o2); break;                         \n\
    case 1: res = (o1 || o2); break;                         \n\
    case 2: res = (!o1); break;                              \n\
  }                                                          \n\
  data_dst [i] = res ? ONE : ZERO;                           \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_fmad1                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const TYPE fac,                                           \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE add)                                           \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE val;                                                  \n\
  val = data_src [i];                                        \n\
  data_dst [i] = MUL (fac, val) + add;                       \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_fmad2                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const TYPE fac,                                           \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  TYPE val;                                                  \n\
  val = data_src1 [i];                                       \n\
  data_dst [i] = MUL (fac, val) + data_src2 [i];             \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_uminus                                                   \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = -data_src [i];                              \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_add1                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE summand)                                       \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = data_src [i] + summand;                     \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_add2                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = data_src1 [i] + data_src2 [i];              \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sub1m                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE minuend)                                       \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = minuend - data_src [i];                     \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sub1s                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE subtrahend)                                    \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = data_src [i] - subtrahend;                  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sub2                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = data_src1 [i] - data_src2 [i];              \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_mul1                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE factor)                                        \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE val;                                                  \n\
  val = data_src [i];                                        \n\
  data_dst [i] = MUL (val, factor);                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_mul2                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v1, v2;                                               \n\
  v1 = data_src1 [i];                                        \n\
  v2 = data_src2 [i];                                        \n\
  data_dst [i] = MUL (v1, v2);                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_mtimes                                                   \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2,                                     \n\
   const ulong s1,                                           \n\
   const ulong len)                                          \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0), j1, j2, k;                   \n\
  TYPE val = ZERO;                                           \n\
  TYPE v1, v2;                                               \n\
  for (k=0; k<len; k++) {                                    \n\
    j1 = (i % s1) +  k       * s1;                           \n\
    j2 =  k       + (i / s1) * len;                          \n\
    v1 = data_src1 [j1];                                     \n\
    v2 = data_src2 [j2];                                     \n\
    val += MUL (v1, v2);                                     \n\
  }                                                          \n\
  data_dst [i] = val;                                        \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_div1n                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE numerator)                                     \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE val;                                                  \n\
  val = data_src [i];                                        \n\
  data_dst [i] = DIV (numerator, val);                       \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_div1d                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE denominator)                                   \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE val;                                                  \n\
  val = data_src [i];                                        \n\
  data_dst [i] = DIV (val, denominator);                     \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_div2                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v1, v2;                                               \n\
  v1 = data_src1 [i];                                        \n\
  v2 = data_src2 [i];                                        \n\
  data_dst [i] = DIV (v1, v2);                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
#define MATH_FUNC(fcn) \\                                    \n\
__kernel void \\                                             \n\
ocl_##fcn \\                                                 \n\
  (__global TYPE *data_dst, \\                               \n\
   const __global TYPE *data_src, \\                         \n\
   const ulong ofs_src) \\                                   \n\
{ \\                                                         \n\
  data_src += ofs_src; \\                                    \n\
  size_t i = get_global_id (0); \\                           \n\
  data_dst [i] = fcn (data_src [i]); \\                      \n\
}                                                            \n\
                                                             \n\
                                                             \n\
#ifdef INTEGER                                               \n\
                                                             \n\
MATH_FUNC(abs)                                               \n\
                                                             \n\
#endif                                                       \n\
                                                             \n\
                                                             \n\
#if defined (FLOATINGPOINT) && ! defined (COMPLEX)           \n\
                                                             \n\
MATH_FUNC(fabs)                                              \n\
MATH_FUNC(acos)                                              \n\
MATH_FUNC(acosh)                                             \n\
MATH_FUNC(asin)                                              \n\
MATH_FUNC(asinh)                                             \n\
MATH_FUNC(atan)                                              \n\
MATH_FUNC(atanh)                                             \n\
MATH_FUNC(cbrt)                                              \n\
MATH_FUNC(ceil)                                              \n\
MATH_FUNC(cos)                                               \n\
MATH_FUNC(cosh)                                              \n\
MATH_FUNC(erf)                                               \n\
MATH_FUNC(erfc)                                              \n\
MATH_FUNC(exp)                                               \n\
MATH_FUNC(expm1)                                             \n\
MATH_FUNC(floor)                                             \n\
MATH_FUNC(isfinite)                                          \n\
MATH_FUNC(isinf)                                             \n\
MATH_FUNC(isnan)                                             \n\
MATH_FUNC(lgamma)                                            \n\
MATH_FUNC(log)                                               \n\
MATH_FUNC(log2)                                              \n\
MATH_FUNC(log10)                                             \n\
MATH_FUNC(log1p)                                             \n\
MATH_FUNC(round)                                             \n\
MATH_FUNC(sign)                                              \n\
MATH_FUNC(sin)                                               \n\
MATH_FUNC(sinh)                                              \n\
MATH_FUNC(sqrt)                                              \n\
MATH_FUNC(tan)                                               \n\
MATH_FUNC(tanh)                                              \n\
MATH_FUNC(tgamma)                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_fix                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v;                                                    \n\
  v = data_src [i];                                          \n\
  if (v < ZERO)                                              \n\
    v = ceil (v);                                            \n\
  else                                                       \n\
    v = floor (v);                                           \n\
  data_dst [i] = v;                                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_power1e                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE exponent)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = pow (data_src [i], exponent);               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_power1b                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE base)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = pow (base, data_src [i]);                   \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_power2                                                   \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = pow (data_src1 [i], data_src2 [i]);         \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_atan2                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = atan2 (data_src1 [i], data_src2 [i]);       \n\
}                                                            \n\
                                                             \n\
#endif                                                       \n\
                                                             \n\
                                                             \n\
#if defined (COMPLEX)                                        \n\
                                                             \n\
#define R_ABS(z) sqrt (z.x*z.x + z.y*z.y)                    \n\
#define R_ARG(z) atan2 (z.y, z.x)                            \n\
                                                             \n\
                                                             \n\
// real to complex functions                                 \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_real2complex_r                                           \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE1 *data_src,                           \n\
   const ulong ofs_src,                                      \n\
   const TYPE1 val)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = (TYPE) (data_src [i], val);                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_real2complex_i                                           \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE1 *data_src,                           \n\
   const ulong ofs_src,                                      \n\
   const TYPE1 val)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = (TYPE) (val, data_src [i]);                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_real2complex_ri                                          \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE1 *data_src1,                          \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE1 *data_src2,                          \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = (TYPE) (data_src1 [i], data_src2 [i]);      \n\
}                                                            \n\
                                                             \n\
                                                             \n\
// complex to real functions                                 \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_real                                                     \n\
  (__global TYPE1 *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = data_src [i].x;                             \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_imag                                                     \n\
  (__global TYPE1 *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = data_src [i].y;                             \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_fabs                                                     \n\
  (__global TYPE1 *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = R_ABS (z);                                  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_arg                                                      \n\
  (__global TYPE1 *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = R_ARG (z);                                  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_isfinite                                                 \n\
  (__global TYPE1 *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = all (isfinite (data_src [i]));              \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_isinf                                                    \n\
  (__global TYPE1 *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = any (isinf (data_src [i]));                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_isnan                                                    \n\
  (__global TYPE1 *data_dst,                                 \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = any (isnan (data_src [i]));                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
// complex to complex functions                              \n\
                                                             \n\
                                                             \n\
MATH_FUNC(ceil)                                              \n\
MATH_FUNC(floor)                                             \n\
MATH_FUNC(round)                                             \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_fix                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE v;                                                    \n\
  v = data_src [i];                                          \n\
  if (v.x < ZERO1)                                           \n\
    v.x = ceil (v.x);                                        \n\
  else                                                       \n\
    v.x = floor (v.x);                                       \n\
  if (v.y < ZERO1)                                           \n\
    v.y = ceil (v.y);                                        \n\
  else                                                       \n\
    v.y = floor (v.y);                                       \n\
  data_dst [i] = v;                                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sign                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  if (IS_NE (z, ZERO))                                       \n\
    z = z / R_ABS (z);                                       \n\
  data_dst [i] = z;                                          \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_conj                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = (TYPE) (z.x, -z.y);                         \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_sqrt (TYPE z)                                              \n\
{                                                            \n\
  TYPE t;                                                    \n\
  t = R_ABS (z) + z.x * (TYPE) (ONE1, -ONE1);                \n\
  t = sqrt (((TYPE1) 0.5) * max (t, ZERO));                  \n\
  t.y *= z.y < ZERO1 ? -ONE1 : ONE1;                         \n\
  return t;                                                  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_exp (TYPE z)                                               \n\
{                                                            \n\
  return exp (z.x) * ((TYPE) (cos (z.y), sin (z.y)));        \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_log (TYPE z)                                               \n\
{                                                            \n\
  return (TYPE) (log (R_ABS (z)), R_ARG (z));                \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_pow (TYPE x, TYPE y)                                       \n\
{                                                            \n\
  return IS_EQ (x, ZERO) ?                                   \n\
    (IS_EQ (y, ZERO) ? ONE : ZERO) :                         \n\
    c_exp (MUL (y, c_log (x)));                              \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_asinh (TYPE z)                                             \n\
{                                                            \n\
  TYPE t = c_log (z + c_sqrt (ONE + MUL (z, z)));            \n\
  if ((z.x == ZERO1) && (z.y < -ONE1))                       \n\
    t.x *= -ONE1;                                            \n\
  return t;                                                  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_acosh (TYPE z)                                             \n\
{                                                            \n\
  return ((TYPE1)2.0) * c_log (c_sqrt (((TYPE1)0.5) * (z + ONE)) \n\
                             + c_sqrt (((TYPE1)0.5) * (z - ONE))); \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_atanh (TYPE z)                                             \n\
{                                                            \n\
  TYPE nd = ONE1 + z.x * (TYPE) (ONE1, -ONE1);               \n\
  nd = z.y*z.y + nd*nd;                                      \n\
  nd.x = ((TYPE1)0.25) * (log (nd.x / nd.y));                \n\
  nd.y = ONE1 - z.x*z.x - z.y*z.y;                           \n\
  nd.y = ((TYPE1)0.5) * atan2 (((TYPE1)2.0) * z.y, nd.y);    \n\
                                                             \n\
  return nd;                                                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_asin (TYPE z)                                              \n\
{                                                            \n\
  TYPE t = (TYPE) (-z.y, z.x);                               \n\
  t = c_log (t + c_sqrt (ONE + MUL (t, t)));                 \n\
  return (TYPE) (t.y, -t.x);                                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_acos (TYPE z)                                              \n\
{                                                            \n\
  return (TYPE) (1.5707963267948966192313216916397514L, ZERO1) - c_asin (z); \n\
}                                                            \n\
                                                             \n\
                                                             \n\
TYPE                                                         \n\
c_atan (TYPE z)                                              \n\
{                                                            \n\
  TYPE nd = z.y + (TYPE) (ONE1, -ONE1);                      \n\
  nd = z.x*z.x + nd*nd;                                      \n\
  nd.y = ((TYPE1)0.25) * (log (nd.x / nd.y));                \n\
  nd.x = ONE1 - z.x*z.x - z.y*z.y;                           \n\
  nd.x = ((TYPE1)0.5) * atan2 (((TYPE1)2.0) * z.x, nd.x);    \n\
                                                             \n\
  return nd;                                                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sqrt                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z, t;                                                 \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_sqrt (z);                                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_exp                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_exp (z);                                  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_log                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_log (z);                                  \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_log2                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_log (z) / log ((TYPE1) 2);                \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_log10                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_log (z) / log ((TYPE1) 10);               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_cos                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = (TYPE) (cos (z.x) * cosh (z.y), -sin (z.x) * sinh (z.y)); \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_cosh                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = (TYPE) (cosh (z.x) * cos (z.y), sinh (z.x) * sin (z.y)); \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sin                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = (TYPE) (sin (z.x) * cosh (z.y), cos (z.x) * sinh (z.y)); \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_sinh                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = (TYPE) (sinh (z.x) * cos (z.y), cosh (z.x) * sin (z.y)); \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_tan                                                      \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z, zs, zc;                                            \n\
  z = data_src [i];                                          \n\
  zs = (TYPE) (sin (z.x) * cosh (z.y), cos (z.x) * sinh (z.y)); \n\
  zc = (TYPE) (cos (z.x) * cosh (z.y), -sin (z.x) * sinh (z.y)); \n\
  data_dst [i] = DIV (zs, zc);                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_tanh                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z, zs, zc;                                            \n\
  z = data_src [i];                                          \n\
  zs = (TYPE) (sinh (z.x) * cos (z.y), cosh (z.x) * sin (z.y)); \n\
  zc = (TYPE) (cosh (z.x) * cos (z.y), sinh (z.x) * sin (z.y)); \n\
  data_dst [i] = DIV (zs, zc);                               \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_acos                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_acos (z);                                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_acosh                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_acosh (z);                                \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_asin                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_asin (z);                                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_asinh                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_asinh (z);                                \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_atan                                                     \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_atan (z);                                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_atanh                                                    \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  TYPE z;                                                    \n\
  z = data_src [i];                                          \n\
  data_dst [i] = c_atanh (z);                                \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_power1e                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE exponent)                                      \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = c_pow (data_src [i], exponent);             \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_power1b                                                  \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src,                            \n\
   const ulong ofs_src,                                      \n\
   const TYPE base)                                          \n\
{                                                            \n\
  data_src += ofs_src;                                       \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = c_pow (base, data_src [i]);                 \n\
}                                                            \n\
                                                             \n\
                                                             \n\
__kernel void                                                \n\
ocl_power2                                                   \n\
  (__global TYPE *data_dst,                                  \n\
   const __global TYPE *data_src1,                           \n\
   const ulong ofs_src1,                                     \n\
   const __global TYPE *data_src2,                           \n\
   const ulong ofs_src2)                                     \n\
{                                                            \n\
  data_src1 += ofs_src1;                                     \n\
  data_src2 += ofs_src2;                                     \n\
  size_t i = get_global_id (0);                              \n\
  data_dst [i] = c_pow (data_src1 [i], data_src2 [i]);       \n\
}                                                            \n\
                                                             \n\
#endif                                                       \n\
";

#define KERNEL_ENTRY( kernel ) \
  case OclArrayKernels::kernel: return "ocl_" #kernel;

const std::string
get_array_prog_kernel_name (OclArrayKernels::Kernel kernel)
{
  // all kernels of OclArrayKernels::Kernel (but the last one) must be registered here
  switch (kernel) {
    KERNEL_ENTRY( fill );
    KERNEL_ENTRY( fill0 );
    KERNEL_ENTRY( eye );
    KERNEL_ENTRY( linspace );
    KERNEL_ENTRY( logspace );
    KERNEL_ENTRY( ndgrid1 );
    KERNEL_ENTRY( repmat1 );
    KERNEL_ENTRY( cat );
    KERNEL_ENTRY( transpose );
    KERNEL_ENTRY( hermitian );
    KERNEL_ENTRY( as_index );
    KERNEL_ENTRY( index );
    KERNEL_ENTRY( assign_el );
    KERNEL_ENTRY( assign );
    KERNEL_ENTRY( assign0 );
    KERNEL_ENTRY( assign_el_logind );
    KERNEL_ENTRY( findfirst );
    KERNEL_ENTRY( findlast );
    KERNEL_ENTRY( all );
    KERNEL_ENTRY( any );
    KERNEL_ENTRY( sum );
    KERNEL_ENTRY( sumsq );
    KERNEL_ENTRY( prod );
    KERNEL_ENTRY( cumsum );
    KERNEL_ENTRY( cumprod );
    KERNEL_ENTRY( mean );
    KERNEL_ENTRY( meansq );
    KERNEL_ENTRY( std );
    KERNEL_ENTRY( max );
    KERNEL_ENTRY( max2 );
    KERNEL_ENTRY( max1 );
    KERNEL_ENTRY( min );
    KERNEL_ENTRY( min2 );
    KERNEL_ENTRY( min1 );
    KERNEL_ENTRY( cummax );
    KERNEL_ENTRY( cummin );
    KERNEL_ENTRY( compare );
    KERNEL_ENTRY( logic );
    KERNEL_ENTRY( fmad1 );
    KERNEL_ENTRY( fmad2 );
    KERNEL_ENTRY( uminus );
    KERNEL_ENTRY( add1 );
    KERNEL_ENTRY( add2 );
    KERNEL_ENTRY( sub1m );
    KERNEL_ENTRY( sub1s );
    KERNEL_ENTRY( sub2 );
    KERNEL_ENTRY( mul1 );
    KERNEL_ENTRY( mul2 );
    KERNEL_ENTRY( mtimes );
    KERNEL_ENTRY( div1n );
    KERNEL_ENTRY( div1d );
    KERNEL_ENTRY( div2 );
    KERNEL_ENTRY( abs );
    KERNEL_ENTRY( fabs );
    KERNEL_ENTRY( acos );
    KERNEL_ENTRY( acosh );
    KERNEL_ENTRY( asin );
    KERNEL_ENTRY( asinh );
    KERNEL_ENTRY( atan );
    KERNEL_ENTRY( atanh );
    KERNEL_ENTRY( cbrt );
    KERNEL_ENTRY( ceil );
    KERNEL_ENTRY( cos );
    KERNEL_ENTRY( cosh );
    KERNEL_ENTRY( erf );
    KERNEL_ENTRY( erfc );
    KERNEL_ENTRY( exp );
    KERNEL_ENTRY( expm1 );
    KERNEL_ENTRY( fix );
    KERNEL_ENTRY( floor );
    KERNEL_ENTRY( isfinite );
    KERNEL_ENTRY( isinf );
    KERNEL_ENTRY( isnan );
    KERNEL_ENTRY( lgamma );
    KERNEL_ENTRY( log );
    KERNEL_ENTRY( log2 );
    KERNEL_ENTRY( log10 );
    KERNEL_ENTRY( log1p );
    KERNEL_ENTRY( round );
    KERNEL_ENTRY( sign );
    KERNEL_ENTRY( sin );
    KERNEL_ENTRY( sinh );
    KERNEL_ENTRY( sqrt );
    KERNEL_ENTRY( tan );
    KERNEL_ENTRY( tanh );
    KERNEL_ENTRY( tgamma );
    KERNEL_ENTRY( power1e );
    KERNEL_ENTRY( power1b );
    KERNEL_ENTRY( power2 );
    KERNEL_ENTRY( atan2 );
    KERNEL_ENTRY( real2complex_r );
    KERNEL_ENTRY( real2complex_i );
    KERNEL_ENTRY( real2complex_ri );
    KERNEL_ENTRY( real );
    KERNEL_ENTRY( imag );
    KERNEL_ENTRY( arg );
    KERNEL_ENTRY( conj );
    default:
      ocl_error ("ocl_array_prog: kernel name not found");
  }
}

#undef KERNEL_ENTRY
