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

/*
 * A minor part of this file is based on content from the following files,
 * originally published with GNU Octave 3.8.0, distributed under the same
 * license (as OCL, see above), with the following Copyright notices:
 *
 * ops.h:
 *   Copyright (C) 1996-2013 John W. Eaton
 *   Copyright (C) 2009 VZLU Prague, a.s.
 */

#include <octave/oct.h>
#include <ops.h>

#include "ocl_octave_versions.h"
#include "ocl_ov_matrix.h"


#if ! defined (OCL_OCTAVE_VERSION_4_4_0_AND_HIGHER) // for octave versions < 4.4.0

#define OCL_INSTALL_UNOP(op, t, f) \
  octave_value_typeinfo::register_unary_op \
  (octave_value::op, t::static_type_id (), f < t >);

#define OCL_INSTALL_NCUNOP(op, t, f) \
  octave_value_typeinfo::register_non_const_unary_op \
  (octave_value::op, t::static_type_id (), f < t >);

#define OCL_INSTALL_BINOP(op, t, t1, t2, f) \
  octave_value_typeinfo::register_binary_op \
  (octave_value::op, t1::static_type_id (), t2::static_type_id (), \
   f < t, t1, t2 >);

#define OCL_INSTALL_ASSIGNOP(op, t1, t2, f) \
  octave_value_typeinfo::register_assign_op \
  (octave_value::op, t1::static_type_id (), t2::static_type_id (), \
   f < t1, t2 >);

#else // for octave versions >= 4.4.0

#include <octave/interpreter.h>

#define OCL_INSTALL_UNOP(op, t, f) \
  octave::interpreter::the_interpreter () -> get_type_info ().install_unary_op \
  (octave_value::op, t::static_type_id (), f < t >);

#define OCL_INSTALL_NCUNOP(op, t, f) \
  octave::interpreter::the_interpreter () -> get_type_info ().install_non_const_unary_op \
  (octave_value::op, t::static_type_id (), f < t >);

#define OCL_INSTALL_BINOP(op, t, t1, t2, f) \
  octave::interpreter::the_interpreter () -> get_type_info ().install_binary_op \
  (octave_value::op, t1::static_type_id (), t2::static_type_id (), \
   f < t, t1, t2 >);

#define OCL_INSTALL_ASSIGNOP(op, t1, t2, f)  \
  octave::interpreter::the_interpreter () -> get_type_info ().install_assign_op \
  (octave_value::op, t1::static_type_id (), t2::static_type_id (), \
   f < t1, t2 >);

#endif

// also uses CONCAT2(x,y) and CONCAT3(x,y,z) macros from ops.h


// ---------- definition of octave_ocl_*matrix classes' operators


template <typename element_type>
static element_type
scalar_ov_cast (const octave_scalar& v)
{ return element_type (v.scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_float_scalar& v)
{ return element_type (v.float_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_complex& v)
{ return element_type (v.complex_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_float_complex& v)
{ return element_type (v.float_complex_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_int8_scalar& v)
{ return element_type (v.int8_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_int16_scalar& v)
{ return element_type (v.int16_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_int32_scalar& v)
{ return element_type (v.int32_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_int64_scalar& v)
{ return element_type (v.int64_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_uint8_scalar& v)
{ return element_type (v.uint8_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_uint16_scalar& v)
{ return element_type (v.uint16_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_uint32_scalar& v)
{ return element_type (v.uint32_scalar_value ()); }


template <typename element_type>
static element_type
scalar_ov_cast (const octave_uint64_scalar& v)
{ return element_type (v.uint64_scalar_value ()); }


// macros to manage operators for OCL matrices


#define OCL_DEFNDUNOP_OP(name, op) \
  template <typename octave_value_type> \
  static octave_value \
  name (const octave_base_value& a) \
  { \
    const octave_value_type& v = dynamic_cast< const octave_value_type& > (a); \
    return new octave_value_type (op v.ocl_array_value ()); \
  }

#define OCL_DEFUNOP_METHOD(name, method) \
  template <typename octave_value_type> \
  static octave_value \
  name (const octave_base_value& a) \
  { \
    const octave_value_type& v = dynamic_cast< const octave_value_type& > (a); \
    return (v.method ()); \
  }

#define OCL_DEFNCUNOP_METHOD(name, method) \
  template <typename octave_value_type> \
  static void \
  name (octave_base_value& a) \
  { \
    octave_value_type& v = dynamic_cast< octave_value_type& > (a); \
    v.method (); \
  }

#define OCL_DEFNDBINOP_OP_MM(name, op) \
  template <typename octave_value_type, typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (const octave_base_value& a1, const octave_base_value& a2) \
  { \
    const octave_value_type1& v1 = dynamic_cast< const octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    return new octave_value_type (v1.ocl_array_value () op v2.ocl_array_value ()); \
  }

#define OCL_DEFNDBINOP_OP_MS(name, op) \
  template <typename octave_value_type, typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (const octave_base_value& a1, const octave_base_value& a2) \
  { \
    const octave_value_type1& v1 = dynamic_cast< const octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    return new octave_value_type (v1.ocl_array_value () op (scalar_ov_cast<typename octave_value_type::element_type> (v2))); \
  }

#define OCL_DEFNDBINOP_OP_SM(name, op) \
  template <typename octave_value_type, typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (const octave_base_value& a1, const octave_base_value& a2) \
  { \
    const octave_value_type1& v1 = dynamic_cast< const octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    return new octave_value_type ((scalar_ov_cast<typename octave_value_type::element_type> (v1)) op v2.ocl_array_value ()); \
  }

#define OCL_DEFNDBINOP_FN_MM(name, f) \
  template <typename octave_value_type, typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (const octave_base_value& a1, const octave_base_value& a2) \
  { \
    const octave_value_type1& v1 = dynamic_cast< const octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    return new octave_value_type (f (v1.ocl_array_value (), v2.ocl_array_value ())); \
  }

#define OCL_DEFNDBINOP_FN_MS(name, f) \
  template <typename octave_value_type, typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (const octave_base_value& a1, const octave_base_value& a2) \
  { \
    const octave_value_type1& v1 = dynamic_cast< const octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    return new octave_value_type (f (v1.ocl_array_value (), scalar_ov_cast<typename octave_value_type::element_type> (v2))); \
  }

#define OCL_DEFNDBINOP_FN_SM(name, f) \
  template <typename octave_value_type, typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (const octave_base_value& a1, const octave_base_value& a2) \
  { \
    const octave_value_type1& v1 = dynamic_cast< const octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    return new octave_value_type (f (scalar_ov_cast<typename octave_value_type::element_type> (v1), v2.ocl_array_value ())); \
  }

#define OCL_DEFNDBINOP_METHOD_MM(name, method) \
  template <typename octave_value_type, typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (const octave_base_value& a1, const octave_base_value& a2) \
  { \
    const octave_value_type1& v1 = dynamic_cast< const octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    typename octave_value_type::array_type v1array = typename octave_value_type::array_type (v1.ocl_array_value ());\
    return new octave_value_type (v1array.method (v2.ocl_array_value ())); \
  }

#define OCL_DEFNDBINOPS_OP(name, op) \
  OCL_DEFNDBINOP_OP_MM (CONCAT2(name, _mm), op) \
  OCL_DEFNDBINOP_OP_MS (CONCAT2(name, _ms), op) \
  OCL_DEFNDBINOP_OP_SM (CONCAT2(name, _sm), op)

#define OCL_DEFNDBINOPS2_OP(name, op) \
  OCL_DEFNDBINOP_OP_MS (CONCAT2(name, _ms), op) \
  OCL_DEFNDBINOP_OP_SM (CONCAT2(name, _sm), op)

#define OCL_DEFNDBINOPS_FN(name, f) \
  OCL_DEFNDBINOP_FN_MM (CONCAT2(name, _mm), f) \
  OCL_DEFNDBINOP_FN_MS (CONCAT2(name, _ms), f) \
  OCL_DEFNDBINOP_FN_SM (CONCAT2(name, _sm), f)

#define OCL_INSTALL_BINOPS(op, tm, ts, f) \
  OCL_INSTALL_BINOP (op, tm, tm, tm, CONCAT2(f, _mm)); \
  OCL_INSTALL_BINOP (op, tm, tm, ts, CONCAT2(f, _ms)); \
  OCL_INSTALL_BINOP (op, tm, ts, tm, CONCAT2(f, _sm))

#define OCL_INSTALL_BINOPS2(op, tm, ts, f) \
  OCL_INSTALL_BINOP (op, tm, tm, ts, CONCAT2(f, _ms)); \
  OCL_INSTALL_BINOP (op, tm, ts, tm, CONCAT2(f, _sm))

#define OCL_INSTALL_BINOPS_C(op, tcm, trm, tcs, f) \
  OCL_INSTALL_BINOP (op, tcm, trm, tcm, CONCAT2(f, _mm)); \
  OCL_INSTALL_BINOP (op, tcm, tcm, trm, CONCAT2(f, _mm)); \
  OCL_INSTALL_BINOP (op, tcm, trm, tcs, CONCAT2(f, _ms)); \
  OCL_INSTALL_BINOP (op, tcm, tcs, trm, CONCAT2(f, _sm))

#define OCL_INSTALL_BINOPS2_C(op, tcm, trm, tcs, f) \
  OCL_INSTALL_BINOP (op, tcm, trm, tcs, CONCAT2(f, _ms)); \
  OCL_INSTALL_BINOP (op, tcm, tcs, trm, CONCAT2(f, _sm))

#define OCL_DEFNDASSIGNOP_FN_M(name, f) \
  template <typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (octave_base_value& a1, const octave_value_list& idx, const octave_base_value& a2) \
  { \
    octave_value_type1& v1 = dynamic_cast< octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    v1.f (idx, v2.ocl_array_value ()); \
    return octave_value (); \
  }

#define OCL_DEFNDASSIGNOP_FN_S(name, f) \
  template <typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (octave_base_value& a1, const octave_value_list& idx, const octave_base_value& a2) \
  { \
    octave_value_type1& v1 = dynamic_cast< octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    v1.f (idx, scalar_ov_cast<typename octave_value_type1::element_type> (v2)); \
    return octave_value (); \
  }

#define OCL_DEFNDASSIGNOP_OP_M(name, op) \
  template <typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (octave_base_value& a1, const octave_value_list& idx, const octave_base_value& a2) \
  { \
    octave_value_type1& v1 = dynamic_cast< octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    assert (idx.empty ()); \
    v1.matrix_ref () op v2.ocl_array_value (); \
    return octave_value (); \
  }

#define OCL_DEFNDASSIGNOP_OP_S(name, op) \
  template <typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (octave_base_value& a1, const octave_value_list& idx, const octave_base_value& a2) \
  { \
    octave_value_type1& v1 = dynamic_cast< octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    assert (idx.empty ()); \
    v1.matrix_ref () op (scalar_ov_cast<typename octave_value_type1::element_type> (v2)); \
    return octave_value (); \
  }

#define OCL_DEFNDASSIGNOP_FNOP_M(name, fnop) \
  template <typename octave_value_type1, typename octave_value_type2> \
  static octave_value \
  name (octave_base_value& a1, const octave_value_list& idx, const octave_base_value& a2) \
  { \
    octave_value_type1& v1 = dynamic_cast< octave_value_type1& > (a1); \
    const octave_value_type2& v2 = dynamic_cast< const octave_value_type2& > (a2); \
    assert (idx.empty ()); \
    fnop (v1.matrix_ref (), v2.ocl_array_value ()); \
    return octave_value (); \
  }


// define operators for OCL matrices


OCL_DEFNDUNOP_OP (oclmat_not, !)
OCL_DEFNDUNOP_OP (oclmat_uplus, +)
OCL_DEFNDUNOP_OP (oclmat_uminus, -)
OCL_DEFUNOP_METHOD (oclmat_transpose, transpose)
OCL_DEFUNOP_METHOD (oclmat_hermitian, hermitian)
OCL_DEFNCUNOP_METHOD (oclmat_incr, increment)
OCL_DEFNCUNOP_METHOD (oclmat_decr, decrement)
OCL_DEFNCUNOP_METHOD (oclmat_changesign, changesign)

OCL_DEFNDBINOPS_OP (oclmat_lt, <)
OCL_DEFNDBINOPS_OP (oclmat_le, <=)
OCL_DEFNDBINOPS_OP (oclmat_gt, >)
OCL_DEFNDBINOPS_OP (oclmat_ge, >=)
OCL_DEFNDBINOPS_OP (oclmat_eq, ==)
OCL_DEFNDBINOPS_OP (oclmat_ne, !=)
OCL_DEFNDBINOPS_OP (oclmat_el_and, &&)
OCL_DEFNDBINOPS_OP (oclmat_el_or, ||)

OCL_DEFNDBINOPS_OP (oclmat_add, +)
OCL_DEFNDBINOPS_OP (oclmat_sub, -)
OCL_DEFNDBINOPS2_OP (oclmat_el_mul, *)
OCL_DEFNDBINOPS2_OP (oclmat_el_div, /)
OCL_DEFNDBINOP_FN_MM (oclmat_el_mul_mm, product)
OCL_DEFNDBINOP_FN_MM (oclmat_el_div_mm, quotient)
OCL_DEFNDBINOP_METHOD_MM (oclmat_mtimes, mtimes)
OCL_DEFNDBINOPS_FN (oclmat_el_pow, pow)

OCL_DEFNDASSIGNOP_FN_M (oclmat_assign_m, assign)
OCL_DEFNDASSIGNOP_OP_M (oclmat_assign_add_m, +=)
OCL_DEFNDASSIGNOP_OP_M (oclmat_assign_sub_m, -=)
OCL_DEFNDASSIGNOP_FNOP_M (oclmat_assign_el_mul_m, product_eq)
OCL_DEFNDASSIGNOP_FNOP_M (oclmat_assign_el_div_m, quotient_eq)

OCL_DEFNDASSIGNOP_FN_S (oclmat_assign_s, assign)
OCL_DEFNDASSIGNOP_OP_S (oclmat_assign_add_s, +=)
OCL_DEFNDASSIGNOP_OP_S (oclmat_assign_sub_s, -=)
OCL_DEFNDASSIGNOP_OP_S (oclmat_assign_mul_s, *=)
OCL_DEFNDASSIGNOP_OP_S (oclmat_assign_div_s, /=)


// define installation procedure for operators for OCL matrices


template < typename octave_value_ocl_matrix_type, typename octave_value_matrix_type, typename octave_value_scalar_type >
static void
oclmat_install (void)
{
  octave_value_ocl_matrix_type::register_type ();

  OCL_INSTALL_UNOP (op_not, octave_value_ocl_matrix_type, oclmat_not);
  OCL_INSTALL_UNOP (op_uplus, octave_value_ocl_matrix_type, oclmat_uplus);
  OCL_INSTALL_UNOP (op_uminus, octave_value_ocl_matrix_type, oclmat_uminus);
  OCL_INSTALL_UNOP (op_transpose, octave_value_ocl_matrix_type, oclmat_transpose);
  OCL_INSTALL_UNOP (op_hermitian, octave_value_ocl_matrix_type, oclmat_hermitian);
  OCL_INSTALL_NCUNOP (op_incr, octave_value_ocl_matrix_type, oclmat_incr);
  OCL_INSTALL_NCUNOP (op_decr, octave_value_ocl_matrix_type, oclmat_decr);
  OCL_INSTALL_NCUNOP (op_uminus, octave_value_ocl_matrix_type, oclmat_changesign);

  OCL_INSTALL_BINOPS(op_lt, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_lt);
  OCL_INSTALL_BINOPS(op_le, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_le);
  OCL_INSTALL_BINOPS(op_gt, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_gt);
  OCL_INSTALL_BINOPS(op_ge, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_ge);
  OCL_INSTALL_BINOPS(op_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_eq);
  OCL_INSTALL_BINOPS(op_ne, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_ne);
  OCL_INSTALL_BINOPS(op_el_and, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_el_and);
  OCL_INSTALL_BINOPS(op_el_or, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_el_or);

  OCL_INSTALL_BINOPS(op_add, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_add);
  OCL_INSTALL_BINOPS(op_sub, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_sub);
  OCL_INSTALL_BINOPS(op_el_mul, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_el_mul);
  OCL_INSTALL_BINOPS(op_el_div, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_el_div);
  OCL_INSTALL_BINOPS2(op_mul, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_el_mul);
  OCL_INSTALL_BINOP (op_mul, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, oclmat_mtimes);
  OCL_INSTALL_BINOP (op_div, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_el_div_ms);
  OCL_INSTALL_BINOPS(op_el_pow, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_el_pow);

  OCL_INSTALL_ASSIGNOP (op_asn_eq, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, oclmat_assign_m);
  OCL_INSTALL_ASSIGNOP (op_add_eq, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, oclmat_assign_add_m);
  OCL_INSTALL_ASSIGNOP (op_sub_eq, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, oclmat_assign_sub_m);
  OCL_INSTALL_ASSIGNOP (op_el_mul_eq, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, oclmat_assign_el_mul_m);
  OCL_INSTALL_ASSIGNOP (op_el_div_eq, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, oclmat_assign_el_div_m);

  OCL_INSTALL_ASSIGNOP (op_asn_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_assign_s);
  OCL_INSTALL_ASSIGNOP (op_add_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_assign_add_s);
  OCL_INSTALL_ASSIGNOP (op_sub_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_assign_sub_s);
  OCL_INSTALL_ASSIGNOP (op_mul_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_assign_mul_s);
  OCL_INSTALL_ASSIGNOP (op_div_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_assign_div_s);
  OCL_INSTALL_ASSIGNOP (op_el_mul_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_assign_mul_s);
  OCL_INSTALL_ASSIGNOP (op_el_div_eq, octave_value_ocl_matrix_type, octave_value_scalar_type, oclmat_assign_div_s);

  if (octave_value_scalar_type::static_type_id () != octave_scalar::static_type_id ()) {
    OCL_INSTALL_BINOPS2(op_lt, octave_value_ocl_matrix_type, octave_scalar, oclmat_lt);
    OCL_INSTALL_BINOPS2(op_le, octave_value_ocl_matrix_type, octave_scalar, oclmat_le);
    OCL_INSTALL_BINOPS2(op_gt, octave_value_ocl_matrix_type, octave_scalar, oclmat_gt);
    OCL_INSTALL_BINOPS2(op_ge, octave_value_ocl_matrix_type, octave_scalar, oclmat_ge);
    OCL_INSTALL_BINOPS2(op_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_eq);
    OCL_INSTALL_BINOPS2(op_ne, octave_value_ocl_matrix_type, octave_scalar, oclmat_ne);
    OCL_INSTALL_BINOPS2(op_el_and, octave_value_ocl_matrix_type, octave_scalar, oclmat_el_and);
    OCL_INSTALL_BINOPS2(op_el_or, octave_value_ocl_matrix_type, octave_scalar, oclmat_el_or);

    OCL_INSTALL_BINOPS2(op_add, octave_value_ocl_matrix_type, octave_scalar, oclmat_add);
    OCL_INSTALL_BINOPS2(op_sub, octave_value_ocl_matrix_type, octave_scalar, oclmat_sub);
    OCL_INSTALL_BINOPS2(op_el_mul, octave_value_ocl_matrix_type, octave_scalar, oclmat_el_mul);
    OCL_INSTALL_BINOPS2(op_el_div, octave_value_ocl_matrix_type, octave_scalar, oclmat_el_div);
    OCL_INSTALL_BINOPS2(op_mul, octave_value_ocl_matrix_type, octave_scalar, oclmat_el_mul);
    OCL_INSTALL_BINOP (op_div, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, octave_scalar, oclmat_el_div_ms);
    OCL_INSTALL_BINOPS2(op_el_pow, octave_value_ocl_matrix_type, octave_scalar, oclmat_el_pow);

    OCL_INSTALL_ASSIGNOP (op_asn_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_assign_s);
    OCL_INSTALL_ASSIGNOP (op_add_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_assign_add_s);
    OCL_INSTALL_ASSIGNOP (op_sub_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_assign_sub_s);
    OCL_INSTALL_ASSIGNOP (op_mul_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_assign_mul_s);
    OCL_INSTALL_ASSIGNOP (op_div_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_assign_div_s);
    OCL_INSTALL_ASSIGNOP (op_el_mul_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_assign_mul_s);
    OCL_INSTALL_ASSIGNOP (op_el_div_eq, octave_value_ocl_matrix_type, octave_scalar, oclmat_assign_div_s);
  }

  if (octave_value_scalar_type::static_type_id () != octave_float_scalar::static_type_id ()) {
    OCL_INSTALL_BINOPS2(op_lt, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_lt);
    OCL_INSTALL_BINOPS2(op_le, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_le);
    OCL_INSTALL_BINOPS2(op_gt, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_gt);
    OCL_INSTALL_BINOPS2(op_ge, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_ge);
    OCL_INSTALL_BINOPS2(op_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_eq);
    OCL_INSTALL_BINOPS2(op_ne, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_ne);
    OCL_INSTALL_BINOPS2(op_el_and, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_el_and);
    OCL_INSTALL_BINOPS2(op_el_or, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_el_or);

    OCL_INSTALL_BINOPS2(op_add, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_add);
    OCL_INSTALL_BINOPS2(op_sub, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_sub);
    OCL_INSTALL_BINOPS2(op_el_mul, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_el_mul);
    OCL_INSTALL_BINOPS2(op_el_div, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_el_div);
    OCL_INSTALL_BINOPS2(op_mul, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_el_mul);
    OCL_INSTALL_BINOP (op_div, octave_value_ocl_matrix_type, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_el_div_ms);
    OCL_INSTALL_BINOPS2(op_el_pow, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_el_pow);

    OCL_INSTALL_ASSIGNOP (op_asn_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_assign_s);
    OCL_INSTALL_ASSIGNOP (op_add_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_assign_add_s);
    OCL_INSTALL_ASSIGNOP (op_sub_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_assign_sub_s);
    OCL_INSTALL_ASSIGNOP (op_mul_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_assign_mul_s);
    OCL_INSTALL_ASSIGNOP (op_div_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_assign_div_s);
    OCL_INSTALL_ASSIGNOP (op_el_mul_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_assign_mul_s);
    OCL_INSTALL_ASSIGNOP (op_el_div_eq, octave_value_ocl_matrix_type, octave_float_scalar, oclmat_assign_div_s);
  }
}


template < typename complex_ocl_matrix_type, typename real_ocl_matrix_type, typename complex_scalar_type >
static void
oclmat_install_c (void)
{
  OCL_INSTALL_BINOPS_C(op_lt, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_lt);
  OCL_INSTALL_BINOPS_C(op_le, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_le);
  OCL_INSTALL_BINOPS_C(op_gt, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_gt);
  OCL_INSTALL_BINOPS_C(op_ge, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_ge);
  OCL_INSTALL_BINOPS_C(op_eq, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_eq);
  OCL_INSTALL_BINOPS_C(op_ne, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_ne);
  OCL_INSTALL_BINOPS_C(op_el_and, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_el_and);
  OCL_INSTALL_BINOPS_C(op_el_or, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_el_or);

  OCL_INSTALL_BINOPS_C(op_add, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_add);
  OCL_INSTALL_BINOPS_C(op_sub, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_sub);
  OCL_INSTALL_BINOPS_C(op_el_mul, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_el_mul);
  OCL_INSTALL_BINOPS_C(op_el_div, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_el_div);
  OCL_INSTALL_BINOPS2_C(op_mul, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_el_mul);
  OCL_INSTALL_BINOP (op_mul, complex_ocl_matrix_type, complex_ocl_matrix_type, real_ocl_matrix_type, oclmat_mtimes);
  OCL_INSTALL_BINOP (op_mul, complex_ocl_matrix_type, real_ocl_matrix_type, complex_ocl_matrix_type, oclmat_mtimes);
  OCL_INSTALL_BINOP (op_div, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_el_div_ms);
  OCL_INSTALL_BINOPS_C(op_el_pow, complex_ocl_matrix_type, real_ocl_matrix_type, complex_scalar_type, oclmat_el_pow);
}


// ---------- public functions


void
install_ocl_matrix_types (void)
{
  oclmat_install < octave_ocl_matrix, octave_matrix, octave_scalar > ();
  oclmat_install < octave_ocl_float_matrix, octave_float_matrix, octave_float_scalar > ();
  oclmat_install < octave_ocl_complex_matrix, octave_complex_matrix, octave_complex > ();
  oclmat_install < octave_ocl_float_complex_matrix, octave_float_complex_matrix, octave_float_complex > ();
  oclmat_install < octave_ocl_int8_matrix, octave_int8_matrix, octave_int8_scalar > ();
  oclmat_install < octave_ocl_int16_matrix, octave_int16_matrix, octave_int16_scalar > ();
  oclmat_install < octave_ocl_int32_matrix, octave_int32_matrix, octave_int32_scalar > ();
  oclmat_install < octave_ocl_int64_matrix, octave_int64_matrix, octave_int64_scalar > ();
  oclmat_install < octave_ocl_uint8_matrix, octave_uint8_matrix, octave_uint8_scalar > ();
  oclmat_install < octave_ocl_uint16_matrix, octave_uint16_matrix, octave_uint16_scalar > ();
  oclmat_install < octave_ocl_uint32_matrix, octave_uint32_matrix, octave_uint32_scalar > ();
  oclmat_install < octave_ocl_uint64_matrix, octave_uint64_matrix, octave_uint64_scalar > ();

  oclmat_install_c < octave_ocl_complex_matrix, octave_ocl_matrix, octave_complex > ();
  oclmat_install_c < octave_ocl_float_complex_matrix, octave_ocl_float_matrix, octave_float_complex > ();
}
