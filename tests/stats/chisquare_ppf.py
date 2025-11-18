#!/usr/bin/env python3
"""
Generator for chisquare::ppf tests (random-grid + broadcasting + safe view-with-weird-strides).
"""
from functools import reduce
import operator
import numpy as np
import scipy.stats as stats
import math
from numpy.lib.stride_tricks import as_strided

tolerance = 1e-4
num_test = 0
np.random.seed(440)

def prod(iterable):
    return int(reduce(operator.mul, iterable, 1))

def fmt_float(v, precision=17):
    v = float(v)
    if math.isnan(v):
        return 'std::numeric_limits<float>::quiet_NaN()'
    if math.isinf(v):
        return 'std::numeric_limits<float>::infinity()' if v > 0 else '-std::numeric_limits<float>::infinity()'
    s = np.format_float_positional(v, precision=precision, unique=False, fractional=True, trim='k')
    if 'e' in s or 'E' in s:
        return s
    if '.' not in s:
        s += '.0'
    return s


def cpp_vec_or_scalar_assignment(varname, arr, shape):
    a = np.asarray(arr).ravel()
    total = prod(shape)
    if total == 1:
        return f'{varname} = {fmt_float(float(a[0]))};'
    items = ', '.join(fmt_float(float(x)) for x in a)
    return f'{varname} = std::vector<float>{{{items}}};'


def flat_initializer(arr):
    a = np.asarray(arr).ravel()
    items = ', '.join(fmt_float(float(x)) for x in a)
    return '{' + items + '}'


def broadcast_shape(shape_a, shape_b):
    a = list(shape_a)[::-1]
    b = list(shape_b)[::-1]
    out = []
    for i in range(max(len(a), len(b))):
        ai = a[i] if i < len(a) else 1
        bi = b[i] if i < len(b) else 1
        if ai == 1:
            outi = bi
        elif bi == 1:
            outi = ai
        elif ai == bi:
            outi = ai
        else:
            raise ValueError(f'incompatible shapes for broadcasting: {shape_a}, {shape_b}')
        out.append(outi)
    return tuple(out[::-1])

def cpp_index_access(varname, flat_index, shape):
    """
    Return access expression for arbitrary variable name in row-major order.
    Examples:
      - shape == (6,) -> 'varname[flat_index]'
      - shape == (2,3) and flat_index==4 -> 'varname[1][1]'
      - scalar -> 'varname[0]'
    """
    total = prod(shape)
    if total == 1:
        return f'{varname}[0]'
    if len(shape) == 1:
        return f'{varname}[{flat_index}]'
    indices = np.unravel_index(flat_index, shape)
    return varname + ''.join(f'[{int(idx)}]' for idx in indices)


def create_test_chi2(dof, n):
    global num_test
    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.ppf_random_test_dof_{dof}')
    lines.append(f'  * @brief Randomly generated chisquare ppf test on {n} matrix with dof = {dof}.')
    lines.append('  *')
    lines.append('  * The test is generated from the script, expecting values from numpy')
    lines.append('  */')
    lines.append(f'TEST(CHISQUARE, ppf_random_test_dof_{dof})')
    lines.append('{')
    num_test += 1

    q = np.linspace(0.05, 0.95, n)
    y = stats.chi2.ppf(q, dof)

    lines.append(f'    Tensor<float> q({{{n}}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('q', q, (n,)))

    lines.append(f'    Tensor<float> k({{1}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('k', [dof], (1,)))

    lines.append(f'    Tensor<float> expected_result({{{n}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(y)};')

    lines.append('    // run Temper chisquare::ppf')
    lines.append('    Tensor<float> result = stats::chisquare::ppf(q, k);')
    lines.append('    // check values')
    for i in range(n):
        lines.append(f'    EXPECT_NEAR(result[{i}], expected_result[{i}], {tolerance});')
    lines.append('}')
    return '\n'.join(lines)


def make_broadcast_test_block(testname, q_shape, q_vals, k_shape, k_vals):
    out_shape = broadcast_shape(q_shape, k_shape)
    q_np = np.asarray(q_vals).reshape(q_shape)
    k_np = np.asarray(k_vals).reshape(k_shape)
    out_b = np.broadcast_to(q_np, out_shape)
    k_b = np.broadcast_to(k_np, out_shape)
    expected = stats.chi2.ppf(out_b, k_b)
    flat_expected = expected.ravel()
    out_elems = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.{testname}')
    lines.append(f'  * @brief Verify broadcasting: {testname.replace("_", " ")}')
    lines.append('  */')
    lines.append(f'TEST(CHISQUARE, {testname})')
    lines.append('{')

    lines.append(f'    Tensor<float> q({{{",".join(str(d) for d in q_shape)}}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('q', np.asarray(q_vals).ravel(), q_shape))

    lines.append(f'    Tensor<float> k({{{",".join(str(d) for d in k_shape)}}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('k', np.asarray(k_vals).ravel(), k_shape))

    lines.append(f'    Tensor<float> expected_result({{{out_elems}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(flat_expected)};')

    lines.append('    Tensor<float> result = stats::chisquare::ppf(q, k);')
    lines.append('    // check values')
    for i in range(out_elems):
        result_access = cpp_index_access('result', i, out_shape)
        lines.append(f'    EXPECT_NEAR({result_access}, expected_result[{i}], {tolerance});')

    lines.append('}')
    return '\n'.join(lines)


def make_view_strides_chi2_test(testname,
                                q_owner_vals, q_offset, q_alias_shape, q_alias_strides,
                                k_owner_vals, k_offset, k_alias_shape, k_alias_strides,
                                memloc='MemoryLocation::DEVICE',
                                tol=1e-6):

    q_owner_np = np.asarray(q_owner_vals, dtype=float)
    k_owner_np = np.asarray(k_owner_vals, dtype=float)

    def build_alias(owner_np, offset, strides_elems, alias_shape):
        base = owner_np[offset:]
        byte_strides = tuple(int(s * owner_np.itemsize) for s in strides_elems)
        return as_strided(base, shape=alias_shape, strides=byte_strides)

    q_alias_np = build_alias(q_owner_np, q_offset, q_alias_strides, tuple(q_alias_shape))
    k_alias_np = build_alias(k_owner_np, k_offset, k_alias_strides, tuple(k_alias_shape))

    expected_np = stats.chi2.ppf(q_alias_np, k_alias_np)
    flat_expected = expected_np.ravel()
    total = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.{testname}')
    lines.append('  * @brief Ensure chisquare::ppf accepts non-contiguous/alias views for x and k')
    lines.append('  */')
    lines.append(f'TEST(CHISQUARE, {testname})')
    lines.append('{')

    lines.append(f'    Tensor<float> q_owner({{{len(q_owner_vals)}}}, {memloc});')
    xv_items = ', '.join(fmt_float(float(v)) for v in q_owner_vals)
    lines.append('    std::vector<float> q_owner_vals = {')
    lines.append(f'        {xv_items}')
    lines.append('    };')
    lines.append('    q_owner = q_owner_vals;')
    lines.append('')

    lines.append('    Tensor<float> q_alias(')
    lines.append('        q_owner,')
    lines.append(f'        std::vector<uint64_t>{{{q_offset}ull}},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(s)) + 'ull' for s in q_alias_shape) + '},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(d)) + 'ull' for d in q_alias_strides) + '}')
    lines.append('    );')
    lines.append('')

    lines.append(f'    Tensor<float> k_owner({{{len(k_owner_vals)}}}, {memloc});')
    kv_items = ', '.join(fmt_float(float(v)) for v in k_owner_vals)
    lines.append('    std::vector<float> k_owner_vals = {')
    lines.append(f'        {kv_items}')
    lines.append('    };')
    lines.append('    k_owner = k_owner_vals;')
    lines.append('')

    lines.append('    Tensor<float> k_alias(')
    lines.append('        k_owner,')
    lines.append(f'        std::vector<uint64_t>{{{k_offset}ull}},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(s)) + 'ull' for s in k_alias_shape) + '},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(d)) + 'ull' for d in k_alias_strides) + '}')
    lines.append('    );')
    lines.append('')

    lines.append('    Tensor<float> result = stats::chisquare::ppf(q_alias, k_alias);')
    lines.append('')
    lines.append(f'    Tensor<float> expected_result({{{", ".join(str(d) for d in q_alias_shape)}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(flat_expected)};')
    lines.append('')
    lines.append('    // compare element-wise (unroll to report precise index on failure)')
    for i in range(total):
        result_access = cpp_index_access('result', i, q_alias_shape)
        expected_access = cpp_index_access('expected_result', i, q_alias_shape)
        lines.append(f'    EXPECT_NEAR({result_access}, {expected_access}, {tolerance});')

    lines.append('}')
    return '\n'.join(lines)


if __name__ == '__main__':
    blocks = []

    blocks.append("""/**
  * @file chisquare_ppf.cpp
  * @brief File generated by tests/stats/chisquare_ppf.py.
  */""")

    for df in range(2, 21):
        blocks.append(create_test_chi2(df, 100))

    bc = []

    q1_shape = (4,)
    q1_vals = [0.05, 0.25, 0.50, 0.95]
    k1_shape = (1,)
    k1_vals = [3.0]
    bc.append(make_broadcast_test_block('ppf_broadcast_q_vector_k_scalar', q1_shape, q1_vals, k1_shape, k1_vals))

    q2_shape = (1,)
    q2_vals = [0.5]
    k2_shape = (3,)
    k2_vals = [3.0, 4.0, 2.0]
    bc.append(make_broadcast_test_block('ppf_broadcast_q_scalar_k_vector', q2_shape, q2_vals, k2_shape, k2_vals))

    q3_shape = (2, 2)
    q3_vals = [0.05, 0.25, 0.5, 0.95]
    k3_shape = (2,)
    k3_vals = [3.0, 2.0]
    bc.append(make_broadcast_test_block('ppf_broadcast_2d_q_k_vector', q3_shape, q3_vals, k3_shape, k3_vals))

    x4_shape = (2, 1)
    x4_vals = [0.33, 0.66]
    k4_shape = (1, 3)
    k4_vals = [2.0, 3.0, 4.0]
    bc.append(make_broadcast_test_block('ppf_broadcast_mixed_shapes_all_operands', x4_shape, x4_vals, k4_shape, k4_vals))

    blocks.extend(bc)

    q_owner_vals = [0.05, 99.0, 0.5, 99.0, 0.95, 99.0]
    q_offset = 0
    q_alias_shape = (3,)
    q_alias_strides = (2,)

    k_owner_vals = [1.0, 99.0, 5.0, 99.0, 10.0, 99.0]
    k_offset = 0
    k_alias_shape = (3,)
    k_alias_strides = (2,)

    blocks.append(make_view_strides_chi2_test(
        'ppf_view_with_weird_strides',
        q_owner_vals, q_offset, q_alias_shape, q_alias_strides,
        k_owner_vals, k_offset, k_alias_shape, k_alias_strides,
        memloc='MemoryLocation::DEVICE',
        tol=1e-6
    ))

    print('\n\n'.join(blocks))
