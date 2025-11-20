#!/usr/bin/env python3
"""
Generator for stats::chisquare::stddev tests validated against scipy.stats.chi2.std.
The expected values are computed with SciPy and used as the oracle.
"""

from functools import reduce
import operator
import numpy as np
import scipy.stats as stats
import math
from numpy.lib.stride_tricks import as_strided

tolerance = 1e-6
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

def cpp_index_access(varname, flat_index, shape):
    total = prod(shape)
    if total == 1:
        return f'{varname}[0]'
    if len(shape) == 1:
        return f'{varname}[{flat_index}]'
    indices = np.unravel_index(flat_index, shape)
    return varname + ''.join(f'[{int(idx)}]' for idx in indices)

def make_std_test(testname, k_shape, k_vals):
    """
    Build a test that constructs k, computes expected = scipy.stats.chi2.std(k),
    and checks stats::stddev(k) elementwise against expected.
    """
    k_np = np.asarray(k_vals).reshape(k_shape)
    # SciPy returns array-like when passed array-like
    expected_np = stats.chi2.std(k_np)
    flat_expected = np.asarray(expected_np).ravel()
    out_elems = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.{testname}')
    lines.append(f'  * @brief Verify stats::chisquare::stddev matches scipy.stats.chi2.std for {testname}')
    lines.append('  */')
    lines.append(f'TEST(CHISQUARE, {testname})')
    lines.append('{')

    # input tensor k
    lines.append(f'    Tensor<float> k({{{",".join(str(d) for d in k_shape)}}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('k', np.asarray(k_vals).ravel(), k_shape))
    lines.append('')
    # call
    lines.append('    Tensor<float> result = stats::chisquare::stddev(k);')
    lines.append('')
    # expected_result
    lines.append(f'    Tensor<float> expected_result({{{out_elems}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(flat_expected)};')
    lines.append('')
    # element-wise check (report index precisely by using result[i] / expected_result[i])
    for i in range(out_elems):
        result_access = cpp_index_access('result', i, k_shape if prod(k_shape)!=1 else (1,))
        # expected_result is flat so index by [i]
        lines.append(f'    EXPECT_NEAR({result_access}, expected_result[{i}], {tolerance});')

    lines.append('}')
    return '\n'.join(lines)


def make_view_strides_std_test(testname,
                                k_owner_vals, k_offset, k_alias_shape, k_alias_strides,
                                memloc='MemoryLocation::DEVICE',
                                tol=1e-6):
    """
    Build a test where k_alias is a non-contiguous view of k_owner.
    Expected values are computed with SciPy using the aliased NumPy view.
    """
    k_owner_np = np.asarray(k_owner_vals, dtype=float)

    def build_alias(owner_np, offset, strides_elems, alias_shape):
        base = owner_np[offset:]
        byte_strides = tuple(int(s * owner_np.itemsize) for s in strides_elems)
        return as_strided(base, shape=tuple(alias_shape), strides=byte_strides)

    k_alias_np = build_alias(k_owner_np, k_offset, k_alias_strides, tuple(k_alias_shape))
    expected_np = stats.chi2.std(k_alias_np)
    flat_expected = expected_np.ravel()
    total = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.{testname}')
    lines.append('  * @brief Ensure stats::chisquare::stddev accepts non-contiguous/alias views for k and matches SciPy')
    lines.append('  */')
    lines.append(f'TEST(CHISQUARE, {testname})')
    lines.append('{')

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

    lines.append('    Tensor<float> result = stats::chisquare::stddev(k_alias);')
    lines.append('')
    lines.append(f'    Tensor<float> expected_result({{{", ".join(str(d) for d in k_alias_shape)}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(flat_expected)};')
    lines.append('')
    for i in range(total):
        result_access = cpp_index_access('result', i, k_alias_shape)
        expected_access = cpp_index_access('expected_result', i, k_alias_shape)
        lines.append(f'    EXPECT_NEAR({result_access}, {expected_access}, {tol});')

    lines.append('}')
    return '\n'.join(lines)


if __name__ == '__main__':
    blocks = []
    blocks.append("""/**
  * @file chisquare_stddev.cpp
  * @brief File generated by tests/stats/chisquare_stddev.py.
  */""")

    # scalar (degrees of freedom = df)
    blocks.append(make_std_test('std_scalar_df_3', (1,), [3.0]))

    # vector of dfs
    blocks.append(make_std_test('std_vector_dfs', (4,), [1.0, 2.0, 3.0, 4.0]))

    # random vector of dfs
    r = np.abs(np.random.randn(6)) + 0.1  # ensure positive df
    blocks.append(make_std_test('std_vector_random_positive', (6,), r.tolist()))

    # 2D shape (matrix of dfs)
    blocks.append(make_std_test('std_matrix_2x3', (2,3),
                                 [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]))

    # view/strides test
    k_owner_vals = [2.0, 99.0, 1.0, 99.0, 5.0, 99.0]
    k_offset = 0
    k_alias_shape = (3,)
    k_alias_strides = (2,)
    blocks.append(make_view_strides_std_test(
        'std_view_with_weird_strides',
        k_owner_vals, k_offset, k_alias_shape, k_alias_strides,
        memloc='MemoryLocation::DEVICE',
        tol=tolerance
    ))

    print('\n\n'.join(blocks))
