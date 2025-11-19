#!/usr/bin/env python3
"""
Generator for chisquare::rvs tests for:
template<typename value_t>
Tensor<value_t> rvs(const Tensor<value_t>& k,
    const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc,
    uint64_t seed)

This script reproduces the exact RNG in the C++ kernel (xorshift64*) using numpy.uint64
and computes expected values with scipy.stats.chi2.ppf.
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


# Reproduce the C++ RNG exactly (vectorized)
def generate_uniform_out(out_shape, seed):
    """
    Reproduce the C++ kernel RNG:
      s = seed ^ (flat + 0x9e3779b97f4a7c15ULL)
      s ^= s >> 12;
      s ^= s << 25;
      s ^= s >> 27;
      rnd = s * 2685821657736338717ULL;
      u = rnd / 18446744073709551616.0;
      clamp to [1e-16, 1-1e-16)
    Returns numpy array of shape out_shape dtype float64.
    """
    total = prod(out_shape)
    if total == 0:
        return np.empty(out_shape, dtype=np.float64)

    const = np.uint64(0x9e3779b97f4a7c15)
    multipl = np.uint64(2685821657736338717)
    idx = np.arange(total, dtype=np.uint64)
    s = np.uint64(seed) ^ (idx + const)

    # xorshift rounds - operate in uint64
    s = s ^ (s >> np.uint64(12))
    s = s ^ (s << np.uint64(25))
    s = s ^ (s >> np.uint64(27))

    rnd = (s * multipl).astype(np.uint64)
    # divide by 2**64 expressed as float
    u = rnd.astype(np.float64) / 18446744073709551616.0
    # clamp to [1e-16, 1-1e-16]
    np.clip(u, 1e-16, 1.0 - 1e-16, out=u)
    return u.reshape(out_shape)


def create_test_chi2(dof, out_n=100, seed=123456789, memloc='MemoryLocation::DEVICE'):
    """
    Create a test where k is a scalar (dof), out_shape is (out_n,),
    seed is non-zero for deterministic generation.
    """
    global num_test
    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.rvs_random_test_dof_{dof}')
    lines.append(f'  * @brief Deterministic chisquare::rvs test on {out_n} outputs with k = {dof} and seed = {seed}.')
    lines.append('  *')
    lines.append('  * The test is generated from the script, expecting values from numpy (reproducing C++ RNG).')
    lines.append('  */')
    lines.append(f'TEST(CHISQUARE, rvs_random_test_dof_{dof})')
    lines.append('{')
    num_test += 1

    out_shape = (out_n,)
    # generate u's with the exact RNG
    u = generate_uniform_out(out_shape, seed)
    y = stats.chi2.ppf(u, dof)

    # define k (scalar)
    lines.append(f'    Tensor<float> k({{1}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('k', [dof], (1,)))

    # pass out_shape as std::vector<uint64_t>
    lines.append(f'    std::vector<uint64_t> out_shape = {{{out_n}ull}};')
    lines.append(f'    uint64_t seed = {seed}ull;')
    lines.append(f'    Tensor<float> expected_result({{{out_n}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(y)};')

    lines.append('    // run Temper chisquare::rvs (k, out_shape, memloc, seed)')
    lines.append(f'    Tensor<float> result = stats::chisquare::rvs(k, out_shape, {memloc}, seed);')
    lines.append('    // check values')
    for i in range(out_n):
        lines.append(f'    EXPECT_NEAR(result[{i}], expected_result[{i}], {tolerance});')
    lines.append('}')
    return '\n'.join(lines)


def make_broadcast_test_block(testname, k_shape, k_vals, out_shape, seed=987654321, memloc='MemoryLocation::DEVICE'):
    """
    k_shape/k_vals specify the k tensor; out_shape specifies the desired output shape.
    k will broadcast to out_shape for ppf.
    """
    k_np = np.asarray(k_vals).reshape(k_shape)
    u = generate_uniform_out(out_shape, seed)
    # rely on numpy broadcasting when calling ppf
    expected = stats.chi2.ppf(u, k_np)
    flat_expected = expected.ravel()
    out_elems = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.{testname}')
    lines.append(f'  * @brief Verify rvs broadcasting: {testname.replace("_", " ")} (seed={seed})')
    lines.append('  */')
    lines.append(f'TEST(CHISQUARE, {testname})')
    lines.append('{')

    # define k tensor
    lines.append(f'    Tensor<float> k({{{",".join(str(d) for d in k_shape)}}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('k', np.asarray(k_vals).ravel(), k_shape))

    # out_shape vector
    lines.append('    std::vector<uint64_t> out_shape = {' + ', '.join(str(int(d)) + 'ull' for d in out_shape) + '};')
    lines.append(f'    uint64_t seed = {seed}ull;')

    lines.append(f'    Tensor<float> expected_result({{{out_elems}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(flat_expected)};')

    lines.append('    Tensor<float> result = stats::chisquare::rvs(k, out_shape, ' + memloc + ', seed);')
    lines.append('    // check values')
    for i in range(out_elems):
        result_access = cpp_index_access('result', i, tuple(out_shape))
        lines.append(f'    EXPECT_NEAR({result_access}, expected_result[{i}], {tolerance});')

    lines.append('}')
    return '\n'.join(lines)


def make_view_strides_chi2_test(testname,
                                k_owner_vals, k_offset, k_alias_shape, k_alias_strides,
                                out_shape=None,
                                memloc='MemoryLocation::DEVICE',
                                seed=135792468):
    """
    Creates a test where k is an alias view (non-contiguous). out_shape defaults to k_alias_shape
    (so q generated by the kernel has same shape). Expected is computed with broadcast of k_alias as needed.
    """
    k_owner_np = np.asarray(k_owner_vals, dtype=float)

    def build_alias(owner_np, offset, strides_elems, alias_shape):
        base = owner_np[offset:]
        byte_strides = tuple(int(s * owner_np.itemsize) for s in strides_elems)
        return as_strided(base, shape=alias_shape, strides=byte_strides)

    k_alias_np = build_alias(k_owner_np, k_offset, k_alias_strides, tuple(k_alias_shape))

    if out_shape is None:
        out_shape = tuple(k_alias_shape)

    u = generate_uniform_out(tuple(out_shape), seed)
    expected_np = stats.chi2.ppf(u, k_alias_np)
    flat_expected = expected_np.ravel()
    total = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CHISQUARE.{testname}')
    lines.append('  * @brief Ensure chisquare::rvs accepts non-contiguous/alias views for k (and produces correct ppf output)')
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

    lines.append('    std::vector<uint64_t> out_shape = {' + ', '.join(str(int(d)) + 'ull' for d in out_shape) + '};')
    lines.append(f'    uint64_t seed = {seed}ull;')
    lines.append('')

    lines.append('    Tensor<float> result = stats::chisquare::rvs(k_alias, out_shape, ' + memloc + ', seed);')
    lines.append('')
    lines.append(f'    Tensor<float> expected_result({{{", ".join(str(d) for d in out_shape)}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(flat_expected)};')
    lines.append('')
    lines.append('    // compare element-wise (unroll to report precise index on failure)')
    for i in range(total):
        result_access = cpp_index_access('result', i, tuple(out_shape))
        expected_access = cpp_index_access('expected_result', i, tuple(out_shape))
        lines.append(f'    EXPECT_NEAR({result_access}, {expected_access}, {tolerance});')

    lines.append('}')
    return '\n'.join(lines)


if __name__ == '__main__':
    blocks = []

    blocks.append("""/**
  * @file chisquare_rvs.cpp
  * @brief File generated by tests/stats/chisquare_rvs.py (rvs(k, out_shape, ...)).
  */""")

    # scalar-k tests for DOFs 2..20 (out_shape length = 100)
    for df in range(2, 21):
        blocks.append(create_test_chi2(df, out_n=100, seed=123456789))

    # broadcast tests: k shapes vs out_shape
    bc = []

    # k scalar, out_shape vector
    bc.append(make_broadcast_test_block('rvs_k_scalar_out_vec', k_shape=(1,), k_vals=[3.0], out_shape=(4,), seed=11111))

    # k vector broadcasting to out_shape larger
    bc.append(make_broadcast_test_block('rvs_k_vector_out_vec', k_shape=(3,), k_vals=[2.0, 3.0, 4.0], out_shape=(3,), seed=22222))

    # k with shape (2,) broadcasting to a 2x2 out_shape
    bc.append(make_broadcast_test_block('rvs_k_vector_broadcast_to_2d', k_shape=(2,), k_vals=[3.0, 2.0], out_shape=(2,2), seed=33333))

    # k with shape (1,3) broadcasting to (2,3)
    bc.append(make_broadcast_test_block('rvs_k_row_broadcast', k_shape=(1,3), k_vals=[2.0, 3.0, 4.0], out_shape=(2,3), seed=44444))

    blocks.extend(bc)

    # view/strides test for k (non-contiguous alias)
    k_owner_vals = [1.0, 99.0, 5.0, 99.0, 10.0, 99.0]
    k_offset = 0
    k_alias_shape = (3,)
    k_alias_strides = (2,)

    blocks.append(make_view_strides_chi2_test(
        'rvs_k_view_with_weird_strides',
        k_owner_vals, k_offset, k_alias_shape, k_alias_strides,
        out_shape=None,
        memloc='MemoryLocation::DEVICE',
        seed=55555
    ))

    print('\n\n'.join(blocks))
