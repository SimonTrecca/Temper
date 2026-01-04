#!/usr/bin/env python3
"""
Generator for math::upsample tests with typed test support.
Generates tests for both ZEROS and NEAREST modes.
"""
from functools import reduce
import operator
import numpy as np
import math

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
        return 'std::numeric_limits<float>::infinity()' if v > 0 else '-std::numeric_limits<float>:: infinity()'
    s = np.format_float_positional(v, precision=precision, unique=False, fractional=True, trim='k')
    if 'e' in s or 'E' in s:
        return s
    if '.' not in s:
        s += '.0'
    return s

def upsample_zeros(input_np, stride):
    """Reference implementation for ZEROS mode."""
    shape = input_np.shape
    channels, in_h, in_w = shape[-3], shape[-2], shape[-1]

    out_h = in_h * stride - (stride - 1)
    out_w = in_w * stride - (stride - 1)

    # Preserve batch dimensions
    out_shape = list(shape[:-2]) + [out_h, out_w]
    output = np.zeros(out_shape, dtype=input_np.dtype)

    # Place input values at strided positions
    output[..., ::stride, ::stride] = input_np

    return output

def upsample_nearest(input_np, stride):
    """Reference implementation for NEAREST mode."""
    shape = input_np.shape
    channels, in_h, in_w = shape[-3], shape[-2], shape[-1]

    out_h = in_h * stride - (stride - 1)
    out_w = in_w * stride - (stride - 1)

    # Use repeat for nearest neighbor
    output = np.repeat(np.repeat(input_np, stride, axis=-2), stride, axis=-1)

    # Trim to exact size (repeat might overshoot by stride-1)
    output = output[..., : out_h, :out_w]

    return output

def create_typed_test_upsample(testname, input_shape, stride, mode, seed):
    """
    Generate a typed test for upsample.
    mode: 'ZEROS' or 'NEAREST'
    """
    global num_test
    np.random.seed(seed)

    # Generate random input
    input_np = np.random.randn(*input_shape).astype(np.float32)

    # Compute expected output
    if mode == 'ZEROS':
        output_np = upsample_zeros(input_np, stride)
        mode_cpp = 'UpsampleMode::ZEROS'
    elif mode == 'NEAREST':
        output_np = upsample_nearest(input_np, stride)
        mode_cpp = 'UpsampleMode::NEAREST'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_shape = output_np.shape
    n_out = prod(out_shape)
    n_in = prod(input_shape)

    lines = []
    lines.append('/**')
    lines.append(f' * @test TypedUpsample.{testname}')
    lines.append(f' * @brief Upsample {mode} mode test: ')
    lines.append(f' *   input shape: {input_shape}')
    lines.append(f' *   stride: {stride}')
    lines.append(f' *   output shape: {out_shape}')
    lines.append(' */')
    lines.append(f'TYPED_TEST(TypedUpsample, {testname})')
    lines.append('{')
    lines.append('    using value_t = TypeParam;')
    lines.append('')

    # Input tensor
    lines.append(f'    Tensor<value_t> input(')
    lines.append('        {' + ', '.join(str(d) for d in input_shape) + '},')
    lines.append('        MemoryLocation:: DEVICE')
    lines.append('    );')
    lines.append('')
    lines.append(f'    std::vector<value_t> input_vals({n_in});')
    for i, val in enumerate(input_np. ravel()):
        lines.append(f'    input_vals[{i}] = static_cast<value_t>({fmt_float(val)});')
    lines.append('    input = input_vals;')
    lines.append('')

    # Run upsample
    lines.append(f'    // Run upsample with {mode} mode')
    lines.append(f'    Tensor<value_t> result = math::upsample<value_t>(')
    lines.append(f'        input, {stride}, {mode_cpp}')
    lines.append('    );')
    lines.append('')

    # Expected output
    lines.append(f'    std::vector<value_t> expected({n_out});')
    for i, val in enumerate(output_np.ravel()):
        lines.append(f'    expected[{i}] = static_cast<value_t>({fmt_float(val)});')
    lines.append('')

    # Copy result to host
    lines.append('    std::vector<value_t> host(expected.size());')
    lines.append('    g_sycl_queue.memcpy(host.data(), result.m_p_data.get(),')
    lines.append('                        host.size() * sizeof(value_t)).wait();')
    lines.append('')

    # Compare
    lines.append('    // Compare results')
    lines.append('    for (size_t i = 0; i < host.size(); ++i)')
    lines.append('    {')
    lines.append('        if constexpr (std::is_floating_point<value_t>::value)')
    lines.append('        {')
    lines.append('            EXPECT_NEAR(static_cast<double>(host[i]),')
    lines.append(f'                        static_cast<double>(expected[i]), {tolerance});')
    lines.append('        }')
    lines.append('        else')
    lines.append('        {')
    lines.append('            EXPECT_EQ(host[i], expected[i]);')
    lines.append('        }')
    lines.append('    }')
    lines.append('}')

    num_test += 1
    return '\n'.join(lines)

def create_typed_test_shape_check(testname, input_shape, stride, mode):
    """Generate a simple shape verification test."""
    channels = input_shape[-3]
    in_h = input_shape[-2]
    in_w = input_shape[-1]
    out_h = in_h * stride - (stride - 1)
    out_w = in_w * stride - (stride - 1)

    out_shape = list(input_shape[:-2]) + [out_h, out_w]

    mode_cpp = f'UpsampleMode::{mode}'

    lines = []
    lines.append('/**')
    lines.append(f' * @test TypedUpsample.{testname}')
    lines.append(f' * @brief Check output shape for {mode} mode')
    lines.append(' */')
    lines.append(f'TYPED_TEST(TypedUpsample, {testname})')
    lines.append('{')
    lines.append('    using value_t = TypeParam;')
    lines.append('')
    lines.append(f'    Tensor<value_t> input(')
    lines.append('        {' + ', '.join(str(d) for d in input_shape) + '},')
    lines.append('        MemoryLocation::DEVICE')
    lines.append('    );')
    lines.append('')
    lines.append(f'    Tensor<value_t> result = math::upsample<value_t>(')
    lines.append(f'        input, {stride}, {mode_cpp}')
    lines.append('    );')
    lines.append('')
    lines.append('    const std::vector<uint64_t>& result_shape = result.get_dimensions();')
    lines.append(f'    std::vector<uint64_t> expected_shape = {{')
    lines.append('        ' + ', '.join(str(d) for d in out_shape))
    lines.append('    };')
    lines.append('')
    lines.append('    EXPECT_EQ(result_shape, expected_shape);')
    lines.append('}')
    return '\n'.join(lines)
def make_view_strides_upsample_test(testname, owner_vals, offset,
                                   alias_shape, alias_strides,
                                   stride, mode,
                                   memloc='MemoryLocation::DEVICE'):
    """
    Test upsample with non-contiguous views.
    """
    from numpy. lib.stride_tricks import as_strided

    owner_np = np.asarray(owner_vals, dtype=np.float32)

    # Build alias view
    base = owner_np[offset:]
    byte_strides = tuple(int(s * owner_np.itemsize) for s in alias_strides)
    alias_np = as_strided(base, shape=tuple(alias_shape), strides=byte_strides)

    # Compute expected with reference implementation
    if mode == 'ZEROS':
        expected_np = upsample_zeros(alias_np. copy(), stride)
        mode_cpp = 'UpsampleMode::ZEROS'
    elif mode == 'NEAREST':
        expected_np = upsample_nearest(alias_np.copy(), stride)
        mode_cpp = 'UpsampleMode::NEAREST'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    flat_expected = expected_np.ravel()
    total = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f' * @test TypedUpsample.{testname}')
    lines.append(' * @brief Ensure math::upsample accepts non-contiguous/alias views')
    lines.append(f' *   Mode: {mode}')
    lines.append(' */')
    lines.append(f'TYPED_TEST(TypedUpsample, {testname})')
    lines.append('{')
    lines.append('    using value_t = TypeParam;')
    lines.append('')

    # Owner tensor
    lines.append(f'    Tensor<value_t> owner({{{len(owner_vals)}}}, {memloc});')
    owner_items = ', '.join('static_cast<value_t>(' + fmt_float(float(v)) + ')' for v in owner_vals)
    lines.append('    std::vector<value_t> owner_vals = {')
    lines.append(f'        {owner_items}')
    lines.append('    };')
    lines.append('    owner = owner_vals;')
    lines.append('')

    # Alias
    lines.append('    Tensor<value_t> alias(')
    lines.append('        owner,')
    lines.append(f'        std::vector<uint64_t>{{{offset}ull}},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(s)) + 'ull' for s in alias_shape) + '},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(d)) + 'ull' for d in alias_strides) + '}')
    lines.append('    );')
    lines.append('')

    # Run
    lines.append(f'    Tensor<value_t> result = math::upsample<value_t>(')
    lines.append(f'        alias, {stride}, {mode_cpp}')
    lines.append('    );')
    lines.append('')

    # Expected
    out_shape = expected_np.shape
    lines. append(f'    std::vector<value_t> expected({total});')
    for i, val in enumerate(flat_expected):
        lines.append(f'    expected[{i}] = static_cast<value_t>({fmt_float(val)});')
    lines.append('')

    # Copy and compare
    lines.append('    std::vector<value_t> host(expected.size());')
    lines.append('    g_sycl_queue.memcpy(host.data(), result.m_p_data. get(),')
    lines.append('                        host.size() * sizeof(value_t)).wait();')
    lines.append('')
    lines.append('    // compare element-wise')
    lines.append('    for (size_t i = 0; i < host.size(); ++i)')
    lines.append('    {')
    lines.append('        if constexpr (std::is_floating_point<value_t>::value)')
    lines.append('        {')
    lines.append('            EXPECT_NEAR(static_cast<double>(host[i]),')
    lines.append(f'                        static_cast<double>(expected[i]), {tolerance});')
    lines.append('        }')
    lines.append('        else')
    lines.append('        {')
    lines.append('            EXPECT_EQ(host[i], expected[i]);')
    lines.append('        }')
    lines.append('    }')

    lines.append('}')
    return '\n'.join(lines)

if __name__ == '__main__':
    blocks = []

    blocks.append("""/**
 * @file ut_upsample.cpp
 * @brief File generated by tests/math/upsample.py.
 * Tests for math::upsample function with ZEROS and NEAREST modes.
 */

#include <gtest/gtest.h>
#include "temper/Math.hpp"
#include "temper/SYCLQueue.hpp"

using namespace temper;
using namespace temper::math;

template<typename T>
class TypedUpsample : public ::testing:: Test {};

using UpsampleTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedUpsample, UpsampleTestTypes);
""")

    seed = 100

    # Shape check tests
    blocks.append(create_typed_test_shape_check(
        'shape_check_zeros_stride2', [3, 4, 4], 2, 'ZEROS'))
    blocks.append(create_typed_test_shape_check(
        'shape_check_nearest_stride3', [2, 5, 5], 3, 'NEAREST'))
    # ZEROS mode tests
    blocks.append(create_typed_test_upsample(
        'zeros_simple_2x2_stride2', [1, 2, 2], 2, 'ZEROS', seed))
    seed += 1
    blocks.append(create_typed_test_upsample(
        'zeros_3channels_3x3_stride2', [3, 3, 3], 2, 'ZEROS', seed))
    seed += 1
    blocks.append(create_typed_test_upsample(
        'zeros_multichannel_4x4_stride3', [2, 4, 4], 3, 'ZEROS', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'zeros_nonsquare_3x5_stride2', [1, 3, 5], 2, 'ZEROS', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'zeros_batch_2x3x4x4_stride2', [2, 3, 4, 4], 2, 'ZEROS', seed))
    seed += 1

    # NEAREST mode tests
    blocks. append(create_typed_test_upsample(
        'nearest_simple_2x2_stride2', [1, 2, 2], 2, 'NEAREST', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'nearest_3channels_3x3_stride2', [3, 3, 3], 2, 'NEAREST', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'nearest_multichannel_4x4_stride3', [2, 4, 4], 3, 'NEAREST', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'nearest_nonsquare_3x5_stride2', [1, 3, 5], 2, 'NEAREST', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'nearest_batch_2x3x4x4_stride2', [2, 3, 4, 4], 2, 'NEAREST', seed))
    seed += 1

    # Edge cases
    blocks.append(create_typed_test_upsample(
        'zeros_stride1_identity', [2, 3, 3], 1, 'ZEROS', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'nearest_stride1_identity', [2, 3, 3], 1, 'NEAREST', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'zeros_large_stride5', [1, 2, 2], 5, 'ZEROS', seed))
    seed += 1

    blocks.append(create_typed_test_upsample(
        'nearest_large_stride5', [1, 2, 2], 5, 'NEAREST', seed))
    seed += 1

        # View with weird strides tests
    # Extract every other element to form (2, 2, 2) with non-contiguous layout
    owner = np.arange(50, dtype=np.float32)
    offset = 0
    alias_shape = (2, 2, 2)
    alias_strides = (16, 8, 2)  # Skip elements in all dimensions

    blocks.append(make_view_strides_upsample_test(
        'zeros_view_with_weird_strides',
        owner, offset, alias_shape, alias_strides,
        2, 'ZEROS',
        memloc='MemoryLocation::DEVICE'
    ))

    blocks.append(make_view_strides_upsample_test(
        'nearest_view_with_weird_strides',
        owner, offset, alias_shape, alias_strides,
        2, 'NEAREST',
        memloc='MemoryLocation:: DEVICE'
    ))

    # Another test:  extract with larger gaps
    owner2 = np.arange(100, dtype=np.float32)
    offset2 = 5
    alias_shape2 = (1, 3, 3)
    alias_strides2 = (27, 9, 3)  # Non-unit strides

    blocks. append(make_view_strides_upsample_test(
        'zeros_view_nonunit_strides',
        owner2, offset2, alias_shape2, alias_strides2,
        3, 'ZEROS',
        memloc='MemoryLocation::DEVICE'
    ))

    blocks.append(make_view_strides_upsample_test(
        'nearest_view_nonunit_strides',
        owner2, offset2, alias_shape2, alias_strides2,
        3, 'NEAREST',
        memloc='MemoryLocation:: DEVICE'
    ))

    print('\n\n'.join(blocks))