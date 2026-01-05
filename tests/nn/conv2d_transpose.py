#!/usr/bin/env python3
"""
Generator for nn:: conv2d_transpose tests with typed test support.
Uses PyTorch's conv_transpose2d for reference.
"""
from functools import reduce
import operator
import numpy as np
import torch
import torch.nn. functional as F
import math
from numpy.lib.stride_tricks import as_strided

tolerance = 1e-4
num_test = 0
np.random.seed(550)

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


def create_typed_test_conv2d_transpose(testname, in_channels, out_channels,
                                      in_h, in_w, kernel_h, kernel_w,
                                      stride, padding, output_padding, seed):
    global num_test
    np.random.seed(seed)

    # Generate random input and kernel
    input_shape = (in_channels, in_h, in_w)
    kernel_shape = (in_channels, out_channels, kernel_h, kernel_w)

    input_np = np.random.randn(*input_shape).astype(np.float32)
    kernel_np = np.random. randn(*kernel_shape).astype(np.float32)

    # Compute expected using PyTorch
    input_torch = torch.from_numpy(input_np).unsqueeze(0)
    kernel_torch = torch.from_numpy(kernel_np)

    output_torch = F.conv_transpose2d(input_torch, kernel_torch,
                                     stride=stride, padding=padding,
                                     output_padding=output_padding)
    output_np = output_torch.squeeze(0).numpy()

    out_shape = output_np.shape
    n_out = prod(output_np.shape)
    n_in = prod(input_shape)
    n_kernel = prod(kernel_shape)

    lines = []
    lines.append('/**')
    lines.append(f' * @test TypedConv2DTranspose. {testname}')
    lines.append(f' * @brief Conv2d transpose test: ')
    lines.append(f' *   input:  {input_shape}, kernel: {kernel_shape}')
    lines.append(f' *   stride: {stride}, padding: {padding}, output_padding: {output_padding}')
    lines.append(' */')
    lines.append(f'TYPED_TEST(TypedConv2DTranspose, {testname})')
    lines.append('{')
    lines.append('    using value_t = TypeParam;')
    lines.append('')

    # Input tensor
    lines.append(f'    Tensor<value_t> input(')
    lines.append('        {' + ', '.join(str(d) for d in input_shape) + '},')
    lines.append('        MemoryLocation::DEVICE')
    lines.append('    );')
    lines.append('')
    lines.append(f'    std::vector<value_t> input_vals({n_in});')
    for i, val in enumerate(input_np. ravel()):
        lines.append(f'    input_vals[{i}] = static_cast<value_t>({fmt_float(val)});')
    lines.append('    input = input_vals;')
    lines.append('')

    # Kernel tensor
    lines.append(f'    Tensor<value_t> kernel(')
    lines.append('        {' + ', '.join(str(d) for d in kernel_shape) + '},')
    lines.append('        MemoryLocation::DEVICE')
    lines.append('    );')
    lines.append('')
    lines.append(f'    std::vector<value_t> kernel_vals({n_kernel});')
    for i, val in enumerate(kernel_np.ravel()):
        lines.append(f'    kernel_vals[{i}] = static_cast<value_t>({fmt_float(val)});')
    lines.append('    kernel = kernel_vals;')
    lines.append('')

    # Run conv2d_transpose
    pad_pair = f'{{{padding[0]}, {padding[1]}}}' if isinstance(padding, tuple) else f'{{{padding}, {padding}}}'
    opad_pair = f'{{{output_padding[0]}, {output_padding[1]}}}' if isinstance(output_padding, tuple) else f'{{{output_padding}, {output_padding}}}'
    lines.append(f'    // Run conv2d_transpose')
    lines.append(f'    Tensor<value_t> result = nn:: conv2d_transpose<value_t>(')
    lines.append(f'        input, kernel, {stride}, {pad_pair}, {opad_pair}')
    lines.append('    );')
    lines.append('')

    # Expected output
    lines.append(f'    std::vector<value_t> expected({n_out});')
    for i, val in enumerate(output_np.ravel()):
        lines.append(f'    expected[{i}] = static_cast<value_t>({fmt_float(val)});')
    lines.append('')

    # Copy result to host
    lines.append('    std::vector<value_t> host(expected.size());')
    lines.append('    g_sycl_queue.memcpy(host.data(), result.m_p_data. get(),')
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


def make_view_strides_conv2d_transpose_test(testname,
                                           input_owner_vals, input_offset,
                                           input_alias_shape, input_alias_strides,
                                           kernel_owner_vals, kernel_offset,
                                           kernel_alias_shape, kernel_alias_strides,
                                           stride, padding, output_padding,
                                           memloc='MemoryLocation::DEVICE'):
    """
    Test conv2d_transpose with non-contiguous views.
    """
    input_owner_np = np.asarray(input_owner_vals, dtype=np.float32)
    kernel_owner_np = np.asarray(kernel_owner_vals, dtype=np.float32)

    def build_alias(owner_np, offset, strides_elems, alias_shape):
        base = owner_np[offset:]
        byte_strides = tuple(int(s * owner_np.itemsize) for s in strides_elems)
        return as_strided(base, shape=alias_shape, strides=byte_strides)

    input_alias_np = build_alias(input_owner_np, input_offset,
                                 input_alias_strides, tuple(input_alias_shape))
    kernel_alias_np = build_alias(kernel_owner_np, kernel_offset,
                                  kernel_alias_strides, tuple(kernel_alias_shape))

    # Compute expected with PyTorch
    input_torch = torch.from_numpy(input_alias_np. copy()).unsqueeze(0)
    kernel_torch = torch.from_numpy(kernel_alias_np. copy())

    output_torch = F.conv_transpose2d(input_torch, kernel_torch,
                                     stride=stride, padding=padding,
                                     output_padding=output_padding)
    expected_np = output_torch.squeeze(0).numpy()
    flat_expected = expected_np.ravel()
    total = flat_expected.size

    lines = []
    lines. append('/**')
    lines.append(f' * @test TypedConv2DTranspose. {testname}')
    lines.append(' * @brief Ensure nn::conv2d_transpose accepts non-contiguous/alias views')
    lines.append(' */')
    lines.append(f'TYPED_TEST(TypedConv2DTranspose, {testname})')
    lines.append('{')
    lines.append('    using value_t = TypeParam;')
    lines.append('')

    # Input owner
    lines.append(f'    Tensor<value_t> input_owner({{{len(input_owner_vals)}}}, {memloc});')
    lines.append(f'    std::vector<value_t> input_owner_vals({len(input_owner_vals)});')
    for i, v in enumerate(input_owner_vals):
        lines.append(f'    input_owner_vals[{i}] = static_cast<value_t>({fmt_float(float(v))});')
    lines.append('    input_owner = input_owner_vals;')
    lines.append('')

    # Input alias
    lines.append('    Tensor<value_t> input_alias(')
    lines.append('        input_owner,')
    lines.append(f'        std::vector<uint64_t>{{{input_offset}ull}},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(s)) + 'ull' for s in input_alias_shape) + '},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(d)) + 'ull' for d in input_alias_strides) + '}')
    lines.append('    );')
    lines.append('')

    # Kernel owner
    lines.append(f'    Tensor<value_t> kernel_owner({{{len(kernel_owner_vals)}}}, {memloc});')
    lines.append(f'    std::vector<value_t> kernel_owner_vals({len(kernel_owner_vals)});')
    for i, v in enumerate(kernel_owner_vals):
        lines.append(f'    kernel_owner_vals[{i}] = static_cast<value_t>({fmt_float(float(v))});')
    lines.append('    kernel_owner = kernel_owner_vals;')
    lines.append('')

    # Kernel alias
    lines.append('    Tensor<value_t> kernel_alias(')
    lines.append('        kernel_owner,')
    lines.append(f'        std::vector<uint64_t>{{{kernel_offset}ull}},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(s)) + 'ull' for s in kernel_alias_shape) + '},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(d)) + 'ull' for d in kernel_alias_strides) + '}')
    lines.append('    );')
    lines.append('')

    # Run
    pad_pair = f'{{{padding[0]}, {padding[1]}}}' if isinstance(padding, tuple) else f'{{{padding}, {padding}}}'
    opad_pair = f'{{{output_padding[0]}, {output_padding[1]}}}' if isinstance(output_padding, tuple) else f'{{{output_padding}, {output_padding}}}'
    lines. append(f'    Tensor<value_t> result = nn::conv2d_transpose<value_t>(')
    lines.append(f'        input_alias, kernel_alias, {stride}, {pad_pair}, {opad_pair}')
    lines.append('    );')
    lines.append('')

    # Expected
    lines.append(f'    std::vector<value_t> expected({total});')
    for i, val in enumerate(flat_expected):
        lines.append(f'    expected[{i}] = static_cast<value_t>({fmt_float(val)});')
    lines.append('')

    # Copy and compare
    lines.append('    std::vector<value_t> host(expected.size());')
    lines.append('    g_sycl_queue.memcpy(host.data(), result.m_p_data.get(),')
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
 * @file ut_conv2d_transpose.cpp
 * @brief File generated by tests/nn/conv2d_transpose.py.
 * Tests for nn::conv2d_transpose function.
 */

#include <gtest/gtest.h>
#include "temper/NN.hpp"
#include "temper/SYCLQueue.hpp"

using namespace temper;

template<typename T>
class TypedConv2DTranspose : public ::testing:: Test {};

using Conv2DTransposeTestTypes = :: testing::Types<float>;
TYPED_TEST_SUITE(TypedConv2DTranspose, Conv2DTransposeTestTypes);
""")

    seed = 200

    # Basic configurations
    blocks.append(create_typed_test_conv2d_transpose(
        'basic_s2_nopad', 1, 1, 3, 3, 3, 3, 2, (0, 0), (0, 0), seed))
    seed += 1

    blocks.append(create_typed_test_conv2d_transpose(
        'basic_s2_pad1', 1, 1, 3, 3, 3, 3, 2, (1, 1), (0, 0), seed))
    seed += 1

    blocks. append(create_typed_test_conv2d_transpose(
        'basic_s2_4x4', 1, 1, 4, 4, 3, 3, 2, (0, 0), (0, 0), seed))
    seed += 1

    # Multiple channels
    blocks.append(create_typed_test_conv2d_transpose(
        'multi_in_channels', 2, 1, 3, 3, 3, 3, 2, (0, 0), (0, 0), seed))
    seed += 1

    blocks.append(create_typed_test_conv2d_transpose(
        'multi_out_channels', 1, 3, 3, 3, 3, 3, 2, (0, 0), (0, 0), seed))
    seed += 1

    blocks.append(create_typed_test_conv2d_transpose(
        'multi_both_channels', 2, 3, 4, 4, 3, 3, 2, (1, 1), (0, 0), seed))
    seed += 1

    # Stride = 1 (no upsampling)
    blocks.append(create_typed_test_conv2d_transpose(
        's1_nopad', 1, 1, 5, 5, 3, 3, 1, (0, 0), (0, 0), seed))
    seed += 1

    blocks.append(create_typed_test_conv2d_transpose(
        's1_pad1', 2, 2, 5, 5, 3, 3, 1, (1, 1), (0, 0), seed))
    seed += 1

    # Non-square kernels
    blocks.append(create_typed_test_conv2d_transpose(
        'nonsquare_kernel_3x2', 1, 1, 4, 4, 3, 2, 2, (0, 0), (0, 0), seed))
    seed += 1

    blocks.append(create_typed_test_conv2d_transpose(
        'nonsquare_kernel_2x3', 2, 2, 3, 5, 2, 3, 2, (1, 0), (0, 0), seed))
    seed += 1

    # Asymmetric padding
    blocks.append(create_typed_test_conv2d_transpose(
        'asym_padding', 1, 1, 3, 3, 3, 3, 2, (1, 2), (0, 0), seed))
    seed += 1

    # With output_padding
    blocks.append(create_typed_test_conv2d_transpose(
        'output_padding_1_1', 1, 1, 3, 3, 3, 3, 2, (0, 0), (1, 1), seed))
    seed += 1

    blocks. append(create_typed_test_conv2d_transpose(
        'output_padding_1_0', 2, 2, 4, 4, 3, 3, 2, (1, 1), (1, 0), seed))
    seed += 1

    # Larger stride
    blocks.append(create_typed_test_conv2d_transpose(
        'large_stride_s3', 1, 1, 2, 2, 3, 3, 3, (0, 0), (0, 0), seed))
    seed += 1

    # View with weird strides test
    input_owner = np.arange(30, dtype=np.float32)
    input_offset = 0
    input_alias_shape = (1, 2, 2)
    input_alias_strides = (8, 4, 2)

    kernel_owner = np.arange(30, dtype=np.float32)
    kernel_offset = 0
    kernel_alias_shape = (1, 1, 2, 2)
    kernel_alias_strides = (8, 8, 4, 2)

    blocks.append(make_view_strides_conv2d_transpose_test(
        'view_weird_strides',
        input_owner, input_offset, input_alias_shape, input_alias_strides,
        kernel_owner, kernel_offset, kernel_alias_shape, kernel_alias_strides,
        2, (0, 0), (0, 0),
        memloc='MemoryLocation::DEVICE'
    ))

    print('\n\n'. join(blocks))