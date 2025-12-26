#!/usr/bin/env python3
"""
Generator for nn:: conv2d tests (random-grid + broadcasting + safe view-with-weird-strides).
Uses true mathematical convolution.
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
    """
    Return access expression for arbitrary variable name in row-major order.
    Examples:
      - shape == (6,) -> 'varname[flat_index]'
      - shape == (2,3) -> 'varname[1][1]'
      - scalar -> 'varname[0]'
    """
    total = prod(shape)
    if total == 1:
        return f'{varname}[0]'
    if len(shape) == 1:
        return f'{varname}[{flat_index}]'
    indices = np.unravel_index(flat_index, shape)
    return varname + ''. join(f'[{int(idx)}]' for idx in indices)


def create_test_conv2d(in_channels, out_channels, in_h, in_w,
                       kernel_h, kernel_w, stride, padding, seed):
    global num_test
    torch.manual_seed(seed)

    # Generate random input and kernel
    input_shape = (in_channels, in_h, in_w)
    kernel_shape = (out_channels, in_channels, kernel_h, kernel_w)

    input_np = np.random.randn(*input_shape).astype(np.float32)
    kernel_np = np.random.randn(*kernel_shape).astype(np.float32)

    # Compute expected using PyTorch with TRUE CONVOLUTION (flip kernel)
    input_torch = torch.from_numpy(input_np).unsqueeze(0)  # Add batch
    kernel_torch = torch.from_numpy(kernel_np)


    kernel_flipped = torch.flip(kernel_torch, dims=[-2, -1])

    output_torch = F.conv2d(input_torch, kernel_flipped,
                           stride=stride, padding=padding)
    output_np = output_torch.squeeze(0).numpy()  # Remove batch

    out_shape = output_np.shape
    n = prod(output_np.shape)

    pad_str = f'{padding[0]}_{padding[1]}' if isinstance(padding, tuple) else f'{padding}_{padding}'
    testname = f'conv2d_random_test_ic{in_channels}_oc{out_channels}_k{kernel_h}x{kernel_w}_s{stride}_p{pad_str}'

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CONV2D. {testname}')
    lines.append(f'  * @brief Randomly generated conv2d test (true convolution):')
    lines.append(f'  *   input: {input_shape}, kernel: {kernel_shape}')
    lines.append(f'  *   stride: {stride}, padding: {padding}')
    lines.append('  *')
    lines.append('  * The test is generated from the script, expecting values from PyTorch')
    lines.append('  * with kernel flipping to match mathematical convolution.')
    lines.append('  */')
    lines.append(f'TEST(CONV2D, {testname})')
    lines.append('{')
    num_test += 1

    # Input tensor
    lines.append(f'    Tensor<float> input({{{", ".join(str(d) for d in input_shape)}}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('input', input_np. ravel(), input_shape))

    lines.append(f'    Tensor<float> kernel({{{", ".join(str(d) for d in kernel_shape)}}});')
    lines.append('    ' + cpp_vec_or_scalar_assignment('kernel', kernel_np.ravel(), kernel_shape))

    # Expected result
    lines.append(f'    Tensor<float> expected_result({{{", ".join(str(d) for d in out_shape)}}});')
    lines.append(f'    expected_result = std::vector<float>{flat_initializer(output_np. ravel())};')
    # Run conv2d
    pad_pair = f'{{{padding[0]}, {padding[1]}}}' if isinstance(padding, tuple) else f'{{{padding}, {padding}}}'
    lines.append(f'    // run Temper nn::conv2d')
    lines.append(f'    Tensor<float> result = nn::conv2d(input, kernel, {stride}, {pad_pair});')
    # Check values
    lines.append('    // check values')
    for i in range(n):
        result_access = cpp_index_access('result', i, out_shape)
        expected_access = cpp_index_access('expected_result', i, out_shape)
        lines.append(f'    EXPECT_NEAR({result_access}, {expected_access}, {tolerance});')
    lines.append('}')
    return '\n'.join(lines)


def make_view_strides_conv2d_test(testname,
                                  input_owner_vals, input_offset,
                                  input_alias_shape, input_alias_strides,
                                  kernel_owner_vals, kernel_offset,
                                  kernel_alias_shape, kernel_alias_strides,
                                  stride, padding,
                                  memloc='MemoryLocation:: DEVICE'):
    """
    Test conv2d with non-contiguous views.
    """
    input_owner_np = np.asarray(input_owner_vals, dtype=np.float32)
    kernel_owner_np = np.asarray(kernel_owner_vals, dtype=np.float32)

    def build_alias(owner_np, offset, strides_elems, alias_shape):
        base = owner_np[offset:]
        byte_strides = tuple(int(s * owner_np. itemsize) for s in strides_elems)
        return as_strided(base, shape=alias_shape, strides=byte_strides)

    input_alias_np = build_alias(input_owner_np, input_offset,
                                 input_alias_strides, tuple(input_alias_shape))
    kernel_alias_np = build_alias(kernel_owner_np, kernel_offset,
                                  kernel_alias_strides, tuple(kernel_alias_shape))

    # Compute expected with PyTorch
    input_torch = torch.from_numpy(input_alias_np. copy()).unsqueeze(0)
    kernel_torch = torch.from_numpy(kernel_alias_np. copy())

    # FLIP kernel for true convolution
    kernel_flipped = torch.flip(kernel_torch, dims=[-2, -1])

    output_torch = F.conv2d(input_torch, kernel_flipped,
                           stride=stride, padding=padding)
    expected_np = output_torch.squeeze(0).numpy()
    flat_expected = expected_np.ravel()
    total = flat_expected.size

    lines = []
    lines.append('/**')
    lines.append(f'  * @test CONV2D.{testname}')
    lines.append('  * @brief Ensure nn::conv2d accepts non-contiguous/alias views')
    lines.append('  */')
    lines.append(f'TEST(CONV2D, {testname})')
    lines.append('{')

    # Input owner
    lines.append(f'    Tensor<float> input_owner({{{len(input_owner_vals)}}}, {memloc});')
    input_items = ', '.join(fmt_float(float(v)) for v in input_owner_vals)
    lines.append('    std::vector<float> input_owner_vals = {')
    lines.append(f'        {input_items}')
    lines.append('    };')
    lines.append('    input_owner = input_owner_vals;')
    lines.append('')

    # Input alias
    lines.append('    Tensor<float> input_alias(')
    lines.append('        input_owner,')
    lines.append(f'        std::vector<uint64_t>{{{input_offset}ull}},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(s)) + 'ull' for s in input_alias_shape) + '},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(d)) + 'ull' for d in input_alias_strides) + '}')
    lines.append('    );')
    lines.append('')

    # Kernel owner
    lines.append(f'    Tensor<float> kernel_owner({{{len(kernel_owner_vals)}}}, {memloc});')
    kernel_items = ', '.join(fmt_float(float(v)) for v in kernel_owner_vals)
    lines.append('    std::vector<float> kernel_owner_vals = {')
    lines.append(f'        {kernel_items}')
    lines.append('    };')
    lines.append('    kernel_owner = kernel_owner_vals;')
    lines.append('')

    # Kernel alias
    lines.append('    Tensor<float> kernel_alias(')
    lines.append('        kernel_owner,')
    lines.append(f'        std::vector<uint64_t>{{{kernel_offset}ull}},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(s)) + 'ull' for s in kernel_alias_shape) + '},')
    lines.append('        std::vector<uint64_t>{' + ', '.join(str(int(d)) + 'ull' for d in kernel_alias_strides) + '}')
    lines.append('    );')
    lines.append('')

    # Run
    pad_pair = f'{{{padding[0]}, {padding[1]}}}' if isinstance(padding, tuple) else f'{{{padding}, {padding}}}'
    lines.append(f'    Tensor<float> result = nn::conv2d(input_alias, kernel_alias, {stride}, {pad_pair});')
    lines.append('')

    # Expected
    out_shape = expected_np.shape
    lines. append(f'    Tensor<float> expected_result({{{", ".join(str(d) for d in out_shape)}}});')
    lines.append(f'    expected_result = std:: vector<float>{flat_initializer(flat_expected)};')
    lines.append('')

    # Check
    lines.append('    // compare element-wise (unroll to report precise index on failure)')
    for i in range(total):
        result_access = cpp_index_access('result', i, out_shape)
        expected_access = cpp_index_access('expected_result', i, out_shape)
        lines.append(f'    EXPECT_NEAR({result_access}, {expected_access}, {tolerance});')

    lines.append('}')
    return '\n'.join(lines)


if __name__ == '__main__':
    blocks = []

    blocks.append("""/**
  * @file conv2d.cpp
  * @brief File generated by tests/nn/conv2d. py.
  * Tests use true mathematical convolution (with kernel flipping).
  */""")

    # Random tests with various configurations
    seed = 42

    # Basic configurations
    blocks.append(create_test_conv2d(1, 1, 5, 5, 3, 3, 1, (0, 0), seed))
    seed += 1
    blocks.append(create_test_conv2d(1, 1, 5, 5, 3, 3, 1, (1, 1), seed))
    seed += 1
    blocks. append(create_test_conv2d(1, 1, 6, 6, 3, 3, 2, (0, 0), seed))
    seed += 1

    # Multiple channels
    blocks.append(create_test_conv2d(3, 1, 5, 5, 3, 3, 1, (0, 0), seed))
    seed += 1
    blocks.append(create_test_conv2d(1, 2, 5, 5, 3, 3, 1, (0, 0), seed))
    seed += 1
    blocks.append(create_test_conv2d(3, 2, 5, 5, 3, 3, 1, (1, 1), seed))
    seed += 1

    # Non-square kernels
    blocks. append(create_test_conv2d(1, 1, 5, 7, 3, 2, 1, (0, 0), seed))
    seed += 1
    blocks.append(create_test_conv2d(2, 2, 7, 5, 2, 3, 1, (1, 0), seed))
    seed += 1

    # Asymmetric padding
    blocks.append(create_test_conv2d(1, 1, 5, 5, 3, 3, 1, (1, 2), seed))
    seed += 1

    # 1x1 convolution
    blocks.append(create_test_conv2d(3, 4, 8, 8, 1, 1, 1, (0, 0), seed))
    seed += 1

    # View with weird strides test
    # Input:  extract every other element to form (1, 2, 2)
    input_owner = np.arange(20, dtype=np.float32)
    input_offset = 0
    input_alias_shape = (1, 2, 2)
    input_alias_strides = (8, 4, 2)  # Skip elements

    # Kernel: extract to form (1, 1, 2, 2)
    kernel_owner = np.arange(20, dtype=np.float32)
    kernel_offset = 0
    kernel_alias_shape = (1, 1, 2, 2)
    kernel_alias_strides = (8, 8, 4, 2)

    blocks.append(make_view_strides_conv2d_test(
        'conv2d_view_with_weird_strides',
        input_owner, input_offset, input_alias_shape, input_alias_strides,
        kernel_owner, kernel_offset, kernel_alias_shape, kernel_alias_strides,
        1, (0, 0),
        memloc='MemoryLocation:: DEVICE'
    ))

    print('\n\n'. join(blocks))