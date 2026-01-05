/**
 * @file NN.hpp
 * @brief Neural network utilities.
 *
 * Provides tensor operations and helpers
 * commonly used in neural network models.
 */

#ifndef TEMPER_NN_HPP
#define TEMPER_NN_HPP

#include "Tensor.hpp"

namespace temper::nn
{

/**
 * @brief 2D convolution operation.
 *
 * Performs 2D convolution between input and kernel tensors with
 * broadcasting support.  The last two dimensions are treated as the
 * spatial dimensions (height, width).
 *
 * Input shape:  (..., in_channels, height, width)
 * Kernel shape: (..., out_channels, in_channels, kernel_h, kernel_w)
 * Output shape: (..., out_channels, out_h, out_w)
 *
 * where out_h = (height + 2*pad_h - kernel_h) / stride + 1
 *       out_w = (width + 2*pad_w - kernel_w) / stride + 1
 *
 * Leading dimensions (...) are broadcasted between input and kernel.
 *
 * @param input Input tensor (rank >= 3).
 * @param kernel Kernel/filter tensor (rank >= 4).
 * @param stride Stride for the convolution (default: 1).
 * @param padding Padding applied to (height, width) (default: {0, 0}).
 * @return Tensor<value_t> Result of convolution.
 */
template <typename value_t>
Tensor<value_t> conv2d(const Tensor<value_t>& input,
    const Tensor<value_t>& kernel,
    uint64_t stride = 1,
    std::pair<uint64_t, uint64_t> padding = {0, 0});
/// \cond
extern template Tensor<float> conv2d<float>
    (const Tensor<float>&, const Tensor<float>&, uint64_t,
     std::pair<uint64_t, uint64_t>);
/// \endcond

/**
 * @brief 2D transposed convolution (also called deconvolution).
 *
 * Performs transposed 2D convolution, which upsamples the input by inserting
 * zeros (controlled by stride) and then convolving.  This is the backward pass
 * of a regular convolution with respect to its input.
 *
 * Input shape:   (... , in_channels, height, width)
 * Kernel shape: (..., in_channels, out_channels, kernel_h, kernel_w)
 *               Note: kernel is (in_channels, out_channels) for transpose conv
 * Output shape: (... , out_channels, out_h, out_w)
 *
 * where:
 *   out_h = (height - 1) * stride + kernel_h - 2*pad_h + output_padding_h
 *   out_w = (width - 1) * stride + kernel_w - 2*pad_w + output_padding_w
 *
 * For the common case with output_padding=0:
 *   out_h = (height - 1) * stride + kernel_h - 2*pad_h
 *   out_w = (width - 1) * stride + kernel_w - 2*pad_w
 *
 * Leading dimensions (...) are broadcasted between input and kernel.
 *
 * @param input Input tensor (rank >= 3).
 * @param kernel Kernel/filter tensor (rank >= 4).
 *        Note: shape is (in_channels, out_channels, kH, kW)
 * @param stride Stride for the transposed convolution (default: 1).
 * @param padding Padding applied to input before upsampling (default: {0, 0}).
 * @param output_padding Additional size added to output (default: {0, 0}).
 * @return Tensor<value_t> Result of transposed convolution.
 */
template <typename value_t>
Tensor<value_t> conv2d_transpose(
    const Tensor<value_t>& input,
    const Tensor<value_t>& kernel,
    uint64_t stride = 1,
    std::pair<uint64_t, uint64_t> padding = {0, 0},
    std::pair<uint64_t, uint64_t> output_padding = {0, 0});
/// \cond
extern template Tensor<float> conv2d_transpose<float>
    (const Tensor<float>&, const Tensor<float>&, uint64_t,
     std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>);
/// \endcond

 /* todo
    all activation functions
    all initialization functions
    pooling
    normalizations
*/
} // namespace temper::nn

#endif // TEMPER_NN_HPP
