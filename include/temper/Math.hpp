/**
 * @file Math.hpp
 * @brief Mathematical operations for tensors.
 *
 * Provides general-purpose mathematical functionality used across the library.
 */

#ifndef TEMPER_MATH_HPP
#define TEMPER_MATH_HPP

#include "Tensor.hpp"

namespace temper::math
{

/**
 * @brief Matrix multiplication between two tensors.
 *
 * Performs a batched or unbatched matrix multiplication between
 * @p first and @p second. Supports the following cases:
 * - Vector × Vector → scalar
 * - Vector × Matrix → vector
 * - Matrix × Vector → vector
 * - Matrix × Matrix → matrix
 * - Batched matrices with broadcasting on batch dimensions
 *
 * The tensors must satisfy the shape constraints:
 * - Last dimension of @p first matches second-to-last of @p second
 * - Batch dimensions are broadcastable
 *
 * The resulting tensor shape is determined according to standard
 * linear algebra rules, with batch dimensions broadcasted if needed.
 *
 * @param first Left-hand side tensor.
 * @param second Right-hand side tensor.
 * @return Tensor<float_t> Result of the matrix multiplication.
 *
 * @throws std::invalid_argument if:
 * - The last dimension of @p first does not match the second-to-last of @p second.
 * - Batch dimensions are not broadcastable.
 * @throws std::overflow_error if:
 * - The resulting tensor would exceed the maximum size representable by uint64_t.
 * @throws std::runtime_error if:
 * - A numeric error occurs during computation (e.g., non-finite values produced).
 * @throws std::bad_alloc if required device memory cannot be allocated.

 */
template <typename float_t>
Tensor<float_t> matmul(const Tensor<float_t> & first,
                        const Tensor<float_t> & second);
/// Explicit instantiation of matmul for float
extern template Tensor<float> matmul<float>
	(const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Reshape a tensor (free function wrapper).
 *
 * Returns a reshaped copy of the input tensor with new dimensions.
 * Unlike the member function, this free function first clones the tensor,
 * so the original tensor is left unmodified.
 *
 * @param tensor Input tensor.
 * @param new_dimensions New shape for the tensor.
 * @return Tensor<float_t> A new tensor with the specified shape.
 *
 * @throws std::invalid_argument If:
 * - @p new_dimensions is empty
 * - any entry in @p new_dimensions is zero
 * - the product of @p new_dimensions differs from the total element count
 *   of the input
 * @throws std::overflow_error If the product of @p new_dimensions
 * would overflow uint64_t.
 */
template <typename float_t>
Tensor<float_t> reshape(const Tensor<float_t> & tensor,
                        const std::vector<uint64_t>& new_dimensions);
/// Explicit instantiation of reshape for float
extern template Tensor<float> reshape<float>
    (const Tensor<float>&, const std::vector<uint64_t>&);

/**
 * @brief Sort tensor elements (free function wrapper).
 *
 * Returns a sorted copy of the input tensor, either flattened (axis = -1)
 * or independently sorted along a single axis. The input tensor is not
 * modified.
 *
 * @param tensor Input tensor.
 * @param axis Axis to sort along, -1 = flatten, otherwise 0..rank-1.
 * @return Tensor<float_t> A new tensor containing the sorted data.
 *
 * @throws std::invalid_argument If @p axis is out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 */
template <typename float_t>
Tensor<float_t> sort(const Tensor<float_t> & tensor, int64_t axis = -1);
/// Explicit instantiation of sort for float
extern template Tensor<float> sort<float>
    (const Tensor<float>&, int64_t);

/**
 * @brief Compute the sum of tensor elements (free function wrapper).
 *
 * Returns a new tensor containing the sums of @p tensor along the specified
 * axis. Delegates to `Tensor::sum`.
 *
 * @param tensor Input tensor.
 * @param axis Axis to sum along, -1 = flatten (sum all elements),
 * otherwise 0..rank-1.
 * @return Tensor<float_t> A new tensor containing the sums.
 *
 * @throws std::invalid_argument If axis is not -1 and out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or the results.
 */
template <typename float_t>
Tensor<float_t> sum(const Tensor<float_t> & tensor, int64_t axis = -1);
/// Explicit instantiation of sum for float
extern template Tensor<float> sum<float>
    (const Tensor<float>&, int64_t);

/**
 * @brief Compute the cumulative sum of tensor elements (free function wrapper).
 *
 * Returns a new tensor containing the cumulative sums of @p tensor along
 * the specified axis. Delegates to `Tensor::cumsum`.
 *
 * @param tensor Input tensor.
 * @param axis Axis to scan along, -1 = flatten (treat as 1D and scan
 * all elements), otherwise 0..rank-1.
 * @return Tensor<float_t> A new tensor containing the cumulative sums.
 *
 * @throws std::invalid_argument If axis is not -1 and is out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or the results.
 */
template <typename float_t>
Tensor<float_t> cumsum(const Tensor<float_t> & tensor, int64_t axis = -1);
/// Explicit instantiation of cumsum for float
extern template Tensor<float> cumsum<float>
    (const Tensor<float>&, int64_t);

/**
 * @brief Transpose a tensor (free function wrapper, full reversal).
 *
 * Returns a new tensor with all axes reversed, leaving the input tensor
 * unmodified. Delegates to `Tensor::transpose()`.
 *
 * @param tensor Input tensor.
 * @return Tensor<float_t> A new tensor view with reversed axes.
 *
 * @throws std::runtime_error If the tensor is empty (rank 0).
 */
template<typename float_t>
Tensor<float_t> transpose(const Tensor<float_t> & tensor);
/// Explicit instantiation of transpose() for float
extern template Tensor<float> transpose<float>(const Tensor<float>&);

/**
 * @brief Transpose a tensor with a custom axis order (free function wrapper).
 *
 * Returns a new tensor with its axes rearranged according to @p axes,
 * leaving the input tensor unmodified. Delegates to
 * `Tensor::transpose(const std::vector<uint64_t>&)`.
 *
 * @param tensor Input tensor.
 * @param axes Vector specifying the new order of axes. Must be a permutation
 * of [0..rank-1].
 * @return Tensor<float_t> A new tensor view with permuted axes.
 *
 * @throws std::invalid_argument If `axes.size()` != rank or if `axes` is not
 * a valid permutation.
 */
template<typename float_t>
Tensor<float_t> transpose(const Tensor<float_t> & tensor,
                        const std::vector<uint64_t> & axes);
/// Explicit instantiation of transpose(axes) for float
extern template Tensor<float> transpose<float>
    (const Tensor<float>&, const std::vector<uint64_t>&);

/**
 * @brief Pad the last two dimensions (height, width) with a constant.
 *
 * @param tensor Input tensor (must have rank >= 2).
 * @param pad_top Rows to add before the top.
 * @param pad_bottom Rows to add after the bottom.
 * @param pad_left Columns to add before the left.
 * @param pad_right Columns to add after the right.
 * @param pad_value Value used to fill padded elements.
 * @return Tensor<float_t> New tensor with requested padding.
 *
 * @throws std::invalid_argument If tensor is empty or rank < 2.
 * @throws std::overflow_error If shape/padding computations overflow.
 * @throws std::bad_alloc If device helper memory cannot be allocated.
 */
template<typename float_t>
Tensor<float_t> pad(const Tensor<float_t> & tensor,
                    uint64_t pad_top,
                    uint64_t pad_bottom,
                    uint64_t pad_left,
                    uint64_t pad_right,
                    float_t pad_value);
/// Explicit instantiation of pad for float
extern template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, uint64_t, uint64_t, float);

/**
 * @brief Symmetric pad helper: pad height/width on both sides.
 *
 * Convenience overload that pads both top/bottom by @p pad_height and
 * both left/right by @p pad_width.
 *
 * @param tensor Input tensor (rank >= 2).
 * @param pad_height Rows added to top and bottom.
 * @param pad_width  Columns added to left and right.
 * @param pad_value  Value to fill padded elements.
 * @return Tensor<float_t> New padded tensor.
 */
template<typename float_t>
Tensor<float_t> pad(const Tensor<float_t> & tensor,
                    uint64_t pad_height,
                    uint64_t pad_width,
                    float_t pad_value);
/// Explicit instantiation of pad (height, width) for float
extern template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, float);

/**
 * @brief Compute indices of maximum values along a specified axis.
 *
 * Returns a vector of indices corresponding to the maximum element of
 * each slice along @p axis. For axis = -1, returns a single global max
 * index. Ties are resolved by taking the first occurrence.
 *
 * @param tensor Input tensor of arbitrary rank.
 * @param axis Axis to reduce (-1 = flatten, otherwise 0..rank-1).
 * @return std::vector<uint64_t> Indices of maximum elements.
 *
 * @throws std::invalid_argument If tensor is empty or axis is out of range.
 * @throws std::runtime_error If any input is NaN or a numeric error occurs.
 * @throws std::bad_alloc If device memory allocation fails.
 */
template<typename float_t>
std::vector<uint64_t> argmax(const Tensor<float_t> & tensor, int64_t axis);
/// Explicit instantiation of argmax for float
extern template std::vector<uint64_t> argmax<float>
    (const Tensor<float>&, int64_t);

/**
 * @brief Elementwise linear interpolation between two tensors.
 *
 * Produces `num` samples interpolated between corresponding elements of
 * `start` and `stop` after broadcasting. The output shape equals the
 * broadcasted shape with an extra axis of length `num` inserted at
 * `axis`. If both inputs are scalar a 1-D tensor of length `num` is
 * returned. Optionally the per-S step values are returned via `step_out`.
 *
 * @param start Left endpoint tensor (broadcastable with `stop`).
 * @param stop Right endpoint tensor (broadcastable with `start`).
 * @param num Number of samples along the new axis (>= 0).
 * @param res_loc Memory location for result and step tensors.
 * @param axis Position to insert the samples axis in the output.
 * @param endpoint If true include `stop` as the last sample; otherwise
 *        use a half-open interval that excludes `stop`.
 * @param step_out If non-null, moved a tensor of per-S step values into
 *        `*step_out` on return.
 * @return Tensor<float_t> Interpolated tensor (broadcasted shape with
 *         inserted axis).
 * @throws std::invalid_argument For empty inputs or out-of-range axis.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error If numeric errors (NaN/Inf/overflow) occur.
 */
template<typename float_t>
Tensor<float_t> linspace(const Tensor<float_t>& start,
                        const Tensor<float_t>& stop,
                        uint64_t num,
                        MemoryLocation res_loc = MemoryLocation::DEVICE,
                        uint64_t axis = 0,
                        bool endpoint = true,
                        Tensor<float_t>* step_out = nullptr);
/// Explicit instantiation of linspace for float
extern template Tensor<float> linspace<float>(const Tensor<float>&,
const Tensor<float>&, uint64_t, MemoryLocation, uint64_t, bool, Tensor<float>*);

/* todo
    arange
    zeros
    integral
    factorial
    log?
    mean
    var
    std
    cov
    eigen
    */
} // namespace temper::math

#endif // TEMPER_MATH_HPP
