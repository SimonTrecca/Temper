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
 * @return Tensor<value_t> Result of the matrix multiplication.
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
template <typename value_t>
Tensor<value_t> matmul(const Tensor<value_t> & first,
                        const Tensor<value_t> & second);
/// Explicit instantiation of matmul for float
extern template Tensor<float> matmul<float>
    (const Tensor<float>&, const Tensor<float>&);
/// Explicit instantiation of matmul for uint64_t
extern template Tensor<uint64_t> matmul<uint64_t>
    (const Tensor<uint64_t>&, const Tensor<uint64_t>&);

/**
 * @brief Reshape a tensor (free function wrapper).
 *
 * Returns a reshaped copy of the input tensor with new dimensions.
 * Unlike the member function, this free function first clones the tensor,
 * so the original tensor is left unmodified.
 *
 * @param tensor Input tensor.
 * @param new_dimensions New shape for the tensor.
 * @return Tensor<value_t> A new tensor with the specified shape.
 *
 * @throws std::invalid_argument If:
 * - @p new_dimensions is empty
 * - any entry in @p new_dimensions is zero
 * - the product of @p new_dimensions differs from the total element count
 *   of the input
 * @throws std::overflow_error If the product of @p new_dimensions
 * would overflow uint64_t.
 */
template <typename value_t>
Tensor<value_t> reshape(const Tensor<value_t> & tensor,
                        const std::vector<uint64_t>& new_dimensions);
/// Explicit instantiation of reshape for float
extern template Tensor<float> reshape<float>
    (const Tensor<float>&, const std::vector<uint64_t>&);
/// Explicit instantiation of reshape for uint64_t
extern template Tensor<uint64_t> reshape<uint64_t>
    (const Tensor<uint64_t>&, const std::vector<uint64_t>&);

/**
 * @brief Sort tensor elements (free function wrapper).
 *
 * Returns a sorted copy of the input tensor, either
 * flattened (axis = std::nullopt, default) or independently along a single axis.
 * The input tensor is not modified.
 * Supports negative axis indexing to start from right to left.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to sort along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @throws std::invalid_argument if @p axis_opt is out of range.
 * @throws std::bad_alloc if required device memory cannot be allocated.
 */
template <typename value_t>
Tensor<value_t> sort(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of sort for float
extern template Tensor<float> sort<float>
    (const Tensor<float>&, std::optional<int64_t>);
/// Explicit instantiation of sort for uint64_t
extern template Tensor<uint64_t> sort<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);

/**
 * @brief Compute the sum of tensor elements (free function wrapper).
 *
 * Returns a new tensor containing the sums of @p tensor along the specified
 * axis. Delegates to `Tensor::sum`.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to sum along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @return Tensor<value_t> A new tensor containing the sums.
 *
 * @throws std::invalid_argument If axis is out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or the results.
 */
template <typename value_t>
Tensor<value_t> sum(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of sum for float
extern template Tensor<float> sum<float>
    (const Tensor<float>&, std::optional<int64_t>);
/// Explicit instantiation of sum for uint64_t
extern template Tensor<uint64_t> sum<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);

/**
 * @brief Compute the cumulative sum of tensor elements (free function wrapper).
 *
 * Returns a new tensor containing the cumulative sums of @p tensor along
 * the specified axis. Delegates to `Tensor::cumsum`.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to cumsum along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @return Tensor<value_t> A new tensor containing the cumulative sums.
 *
 * @throws std::invalid_argument If axis is not -1 and is out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or the results.
 */
template <typename value_t>
Tensor<value_t> cumsum(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of cumsum for float
extern template Tensor<float> cumsum<float>
    (const Tensor<float>&, std::optional<int64_t>);
/// Explicit instantiation of cumsum for uint64_t
extern template Tensor<uint64_t> cumsum<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);

/**
 * @brief Transpose a tensor (free function wrapper, full reversal).
 *
 * Returns a new tensor with all axes reversed, leaving the input tensor
 * unmodified. Delegates to `Tensor::transpose()`.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> A new tensor view with reversed axes.
 *
 * @throws std::runtime_error If the tensor is empty (rank 0).
 */
template<typename value_t>
Tensor<value_t> transpose(const Tensor<value_t> & tensor);
/// Explicit instantiation of transpose() for float
extern template Tensor<float> transpose<float>(const Tensor<float>&);
/// Explicit instantiation of transpose() for uint64_t
extern template Tensor<uint64_t> transpose<uint64_t>(const Tensor<uint64_t>&);

/**
 * @brief Transpose a tensor with a custom axis order (free function wrapper).
 *
 * Returns a new tensor with its axes rearranged according to @p axes,
 * leaving the input tensor unmodified. Delegates to
 * `Tensor::transpose(const std::vector<uint64_t>&)`.
 *
 * @param tensor Input tensor.
 * @param axes Vector specifying the new order of axes. Must be a permutation
 * of [-rank..rank-1].
 * @return Tensor<value_t> A new tensor view with permuted axes.
 *
 * @throws std::invalid_argument If `axes.size()` != rank or if `axes` is not
 * a valid permutation.
 */
template<typename value_t>
Tensor<value_t> transpose(const Tensor<value_t> & tensor,
                        const std::vector<int64_t> & axes);
/// Explicit instantiation of transpose(axes) for float
extern template Tensor<float> transpose<float>
    (const Tensor<float>&, const std::vector<int64_t>&);
/// Explicit instantiation of transpose(axes) for uint64_t
extern template Tensor<uint64_t> transpose<uint64_t>
    (const Tensor<uint64_t>&, const std::vector<int64_t>&);

/**
 * @brief Pad the last two dimensions (height, width) with a constant.
 *
 * @param tensor Input tensor (must have rank >= 2).
 * @param pad_top Rows to add before the top.
 * @param pad_bottom Rows to add after the bottom.
 * @param pad_left Columns to add before the left.
 * @param pad_right Columns to add after the right.
 * @param pad_value Value used to fill padded elements.
 * @return Tensor<value_t> New tensor with requested padding.
 *
 * @throws std::invalid_argument If tensor is empty or rank < 2.
 * @throws std::overflow_error If shape/padding computations overflow.
 * @throws std::bad_alloc If device helper memory cannot be allocated.
 */
template<typename value_t>
Tensor<value_t> pad(const Tensor<value_t> & tensor,
    uint64_t pad_top,
    uint64_t pad_bottom,
    uint64_t pad_left,
    uint64_t pad_right,
    value_t pad_value);
/// Explicit instantiation of pad for float
extern template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, uint64_t, uint64_t, float);
/// Explicit instantiation of pad for uint64_t
extern template Tensor<uint64_t> pad<uint64_t>
    (const Tensor<uint64_t>&, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

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
 * @return Tensor<value_t> New padded tensor.
 */
template<typename value_t>
Tensor<value_t> pad(const Tensor<value_t> & tensor,
    uint64_t pad_height,
    uint64_t pad_width,
    value_t pad_value);
/// Explicit instantiation of pad (height, width) for float
extern template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, float);
/// Explicit instantiation of pad (height, width) for uint64_t
extern template Tensor<uint64_t> pad<uint64_t>
    (const Tensor<uint64_t>&, uint64_t, uint64_t, uint64_t);

/**
* @brief Compute indices of maximum elements along a specified axis.
*
* Finds the index of the maximum value for each slice of the input tensor
* along the specified axis. If @p axis_opt is std::nullopt, the input tensor
* is flattened and a single global maximum index is returned.
*
* The result tensor has the same rank as the input, with the reduced axis
* set to size 1, and uses the same memory location (HOST or DEVICE) as the input.
* Ties are resolved by taking the first occurrence of the maximum value.
*
* @param tensor Input tensor.
* @param axis_opt Axis to reduce, or std::nullopt to flatten before reduction.
* @return Tensor<uint64_t> Tensor of argmax indices.
*
* @throws std::invalid_argument If the tensor is empty or the axis is invalid.
* @throws std::runtime_error If any input contains NaN or a numeric error occurs.
* @throws std::bad_alloc If memory allocation fails.
*/
template<typename value_t>
Tensor<uint64_t> argmax(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of argmax for float
extern template Tensor<uint64_t> argmax<float>
    (const Tensor<float>&, std::optional<int64_t>);
/// Explicit instantiation of argmax for uint64_t
extern template Tensor<uint64_t> argmax<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);

/**
 * @brief Compute the indices that would sort a tensor.
 *
 * Returns a tensor of `uint64_t` indices such that applying them along
 * the selected axis orders elements in ascending (default) or descending
 * order. If `axis_opt` is `std::nullopt` the input is flattened and a
 * 1-D index tensor is returned. Negative axis values are supported.
 *
 * @param tensor Input tensor (must contain at least one element).
 * @param axis_opt Optional axis to sort along; `std::nullopt` means flatten.
 * @param descending If true, produce indices for descending order.
 * @return Tensor<uint64_t> Tensor of indices (same memory location as input).
 *
 * @throws std::invalid_argument If the input is empty or axis is out of range.
 * @throws std::bad_alloc If required device/host memory allocation fails.
 * @throws std::runtime_error If numeric or device execution errors occur.
 */
template<typename value_t>
Tensor<uint64_t> argsort(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt,
    bool descending = false);
/// Explicit instantiation of argsort for float
extern template Tensor<uint64_t> argsort<float>
(const Tensor<float>&, std::optional<int64_t>, bool);
/// Explicit instantiation of argsort for uint64_t
extern template Tensor<uint64_t> argsort<uint64_t>
(const Tensor<uint64_t>&, std::optional<int64_t>, bool);

/**
 * @brief Gather elements from a tensor using integer indices.
 *
 * Supports two modes:
 * - **Flattened** (@p axis_opt is @c std::nullopt): @p indexes must be 1-D
 * and its length must equal the total number of elements in @p tensor.
 * - **Axis-based** (when @p axis_opt is provided): @p indexes
 * (rank ≤ tensor.rank()) must be broadcastable to the input shape and supplies,
 * for each output location, an index selecting an element along the chosen axis.
 *
 * @param tensor Source tensor.
 * @param indexes Index tensor (uint64_t).
 * @param axis_opt Optional axis to gather along (nullopt = flattened).
 * @return Tensor<value_t> Result with same shape and memory location
 * as `tensor`.
 *
 * @throws std::invalid_argument On bad ranks/shape/axis or
 * incompatible broadcast.
 * @throws std::out_of_range On index values outside allowed range.
 * @throws std::bad_alloc On allocation failure.
 * @throws std::runtime_error On device/kernel numeric errors.
 */
template<typename value_t>
Tensor<value_t> gather(const Tensor<value_t> & tensor,
    const Tensor<uint64_t> & indexes,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of gather for float
extern template Tensor<float> gather<float>
(const Tensor<float>&, const Tensor<uint64_t>&, std::optional<int64_t>);
/// Explicit instantiation of gather for uint64_t
extern template Tensor<uint64_t> gather<uint64_t>
(const Tensor<uint64_t>&, const Tensor<uint64_t>&, std::optional<int64_t>);

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
 * @return Tensor<value_t> Interpolated tensor (broadcasted shape with
 *         inserted axis).
 *
 * @throws std::invalid_argument For empty inputs or out-of-range axis.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error If numeric errors (NaN/Inf/overflow) occur.
 */
template<typename value_t>
Tensor<value_t> linspace(const Tensor<value_t>& start,
    const Tensor<value_t>& stop,
    uint64_t num,
    MemoryLocation res_loc = MemoryLocation::DEVICE,
    int64_t axis = 0,
    bool endpoint = true,
    Tensor<value_t>* step_out = nullptr);
/// Explicit instantiation of linspace for float
extern template Tensor<float> linspace<float>(const Tensor<float>&,
const Tensor<float>&, uint64_t, MemoryLocation, int64_t, bool, Tensor<float>*);

/**
 * @brief Generate a 1-D tensor of linearly spaced samples between two scalars.
 *
 * @param start Start value.
 * @param stop  Stop value.
 * @param num   Number of samples.
 * @param res_loc Memory location for result (default DEVICE).
 * @param endpoint Include stop when true.
 * @param step_out If non-null, moved a length-1 tensor with the step.
 * @return 1-D Tensor<value_t> of length `num`.
 *
 * @throws std::invalid_argument For empty inputs or out-of-range axis.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error If numeric errors (NaN/Inf/overflow) occur.
 */
template<typename value_t>
Tensor<value_t> linspace(value_t start,
    value_t stop,
    uint64_t num,
    MemoryLocation res_loc = MemoryLocation::DEVICE,
    bool endpoint = true,
    Tensor<value_t>* step_out = nullptr);
/// Explicit instantiation of linspace(scalars) for float
extern template Tensor<float> linspace<float>(float,
float, uint64_t, MemoryLocation, bool, Tensor<float>*);

/**
 * @brief Generate a 1-D tensor with values in the half-open interval
 * [start, stop) using a fixed step.
 *
 * Produces a sequence of values starting at @p start, incremented by @p step,
 * stopping before reaching @p stop. The output length is computed as:
 *   `ceil((stop - start)/step)` for non-zero step. Supports positive
 * or negative step values.
 *
 * @param start First value of the sequence.
 * @param stop Upper bound of the sequence (exclusive).
 * @param step Increment between consecutive elements (non-zero).
 * @param res_loc Memory location for the resulting tensor.
 * @return Tensor<value_t> 1-D tensor containing the generated sequence.
 *
 * @throws std::invalid_argument If @p step is zero.
 * @throws std::runtime_error If any input is NaN/Inf or if a numeric error
 * (NaN/Inf/overflow) occurs during generation.
 * @throws std::bad_alloc If device memory allocation fails.
 */
template<typename value_t>
Tensor<value_t> arange(value_t start,
    value_t stop,
    value_t step,
    MemoryLocation res_loc);
/// Explicit instantiation of arange for float
extern template Tensor<float> arange<float>(float, float, float, MemoryLocation);
/// Explicit instantiation of arange for uint64_t
extern template Tensor<uint64_t> arange<uint64_t>
(uint64_t, uint64_t, uint64_t, MemoryLocation);

/**
 * @brief Generate a 1-D tensor with values from 0 up to stop-1.
 *
 * Equivalent to `arange(0, stop, 1)`. Produces a 1-D tensor with integer-like
 * increments cast to @p value_t type.
 *
 * @param stop Upper bound of the sequence (exclusive).
 * @param res_loc Memory location for the resulting tensor.
 * @return Tensor<value_t> 1-D tensor containing the generated sequence
 * from 0 to stop-1.
 *
 * @throws std::runtime_error If @p stop is NaN/Inf.
 * @throws std::bad_alloc If device memory allocation fails.
 */
template<typename value_t>
Tensor<value_t> arange(value_t stop,
    MemoryLocation res_loc = MemoryLocation::DEVICE);
/// Explicit instantiation of arange(stop) for float
extern template Tensor<float> arange<float>(float, MemoryLocation);
/// Explicit instantiation of arange(stop) for uint64_t
extern template Tensor<uint64_t> arange<uint64_t>(uint64_t, MemoryLocation);

/**
 * @brief Create a tensor filled with zeros.
 *
 * Constructs a tensor with the given @p shape in which every element is
 * zero-initialized (value `0` converted to @p value_t). Memory for the
 * tensor is allocated in the location specified by @p res_loc.
 *
 * @param shape Vector of dimension sizes for the tensor. The product of the
 *        entries defines the total number of elements.
 * @param res_loc Memory location for the resulting tensor (default: DEVICE).
 * @return Tensor<value_t> Tensor of the given shape with all elements equal to zero.
 *
 * @note The default Tensor builder used by this implementation already
 *       zero-initializes allocated memory.
 *
 */
template<typename value_t>
Tensor<value_t> zeros(const std::vector<uint64_t> & shape,
    MemoryLocation res_loc = MemoryLocation::DEVICE);
/// Explicit instantiation of zeros for float
extern template Tensor<float> zeros<float>
    (const std::vector<uint64_t>&, MemoryLocation);
/// Explicit instantiation of zeros for uint64_t
extern template Tensor<uint64_t> zeros<uint64_t>
    (const std::vector<uint64_t>&, MemoryLocation);

/**
 * @brief Approximate the integral of f over [a, b] using Simpson's rule.
 *
 * Splits [a, b] into @p n_bins equal intervals and applies composite
 * Simpson's rule to estimate the area under @p f.
 *
 * @param f Integrand function.
 * @param a Lower bound of integration.
 * @param b Upper bound of integration.
 * @param n_bins Number of subintervals (must be >= 1).
 * @return Approximate value of the integral.
 *
 * @throws std::invalid_argument If @p n_bins is less than 1.
 */
template <typename value_t>
value_t integral(std::function<value_t(value_t)> f,
    value_t a,
    value_t b,
    uint64_t n_bins = 1000);
/// Explicit instantiation of integral for float
extern template float integral<float>
    (std::function<float(float)>, float, float, uint64_t);

/**
 * @brief Elementwise factorial computed on the device.
 *
 * Computes the factorial of each element and returns a tensor with the same
 * shape and memory location.
 *
 * @param tensor Input tensor (elements must be non-negative integers
 * within a small tolerance).
 * @return Tensor<value_t> Tensor of elementwise factorials.
 * @throws std::invalid_argument If tensor is empty or contains
 * negative/non-integer values.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error On NaN/Inf/overflow during
 * input checks or accumulation.
 */
template<typename value_t>
Tensor<value_t> factorial(const Tensor<value_t> & tensor);
/// Explicit instantiation of factorial for float
extern template Tensor<float> factorial<float>(const Tensor<float>&);
/// Explicit instantiation of factorial for uint64_t
extern template Tensor<uint64_t> factorial<uint64_t>(const Tensor<uint64_t>&);

/**
 * @brief Elementwise natural logarithm computed on the device.
 *
 * Computes ln(x) for every element of @p tensor and returns a tensor with
 * the same shape and memory location.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> Tensor containing elementwise natural logs.
 *
 * @throws std::invalid_argument If @p tensor is empty.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error If inputs contain NaN, or if any computed
 * output is non-finite (Inf / -Inf / NaN) during computation.
 */
template<typename value_t>
Tensor<value_t> log(const Tensor<value_t> & tensor);
/// Explicit instantiation of log for float
extern template Tensor<float> log<float>(const Tensor<float>&);

/**
 * @brief Compute the mean (average) of tensor elements.
 *
 * Reduces the tensor by computing the arithmetic mean either over all
 * elements (axis = nullopt) or independently along a single axis.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to compute mean along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @return Tensor<value_t> Tensor with the specified axis reduced (result
 * uses the same memory location as the input).
 *
 * @throws std::invalid_argument If the input tensor has no elements,
 * if @p axis is out of range, or if the selected axis
 * has zero length.
 * @throws std::bad_alloc If required device/host memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in inputs or produced by the reduction/division.
 */
template <typename value_t>
Tensor<value_t> mean(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of mean for float
extern template Tensor<float> mean<float>
    (const Tensor<float>&, std::optional<int64_t>);

/**
 * @brief Compute the variance of tensor elements.
 *
 * Reduces the tensor by computing the arithmetic variance either over all
 * elements (axis = nullopt) or independently along a single axis.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to reduce along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @param ddof Delta degrees of freedom (0 => population variance).
 * @return Tensor<value_t> Tensor with the specified axis reduced.
 *
 * @throws std::invalid_argument If the input tensor has no elements,
 * if @p axis is out of range, if the selected axis has zero length,
 * or if (N - ddof) <= 0.
 * @throws std::bad_alloc If required memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered.
 */
template <typename value_t>
Tensor<value_t> var(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt,
    int64_t ddof = 0);
/// Explicit instantiation of var for float
extern template Tensor<float> var<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);

/**
 * @brief Compute covariance matrices over specified sample and event axes.
 *
 * Treats axes in @p sample_axes as sample dimensions and axes in
 * @p event_axes as event (feature) dimensions. Remaining axes are batch
 * axes. Result shape is:
 *   { <batch dims...>, event_total, event_total }
 * where event_total = product(lengths of @p event_axes).
 *
 * @param tensor      Input tensor.
 * @param sample_axes Non-empty vector of axis indices to flatten as samples.
 *                    Order matters. Entries must be distinct and in
 *                    [-rank, rank-1].
 * @param event_axes  Non-empty vector of axis indices to flatten as events.
 *                    Order matters. Entries must be distinct, disjoint from
 *                    @p sample_axes, and in [-rank, rank-1].
 * @param ddof        Delta degrees of freedom (>= 0). Divisor is
 *                    (N - ddof) where N = product(lengths of sample axes).
 * @return Tensor<value_t> Covariance matrices with shape
 *         `{ <batch dims...>, event_total, event_total }`. Allocated in the
 *         same memory location (host/device) as the input where possible.
 *
 * @throws std::invalid_argument if:
 * - input tensor has no elements.
 * - @p sample_axes or @p event_axes is empty.
 * - @p ddof is negative.
 * - tensor rank < 2.
 * - any axis index is out of range.
 * - the same axis appears more than once (within or across vectors).
 * - ddof >= number of samples (N).
 * @throws std::out_of_range if:
 * - internal view/alias construction would exceed the owner's bounds.
 * @throws std::bad_alloc if:
 * - required host/device memory allocation failed.
 * @throws std::runtime_error if:
 * - NaN or non-finite values encountered, or device/kernel errors during
 *   reduction or matrix multiplication.
 */
template <typename value_t>
Tensor<value_t> cov(const Tensor<value_t> & tensor,
                    std::vector<int64_t> sample_axes,
                    std::vector<int64_t> event_axes,
                    int64_t ddof = 0);
/// Explicit instantiation of cov for float
extern template Tensor<float> cov<float> (const Tensor<float>&,
    std::vector<int64_t>, std::vector<int64_t>, int64_t);

/**
 * @brief Convenience overload — covariance for the last two axes.
 *
 * Equivalent to:
 *   cov(tensor, { rank-2 }, { rank-1 }, ddof)
 *
 * For a 2-D tensor of shape {N, M} this returns the M×M covariance of the
 * N samples. For shape {B1,..., N, M} it returns {B1,..., M, M}.
 *
 * @param tensor Input tensor.
 * @param ddof Delta degrees of freedom (>= 0). Must satisfy ddof < N,
 *             where N is the length of axis `rank-2`.
 * @return Tensor<value_t> Covariance matrices for the last two axes.
 *
 * @throws std::invalid_argument if:
 * - tensor rank < 2.
 * - @p ddof is negative.
 * - ddof is invalid for the sample count (ddof >= N).
 * @throws std::bad_alloc if:
 * - required memory allocation failed.
 * @throws std::runtime_error if:
 * - NaN or non-finite values encountered, or device/kernel errors.
 */
template <typename value_t>
Tensor<value_t> cov(const Tensor<value_t> & tensor, int64_t ddof = 0);
/// Explicit instantiation of cov(no axes) for float
extern template Tensor<float> cov<float> (const Tensor<float>&, int64_t);

/**
 * @brief Compute the standard deviation of tensor elements.
 *
 * Reduces the tensor by computing the square root of the variance,
 * either over all elements (axis = nullopt)or independently
 * along a single axis.
 *
 * The computation follows the same semantics as `var()`, using the given
 * delta degrees of freedom (ddof) to adjust the divisor (N - ddof). When
 * ddof = 0, the result is the population standard deviation; when ddof = 1,
 * the result is the sample standard deviation.
 *
 * @param input Input tensor.
 * @param axis_opt Axis to reduce along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @param ddof Delta degrees of freedom (0 => population std).
 * @return Tensor<value_t> Tensor with the specified axis reduced.
 *
 * @throws std::invalid_argument If the input tensor has no elements,
 * if @p axis is out of range, if the selected axis has
 * zero length, or if (N - ddof) <= 0.
 * @throws std::bad_alloc If required memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or produced during the sqrt computation.
 */
template<typename value_t>
Tensor<value_t> stddev(const Tensor<value_t>& input,
    std::optional<int64_t> axis_opt = std::nullopt,
    int64_t ddof = 0);
/// Explicit instantiation of stddev for float
extern template Tensor<float> stddev<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);

/**
 * @brief Compute the eigenvalues and right eigenvectors of the last
 * two axes using a batched Jacobi method.
 *
 * This routine interprets the last two dimensions of the tensor as a
 * stack of square matrices `{B..., N, N}` and applies a Jacobi
 * (Givens-rotation) diagonalization to each of them in parallel. It
 * produces all `N` eigenvalues and a full orthonormal set of right
 * eigenvectors for every matrix in the batch.
 *
 * The diagonalization is iterative. For each sweep over the matrix
 * entries, a sequence of Givens rotations progressively reduces the
 * off–diagonal elements. The process stops when either the largest
 * off–diagonal magnitude falls below `tol`, or `max_iters` sweeps
 * have completed, or a numerical/device error is detected.
 *
 * @param tensor Input tensor containing one or more square matrices in
 * the last two axes.
 * @param max_iters Maximum number of full Jacobi sweeps to attempt.
 * @param tol Convergence threshold for the largest off–diagonal entry.
 * @return std::pair<Tensor<value_t>, Tensor<value_t>>
 *     First:  eigenvalues of shape `{B..., 1, N}`.
 *     Second: eigenvectors of shape `{B..., N, N}`, stored by
 *     columns.
 *
 * @throws std::invalid_argument
 *     If `rank < 2` or the last two dimensions are not square.
 * @throws std::runtime_error
 *     If NaN or non-finite values are detected in the input or
 *     during computation, or a division by zero is encountered
 *     when forming Givens coefficients.
 */
template <typename value_t>
std::pair<Tensor<value_t>, Tensor<value_t>> eig(const Tensor<value_t> & tensor,
    uint64_t max_iters = 100,
    value_t tol = static_cast<value_t>(1e-4));
/// Explicit instantiation of eig for float
extern template std::pair<Tensor<float>, Tensor<float>> eig<float>
    (const Tensor<float>&, uint64_t, float);

/**
 * @brief Elementwise square root.
 *
 * Returns a new tensor with each element equal to the square root of the
 * corresponding input element. The input tensor is not modified.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> New tensor with elementwise sqrt applied.
 *
 * @throws std::invalid_argument If the tensor has no elements.
 * @throws std::bad_alloc If device memory allocation fails.
 * @throws std::runtime_error If NaN values are found in inputs or
 *         non-finite results are produced.
 */
template <typename value_t>
Tensor<value_t> sqrt(const Tensor<value_t>& tensor);
/// Explicit instantiation of sqrt for float
extern template Tensor<float> sqrt<float>(const Tensor<float>& tensor);

/**
 * @brief Elementwise power.
 *
 * Computes elementwise `a^b`, supporting broadcasting between the two inputs.
 *
 * @param a Base tensor.
 * @param b Exponent tensor.
 * @return Tensor<value_t> Tensor containing the elementwise powers.
 *
 * @throws std::invalid_argument If either input tensor is empty or the shapes
 *         are not broadcastable.
 * @throws std::bad_alloc If required host or device memory cannot be allocated.
 * @throws std::runtime_error If NaN/Inf or other numeric/device errors occur
 *         during computation.
 */
template<typename value_t>
Tensor<value_t> pow(const Tensor<value_t> & a, const Tensor<value_t> & b);
/// Explicit instantiation of pow for float
extern template Tensor<float> pow<float>
    (const Tensor<float>&, const Tensor<float>&);
/// Explicit instantiation of pow for uint64_t
extern template Tensor<uint64_t> pow<uint64_t>
    (const Tensor<uint64_t>&, const Tensor<uint64_t>&);

/**
 * @brief Elementwise natural exponential.
 *
 * Computes `exp(x)` for every element of @p tensor and returns a tensor
 * with the same shape and memory location.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> Tensor containing elementwise natural exponentials.
 *
 * @throws std::invalid_argument If @p tensor is empty.
 * @throws std::bad_alloc If required device helper memory cannot be allocated.
 * @throws std::runtime_error If:
 * - Inputs contain NaN, or
 * - A computed output is non-finite (Inf / -Inf / NaN), or
 * - A numeric error occurs during device/kernel execution.
 */
template<typename value_t>
Tensor<value_t> exp(const Tensor<value_t> & tensor);
/// Explicit instantiation of exp for float
extern template Tensor<float> exp<float>(const Tensor<float>&);

/* todo
    diag
*/
} // namespace temper::math

#endif // TEMPER_MATH_HPP
