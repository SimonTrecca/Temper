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
 */
template <typename value_t>
Tensor<value_t> matmul(const Tensor<value_t> & first,
    const Tensor<value_t> & second);
/// \cond
extern template Tensor<float> matmul<float>
    (const Tensor<float>&, const Tensor<float>&);
extern template Tensor<uint64_t> matmul<uint64_t>
    (const Tensor<uint64_t>&, const Tensor<uint64_t>&);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> reshape(const Tensor<value_t> & tensor,
    const std::vector<uint64_t>& new_dimensions);
/// \cond
extern template Tensor<float> reshape<float>
    (const Tensor<float>&, const std::vector<uint64_t>&);
extern template Tensor<uint64_t> reshape<uint64_t>
    (const Tensor<uint64_t>&, const std::vector<uint64_t>&);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> sort(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// \cond
extern template Tensor<float> sort<float>
    (const Tensor<float>&, std::optional<int64_t>);
extern template Tensor<uint64_t> sort<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> sum(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// \cond
extern template Tensor<float> sum<float>
    (const Tensor<float>&, std::optional<int64_t>);
extern template Tensor<uint64_t> sum<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> cumsum(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// \cond
extern template Tensor<float> cumsum<float>
    (const Tensor<float>&, std::optional<int64_t>);
extern template Tensor<uint64_t> cumsum<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);
/// \endcond

/**
 * @brief Transpose a tensor (free function wrapper, full reversal).
 *
 * Returns a new tensor with all axes reversed, leaving the input tensor
 * unmodified. Delegates to `Tensor::transpose()`.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> A new tensor view with reversed axes.
 */
template<typename value_t>
Tensor<value_t> transpose(const Tensor<value_t> & tensor);
/// \cond
extern template Tensor<float> transpose<float>(const Tensor<float>&);
extern template Tensor<uint64_t> transpose<uint64_t>(const Tensor<uint64_t>&);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> transpose(const Tensor<value_t> & tensor,
    const std::vector<int64_t> & axes);
/// \cond
extern template Tensor<float> transpose<float>
    (const Tensor<float>&, const std::vector<int64_t>&);
extern template Tensor<uint64_t> transpose<uint64_t>
    (const Tensor<uint64_t>&, const std::vector<int64_t>&);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> pad(const Tensor<value_t> & tensor,
    uint64_t pad_top,
    uint64_t pad_bottom,
    uint64_t pad_left,
    uint64_t pad_right,
    value_t pad_value);
/// \cond
extern template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, uint64_t, uint64_t, float);
extern template Tensor<uint64_t> pad<uint64_t>
    (const Tensor<uint64_t>&, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
/// \endcond

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
/// \cond
extern template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, float);
extern template Tensor<uint64_t> pad<uint64_t>
    (const Tensor<uint64_t>&, uint64_t, uint64_t, uint64_t);
/// \endcond

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
*/
template<typename value_t>
Tensor<uint64_t> argmax(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// \cond
extern template Tensor<uint64_t> argmax<float>
    (const Tensor<float>&, std::optional<int64_t>);
extern template Tensor<uint64_t> argmax<uint64_t>
    (const Tensor<uint64_t>&, std::optional<int64_t>);
/// \endcond

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
 */
template<typename value_t>
Tensor<uint64_t> argsort(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt,
    bool descending = false);
/// \cond
extern template Tensor<uint64_t> argsort<float>
(const Tensor<float>&, std::optional<int64_t>, bool);
extern template Tensor<uint64_t> argsort<uint64_t>
(const Tensor<uint64_t>&, std::optional<int64_t>, bool);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> gather(const Tensor<value_t> & tensor,
    const Tensor<uint64_t> & indexes,
    std::optional<int64_t> axis_opt = std::nullopt);
/// \cond
extern template Tensor<float> gather<float>
(const Tensor<float>&, const Tensor<uint64_t>&, std::optional<int64_t>);
extern template Tensor<uint64_t> gather<uint64_t>
(const Tensor<uint64_t>&, const Tensor<uint64_t>&, std::optional<int64_t>);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> linspace(const Tensor<value_t>& start,
    const Tensor<value_t>& stop,
    uint64_t num,
    MemoryLocation res_loc = MemoryLocation::DEVICE,
    int64_t axis = 0,
    bool endpoint = true,
    Tensor<value_t>* step_out = nullptr);
/// \cond
extern template Tensor<float> linspace<float>(const Tensor<float>&,
const Tensor<float>&, uint64_t, MemoryLocation, int64_t, bool, Tensor<float>*);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> linspace(value_t start,
    value_t stop,
    uint64_t num,
    MemoryLocation res_loc = MemoryLocation::DEVICE,
    bool endpoint = true,
    Tensor<value_t>* step_out = nullptr);
/// \cond
extern template Tensor<float> linspace<float>(float,
float, uint64_t, MemoryLocation, bool, Tensor<float>*);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> arange(value_t start,
    value_t stop,
    value_t step,
    MemoryLocation res_loc);
/// \cond
extern template Tensor<float> arange<float>(float, float, float, MemoryLocation);
extern template Tensor<uint64_t> arange<uint64_t>
(uint64_t, uint64_t, uint64_t, MemoryLocation);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> arange(value_t stop,
    MemoryLocation res_loc = MemoryLocation::DEVICE);
/// \cond
extern template Tensor<float> arange<float>(float, MemoryLocation);
extern template Tensor<uint64_t> arange<uint64_t>(uint64_t, MemoryLocation);
/// \endcond

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
/// \cond
extern template Tensor<float> zeros<float>
    (const std::vector<uint64_t>&, MemoryLocation);
extern template Tensor<uint64_t> zeros<uint64_t>
    (const std::vector<uint64_t>&, MemoryLocation);
/// \endcond

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
 */
template <typename value_t>
value_t integral(std::function<value_t(value_t)> f,
    value_t a,
    value_t b,
    uint64_t n_bins = 1000);
/// \cond
extern template float integral<float>
    (std::function<float(float)>, float, float, uint64_t);
/// \endcond

/**
 * @brief Elementwise factorial computed on the device.
 *
 * Computes the factorial of each element and returns a tensor with the same
 * shape and memory location.
 *
 * @param tensor Input tensor (elements must be non-negative integers
 * within a small tolerance).
 * @return Tensor<value_t> Tensor of elementwise factorials.
 */
template<typename value_t>
Tensor<value_t> factorial(const Tensor<value_t> & tensor);
/// \cond
extern template Tensor<float> factorial<float>(const Tensor<float>&);
extern template Tensor<uint64_t> factorial<uint64_t>(const Tensor<uint64_t>&);
/// \endcond

/**
 * @brief Elementwise natural logarithm computed on the device.
 *
 * Computes ln(x) for every element of @p tensor and returns a tensor with
 * the same shape and memory location.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> Tensor containing elementwise natural logs.
 */
template<typename value_t>
Tensor<value_t> log(const Tensor<value_t> & tensor);
/// \cond
extern template Tensor<float> log<float>(const Tensor<float>&);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> mean(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// \cond
extern template Tensor<float> mean<float>
    (const Tensor<float>&, std::optional<int64_t>);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> var(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt,
    int64_t ddof = 0);
/// \cond
extern template Tensor<float> var<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> cov(const Tensor<value_t> & tensor,
    std::vector<int64_t> sample_axes,
    std::vector<int64_t> event_axes,
    int64_t ddof = 0);
/// \cond
extern template Tensor<float> cov<float> (const Tensor<float>&,
    std::vector<int64_t>, std::vector<int64_t>, int64_t);
/// \endcond

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
 */
template <typename value_t>
Tensor<value_t> cov(const Tensor<value_t> & tensor, int64_t ddof = 0);
/// \cond
extern template Tensor<float> cov<float> (const Tensor<float>&, int64_t);
/// \endcond

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
 */
template<typename value_t>
Tensor<value_t> stddev(const Tensor<value_t>& input,
    std::optional<int64_t> axis_opt = std::nullopt,
    int64_t ddof = 0);
/// \cond
extern template Tensor<float> stddev<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);
/// \endcond

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
 */
template <typename value_t>
std::pair<Tensor<value_t>, Tensor<value_t>> eig(const Tensor<value_t> & tensor,
    uint64_t max_iters = 100,
    value_t tol = static_cast<value_t>(1e-4));
/// \cond
extern template std::pair<Tensor<float>, Tensor<float>> eig<float>
    (const Tensor<float>&, uint64_t, float);
/// \endcond

/**
 * @brief Elementwise square root.
 *
 * Returns a new tensor with each element equal to the square root of the
 * corresponding input element. The input tensor is not modified.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> New tensor with elementwise sqrt applied.
 */
template <typename value_t>
Tensor<value_t> sqrt(const Tensor<value_t>& tensor);
/// \cond
extern template Tensor<float> sqrt<float>(const Tensor<float>& tensor);
/// \endcond

/**
 * @brief Elementwise power.
 *
 * Computes elementwise `a^b`, supporting broadcasting between the two inputs.
 *
 * @param a Base tensor.
 * @param b Exponent tensor.
 * @return Tensor<value_t> Tensor containing the elementwise powers.
 */
template<typename value_t>
Tensor<value_t> pow(const Tensor<value_t> & a, const Tensor<value_t> & b);
/// \cond
extern template Tensor<float> pow<float>
    (const Tensor<float>&, const Tensor<float>&);
extern template Tensor<uint64_t> pow<uint64_t>
    (const Tensor<uint64_t>&, const Tensor<uint64_t>&);
/// \endcond

/**
 * @brief Elementwise natural exponential.
 *
 * Computes `exp(x)` for every element of @p tensor and returns a tensor
 * with the same shape and memory location.
 *
 * @param tensor Input tensor.
 * @return Tensor<value_t> Tensor containing elementwise natural exponentials.
 */
template<typename value_t>
Tensor<value_t> exp(const Tensor<value_t> & tensor);
/// \cond
extern template Tensor<float> exp<float>(const Tensor<float>&);
/// \endcond

/**
 * @brief Upsampling modes for spatial upsampling operations.
 */
enum class UpsampleMode
{
    ZEROS,   ///< Insert zeros between elements (for transposed convolution)
    NEAREST  ///< Nearest neighbor upsampling (repeat values)
};

/**
 * @brief Upsample the last two spatial dimensions by inserting values.
 *
 * Upsamples the height and width dimensions (last two) of the input tensor
 * by inserting elements according to the specified mode. The input tensor
 * must have rank >= 3, with the last three dimensions interpreted as
 * (channels, height, width). Batch dimensions are preserved.
 *
 * For ZEROS mode with stride s:
 *   out_height = in_height * s - (s - 1)
 *   out_width  = in_width * s - (s - 1)
 *
 * @param tensor Input tensor (must have rank >= 3).
 * @param stride Upsampling factor for both height and width.
 * @param mode Upsampling mode (default:  ZEROS).
 * @return Tensor<value_t> Upsampled tensor with same batch dimensions
 *         and channel count.
 */
template<typename value_t>
Tensor<value_t> upsample(const Tensor<value_t> & tensor,
    uint64_t stride,
    UpsampleMode mode = UpsampleMode::ZEROS);
/// \cond
extern template Tensor<float> upsample<float>
    (const Tensor<float>&, uint64_t, UpsampleMode);
extern template Tensor<uint64_t> upsample<uint64_t>
    (const Tensor<uint64_t>&, uint64_t, UpsampleMode);
/// \endcond


/* todo
    diag
*/
} // namespace temper::math

#endif // TEMPER_MATH_HPP
