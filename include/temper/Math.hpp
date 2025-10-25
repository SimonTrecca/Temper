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
template <typename float_t>
Tensor<float_t> sort(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of sort for float
extern template Tensor<float> sort<float>
    (const Tensor<float>&, std::optional<int64_t>);

/**
 * @brief Compute the sum of tensor elements (free function wrapper).
 *
 * Returns a new tensor containing the sums of @p tensor along the specified
 * axis. Delegates to `Tensor::sum`.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to sum along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @return Tensor<float_t> A new tensor containing the sums.
 *
 * @throws std::invalid_argument If axis is out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or the results.
 */
template <typename float_t>
Tensor<float_t> sum(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of sum for float
extern template Tensor<float> sum<float>
    (const Tensor<float>&, std::optional<int64_t>);

/**
 * @brief Compute the cumulative sum of tensor elements (free function wrapper).
 *
 * Returns a new tensor containing the cumulative sums of @p tensor along
 * the specified axis. Delegates to `Tensor::cumsum`.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to cumsum along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @return Tensor<float_t> A new tensor containing the cumulative sums.
 *
 * @throws std::invalid_argument If axis is not -1 and is out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or the results.
 */
template <typename float_t>
Tensor<float_t> cumsum(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of cumsum for float
extern template Tensor<float> cumsum<float>
    (const Tensor<float>&, std::optional<int64_t>);

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
 * of [-rank..rank-1].
 * @return Tensor<float_t> A new tensor view with permuted axes.
 *
 * @throws std::invalid_argument If `axes.size()` != rank or if `axes` is not
 * a valid permutation.
 */
template<typename float_t>
Tensor<float_t> transpose(const Tensor<float_t> & tensor,
                        const std::vector<int64_t> & axes);
/// Explicit instantiation of transpose(axes) for float
extern template Tensor<float> transpose<float>
    (const Tensor<float>&, const std::vector<int64_t>&);

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
 * each slice along @p axis_opt. For axis = nullopt, returns a single global max
 * index. Ties are resolved by taking the first occurrence.
 *
 * @param tensor Input tensor of arbitrary rank.
 * @param axis_opt Axis to reduce (nullopt = flatten, otherwise -rank..rank-1).
 * @return std::vector<uint64_t> Indices of maximum elements.
 *
 * @throws std::invalid_argument If tensor is empty or axis is out of range.
 * @throws std::runtime_error If any input is NaN or a numeric error occurs.
 * @throws std::bad_alloc If device memory allocation fails.
 */
template<typename float_t>
std::vector<uint64_t> argmax(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of argmax for float
extern template std::vector<uint64_t> argmax<float>
    (const Tensor<float>&, std::optional<int64_t>);

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
 *
 * @throws std::invalid_argument For empty inputs or out-of-range axis.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error If numeric errors (NaN/Inf/overflow) occur.
 */
template<typename float_t>
Tensor<float_t> linspace(const Tensor<float_t>& start,
                        const Tensor<float_t>& stop,
                        uint64_t num,
                        MemoryLocation res_loc = MemoryLocation::DEVICE,
                        int64_t axis = 0,
                        bool endpoint = true,
                        Tensor<float_t>* step_out = nullptr);
/// Explicit instantiation of linspace for float
extern template Tensor<float> linspace<float>(const Tensor<float>&,
const Tensor<float>&, uint64_t, MemoryLocation, int64_t, bool, Tensor<float>*);

template<typename float_t>
Tensor<float_t> linspace(float_t start,
                        float_t stop,
                        uint64_t num,
                        MemoryLocation res_loc = MemoryLocation::DEVICE,
                        int64_t axis = 0,
                        bool endpoint = true,
                        Tensor<float_t>* step_out = nullptr);
/// Explicit instantiation of linspace(scalars) for float
extern template Tensor<float> linspace<float>(float,
float, uint64_t, MemoryLocation, int64_t, bool, Tensor<float>*);

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
 * @return Tensor<float_t> 1-D tensor containing the generated sequence.
 *
 * @throws std::invalid_argument If @p step is zero.
 * @throws std::runtime_error If any input is NaN/Inf or if a numeric error
 * (NaN/Inf/overflow) occurs during generation.
 * @throws std::bad_alloc If device memory allocation fails.
 */
template<typename float_t>
Tensor<float_t> arange(float_t start,
                       float_t stop,
                       float_t step,
                       MemoryLocation res_loc);
/// Explicit instantiation of arange for float
extern template Tensor<float> arange<float>(float, float, float, MemoryLocation);

/**
 * @brief Generate a 1-D tensor with values from 0 up to stop-1.
 *
 * Equivalent to `arange(0, stop, 1)`. Produces a 1-D tensor with integer-like
 * increments cast to @p float_t type.
 *
 * @param stop Upper bound of the sequence (exclusive).
 * @param res_loc Memory location for the resulting tensor.
 * @return Tensor<float_t> 1-D tensor containing the generated sequence
 * from 0 to stop-1.
 *
 * @throws std::runtime_error If @p stop is NaN/Inf.
 * @throws std::bad_alloc If device memory allocation fails.
 */
template<typename float_t>
Tensor<float_t> arange(float_t stop,
    MemoryLocation res_loc = MemoryLocation::DEVICE);
/// Explicit instantiation of arange(stop) for float
extern template Tensor<float> arange<float>(float, MemoryLocation);

/**
 * @brief Create a tensor filled with zeros.
 *
 * Constructs a tensor with the given @p shape in which every element is
 * zero-initialized (value `0` converted to @p float_t). Memory for the
 * tensor is allocated in the location specified by @p res_loc.
 *
 * @param shape Vector of dimension sizes for the tensor. The product of the
 *        entries defines the total number of elements.
 * @param res_loc Memory location for the resulting tensor (default: DEVICE).
 * @return Tensor<float_t> Tensor of the given shape with all elements equal to zero.
 *
 * @note The default Tensor builder used by this implementation already
 *       zero-initializes allocated memory.
 *
 */
template<typename float_t>
Tensor<float_t> zeros(const std::vector<uint64_t> & shape,
    MemoryLocation res_loc = MemoryLocation::DEVICE);
/// Explicit instantiation of zeros for float
extern template Tensor<float> zeros<float>
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
template <typename float_t>
float_t integral(std::function<float_t(float_t)> f,
                        float_t a,
                        float_t b,
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
 * @return Tensor<float_t> Tensor of elementwise factorials.
 * @throws std::invalid_argument If tensor is empty or contains
 * negative/non-integer values.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error On NaN/Inf/overflow during
 * input checks or accumulation.
 */
template<typename float_t>
Tensor<float_t> factorial(const Tensor<float_t> & tensor);
/// Explicit instantiation of factorial for float
extern template Tensor<float> factorial<float>(const Tensor<float>&);

/**
 * @brief Elementwise natural logarithm computed on the device.
 *
 * Computes ln(x) for every element of @p tensor and returns a tensor with
 * the same shape and memory location.
 *
 * @param tensor Input tensor.
 * @return Tensor<float_t> Tensor containing elementwise natural logs.
 *
 * @throws std::invalid_argument If @p tensor is empty.
 * @throws std::bad_alloc On device allocation failure.
 * @throws std::runtime_error If inputs contain NaN, or if any computed
 * output is non-finite (Inf / -Inf / NaN) during computation.
 */
template<typename float_t>
Tensor<float_t> log(const Tensor<float_t> & tensor);
/// Explicit instantiation of log for float
extern template Tensor<float> log<float>(const Tensor<float>&);

/**
 * @brief Compute the mean of tensor elements (free function wrapper).
 *
 * Returns a new tensor containing the arithmetic means of @p tensor along the
 * specified axis. Delegates to `Tensor::mean`.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to compute mean along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @return Tensor<float_t> A new tensor containing the means.
 *
 * @throws std::invalid_argument If the tensor is empty or if @p axis
 * is out of range.
 * @throws std::bad_alloc If required device memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered.
 */
template <typename float_t>
Tensor<float_t> mean(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of mean for float
extern template Tensor<float> mean<float>
    (const Tensor<float>&, std::optional<int64_t>);

/**
 * @brief Compute the variance of tensor elements (free function wrapper).
 *
 * Functional wrapper around `Tensor::var()`. Reduces the input tensor by
 * computing the arithmetic variance either over all elements (axis = nullopt)
 * or independently along a single axis. The divisor uses (N - ddof),
 * where N is the number of elements being reduced.
 *
 * @param tensor Input tensor.
 * @param axis_opt Axis to reduce along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @param ddof Delta degrees of freedom (0 => population variance).
 * @return Tensor<float_t> Tensor with the specified axis reduced.
 *
 * @throws std::invalid_argument If the input tensor has no elements,
 * if @p axis is out of range, if the selected axis has zero length,
 * or if (N - ddof) <= 0.
 * @throws std::bad_alloc If required memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or in computed results.
 */
template <typename float_t>
Tensor<float_t> var(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt,
    int64_t ddof = 0);
/// Explicit instantiation of var for float
extern template Tensor<float> var<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);

/**
 * @brief Compute covariance (free function wrapper).
 *
 * Functional wrapper that delegates to `Tensor::cov(sample_axes,
 * event_axes, ddof)`. Treats @p sample_axes as sample dimensions and
 * @p event_axes as event (feature) dimensions; remaining axes are batch
 * axes. Result shape is `{ <batch dims...>, event_total, event_total }`.
 *
 * @param tensor      Input tensor.
 * @param sample_axes Non-empty vector of axis indices to flatten as
 *                    samples. Order matters; entries must be distinct and
 *                    in [-rank, rank-1].
 * @param event_axes  Non-empty vector of axis indices to flatten as
 *                    events. Order matters; entries must be distinct,
 *                    disjoint from @p sample_axes, and in [-rank, rank-1].
 * @param ddof        Delta degrees of freedom (>= 0). Divisor is
 *                    (N - ddof) where N is product(lengths of sample axes).
 *
 * @return Tensor<float_t> Covariance matrices with shape
 *         `{ <batch dims...>, event_total, event_total }`.
 *
 * @throws std::invalid_argument
 * - input tensor has no elements.
 * - @p sample_axes or @p event_axes is empty.
 * - @p ddof is negative.
 * - tensor rank < 2.
 * - any axis index is out of range.
 * - the same axis appears more than once (within or across vectors).
 * - ddof >= number of samples (N).
 * @throws std::out_of_range
 * - internal view/alias construction would exceed the owner's bounds.
 * @throws std::bad_alloc
 * - required host/device memory allocation failed.
 * @throws std::runtime_error
 * - NaN or non-finite values encountered, or device/kernel errors during
 *   reduction or matrix multiplication.
 */
template <typename float_t>
Tensor<float_t> cov(const Tensor<float_t> & tensor,
                    std::vector<int64_t> sample_axes,
                    std::vector<int64_t> event_axes,
                    int64_t ddof = 0);
/// Explicit instantiation of cov for float
extern template Tensor<float> cov<float> (const Tensor<float>&,
    std::vector<int64_t>, std::vector<int64_t>, int64_t);

/**
 * @brief Convenience covariance wrapper using last two axes.
 *
 * Delegates to `Tensor::cov(ddof)` which is equivalent to
 * `cov(tensor, {rank-2}, {rank-1}, ddof)`. For shape {N, M} returns the
 * M×M covariance of the N samples; for {B1,...,N,M} returns {B1,...,M,M}.
 *
 * @param tensor Input tensor.
 * @param ddof   Delta degrees of freedom (>= 0). Must satisfy ddof < N,
 *               where N is the length of axis `rank-2`.
 *
 * @return Tensor<float_t> Covariance matrices for the last two axes.
 *
 * @throws std::invalid_argument
 * - tensor rank < 2.
 * - @p ddof is negative.
 * - ddof is invalid for the sample count (ddof >= N).
 *
 * @throws std::bad_alloc
 * - required memory allocation failed.
 *
 * @throws std::runtime_error
 * - NaN or non-finite values encountered, or device/kernel errors.
 */
template <typename float_t>
Tensor<float_t> cov(const Tensor<float_t> & tensor, int64_t ddof = 0);
/// Explicit instantiation of cov(no axes) for float
extern template Tensor<float> cov<float> (const Tensor<float>&, int64_t);

/**
 * @brief Compute the standard deviation of
 * tensor elements (free function wrapper).
 *
 * Functional wrapper around `Tensor::stddev()`. Reduces the input tensor by
 * computing the square root of its variance, either across all elements
 * (axis = nullopt) or independently along a single axis.
 *
 * @param input Input tensor.
 * @param axis_opt Axis to reduce along, nullopt = flatten,
 * otherwise -rank..rank-1.
 * @param ddof Delta degrees of freedom (0 => population std).
 * @return Tensor<float_t> Tensor with the specified axis reduced.
 *
 * @throws std::invalid_argument If the input tensor has no elements,
 * if @p axis is not -1 and out of range, if the selected axis has zero length,
 * or if (N - ddof) <= 0.
 * @throws std::bad_alloc If required memory cannot be allocated.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * in the inputs or during the sqrt computation.
 */
template<typename float_t>
Tensor<float_t> stddev(const Tensor<float_t>& input,
    std::optional<int64_t> axis_opt = std::nullopt,
    int64_t ddof = 0);
/// Explicit instantiation of stddev for float
extern template Tensor<float> stddev<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);

/**
 * @brief Compute eigenvalues and right eigenvectors for the last two axes.
 *
 * For each square matrix in `input` shaped `{B..., N, N}` this routine
 * computes up to `N` eigenpairs using power-iteration + rank-1 deflation.
 *
 * @param input     Input tensor containing one or more square matrices in
 *                  the last two axes.
 * @param max_iters Maximum iterations per power iteration (default 100).
 * @param tol       Convergence tolerance for the L2 iterate difference
 *                  (default 1e-4).
 * @return std::pair<Tensor<float_t>, Tensor<float_t>>
 *         First: eigenvalues tensor of shape `{B..., N}`.
 *         Second: right eigenvectors tensor of shape `{B..., N, N}` (columns).
 *
 * @throws std::invalid_argument if input rank < 2 or last two dims are not equal.
 * @throws std::runtime_error if random init vectors are zero, left/right inner
 *         product is zero, or device/kernel errors occur.
 */
template <typename float_t>
std::pair<Tensor<float_t>, Tensor<float_t>> eig(const Tensor<float_t> & input,
    uint64_t max_iters = 100,
    float_t tol = static_cast<float_t>(1e-4));
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
 * @return Tensor<float_t> New tensor with elementwise sqrt applied.
 *
 * @throws std::invalid_argument If the tensor has no elements.
 * @throws std::bad_alloc If device memory allocation fails.
 * @throws std::runtime_error If NaN values are found in inputs or
 *         non-finite results are produced.
 */
template <typename float_t>
Tensor<float_t> sqrt(const Tensor<float_t>& tensor);
/// Explicit instantiation of sqrt for float
extern template Tensor<float> sqrt<float>(const Tensor<float>& tensor);

/**
 * @brief Elementwise natural exponential.
 *
 * Computes `exp(x)` for every element of @p tensor and returns a tensor
 * with the same shape and memory location.
 *
 * @param tensor Input tensor.
 * @return Tensor<float_t> Tensor containing elementwise natural exponentials.
 *
 * @throws std::invalid_argument If @p tensor is empty.
 * @throws std::bad_alloc If required device helper memory cannot be allocated.
 * @throws std::runtime_error If:
 * - Inputs contain NaN, or
 * - A computed output is non-finite (Inf / -Inf / NaN), or
 * - A numeric error occurs during device/kernel execution.
 */
template<typename float_t>
Tensor<float_t> exp(const Tensor<float_t> & tensor);
/// Explicit instantiation of exp for float
extern template Tensor<float> exp<float>(const Tensor<float>&);

/* todo
    diag
*/
} // namespace temper::math

#endif // TEMPER_MATH_HPP
