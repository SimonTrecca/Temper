/**
 * @file Stats.hpp
 * @brief Statistical distribution utilities.
 *
 * Provides common functions for statistical distributions.
 */

#ifndef TEMPER_STATS_HPP
#define TEMPER_STATS_HPP

#include "Tensor.hpp"

namespace temper::stats
{

/**
 * @brief Draw samples from the standard normal distribution.
 *
 * Convenience wrapper that returns samples ~ N(0,1). Implemented by calling
 * `temper::stats::norm::rvs` with `loc = 0` and `scale = 1`.
 *
 * @param out_shape Output shape for the returned tensor (e.g. {N} or {H,W}).
 * @param res_loc MemoryLocation where the result should be allocated.
 *                Defaults to the project's device-oriented convention.
 * @param seed Optional RNG seed. If zero, the implementation seeds from
 *             `std::random_device` the same way as `norm::rvs`.
 * @return Tensor<value_t> Tensor of shape `out_shape` with samples ~ N(0,1).
 */
template<typename value_t>
Tensor<value_t> randn(const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc = MemoryLocation::DEVICE,
    uint64_t seed = 0ULL);
/// Explicit instantiation of randn for float
extern template Tensor<float> randn<float>
(const std::vector<uint64_t>&, MemoryLocation, uint64_t);

namespace norm
{
/**
 * @brief Probability density function of the normal distribution.
 *
 * Computes the value of the normal PDF element-wise:
 * pdf(x; loc, scale) = (1 / (scale * sqrt(2 * pi))) *
 * exp(-0.5 * ((x - loc) / scale)^2)
 *
 * Inputs `x`, `loc` (mean) and `scale` (standard deviation) are broadcast
 * together to produce the output shape.
 *
 * @param x Values at which to evaluate the PDF. Must be non-empty.
 * @param loc Mean(s) of the normal distribution. Must be non-empty.
 * @param scale Standard deviation(s) of the normal distribution. Must be
 * non-empty and strictly positive.
 * @return Tensor<value_t> Tensor containing PDF values with the broadcasted
 * shape.
 *
 * @throws std::invalid_argument if any input tensor is empty, if any
 * `scale` element is non-positive, or if NaN is detected in inputs.
 * @throws std::runtime_error if a non-finite result (overflow or Inf) or
 * other numeric error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> pdf(const Tensor<value_t>& x,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale);
/// Explicit instantiation of norm::pdf for float
extern template Tensor<float> pdf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Logarithm of the normal probability density function.
 *
 * Computes the log-PDF element-wise:
 * logpdf(x; loc, scale) = -0.5 * ((x - loc) / scale)^2 
 *  - log(scale) - 0.5*log(2*pi)
 * 
 * Inputs `x`, `loc` and `scale` are broadcast together to produce the
 * output shape.
 *
 * @param x Values at which to evaluate the log-PDF. Must be non-empty.
 * @param loc Mean(s) of the normal distribution. Must be non-empty.
 * @param scale Std-dev(s) of the normal distribution. Must be non-empty
 * and strictly positive.
 * @return Tensor<value_t> Tensor containing log-PDF values with the
 * broadcasted shape.
 *
 * @throws std::invalid_argument if any input tensor is empty, if any
 * `scale` element is non-positive, or if NaN is detected in inputs.
 * @throws std::runtime_error if a non-finite result (overflow or Inf)
 * or other numeric error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> logpdf(const Tensor<value_t>& x,
const Tensor<value_t>& loc,
const Tensor<value_t>& scale);
/// Explicit instantiation of norm::logpdf for float
extern template Tensor<float> logpdf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Cumulative distribution function of the normal distribution.
 *
 * Computes the normal CDF element-wise:
 *     cdf(x; loc, scale) = 0.5 * (1 + erf((x - loc) / (scale * sqrt(2))))
 *
 * Inputs `x`, `loc` (mean) and `scale` (standard deviation) are broadcast
 * together to produce the output shape.
 *
 * @param x Values at which to evaluate the CDF. Must be non-empty.
 * @param loc Mean(s) of the normal distribution. Must be non-empty.
 * @param scale Standard deviation(s) of the normal distribution. Must be
 * non-empty and strictly positive.
 * @return Tensor<value_t> Tensor containing CDF values in [0, 1] with the
 * broadcasted shape.
 *
 * @throws std::invalid_argument if any input tensor is empty, if any
 * `scale` element is non-positive, or if NaN is detected in inputs.
 * @throws std::runtime_error if a non-finite result (overflow or Inf) or
 * other numeric error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> cdf(const Tensor<value_t>& x,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale);
/// Explicit instantiation of norm::cdf for float
extern template Tensor<float> cdf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Inverse CDF (percent-point function) for the normal distribution.
 *
 * Computes x = loc + scale * inverse_normal_cdf(q) with broadcasting.
 *
 * @param q Probabilities in [0,1], broadcastable to output shape.
 * @param loc Mean tensor, broadcastable.
 * @param scale Std-dev tensor, broadcastable.
 * @return Tensor<value_t> Quantiles with the broadcasted shape.
 *
 * @throws std::invalid_argument if any input is empty, any q is outside
 *         [0,1], or any scale is <= 0.
 * @throws std::runtime_error if NaN or non-finite results occur.
 */
template<typename value_t>
Tensor<value_t> ppf(const Tensor<value_t>& q,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale);
/// Explicit instantiation of norm::ppf for float
extern template Tensor<float> ppf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Inverse survival function (ISF) / quantile of the normal.
 *
 * Computes the inverse survival function element-wise. The relation
 * used is: isf(q; loc, scale) == ppf(1 - q; loc, scale). Inputs `q`,
 * `loc` and `scale` are broadcast together to produce the output shape.
 *
 * For a scalar q this corresponds to the value x such that
 *     q = 1 - CDF(x; loc, scale)
 * i.e. the returned x satisfies CDF(x; loc, scale) = 1 - q.
 *
 * Note: q values of exactly 0 or 1 lead to infinite intermediate values
 * inside the kernel (the implementation maps those to +/-Inf) which are
 * treated as numeric non-finite results and will be reported as errors.
 *
 * @param q Probabilities in [0,1], broadcastable to the output shape.
 * Must be non-empty.
 * @param loc Mean(s) of the normal distribution. Must be non-empty.
 * @param scale Std-dev(s) of the normal distribution. Must be non-empty
 * and strictly positive.
 * @return Tensor<value_t> Quantiles (ISF values) with the broadcasted
 * shape.
 *
 * @throws std::invalid_argument if any input tensor is empty, if any
 * `q` value is outside [0,1], if any `scale` element is
 * non-positive, or if NaN is detected in inputs.
 * @throws std::runtime_error if a non-finite result (overflow or Inf)
 * or other numeric error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> isf(const Tensor<value_t>& q,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale);
/// Explicit instantiation of norm::isf for float
extern template Tensor<float> isf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Draw samples from Normal(loc, scale) with shape out_shape.
 *
 * Generates device uniform variates (xorshift64*), maps them through ppf,
 * and returns a tensor of samples with shape out_shape.
 *
 * @param loc Mean tensor, broadcastable to out_shape.
 * @param scale Std-dev tensor, broadcastable to out_shape.
 * @param out_shape Desired output shape.
 * @param res_loc MemoryLocation for the returned tensor.
 * @param seed RNG seed (0 => seeded from std::random_device).
 *        Non-zero seed yields deterministic output.
 * @return Tensor<value_t> Samples with shape out_shape.
 *
 * @throws std::invalid_argument if loc or scale are empty; ppf errors
 *         (e.g. non-positive scale) are propagated as invalid_argument.
 * @throws std::runtime_error on NaN or numeric errors during uniform
 *         generation.
 */
template<typename value_t>
Tensor<value_t> rvs(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale,
    const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc = MemoryLocation::DEVICE,
    uint64_t seed = 0ULL);
/// Explicit instantiation of norm::rvs for float
extern template Tensor<float> rvs<float>(const Tensor<float>&,
const Tensor<float>&, const std::vector<uint64_t>&, MemoryLocation, uint64_t);

/**
 * @brief Mean of the normal distribution.
 *
 * Returns the mean of Normal(loc, scale) element-wise.
 *
 * For the normal distribution, the mean is equal to `loc`.
 *
 * @param loc Mean tensor of the distribution. Must be non-empty.
 * @param scale Standard deviation tensor of the distribution. Not needed;
 * kept for consistency.
 * @return Tensor<value_t> Tensor containing the mean values.
 */
template<typename value_t>
Tensor<value_t> mean(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale);
/// Explicit instantiation of norm::mean for float
extern template Tensor<float> mean<float>
(const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Computes the variance of a normal distribution.
 *
 * Returns a tensor containing the variance values of Normal(loc, scale).
 * The variance is the square of the `scale` tensor.
 *
 * @param loc Mean tensor. Not needed; kept for consistency.
 * @param scale Standard deviation tensor. Must be non-empty and positive.
 * @return Tensor<value_t> Tensor containing variance values, broadcasted
 *         to the shape of `loc` and `scale`.
 *
 * @throws std::invalid_argument If scale is empty.
 * @throws std::bad_alloc If required host or device memory cannot be allocated.
 * @throws std::runtime_error If NaN/Inf or other numeric/device errors occur
 *         during computation.
 */
template<typename value_t>
Tensor<value_t> var(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale);
/// Explicit instantiation of norm::var for float
extern template Tensor<float> var<float>
(const Tensor<float>&, const Tensor<float>&);


/**
 * @brief Computes the standard deviation of a normal distribution.
 *
 * Returns a tensor containing the standard deviation values of
 * Normal(loc, scale).
 * The standard deviation is the same as the `scale` tensor.
 *
 * @param loc Mean tensor. Not needed; kept for consistency.
 * @param scale Standard deviation tensor. Must be non-empty and positive.
 * @return Tensor<value_t> Tensor containing standard deviation values.
 */
template<typename value_t>
Tensor<value_t> stddev(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale);
/// Explicit instantiation of norm::stddev for float
extern template Tensor<float> stddev<float>
(const Tensor<float>&, const Tensor<float>&);

} // namespace norm

    namespace beta
    {
        /*todo
        pdf
        cdf
        ppf
        rvs
        isf
        logpdf
        mean
        var
        std
        */
    } // namespace beta

    namespace expon
    {
        /*todo
        pdf
        cdf
        ppf
        rvs
        isf
        logpdf
        mean
        var
        std
        */
    } // namespace expon

    namespace poisson
    {
        /*todo
        pmf
        cdf
        ppf
        rvs
        isf
        logpmf
        mean
        var
        std
        */
    } // namespace poisson

    namespace t
    {
        /*todo
        pdf
        cdf
        ppf
        rvs
        isf
        logpdf
        mean
        var
        std
        */
    } // namespace t

namespace chisquare
{
/**
 * @brief Probability density function of the chi-square distribution.
 *
 * Computes the chi-square PDF element-wise:
 *     pdf(x; k) = (1 / (2^(k/2) * Gamma(k/2))) * x^(k/2 - 1) * exp(-x/2)
 *
 * Inputs `x` (values) and `k` (degrees of freedom) are broadcast together
 * to produce the output shape. The PDF is defined for x >= 0 and for
 * positive degrees of freedom `k`.
 *
 * @param x Values at which to evaluate the PDF. Must be non-empty. Values
 *          less than zero are invalid and will be reported as an error.
 * @param k Degrees of freedom. Must be non-empty. Values
 *          less than zero are invalid and will be reported as an error.
 * @return Tensor<value_t> Tensor containing PDF values with the broadcasted
 *         shape.
 *
 * @throws std::invalid_argument if any input tensor is empty or NaN is
 *         detected in inputs.
 * @throws std::invalid_argument if any evaluated `x` element or 'k' element
 *         is invalid.
 * @throws std::runtime_error if a non-finite result (overflow or Inf) or
 *         other numeric/device error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> pdf(const Tensor<value_t>& x,
    const Tensor<value_t>& k);
/// Explicit instantiation of chisquare::pdf for float
extern template Tensor<float> pdf<float>
(const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Logarithm of the chi-square probability density function.
 *
 * Computes the log-PDF element-wise:
 *     logpdf(x; k) = (k/2 - 1) * log(x) - x/2
 *                     - (k/2) * log(2) - lgamma(k/2)
 *
 * Inputs `x` (values) and `k` (degrees of freedom) are broadcast together
 * to produce the output shape.
 *
 * @param x Values at which to evaluate the log-PDF. Must be non-empty.
 *          Values less than zero are invalid and reported as an error.
 * @param k Degrees of freedom. Must be non-empty and strictly positive.
 *          Non-positive values are invalid and reported as an error.
 * @return Tensor<value_t> Tensor containing element-wise log-PDF values
 *         with the broadcasted shape.
 *
 * @throws std::invalid_argument if any input tensor is empty, if NaN is
 *         detected in the inputs, or if any element of `k` is non-positive
 *         or any element of `x` is negative.
 * @throws std::runtime_error if a non-finite result (overflow or Inf) or
 *         other numeric/device error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> logpdf(const Tensor<value_t>& x,
    const Tensor<value_t>& k);
/// Explicit instantiation of chisquare::logpdf for float
extern template Tensor<float> logpdf<float>
(const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Cumulative distribution function of the chi-square distribution.
 *
 * Computes the chi-square CDF element-wise:
 *     cdf(x; k) = P(k/2, x/2)
 * where P is the lower regularized gamma function.
 *
 * Inputs `x` (values) and `k` (degrees of freedom) are broadcast together
 * to produce the output shape. The CDF is defined for x >= 0 and k > 0.
 *
 * @param x Values at which to evaluate the CDF. Must be non-empty. Values
 *          less than zero are invalid and will be reported as an error.
 * @param k Degrees of freedom. Must be non-empty and positive. Values
 *          less than or equal to zero are invalid.
 * @return Tensor<value_t> Tensor containing CDF values in [0,1] with the
 *         broadcasted shape.
 *
 * @throws std::invalid_argument if any input tensor is empty, contains NaN,
 *         or if any element of x < 0 or k <= 0.
 * @throws std::runtime_error if a non-finite result (overflow, Inf) or
 *         other numeric/device error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> cdf(const Tensor<value_t>& x,
    const Tensor<value_t>& k);
/// Explicit instantiation of chisquare::cdf for float
extern template Tensor<float> cdf<float>
(const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Inverse CDF (percent-point function) of the chi-square distribution.
 *
 * Computes the chi-square quantile x for given cumulative probabilities `q`
 * and degrees of freedom `k`, element-wise with broadcasting:
 *     x = 2 * inverse_regularized_gamma(k/2, q)
 *
 * Inputs `q` (probabilities) and `k` (degrees of freedom) are broadcast
 * together to produce the output shape. `q` values must lie in [0,1]; `k`
 * elements must be strictly positive. Note that `q == 0` maps to `x == 0`,
 * while `q == 1` can produce infinite intermediate values in the kernel and
 * will be treated as a numeric error by this implementation.
 *
 * @param q Probabilities in [0,1]. Must be non-empty.
 * @param k Degrees of freedom, must be non-empty and strictly positive.
 * @return Tensor<value_t> Quantiles with the broadcasted shape.
 *
 * @throws std::invalid_argument if any input tensor is empty or contains NaN.
 * @throws std::invalid_argument if any element of `k` is non-positive.
 * @throws std::invalid_argument if any `q` value is outside [0,1].
 * @throws std::runtime_error if a non-finite result (overflow or Inf) is
 *         produced by the underlying kernel or if other numeric/device
 *         errors occur during computation.
 */
template<typename value_t>
Tensor<value_t> ppf(const Tensor<value_t>& q,
    const Tensor<value_t>& k);
/// Explicit instantiation of chisquare::ppf for float
extern template Tensor<float> ppf<float>
(const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Inverse survival function (ISF) / upper-tail quantile of
 * the chi-square.
 *
 * Computes the inverse survival function element-wise using the identity
 *     isf(q; k) = ppf(1 - q; k)
 * with standard broadcasting of `q` and `k` to determine the output shape.
 *
 * For a scalar q this returns the value x such that
 *     q = 1 - CDF(x; k)
 * i.e. the returned x satisfies CDF(x; k) = 1 - q.
 *
 * Inputs `q` (upper-tail probabilities) and `k` (degrees of freedom) must be
 * non-empty and broadcastable. Values of `q` must lie in [0,1], and `k` values
 * must be strictly positive. Note that `q == 0` maps to ppf(1), which may lead
 * to infinite intermediate values and is treated as a numeric error by this
 * implementation.
 *
 * @param q Probabilities in [0,1], broadcastable to the output shape. Must be
 *          non-empty.
 * @param k Degrees of freedom, must be non-empty and strictly positive.
 * @return Tensor<value_t> Quantiles (ISF values) with the broadcasted shape.
 *
 * @throws std::invalid_argument if any input tensor is empty, if any `q`
 *         value is outside [0,1], if any `k` element is non-positive, or if
 *         NaN is detected in the inputs.
 * @throws std::runtime_error if a non-finite result (overflow or Inf) or
 *         other numeric/device error occurs during computation.
 */
template<typename value_t>
Tensor<value_t> isf(const Tensor<value_t>& q,
    const Tensor<value_t>& k);
/// Explicit instantiation of chisquare::isf for float
extern template Tensor<float> isf<float>
(const Tensor<float>&, const Tensor<float>&);

/**
 * @brief Draw samples from the chi-square distribution.
 *
 * Generates samples from ChiSquare(k) with shape `out_shape`. The implementation
 * generates uniform variates on the device using a xorshift64* style generator,
 * clamps them to the open interval (1e-16, 1-1e-16) to avoid extreme
 * probabilities, and maps them through the chi-square ppf:
 *     samples = ppf(u, k)
 *
 * Inputs `k` (degrees of freedom) is broadcastable to `out_shape`. The returned
 * tensor has shape `out_shape` and is allocated in `res_loc`.
 *
 * @param k Degrees of freedom tensor. Must be non-empty.
 * @param out_shape Desired output shape for the samples. Must be non-empty.
 * @param res_loc MemoryLocation where the result will be allocated.
 * @param seed RNG seed. If zero the implementation seeds from
 * std::random_device; non-zero seeds produce deterministic output.
 * @return Tensor<value_t> Tensor of shape `out_shape` containing chi-square
 *         samples.
 *
 * @throws std::invalid_argument if `k` is empty, if `out_shape` is empty, or if
 *         inputs contain NaN.
 * @throws std::runtime_error if numeric or device errors occur during uniform
 *         variate generation or other internal computations.
 */
template<typename value_t>
Tensor<value_t> rvs(const Tensor<value_t>& k,
    const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc = MemoryLocation::DEVICE,
    uint64_t seed = 0ULL);
/// Explicit instantiation of chisquare::rvs for float
extern template Tensor<float> rvs<float>(const Tensor<float>&,
const std::vector<uint64_t>&, MemoryLocation, uint64_t);

    /*todo
    logpdf
    mean
    var
    std
    */
} // namespace chisquare

} // namespace temper::stats

#endif // TEMPER_STATS_HPP
