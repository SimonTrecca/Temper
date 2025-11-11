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
    MemoryLocation res_loc,
    uint64_t seed = 0ULL);
/// Explicit instantiation of norm::rvs for float
extern template Tensor<float> rvs<float>(const Tensor<float>&,
const Tensor<float>&, const std::vector<uint64_t>&, MemoryLocation, uint64_t);
        /*todo
        isf
        logpdf
        mean
        var
        std
        */
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
    } // namespace chisquare

} // namespace temper::stats

#endif // TEMPER_STATS_HPP
