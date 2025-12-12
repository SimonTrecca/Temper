/**
 * @file SYCLUtils.hpp
 * @brief Small, inline helpers for use inside SYCL kernels.
 *
 */

#ifndef TEMPER_SYCLUTILS_HPP
#define TEMPER_SYCLUTILS_HPP

#include <sycl/sycl.hpp>
#include <cstdint>

namespace temper::sycl_utils
{

/**
 * @brief Map a logical row-major linear index to a physical offset using
 *        precomputed divisors and strides.
 *
 * @param logical_idx linear index (0..N-1)
 * @param p_divisors  pointer to divisors array (length == rank)
 * @param p_strides   pointer to strides array (length == rank)
 * @param rank        number of dimensions
 * @return offset into the underlying flat data buffer
 */
inline uint64_t idx_of(uint64_t logical_idx,
                       const uint64_t* p_divisors,
                       const uint64_t* p_strides,
                       int64_t rank)
{
    uint64_t rem = logical_idx;
    uint64_t off = 0;
    for (int64_t d = 0; d < rank; ++d)
    {
        uint64_t div = p_divisors[d];
        uint64_t coord = 0;
        if (div != 0)
        {
            coord = rem / div;
            rem = rem % div;
        }
        off += coord * p_strides[d];
    }
    return off;
}

/**
 * @brief Safe isnan wrapper that is valid for both floating and integer types.
 *
 * For floating types it forwards to sycl::isnan(). For integral types it
 * returns false (integers can't be NaN).
 */
template <typename value_t>
inline bool is_nan(value_t v)
{
    if constexpr (std::is_floating_point_v<value_t>)
    {
        return sycl::isnan(v);
    }
    else
    {
        (void)v;
        return false;
    }
}

/**
 * @brief Safe isfinite wrapper for floating/integral types.
 *
 * For floating types it forwards to sycl::isfinite(). For integral types it
 * returns true (integers are always finite).
 */
template <typename value_t>
inline bool is_finite(value_t v)
{
    if constexpr (std::is_floating_point_v<value_t>)
    {
        return sycl::isfinite(v);
    }
    else
    {
        (void)v;
        return true;
    }
}

/**
 * @brief Safe sqrt wrapper.
 *
 * For floating types it forwards to sycl::sqrt(). For integral types it casts
 * to a floating type, computes the sqrt, then casts back (flooring the result).
 * If you prefer other semantics for integers, change the cast or behavior.
 */
template <typename value_t>
inline value_t sqrt(value_t v)
{
    if constexpr (std::is_floating_point_v<value_t>)
    {
        return sycl::sqrt(v);
    }
    else
    {
        float tmp = sycl::sqrt(static_cast<double>(v));
        return static_cast<value_t>(tmp);
    }
}

/**
 * @brief Safe power wrapper.
 *
 * For floating types, forwards to sycl::pow(). For integral types, uses
 * fast exponentiation by squaring. Negative integer exponents are computed
 * via floating-point promotion.
 *
 * @param a Base value
 * @param b Exponent value
 * @return a raised to the power b
 */
template <typename value_t>
inline value_t pow(value_t a, value_t b)
{
    if constexpr (std::is_floating_point_v<value_t>)
    {
        return sycl::pow(a, b);
    }
    else
    {
        if constexpr (std::is_signed_v<value_t>)
        {
            if (b < 0)
            {
                double r = sycl::pow(static_cast<double>(a),
                                     static_cast<double>(b));
                return static_cast<value_t>(r);
            }
        }
        value_t base = a;
        value_t exp = b;
        value_t acc = 1;
        while (exp)
        {
            if (exp & 1) acc = acc * base;
            exp >>= 1;
            if (exp) base = base * base;
        }
        return acc;
    }
}

/**
 * @brief Safe fabs wrapper.
 *
 * For floating types forwards to sycl::fabs; for integral types returns the
 * absolute value using ordinary integer ops.
 */
template <typename value_t>
inline value_t fabs(value_t v)
{
    if constexpr (std::is_floating_point_v<value_t>)
    {
        return sycl::fabs(v);
    }
    else
    {
        if constexpr (std::is_signed_v<value_t>)
        {
            if (v < 0)
            {
                return -v;
            }
            else
            {
                return v;
            }
        }
        else
        {
            return v;
        }
    }
}

/**
 * @brief Safe floor wrapper.
 *
 * For floating types it forwards to sycl::floor(). For integral types it
 * returns the input unchanged (floor of an integer is itself). If you prefer
 * different semantics (e.g. cast to floating and floor), change accordingly.
 */
template <typename value_t>
inline value_t floor(value_t v)
{
    if constexpr (std::is_floating_point_v<value_t>)
    {
        return sycl::floor(v);
    }
    else
    {

        (void)v;
        return v;
    }
}

/**
 * @brief Safe round wrapper.
 *
 * For floating types it forwards to sycl::round(). For integral types it
 * returns the input unchanged (round of an integer is itself). Adjust if you
 * want different semantics for integers.
 */
template <typename value_t>
inline value_t round(value_t v)
{
    if constexpr (std::is_floating_point_v<value_t>)
    {
        return sycl::round(v);
    }
    else
    {
        (void)v;
        return v;
    }
}

/**
 * @brief Atomically set an error flag if a NaN is observed.
 *
 * @param v value to test
 * @param p_err pointer to int32_t error flag in shared/global memory
 *
 * Error codes:
 *   1 -> NaN detected
 */
template<typename value_t>
inline void device_check_nan_and_set(value_t v, int32_t* p_err)
{
#if !defined(TEMPER_DISABLE_ERROR_CHECKS)
    if (sycl_utils::is_nan(v))
    {
        auto atomic_err = sycl::atomic_ref<int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*p_err);
        int32_t expected = 0;
        atomic_err.compare_exchange_strong(expected, 1);
    }
#else
    (void)v;
    (void)p_err;
#endif
}

/**
 * @brief Atomically set an error flag if a value is non-finite (Inf/Nan).
 *
 * @param v value to test
 * @param p_err pointer to int32_t error flag in shared/global memory
 *
 * Error codes:
 *   2 -> non-finite detected (Inf/overflow/result)
 */
template<typename value_t>
inline void device_check_finite_and_set(value_t v, int32_t* p_err)
{
#if !defined(TEMPER_DISABLE_ERROR_CHECKS)
    if (!sycl_utils::is_finite(v))
    {
        auto atomic_err = sycl::atomic_ref<int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*p_err);
        int32_t expected = 0;
        atomic_err.compare_exchange_strong(expected, 2);
    }
#else
    // checks disabled: avoid unused-parameter warnings and ensure no side-effects
    (void)v;
    (void)p_err;
#endif
}

inline double erfinv(double x)
{
    const double a[] = {
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    };

    const double b[] = {
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    };

    const double c[] = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    };

    const double d[] = {
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    };

    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;

    if (x <= -1.0) return -std::numeric_limits<double>::infinity();
    if (x >=  1.0) return  std::numeric_limits<double>::infinity();

    double p = (x + 1.0) * 0.5;

    double q, r;
    if (p < p_low)
    {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
    else if (p <= p_high)
    {
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    }
    else
    {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
}

/**
* @brief Compute the lower regularized incomplete gamma function P(s, x).
*
* This function calculates the regularized lower incomplete gamma function
* P(s, x) = (1 / Gamma(s)) * integral_0^x t^{s-1} e^{-t} dt.
*
* Implementation notes:
* - The function operates in log-space for normalization using std::lgamma()
* to avoid overflow/underflow when s or x are large.
* - For x < s + 1 a series expansion (convergent for small x) is used.
* - For x >= s + 1 Lentz's continued fraction is used to evaluate the
* complementary value Q(s,x) = Gamma(s,x)/Gamma(s) and P = 1 - Q.
*
* @param s Shape parameter (must be > 0).
* @param x Upper limit of integration (must be >= 0).
* @return P(s, x) in [0,1], or NaN if inputs are invalid.
*/
inline double regularized_gamma(double s, double x)
{
    if (s <= 0.0 || x < 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return 0.0;

    const int MAX_ITER = 2000;
    const double EPS = std::numeric_limits<double>::epsilon();
    const double FPMIN = std::numeric_limits<double>::min() / EPS;

    double log_front = -x + s * std::log(x) - std::lgamma(s);

    if (x < s + 1.0)
    {
        double sum = 1.0 / s;
        double term = sum;
        for (int n = 1; n < MAX_ITER; ++n)
        {
            term *= x / (s + n);
            sum += term;
            if (std::fabs(term) < EPS * std::fabs(sum)) break;
        }
        double front = std::exp(log_front);
        double result = front * sum;

        if (!std::isfinite(result))
        {
            if (log_front < -700) return 0.0;
            return std::numeric_limits<double>::quiet_NaN();
        }
        return result;
    }

    double b = x + 1.0 - s;
    double c = 1.0 / FPMIN;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < MAX_ITER; ++i)
    {
        double an = -static_cast<double>(i) * (static_cast<double>(i) - s);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) < EPS) break;
    }

    double front = std::exp(log_front);
    double Q = front * h;
    double P = 1.0 - Q;
    if (P < 0.0 && P > -1e-16) P = 0.0;
    return P;
}

/**
* @brief Compute the inverse of the lower regularized incomplete gamma.
*
* Solve for x given a>0 and 0<p<1 such that P(a, x) == p. The routine
* returns x in (0, +inf). The solver uses a Wilson–Hilferty initial guess
* for a>1 and a small-a heuristic otherwise. A hybrid Newton + bisection
* iteration follows, with the PDF computed in log-space to preserve
* numerical stability.
*
* @param a Shape parameter (must be > 0).
* @param p Target probability in (0,1).
* @return x such that P(a, x) = p, or NaN/inf for invalid inputs.
*/
inline double inverse_regularized_gamma(double a, double p)
{
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    if (a <= 0.0) return std::numeric_limits<double>::quiet_NaN();

    const int MAX_ITER = 200;
    const double TOL = 1e-14;

    double x;
    if (a > 1.0)
    {
        double z = std::sqrt(2.0) * erfinv(2.0 * p - 1.0);
        double w = 1.0 - 1.0 / (9.0 * a) + z * std::sqrt(1.0 / (9.0 * a));
        x = a * w * w * w;
        if (x <= 0.0) x = std::numeric_limits<double>::min();
    }
    else
    {
        double log_gamma_ap1 = std::lgamma(a + 1.0);
        double log_x = (std::log(p) + log_gamma_ap1) / a;
        if (log_x < -700.0) x = std::exp(log_x);
        else x = std::exp(log_x);
        if (!(x > 0.0) || x < 1e-300)
        {
            x = std::min(1e-2, std::max(std::exp(std::log(p) / a), 1e-8));
        }
    }

    double lo = 0.0;
    double hi = x;
    if (!(hi > 0.0)) hi = 1.0;
    double P_hi = regularized_gamma(a, hi);
    int safety = 0;
    while (P_hi < p && safety < 2000)
    {
        hi *= 2.0;
        P_hi = regularized_gamma(a, hi);
        ++safety;
        if (hi > 1e300) break;
    }

    if (!std::isfinite(P_hi))
    {
        hi = std::max(hi * 1e-6, std::numeric_limits<double>::min());
        P_hi = regularized_gamma(a, hi);
        if (!std::isfinite(P_hi))
        {
            return x;
        }
    }

    if (std::fabs(P_hi - p) < 1e-16) return hi;

    double x_curr = std::min(std::max(x, std::numeric_limits<double>::min()),
        hi);
    double f_curr = regularized_gamma(a, x_curr) - p;
    if (f_curr > 0.0)
    {
        hi = x_curr;
        P_hi = regularized_gamma(a, hi);
    } else
    {
        lo = x_curr;
    }

    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        double log_pdf;
        if (x_curr <= 0.0)
        {
            log_pdf = (a - 1.0) * std::log(std::numeric_limits<double>::min()) -
                std::numeric_limits<double>::max();
        }
        else
        {
            log_pdf = (a - 1.0) * std::log(x_curr) - x_curr - std::lgamma(a);
        }
        double pdf = std::exp(std::max(-700.0, std::min(700.0, log_pdf)));
        double F = regularized_gamma(a, x_curr);
        double f = F - p;

        double x_new;
        if (pdf > 0.0)
        {
            x_new = x_curr - f / pdf;
            if (!(x_new > lo && x_new < hi) || !std::isfinite(x_new))
            {
                x_new = 0.5 * (lo + hi);
            }
        }
        else
        {
            x_new = 0.5 * (lo + hi);
        }

        double F_new = regularized_gamma(a, x_new);
        double f_new = F_new - p;

        if (f_new > 0.0)
        {
            hi = x_new;
            P_hi = F_new;
        }
        else
        {
            lo = x_new;
        }

        x_curr = x_new;
        f_curr = f_new;

        if (std::fabs(hi - lo) <= TOL * (1.0 + std::fabs(x_curr)))
            break;

        if (std::fabs(f_curr) <= 1e-15) break;
    }

    return x_curr;
}

/**
 * @brief Atomically set an error flag if divisor is zero.
 *
 * @tparam value_t Floating-point type
 * @param b_val Divisor value
 * @param p_err Pointer to int32_t error flag in shared/global memory
 *
 * Error codes:
 *   3 -> division by zero detected
 */
template<typename value_t>
inline void device_check_divzero_and_set(value_t b_val, int32_t* p_err)
{
    if (b_val == static_cast<value_t>(0))
    {
        auto atomic_err = sycl::atomic_ref<int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*p_err);

        int32_t expected = 0;
        atomic_err.compare_exchange_strong(expected, 3);
    }
}

/**
 * @brief Partition helper for merge-path parallel merge.
 *
 * Given sorted runs A=[left,mid) and B=[mid,right) (logical indices),
 * returns i in [0..len_left] such that i + j = k partitions the combined run.
 *
 * Treats NaN as larger than any numeric value (so non-NaN < NaN).
 *
 * @param k combined rank (0..total_len)
 * @param left logical start of A
 * @param mid  logical start of B (end of A)
 * @param right logical end of B
 * @param p_divs device pointer to divisors array (for idx_of)
 * @param p_strides device pointer to strides array (for idx_of)
 * @param rank number of dimensions for idx_of
 * @param merge_input pointer to data buffer
 * @return partition index i inside A (0..len_left)
 */
template<typename value_t>
inline uint64_t merge_path_partition(uint64_t k,
    uint64_t left,
    uint64_t mid,
    uint64_t right,
    const uint64_t* p_divs,
    const uint64_t* p_strides,
    int64_t rank,
    const value_t* merge_input)
{
    const uint64_t len_left  = mid - left;
    const uint64_t len_right = right - mid;

    uint64_t i_min;
    if (k > len_right)
    {
        i_min = k - len_right;
    }
    else
    {
        i_min = 0;
    }

    uint64_t i_max;
    if (k < len_left)
    {
        i_max = k;
    }
    else
    {
        i_max = len_left;
    }

    while (i_min < i_max)
    {
        uint64_t i_mid = (i_min + i_max) / 2;
        uint64_t j_mid = k - i_mid;

        uint64_t off_a = idx_of(left + i_mid, p_divs, p_strides, rank);
        uint64_t off_b = idx_of(mid + j_mid - 1, p_divs, p_strides, rank);

        value_t a = merge_input[off_a];
        value_t b = merge_input[off_b];

        bool a_is_nan = sycl_utils::is_nan(a);
        bool b_is_nan = sycl_utils::is_nan(b);

        bool cmp;
        if (!a_is_nan && b_is_nan)
        {
            cmp = true;
        }
        else if (a < b)
        {
            cmp = true;
        }
        else
        {
            cmp = false;
        }

        if (cmp)
        {
            i_min = i_mid + 1;
        }
        else
        {
            i_max = i_mid;
        }
    }

    return i_min;
}

} // namespace temper::sycl_utils

#endif // TEMPER_SYCLUTILS_HPP
