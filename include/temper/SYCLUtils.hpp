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
    if (sycl_utils::is_nan(v))
    {
        auto atomic_err = sycl::atomic_ref<int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*p_err);
        int32_t expected = 0;
        atomic_err.compare_exchange_strong(expected, 1);
    }
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
    if (!sycl_utils::is_finite(v))
    {
        auto atomic_err = sycl::atomic_ref<int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*p_err);
        int32_t expected = 0;
        atomic_err.compare_exchange_strong(expected, 2);
    }
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
