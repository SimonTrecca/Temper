/**
 * @file Utils.hpp
 * @brief General-purpose tensor utility functions.
 *
 * Provides helpers for tensor shape alignment, broadcasting,
 * stride/divisor computation, and index translation.
 */

#ifndef TEMPER_UTILS_HPP
#define TEMPER_UTILS_HPP

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <numeric>

namespace temper::utils
{

/**
 * @brief Descriptor of a tensor's layout.
 *
 * Encapsulates the shape, strides, and optionally
 * divisors (used for index calculations).
 */
struct TensorDesc
{
    std::vector<uint64_t> shape;    ///< Sizes of each dimension.
    std::vector<uint64_t> strides;  ///< Strides for each dimension.
    std::vector<uint64_t> divisors; ///< Precomputed divisors for fast indexing.
};

/**
 * @brief Broadcast metadata and broadcast-aware strides.
 *
 * Contains the output tensor descriptor (shape and divisors) that
 * results from broadcasting two aligned tensors, together with the
 * strides to use for operands A and B after broadcasting. Strides
 * set to 0 indicate the dimension is broadcasted.
 */
struct BroadcastResult
{
    TensorDesc out;                  ///< Output tensor descriptor.
    std::vector<uint64_t> a_strides; ///< Broadcast-aware strides for A.
    std::vector<uint64_t> b_strides; ///< Broadcast-aware strides for B.
};

/**
 * @brief Precompute divisors for fast index translation.
 *
 * @param shape The tensor shape.
 * @return A vector of divisors matching the rank.
 */
inline std::vector<uint64_t>
compute_divisors(const std::vector<uint64_t>& shape)
{
    int64_t rank = static_cast<uint64_t>(shape.size());
    std::vector<uint64_t> divs(rank, 1);

    for (int64_t i = 0; i < rank; ++i)
    {
        uint64_t prod = 1;
        for (int64_t j = i + 1; j < rank; ++j)
        {
            prod *= shape[j];
        }
        divs[i] = prod;
    }
    return divs;
}

/**
 * @brief Align a tensor descriptor to a target rank by left-padding.
 *
 * The input descriptor is left-padded with dimensions of size 1 and
 * strides of 0 so that the returned descriptor has exactly `rank`
 * dimensions.
 *
 * @param t The input tensor descriptor.
 * @param rank The desired rank.
 * @return A new descriptor with shape and strides expanded to `rank`.
 */
inline TensorDesc
align_tensor(const TensorDesc& t, int64_t rank)
{
    TensorDesc out;
    out.shape.assign(rank, 1);
    out.strides.assign(rank, 0);

    int64_t offset = rank - static_cast<int64_t>(t.shape.size());
    for (int64_t i = 0; i < static_cast<int64_t>(t.shape.size()); ++i)
    {
        out.shape[offset + i]   = t.shape[i];
        out.strides[offset + i] = t.strides[i];
    }
    return out;
}

/**
 * @brief Compute broadcast shape/divisors and broadcast-aware strides.
 *
 * Both inputs must already be aligned to the same rank (left-padded to
 * equal length). For each dimension a.shape[d] and b.shape[d] must be
 * either equal or one of them must be 1. If a dimension of an operand
 * is 1, its stride for that dimension becomes 0 in the result.
 *
 * @param a Left operand descriptor (aligned).
 * @param b Right operand descriptor (aligned).
 * @return BroadcastResult containing output descriptor and strides.
 * @throws std::invalid_argument If descriptors have differing rank or
 *         incompatible sizes for broadcasting.
 */
inline BroadcastResult
compute_broadcast(const TensorDesc& a, const TensorDesc& b)
{
    if (a.shape.size() != b.shape.size())
    {
        throw std::invalid_argument(R"(compute_broadcast:
            descriptors must have same rank)");
    }

    const int64_t rank = static_cast<int64_t>(a.shape.size());
    BroadcastResult res;
    res.out.shape.resize(rank);
    res.out.strides.assign(rank, 0);
    res.a_strides.assign(rank, 0);
    res.b_strides.assign(rank, 0);

    for (int64_t d = 0; d < rank; ++d)
    {
        const uint64_t asz = a.shape[d];
        const uint64_t bsz = b.shape[d];

        if (asz == bsz)
        {
            res.out.shape[d] = asz;
            res.a_strides[d] = a.strides[d];
            res.b_strides[d] = b.strides[d];
        }
        else if (asz == 1)
        {
            res.out.shape[d] = bsz;
            res.a_strides[d] = 0;
            res.b_strides[d] = b.strides[d];
        }
        else if (bsz == 1)
        {
            res.out.shape[d] = asz;
            res.a_strides[d] = a.strides[d];
            res.b_strides[d] = 0;
        }
        else
        {
            throw std::invalid_argument("compute_broadcast: incompatible shapes for broadcasting");
        }
    }

    res.out.divisors = compute_divisors(res.out.shape);

    return res;
}

/**
 * @brief Choose a power-of-two work-group size for device and axis.
 *
 * Selects the largest power-of-two work-group size `wg` satisfying:
 *  - wg <= device_max_work_group_size
 *  - wg <= preferred_cap (if preferred_cap < device max)
 *  - wg <= effective_axis_size (if effective_axis_size != 0)
 *
 * If `effective_axis_size` is 0 it is treated as 1 so this returns at
 * least a work-group size of 1.
 *
 * @note This function queries the device via q.get_device() to read
 *       sycl::info::device::max_work_group_size.
 *
 * @param q SYCL queue whose device will be queried for limits.
 * @param effective_axis_size The axis length to parallelize over. If 0,
 *                            treated as 1.
 * @param preferred_cap Optional upper bound preference for wg (default
 *                      256). The chosen size will not exceed this.
 * @return The selected work-group size (power of two, >= 1).
 */
inline size_t compute_pow2_workgroup_size(
    const sycl::queue &q,
    size_t effective_axis_size,
    size_t preferred_cap = 256)
{
    auto dev = q.get_device();
    size_t device_max = dev.get_info<sycl::info::device::max_work_group_size>();

    size_t cap = device_max;
    if (preferred_cap < cap) cap = preferred_cap;

    size_t axis_size_for_wg = 1;
    if (effective_axis_size != 0) axis_size_for_wg = effective_axis_size;

    size_t max_allowed = cap;
    if (axis_size_for_wg < cap) max_allowed = axis_size_for_wg;

    size_t wg = 1;
    while (wg * 2 <= max_allowed) wg *= 2;

    return wg;
}

/**
 * @brief Compute both work-group size and number of groups.
 *
 * Uses compute_pow2_workgroup_size to pick a power-of-two work-group
 * size `wg`, then computes the number of groups as:
 *
 *   num_groups = ceil(effective_axis_size / wg)
 *
 * Special cases:
 * - If effective_axis_size is 0, num_groups is returned as 1.
 * - Ensures num_groups is at least 1.
 *
 * @param q SYCL queue whose device will be queried for limits.
 * @param effective_axis_size The axis length to parallelize over. If 0,
 *                            treated specially and yields 1 group.
 * @param preferred_cap Optional upper bound preference for wg (default
 *                      256).
 * @return A pair {wg, num_groups} where wg is the chosen work-group
 *         size and num_groups is the computed dispatch group count.
 */
inline std::pair<size_t,size_t> compute_wg_and_groups(
    const sycl::queue &q,
    size_t effective_axis_size,
    size_t preferred_cap = 256)
{
    size_t wg = compute_pow2_workgroup_size(q, effective_axis_size, preferred_cap);

    size_t axis_size_for_wg = 1;
    if (effective_axis_size != 0) axis_size_for_wg = effective_axis_size;

    size_t num_groups = 0;
    if (axis_size_for_wg == 0) {
        num_groups = 1;
    } else {
        num_groups = (axis_size_for_wg + wg - 1) / wg;
    }
    if (num_groups == 0) num_groups = 1;

    return {wg, num_groups};
}

} // namespace temper::utils

#endif // TEMPER_UTILS_HPP
