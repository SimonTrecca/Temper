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

#include "Errors.hpp"

namespace temper::utils
{

/**
 * @brief Descriptor of a tensor's layout.
 *
 * Encapsulates the shape and strides.
 */
struct TensorDesc
{
    std::vector<uint64_t> shape;    ///< Sizes of each dimension.
    std::vector<uint64_t> strides;  ///< Strides for each dimension.
};

/**
* @brief Broadcast metadata and broadcast-aware strides.
*
* Holds the result of broadcasting a set of input tensors.
* - `shape`: Output shape after broadcasting.
* - `divisors`: Precomputed divisors for indexing the broadcasted output.
* - `strides`: Per-operand, broadcast-aware strides
* (0 if the operand is broadcasted).
*/
struct BroadcastResult
{
    std::vector<uint64_t> shape;    ///< Sizes of each dimension.
    std::vector<uint64_t> divisors; ///< Precomputed divisors for fast indexing.

    /// Broadcast-aware strides.
    std::vector<std::vector<uint64_t>> strides;
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
 * @brief Compute broadcast shape, divisors, and broadcast-aware strides.
 *
 * Aligns input tensor descriptors, computes the broadcasted output shape,
 * per-operand broadcast-aware strides, and precomputed divisors for indexing.
 *
 * @param inputs A list of tensor descriptors to broadcast.
 * @return BroadcastResult containing the broadcasted shape, divisors, and
 * strides.
 *
 * @throws std::invalid_argument if `inputs` is empty or shapes are
 * incompatible.
 */
inline BroadcastResult compute_broadcast(const std::vector<TensorDesc>& inputs)
{
    TEMPER_CHECK(inputs.empty(),
        validation_error,
        "compute_broadcast: no inputs provided");

    int64_t max_rank = 0;
    for (const auto &t : inputs)
    {
        max_rank = std::max(max_rank, static_cast<int64_t>(t.shape.size()));
    }

    std::vector<TensorDesc> aligned;
    aligned.reserve(inputs.size());
    for (const auto &t : inputs)
    {
        TensorDesc a;
        a.shape.assign(max_rank, 1);
        a.strides.assign(max_rank, 0);
        int64_t t_rank = static_cast<int64_t>(t.shape.size());
        int64_t offset = max_rank - t_rank;
        for (int64_t i = 0; i < t_rank; ++i)
        {
            a.shape[offset + i] = t.shape[i];
            a.strides[offset + i] = t.strides[i];
        }
        aligned.push_back(std::move(a));
    }

    BroadcastResult res;
    res.shape.assign(max_rank, 1);
    res.strides.assign(aligned.size(), std::vector<uint64_t>(max_rank, 0));

    for (int64_t d = 0; d < max_rank; ++d)
    {
        uint64_t out_sz = 1;
        for (const auto &a : aligned)
        {
            out_sz = std::max(out_sz, a.shape[d]);
        }

        for (size_t i = 0; i < aligned.size(); ++i)
        {
            uint64_t asz = aligned[i].shape[d];
            if (asz == out_sz)
            {
                res.strides[i][d] = aligned[i].strides[d];
            }
            else if (asz == 1)
            {
                res.strides[i][d] = 0;
            }
            else
            {
                TEMPER_CHECK(true,
                    validation_error,
                    R"(compute_broadcast:
                        incompatible shapes for broadcasting)");
            }
        }
        res.shape[d] = out_sz;
    }

    res.divisors = compute_divisors(res.shape);
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
