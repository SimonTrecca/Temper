/**
 * @file ML.cpp
 * @brief Machine learning utility definitions.
 */

#include "temper/ML.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Utils.hpp"
#include "temper/Errors.hpp"
#include "temper/Math.hpp"

namespace temper::ml
{
template <typename value_t>
Tensor<value_t> one_hot_expand_at(const Tensor<value_t>& tensor,
	int64_t axis,
	uint64_t axis_index,
	uint64_t depth,
	value_t on_value,
	value_t off_value)
{
    TEMPER_CHECK(depth == 0,
        validation_error,
        "one_hot_expand_at: depth must be > 0.");

    const int64_t rank = tensor.get_rank();
    TEMPER_CHECK(rank == 0,
        validation_error,
        R"(one_hot_expand_at: input tensor has no elements.)");

    if (axis < 0)
    {
        axis += rank;
    }
     TEMPER_CHECK(axis < 0 || axis >= rank,
        bounds_error,
        "one_hot_expand_at:  axis out of range.");

    const std::vector<uint64_t> & in_shape = tensor.get_dimensions();
    TEMPER_CHECK(axis_index >= in_shape[axis],
        bounds_error,
        "one_hot_expand_at: axis_index out of range.");

    // Build output shape: we remove one element from axis (the label)
    // and expand to 'depth' slots: out_axis_len = (in_axis_len - 1) + depth.
    std::vector<uint64_t> out_shape = in_shape;
    out_shape[axis] = (in_shape[axis] - 1) + depth;

    MemoryLocation res_loc = tensor.get_memory_location();
    Tensor<value_t> result(out_shape, res_loc);

    const std::vector<uint64_t> in_divs =
    	temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = tensor.get_strides();
    const std::vector<uint64_t> out_divs =
    	temper::utils::compute_divisors(out_shape);

    sycl_utils::SyclArray<uint64_t> in_divs_arr(g_sycl_queue,
        in_divs, MemoryLocation:: DEVICE);
    sycl_utils::SyclArray<uint64_t> in_strides_arr(g_sycl_queue,
        in_strides, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> in_shape_arr(g_sycl_queue,
        in_shape, MemoryLocation:: DEVICE);
    sycl_utils::SyclArray<uint64_t> out_divs_arr(g_sycl_queue,
        out_divs, MemoryLocation:: DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_arr(g_sycl_queue,
        1, MemoryLocation::HOST);

    const uint64_t* p_in_divs = in_divs_arr;
    const uint64_t* p_in_strides = in_strides_arr;
    const uint64_t* p_in_shape = in_shape_arr;
    const uint64_t* p_out_divs = out_divs_arr;
    int32_t* p_error_flag = error_flag_arr;

    *p_error_flag = 0;

    const uint64_t total_in_elems = tensor.get_num_elements();
    const uint64_t total_out_elems = result.get_num_elements();

    const value_t integer_eps = static_cast<value_t>(1e-3);

    const value_t* p_in_data = tensor.get_data();
    value_t* p_out_data = result.get_data();

    // Initialize output to off_value.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_out_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);
            p_out_data[flat] = off_value;
        });
    }).wait();

    // Either copy into shifted output or set on_value.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_in_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t in_idx =
            	temper::sycl_utils::idx_of(flat, p_in_divs, p_in_strides, rank);
            value_t in_val = p_in_data[in_idx];

            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(in_val), p_error_flag, 1);

            uint64_t coord_a = (flat / p_in_divs[axis]) % p_in_shape[axis];

            if (coord_a != axis_index)
            {
                uint64_t dest_flat = 0;
                for (int64_t d = 0; d < rank; ++d)
                {
                    uint64_t coord = (flat / p_in_divs[d]) % p_in_shape[d];
                    uint64_t dest_coord;
                    if (d == axis)
                    {
                        if (coord < axis_index)
                        {
                            dest_coord = coord;
                        }
                        else
                        {
                            dest_coord = (coord - 1) + depth;
                        }
                    }
                    else
                    {
                        dest_coord = coord;
                    }
                    dest_flat += dest_coord * p_out_divs[d];
                }
                p_out_data[dest_flat] = in_val;
            }
            else
            {
                // Integer check.
                value_t rounded = sycl_utils::round(in_val);
                value_t diff = sycl_utils::fabs(in_val - rounded);

                TEMPER_DEVICE_CHECK(diff > integer_eps, p_error_flag, 3);

                int64_t lbl_ll = static_cast<int64_t>(rounded);

                TEMPER_DEVICE_CHECK(lbl_ll < 0 ||
                    static_cast<uint64_t>(lbl_ll) >= depth, p_error_flag, 3);

                const uint64_t lbl = static_cast<uint64_t>(lbl_ll);

                uint64_t dest_flat = 0;
                for (int64_t d = 0; d < rank; ++d)
                {
                    uint64_t coord = (flat / p_in_divs[d]) % p_in_shape[d];
                    uint64_t dest_coord;
                    if (d == axis)
                    {
                        dest_coord = axis_index + lbl;
                    }
                    else
                    {
                        dest_coord = coord;
                    }
                    dest_flat += dest_coord * p_out_divs[d];
                }

                p_out_data[dest_flat] = on_value;
            }
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(one_hot_expand_at: NaN detected in inputs.)");

    TEMPER_CHECK(err == 3,
        validation_error,
        R"(one_hot_expand_at: label non-integer or out of range.)");

    return result;
}
template Tensor<float> one_hot_expand_at<float>
    (const Tensor<float>&, int64_t, uint64_t, uint64_t, float, float);
template Tensor<uint64_t> one_hot_expand_at<uint64_t>
    (const Tensor<uint64_t>&, int64_t, uint64_t, uint64_t, uint64_t, uint64_t);

template<typename value_t>
Tensor<value_t> softmax(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt)
{
    const int64_t rank = tensor.get_rank();
    TEMPER_CHECK(rank == 0,
        validation_error,
        R"(softmax: input tensor has no elements.)");

    const bool flatten = !axis_opt.has_value();
    if (!flatten)
    {
        int64_t axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }
        TEMPER_CHECK(axis < 0 || axis >= rank,
            bounds_error,
            R"(softmax: axis out of range.)");
    }

    Tensor<value_t> ex = math::exp(tensor);
    Tensor<value_t> denom = math::sum(ex, axis_opt);
    Tensor<value_t> out = ex / denom;

    return out;
}
template Tensor<float> softmax<float>
    (const Tensor<float>&, std::optional<int64_t>);

template<typename value_t>
Tensor<value_t> cross_entropy(const Tensor<value_t> & logits,
    const Tensor<value_t> & labels,
    std::optional<int64_t> axis_opt,
    bool from_logits,
    bool reduction_mean)
{
    const int64_t rank_logits = logits.get_rank();
    TEMPER_CHECK(rank_logits == 0,
        validation_error,
        R"(cross_entropy: input logits tensor has no elements.)");

    const int64_t rank_labels = labels.get_rank();
    TEMPER_CHECK(rank_labels == 0,
        validation_error,
        R"(cross_entropy: labels tensor has no elements.)");

    const bool flatten = !axis_opt.has_value();
    const int64_t max_rank = std::max(rank_logits, rank_labels);

    std::optional<int64_t> axis_norm;
    std::optional<int64_t> axis_aligned;

    if (flatten)
    {
        axis_norm = std::nullopt;
        axis_aligned = std::nullopt;
    }
    else
    {
        int64_t axis = axis_opt.value();
        if (axis < 0)
        {
            axis += max_rank;
        }
        TEMPER_CHECK(axis < 0 || axis >= max_rank,
            bounds_error,
            R"(cross_entropy: axis out of bounds (aligned shape))");
        axis_aligned = axis;

        int64_t axis_for_logits = axis - (max_rank - rank_logits);
        if (from_logits)
        {
            TEMPER_CHECK(axis_for_logits < 0 || axis_for_logits >= rank_logits,
                bounds_error,
                R"(cross_entropy:
                    axis does not exist on logits (required for softmax))");
            axis_norm = axis_for_logits;
        }
        else
        {
            axis_norm = axis_for_logits;
        }
    }

    Tensor<value_t> probs;
    if (from_logits)
    {
        probs = softmax(logits, axis_norm);
    }
    else
    {
        probs = logits;
    }

    Tensor<value_t> logp = temper::math::log(probs);
    Tensor<value_t> mul = labels * logp;

    Tensor<value_t> summed = temper::math::sum(mul, axis_aligned);
    Tensor<value_t> loss = -summed;

    if (reduction_mean)
    {
        return temper::math::mean(loss, std::nullopt);
    }
    else
    {
        return loss;
    }
}
template Tensor<float> cross_entropy<float>
(const Tensor<float>&, const Tensor<float>&, std::optional<int64_t>, bool, bool);

template<typename value_t>
Tensor<value_t> mean_squared_error(const Tensor<value_t>& predictions,
    const Tensor<value_t>& targets,
    std::optional<int64_t> axis_opt,
    bool reduction_mean)
{
    const int64_t rank_pred = predictions.get_rank();
    const int64_t rank_tgt = targets.get_rank();
    TEMPER_CHECK(rank_pred == 0,
        validation_error,
        R"(mean_squared_error: predictions tensor has no elements.)");

    TEMPER_CHECK(rank_tgt == 0,
        validation_error,
        R"(mean_squared_error: targets tensor has no elements.)");

    const int64_t max_rank = std::max(rank_pred, rank_tgt);
    const bool flatten = !axis_opt.has_value();

    std::optional<int64_t> axis_aligned;
    if (flatten)
    {
        axis_aligned = std::nullopt;
    }
    else
    {
        int64_t axis = axis_opt.value();
        if (axis < 0)
        {
            axis += max_rank;
        }
        TEMPER_CHECK(axis < 0 || axis >= max_rank,
            bounds_error,
            R"(mean_squared_error: axis out of bounds (aligned shape))");
        axis_aligned = axis;
    }

    Tensor<value_t> diff = predictions - targets;
    Tensor<value_t> sq = diff * diff;

    Tensor<value_t> summed = temper::math::sum(sq, axis_aligned);

    if (reduction_mean)
    {
        return temper::math::mean(summed, std::nullopt);
    }
    else
    {
        return summed;
    }
}
template Tensor<float> mean_squared_error<float>
(const Tensor<float>&, const Tensor<float>&, std::optional<int64_t>, bool);

template<typename value_t>
PCAResult<value_t> pca(const Tensor<value_t> & data,
    std::optional<uint64_t> n_components,
    bool standardize)
{
    const int64_t rank = data.get_rank();

    TEMPER_CHECK(rank < 2,
        validation_error,
        R"(pca: rank must be >= 2.)");

    uint64_t k = data.get_dimensions()[rank - 1];
    if (n_components.has_value())
    {
        uint64_t value = n_components.value();
        // Check whether the number of components given is valid.
        TEMPER_CHECK(value > k || value == 0,
            validation_error,
            R"(pca: invalid number of components.)");
        k = value;
    }

    Tensor<value_t> data_scaled;
    if (standardize)
    {
        Tensor<value_t> data_mean = math::mean(data, -2);
        Tensor<value_t> data_std = math::stddev(data, -2, /*ddof=*/1);
        data_scaled = (data - data_mean) / data_std;
    }
    else
    {
        Tensor<value_t> data_mean = math::mean(data, -2);
        data_scaled = data - data_mean;
    }

    Tensor<value_t> data_cov = math::cov(data_scaled, /*ddof=*/1);

    auto eigs = math::eig(data_cov, 100);

    Tensor<uint64_t> sort_order = math::argsort(eigs.first, -1, true);

    std::vector<uint64_t> k_eigvals_shape = eigs.first.get_dimensions();
    k_eigvals_shape[rank - 1] = k;

    std::vector<uint64_t> k_eigvecs_shape = eigs.second.get_dimensions();
    k_eigvecs_shape[rank - 1] = k;

    Tensor<value_t> eigvals = math::gather(eigs.first, sort_order, -1);
    Tensor<value_t> eigvecs = math::gather(eigs.second, sort_order, -1);

    PCAResult<value_t> result;

    std::vector<uint64_t> start_indices(rank, 0);

    Tensor<value_t> eigvals_k_view(eigvals, start_indices, k_eigvals_shape);

    result.explained_variance = std::move(eigvals_k_view);

    Tensor<value_t> eigvecs_k_view(eigvecs, start_indices, k_eigvecs_shape);
    result.loadings = std::move(eigvecs_k_view);
    result.projections = math::matmul(data_scaled, result.loadings);

    return result;
}
template PCAResult<float> pca<float>
(const Tensor<float>&, std::optional<uint64_t>, bool);

} // namespace temper::ml