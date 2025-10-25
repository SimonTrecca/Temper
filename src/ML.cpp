/**
 * @file ML.cpp
 * @brief Machine learning utility definitions.
 */

#include "temper/ML.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Utils.hpp"
#include "temper/Math.hpp"

namespace temper::ml
{
template <typename float_t>
Tensor<float_t> one_hot_expand_at(const Tensor<float_t>& tensor,
	int64_t axis,
	uint64_t axis_index,
	uint64_t depth,
	float_t on_value,
	float_t off_value)
{
    if (depth == 0)
    {
        throw std::invalid_argument("one_hot_expand_at: depth must be > 0.");
    }

    const int64_t rank = tensor.get_rank();
    if (rank == 0)
    {
        throw std::invalid_argument(R"(one_hot_expand_at:
			input tensor has no elements.)");
    }

    if (axis < 0)
    {
        axis += rank;
    }
    if (axis < 0 || axis >= rank)
    {
        throw std::invalid_argument("one_hot_expand_at: axis out of range.");
    }

    const std::vector<uint64_t> in_shape = tensor.get_dimensions();
    if (axis_index >= in_shape[axis])
    {
        throw std::out_of_range("one_hot_expand_at: axis_index out of range.");
    }

    // Build output shape: we remove one element from axis (the label)
    // and expand to 'depth' slots: out_axis_len = (in_axis_len - 1) + depth.
    std::vector<uint64_t> out_shape = in_shape;
    out_shape[axis] = (in_shape[axis] - 1) + depth;

    MemoryLocation res_loc = tensor.get_memory_location();
    Tensor<float_t> result(out_shape, res_loc);

    const std::vector<uint64_t> in_divs =
    	temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = tensor.get_strides();
    const std::vector<uint64_t> out_divs =
    	temper::utils::compute_divisors(out_shape);

    uint64_t* p_in_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_in_shape = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_out_shape = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_in_divs && p_in_strides && p_in_shape &&
                     p_out_divs && p_out_shape && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_in_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_in_shape, g_sycl_queue);
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_out_shape, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_in_divs,
    	in_divs.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_in_strides,
    	in_strides.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_in_shape,
    	in_shape.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_out_divs,
    	out_divs.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_out_shape,
    	out_shape.data(), sizeof(uint64_t) * rank).wait();

    *p_error_flag = 0;

    const uint64_t total_in_elems = tensor.get_num_elements();
    const uint64_t total_out_elems = result.get_num_elements();

    const float_t integer_eps = static_cast<float_t>(1e-3);

    const float_t* p_in_data = tensor.get_data();
    float_t* p_out_data = result.get_data();

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
            float_t in_val = p_in_data[in_idx];

            temper::sycl_utils::device_check_nan_and_set<float_t>
            	(in_val, p_error_flag);
            if (*p_error_flag != 0) { return; }

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
                temper::sycl_utils::device_check_finite_and_set<float_t>
                	(in_val, p_error_flag);
                if (*p_error_flag != 0) { return; }

                // Integer check.
                float_t rounded = sycl::round(in_val);
                float_t diff = sycl::fabs(in_val - rounded);
                if (diff > integer_eps)
                {
                    p_error_flag[0] = 3;
                    return;
                }

                int64_t lbl_ll = static_cast<int64_t>(rounded);
                if (lbl_ll < 0 || static_cast<uint64_t>(lbl_ll) >= depth)
                {
                    p_error_flag[0] = 3;
                    return;
                }
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

    sycl::free(p_in_divs, g_sycl_queue);
    sycl::free(p_in_strides, g_sycl_queue);
    sycl::free(p_in_shape, g_sycl_queue);
    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_out_shape, g_sycl_queue);

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(one_hot_expand_at:
            	NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(one_hot_expand_at:
            	non-finite result produced.)");
        }
        if (err == 3)
        {
            throw std::runtime_error(R"(one_hot_expand_at:
            	label non-integer or out of range.)");
        }
        throw std::runtime_error(R"(one_hot_expand_at:
        	numeric error during computation.)");
    }

    return result;
}
template Tensor<float> one_hot_expand_at<float>
    (const Tensor<float>&, int64_t, uint64_t, uint64_t, float, float);

template<typename float_t>
Tensor<float_t> softmax(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt)
{
    const int64_t rank = tensor.get_rank();
    if (rank == 0)
    {
        throw std::invalid_argument(R"(softmax:
            input tensor has no elements.)");
    }

    const bool flatten = !axis_opt.has_value();
    if (!flatten)
    {
        int64_t axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || axis >= rank)
        {
            throw std::invalid_argument(R"(softmax: axis out of range.)");
        }
    }

    Tensor<float_t> ex = math::exp(tensor);
    Tensor<float_t> denom = math::sum(ex, axis_opt);
    Tensor<float_t> out = ex / denom;

    return out;
}
template Tensor<float> softmax<float>
    (const Tensor<float>&, std::optional<int64_t>);

} // namespace temper::ml