/**
 * @file Math.cpp
 * @brief Mathematical tensor operation definitions.
 */

#include "temper/Math.hpp"

namespace temper::math
{

template <typename float_t>
Tensor<float_t> matmul(const Tensor<float_t> & first,
						const Tensor<float_t> & second)
{
    if (first.get_dimensions().empty() || second.get_dimensions().empty())
    {
        throw std::invalid_argument(R"(matmul:
			either tensor has no elements.)");
    }

    const uint64_t a_rank_orig = first.get_rank();
    const uint64_t b_rank_orig = second.get_rank();

    std::vector<uint64_t> a_shape_prom;
    std::vector<uint64_t> a_strides_prom;
    if (a_rank_orig == 1)
    {
        uint64_t K = first.get_dimensions()[0];
        a_shape_prom = {1, K};
        uint64_t orig_stride = first.get_strides()[0];
        a_strides_prom = { orig_stride * K, orig_stride };
    }
    else
    {
        a_shape_prom = first.get_dimensions();
        a_strides_prom = first.get_strides();
    }

    std::vector<uint64_t> b_shape_prom;
    std::vector<uint64_t> b_strides_prom;
    if (b_rank_orig == 1)
    {
        uint64_t K = second.get_dimensions()[0];
        b_shape_prom = { K, 1 };
        uint64_t orig_stride = second.get_strides()[0];
        b_strides_prom = { orig_stride, orig_stride };
    }
    else
    {
        b_shape_prom = second.get_dimensions();
        b_strides_prom = second.get_strides();
    }

    const uint64_t a_rank = static_cast<uint64_t>(a_shape_prom.size());
    const uint64_t b_rank = static_cast<uint64_t>(b_shape_prom.size());

    const uint64_t m = a_shape_prom[a_rank - 2];
    const uint64_t k_a = a_shape_prom[a_rank - 1];
    const uint64_t k_b = b_shape_prom[b_rank - 2];
    const uint64_t n = b_shape_prom[b_rank - 1];

    if (k_a != k_b)
    {
        throw std::invalid_argument(R"(matmul:
			inner dimensions must match.)");
    }
    const uint64_t K = k_a;

    const uint64_t a_batch_rank = a_rank - 2;
    const uint64_t b_batch_rank = b_rank - 2;
    const uint64_t out_batch_rank = std::max(a_batch_rank, b_batch_rank);

    std::vector<uint64_t> a_batch_shapes(out_batch_rank, 1);
    std::vector<uint64_t> b_batch_shapes(out_batch_rank, 1);
    std::vector<uint64_t> a_batch_strides(out_batch_rank, 0);
    std::vector<uint64_t> b_batch_strides(out_batch_rank, 0);

    for (uint64_t i = 0; i < a_batch_rank; ++i)
    {
        a_batch_shapes[out_batch_rank - a_batch_rank + i] =
            a_shape_prom[i];
        a_batch_strides[out_batch_rank - a_batch_rank + i] =
            a_strides_prom[i];
    }
    for (uint64_t i = 0; i < b_batch_rank; ++i)
    {
        b_batch_shapes[out_batch_rank - b_batch_rank + i] =
            b_shape_prom[i];
        b_batch_strides[out_batch_rank - b_batch_rank + i] =
            b_strides_prom[i];
    }

    std::vector<uint64_t> out_batch_shape(out_batch_rank);
    for (uint64_t d = 0; d < out_batch_rank; ++d)
    {
        uint64_t as = a_batch_shapes[d];
        uint64_t bs = b_batch_shapes[d];
        if (as == bs)
        {
            out_batch_shape[d] = as;
        }
        else if (as == 1)
        {
            out_batch_shape[d] = bs;
        }
        else if (bs == 1)
        {
            out_batch_shape[d] = as;
        }
        else
        {
            throw std::invalid_argument(R"(matmul:
				incompatible batch shapes for broadcasting.)");
        }
    }

    std::vector<uint64_t> out_shape;
    if (a_rank_orig == 1 && b_rank_orig == 1)
    {
        out_shape = { 1 };
    }
    else if (a_rank_orig == 1 && b_rank_orig >= 2)
    {
        out_shape = out_batch_shape;
        out_shape.push_back(n);
    }
    else if (a_rank_orig >= 2 && b_rank_orig == 1)
    {
        out_shape = out_batch_shape;
        out_shape.push_back(m);
    }
    else
    {
        out_shape = out_batch_shape;
        out_shape.push_back(m);
        out_shape.push_back(n);
    }

	MemoryLocation res_loc;
    if (first.get_memory_location() == MemoryLocation::DEVICE ||
        second.get_memory_location() == MemoryLocation::DEVICE)
    {
        res_loc = MemoryLocation::DEVICE;
    }
    else
    {
        res_loc = MemoryLocation::HOST;
    }
    Tensor<float_t> result(out_shape, res_loc);

    uint64_t batch_count = 1;
    for (uint64_t d : out_batch_shape)
    {
        batch_count *= d;
    }
    const uint64_t matrix_size = m * n;
	uint64_t total_output_elems = batch_count * matrix_size;

	if (a_rank_orig == 1 && b_rank_orig == 1)
	{
	    total_output_elems = 1;
	}

    std::vector<uint64_t> out_batch_divs(out_batch_rank, 1);
    if (out_batch_rank >= 2)
    {
        for (int64_t i = static_cast<int64_t>(out_batch_rank) - 2;
			i >= 0;
			--i)
        {
            out_batch_divs[static_cast<size_t>(i)] =
                out_batch_divs[static_cast<size_t>(i + 1)] *
                out_batch_shape[static_cast<size_t>(i + 1)];
        }
    }

    const uint64_t full_rank = out_batch_rank + 2;
    std::vector<uint64_t> a_full_strides(full_rank, 0);
    std::vector<uint64_t> b_full_strides(full_rank, 0);
    for (uint64_t d = 0; d < out_batch_rank; ++d)
    {
		if (a_batch_shapes[d] == 1)
		{
			a_full_strides[d] = 0;
		}
		else
		{
			a_full_strides[d] = a_batch_strides[d];
		}

		if (b_batch_shapes[d] == 1)
		{
			b_full_strides[d] = 0;
		}
		else
		{
			b_full_strides[d] = b_batch_strides[d];
		}
	}

	a_full_strides[full_rank - 2] = a_strides_prom[a_rank - 2];
	a_full_strides[full_rank - 1] = a_strides_prom[a_rank - 1];

    b_full_strides[full_rank - 2] = b_strides_prom[b_rank - 2];
    b_full_strides[full_rank - 1] = b_strides_prom[b_rank - 1];

    std::vector<uint64_t> res_batch_strides(out_batch_rank, 0);
    const std::vector<uint64_t> & res_strides_full = result.get_strides();
    const uint64_t res_rank = result.get_rank();
    uint64_t res_trailing = 0;
    if (a_rank_orig == 1 && b_rank_orig == 1)
    {
        res_trailing = 1;
    }
    else if (a_rank_orig == 1 && b_rank_orig >= 2)
    {
        res_trailing = 1;
    }
    else if (a_rank_orig >= 2 && b_rank_orig == 1)
    {
        res_trailing = 1;
    }
    else
    {
        res_trailing = 2;
    }

    uint64_t res_batch_rank = 0;
	if (res_rank > res_trailing)
	{
	    res_batch_rank = res_rank - res_trailing;
	}
    for (uint64_t d = 0;
		d < std::min<uint64_t>(res_batch_rank, out_batch_rank);
		++d)
    {
        res_batch_strides[d + (out_batch_rank - res_batch_rank)] =
            res_strides_full[d];
    }
    uint64_t res_stride_m = 0, res_stride_n = 0;
    if (res_trailing == 2)
    {
        res_stride_m = res_strides_full[res_rank - 2];
        res_stride_n = res_strides_full[res_rank - 1];
    }
    else if (res_trailing == 1)
    {
        if (a_rank_orig == 1 && b_rank_orig >= 2)
        {
            res_stride_n = res_strides_full[res_rank - 1];
        }
        else
        {
            res_stride_m = res_strides_full[res_rank - 1];
        }
    }
    else
    {
        // Do nothing.
    }

    uint64_t * p_out_batch_divs = nullptr;
    uint64_t * p_a_full_strides = nullptr;
    uint64_t * p_b_full_strides = nullptr;
    uint64_t * p_res_batch_strides = nullptr;

    if (out_batch_rank > 0)
    {
        p_out_batch_divs = static_cast<uint64_t*>(sycl::malloc_device
			(sizeof(uint64_t) * out_batch_rank, g_sycl_queue));
        g_sycl_queue.memcpy(p_out_batch_divs, out_batch_divs.data(),
                            sizeof(uint64_t) * out_batch_rank).wait();

        p_a_full_strides = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));
        p_b_full_strides = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));
        p_res_batch_strides = static_cast<uint64_t*>(sycl::malloc_device
			(sizeof(uint64_t) * out_batch_rank, g_sycl_queue));

        g_sycl_queue.memcpy(p_a_full_strides, a_full_strides.data(),
                            sizeof(uint64_t) * full_rank).wait();
        g_sycl_queue.memcpy(p_b_full_strides, b_full_strides.data(),
                            sizeof(uint64_t) * full_rank).wait();
        g_sycl_queue.memcpy(p_res_batch_strides, res_batch_strides.data(),
                            sizeof(uint64_t) * out_batch_rank).wait();
    }
    else
    {
        p_a_full_strides = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));
        p_b_full_strides = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));
        g_sycl_queue.memcpy(p_a_full_strides, a_full_strides.data(),
                            sizeof(uint64_t) * full_rank).wait();
        g_sycl_queue.memcpy(p_b_full_strides, b_full_strides.data(),
                            sizeof(uint64_t) * full_rank).wait();
    }

    // Shared error flag (0 = OK, 1 = NaN in inputs, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    float_t* p_a_data = const_cast<float_t*>(first.get_data());
    float_t* p_b_data = const_cast<float_t*>(second.get_data());
    float_t* p_r_data = result.get_data();

    // Launch kernel: each output element -> compute dot product across K.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> idx)
        {
            auto atomic_err = sycl::atomic_ref<int32_t,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*(p_error_flag));

            const uint64_t flat = static_cast<uint64_t>(idx[0]);

            uint64_t batch_idx = 0;
            uint64_t local_matrix_idx = 0;
            if (!(a_rank_orig == 1 && b_rank_orig == 1))
            {
                batch_idx = flat / (m * n);
                local_matrix_idx = flat % (m * n);
            }
            uint64_t i = 0, j = 0;
            if (a_rank_orig == 1 && b_rank_orig == 1)
            {
                i = 0; j = 0;
            }
            else if (a_rank_orig == 1 && b_rank_orig >= 2)
            {
                i = 0;
                j = local_matrix_idx;
            }
            else if (a_rank_orig >= 2 && b_rank_orig == 1)
            {
                i = local_matrix_idx;
                j = 0;
            }
            else
            {
                i = local_matrix_idx / n;
                j = local_matrix_idx % n;
            }

            uint64_t base_a = 0;
            uint64_t base_b = 0;
            uint64_t base_r = 0;

            uint64_t rem = batch_idx;
            for (uint64_t d = 0; d < out_batch_rank; ++d)
            {
                uint64_t coord = 0;
                if (out_batch_rank > 0)
                {
                    uint64_t div = p_out_batch_divs[d];
                    coord = rem / div;
                    rem = rem % div;
                }
                base_a += coord * p_a_full_strides[d];
                base_b += coord * p_b_full_strides[d];
                if (out_batch_rank > 0)
                {
                    base_r += coord * p_res_batch_strides[d];
                }
            }

            const uint64_t a_stride_m = p_a_full_strides[full_rank - 2];
            const uint64_t a_stride_k = p_a_full_strides[full_rank - 1];
            const uint64_t b_stride_k = p_b_full_strides[full_rank - 2];
            const uint64_t b_stride_n = p_b_full_strides[full_rank - 1];

            uint64_t a_offset_base = base_a + i * a_stride_m;
            uint64_t b_offset_base = base_b + j * b_stride_n;

            float_t acc = float_t{0};
            for (uint64_t t = 0; t < K; ++t)
            {
                float_t a_val = p_a_data[a_offset_base + t * a_stride_k];
                float_t b_val = p_b_data[b_offset_base + t * b_stride_k];

                if (std::isnan(a_val) || std::isnan(b_val))
                {
                    int32_t expected = 0;
                    atomic_err.compare_exchange_strong(expected, 1);
                    if (std::numeric_limits<float_t>::has_quiet_NaN)
                    {
                        acc = std::numeric_limits<float_t>::quiet_NaN();
                    }
                    else
                    {
                        acc = a_val * b_val + acc;
                    }
                    break;
                }
                acc += a_val * b_val;
            }

            if (std::isnan(acc))
            {
                uint64_t dst_off = 0;
                if (a_rank_orig == 1 && b_rank_orig == 1)
                {
                    dst_off = 0;
                }
                else if (a_rank_orig == 1 && b_rank_orig >= 2)
                {
                    dst_off = base_r + j * res_stride_n;
                }
                else if (a_rank_orig >= 2 && b_rank_orig == 1)
                {
                    dst_off = base_r + i * res_stride_m;
                }
                else
                {
                    dst_off = base_r + i * res_stride_m + j * res_stride_n;
                }
                p_r_data[dst_off] = acc;
                return;
            }

            if (!std::isfinite(acc))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 2);
                uint64_t dst_off = 0;
                if (a_rank_orig == 1 && b_rank_orig == 1)
                {
                    dst_off = 0;
                }
                else if (a_rank_orig == 1 && b_rank_orig >= 2)
                {
                    dst_off = base_r + j * res_stride_n;
                }
                else if (a_rank_orig >= 2 && b_rank_orig == 1)
                {
                    dst_off = base_r + i * res_stride_m;
                }
                else
                {
                    dst_off = base_r + i * res_stride_m + j * res_stride_n;
                }
                p_r_data[dst_off] = acc;
                return;
            }

            uint64_t dst_off = 0;
            if (a_rank_orig == 1 && b_rank_orig == 1)
            {
                dst_off = 0;
            }
            else if (a_rank_orig == 1 && b_rank_orig >= 2)
            {
                dst_off = base_r + j * res_stride_n;
            }
            else if (a_rank_orig >= 2 && b_rank_orig == 1)
            {
                dst_off = base_r + i * res_stride_m;
            }
            else
            {
                dst_off = base_r + i * res_stride_m + j * res_stride_n;
            }
            p_r_data[dst_off] = acc;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_a_full_strides, g_sycl_queue);
    sycl::free(p_b_full_strides, g_sycl_queue);
    if (p_out_batch_divs)
    {
		sycl::free(p_out_batch_divs, g_sycl_queue);
    }
    if (p_res_batch_strides)
    {
		sycl::free(p_res_batch_strides, g_sycl_queue);
    }

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(matmul: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(matmul:
				non-finite result (overflow or Inf).)");
        }
        throw std::runtime_error(R"(matmul: numeric error during matmul.)");
    }

    return result;
}
template Tensor<float> matmul<float>
	(const Tensor<float>&, const Tensor<float>&);

} // namespace temper::math
