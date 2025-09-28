/**
 * @file Math.cpp
 * @brief Mathematical tensor operation definitions.
 */

#include "temper/Math.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Utils.hpp"

namespace temper::math
{

template <typename float_t>
temper::Tensor<float_t> matmul(const temper::Tensor<float_t> & first,
                               const temper::Tensor<float_t> & second)
{
    if (first.get_dimensions().empty() || second.get_dimensions().empty())
    {
        throw std::invalid_argument(R"(matmul:
            either tensor has no elements.)");
    }

    const uint64_t a_rank_orig = first.get_rank();
    const uint64_t b_rank_orig = second.get_rank();

    temper::utils::TensorDesc a_desc;
    temper::utils::TensorDesc b_desc;

    if (a_rank_orig == 1)
    {
        const uint64_t K = first.get_dimensions()[0];
        a_desc.shape   = {1, K};
        const uint64_t orig_stride = first.get_strides()[0];
        a_desc.strides = { orig_stride * K, orig_stride };
    }
    else
    {
        a_desc.shape   = first.get_dimensions();
        a_desc.strides = first.get_strides();
    }

    if (b_rank_orig == 1)
    {
        const uint64_t K = second.get_dimensions()[0];
        b_desc.shape   = {K, 1};
        const uint64_t orig_stride = second.get_strides()[0];
        b_desc.strides = { orig_stride, orig_stride };
    }
    else
    {
        b_desc.shape   = second.get_dimensions();
        b_desc.strides = second.get_strides();
    }

    const uint64_t a_rank = static_cast<uint64_t>(a_desc.shape.size());
    const uint64_t b_rank = static_cast<uint64_t>(b_desc.shape.size());

    const uint64_t m = a_desc.shape[a_rank - 2];
    const uint64_t k_a = a_desc.shape[a_rank - 1];
    const uint64_t k_b = b_desc.shape[b_rank - 2];
    const uint64_t n = b_desc.shape[b_rank - 1];

    if (k_a != k_b)
    {
        throw std::invalid_argument(R"(matmul: inner dimensions must match.)");
    }
    const uint64_t K = k_a;

    const uint64_t a_batch_rank = a_rank - 2;
    const uint64_t b_batch_rank = b_rank - 2;
    const uint64_t out_batch_rank = std::max(a_batch_rank, b_batch_rank);
    const uint64_t full_rank = out_batch_rank + 2;

    temper::utils::TensorDesc a_al =
        temper::utils::align_tensor(a_desc, full_rank);
    temper::utils::TensorDesc b_al =
        temper::utils::align_tensor(b_desc, full_rank);

    std::vector<uint64_t> out_batch_shape;
    std::vector<uint64_t> a_batch_broadcast_strides;
    std::vector<uint64_t> b_batch_broadcast_strides;

    if (out_batch_rank > 0)
    {
        temper::utils::TensorDesc a_batch_desc;
        temper::utils::TensorDesc b_batch_desc;
        a_batch_desc.shape.assign
            (a_al.shape.begin(), a_al.shape.begin() + out_batch_rank);
        a_batch_desc.strides.assign
            (a_al.strides.begin(), a_al.strides.begin() + out_batch_rank);
        b_batch_desc.shape.assign
            (b_al.shape.begin(), b_al.shape.begin() + out_batch_rank);
        b_batch_desc.strides.assign
            (b_al.strides.begin(), b_al.strides.begin() + out_batch_rank);

        temper::utils::BroadcastResult batch_bres =
            temper::utils::compute_broadcast(a_batch_desc, b_batch_desc);

        out_batch_shape = batch_bres.out.shape;
        a_batch_broadcast_strides = batch_bres.a_strides;
        b_batch_broadcast_strides = batch_bres.b_strides;
    }
    else
    {
        out_batch_shape.clear();
        a_batch_broadcast_strides.clear();
        b_batch_broadcast_strides.clear();
    }

    std::vector<uint64_t> out_full_shape;
    out_full_shape.reserve(full_rank);
    for (uint64_t d = 0; d < out_batch_rank; ++d)
    {
        out_full_shape.push_back(out_batch_shape[d]);
    }
    out_full_shape.push_back(m);
    out_full_shape.push_back(n);

    std::vector<uint64_t> out_shape;
    if (a_rank_orig == 1 && b_rank_orig == 1)
    {
        out_shape = {1};
    }
    else if (a_rank_orig == 1 && b_rank_orig >= 2)
    {
        out_shape.assign
            (out_full_shape.begin(), out_full_shape.begin() + out_batch_rank);
        out_shape.push_back(n);
    }
    else if (a_rank_orig >= 2 && b_rank_orig == 1)
    {
        out_shape.assign
            (out_full_shape.begin(), out_full_shape.begin() + out_batch_rank);
        out_shape.push_back(m);
    }
    else
    {
        out_shape.assign
            (out_full_shape.begin(), out_full_shape.begin() + out_batch_rank);
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

    std::vector<uint64_t> out_divs =
        temper::utils::compute_divisors(out_full_shape);

    std::vector<uint64_t> a_full_strides(full_rank, 0);
    std::vector<uint64_t> b_full_strides(full_rank, 0);

    for (uint64_t d = 0; d < out_batch_rank; ++d)
    {
        a_full_strides[d] = a_batch_broadcast_strides[d];
        b_full_strides[d] = b_batch_broadcast_strides[d];
    }
    a_full_strides[full_rank - 2] = a_al.strides[full_rank - 2];
    a_full_strides[full_rank - 1] = a_al.strides[full_rank - 1];

    b_full_strides[full_rank - 2] = b_al.strides[full_rank - 2];
    b_full_strides[full_rank - 1] = b_al.strides[full_rank - 1];

    std::vector<uint64_t> a_batch_only_strides = a_full_strides;
    std::vector<uint64_t> b_batch_only_strides = b_full_strides;
    if (full_rank >= 2)
    {
        a_batch_only_strides[full_rank - 2] = 0;
        a_batch_only_strides[full_rank - 1] = 0;
        b_batch_only_strides[full_rank - 2] = 0;
        b_batch_only_strides[full_rank - 1] = 0;
    }

    uint64_t res_trailing = 0;
    bool res_trailing_is_n = false;
    if (a_rank_orig == 1 && b_rank_orig == 1)
    {
        res_trailing = 1;
        res_trailing_is_n = true;
    }
    else if (a_rank_orig == 1 && b_rank_orig >= 2)
    {
        res_trailing = 1;
        res_trailing_is_n = true;
    }
    else if (a_rank_orig >= 2 && b_rank_orig == 1)
    {
        res_trailing = 1;
        res_trailing_is_n = false;
    }
    else
    {
        res_trailing = 2;
    }

    const std::vector<uint64_t> res_strides = result.get_strides();
    const uint64_t res_rank = static_cast<uint64_t>(res_strides.size());
    uint64_t res_batch_rank = 0;
    if (res_rank > res_trailing)
    {
        res_batch_rank = res_rank - res_trailing;
    }

    std::vector<uint64_t> res_strides_full(full_rank, 0);

    if (res_batch_rank > 0)
    {
        uint64_t start = out_batch_rank - res_batch_rank;
        for (uint64_t d = 0; d < res_batch_rank; ++d)
        {
            res_strides_full[start + d] = res_strides[d];
        }
    }

    if (res_trailing == 2)
    {
        res_strides_full[full_rank - 2] = res_strides[res_batch_rank + 0];
        res_strides_full[full_rank - 1] = res_strides[res_batch_rank + 1];
    }
    else if (res_trailing == 1)
    {
        if (res_trailing_is_n)
        {
            res_strides_full[full_rank - 1] = res_strides[res_batch_rank];
        }
        else
        {
            res_strides_full[full_rank - 2] = res_strides[res_batch_rank];
        }
    }

    const uint64_t a_stride_m = a_full_strides[full_rank - 2];
    const uint64_t a_stride_k = a_full_strides[full_rank - 1];
    const uint64_t b_stride_k = b_full_strides[full_rank - 2];
    const uint64_t b_stride_n = b_full_strides[full_rank - 1];

    uint64_t batch_count = 1;
    for (uint64_t d = 0; d < out_batch_rank; ++d)
    {
        batch_count *= out_batch_shape[d];
    }

    const uint64_t matrix_size = m * n;

    uint64_t total_output_elems;
    if (a_rank_orig == 1 && b_rank_orig == 1)
    {
        total_output_elems = 1;
    }
    else
    {
        total_output_elems = batch_count * matrix_size;
    }

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));
    uint64_t* p_a_batch_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));
    uint64_t* p_b_batch_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));
    uint64_t* p_res_full_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * full_rank, g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_a_batch_strides && p_b_batch_strides &&
                     p_res_full_strides && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_a_batch_strides, g_sycl_queue);
        sycl::free(p_b_batch_strides, g_sycl_queue);
        sycl::free(p_res_full_strides, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);

        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * full_rank).wait();
    g_sycl_queue.memcpy(p_a_batch_strides,
        a_batch_only_strides.data(), sizeof(uint64_t) * full_rank).wait();
    g_sycl_queue.memcpy(p_b_batch_strides,
        b_batch_only_strides.data(), sizeof(uint64_t) * full_rank).wait();
    g_sycl_queue.memcpy(p_res_full_strides,
        res_strides_full.data(), sizeof(uint64_t) * full_rank).wait();

    *p_error_flag = 0;

    const float_t* p_a_data = first.get_data();
    const float_t* p_b_data = second.get_data();
    float_t* p_r_data = result.get_data();


    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            if (a_rank_orig == 1 && b_rank_orig == 1)
            {
                float_t acc = float_t{0};
                for (uint64_t t = 0; t < K; ++t)
                {
                    float_t av = p_a_data[t * a_stride_k];
                    float_t bv = p_b_data[t * b_stride_k];

                    temper::sycl_utils::device_check_nan_and_set<float_t>
                            (av, p_error_flag);
                    temper::sycl_utils::device_check_nan_and_set<float_t>
                        (bv, p_error_flag);
                    acc += av * bv;
                }
                temper::sycl_utils::device_check_finite_and_set<float_t>
                    (acc, p_error_flag);
                p_r_data[0] = acc;
                return;
            }

            uint64_t i = 0, j = 0;
            const uint64_t div_m = p_out_divs[full_rank - 2];
            const uint64_t div_n = p_out_divs[full_rank - 1];
            if (div_m != 0) i = (flat / div_m) % static_cast<uint64_t>(m);
            if (div_n != 0) j = (flat / div_n) % static_cast<uint64_t>(n);

            uint64_t base_a = temper::sycl_utils::idx_of
                (flat, p_out_divs, p_a_batch_strides, full_rank);
            uint64_t base_b = temper::sycl_utils::idx_of
                (flat, p_out_divs, p_b_batch_strides, full_rank);
            uint64_t base_r = temper::sycl_utils::idx_of
                (flat, p_out_divs, p_res_full_strides, full_rank);

            const uint64_t a_offset_base = base_a + i * a_stride_m;
            const uint64_t b_offset_base = base_b + j * b_stride_n;

            float_t acc = float_t{0};
            for (uint64_t t = 0; t < K; ++t)
            {
                float_t a_val = p_a_data[a_offset_base + t * a_stride_k];
                float_t b_val = p_b_data[b_offset_base + t * b_stride_k];

                temper::sycl_utils::device_check_nan_and_set<float_t>
                    (a_val, p_error_flag);
                temper::sycl_utils::device_check_nan_and_set<float_t>
                    (b_val, p_error_flag);
                acc += a_val * b_val;
            }
            temper::sycl_utils::device_check_finite_and_set<float_t>
                (acc, p_error_flag);

            p_r_data[base_r] = acc;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_a_batch_strides, g_sycl_queue);
    sycl::free(p_b_batch_strides, g_sycl_queue);
    sycl::free(p_res_full_strides, g_sycl_queue);

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

template <typename float_t>
Tensor<float_t> reshape(const Tensor<float_t> & tensor,
                        const std::vector<uint64_t>& new_dimensions)
{
    Tensor<float_t> t = tensor.clone();
    t.reshape(new_dimensions);
    return t;
}
template Tensor<float> reshape<float>
    (const Tensor<float>&, const std::vector<uint64_t>&);

template <typename float_t>
Tensor<float_t> sort(const Tensor<float_t> & tensor, int64_t axis)
{
    Tensor<float_t> t = tensor.clone();
    t.sort(axis);
    return t;
}
template Tensor<float> sort<float>
    (const Tensor<float>&, int64_t);

template <typename float_t>
Tensor<float_t> sum(const Tensor<float_t> & tensor, int64_t axis)
{
    Tensor<float_t> t = tensor.sum(axis);
    return t;
}
template Tensor<float> sum<float>
    (const Tensor<float>&, int64_t axis);

template <typename float_t>
Tensor<float_t> cumsum(const Tensor<float_t> & tensor, int64_t axis)
{
    Tensor<float_t> t = tensor.cumsum(axis);
    return t;
}
template Tensor<float> cumsum<float>
    (const Tensor<float>&, int64_t);

template<typename float_t>
Tensor<float_t> transpose(const Tensor<float_t> & tensor)
{
    Tensor t = tensor.transpose();
    return t;
}
template Tensor<float> transpose<float>(const Tensor<float>&);

template<typename float_t>
Tensor<float_t> transpose(const Tensor<float_t> & tensor,
                        const std::vector<uint64_t> & axes)
{
    Tensor t = tensor.transpose(axes);
    return t;
}
template Tensor<float> transpose<float>
    (const Tensor<float>&, const std::vector<uint64_t>&);

} // namespace temper::math
