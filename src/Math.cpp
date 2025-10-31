/**
 * @file Math.cpp
 * @brief Mathematical tensor operation definitions.
 */

#include "temper/Math.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Utils.hpp"

namespace temper::math
{

template <typename value_t>
temper::Tensor<value_t> matmul(const temper::Tensor<value_t> & first,
                               const temper::Tensor<value_t> & second)
{
    if (first.get_dimensions().empty() || second.get_dimensions().empty())
    {
        throw std::invalid_argument(R"(matmul:
            either tensor has no elements.)");
    }

    const int64_t a_rank_orig = first.get_rank();
    const int64_t b_rank_orig = second.get_rank();

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

    const int64_t a_rank = static_cast<uint64_t>(a_desc.shape.size());
    const int64_t b_rank = static_cast<uint64_t>(b_desc.shape.size());

    const uint64_t m = a_desc.shape[a_rank - 2];
    const uint64_t k_a = a_desc.shape[a_rank - 1];
    const uint64_t k_b = b_desc.shape[b_rank - 2];
    const uint64_t n = b_desc.shape[b_rank - 1];

    if (k_a != k_b)
    {
        throw std::invalid_argument(R"(matmul: inner dimensions must match.)");
    }
    const uint64_t K = k_a;

    const int64_t a_batch_rank = a_rank - 2;
    const int64_t b_batch_rank = b_rank - 2;
    const int64_t out_batch_rank = std::max(a_batch_rank, b_batch_rank);
    const int64_t full_rank = out_batch_rank + 2;

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
    for (int64_t d = 0; d < out_batch_rank; ++d)
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

    Tensor<value_t> result(out_shape, res_loc);

    std::vector<uint64_t> out_divs =
        temper::utils::compute_divisors(out_full_shape);

    std::vector<uint64_t> a_full_strides(full_rank, 0);
    std::vector<uint64_t> b_full_strides(full_rank, 0);

    for (int64_t d = 0; d < out_batch_rank; ++d)
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

    int64_t res_trailing = 0;
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
    const int64_t res_rank = static_cast<int64_t>(res_strides.size());
    int64_t res_batch_rank = 0;
    if (res_rank > res_trailing)
    {
        res_batch_rank = res_rank - res_trailing;
    }

    std::vector<uint64_t> res_strides_full(full_rank, 0);

    if (res_batch_rank > 0)
    {
        int64_t start = out_batch_rank - res_batch_rank;
        for (int64_t d = 0; d < res_batch_rank; ++d)
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
    for (int64_t d = 0; d < out_batch_rank; ++d)
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

    const value_t* p_a_data = first.get_data();
    const value_t* p_b_data = second.get_data();
    value_t* p_r_data = result.get_data();


    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            if (a_rank_orig == 1 && b_rank_orig == 1)
            {
                value_t acc = value_t{0};
                for (uint64_t t = 0; t < K; ++t)
                {
                    value_t av = p_a_data[t * a_stride_k];
                    value_t bv = p_b_data[t * b_stride_k];

                    temper::sycl_utils::device_check_nan_and_set<value_t>
                            (av, p_error_flag);
                    temper::sycl_utils::device_check_nan_and_set<value_t>
                        (bv, p_error_flag);
                    acc += av * bv;
                }
                temper::sycl_utils::device_check_finite_and_set<value_t>
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

            value_t acc = value_t{0};
            for (uint64_t t = 0; t < K; ++t)
            {
                value_t a_val = p_a_data[a_offset_base + t * a_stride_k];
                value_t b_val = p_b_data[b_offset_base + t * b_stride_k];

                temper::sycl_utils::device_check_nan_and_set<value_t>
                    (a_val, p_error_flag);
                temper::sycl_utils::device_check_nan_and_set<value_t>
                    (b_val, p_error_flag);
                acc += a_val * b_val;
            }
            temper::sycl_utils::device_check_finite_and_set<value_t>
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

template <typename value_t>
Tensor<value_t> reshape(const Tensor<value_t> & tensor,
                        const std::vector<uint64_t>& new_dimensions)
{
    Tensor<value_t> t = tensor.clone();
    t.reshape(new_dimensions);
    return t;
}
template Tensor<float> reshape<float>
    (const Tensor<float>&, const std::vector<uint64_t>&);

template <typename value_t>
Tensor<value_t> sort(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt)
{
    Tensor<value_t> t = tensor.clone();
    t.sort(axis_opt);
    return t;
}
template Tensor<float> sort<float>
    (const Tensor<float>&, std::optional<int64_t>);

template <typename value_t>
Tensor<value_t> sum(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt)
{
    Tensor<value_t> t = tensor.sum(axis_opt);
    return t;
}
template Tensor<float> sum<float>
    (const Tensor<float>&, std::optional<int64_t>);

template <typename value_t>
Tensor<value_t> cumsum(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt)
{
    Tensor<value_t> t = tensor.cumsum(axis_opt);
    return t;
}
template Tensor<float> cumsum<float>
    (const Tensor<float>&, std::optional<int64_t>);

template<typename value_t>
Tensor<value_t> transpose(const Tensor<value_t> & tensor)
{
    Tensor t = tensor.transpose();
    return t;
}
template Tensor<float> transpose<float>(const Tensor<float>&);

template<typename value_t>
Tensor<value_t> transpose(const Tensor<value_t> & tensor,
                        const std::vector<int64_t> & axes)
{
    Tensor t = tensor.transpose(axes);
    return t;
}
template Tensor<float> transpose<float>
    (const Tensor<float>&, const std::vector<int64_t>&);

template<typename value_t>
Tensor<value_t> pad(const Tensor<value_t> & tensor,
                    uint64_t pad_top,
                    uint64_t pad_bottom,
                    uint64_t pad_left,
                    uint64_t pad_right,
                    value_t pad_value)
{

    const std::vector<uint64_t> & input_shape = tensor.get_dimensions();
    const int64_t rank = tensor.get_rank();

    if (input_shape.empty())
    {
        throw std::invalid_argument(R"(pad:
            input tensor has no elements.)");
    }

    if (rank < 2)
    {
        throw std::invalid_argument(R"(pad:
            input tensor has less than 2 dimensions.)");
    }

    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    std::vector<uint64_t> res_shape = input_shape;
    if (pad_left > U64_MAX - pad_right)
    {
        throw std::overflow_error(R"(pad:
            pad_left + pad_right overflows uint64_t)");
    }
    uint64_t add_width = pad_left + pad_right;
    if (res_shape[rank - 1] > U64_MAX - add_width)
    {
        throw std::overflow_error("pad: result width overflows uint64_t");
    }
    res_shape[rank - 1] += add_width;
    if (pad_top > U64_MAX - pad_bottom)
    {
        throw std::overflow_error(R"(pad:
            pad_top + pad_bottom overflows uint64_t)");
    }
    uint64_t add_height = pad_top + pad_bottom;
    if (res_shape[rank - 2] > U64_MAX - add_height)
    {
        throw std::overflow_error(R"(pad:
            result height overflows uint64_t)");
    }
    res_shape[rank - 2] += add_height;

    MemoryLocation res_loc = tensor.get_memory_location();

    Tensor<value_t> result (res_shape, res_loc);
    (void) pad_value;

    uint64_t* p_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_res_shape = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));

    bool alloc_ok = (p_divs && p_in_strides && p_res_shape);
    if (!alloc_ok)
    {
        sycl::free(p_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_res_shape, g_sycl_queue);
        throw std::bad_alloc();
    }

    std::vector<uint64_t> res_divs =
        temper::utils::compute_divisors(res_shape);
    std::vector<uint64_t> in_batch_strides = tensor.get_strides();

    // Removed the last two strides in order to get the base of the channel.
    // Saved them for final offset calculation.
    const uint64_t in_row_stride = in_batch_strides[rank - 2];
    const uint64_t in_col_stride = in_batch_strides[rank - 1];
    in_batch_strides[rank - 1] = 0;
    in_batch_strides[rank -2] = 0;

    g_sycl_queue.memcpy(p_in_strides,
        in_batch_strides.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_res_shape,
        res_shape.data(), sizeof(uint64_t) * rank).wait();

    // Only the result divisors are needed, because they only differ from the
    // input one by the last two and input does not use them.
    g_sycl_queue.memcpy(p_divs,
        res_divs.data(), sizeof(uint64_t) * rank).wait();

    const value_t* p_in_data = tensor.get_data();
    value_t* p_res_data = result.get_data();

    const uint64_t pad_t = pad_top;
    const uint64_t pad_l = pad_left;
    const uint64_t in_h = input_shape[rank - 2];
    const uint64_t in_w = input_shape[rank - 1];
    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);
            value_t val = pad_value;

            uint64_t res_col = (flat / p_divs[rank - 1]) % p_res_shape[rank - 1];
            uint64_t res_row = (flat / p_divs[rank - 2]) % p_res_shape[rank - 2];

            // Check whether this output location maps into
            // the input image region.
            bool inside_col = (res_col >= pad_l) && (res_col < pad_l + in_w);
            bool inside_row = (res_row >= pad_t) && (res_row < pad_t + in_h);

            if (inside_row && inside_col)
            {
                uint64_t in_row = res_row - pad_t;
                uint64_t in_col = res_col - pad_l;

                uint64_t in_base = temper::sycl_utils::idx_of
                    (flat, p_divs, p_in_strides, rank);
                uint64_t in_idx = in_base +
                                in_row * in_row_stride +
                                in_col * in_col_stride;
                val = p_in_data[in_idx];
            }

            p_res_data[flat] = val;
        });
    }).wait();

    return result;
}
template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, uint64_t, uint64_t, float);

template<typename value_t>
Tensor<value_t> pad(const Tensor<value_t> & tensor,
                    uint64_t pad_height,
                    uint64_t pad_width,
                    value_t pad_value)
{
    return pad(tensor, pad_height, pad_height, pad_width, pad_width, pad_value);
}
template Tensor<float> pad<float>
    (const Tensor<float>&, uint64_t, uint64_t, float);

template<typename value_t>
std::vector<uint64_t> argmax(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt)
{
    const std::vector<uint64_t> & in_shape = tensor.get_dimensions();
    const int64_t rank = tensor.get_rank();

    if (in_shape.empty())
    {
        throw std::invalid_argument(R"(argmax:
            input tensor has no elements.)");
    }

    const bool flatten = !axis_opt.has_value();

    int64_t axis;

    if (!flatten)
    {
        axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || axis >= rank)
        {
            throw std::invalid_argument("argmax: axis out of bounds");
        }
    }

    uint64_t total_output_elems = 1;
    if (flatten)
    {
        total_output_elems = 1;
    }
    else
    {
        total_output_elems = 1;
        for (int64_t d = 0; d < rank; ++d)
        {
            if (d == axis)
            {
                continue;
            }
            total_output_elems *= in_shape[d];
        }
    }

    const uint64_t total_input_elems = tensor.get_num_elements();

    const std::vector<uint64_t> in_divs =
        temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = tensor.get_strides();

    std::vector<uint64_t> res_full_shape;
    if (!flatten)
    {
        res_full_shape = in_shape;
        res_full_shape[axis] = 1;
    }

    uint64_t* p_in_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_res_full_divs = nullptr;
    if (!flatten)
    {
        p_res_full_divs = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    }

    uint64_t* p_out_dev = static_cast<uint64_t*>(sycl::malloc_device
        (sizeof(uint64_t) * static_cast<size_t>(total_output_elems),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_in_divs && p_in_strides && p_error_flag &&
                     (flatten || p_res_full_divs) && p_out_dev);
    if (!alloc_ok)
    {
        sycl::free(p_in_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_res_full_divs, g_sycl_queue);
        sycl::free(p_out_dev, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_in_divs,
        in_divs.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_in_strides,
        in_strides.data(), sizeof(uint64_t) * rank).wait();
    if (!flatten)
    {
        const std::vector<uint64_t> res_full_divs =
            temper::utils::compute_divisors(res_full_shape);
        g_sycl_queue.memcpy(p_res_full_divs,
            res_full_divs.data(), sizeof(uint64_t) * rank).wait();
    }

    *p_error_flag = 0;

    const value_t* p_in_data = tensor.get_data();
    uint64_t axis_dim = 0;
    uint64_t axis_stride = 0;
    if (!flatten)
    {
        axis_dim = in_shape[axis];
        axis_stride = in_strides[axis];
    }

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t out_flat = static_cast<uint64_t>(id[0]);

            if (flatten)
            {
                uint64_t best_idx = 0;
                uint64_t first_offset = temper::sycl_utils::idx_of(
                    0, p_in_divs, p_in_strides, rank);
                value_t best_val = p_in_data[first_offset];
                temper::sycl_utils::device_check_nan_and_set<value_t>
                    (best_val, p_error_flag);

                for (uint64_t t = 1; t < total_input_elems; ++t)
                {
                    uint64_t off = temper::sycl_utils::idx_of(
                        t, p_in_divs, p_in_strides, rank);
                    value_t v = p_in_data[off];
                    temper::sycl_utils::device_check_nan_and_set<value_t>
                        (v, p_error_flag);
                    if (v > best_val)
                    {
                        best_val = v;
                        best_idx = t;
                    }
                }
                p_out_dev[0] = best_idx;
                return;
            }

            uint64_t base = temper::sycl_utils::idx_of(
                out_flat, p_res_full_divs, p_in_strides, rank);

            uint64_t best_rel = 0;
            value_t best_val = p_in_data[base + 0 * axis_stride];
            temper::sycl_utils::device_check_nan_and_set<value_t>
                (best_val, p_error_flag);

            for (uint64_t t = 1; t < axis_dim; ++t)
            {
                value_t v = p_in_data[base + t * axis_stride];
                temper::sycl_utils::device_check_nan_and_set<value_t>
                    (v, p_error_flag);
                if (v > best_val)
                {
                    best_val = v;
                    best_rel = t;
                }
            }

            p_out_dev[out_flat] = best_rel;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_in_divs, g_sycl_queue);
    sycl::free(p_in_strides, g_sycl_queue);
    sycl::free(p_res_full_divs, g_sycl_queue);

    if (err != 0)
    {
        sycl::free(p_out_dev, g_sycl_queue);
        if (err == 1)
        {
            throw std::runtime_error(R"(argmax: NaN detected in inputs.)");
        }
        throw std::runtime_error(R"(argmax: numeric error during argmax.)");
    }

    std::vector<uint64_t> host_out(static_cast<size_t>(total_output_elems));
    g_sycl_queue.memcpy(host_out.data(), p_out_dev,
        sizeof(uint64_t) * static_cast<size_t>(total_output_elems)).wait();

    sycl::free(p_out_dev, g_sycl_queue);

    return host_out;
}
template std::vector<uint64_t> argmax<float>
    (const Tensor<float>&, std::optional<int64_t>);

template<typename value_t>
Tensor<value_t> linspace(const Tensor<value_t>& start,
                        const Tensor<value_t>& stop,
                        uint64_t num,
                        MemoryLocation res_loc,
                        int64_t axis,
                        bool endpoint,
                        Tensor<value_t>* step_out)
{
    const std::vector<uint64_t> & start_shape = start.get_dimensions();
    const std::vector<uint64_t> & stop_shape = stop.get_dimensions();

    if (start_shape.empty())
    {
        throw std::invalid_argument(R"(linspace:
            start tensor has no elements.)");
    }

    if (stop_shape.empty())
    {
        throw std::invalid_argument(R"(linspace:
            stop tensor has no elements.)");
    }

    if (num == 0)
    {
        Tensor<value_t> result({1}, res_loc);
        Tensor<value_t> step_tensor({1}, res_loc);
        if (step_out != nullptr)
        {
            *step_out = std::move(step_tensor);
        }
        return result;
    }

    temper::utils::TensorDesc a_desc, b_desc;
    a_desc.shape = start_shape;
    b_desc.shape = stop_shape;
    a_desc.strides = start.get_strides();
    b_desc.strides = stop.get_strides();

    const int64_t a_rank = start.get_rank();
    const int64_t b_rank = stop.get_rank();
    int64_t full_rank = std::max(a_rank, b_rank);

    a_desc = temper::utils::align_tensor(a_desc, full_rank);
    b_desc = temper::utils::align_tensor(b_desc, full_rank);

    temper::utils::BroadcastResult b_res =
        temper::utils::compute_broadcast(a_desc, b_desc);

    const std::vector<uint64_t> res_shape = b_res.out.shape;
    const std::vector<uint64_t> a_bcast_strides = b_res.a_strides;
    const std::vector<uint64_t> b_bcast_strides = b_res.b_strides;
    const int64_t res_rank = static_cast<int64_t>(res_shape.size());

    const bool start_is_shape1 =
        (start_shape.size() == 1 && start_shape[0] == 1);
    const bool stop_is_shape1  =
        (stop_shape.size() == 1 && stop_shape[0] == 1);
    const bool both_scalars = (start_is_shape1 && stop_is_shape1);

    int64_t out_rank;
    std::vector<uint64_t> out_shape;

    if (both_scalars)
    {
        out_rank = 1;
        out_shape = { num };
        if (axis < 0)
        {
            axis += out_rank;
        }
        if (axis < 0 || axis >= out_rank)
        {
            throw std::invalid_argument(R"(linspace: axis out of range)");
        }
    }
    else
    {
        out_rank = res_rank + 1;
        if (axis < 0)
        {
            axis += out_rank;
        }
        if (axis < 0 || axis >= out_rank)
        {
            throw std::invalid_argument(R"(linspace: axis out of range)");
        }
        out_shape.reserve(out_rank);
        for (int64_t d = 0; d < axis; ++d)
        {
            out_shape.push_back(res_shape[d]);
        }
        out_shape.push_back(num);
        for (int64_t d = axis; d < res_rank; ++d)
        {
            out_shape.push_back(res_shape[d]);
        }
    }

    Tensor<value_t> result(out_shape, res_loc);

    std::vector<uint64_t> step_shape = res_shape;
    Tensor<value_t> step_tensor(step_shape, res_loc);

    std::vector<uint64_t> out_divs = temper::utils::compute_divisors(out_shape);
    std::vector<uint64_t> res_divs = temper::utils::compute_divisors(res_shape);

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));

    uint64_t* p_a_bcast_strides_expanded = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));
    uint64_t* p_b_bcast_strides_expanded = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));
    uint64_t* p_res_strides_expanded = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs &&
                     p_a_bcast_strides_expanded && p_b_bcast_strides_expanded &&
                     p_res_strides_expanded && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_a_bcast_strides_expanded, g_sycl_queue);
        sycl::free(p_b_bcast_strides_expanded, g_sycl_queue);
        sycl::free(p_res_strides_expanded, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * out_rank).wait();

    std::vector<uint64_t> a_bcast_strides_expanded(out_rank, 0);
    std::vector<uint64_t> b_bcast_strides_expanded(out_rank, 0);
    std::vector<uint64_t> res_strides_expanded(out_rank, 0);

    for (int64_t d = 0; d < axis; ++d)
    {
        if (d < res_rank)
        {
            a_bcast_strides_expanded[d] = a_bcast_strides[d];
            b_bcast_strides_expanded[d] = b_bcast_strides[d];
        }
    }
    a_bcast_strides_expanded[axis] = 0;
    b_bcast_strides_expanded[axis] = 0;
    for (int64_t d = axis; d < res_rank; ++d)
    {
        a_bcast_strides_expanded[d + 1] = a_bcast_strides[d];
        b_bcast_strides_expanded[d + 1] = b_bcast_strides[d];
    }

    for (int64_t d = 0; d < axis; ++d)
    {
        if (d < res_rank)
        {
            res_strides_expanded[d] = res_divs[d];
        }
    }
    res_strides_expanded[axis] = 0;
    for (int64_t d = axis; d < res_rank; ++d)
    {
        res_strides_expanded[d + 1] = res_divs[d];
    }

    g_sycl_queue.memcpy(p_a_bcast_strides_expanded,
        a_bcast_strides_expanded.data(), sizeof(uint64_t) * out_rank).wait();
    g_sycl_queue.memcpy(p_b_bcast_strides_expanded,
        b_bcast_strides_expanded.data(), sizeof(uint64_t) * out_rank).wait();
    g_sycl_queue.memcpy(p_res_strides_expanded,
        res_strides_expanded.data(), sizeof(uint64_t) * out_rank).wait();

    *p_error_flag = 0;

    const value_t* p_a_data = start.get_data();
    const value_t* p_b_data = stop.get_data();
    value_t* p_out = result.get_data();
    value_t* p_step_out = step_tensor.get_data();

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t pos_axis = 0;
            pos_axis = (flat / p_out_divs[axis]) % static_cast<uint64_t>(num);

            const uint64_t a_idx = temper::sycl_utils::idx_of(flat,
                p_out_divs,
                p_a_bcast_strides_expanded,
                out_rank);
            const uint64_t b_idx = temper::sycl_utils::idx_of(flat,
                p_out_divs,
                p_b_bcast_strides_expanded,
                out_rank);
            const uint64_t res_flat = temper::sycl_utils::idx_of(flat,
                p_out_divs,
                p_res_strides_expanded,
                out_rank);

            value_t a_val = p_a_data[a_idx];
            value_t b_val = p_b_data[b_idx];
            temper::sycl_utils::device_check_nan_and_set<value_t>
                (a_val, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<value_t>
                (b_val, p_error_flag);

            value_t step;
            if (num == 1)
            {
                step = value_t{0};
            }
            else
            {
                if (endpoint)
                {
                    step = (b_val - a_val) / static_cast<value_t>(num - 1);
                }
                else
                {
                    step = (b_val - a_val) / static_cast<value_t>(num);
                }
            }
            temper::sycl_utils::device_check_finite_and_set<value_t>
                (step, p_error_flag);

            if (pos_axis == 0)
            {
                p_step_out[res_flat] = step;
            }

            value_t val;
            if (num == 1)
            {
                val = a_val;
            }
            else
            {
                val = a_val + step * static_cast<value_t>(pos_axis);
            }
            temper::sycl_utils::device_check_finite_and_set<value_t>
                (val, p_error_flag);
            p_out[flat] = val;
        });
    }).wait();


    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_a_bcast_strides_expanded, g_sycl_queue);
    sycl::free(p_b_bcast_strides_expanded, g_sycl_queue);
    sycl::free(p_res_strides_expanded, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(linspace:
                NaN detected in inputs or computed values.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(linspace:
                non-finite result (overflow or Inf) produced.)");
        }
        throw std::runtime_error(R"(linspace:
            numeric error during linspace generation.)");
    }

    if (step_out != nullptr)
    {
        *step_out = std::move(step_tensor);
    }

    return result;
}
template Tensor<float> linspace<float>(const Tensor<float>&,
const Tensor<float>&, uint64_t, MemoryLocation, int64_t, bool, Tensor<float>*);

template<typename value_t>
Tensor<value_t> linspace(value_t start,
                        value_t stop,
                        uint64_t num,
                        MemoryLocation res_loc,
                        int64_t axis,
                        bool endpoint,
                        Tensor<value_t>* step_out)
{
    Tensor<value_t> start_t;
    start_t = start;
    Tensor<value_t> stop_t;
    stop_t = stop;
    return linspace(start_t, stop_t, num, res_loc, axis, endpoint, step_out);
}
template Tensor<float> linspace<float>(float,
float, uint64_t, MemoryLocation, int64_t, bool, Tensor<float>*);

template<typename value_t>
Tensor<value_t> arange(value_t start,
                       value_t stop,
                       value_t step,
                       MemoryLocation res_loc)
{
    if (step == static_cast<value_t>(0))
    {
        throw std::invalid_argument(R"(arange: step must be non-zero.)");
    }

    if (!std::isfinite(static_cast<double>(start)) ||
        !std::isfinite(static_cast<double>(stop))  ||
        !std::isfinite(static_cast<double>(step)))
    {
        throw std::runtime_error(R"(arange:
            non-finite start/stop/step provided.)");
    }

    double raw   = static_cast<double>(stop) - static_cast<double>(start);
    double ratio = raw / static_cast<double>(step);

    uint64_t num = 0;
    if (step > static_cast<value_t>(0))
    {
        if (ratio > 0.0)
        {
            num = static_cast<uint64_t>(std::ceil(ratio));
        }
    }
    else
    {
        if (ratio > 0.0)
        {
            num = static_cast<uint64_t>(std::ceil(ratio));
        }
    }

    if (num == 0)
    {
        Tensor<value_t> result({1}, res_loc);
        return result;
    }

    std::vector<uint64_t> out_shape = { num };
    Tensor<value_t> result(out_shape, res_loc);

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    if (!p_error_flag)
    {
        throw std::bad_alloc();
    }
    *p_error_flag = 0;

    value_t* p_out = result.get_data();
    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t idx = static_cast<uint64_t>(id[0]);
            value_t val = static_cast<value_t>
                (start + static_cast<double>(idx) * static_cast<double>(step));

            temper::sycl_utils::device_check_nan_and_set<value_t>
                (val, p_error_flag);
            temper::sycl_utils::device_check_finite_and_set<value_t>
                (val, p_error_flag);

            p_out[idx] = val;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(arange:
                NaN detected in computed values.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(arange:
                non-finite result (overflow or Inf) produced.)");
        }
        throw std::runtime_error(R"(arange:
            numeric error during generation.)");
    }

    return result;
}
template Tensor<float> arange<float>(float, float, float, MemoryLocation);

template<typename value_t>
Tensor<value_t> arange(value_t stop, MemoryLocation res_loc)
{
    return arange<value_t>
        (static_cast<value_t>(0), stop, static_cast<value_t>(1), res_loc);
}
template Tensor<float> arange<float>(float, MemoryLocation);

template<typename value_t>
Tensor<value_t> zeros(const std::vector<uint64_t> & shape,
    MemoryLocation res_loc)
{
    // Default builder already zero-initializes.
    return Tensor<float>(shape, res_loc);
}
template Tensor<float> zeros<float>
    (const std::vector<uint64_t>&, MemoryLocation);

template <typename value_t>
value_t integral(std::function<value_t(value_t)> f,
                        value_t a,
                        value_t b,
                        uint64_t n_bins)
{
    if (n_bins < 1)
    {
        throw std::invalid_argument(R"(integral:
            there need to be at least 1 interval)");
    }

    const value_t delta = (b - a) / static_cast<value_t>(n_bins);
    value_t x = a;
    value_t result = static_cast<value_t>(0);

    for (uint64_t i = 0; i < n_bins; ++i)
    {
        const value_t x_left  = x;
        const value_t x_right = x + delta;
        const value_t x_mid   = static_cast<value_t>(0.5) * (x_left + x_right);

        result += f(x_left) + static_cast<value_t>(4) * f(x_mid) + f(x_right);
        x = x_right;
    }

    result *= (delta / static_cast<value_t>(6));
    return result;
}
template float integral<float>
    (std::function<float(float)>, float, float, uint64_t);

template<typename value_t>
Tensor<value_t> factorial(const Tensor<value_t> & tensor)
{
    const std::vector<uint64_t> & in_shape = tensor.get_dimensions();
    const uint64_t arr_len = static_cast<uint64_t>(in_shape.size());

    if (in_shape.empty())
    {
        throw std::invalid_argument(R"(factorial:
            input tensor has no elements.)");
    }

    const uint64_t total_output_elems = tensor.get_num_elements();
    MemoryLocation res_loc = tensor.get_memory_location();
    Tensor<value_t> result(in_shape, res_loc);

    const std::vector<uint64_t> in_divs =
        temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = tensor.get_strides();

    uint64_t* p_in_divs = static_cast<uint64_t*>(sycl::malloc_device
        (sizeof(uint64_t) * static_cast<size_t>(arr_len), g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(sycl::malloc_device
        (sizeof(uint64_t) * static_cast<size_t>(arr_len), g_sycl_queue));
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_in_divs && p_in_strides && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_in_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_in_divs,
        in_divs.data(), sizeof(uint64_t) * arr_len).wait();
    g_sycl_queue.memcpy(p_in_strides,
        in_strides.data(), sizeof(uint64_t) * arr_len).wait();
    *p_error_flag = 0;

    const value_t* p_in_data = tensor.get_data();
    value_t* p_out = result.get_data();

    const value_t eps = static_cast<value_t>(1e-6);

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t in_idx = temper::sycl_utils::idx_of
                (flat, p_in_divs, p_in_strides, arr_len);
            value_t v = p_in_data[in_idx];

            temper::sycl_utils::device_check_nan_and_set<value_t>
                (v, p_error_flag);
            temper::sycl_utils::device_check_finite_and_set<value_t>
                (v, p_error_flag);
            if (*p_error_flag != 0)
            {
                p_out[flat] = static_cast<value_t>(0);
                return;
            }

            if (v < static_cast<value_t>(0))
            {
                p_error_flag[0] = 3;
                p_out[flat] = static_cast<value_t>(0);
                return;
            }

            value_t rounded = sycl::floor(v + static_cast<value_t>(0.5));
            value_t diff = sycl::fabs(v - rounded);
            if (diff > eps)
            {
                p_error_flag[0] = 3;
                p_out[flat] = static_cast<value_t>(0);
                return;
            }

            const uint64_t n = static_cast<uint64_t>(rounded);

            value_t acc = static_cast<value_t>(1);
            for (uint64_t t = 1; t <= n; ++t)
            {
                acc = acc * static_cast<value_t>(t);
            }
            temper::sycl_utils::device_check_finite_and_set<value_t>
                    (acc, p_error_flag);
            p_out[flat] = acc;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_in_divs, g_sycl_queue);
    sycl::free(p_in_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(factorial: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(factorial:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 3)
        {
            throw std::invalid_argument(R"(factorial:
                input contains negative or non-integer values.)");
        }
        throw std::runtime_error(R"(factorial:
            numeric error during factorial.)");
    }

    return result;
}
template Tensor<float> factorial<float>(const Tensor<float>&);

template<typename value_t>
Tensor<value_t> log(const Tensor<value_t> & tensor)
{
    const std::vector<uint64_t> & in_shape = tensor.get_dimensions();
    if (in_shape.empty())
    {
        throw std::invalid_argument(R"(log: input tensor has no elements.)");
    }

    const uint64_t arr_len = static_cast<uint64_t>(in_shape.size());
    const uint64_t total_output_elems = tensor.get_num_elements();
    MemoryLocation res_loc = tensor.get_memory_location();
    Tensor<value_t> result(in_shape, res_loc);

    const std::vector<uint64_t> in_divs =
        temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = tensor.get_strides();

    uint64_t* p_in_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(arr_len),
                            g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(arr_len),
                            g_sycl_queue));
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_in_divs && p_in_strides && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_in_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_in_divs, in_divs.data(),
        sizeof(uint64_t) * arr_len).wait();
    g_sycl_queue.memcpy(p_in_strides, in_strides.data(),
        sizeof(uint64_t) * arr_len).wait();
    *p_error_flag = 0;

    const value_t* p_in_data = tensor.get_data();
    value_t* p_out = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t in_idx = temper::sycl_utils::idx_of(flat,
                                                    p_in_divs,
                                                    p_in_strides,
                                                    arr_len);
            value_t v = p_in_data[in_idx];

            temper::sycl_utils::device_check_nan_and_set<value_t>
                (v, p_error_flag);
            value_t outv = sycl::log(v);

            temper::sycl_utils::device_check_finite_and_set<value_t>
                (outv, p_error_flag);
            p_out[flat] = outv;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_in_divs, g_sycl_queue);
    sycl::free(p_in_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(log: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(log:
                non-finite result (Inf/overflow/NaN) produced.)");
        }
        throw std::runtime_error(R"(log:
            numeric error during log computation.)");
    }

    return result;
}
template Tensor<float> log<float>(const Tensor<float>&);

template <typename value_t>
Tensor<value_t> mean(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt)
{
    const int64_t rank = tensor.get_rank();
    if (rank == 0)
    {
        throw std::invalid_argument(R"(mean:
            input tensor has no elements.)");
    }
    const bool flatten = !axis_opt.has_value();

    uint64_t denom_u = 1;
    if (flatten)
    {
        denom_u = tensor.get_num_elements();
    }
    else
    {
        int64_t axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || axis >= rank)
        {
            throw std::invalid_argument("mean: axis out of bounds");
        }
        denom_u = tensor.get_dimensions()[axis];
    }
    Tensor<value_t> s = math::sum(tensor, axis_opt);

    value_t denom_val = static_cast<value_t>(denom_u);
    MemoryLocation loc = tensor.get_memory_location();
    Tensor<value_t> denom_t({1}, loc);
    denom_t = denom_val;

    Tensor<value_t> result = s / denom_t;
    return result;
}
template Tensor<float> mean<float>
    (const Tensor<float>&, std::optional<int64_t> axis_opt);

template <typename value_t>
Tensor<value_t> var(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt,
    int64_t ddof)
{
    const uint64_t total_elems = tensor.get_num_elements();
    if (total_elems == 0)
    {
        throw std::invalid_argument(R"(var:
            input tensor has no elements.)");
    }

    if (ddof < 0)
    {
        throw std::invalid_argument(R"(var:
            ddof must be non-negative.)");
    }

    const bool flatten = !axis_opt.has_value();

    uint64_t N = 0;
    if (flatten)
    {
        if (static_cast<uint64_t>(ddof) >= total_elems)
        {
            throw std::invalid_argument
                (R"(var: ddof >= number of elements.)");
        }
        N = total_elems;
    }
    else
    {
        const int64_t rank = tensor.get_rank();
        int64_t axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || axis >= rank)
        {
            throw std::invalid_argument("var: axis out of bounds");
        }
        const uint64_t axis_len = tensor.get_dimensions()[axis];
        if (static_cast<uint64_t>(ddof) >= axis_len)
        {
            throw std::invalid_argument(R"(var:
                ddof >= axis length.)");
        }
        N = axis_len;
    }

    uint64_t denom_u = N - static_cast<uint64_t>(ddof);
    Tensor<value_t> m = math::mean(tensor, axis_opt);

    Tensor<value_t> diff = tensor - m;
    Tensor<value_t> sq = diff * diff;
    Tensor<value_t> sumsq = math::sum(sq, axis_opt);

    value_t denom_val = static_cast<value_t>(denom_u);
    MemoryLocation loc = tensor.get_memory_location();
    Tensor<value_t> denom_t({1}, loc);
    denom_t = denom_val;

    Tensor<value_t> result = sumsq / denom_t;
    return result;
}
template Tensor<float> var<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);

template <typename value_t>
Tensor<value_t> cov(const Tensor<value_t> & tensor,
                    std::vector<int64_t> sample_axes,
                    std::vector<int64_t> event_axes,
                    int64_t ddof)
{
    const uint64_t total_elems = tensor.get_num_elements();
    const std::vector<uint64_t> & original_shape = tensor.get_dimensions();
    const int64_t rank = tensor.get_rank();
    if (total_elems == 0)
    {
        throw std::invalid_argument(R"(cov:
            input tensor has no elements.)");
    }

    if(sample_axes.empty() || event_axes.empty())
    {
        throw std::invalid_argument(R"(cov:
            axes arguments cannot be empty.)");
    }

    if (ddof < 0)
    {
        throw std::invalid_argument(R"(cov:
            ddof must be non-negative.)");
    }

    if (rank < 2)
    {
        throw std::invalid_argument(R"(cov:
            rank must be >= 2.)");
    }

    const uint64_t num_sample_axes = static_cast<int64_t>(sample_axes.size());
    const uint64_t num_event_axes  = static_cast<int64_t>(event_axes.size());
    // Check if the sample axes are regular
    // (within tensor range, not replicated).
    std::vector<bool> seen(rank, false);
    for (uint64_t i = 0; i < sample_axes.size(); ++i)
    {
        int64_t axis = sample_axes[i];
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || axis >= rank)
        {
            throw std::invalid_argument("cov: axis out of bounds");
        }
        if (seen[axis])
        {
            throw std::invalid_argument(R"(cov:
                the same axis cannot be used twice)");
        }
        seen[axis] = true;
        sample_axes[i] = axis;
    }

    // We do the same for event axes.
    for (uint64_t i = 0; i < event_axes.size(); ++i)
    {
        int64_t axis = event_axes[i];
        if (axis < 0)
        {
            axis += rank;
        }
        if (axis < 0 || axis >= rank)
        {
            throw std::invalid_argument("cov: axis out of bounds");
        }
        if (seen[axis])
        {
            throw std::invalid_argument(R"(cov:
                the same axis cannot be used twice)");
        }
        seen[axis] = true;
        event_axes[i] = axis;
    }

    // Build the tensor shape: batch axes -> sample axes -> event axes.
    std::vector<int64_t> t_shape;
    t_shape.reserve(rank);
    for (int64_t i = 0; i < rank; ++i)
    {
        if (seen[i])
        {
            continue;
        }
        t_shape.push_back(i);
    }
    t_shape.insert(t_shape.end(), sample_axes.begin(), sample_axes.end());
    t_shape.insert(t_shape.end(), event_axes.begin(), event_axes.end());

    Tensor<value_t> t_tensor = tensor.transpose(t_shape);

    // We clone the transposed tensor because reshape does not work on views.
    Tensor<value_t> t_tensor_clone = t_tensor.clone();

    uint64_t sample_total = 1, event_total = 1;
    // Compute the final shape of the tensor we need to operate on.
    for (int64_t axis : sample_axes)
    {
        sample_total *= original_shape[axis];
    }
    if (static_cast<uint64_t>(ddof) >= sample_total)
    {
        throw std::invalid_argument(R"(cov:
                not enough samples for ddof.)");
    }
    for (int64_t axis : event_axes)
    {
        event_total *= original_shape[axis];
    }

    const std::vector<uint64_t> & transposed_shape =
        t_tensor_clone.get_dimensions();

    const int64_t batch_len = rank - (num_sample_axes + num_event_axes);

    std::vector<uint64_t> final_shape;
    final_shape.reserve(batch_len + 2);
    for (int64_t i = 0; i < batch_len; ++i)
    {
        final_shape.push_back(transposed_shape[i]);
    }
    final_shape.push_back(sample_total);
    final_shape.push_back(event_total);

    t_tensor_clone.reshape(final_shape);

    Tensor<value_t> mu = math::mean(t_tensor_clone, batch_len);
    Tensor<value_t> centered = t_tensor_clone - mu;

    Tensor<value_t> denom({1}, tensor.get_memory_location());
    denom = 1.0f / (sample_total - ddof);

    std::vector<int64_t> transpose_order;
    transpose_order.reserve(centered.get_rank());
    for (int64_t i = 0; i < batch_len; ++i)
    {
        transpose_order.push_back(i);
    }
    transpose_order.push_back(batch_len + 1);
    transpose_order.push_back(batch_len);

    Tensor<value_t> centered_t = centered.transpose(transpose_order);

    Tensor<value_t> result = denom * math::matmul(centered_t, centered);

    return result;
}
template Tensor<float> cov<float> (const Tensor<float>&,
    std::vector<int64_t>, std::vector<int64_t>, int64_t);

template <typename value_t>
Tensor<value_t> cov(const Tensor<value_t> & tensor, int64_t ddof)
{
    const int64_t rank = tensor.get_rank();
    if (rank < 2)
    {
        throw std::invalid_argument(R"(cov:
            rank must be >= 2.)");
    }

    return math::cov(tensor, {rank - 2}, {rank - 1}, ddof);
}
template Tensor<float> cov<float> (const Tensor<float>&, int64_t);

template <typename value_t>
Tensor<value_t> stddev(const Tensor<value_t> & tensor,
    std::optional<int64_t> axis_opt,
    int64_t ddof)
{
    const int64_t rank = tensor.get_rank();
    if (rank == 0)
    {
        throw std::invalid_argument(R"(std:
            input tensor has no elements.)");
    }
    Tensor<value_t> v = math::var(tensor, axis_opt, ddof);

    const std::vector<uint64_t> & in_shape = v.get_dimensions();

    const uint64_t total_output_elems = v.get_num_elements();
    MemoryLocation res_loc = v.get_memory_location();
    Tensor<value_t> result(in_shape, res_loc);

    const std::vector<uint64_t> in_divs =
        temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = v.get_strides();

    uint64_t* p_in_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_in_divs && p_in_strides && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_in_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_in_divs, in_divs.data(),
        sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_in_strides, in_strides.data(),
        sizeof(uint64_t) * rank).wait();
    *p_error_flag = 0;

    const value_t* p_in_data = v.get_data();
    value_t* p_out = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);
            uint64_t in_idx = temper::sycl_utils::idx_of(flat,
                                                    p_in_divs,
                                                    p_in_strides,
                                                    rank);
            value_t val = p_in_data[in_idx];
            temper::sycl_utils::device_check_nan_and_set<value_t>
                (val, p_error_flag);
            value_t outv = sycl_utils::sqrt(val);
            temper::sycl_utils::device_check_finite_and_set<value_t>
                (outv, p_error_flag);
            p_out[flat] = outv;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_in_divs, g_sycl_queue);
    sycl::free(p_in_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(std: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(std:
                non-finite result (Inf/overflow/NaN) produced.)");
        }
        throw std::runtime_error(R"(std:
            numeric error during sqrt computation.)");
    }

    return result;
}
template Tensor<float> stddev<float>
    (const Tensor<float>&, std::optional<int64_t>, int64_t);

template <typename value_t>
std::pair<Tensor<value_t>, Tensor<value_t>> eig(const Tensor<value_t> & tensor,
    uint64_t max_iters,
    value_t tol)
{
    const int64_t rank = tensor.get_rank();
    if (rank < 2)
    {
        throw std::invalid_argument("eig: rank must be >= 2.");
    }

    const std::vector<uint64_t> tensor_dims = tensor.get_dimensions();
    const std::vector<uint64_t> tensor_strides = tensor.get_strides();

    if (tensor_dims[rank - 1] != tensor_dims[rank - 2])
    {
        throw std::invalid_argument(R"(eig:
            last two dims must be square.)");
    }
    const uint64_t n = tensor_dims[rank - 1];

    std::vector<uint64_t> batch_shape;
    for (int64_t i = 0; i + 2 < rank; ++i)
    {
        batch_shape.push_back(tensor_dims[i]);
    }

    uint64_t batch_count = 1;
    for (uint64_t d : batch_shape)
    {
        batch_count *= d;
    }

    std::vector<uint64_t> eigvals_shape = batch_shape;
    eigvals_shape.push_back(n);
    std::vector<uint64_t> eigvecs_shape = batch_shape;
    eigvecs_shape.push_back(n);
    eigvecs_shape.push_back(n);

    MemoryLocation res_loc = tensor.get_memory_location();
    Tensor<value_t> eigvals_tensor(eigvals_shape, res_loc);
    Tensor<value_t> eigvecs_tensor(eigvecs_shape, res_loc);

    const uint64_t matrix_size = n * n;
    const uint64_t total_matrix_elems = batch_count * matrix_size;

    std::vector<uint64_t> batch_divisors;
    if (!batch_shape.empty())
    {
        batch_divisors = utils::compute_divisors(batch_shape);
    }

    std::vector<uint64_t> batch_strides;
    for (int64_t i = 0; i < static_cast<int64_t>(batch_shape.size()); ++i)
    {
        batch_strides.push_back(tensor_strides[i]);
    }

    value_t* p_A = nullptr;
    value_t* p_Q = nullptr;
    value_t* p_temp = nullptr;
    int32_t* p_error_flag = nullptr;
    uint64_t* p_batch_divisors = nullptr;
    uint64_t* p_batch_strides = nullptr;

    p_A = static_cast<value_t*>(sycl::malloc_device(
        sizeof(value_t) * total_matrix_elems, g_sycl_queue));
    p_Q = static_cast<value_t*>(sycl::malloc_device(
        sizeof(value_t) * total_matrix_elems, g_sycl_queue));
    p_temp = static_cast<value_t*>(sycl::malloc_device(
        sizeof(value_t) * total_matrix_elems, g_sycl_queue));

    p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    if (!batch_shape.empty())
    {
        p_batch_divisors = static_cast<uint64_t*>( sycl::malloc_device
            (sizeof(uint64_t) * batch_shape.size(), g_sycl_queue));
        p_batch_strides = static_cast<uint64_t*>(sycl::malloc_device
            (sizeof(uint64_t) * batch_shape.size(), g_sycl_queue));
    }

    if (!p_A || !p_Q || !p_temp || !p_error_flag ||
        (!batch_shape.empty() && (!p_batch_divisors || !p_batch_strides)))
    {
        sycl::free(p_A, g_sycl_queue);
        sycl::free(p_Q, g_sycl_queue);
        sycl::free(p_temp, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        sycl::free(p_batch_divisors, g_sycl_queue);
        sycl::free(p_batch_strides, g_sycl_queue);
        throw std::bad_alloc();
    }

    *p_error_flag = 0;

    const value_t* p_src = tensor.get_data();
    const uint64_t stride_row = tensor_strides[rank - 2];
    const uint64_t stride_col = tensor_strides[rank - 1];
    const int64_t batch_rank = static_cast<int64_t>(batch_shape.size());

    if (!batch_shape.empty())
    {
        g_sycl_queue.memcpy(p_batch_divisors, batch_divisors.data(),
                            sizeof(uint64_t) * batch_rank).wait();
        g_sycl_queue.memcpy(p_batch_strides, batch_strides.data(),
                            sizeof(uint64_t) * batch_rank).wait();
    }

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(total_matrix_elems),
            [=](sycl::id<1> idx)
        {
            uint64_t flat = static_cast<uint64_t>(idx[0]);
            uint64_t batch_idx = flat / matrix_size;
            uint64_t in_mat = flat % matrix_size;
            uint64_t row = in_mat / n;
            uint64_t col = in_mat % n;
            uint64_t batch_offset = 0;
            if (batch_rank > 0)
            {
                batch_offset = sycl_utils::idx_of(batch_idx,
                    p_batch_divisors, p_batch_strides, batch_rank);
            }
            uint64_t src_offset = batch_offset +
                row * stride_row + col * stride_col;
            value_t v = p_src[src_offset];

            sycl_utils::device_check_nan_and_set(v, p_error_flag);
            sycl_utils::device_check_finite_and_set(v, p_error_flag);

            p_A[flat] = v;
        });
    }).wait();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(total_matrix_elems),
            [=](sycl::id<1> idx)
        {
            uint64_t flat = static_cast<uint64_t>(idx[0]);
            uint64_t in_mat = flat % matrix_size;
            uint64_t row = in_mat / n;
            uint64_t col = in_mat % n;
            if (row == col)
            {
                // Diagonal entry of identity.
                p_Q[flat] = value_t{1};
            }
            else
            {
                // Off-diagonal entry of identity.
                p_Q[flat] = value_t{0};
            }
        });
    }).wait();

    for (uint64_t iter = 0; iter < max_iters; ++iter)
    {
        for (uint64_t col = 0; col < n - 1; ++col)
        {
            for (uint64_t row = col + 1; row < n; ++row)
            {
                g_sycl_queue.submit([&](sycl::handler& cgh)
                {
                    cgh.parallel_for(sycl::range<1>(batch_count),
                        [=](sycl::id<1> b_id)
                    {
                        uint64_t b = static_cast<uint64_t>(b_id[0]);
                        uint64_t base = b * matrix_size;
                        value_t a = p_A[base + col * n + col];
                        value_t b_val = p_A[base + row * n + col];

                        value_t r = sycl_utils::sqrt(a * a + b_val * b_val);

                        sycl_utils::device_check_divzero_and_set
                            (r, p_error_flag);
                        if (r == value_t{0}) return;

                        value_t c = a / r;
                        value_t s = -b_val / r;

                        for (uint64_t k = col; k < n; ++k)
                        {
                            value_t a_ck = p_A[base + col * n + k];
                            value_t a_rk = p_A[base + row * n + k];
                            value_t new_ck = c * a_ck - s * a_rk;
                            value_t new_rk = s * a_ck + c * a_rk;
                            p_A[base + col * n + k] = new_ck;
                            p_A[base + row * n + k] = new_rk;
                        }

                        for (uint64_t k = 0; k < n; ++k)
                        {
                            value_t q_kc = p_Q[base + k * n + col];
                            value_t q_kr = p_Q[base + k * n + row];
                            value_t new_qc = c * q_kc - s * q_kr;
                            value_t new_qr = s * q_kc + c * q_kr;
                            p_Q[base + k * n + col] = new_qc;
                            p_Q[base + k * n + row] = new_qr;
                        }
                    });
                }).wait();
            }
        }

        g_sycl_queue.submit([&](sycl::handler& cgh)
        {
            cgh.parallel_for(sycl::range<1>(total_matrix_elems),
                [=](sycl::id<1> idx)
            {
                uint64_t flat = static_cast<uint64_t>(idx[0]);
                uint64_t b = flat / matrix_size;
                uint64_t in_mat = flat % matrix_size;
                uint64_t row = in_mat / n;
                uint64_t col = in_mat % n;
                uint64_t base = b * matrix_size;

                value_t sum = value_t{0};
                for (uint64_t k = 0; k < n; ++k)
                {
                    sum += p_A[base + row * n + k] * p_Q[base + col * n + k];
                }
                sycl_utils::device_check_finite_and_set(sum, p_error_flag);
                p_temp[flat] = sum;
            });
        }).wait();

        g_sycl_queue.memcpy(p_A, p_temp,
            sizeof(value_t) * total_matrix_elems).wait();

        if (iter % 10 == 9)
        {
            value_t* p_max_off_diag = static_cast<value_t*>(
                sycl::malloc_device(sizeof(value_t), g_sycl_queue));

            g_sycl_queue.memset(p_max_off_diag, 0, sizeof(value_t)).wait();
            // Compute maximum off-diagonal value on device.
            g_sycl_queue.submit([&](sycl::handler& cgh)
            {
                cgh.parallel_for(sycl::range<1>(batch_count * matrix_size),
                    [=](sycl::id<1> idx)
                {
                    uint64_t flat = static_cast<uint64_t>(idx[0]);
                    uint64_t b = flat / matrix_size;
                    uint64_t in_mat = flat % matrix_size;
                    uint64_t i = in_mat / n;
                    uint64_t j = in_mat % n;
                    if (i != j)
                    {
                        value_t val = sycl_utils::fabs
                            (p_A[b * matrix_size + i * n + j]);

                        sycl_utils::device_check_nan_and_set
                            (val, p_error_flag);
                        sycl_utils::device_check_finite_and_set
                            (val, p_error_flag);

                        sycl::atomic_ref<value_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_max(*p_max_off_diag);

                        value_t old_max = atomic_max.load();
                        while (val > old_max &&
                            !atomic_max.compare_exchange_strong(old_max, val))
                        {
                            old_max = atomic_max.load();
                        }
                    }
                });
            }).wait();

            value_t max_off_diag;
            g_sycl_queue.memcpy(&max_off_diag,
                p_max_off_diag, sizeof(value_t)).wait();

            sycl::free(p_max_off_diag, g_sycl_queue);

            if (max_off_diag < tol) break;
            if (*p_error_flag != 0) break;
        }
    }

    value_t* p_eigvals = eigvals_tensor.get_data();
    value_t* p_eigvecs = eigvecs_tensor.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(batch_count * n),
            [=](sycl::id<1> idx)
        {
            uint64_t flat = static_cast<uint64_t>(idx[0]);
            uint64_t b = flat / n;
            uint64_t i = flat % n;
            value_t v = p_A[b * matrix_size + i * n + i];
            sycl_utils::device_check_nan_and_set(v, p_error_flag);
            sycl_utils::device_check_finite_and_set(v, p_error_flag);
            p_eigvals[flat] = v;
        });
    }).wait();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(total_matrix_elems),
            [=](sycl::id<1> idx)
        {
            uint64_t flat = static_cast<uint64_t>(idx[0]);
            value_t v = p_Q[flat];
            sycl_utils::device_check_nan_and_set(v, p_error_flag);
            sycl_utils::device_check_finite_and_set(v, p_error_flag);
            p_eigvecs[flat] = v;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_A, g_sycl_queue);
    sycl::free(p_Q, g_sycl_queue);
    sycl::free(p_temp, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_batch_divisors, g_sycl_queue);
    sycl::free(p_batch_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(eig:
                NaN detected in inputs or during computation.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(eig:
                non-finite result (overflow or Inf) during computation.)");
        }
        if (err == 3)
        {
            throw std::runtime_error(R"(eig:
                division by zero detected during computation.)");
        }
        throw std::runtime_error(R"(eig:
            numeric error during eigendecomposition.)");
    }

    return std::make_pair(eigvals_tensor, eigvecs_tensor);
}
template std::pair<Tensor<float>, Tensor<float>> eig<float>
    (const Tensor<float>&, uint64_t, float);

template <typename value_t>
Tensor<value_t> sqrt(const Tensor<value_t>& tensor)
{
    const int64_t rank = tensor.get_rank();
    if (rank == 0)
    {
        throw std::invalid_argument("sqrt: input tensor has no elements.");
    }
    const std::vector<uint64_t>& dims = tensor.get_dimensions();

    const uint64_t total_elems = tensor.get_num_elements();
    MemoryLocation mem_loc = tensor.get_memory_location();
    Tensor<value_t> result(dims, mem_loc);

    const std::vector<uint64_t> divisors =
        temper::utils::compute_divisors(dims);
    const std::vector<uint64_t> strides = tensor.get_strides();

    uint64_t* p_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    if (!p_divs || !p_strides || !p_error_flag)
    {
        sycl::free(p_divs, g_sycl_queue);
        sycl::free(p_strides, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_divs, divisors.data(),
                        sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_strides, strides.data(),
                        sizeof(uint64_t) * rank).wait();
    *p_error_flag = 0;

    const value_t* p_in = tensor.get_data();
    value_t* p_out = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);
            uint64_t idx = temper::sycl_utils::idx_of(
                flat, p_divs, p_strides, rank);

            value_t val = p_in[idx];
            temper::sycl_utils::device_check_nan_and_set<value_t>
                (val, p_error_flag);

            value_t outv = sycl::sqrt(val);
            temper::sycl_utils::device_check_finite_and_set<value_t>
                (outv, p_error_flag);
            p_out[flat] = outv;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_divs, g_sycl_queue);
    sycl::free(p_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error("sqrt: NaN detected in inputs.");
        }
        if (err == 2)
        {
            throw std::runtime_error("sqrt: non-finite result produced.");
        }
        throw std::runtime_error("sqrt: numeric error during computation.");
    }

    return result;
}
template Tensor<float> sqrt<float>(const Tensor<float>& tensor);

template<typename value_t>
Tensor<value_t> exp(const Tensor<value_t> & tensor)
{
    const std::vector<uint64_t> & in_shape = tensor.get_dimensions();
    if (in_shape.empty())
    {
        throw std::invalid_argument(R"(exp: input tensor has no elements.)");
    }

    const uint64_t arr_len = static_cast<uint64_t>(in_shape.size());
    const uint64_t total_output_elems = tensor.get_num_elements();
    MemoryLocation res_loc = tensor.get_memory_location();
    Tensor<value_t> result(in_shape, res_loc);

    const std::vector<uint64_t> in_divs =
        temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = tensor.get_strides();

    uint64_t* p_in_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(arr_len),
                            g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(arr_len),
                            g_sycl_queue));
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_in_divs && p_in_strides && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_in_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_in_divs, in_divs.data(),
        sizeof(uint64_t) * arr_len).wait();
    g_sycl_queue.memcpy(p_in_strides, in_strides.data(),
        sizeof(uint64_t) * arr_len).wait();
    *p_error_flag = 0;

    const value_t* p_in_data = tensor.get_data();
    value_t* p_out = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t in_idx = temper::sycl_utils::idx_of(flat,
                                                    p_in_divs,
                                                    p_in_strides,
                                                    arr_len);
            value_t v = p_in_data[in_idx];

            temper::sycl_utils::device_check_nan_and_set<value_t>
                (v, p_error_flag);

            value_t outv = sycl::exp(v);

            temper::sycl_utils::device_check_finite_and_set<value_t>
                (outv, p_error_flag);
            p_out[flat] = outv;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_in_divs, g_sycl_queue);
    sycl::free(p_in_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(exp: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(exp:
                non-finite result (Inf/overflow/NaN) produced.)");
        }
        throw std::runtime_error(R"(exp: numeric error during exp computation.)");
    }

    return result;
}
template Tensor<float> exp<float>(const Tensor<float>&);

} // namespace temper::math
