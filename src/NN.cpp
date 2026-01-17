/**
 * @file NN.cpp
 * @brief Neural network utility definitions.
 */

#include "temper/NN.hpp"
#include "temper/Math.hpp"
#include "temper/Utils.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Errors.hpp"

namespace temper::nn
{

template<typename value_t>
Tensor<value_t> conv2d(const Tensor<value_t>& input,
    const Tensor<value_t>& kernel,
    uint64_t stride,
    std::pair<uint64_t, uint64_t> padding)
{
    const std::vector<uint64_t>& in_shape = input.get_dimensions();
    const std::vector<uint64_t>& ker_shape = kernel.get_dimensions();
    const int64_t in_rank = input.get_rank();
    const int64_t ker_rank = kernel.get_rank();

    TEMPER_CHECK(in_rank >= 3,
        validation_error,
        R"(conv2d: input tensor must have rank >= 3
           (at least in_channels, height, width).)");

    TEMPER_CHECK(ker_rank >= 4,
        validation_error,
        R"(conv2d: kernel tensor must have rank >= 4
           (at least out_channels, in_channels, kernel_h, kernel_w).)");

    TEMPER_CHECK(stride > 0,
        validation_error,
        R"(conv2d: stride must be positive.)");

    const uint64_t in_channels = in_shape[in_rank - 3];
    const uint64_t in_height = in_shape[in_rank - 2];
    const uint64_t in_width = in_shape[in_rank - 1];

    const uint64_t out_channels = ker_shape[ker_rank - 4];
    const uint64_t ker_in_channels = ker_shape[ker_rank - 3];
    const uint64_t ker_height = ker_shape[ker_rank - 2];
    const uint64_t ker_width = ker_shape[ker_rank - 1];

    const uint64_t pad_h = padding.first;
    const uint64_t pad_w = padding.second;

    TEMPER_CHECK(in_channels == ker_in_channels,
        validation_error,
        R"(conv2d: input channels must match kernel input channels.)");

    TEMPER_CHECK(in_height + 2 * pad_h >= ker_height,
        validation_error,
        R"(conv2d: kernel height larger than padded input height.)");

    TEMPER_CHECK(in_width + 2 * pad_w >= ker_width,
        validation_error,
        R"(conv2d: kernel width larger than padded input width.)");

    const uint64_t out_height =
        (in_height + 2 * pad_h - ker_height) / stride + 1;
    const uint64_t out_width =
        (in_width + 2 * pad_w - ker_width) / stride + 1;

    // Handle broadcasting on leading dimensions.
    const int64_t in_leading_rank = in_rank - 3;
    const int64_t ker_leading_rank = ker_rank - 4;

    std::vector<uint64_t> in_leading_shape(in_shape.begin(),
        in_shape.begin() + in_leading_rank);
    std::vector<uint64_t> ker_leading_shape(ker_shape.begin(),
        ker_shape.begin() + ker_leading_rank);

    std::vector<uint64_t> leading_shape;
    std::vector<uint64_t> in_leading_strides;
    std::vector<uint64_t> ker_leading_strides;

    if (in_leading_rank > 0 || ker_leading_rank > 0)
    {
        temper::utils::TensorDesc in_leading_desc{
            in_leading_shape,
            std::vector<uint64_t>(input.get_strides().begin(),
                input.get_strides().begin() + in_leading_rank)
        };
        temper::utils::TensorDesc ker_leading_desc{
            ker_leading_shape,
            std::vector<uint64_t>(kernel.get_strides().begin(),
                kernel.get_strides().begin() + ker_leading_rank)
        };

        auto bcast = temper::utils::compute_broadcast(
            {in_leading_desc, ker_leading_desc});
        leading_shape = std::move(bcast.shape);
        in_leading_strides = std::move(bcast.strides[0]);
        ker_leading_strides = std::move(bcast.strides[1]);
    }

    // Construct output shape.
    std::vector<uint64_t> out_shape = leading_shape;
    out_shape.push_back(out_channels);
    out_shape.push_back(out_height);
    out_shape.push_back(out_width);

    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    MemoryLocation res_loc = MemoryLocation::HOST;
    if (input.get_memory_location() == MemoryLocation:: DEVICE ||
        kernel.get_memory_location() == MemoryLocation::DEVICE)
    {
        res_loc = MemoryLocation::DEVICE;
    }

    Tensor<value_t> result(out_shape, res_loc);

    uint64_t leading_count = 1;
    for (uint64_t dim : leading_shape)
    {
        leading_count *= dim;
    }

    const uint64_t total_output_elems =
        leading_count * out_channels * out_height * out_width;

    std::vector<uint64_t> out_divs =
        temper::utils::compute_divisors(out_shape);

    sycl_utils::SyclArray<uint64_t> out_divs_arr(g_sycl_queue,
        out_divs, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_arr(g_sycl_queue,
        1, MemoryLocation::HOST);

    const uint64_t* p_out_divs = out_divs_arr;
    int32_t* p_error_flag = error_flag_arr;

    *p_error_flag = 0;

    const std::vector<uint64_t>& in_strides = input.get_strides();
    const std::vector<uint64_t>& ker_strides = kernel.get_strides();

    const uint64_t in_channel_stride = in_strides[in_rank - 3];
    const uint64_t in_row_stride = in_strides[in_rank - 2];
    const uint64_t in_col_stride = in_strides[in_rank - 1];

    const uint64_t ker_out_channel_stride = ker_strides[ker_rank - 4];
    const uint64_t ker_in_channel_stride = ker_strides[ker_rank - 3];
    const uint64_t ker_row_stride = ker_strides[ker_rank - 2];
    const uint64_t ker_col_stride = ker_strides[ker_rank - 1];

    std::optional<sycl_utils::SyclArray<uint64_t>>
        in_leading_strides_arr_opt;
    std::optional<sycl_utils::SyclArray<uint64_t>>
        ker_leading_strides_arr_opt;
    std:: optional<sycl_utils:: SyclArray<uint64_t>>
        leading_divs_arr_opt;

    const uint64_t* p_in_leading_strides = nullptr;
    const uint64_t* p_ker_leading_strides = nullptr;
    const uint64_t* p_leading_divs = nullptr;

    const int64_t leading_rank =
        static_cast<int64_t>(leading_shape.size());

    if (leading_rank > 0)
    {
        std::vector<uint64_t> leading_divs =
            temper::utils::compute_divisors(leading_shape);

        in_leading_strides_arr_opt.emplace(g_sycl_queue,
            in_leading_strides, MemoryLocation::DEVICE);
        ker_leading_strides_arr_opt.emplace(g_sycl_queue,
            ker_leading_strides, MemoryLocation:: DEVICE);
        leading_divs_arr_opt.emplace(g_sycl_queue,
            leading_divs, MemoryLocation::DEVICE);

        p_in_leading_strides = in_leading_strides_arr_opt->data();
        p_ker_leading_strides = ker_leading_strides_arr_opt->data();
        p_leading_divs = leading_divs_arr_opt->data();
    }

    const value_t* p_input = input.get_data();
    const value_t* p_kernel = kernel.get_data();
    value_t* p_output = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl:: range<1>(
            static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t out_col_idx = (flat / p_out_divs[out_rank - 1]) %
                out_width;
            uint64_t out_row_idx = (flat / p_out_divs[out_rank - 2]) %
                out_height;
            uint64_t out_ch_idx = (flat / p_out_divs[out_rank - 3]) %
                out_channels;
            uint64_t leading_flat = flat / (out_channels * out_height * out_width);

            uint64_t in_leading_offset = 0;
            uint64_t ker_leading_offset = 0;

            if (leading_rank > 0)
            {
                in_leading_offset = temper::sycl_utils::idx_of(leading_flat,
                    p_leading_divs, p_in_leading_strides, leading_rank);
                ker_leading_offset = temper:: sycl_utils::idx_of(leading_flat,
                    p_leading_divs, p_ker_leading_strides, leading_rank);
            }

            value_t acc = value_t{0};

            for (uint64_t ic = 0; ic < in_channels; ++ic)
            {
                for (uint64_t kh = 0; kh < ker_height; ++kh)
                {
                    for (uint64_t kw = 0; kw < ker_width; ++kw)
                    {
                        int64_t in_row =
                            static_cast<int64_t>(
                                out_row_idx * stride + kh)
                            - static_cast<int64_t>(pad_h);
                        int64_t in_col =
                            static_cast<int64_t>(
                                out_col_idx * stride + kw)
                            - static_cast<int64_t>(pad_w);

                        if (in_row >= 0 &&
                            in_row < static_cast<int64_t>(in_height) &&
                            in_col >= 0 &&
                            in_col < static_cast<int64_t>(in_width))
                        {
                            uint64_t in_offset = in_leading_offset +
                                ic * in_channel_stride +
                                static_cast<uint64_t>(in_row) *
                                    in_row_stride +
                                static_cast<uint64_t>(in_col) *
                                    in_col_stride;

                            uint64_t ker_kh_flip = ker_height - 1 - kh;
                            uint64_t ker_kw_flip = ker_width - 1 - kw;

                            uint64_t ker_offset = ker_leading_offset +
                                out_ch_idx * ker_out_channel_stride +
                                ic * ker_in_channel_stride +
                                ker_kh_flip * ker_row_stride +
                                ker_kw_flip * ker_col_stride;

                            value_t in_val = p_input[in_offset];
                            value_t ker_val = p_kernel[ker_offset];

                            TEMPER_DEVICE_ASSERT(
                                !sycl_utils::is_nan(in_val),
                                p_error_flag, 1);
                            TEMPER_DEVICE_ASSERT(
                                !sycl_utils::is_nan(ker_val),
                                p_error_flag, 1);

                            acc += in_val * ker_val;
                        }
                    }
                }
            }

            TEMPER_DEVICE_ASSERT(sycl_utils::is_finite(acc),
                p_error_flag, 2);

            p_output[flat] = acc;
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err != 1,
        nan_error,
        R"(conv2d: NaN detected in inputs.)");

    TEMPER_CHECK(err != 2,
        nonfinite_error,
        R"(conv2d: non-finite result (overflow or Inf) produced.)");

    return result;
}
template Tensor<float> conv2d<float>
    (const Tensor<float>&, const Tensor<float>&, uint64_t,
     std::pair<uint64_t, uint64_t>);

template <typename value_t>
Tensor<value_t> conv2d_transpose(
    const Tensor<value_t>& input,
    const Tensor<value_t>& kernel,
    uint64_t stride,
    std::pair<uint64_t, uint64_t> padding,
    std::pair<uint64_t, uint64_t> output_padding)
{
    const std::vector<uint64_t>& input_shape = input.get_dimensions();
    const std::vector<uint64_t>& kernel_shape = kernel.get_dimensions();

    const int64_t input_rank  = input.get_rank();
    const int64_t kernel_rank = kernel.get_rank();

    TEMPER_CHECK(input_rank >= 3,
        validation_error,
        "conv2d_transpose: input must have rank >= 3");

    TEMPER_CHECK(kernel_rank >= 4,
        validation_error,
        "conv2d_transpose: kernel must have rank >= 4");

    TEMPER_CHECK(stride > 0,
        validation_error,
        "conv2d_transpose: stride must be positive");

    const uint64_t in_channels_input = input_shape[input_rank - 3];
    const uint64_t ker_in_channels   = kernel_shape[kernel_rank - 4];
    const uint64_t kernel_h          = kernel_shape[kernel_rank - 2];
    const uint64_t kernel_w          = kernel_shape[kernel_rank - 1];

    TEMPER_CHECK(in_channels_input == ker_in_channels,
        validation_error,
        "conv2d_transpose: input channels must match between input and kernel");

    // Upsample input.
    Tensor<value_t> upsampled =
        math::upsample(input, stride, math::UpsampleMode::ZEROS);

    // Apply output_padding before convolution.
    if (output_padding.first > 0 || output_padding.second > 0)
    {
        upsampled = math::pad(
            upsampled,
            0, output_padding.first,
            0, output_padding.second,
            static_cast<value_t>(0)
        );
    }

    // Transpose kernel (in_ch, out_ch, kH, kW) -> (out_ch, in_ch, kH, kW).
    std::vector<int64_t> transpose_axes;
    for (int64_t i = 0; i < kernel_rank - 4; ++i)
        transpose_axes.push_back(i);

    transpose_axes.push_back(kernel_rank - 3);
    transpose_axes.push_back(kernel_rank - 4);
    transpose_axes.push_back(kernel_rank - 2);
    transpose_axes.push_back(kernel_rank - 1);

    Tensor<value_t> kernel_transposed =
        math::transpose(kernel, transpose_axes);

    // Convert padding for regular conv2d.
    const uint64_t conv_pad_h =
        (kernel_h > padding.first) ? (kernel_h - 1 - padding.first) : 0;
    const uint64_t conv_pad_w =
        (kernel_w > padding.second) ? (kernel_w - 1 - padding.second) : 0;

    // Regular conv2d with stride = 1.
    return conv2d(
        upsampled,
        kernel_transposed,
        1,
        {conv_pad_h, conv_pad_w}
    );
}
template Tensor<float> conv2d_transpose<float>
    (const Tensor<float>&, const Tensor<float>&, uint64_t,
     std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>);

template<typename value_t>
Tensor<value_t> relu(const Tensor<value_t>& tensor)
{
    const int64_t rank = tensor.get_rank();
    TEMPER_CHECK(rank > 0,
        validation_error,
        R"(relu: input tensor has no elements.)");

    const std::vector<uint64_t>& shape = tensor.get_dimensions();
    const uint64_t num_elements = tensor.get_num_elements();

    MemoryLocation res_loc = tensor.get_memory_location();
    Tensor<value_t> result(shape, res_loc);

    const std::vector<uint64_t>& in_strides = tensor.get_strides();
    std::vector<uint64_t> divs = temper::utils::compute_divisors(shape);

    sycl_utils::SyclArray<uint64_t> divs_arr(g_sycl_queue,
        divs, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> strides_arr(g_sycl_queue,
        in_strides, MemoryLocation:: DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_arr(g_sycl_queue,
        1, MemoryLocation::HOST);

    const uint64_t* p_divs = divs_arr;
    const uint64_t* p_strides = strides_arr;
    int32_t* p_error_flag = error_flag_arr;

    *p_error_flag = 0;

    const value_t* p_input = tensor.get_data();
    value_t* p_output = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl:: range<1>(
            static_cast<size_t>(num_elements)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t in_idx = temper::sycl_utils:: idx_of(
                flat, p_divs, p_strides, rank);

            value_t val = p_input[in_idx];

            TEMPER_DEVICE_ASSERT(!sycl_utils::is_nan(val),
                p_error_flag, 1);

            value_t out_val = val > value_t{0} ? val : value_t{0};

            TEMPER_DEVICE_ASSERT(sycl_utils::is_finite(out_val),
                p_error_flag, 2);

            p_output[flat] = out_val;
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err != 1,
        nan_error,
        R"(relu: NaN detected in input.)");

    TEMPER_CHECK(err != 2,
        nonfinite_error,
        R"(relu: non-finite result produced.)");

    return result;
}
template Tensor<float> relu<float>(const Tensor<float>&);

} // namespace temper::nn