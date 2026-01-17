/**
 * @file ut_NN.cpp
 * @brief Google Test suite for neural network utilities.
 *
 * Contains unit tests ensuring correctness of functions implemented
 * in the NN module.
 */

#include <gtest/gtest.h>

#include "temper/Errors.hpp"

#define private public
#define protected public
#include "temper/NN.hpp"
#undef private
#undef protected

using namespace temper;

namespace Test
{

#include "nn/conv2d.cpp"

/**
  * @test CONV2D.conv2d_broadcast_input_batch
  * @brief Input has leading batch dimension, kernel broadcasts over it.
  */
TEST(CONV2D, conv2d_broadcast_input_batch)
{

    Tensor<float> input({2, 1, 2, 2});
    input = std::vector<float>{
        // Batch 0
        1.0, 2.0,
        3.0, 4.0,
        // Batch 1
        5.0, 6.0,
        7.0, 8.0
    };

    Tensor<float> kernel({1, 1, 2, 2});
    kernel = std:: vector<float>{
        1.0, 0.0,
        0.0, 1.0
    };
    Tensor<float> result = nn::conv2d(input, kernel, 1, {0, 0});

    EXPECT_EQ(result.get_dimensions(), std::vector<uint64_t>({2, 1, 1, 1}));
    EXPECT_NEAR(result[0][0][0][0], 5.0, 0.0001);
    EXPECT_NEAR(result[1][0][0][0], 13.0, 0.0001);
}

/**
  * @test CONV2D. conv2d_broadcast_kernel_batch
  * @brief Kernel has leading batch dimension, input broadcasts over it.
  */
TEST(CONV2D, conv2d_broadcast_kernel_batch)
{

    Tensor<float> input({1, 2, 2});
    input = std::vector<float>{
        1.0, 2.0,
        3.0, 4.0
    };

    Tensor<float> kernel({2, 1, 1, 2, 2});
    kernel = std::vector<float>{
        1.0, 1.0,
        1.0, 1.0,

        1.0, 0.0,
        0.0, 1.0
    };

    Tensor<float> result = nn::conv2d(input, kernel, 1, {0, 0});

    EXPECT_EQ(result.get_dimensions(), std::vector<uint64_t>({2, 1, 1, 1}));
    EXPECT_NEAR(result[0][0][0][0], 10.0, 0.0001);
    EXPECT_NEAR(result[1][0][0][0], 5.0, 0.0001);
}

/**
  * @test CONV2D.conv2d_broadcast_both_leading
  * @brief Both input and kernel have broadcastable leading dimensions.
  */
TEST(CONV2D, conv2d_broadcast_both_leading)
{
    Tensor<float> input({2, 1, 1, 2, 2});
    input = std::vector<float>{
        1.0, 2.0,
        3.0, 4.0,

        5.0, 6.0,
        7.0, 8.0
    };

    Tensor<float> kernel({1, 2, 1, 1, 2, 2});
    kernel = std::vector<float>{
        1.0, 1.0,
        1.0, 1.0,

        1.0, 0.0,
        0.0, 1.0
    };

    Tensor<float> result = nn::conv2d(input, kernel, 1, {0, 0});

    EXPECT_EQ(result. get_dimensions(),
              std::vector<uint64_t>({2, 2, 1, 1, 1}));
    EXPECT_NEAR(result[0][0][0][0][0], 10.0, 0.0001);
    EXPECT_NEAR(result[0][1][0][0][0], 5.0, 0.0001);
    EXPECT_NEAR(result[1][0][0][0][0], 26.0, 0.0001);
    EXPECT_NEAR(result[1][1][0][0][0], 13.0, 0.0001);
}

/**
  * @test CONV2D.conv2d_broadcast_scalar_like_kernel
  * @brief Kernel with singleton leading dimension broadcasts.
  */
TEST(CONV2D, conv2d_broadcast_scalar_like_kernel)
{
    Tensor<float> input({3, 1, 2, 2});
    input = std::vector<float>{
        // Batch 0
        1.0, 1.0,
        1.0, 1.0,
        // Batch 1
        2.0, 2.0,
        2.0, 2.0,
        // Batch 2
        3.0, 3.0,
        3.0, 3.0
    };

    Tensor<float> kernel({1, 1, 1, 2, 2});
    kernel = std::vector<float>{
        1.0, 1.0,
        1.0, 1.0
    };

    Tensor<float> result = nn::conv2d(input, kernel, 1, {0, 0});

    EXPECT_EQ(result.get_dimensions(), std::vector<uint64_t>({3, 1, 1, 1}));
    EXPECT_NEAR(result[0][0][0][0], 4.0, 0.0001);
    EXPECT_NEAR(result[1][0][0][0], 8.0, 0.0001);
    EXPECT_NEAR(result[2][0][0][0], 12.0, 0.0001);
}

/**
  * @test CONV2D.conv2d_input_rank_too_low
  * @brief Input tensor must have rank >= 3.
  */
TEST(CONV2D, conv2d_input_rank_too_low)
{
    Tensor<float> input({3, 3});
    Tensor<float> kernel({1, 1, 2, 2});

    EXPECT_THROW({
        nn::conv2d(input, kernel, 1, {0, 0});
    }, temper::validation_error);
}

/**
  * @test CONV2D.conv2d_kernel_rank_too_low
  * @brief Kernel tensor must have rank >= 4.
  */
TEST(CONV2D, conv2d_kernel_rank_too_low)
{
    Tensor<float> input({1, 3, 3});
    Tensor<float> kernel({2, 2});

    EXPECT_THROW({
        nn::conv2d(input, kernel, 1, {0, 0});
    }, temper::validation_error);
}

/**
  * @test CONV2D.conv2d_stride_zero
  * @brief Stride must be positive (non-zero).
  */
TEST(CONV2D, conv2d_stride_zero)
{
    Tensor<float> input({1, 3, 3});
    Tensor<float> kernel({1, 1, 2, 2});

    EXPECT_THROW({
        nn::conv2d(input, kernel, 0, {0, 0});
    }, temper:: validation_error);
}

/**
  * @test CONV2D.conv2d_channel_mismatch
  * @brief Input channels must match kernel input channels.
  */
TEST(CONV2D, conv2d_channel_mismatch)
{
    Tensor<float> input({2, 3, 3});
    Tensor<float> kernel({1, 3, 2, 2});

    EXPECT_THROW({
        nn::conv2d(input, kernel, 1, {0, 0});
    }, temper::validation_error);
}

/**
  * @test CONV2D. conv2d_kernel_height_too_large
  * @brief Kernel height cannot exceed padded input height.
  */
TEST(CONV2D, conv2d_kernel_height_too_large)
{
    Tensor<float> input({1, 3, 3});
    Tensor<float> kernel({1, 1, 5, 2});

    EXPECT_THROW({
        nn:: conv2d(input, kernel, 1, {0, 0});
    }, temper::validation_error);
}

/**
  * @test CONV2D. conv2d_kernel_width_too_large
  * @brief Kernel width cannot exceed padded input width.
  */
TEST(CONV2D, conv2d_kernel_width_too_large)
{
    Tensor<float> input({1, 3, 3});
    Tensor<float> kernel({1, 1, 2, 5});

    EXPECT_THROW({
        nn::conv2d(input, kernel, 1, {0, 0});
    }, temper::validation_error);
}

/**
  * @test CONV2D.conv2d_kernel_too_large_with_padding
  * @brief Kernel can be larger than input if padding helps, but still limited.
  */
TEST(CONV2D, conv2d_kernel_too_large_with_padding)
{
    Tensor<float> input({1, 3, 3});
    Tensor<float> kernel({1, 1, 5, 5});
    EXPECT_NO_THROW({
        nn:: conv2d(input, kernel, 1, {1, 1});
    });
    EXPECT_THROW({
        nn::conv2d(input, kernel, 1, {0, 0});
    }, temper::validation_error);
}

/**
  * @test CONV2D.conv2d_nan_input_throws
  * @brief NaN values in input should be detected.
  */
TEST(CONV2D, conv2d_nan_input_throws)
{
    Tensor<float> input({1, 2, 2});
    input = std::vector<float>{
        1.0, std::numeric_limits<float>::quiet_NaN(),
        3.0, 4.0
    };

    Tensor<float> kernel({1, 1, 2, 2});
    kernel = std::vector<float>{1.0, 1.0, 1.0, 1.0};
    EXPECT_THROW({
        nn::conv2d(input, kernel, 1, {0, 0});
    }, temper::nan_error);
}

/**
  * @test CONV2D.conv2d_nan_kernel_throws
  * @brief NaN values in kernel should be detected.
  */
TEST(CONV2D, conv2d_nan_kernel_throws)
{
    Tensor<float> input({1, 2, 2});
    input = std::vector<float>{1.0, 2.0, 3.0, 4.0};
    Tensor<float> kernel({1, 1, 2, 2});
    kernel = std::vector<float>{
        1.0, std::numeric_limits<float>::quiet_NaN(),
        3.0, 4.0
    };
    EXPECT_THROW({
        nn:: conv2d(input, kernel, 1, {0, 0});
    }, temper::nan_error);
}

/**
  * @test CONV2D.conv2d_inf_throws
  * @brief Overflow producing Inf should be detected.
  */
TEST(CONV2D, conv2d_inf_throws)
{
    Tensor<float> input({1, 2, 2});
    float large = std::numeric_limits<float>:: max();
    input = std::vector<float>{large, large, large, large};

    Tensor<float> kernel({1, 1, 2, 2});
    kernel = std::vector<float>{large, large, large, large};

    EXPECT_THROW({
        nn::conv2d(input, kernel, 1, {0, 0});
    }, temper::nonfinite_error);
}

#include "nn/conv2d_transpose.cpp"

/**
 * @test TypedConv2DTranspose.broadcast_high_rank
 * @brief conv2d_transpose should correctly broadcast kernel over
 *        higher-rank leading dimensions
 */
TYPED_TEST(TypedConv2DTranspose, broadcast_high_rank)
{
    using value_t = TypeParam;

    /*
     * input shape:
     *   [2, 3, 4, 5, 5]
     *   leading dims = [2, 3]
     *   channels     = 4
     *   H, W         = 5, 5
     */
    Tensor<value_t> input(
        {2, 3, 4, 5, 5},
        MemoryLocation::DEVICE
    );

    /*
     * kernel shape:
     *   [4, 6, 3, 3]
     *   no leading dims
     *   in_ch = 4, out_ch = 6
     */
    Tensor<value_t> kernel(
        {4, 6, 3, 3},
        MemoryLocation::DEVICE
    );

    Tensor<value_t> output;

    EXPECT_NO_THROW(
        output = nn::conv2d_transpose(
            input,
            kernel,
            2,
            {1, 1},
            {0, 0}
        )
    );

    const auto &out_shape = output.get_dimensions();

    EXPECT_EQ(out_shape.size(), 5u);
    EXPECT_EQ(out_shape[0], 2u);
    EXPECT_EQ(out_shape[1], 3u);
    EXPECT_EQ(out_shape[2], 6u);
}

/**
 * @test TypedConv2DTranspose.broadcast_reverse_high_rank
 * @brief conv2d_transpose should broadcast input over
 *        higher-rank kernel leading dimensions
 */
TYPED_TEST(TypedConv2DTranspose, broadcast_reverse_high_rank)
{
    using value_t = TypeParam;

    /*
     * input shape:
     *   [4, 5, 5]
     *   channels = 4
     */
    Tensor<value_t> input(
        {4, 5, 5},
        MemoryLocation::DEVICE
    );

    /*
     * kernel shape:
     *   [2, 3, 4, 6, 3, 3]
     *   leading dims = [2, 3]
     *   in_ch = 4, out_ch = 6
     */
    Tensor<value_t> kernel(
        {2, 3, 4, 6, 3, 3},
        MemoryLocation::DEVICE
    );

    Tensor<value_t> output;

    EXPECT_NO_THROW(
        output = nn::conv2d_transpose(
            input,
            kernel,
            1,
            {0, 0},
            {0, 0}
        )
    );

    const auto &out_shape = output.get_dimensions();

    EXPECT_EQ(out_shape.size(), 5u);
    EXPECT_EQ(out_shape[0], 2u);
    EXPECT_EQ(out_shape[1], 3u);
    EXPECT_EQ(out_shape[2], 6u);
}

/**
 * @test TypedConv2DTranspose.broadcast_intertwined_high_rank
 * @brief conv2d_transpose should correctly broadcast when
 *        input and kernel leading dimensions intertwine
 */
TYPED_TEST(TypedConv2DTranspose, broadcast_intertwined_high_rank)
{
    using value_t = TypeParam;

    /*
     * input shape:
     *   [2, 1, 4, 5, 5]
     *   leading dims = [2, 1]
     */
    Tensor<value_t> input(
        {2, 1, 4, 5, 5},
        MemoryLocation::DEVICE
    );

    /*
     * kernel shape:
     *   [1, 3, 4, 6, 3, 3]
     *   leading dims = [1, 3]
     */
    Tensor<value_t> kernel(
        {1, 3, 4, 6, 3, 3},
        MemoryLocation::DEVICE
    );

    Tensor<value_t> output;

    EXPECT_NO_THROW(
        output = nn::conv2d_transpose(
            input,
            kernel,
            2,
            {1, 1},
            {0, 0}
        )
    );

    const auto &out_shape = output.get_dimensions();

    EXPECT_EQ(out_shape.size(), 5u);
    EXPECT_EQ(out_shape[0], 2u);
    EXPECT_EQ(out_shape[1], 3u);
    EXPECT_EQ(out_shape[2], 6u);
}


/**
 * @test TypedConv2DTranspose.error_input_rank_too_low
 * @brief conv2d_transpose should throw validation_error for input
 *        rank < 3
 * Corresponds to: TEMPER_CHECK(input_rank < 3, ...)
 */
TYPED_TEST(TypedConv2DTranspose, error_input_rank_too_low)
{
    using value_t = TypeParam;

    Tensor<value_t> input(
        {4, 4},
        MemoryLocation::DEVICE
    );
    Tensor<value_t> kernel(
        {1, 1, 3, 3},
        MemoryLocation::DEVICE
    );

    EXPECT_THROW(
        nn::conv2d_transpose(
            input,
            kernel,
            1,
            {0, 0},
            {0, 0}
        ),
        temper::validation_error
    );
}

/**
 * @test TypedConv2DTranspose.error_kernel_rank_too_low
 * @brief conv2d_transpose should throw validation_error for kernel
 *        rank < 4
 * Corresponds to: TEMPER_CHECK(kernel_rank < 4, ...)
 */
TYPED_TEST(TypedConv2DTranspose, error_kernel_rank_too_low)
{
    using value_t = TypeParam;

    Tensor<value_t> input(
        {1, 4, 4},
        MemoryLocation::DEVICE
    );
    Tensor<value_t> kernel(
        {3, 3, 3},
        MemoryLocation::DEVICE
    );

    EXPECT_THROW(
        nn::conv2d_transpose(
            input,
            kernel,
            1,
            {0, 0},
            {0, 0}
        ),
        temper::validation_error
    );
}

/**
 * @test TypedConv2DTranspose.error_stride_zero
 * @brief conv2d_transpose should throw validation_error for
 *        stride == 0
 * Corresponds to: TEMPER_CHECK(stride == 0, ...)
 */
TYPED_TEST(TypedConv2DTranspose, error_stride_zero)
{
    using value_t = TypeParam;

    Tensor<value_t> input(
        {1, 4, 4},
        MemoryLocation::DEVICE
    );
    Tensor<value_t> kernel(
        {1, 1, 3, 3},
        MemoryLocation::DEVICE
    );

    EXPECT_THROW(
        nn::conv2d_transpose(
            input,
            kernel,
            0,
            {0, 0},
            {0, 0}
        ),
        temper::validation_error
    );
}

/**
 * @test TypedConv2DTranspose.error_channel_mismatch
 * @brief conv2d_transpose should throw validation_error when
 *        input channels != kernel input channels
 * Corresponds to:
 *   TEMPER_CHECK(in_channels_input != ker_in_channels, ...)
 */
TYPED_TEST(TypedConv2DTranspose, error_channel_mismatch)
{
    using value_t = TypeParam;

    Tensor<value_t> input(
        {2, 4, 4},
        MemoryLocation::DEVICE
    );
    Tensor<value_t> kernel(
        {3, 1, 3, 3},
        MemoryLocation::DEVICE
    );

    EXPECT_THROW(
        nn::conv2d_transpose(
            input,
            kernel,
            1,
            {0, 0},
            {0, 0}
        ),
        temper::validation_error
    );
}

template<typename T>
class TypedRelu : public :: testing::Test {};

using ReluTestTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(TypedRelu, ReluTestTypes);

/**
 * @test TypedRelu.relu_basic_positive
 * @brief ReLU should pass positive values through unchanged.
 */
TYPED_TEST(TypedRelu, relu_basic_positive)
{
    using value_t = TypeParam;

    Tensor<value_t> input({4}, MemoryLocation:: DEVICE);
    input = std::vector<value_t>{
        static_cast<value_t>(1.0),
        static_cast<value_t>(2.0),
        static_cast<value_t>(3.0),
        static_cast<value_t>(4.0)
    };

    Tensor<value_t> result = nn::relu(input);

    EXPECT_EQ(result.get_dimensions(), input.get_dimensions());
    EXPECT_NEAR(static_cast<double>(result[0]), 1.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1]), 2.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[2]), 3.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[3]), 4.0, 1e-6);
}

/**
 * @test TypedRelu.relu_basic_negative
 * @brief ReLU should clamp negative values to zero.
 */
TYPED_TEST(TypedRelu, relu_basic_negative)
{
    using value_t = TypeParam;

    Tensor<value_t> input({4}, MemoryLocation::DEVICE);
    input = std::vector<value_t>{
        static_cast<value_t>(-1.0),
        static_cast<value_t>(-2.0),
        static_cast<value_t>(-3.0),
        static_cast<value_t>(-4.0)
    };

    Tensor<value_t> result = nn::relu(input);

    EXPECT_EQ(result.get_dimensions(), input.get_dimensions());
    EXPECT_NEAR(static_cast<double>(result[0]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[2]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[3]), 0.0, 1e-6);
}

/**
 * @test TypedRelu.relu_mixed_values
 * @brief ReLU should handle mixed positive and negative values.
 */
TYPED_TEST(TypedRelu, relu_mixed_values)
{
    using value_t = TypeParam;

    Tensor<value_t> input({2, 3}, MemoryLocation::DEVICE);
    input = std::vector<value_t>{
        static_cast<value_t>(-1.0), static_cast<value_t>(0.0),
        static_cast<value_t>(1.0), static_cast<value_t>(-2.0),
        static_cast<value_t>(2.0), static_cast<value_t>(-0.5)
    };

    Tensor<value_t> result = nn:: relu(input);

    EXPECT_EQ(result.get_dimensions(), std::vector<uint64_t>({2, 3}));
    EXPECT_NEAR(static_cast<double>(result[0][0]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[0][1]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[0][2]), 1.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1][0]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1][1]), 2.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1][2]), 0.0, 1e-6);
}

/**
 * @test TypedRelu.relu_zero_input
 * @brief ReLU of zero should be zero.
 */
TYPED_TEST(TypedRelu, relu_zero_input)
{
    using value_t = TypeParam;

    Tensor<value_t> input({3}, MemoryLocation::DEVICE);
    input = std::vector<value_t>{
        static_cast<value_t>(0.0),
        static_cast<value_t>(0.0),
        static_cast<value_t>(0.0)
    };

    Tensor<value_t> result = nn::relu(input);

    EXPECT_NEAR(static_cast<double>(result[0]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[2]), 0.0, 1e-6);
}

/**
 * @test TypedRelu.relu_empty_tensor_throws
 * @brief ReLU on an empty/default tensor should throw validation_error.
 */
TYPED_TEST(TypedRelu, relu_empty_tensor_throws)
{
    using value_t = TypeParam;

    Tensor<value_t> input;

    EXPECT_THROW({
        nn::relu(input);
    }, temper::validation_error);
}

/**
 * @test TypedRelu.relu_nan_input_throws
 * @brief ReLU should throw nan_error if input contains NaN.
 */
TYPED_TEST(TypedRelu, relu_nan_input_throws)
{
    using value_t = TypeParam;

    if constexpr (! std::is_floating_point_v<value_t>)
        return;

    Tensor<value_t> input({3}, MemoryLocation::DEVICE);
    input = std::vector<value_t>{
        static_cast<value_t>(1.0),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(3.0)
    };

    EXPECT_THROW({
        nn::relu(input);
    }, temper::nan_error);
}

/**
 * @test TypedRelu.relu_alias_view_strided
 * @brief ReLU on a non-contiguous strided alias view.
 */
TYPED_TEST(TypedRelu, relu_alias_view_strided)
{
    using value_t = TypeParam;

    Tensor<value_t> owner({2, 4}, MemoryLocation::DEVICE);
    owner = std::vector<value_t>{
        static_cast<value_t>(-1.0), static_cast<value_t>(2.0),
        static_cast<value_t>(-3.0), static_cast<value_t>(4.0),
        static_cast<value_t>(-5.0), static_cast<value_t>(6.0),
        static_cast<value_t>(-7.0), static_cast<value_t>(8.0)
    };

    // Create a view with stride 2 along last axis:  selects columns 0, 2
    std::vector<uint64_t> start = {0, 0};
    std::vector<uint64_t> dims = {2, 2};
    std::vector<uint64_t> strides = {4, 2};
    Tensor<value_t> view(owner, start, dims, strides);

    Tensor<value_t> result = nn::relu(view);

    EXPECT_EQ(result.get_dimensions(), std::vector<uint64_t>({2, 2}));
    EXPECT_NEAR(static_cast<double>(result[0][0]), 0.0, 1e-6);  // relu(-1) = 0
    EXPECT_NEAR(static_cast<double>(result[0][1]), 0.0, 1e-6);  // relu(-3) = 0
    EXPECT_NEAR(static_cast<double>(result[1][0]), 0.0, 1e-6);  // relu(-5) = 0
    EXPECT_NEAR(static_cast<double>(result[1][1]), 0.0, 1e-6);  // relu(-7) = 0
}

/**
 * @test TypedRelu.relu_3d_tensor
 * @brief ReLU on a 3D tensor.
 */
TYPED_TEST(TypedRelu, relu_3d_tensor)
{
    using value_t = TypeParam;

    Tensor<value_t> input({2, 2, 2}, MemoryLocation::DEVICE);
    input = std::vector<value_t>{
        static_cast<value_t>(-1.0), static_cast<value_t>(2.0),
        static_cast<value_t>(3.0), static_cast<value_t>(-4.0),
        static_cast<value_t>(5.0), static_cast<value_t>(-6.0),
        static_cast<value_t>(-7.0), static_cast<value_t>(8.0)
    };

    Tensor<value_t> result = nn::relu(input);

    EXPECT_EQ(result.get_dimensions(), std::vector<uint64_t>({2, 2, 2}));
    EXPECT_NEAR(static_cast<double>(result[0][0][0]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[0][0][1]), 2.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[0][1][0]), 3.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[0][1][1]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1][0][0]), 5.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1][0][1]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1][1][0]), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(result[1][1][1]), 8.0, 1e-6);
}

} // namespace Test