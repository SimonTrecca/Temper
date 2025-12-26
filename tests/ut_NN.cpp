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

} // namespace Test