/**
 * @file ut_ML.cpp
 * @brief Google Test suite for machine learning utilities.
 *
 * Contains unit tests ensuring correctness of functions implemented
 * in the ML module.
 */

#include <gtest/gtest.h>

#define private public
#define protected public
#include "temper/ML.hpp"
#undef private
#undef protected

using namespace temper;

namespace Test
{
/**
 * @test ONEHOT.non_integer_label_throws
 * @brief Non-integer label values at the selected axis index
 * should cause an error.
 */
TEST(ONEHOT, non_integer_label_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {10.0f, 1.5f, 2.0f};
    t = vals;

    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/0, /*axis_index=*/1,
		/*depth=*/3), std::runtime_error);
}

/**
 * @test ONEHOT.label_out_of_range_throws
 * @brief Integer label >= depth should produce an error.
 */
TEST(ONEHOT, label_out_of_range_throws)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 5.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    t = vals;

    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/1, /*axis_index=*/1,
		/*depth=*/3), std::runtime_error);
}

/**
 * @test ONEHOT.invalid_depth_throws
 * @brief depth == 0 should throw std::invalid_argument.
 */
TEST(ONEHOT, invalid_depth_throws)
{
    Tensor<float> t({2,2}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 0.0f, 2.0f, 1.0f};
    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/1, /*axis_index=*/1,
    	/*depth=*/0), std::invalid_argument);
}

/**
 * @test ONEHOT.empty_tensor_throws
 * @brief calling on an empty (default) tensor should throw std::invalid_argument.
 */
TEST(ONEHOT, empty_tensor_throws)
{
    Tensor<float> t;
    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/0, /*axis_index=*/0,
    	/*depth=*/2), std::invalid_argument);
}

/**
 * @test ONEHOT.axis_out_of_range_throws
 * @brief axis >= rank should throw std::invalid_argument.
 */
TEST(ONEHOT, axis_out_of_range_throws)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1,2,3,4,5,6};
    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/2,
    	/*axis_index=*/0, /*depth=*/2), std::invalid_argument);
}

/**
 * @test ONEHOT.axis_index_out_of_range_throws
 * @brief axis_index >= axis length should throw std::out_of_range.
 */
TEST(ONEHOT, axis_index_out_of_range_throws)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1,2,3,4,5,6};
    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/1, /*axis_index=*/3,
    	/*depth=*/2), std::out_of_range);
}

/**
 * @test ONEHOT.nan_label_throws
 * @brief NaN at the targeted axis index should produce std::runtime_error.
 */
TEST(ONEHOT, nan_label_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f,
    	std::numeric_limits<float>::quiet_NaN(), 2.0f};
    t = vals;
    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/0, /*axis_index=*/1,
    	/*depth=*/3), std::runtime_error);
}

/**
 * @test ONEHOT.negative_label_throws
 * @brief Negative integer labels should cause std::runtime_error.
 */
TEST(ONEHOT, negative_label_throws)
{
    Tensor<float> t({2,1}, MemoryLocation::DEVICE);
    // put -1 in the targeted axis element
    std::vector<float> vals = { -1.0f, 0.0f };
    t = vals;
    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/1, /*axis_index=*/0,
    	/*depth=*/3), std::runtime_error);
}

/**
 * @test ONEHOT.non_integer_label_throws_2d
 * @brief Non-integer label in a 2D tensor throws (redundant safety test).
 */
TEST(ONEHOT, non_integer_label_throws_2d)
{
    Tensor<float> t({2,2}, MemoryLocation::DEVICE);
    std::vector<float> vals = { 1.0f, 1.5f, 3.0f, 4.0f };
    t = vals;
    EXPECT_THROW(ml::one_hot_expand_at<float>(t, /*axis=*/1, /*axis_index=*/1,
    	/*depth=*/3), std::runtime_error);
}

/**
 * @test ONEHOT.basic_2d
 * @brief One-hot expand a chosen column in a 2x4 contiguous tensor.
 *
 * Input:
 *  [1, 2, 0, 4]
 *  [5, 6, 2, 8]
 *
 * axis=1, axis_index=2, depth=3
 * Output shape: {2, 6}
 * Expected rows:
 *  [1,2, 1,0,0, 4]
 *  [5,6, 0,0,1, 8]
 */
TEST(ONEHOT, basic_2d)
{
    Tensor<float> t({2, 4}, MemoryLocation::DEVICE);
    // row-major input
    std::vector<float> vals = {
        1.0f, 2.0f, 0.0f, 4.0f,
        5.0f, 6.0f, 2.0f, 8.0f
    };
    t = vals;

    Tensor<float> out = ml::one_hot_expand_at<float>(t, /*axis=*/1,
    	/*axis_index=*/2, /*depth=*/3, /*on_value=*/1.0f, /*off_value=*/0.0f);

    std::vector<uint64_t> expected_shape = {2ull, 6ull};
    EXPECT_EQ(out.get_dimensions().size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); ++i)
    {
        EXPECT_EQ(out.get_dimensions()[i], expected_shape[i]);
    }

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 12u);

    std::vector<float> host(N, -1.0f);
    g_sycl_queue.memcpy(host.data(),
    	out.m_p_data.get(), sizeof(float) * N).wait();

    std::vector<float> expected = {
        1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 4.0f,
        5.0f, 6.0f, 0.0f, 0.0f, 1.0f, 8.0f
    };

    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test ONEHOT.alias_view_1d
 * @brief One-hot expand on a 1D alias view (non-contiguous stride).
 *
 * Owner: 6 elements, alias view picks indices {0,2,4} via stride=2.
 * Alias values: {7, 1, 2}
 * axis=0, axis_index=1 (the middle element of the alias is treated as label)
 * depth=3 -> output length = (3-1)+3 = 5
 *
 * Expected alias output: [7, off, on, off, 2]  (off=0, on=1)
 */
TEST(ONEHOT, alias_view_1d)
{
    Tensor<float> owner({6}, MemoryLocation::DEVICE);
    std::vector<float> owner_vals = {7.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f};
    owner = owner_vals;

    std::vector<uint64_t> start = {0ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> v(owner, start, dims, strides);

    Tensor<float> out = ml::one_hot_expand_at<float>(v, /*axis=*/0,
    	/*axis_index=*/1, /*depth=*/3, /*on_value=*/1.0f, /*off_value=*/0.0f);

    ASSERT_EQ(out.get_num_elements(), 5u);
    std::vector<float> host(5, -1.0f);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(), sizeof(float) * 5).wait();

    std::vector<float> expected = {7.0f, 0.0f, 1.0f, 0.0f, 2.0f};
    for (uint64_t i = 0; i < 5; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test ONEHOT.custom_on_off_axis0_first_index
 * @brief Check one_hot_expand_at when axis=0 and axis_index=0 with custom
 * on/off values and verify the full output content.
 *
 * Input (2x2):
 *  row0 (labels): [1.0, 0.0]
 *  row1 (data)  : [2.5, 3.5]
 *
 * With axis=0, axis_index=0, depth=2, on=5, off=-1:
 *  - row0 is replaced by depth slots at positions 0..1 containing the
 *    one-hot encodings of the labels (placed at axis_index + label).
 *  - row1 (coord != axis_index) is shifted to output row (1-1)+depth = 2.
 *
 * Expected output (3x2):
 *  row0: [ -1,  5 ]   // label 0 -> sets row0,col1
 *  row1: [  5, -1 ]   // label 1 -> sets row1,col0
 *  row2: [2.5, 3.5]   // original row1 copied to row2
 */
TEST(ONEHOT, custom_on_off_axis0_first_index)
{
    Tensor<float> t({2, 2}, MemoryLocation::DEVICE);

    std::vector<float> vals = {
        1.0f, 0.0f, // labels along axis 0, index 0 (row 0)
        2.5f, 3.5f  // remaining row to be shifted/copied
    };
    t = vals;

    const uint64_t axis = 0;
    const uint64_t axis_index = 0;
    const uint64_t depth = 2;
    const float on_value  = 5.0f;
    const float off_value = -1.0f;

    Tensor<float> out = temper::ml::one_hot_expand_at<float>(
        t, axis, axis_index, depth, on_value, off_value);

    ASSERT_EQ(out.get_rank(), t.get_rank());
    ASSERT_EQ(out.get_dimensions()[0], (t.get_dimensions()[0] - 1) + depth);
    ASSERT_EQ(out.get_dimensions()[1], t.get_dimensions()[1]);

    const uint64_t N = out.get_num_elements();
    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
    	out.m_p_data.get(), sizeof(float) * N).wait();

    std::vector<float> expected = {
        -1.0f, 5.0f,
         5.0f,-1.0f,
         2.5f, 3.5f
    };
    ASSERT_EQ(N, expected.size());

    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test ONEHOT.depth_one_identity_like
 * @brief depth == 1 should not change axis length (D -> (D-1)+1 == D) and
 * should place on_value at axis_index + lbl (lbl must be 0).
 */
TEST(ONEHOT, depth_one_identity_like)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    // choose axis 1 index 1 depth 1; label must be 0 to place on at axis_index
    std::vector<float> vals = {
        7.0f, 0.0f, 9.0f,
        1.0f, 0.0f, 2.0f
    };
    t = vals;

    Tensor<float> out = ml::one_hot_expand_at<float>(t, /*axis=*/1,
    	/*axis_index=*/1, /*depth=*/1, /*on_value=*/5.5f, /*off_value=*/-2.2f);

    EXPECT_EQ(out.get_dimensions(), t.get_dimensions());

    std::vector<float> host(out.get_num_elements());
    g_sycl_queue.memcpy(host.data(),
    	out.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0 * 3 + 0], 7.0f);
    EXPECT_FLOAT_EQ(host[0 * 3 + 1], 5.5f);
    EXPECT_FLOAT_EQ(host[0 * 3 + 2], 9.0f);

    EXPECT_FLOAT_EQ(host[1 * 3 + 0], 1.0f);
    EXPECT_FLOAT_EQ(host[1 * 3 + 1], 5.5f);
    EXPECT_FLOAT_EQ(host[1 * 3 + 2], 2.0f);
}

/**
 * @test ONEHOT.onehot_3d_axis2_example
 * @brief Small 3D tensor and expansion along axis=2 (last axis)
 *
 * Input shape {1,1,3} values {7, 2(label), 9}
 * axis=2 axis_index=1 depth=3 on=11 off=0
 * Output shape {1,1,5} expected flattened along last axis:
 * [7, off, off, on, 9] i.e. [7, 0, 0, 11, 9]
 */
TEST(ONEHOT, onehot_3d_axis2_example)
{
    Tensor<float> t({1,1,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = { 7.0f, 2.0f, 9.0f };
    t = vals;

    Tensor<float> out = ml::one_hot_expand_at<float>(t, /*axis=*/2,
    	/*axis_index=*/1, /*depth=*/3, /*on_value=*/11.0f, /*off_value=*/0.0f);

    ASSERT_EQ(out.get_num_elements(), 5u);
    std::vector<float> host(5);
    g_sycl_queue.memcpy(host.data(),
    	out.m_p_data.get(), sizeof(float) * 5).wait();

    EXPECT_FLOAT_EQ(host[0], 7.0f);
    EXPECT_FLOAT_EQ(host[1], 0.0f);
    EXPECT_FLOAT_EQ(host[2], 0.0f);
    EXPECT_FLOAT_EQ(host[3], 11.0f);
    EXPECT_FLOAT_EQ(host[4], 9.0f);
}

/**
 * @test ONEHOT.alias_view_2d_column_axis
 * @brief One-hot expand where labels are in a column in an alias (strided)
 * view. Verifies content (not just shape) including custom on/off values.
 */
TEST(ONEHOT, alias_view_2d_column_axis)
{
    Tensor<float> owner({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> owner_vals = {
        1.0f, 0.0f, 9.0f,
        2.0f, 1.0f, 8.0f
    };
    owner = owner_vals;

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims  = {2ull, 2ull};
    std::vector<uint64_t> strides = {3ull, 1ull};
    Tensor<float> alias2(owner, start, dims, strides);

    const uint64_t axis = 1;
    const uint64_t axis_index = 1;
    const uint64_t depth = 2;
    const float on_value  = 7.0f;
    const float off_value = -2.0f;

    Tensor<float> out = temper::ml::one_hot_expand_at<float>(
        alias2, axis, axis_index, depth, on_value, off_value);

    ASSERT_EQ(out.get_rank(), alias2.get_rank());
    ASSERT_EQ(out.get_dimensions()[0], alias2.get_dimensions()[0]);
    ASSERT_EQ(out.get_dimensions()[1], (alias2.get_dimensions()[1] - 1) + depth);

    const uint64_t N = out.get_num_elements();
    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
    	out.m_p_data.get(), sizeof(float) * N).wait();

    std::vector<float> expected = {
         1.0f,  7.0f, -2.0f,
         2.0f, -2.0f,  7.0f
    };
    ASSERT_EQ(N, expected.size());

    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

} // namespace Test