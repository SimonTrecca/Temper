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
 * @test ONEHOT.basic_axis_negative
 * @brief One-hot expand a chosen column in a 2x4 contiguous tensor
 * using negative indexing.
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
TEST(ONEHOT, basic_axis_negative)
{
    Tensor<float> t({2, 4}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 0.0f, 4.0f,
        5.0f, 6.0f, 2.0f, 8.0f
    };
    t = vals;

    Tensor<float> out = ml::one_hot_expand_at<float>(t, /*axis=*/-1,
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

/**
 * @test SOFTMAX.empty_tensor_throws
 * @brief Softmax on an empty/default tensor should throw std::invalid_argument.
 */
TEST(SOFTMAX, empty_tensor_throws)
{
    Tensor<float> t;
    EXPECT_THROW(ml::softmax<float>(t, /*axis=*/0), std::invalid_argument);
}

/**
 * @test SOFTMAX.axis_out_of_range_throws
 * @brief Passing axis >= rank should throw std::invalid_argument.
 */
TEST(SOFTMAX, axis_out_of_range_throws)
{
    Tensor<float> t({2,2}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(ml::softmax<float>(t, /*axis=*/2), std::invalid_argument);
}

/**
 * @test SOFTMAX.nan_input_throws
 * @brief NaN in input should produce a runtime_error (propagated from exp).
 */
TEST(SOFTMAX, nan_input_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f,
        std::numeric_limits<float>::quiet_NaN(), 2.0f};
    t = vals;
    EXPECT_THROW(ml::softmax<float>(t, /*axis=*/0), std::runtime_error);
}

/**
 * @test SOFTMAX.inf_input_throws
 * @brief Inf in input should produce a runtime_error (non-finite result from exp).
 */
TEST(SOFTMAX, inf_input_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, std::numeric_limits<float>::infinity()};
    t = vals;
    EXPECT_THROW(ml::softmax<float>(t, /*axis=*/0), std::runtime_error);
}

/**
 * @test SOFTMAX.softmax_basic_1d
 * @brief Basic softmax 1D.
 */
TEST(SOFTMAX, softmax_basic_1d)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    t = vals;

    Tensor<float> out = ml::softmax<float>(t, /*axis=*/0);

    ASSERT_EQ(out.get_dimensions(), t.get_dimensions());
    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 3u);

    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    const double e1 = 2.71828182845904523536;
    const double e2 = 7.38905609893065022723;
    const double e3 = 20.08553692318766774092;
    const double S  = e1 + e2 + e3;
    const double ref0 = e1 / S;
    const double ref1 = e2 / S;
    const double ref2 = e3 / S;

    EXPECT_NEAR(static_cast<double>(host[0]), ref0, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[1]), ref1, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[2]), ref2, 1e-6);
}

/**
 * @test SOFTMAX.softmax_2d_axis1
 * @brief Softmax per-row (axis=1).
 */
TEST(SOFTMAX, softmax_2d_axis1)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f,
        3.0f, 2.0f, 1.0f
    };
    t = vals;

    Tensor<float> out = ml::softmax<float>(t, /*axis=*/1);

    ASSERT_EQ(out.get_dimensions(), t.get_dimensions());
    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 6u);

    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    const double e1 = 2.71828182845904523536;
    const double e2 = 7.38905609893065022723;
    const double e3 = 20.08553692318766774092;
    const double S  = e1 + e2 + e3;

    EXPECT_NEAR(static_cast<double>(host[0]), e1 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[1]), e2 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[2]), e3 / S, 1e-6);

    EXPECT_NEAR(static_cast<double>(host[3]), e3 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[4]), e2 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[5]), e1 / S, 1e-6);
}

/**
 * @test SOFTMAX.softmax_2d_axis0
 * @brief Softmax along axis 0 (per-column) for identical rows -> 0.5 each.
 */
TEST(SOFTMAX, softmax_2d_axis0)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f
    };
    t = vals;

    Tensor<float> out = ml::softmax<float>(t, /*axis=*/0);

    ASSERT_EQ(out.get_dimensions(), t.get_dimensions());
    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 6u);

    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(), sizeof(float) * N).wait();

    for (uint64_t r = 0; r < 2; ++r)
    {
        for (uint64_t c = 0; c < 3; ++c)
        {
            const double got = static_cast<double>(host[r * 3 + c]);
            EXPECT_NEAR(got, 0.5, 1e-6);
        }
    }
}

/**
 * @test SOFTMAX.alias_view_weird_strides
 * @brief Softmax on an alias view with non-standard strides.
 */
TEST(SOFTMAX, alias_view_weird_strides)
{
    Tensor<float> owner({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> owner_vals = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    owner = owner_vals;

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims  = {2ull, 2ull};
    std::vector<uint64_t> strides = {3ull, 2ull};
    Tensor<float> alias(owner, start, dims, strides);

    Tensor<float> out = ml::softmax<float>(alias, /*axis=*/1);

    ASSERT_EQ(out.get_rank(), alias.get_rank());
    ASSERT_EQ(out.get_dimensions(), alias.get_dimensions());
    ASSERT_EQ(out.get_num_elements(), 4u);

    std::vector<float> host(4);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 4).wait();

    const double r0_c0 = 0.11920292202211755;
    const double r0_c1 = 0.88079707797788245;

    EXPECT_NEAR(static_cast<double>(host[0]), r0_c0, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[1]), r0_c1, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[2]), r0_c0, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[3]), r0_c1, 1e-6);
}

/**
 * @test SOFTMAX.softmax_negative_indexing
 * @brief Negative axis indexing: axis = -1 should behave like axis = rank-1.
 */
TEST(SOFTMAX, softmax_negative_indexing)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f,
        3.0f, 2.0f, 1.0f
    };
    t = vals;

    Tensor<float> out = ml::softmax<float>(t, /*axis=*/-1);

    ASSERT_EQ(out.get_dimensions(), t.get_dimensions());
    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 6u);

    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    const double e1 = 2.71828182845904523536;
    const double e2 = 7.38905609893065022723;
    const double e3 = 20.08553692318766774092;
    const double S  = e1 + e2 + e3;

    EXPECT_NEAR(static_cast<double>(host[0]), e1 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[1]), e2 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[2]), e3 / S, 1e-6);

    EXPECT_NEAR(static_cast<double>(host[3]), e3 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[4]), e2 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[5]), e1 / S, 1e-6);
}

/**
 * @test SOFTMAX.softmax_flatten
 * @brief Softmax with no axis (flatten)should compute softmax
 * over all elements.
 */
TEST(SOFTMAX, softmax_flatten)
{
    Tensor<float> t({2,2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        0.0f, 1.0f,
        2.0f, 3.0f
    };
    t = vals;

    Tensor<float> out = ml::softmax<float>(t);

    ASSERT_EQ(out.get_dimensions(), t.get_dimensions());
    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 4u);

    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    const double e0 = std::exp(0.0);
    const double e1 = std::exp(1.0);
    const double e2 = std::exp(2.0);
    const double e3 = std::exp(3.0);
    const double S  = e0 + e1 + e2 + e3;

    EXPECT_NEAR(static_cast<double>(host[0]), e0 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[1]), e1 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[2]), e2 / S, 1e-6);
    EXPECT_NEAR(static_cast<double>(host[3]), e3 / S, 1e-6);
}

/**
 * @test CROSS_ENTROPY.empty_logits_throws
 * @brief empty/default logits tensor should throw std::invalid_argument.
 */
TEST(CROSS_ENTROPY, empty_logits_throws)
{
    Tensor<float> logits;
    Tensor<float> labels({1,2}, MemoryLocation::DEVICE);
    labels = std::vector<float>{1.0f, 0.0f};

    EXPECT_THROW(ml::cross_entropy<float>(logits, labels),
        std::invalid_argument);
}

/**
 * @test CROSS_ENTROPY.empty_labels_throws
 * @brief empty/default labels tensor should throw std::invalid_argument.
 */
TEST(CROSS_ENTROPY, empty_labels_throws)
{
    Tensor<float> logits({1,2}, MemoryLocation::DEVICE);
    Tensor<float> labels;
    logits = std::vector<float>{1.0f, 0.0f};

    EXPECT_THROW(ml::cross_entropy<float>(logits, labels),
        std::invalid_argument);
}

/**
 * @test CROSS_ENTROPY.axis_out_of_range_throws
 * @brief axis >= rank_logits should throw std::invalid_argument.
 */
TEST(CROSS_ENTROPY, axis_out_of_range_throws)
{
    Tensor<float> logits({2,2}, MemoryLocation::DEVICE);
    logits = std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
    Tensor<float> labels({2,2}, MemoryLocation::DEVICE);
    labels = std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f};

    std::optional<int64_t> axis = 2;
    EXPECT_THROW(ml::cross_entropy<float>(logits, labels, axis),
        std::invalid_argument);
}

/**
 * @test CROSS_ENTROPY.broadcast_labels_mean
 * @brief Labels tensor has higher rank than logits: test aligning+broadcasting
 */
TEST(CROSS_ENTROPY, broadcast_labels_mean)
{
    Tensor<float> logits({2,2}, MemoryLocation::DEVICE);
    logits = std::vector<float>{0.0f, 0.0f,
                                0.0f, 0.0f};

    Tensor<float> labels({1,2,2}, MemoryLocation::DEVICE);
    labels = std::vector<float>{
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    std::optional<int64_t> axis = 2;
    Tensor<float> out = ml::cross_entropy<float>(logits, labels,
        axis, /*from_logits=*/true, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    const double expected = -std::log(0.5);
    EXPECT_NEAR(static_cast<double>(host[0]), expected, 1e-6);
}

/**
 * @test CROSS_ENTROPY.broadcast_labels_no_reduction
 * @brief Same broadcast case as above but with reduction_mean = false.
 */
TEST(CROSS_ENTROPY, broadcast_labels_no_reduction)
{
    Tensor<float> logits({2,2}, MemoryLocation::DEVICE);
    logits = std::vector<float>{0.0f, 0.0f,
                                0.0f, 0.0f};

    Tensor<float> labels({1,2,2}, MemoryLocation::DEVICE);
    labels = std::vector<float>{
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    std::optional<int64_t> axis = 2;
    Tensor<float> out = ml::cross_entropy<float>(logits, labels,
        axis, /*from_logits=*/true, /*reduction_mean=*/false);

    std::vector<uint64_t> expected_shape = {1ull, 2ull, 1ull};
    ASSERT_EQ(out.get_dimensions(), expected_shape);

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 2u);
    std::vector<float> host(N, -1.0f);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    const double expected = -std::log(0.5);
    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host[i]), expected, 1e-6);
    }
}

/**
 * @test CROSS_ENTROPY.from_probs_no_logits
 * @brief When from_logits=false, `logits` are treated as probabilities.
 */
TEST(CROSS_ENTROPY, from_probs_no_logits)
{
    Tensor<float> probs({2,2}, MemoryLocation::DEVICE);
    probs = std::vector<float>{
        0.9f, 0.1f,
        0.1f, 0.9f
    };

    Tensor<float> labels({2,2}, MemoryLocation::DEVICE);
    labels = std::vector<float>{
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    std::optional<int64_t> axis = 1;
    Tensor<float> out = ml::cross_entropy<float>(probs, labels, axis,
        /*from_logits=*/false, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    const double expected = -std::log(0.9);
    EXPECT_NEAR(static_cast<double>(host[0]), expected, 1e-6);
}

/**
 * @test CROSS_ENTROPY.negative_axis_equals_positive
 * @brief Negative axis indexing should be handled equivalent to rank-1 indexing.
 */
TEST(CROSS_ENTROPY, negative_axis_equals_positive)
{
    Tensor<float> logits({2,2}, MemoryLocation::DEVICE);
    logits = std::vector<float>{0.0f, 0.0f,
                                0.0f, 0.0f};

    Tensor<float> labels({1,2,2}, MemoryLocation::DEVICE);
    labels = std::vector<float>{
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    std::optional<int64_t> axis_neg = -1;
    std::optional<int64_t> axis_pos = 1;

    Tensor<float> out_neg = ml::cross_entropy<float>(logits, labels,
        axis_neg, /*from_logits=*/true, /*reduction_mean=*/true);
    Tensor<float> out_pos = ml::cross_entropy<float>(logits, labels,
        axis_pos, /*from_logits=*/true, /*reduction_mean=*/true);

    std::vector<float> host_neg(1), host_pos(1);
    g_sycl_queue.memcpy(host_neg.data(),
        out_neg.m_p_data.get(), sizeof(float) * 1).wait();
    g_sycl_queue.memcpy(host_pos.data(),
        out_pos.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host_neg[0]),
        static_cast<double>(host_pos[0]), 1e-7);
}

/**
 * @test CROSS_ENTROPY.alias_logits_weird_strides_mean
 * @brief logits provided as an alias view with non-standard strides.
 */
TEST(CROSS_ENTROPY, alias_logits_weird_strides_mean)
{
    Tensor<float> owner({2,3}, MemoryLocation::DEVICE);
    std::vector<float> owner_vals = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    owner = owner_vals;

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims  = {2ull, 2ull};
    std::vector<uint64_t> strides = {3ull, 2ull};
    Tensor<float> logits_alias(owner, start, dims, strides);

    Tensor<float> labels({2,2}, MemoryLocation::DEVICE);
    labels = std::vector<float>{
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    std::optional<int64_t> axis = 1;
    Tensor<float> out = ml::cross_entropy<float>(logits_alias, labels,
        axis, /*from_logits=*/true, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    const double e1 = std::exp(1.0), e3 = std::exp(3.0);
    const double loss0 = -std::log(e1 / (e1 + e3));

    const double e4 = std::exp(4.0), e6 = std::exp(6.0);
    const double loss1 = -std::log(e6 / (e4 + e6));

    const double expected_mean = 0.5 * (loss0 + loss1);

    EXPECT_NEAR(static_cast<double>(host[0]), expected_mean, 1e-6);
}


/**
 * @test CROSS_ENTROPY.alias_labels_weird_strides_mean
 * @brief labels provided as an alias view with non-standard strides.
 */
TEST(CROSS_ENTROPY, alias_labels_weird_strides_mean)
{
    Tensor<float> logits({2,2}, MemoryLocation::DEVICE);
    logits = std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f};

    Tensor<float> owner_labels({2,3}, MemoryLocation::DEVICE);

    owner_labels = std::vector<float>{
        1.0f, 0.0f, 9.0f,
        0.0f, 1.0f, 8.0f
    };

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims  = {2ull, 2ull};
    std::vector<uint64_t> strides = {3ull, 1ull};
    Tensor<float> labels_alias(owner_labels, start, dims, strides);

    std::optional<int64_t> axis = 1;
    Tensor<float> out = ml::cross_entropy<float>(logits, labels_alias,
        axis, /*from_logits=*/true, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(), sizeof(float) * 1).wait();

    const double expected = -std::log(0.5);
    EXPECT_NEAR(static_cast<double>(host[0]), expected, 1e-6);
}

/**
 * @test CROSS_ENTROPY.alias_both_weird_strides
 * @brief both logits and labels are alias views (non-contiguous); verifies
 * the function works regardless of underlying owner memory layout.
 *
 * Reuse the owners from previous cases to build a combined non-contiguous test.
 */
TEST(CROSS_ENTROPY, alias_both_weird_strides)
{
    Tensor<float> owner_log({2,3}, MemoryLocation::DEVICE);
    owner_log = std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    Tensor<float> logits_alias(owner_log, std::vector<uint64_t>{0,0},
        std::vector<uint64_t>{2,2}, std::vector<uint64_t>{3,2});

    Tensor<float> owner_lbl({2,4}, MemoryLocation::DEVICE);
    owner_lbl = std::vector<float>{
        1.0f, 9.0f, 0.0f, 7.0f,
        0.0f, 8.0f, 1.0f, 6.0f
    };
    Tensor<float> labels_alias(owner_lbl, std::vector<uint64_t>{0,0},
        std::vector<uint64_t>{2,2}, std::vector<uint64_t>{4,2});

    std::optional<int64_t> axis = 1;
    Tensor<float> out = ml::cross_entropy<float>
        (logits_alias, labels_alias,
        axis, /*from_logits=*/true, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    const double e1 = std::exp(1.0), e3 = std::exp(3.0);
    const double loss0 = -std::log(e1 / (e1 + e3));
    const double e4 = std::exp(4.0), e6 = std::exp(6.0);
    const double loss1 = -std::log(e6 / (e4 + e6));
    const double expected_mean = 0.5 * (loss0 + loss1);

    EXPECT_NEAR(static_cast<double>(host[0]), expected_mean, 1e-6);
}

/**
 * @test MSE.empty_predictions_throws
 * @brief empty/default predictions tensor should throw std::invalid_argument.
 */
TEST(MSE, empty_predictions_throws)
{
    Tensor<float> preds;
    Tensor<float> targets({1,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{1.0f, 1.0f};

    EXPECT_THROW(ml::mean_squared_error<float>
        (preds, targets, std::nullopt, true),
        std::invalid_argument);
}

/**
 * @test MSE.empty_targets_throws
 * @brief empty/default targets tensor should throw std::invalid_argument.
 */
TEST(MSE, empty_targets_throws)
{
    Tensor<float> preds({1,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{1.0f, 1.0f};
    Tensor<float> targets;

    EXPECT_THROW(ml::mean_squared_error<float>
        (preds, targets, std::nullopt, true),
        std::invalid_argument);
}

/**
 * @test MSE.flatten_mean_basic
 * @brief Flatten (no axis) + reduction_mean=true returns mean(...) of the
 * summed tensor. Current implementation: sum(sq) -> scalar, then mean(scalar)
 * which equals the scalar sum (14.0) for this input.
 */
TEST(MSE, flatten_mean_basic)
{
    Tensor<float> preds({2,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};

    Tensor<float> targets({2,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f};

    Tensor<float> out = ml::mean_squared_error<float>
        (preds, targets, std::nullopt, true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host[0]), 14.0, 1e-6);
}

/**
 * @test MSE.flatten_no_reduction
 * @brief Flatten (no axis) + reduction_mean=false returns
 * the sum of squared errors.
 */
TEST(MSE, flatten_no_reduction)
{
    Tensor<float> preds({2,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};

    Tensor<float> targets({2,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f};

    Tensor<float> out = ml::mean_squared_error<float>
        (preds, targets, std::nullopt, false);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host[0]), 14.0, 1e-6);
}

/**
 * @test MSE.axis_out_of_range_throws
 * @brief Passing axis >= max_rank should throw std::invalid_argument.
 */
TEST(MSE, axis_out_of_range_throws)
{
    Tensor<float> preds({2,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f};
    Tensor<float> targets({2,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f};

    std::optional<int64_t> axis = 2;
    EXPECT_THROW(ml::mean_squared_error<float>(preds, targets, axis, true),
        std::invalid_argument);
}

/**
 * @test MSE.broadcast_targets_mean
 * @brief targets have higher rank; check alignment + reduction_mean=true.
 */
TEST(MSE, broadcast_targets_mean)
{
    Tensor<float> preds({2,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f};

    Tensor<float> targets({1,2,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f};

    std::optional<int64_t> axis = 2;
    Tensor<float> out = ml::mean_squared_error<float>
        (preds, targets, axis, true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host[0]), 2.0, 1e-6);
}

/**
 * @test MSE.broadcast_targets_no_reduction
 * @brief Same broadcast case but with reduction_mean=false:
 * expect per-batch sums.
 */
TEST(MSE, broadcast_targets_no_reduction)
{
    Tensor<float> preds({2,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f};

    Tensor<float> targets({1,2,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f};

    std::optional<int64_t> axis = 2;
    Tensor<float> out = ml::mean_squared_error<float>
        (preds, targets, axis, false);

    std::vector<uint64_t> expected_shape = {1ull, 2ull, 1ull};
    ASSERT_EQ(out.get_dimensions(), expected_shape);

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 2u);
    std::vector<float> host(N, -1.0f);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host[i]), 2.0, 1e-6);
    }
}

/**
 * @test MSE.negative_axis_equals_positive
 * @brief Negative axis indexing should be handled equivalent to positive axis.
 */
TEST(MSE, negative_axis_equals_positive)
{
    Tensor<float> preds({2,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f};

    Tensor<float> targets({1,2,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f};

    std::optional<int64_t> axis_neg = -1;
    std::optional<int64_t> axis_pos = 1;

    Tensor<float> out_neg = ml::mean_squared_error<float>(preds, targets,
        axis_neg, /*reduction_mean=*/true);
    Tensor<float> out_pos = ml::mean_squared_error<float>(preds, targets,
        axis_pos, /*reduction_mean=*/true);

    std::vector<float> host_neg(1), host_pos(1);
    g_sycl_queue.memcpy(host_neg.data(),
        out_neg.m_p_data.get(), sizeof(float) * 1).wait();
    g_sycl_queue.memcpy(host_pos.data(),
        out_pos.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host_neg[0]),
        static_cast<double>(host_pos[0]), 1e-7);
}

/**
 * @test MSE.alias_preds_weird_strides_mean
 * @brief Predictions provided as an alias view with non-standard strides.
 */
TEST(MSE, alias_preds_weird_strides_mean)
{
    Tensor<float> owner({2,3}, MemoryLocation::DEVICE);
    owner = std::vector<float>{1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims  = {2ull, 2ull};
    std::vector<uint64_t> strides = {3ull, 2ull};
    Tensor<float> preds_alias(owner, start, dims, strides);

    Tensor<float> targets({2,2}, MemoryLocation::DEVICE);
    targets = std::vector<float>{1.0f, 1.0f,
                                 1.0f, 1.0f};

    std::optional<int64_t> axis = 1;
    Tensor<float> out = ml::mean_squared_error<float>(preds_alias, targets,
        axis, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host[0]), 19.0, 1e-6);
}

/**
 * @test MSE.alias_targets_weird_strides_mean
 * @brief Targets provided as an alias view with non-standard strides.
 */
TEST(MSE, alias_targets_weird_strides_mean)
{
    Tensor<float> preds({2,2}, MemoryLocation::DEVICE);
    preds = std::vector<float>{1.0f, 3.0f,
                               4.0f, 6.0f};

    Tensor<float> owner_labels({2,3}, MemoryLocation::DEVICE);
    owner_labels = std::vector<float>{
        1.0f, 9.0f, 0.0f,
        1.0f, 8.0f, 0.0f
    };

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims  = {2ull, 2ull};
    std::vector<uint64_t> strides = {3ull, 1ull};
    Tensor<float> targets_alias(owner_labels, start, dims, strides);

    std::optional<int64_t> axis = 1;
    Tensor<float> out = ml::mean_squared_error<float>(preds, targets_alias,
        axis, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host[0]), 24.5, 1e-6);
}

/**
 * @test MSE.alias_both_weird_strides
 * @brief Both preds and targets are alias views (non-contiguous); verifies
 * the function works regardless of underlying owner memory layout.
 */
TEST(MSE, alias_both_weird_strides)
{
    Tensor<float> owner_pred({2,3}, MemoryLocation::DEVICE);
    owner_pred = std::vector<float>{1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f};
    Tensor<float> preds_alias(owner_pred, std::vector<uint64_t>{0,0},
        std::vector<uint64_t>{2,2}, std::vector<uint64_t>{3,2});

    Tensor<float> owner_tgt({2,4}, MemoryLocation::DEVICE);
    owner_tgt = std::vector<float>{
        1.0f, 9.0f, 1.0f, 7.0f,
        1.0f, 8.0f, 1.0f, 6.0f
    };
    Tensor<float> targets_alias(owner_tgt, std::vector<uint64_t>{0,0},
        std::vector<uint64_t>{2,2}, std::vector<uint64_t>{4,2});

    std::optional<int64_t> axis = 1;
    Tensor<float> out = ml::mean_squared_error<float>
        (preds_alias, targets_alias,
        axis, /*reduction_mean=*/true);

    ASSERT_EQ(out.get_num_elements(), 1u);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 1).wait();

    EXPECT_NEAR(static_cast<double>(host[0]), 19.0, 1e-6);
}

} // namespace Test