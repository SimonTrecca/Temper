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

template<typename T>
class TypedOnehot : public ::testing::Test {};

using OnehotTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedOnehot, OnehotTestTypes);

/**
 * @test TypedOnehot.non_integer_label_throws
 * @brief Non-integer label values at the selected axis index should error.
 */
TYPED_TEST(TypedOnehot, non_integer_label_throws)
{
    using value_t = TypeParam;

    // only floating types can represent non-integer labels like 1.5
    if constexpr (!std::is_floating_point_v<value_t>)
        return;

    Tensor<value_t> t({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(10.0), static_cast<value_t>(1.5),
        static_cast<value_t>(2.0)
    };
    t = vals;

    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/0, /*axis_index=*/1, /*depth=*/3),
        std::runtime_error);
}

/**
 * @test TypedOnehot.label_out_of_range_throws
 * @brief Integer label >= depth should produce an error.
 */
TYPED_TEST(TypedOnehot, label_out_of_range_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(5),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/1, /*axis_index=*/1, /*depth=*/3),
        std::runtime_error);
}

/**
 * @test TypedOnehot.invalid_depth_throws
 * @brief depth == 0 should throw std::invalid_argument.
 */
TYPED_TEST(TypedOnehot, invalid_depth_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::DEVICE);
    t = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(2), static_cast<value_t>(1)
    };

    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/1, /*axis_index=*/1, /*depth=*/0),
        std::invalid_argument);
}

/**
 * @test TypedOnehot.empty_tensor_throws
 * @brief calling on an empty (default) tensor should throw.
 */
TYPED_TEST(TypedOnehot, empty_tensor_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/0, /*axis_index=*/0, /*depth=*/2),
        std::invalid_argument);
}

/**
 * @test TypedOnehot.axis_out_of_range_throws
 * @brief axis >= rank should throw std::invalid_argument.
 */
TYPED_TEST(TypedOnehot, axis_out_of_range_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/2, /*axis_index=*/0, /*depth=*/2),
        std::invalid_argument);
}

/**
 * @test TypedOnehot.axis_index_out_of_range_throws
 * @brief axis_index >= axis length should throw std::out_of_range.
 */
TYPED_TEST(TypedOnehot, axis_index_out_of_range_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/1, /*axis_index=*/3, /*depth=*/2),
        std::out_of_range);
}

/**
 * @test TypedOnehot.nan_label_throws
 * @brief NaN at the targeted axis index should produce runtime_error.
 */
TYPED_TEST(TypedOnehot, nan_label_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point_v<value_t>)
    {
        return;
    }

    Tensor<value_t> t({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(2)
    };
    t = vals;

    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/0, /*axis_index=*/1, /*depth=*/3),
        std::runtime_error);
}

/**
 * @test TypedOnehot.negative_label_throws
 * @brief Negative integer labels should cause std::runtime_error.
 */
TYPED_TEST(TypedOnehot, negative_label_throws)
{
    using value_t = TypeParam;

    // unsigned types cannot represent negative labels meaningfully
    if constexpr (!std::is_signed_v<value_t>)
    {
        return;
    }

    Tensor<value_t> t({2,1}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(-1), static_cast<value_t>(0)
    };
    t = vals;

    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/1, /*axis_index=*/0, /*depth=*/3),
        std::runtime_error);
}

/**
 * @test TypedOnehot.non_integer_label_throws_2d
 * @brief Non-integer label in a 2D tensor throws (redundant check).
 */
TYPED_TEST(TypedOnehot, non_integer_label_throws_2d)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point_v<value_t>)
    {
        return;
    }

    Tensor<value_t> t({2,2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0), static_cast<value_t>(1.5),
        static_cast<value_t>(3.0), static_cast<value_t>(4.0)
    };
    t = vals;

    EXPECT_THROW(ml::one_hot_expand_at<value_t>(
        t, /*axis=*/1, /*axis_index=*/1, /*depth=*/3),
        std::runtime_error);
}

/**
 * @test TypedOnehot.basic_2d
 * @brief One-hot expand a chosen column in a 2x4 contiguous tensor.
 */
TYPED_TEST(TypedOnehot, basic_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 4}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(2), static_cast<value_t>(8)
    };
    t = vals;

    Tensor<value_t> out = ml::one_hot_expand_at<value_t>(
        t, /*axis=*/1, /*axis_index=*/2, /*depth=*/3,
        static_cast<value_t>(1), static_cast<value_t>(0));

    std::vector<uint64_t> expected_shape = {2ull, 6ull};
    EXPECT_EQ(out.get_dimensions().size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); ++i)
        EXPECT_EQ(out.get_dimensions()[i], expected_shape[i]);

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 12u);

    std::vector<value_t> host(N);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * N).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(1), static_cast<value_t>(8)
    };
    for (uint64_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        }
        else
        {
            EXPECT_EQ(host[i], expected[i]);
        }
    }
}

/**
 * @test TypedOnehot.basic_axis_negative
 * @brief One-hot expand a chosen column using negative indexing.
 */
TYPED_TEST(TypedOnehot, basic_axis_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 4}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(2), static_cast<value_t>(8)
    };
    t = vals;

    Tensor<value_t> out = ml::one_hot_expand_at<value_t>(
        t, /*axis=*/-1, /*axis_index=*/2, /*depth=*/3,
        static_cast<value_t>(1), static_cast<value_t>(0));

    std::vector<uint64_t> expected_shape = {2ull, 6ull};
    EXPECT_EQ(out.get_dimensions().size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); ++i)
        EXPECT_EQ(out.get_dimensions()[i], expected_shape[i]);

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 12u);

    std::vector<value_t> host(N);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * N).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(1), static_cast<value_t>(8)
    };
    for (uint64_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        }
        else
        {
            EXPECT_EQ(host[i], expected[i]);
        }
    }
}

/**
 * @test TypedOnehot.alias_view_1d
 * @brief One-hot expand on a 1D alias view (non-contiguous stride).
 */
TYPED_TEST(TypedOnehot, alias_view_1d)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({6}, MemoryLocation::DEVICE);

    std::vector<value_t> owner_vals = {
        static_cast<value_t>(7), static_cast<value_t>(0),
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(2), static_cast<value_t>(0)
    };
    owner = owner_vals;

    std::vector<uint64_t> start = {0ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<value_t> v(owner, start, dims, strides);

    Tensor<value_t> out = ml::one_hot_expand_at<value_t>(
        v, /*axis=*/0, /*axis_index=*/1, /*depth=*/3,
        static_cast<value_t>(1), static_cast<value_t>(0));

    ASSERT_EQ(out.get_num_elements(), 5u);
    std::vector<value_t> host(5);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * 5).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(7), static_cast<value_t>(0),
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(2)
    };
    for (uint64_t i = 0; i < 5; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        }
        else
        {
            EXPECT_EQ(host[i], expected[i]);
        }
    }
}

/**
 * @test TypedOnehot.custom_on_off_axis0_first_index
 * @brief one_hot_expand_at when axis=0 and axis_index=0 with custom vals.
 */
TYPED_TEST(TypedOnehot, custom_on_off_axis0_first_index)
{
    using value_t = TypeParam;

    // negative off_value does not make sense for unsigned value_t
    if constexpr (!std::is_signed_v<value_t>)
    {
        return;
    }

    Tensor<value_t> t({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(2.5), static_cast<value_t>(3.5)
    };

    t = vals;

    const uint64_t axis = 0;
    const uint64_t axis_index = 0;
    const uint64_t depth = 2;

    value_t on_value;
    value_t off_value;
    if constexpr (std::is_floating_point_v<value_t>)
    {
        on_value = static_cast<value_t>(5.0);
        off_value = static_cast<value_t>(-1.0);
    }
    else
    {
        on_value = static_cast<value_t>(5);
        off_value = static_cast<value_t>(0);
    }

    Tensor<value_t> out = ml::one_hot_expand_at<value_t>(
        t, axis, axis_index, depth, on_value, off_value);

    ASSERT_EQ(out.get_rank(), t.get_rank());
    ASSERT_EQ(out.get_dimensions()[0],
              (t.get_dimensions()[0] - 1) + depth);
    ASSERT_EQ(out.get_dimensions()[1], t.get_dimensions()[1]);

    const uint64_t N = out.get_num_elements();
    std::vector<value_t> host(N);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * N).wait();

    std::vector<value_t> expected;
    if constexpr (std::is_floating_point_v<value_t>)
    {
        expected = {
            static_cast<value_t>(-1.0), static_cast<value_t>(5.0),
            static_cast<value_t>(5.0),  static_cast<value_t>(-1.0),
            static_cast<value_t>(2.5),  static_cast<value_t>(3.5)
        };
        for (uint64_t i = 0; i < N; ++i)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        }
    }
    else
    {
        expected = {
            static_cast<value_t>(0), static_cast<value_t>(5),
            static_cast<value_t>(5), static_cast<value_t>(0),
            static_cast<value_t>(2), static_cast<value_t>(3)
        };
        for (uint64_t i = 0; i < N; ++i)
        {
            EXPECT_EQ(host[i], expected[i]);
        }
    }
}

/**
 * @test TypedOnehot.depth_one_identity_like
 * @brief depth == 1 should not change axis length and place on_value.
 */
TYPED_TEST(TypedOnehot, depth_one_identity_like)
{
    using value_t = TypeParam;

    // test uses fractional on/off; only floating types are meaningful here
    if constexpr (!std::is_floating_point_v<value_t>)
    {
        return;
    }

    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(7.0), static_cast<value_t>(0.0),
        static_cast<value_t>(9.0), static_cast<value_t>(1.0),
        static_cast<value_t>(0.0), static_cast<value_t>(2.0)
    };
    t = vals;

    Tensor<value_t> out = ml::one_hot_expand_at<value_t>(
        t, /*axis=*/1, /*axis_index=*/1, /*depth=*/1,
        static_cast<value_t>(5.5), static_cast<value_t>(-2.2));

    EXPECT_EQ(out.get_dimensions(), t.get_dimensions());

    std::vector<value_t> host(out.get_num_elements());
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0 * 3 + 0], static_cast<value_t>(7.0));
    EXPECT_FLOAT_EQ(host[0 * 3 + 1], static_cast<value_t>(5.5));
    EXPECT_FLOAT_EQ(host[0 * 3 + 2], static_cast<value_t>(9.0));
    EXPECT_FLOAT_EQ(host[1 * 3 + 0], static_cast<value_t>(1.0));
    EXPECT_FLOAT_EQ(host[1 * 3 + 1], static_cast<value_t>(5.5));
    EXPECT_FLOAT_EQ(host[1 * 3 + 2], static_cast<value_t>(2.0));
}

/**
 * @test TypedOnehot.onehot_3d_axis2_example
 * @brief Small 3D tensor and expansion along axis=2 (last axis).
 */
TYPED_TEST(TypedOnehot, onehot_3d_axis2_example)
{
    using value_t = TypeParam;
    Tensor<value_t> t({1,1,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(7), static_cast<value_t>(2),
        static_cast<value_t>(9)
    };
    t = vals;

    Tensor<value_t> out = ml::one_hot_expand_at<value_t>(
        t, /*axis=*/2, /*axis_index=*/1, /*depth=*/3,
        static_cast<value_t>(11), static_cast<value_t>(0));

    ASSERT_EQ(out.get_num_elements(), 5u);
    std::vector<value_t> host(5);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * 5).wait();

    if constexpr (std::is_floating_point_v<value_t>)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(7.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(0.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(0.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(11.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[4]),
                        static_cast<double>(9.0));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(7));
        EXPECT_EQ(host[1], static_cast<value_t>(0));
        EXPECT_EQ(host[2], static_cast<value_t>(0));
        EXPECT_EQ(host[3], static_cast<value_t>(11));
        EXPECT_EQ(host[4], static_cast<value_t>(9));
    }
}

/**
 * @test TypedOnehot.alias_view_2d_column_axis
 * @brief One-hot expand where labels are in a column in an alias (strided)
 * view. Verifies content including custom on/off values.
 */
TYPED_TEST(TypedOnehot, alias_view_2d_column_axis)
{
    using value_t = TypeParam;

    // negative off_value not representable for unsigned types
    if constexpr (!std::is_signed_v<value_t>)
        return;

    Tensor<value_t> owner({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> owner_vals = {
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(9), static_cast<value_t>(2),
        static_cast<value_t>(1), static_cast<value_t>(8)
    };
    owner = owner_vals;

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims  = {2ull, 2ull};
    std::vector<uint64_t> strides = {3ull, 1ull};
    Tensor<value_t> alias2(owner, start, dims, strides);

    const uint64_t axis = 1;
    const uint64_t axis_index = 1;
    const uint64_t depth = 2;

    value_t on_value = static_cast<value_t>(7);
    value_t off_value = static_cast<value_t>(-2);

    Tensor<value_t> out = ml::one_hot_expand_at<value_t>(
        alias2, axis, axis_index, depth, on_value, off_value);

    ASSERT_EQ(out.get_rank(), alias2.get_rank());
    ASSERT_EQ(out.get_dimensions()[0], alias2.get_dimensions()[0]);
    ASSERT_EQ(out.get_dimensions()[1],
              (alias2.get_dimensions()[1] - 1) + depth);

    const uint64_t N = out.get_num_elements();
    std::vector<value_t> host(N);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * N).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(1), static_cast<value_t>(7),
        static_cast<value_t>(-2), static_cast<value_t>(2),
        static_cast<value_t>(-2), static_cast<value_t>(7)
    };
    ASSERT_EQ(N, expected.size());

    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                        static_cast<double>(expected[i]));
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
            EXPECT_NEAR(got, 0.5, 1e-4);
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

    EXPECT_NEAR(static_cast<double>(host[0]), r0_c0, 1e-4);
    EXPECT_NEAR(static_cast<double>(host[1]), r0_c1, 1e-4);
    EXPECT_NEAR(static_cast<double>(host[2]), r0_c0, 1e-4);
    EXPECT_NEAR(static_cast<double>(host[3]), r0_c1, 1e-4);
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
    EXPECT_NEAR(static_cast<double>(host[0]), expected, 1e-4);
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
        EXPECT_NEAR(static_cast<double>(host[i]), expected, 1e-4);
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

    EXPECT_NEAR(static_cast<double>(host[0]), expected_mean, 1e-4);
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
    EXPECT_NEAR(static_cast<double>(host[0]), expected, 1e-4);
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

    EXPECT_NEAR(static_cast<double>(host[0]), expected_mean, 1e-4);
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

/**
 * @test PCA.pca_basic
 * @brief Basic PCA on a 3×3 matrix with standardization; checks eigenvalues,
 * loadings, and projections.
 */
TEST(PCA, pca_basic)
{
    Tensor<float> t({3, 3});
    t = {
       1,  2,  3,
       4,  5,  6,
       2,  3,  2,
    };

    ml::PCAResult result = ml::pca(t, std::nullopt, true);

    Tensor<float> expected_expv({1, 3});
    expected_expv =
    {
        2.78708718f, 0.21291282f, 7.03016493e-16f,
    };

    Tensor<float> expected_loadings({3, 3});
    expected_loadings =
    {
        -0.58916765f, -0.39100062f, -0.70710678f,
        -0.58916765f, -0.39100062f,  0.70710678f,
        -0.55295837f,  0.83320888f, -1.22124533e-15f,
    };

    Tensor<float> expected_projections({3, 3});
    expected_projections =
    {
         1.20562378f,  0.41574624f, -7.20766725e-16f,
        -1.90547864f,  0.08070810f, -4.94177148e-17f,
         0.69985486f, -0.49645434f,  7.24197496e-16f,
    };

    const double tol = 5e-3;

    // Check explained variance.
    {
        auto it_a = result.explained_variance.begin();
        auto it_b = expected_expv.begin();
        for (; it_a != result.explained_variance.end() &&
            it_b != expected_expv.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(*it_a, *it_b, tol);
        }
    }

    // Check loadings.
    {
        auto it_a = result.loadings.begin();
        auto it_b = expected_loadings.begin();
        for (; it_a != result.loadings.end() &&
            it_b != expected_loadings.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }

    // Check projections.
    {
        auto it_a = result.projections.begin();
        auto it_b = expected_projections.begin();
        for (; it_a != result.projections.end() &&
            it_b != expected_projections.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }
}

/**
 * @test PCA.pca_no_standardize_basic
 * @brief PCA without standardization on a 3×2 matrix; verifies unscaled
 * covariance, loadings, and projections.
 */

TEST(PCA, pca_no_standardize_basic)
{
    Tensor<float> t({3, 2});
    t = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    };

    ml::PCAResult result = ml::pca(t, std::nullopt, /*standardize=*/false);

    Tensor<float> expected_expv({1, 2});
    expected_expv = {
        8.f, 0.f
    };

    Tensor<float> expected_loadings({2, 2});
    expected_loadings = {
         0.70710678f,  0.70710678f,
         0.70710678f, -0.70710678f
    };

    Tensor<float> expected_proj({3, 2});
    expected_proj = {
        -2.82842712f,  0.f,
         0.f,          0.f,
         2.82842712f,  0.f
    };

    const double tol = 5e-3;

    // Check explained variance.
    {
        auto a = result.explained_variance.begin();
        auto b = expected_expv.begin();
        for (; a != result.explained_variance.end() &&
            b != expected_expv.end(); ++a, ++b)
        {
            EXPECT_NEAR(*a, *b, tol);
        }
    }

    // Check loadings.
    {
        auto a = result.loadings.begin();
        auto b = expected_loadings.begin();
        for (; a != result.loadings.end() &&
            b != expected_loadings.end(); ++a, ++b)
        {
            EXPECT_NEAR(std::fabs(*a), std::fabs(*b), tol);
        }
    }

    // Check projections.
    {
        auto a = result.projections.begin();
        auto b = expected_proj.begin();
        for (; a != result.projections.end() &&
            b != expected_proj.end(); ++a, ++b)
        {
            EXPECT_NEAR(std::fabs(*a), std::fabs(*b), tol);
        }
    }
}

/**
 * @test PCA.pca_basic_batched
 * @brief Batched PCA on a 3×3 dataset replicated over batch dimension;
 * validates per-batch eigenvalues, loadings, and projections.
 */
TEST(PCA, pca_basic_batched)
{
    Tensor<float> t({2, 3, 3});
    t = {
       1,  2,  3,
       4,  5,  6,
       2,  3,  2,

       1,  3,  5,
       7,  9, 45,
       2,  4, 10
    };

    ml::PCAResult result = ml::pca(t, std::nullopt, true);

    Tensor<float> expected_expv({2, 1, 3});
    expected_expv =
    {
        2.78708718f, 0.21291282f, 7.03016493e-16f,
        2.99886771f, 0.00113229f, 7.05596321e-16f
    };

    Tensor<float> expected_loadings({2, 3, 3});
    expected_loadings =
    {
        -0.58916765f, -0.39100062f, -0.70710678f,
        -0.58916765f, -0.39100062f,  0.70710678f,
        -0.55295837f,  0.83320888f, -1.22124533e-15f,

        -0.57740479f, -0.40817118f, -0.70710678f,
        -0.57740479f, -0.40817118f,  0.70710678f,
        -0.57724122f,  0.81657368f, -1.33559830e-13f
    };

    Tensor<float> expected_projections({2, 3, 3});
    expected_projections =
    {
         1.20562378f,  0.41574624f, -7.20766725e-16f,
        -1.90547864f,  0.08070810f, -4.94177148e-17f,
         0.69985486f, -0.49645434f,  7.24197496e-16f,

         1.23552187f,  0.03055077f, -5.14396459e-15f,
        -1.97937101f,  0.00551589f, -3.04875468e-16f,
         0.74384913f, -0.03606666f,  5.87388085e-15f
    };

    const double tol = 5e-3;

    // Check explained variance.
    {
        auto it_a = result.explained_variance.begin();
        auto it_b = expected_expv.begin();
        for (; it_a != result.explained_variance.end() &&
            it_b != expected_expv.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(*it_a, *it_b, tol);
        }
    }

    // Check loadings.
    {
        auto it_a = result.loadings.begin();
        auto it_b = expected_loadings.begin();
        for (; it_a != result.loadings.end() &&
            it_b != expected_loadings.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }

    // Check projections.
    {
        auto it_a = result.projections.begin();
        auto it_b = expected_projections.begin();
        for (; it_a != result.projections.end() &&
            it_b != expected_projections.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }
}

/**
 * @test PCA.pca_4d
 * @brief PCA on a 4D tensor where the last two dims form square matrices;
 * checks batched PCA across two leading dimensions.
 */
TEST(PCA, pca_4d)
{
    Tensor<float> t({2, 2, 3, 3});
    t =
    {
        1,  2,  3,
        4,  5,  6,
        2,  3,  2,

        1,  3,  5,
        7,  9, 45,
        2,  4, 10,

        1,  2,  3,
        4,  5,  6,
        2,  3,  2,

        1,  3,  5,
        7,  9, 45,
        2,  4, 10
    };

    ml::PCAResult result = ml::pca(t, std::nullopt, true);

    Tensor<float> expected_expv({2, 2, 1, 3});
    expected_expv =
    {
        2.78708718f, 0.21291282f, 6.85401860e-16f,
        2.99886771f, 0.00113229f, 5.83854862e-16f,
        2.78708718f, 0.21291282f, 6.85401860e-16f,
        2.99886771f, 0.00113229f, 5.83854862e-16f
    };

    Tensor<float> expected_loadings({2, 2, 3, 3});
    expected_loadings =
    {
        -0.58916765f, -0.39100062f, -0.70710678f,
        -0.58916765f, -0.39100062f,  0.70710678f,
        -0.55295837f,  0.83320888f,  5.61242657e-17f,

        -0.57740479f, -0.40817118f, -0.70710678f,
        -0.57740479f, -0.40817118f,  0.70710678f,
        -0.57724122f,  0.81657368f, -3.15465886e-17f,

        -0.58916765f, -0.39100062f, -0.70710678f,
        -0.58916765f, -0.39100062f,  0.70710678f,
        -0.55295837f,  0.83320888f,  5.61242657e-17f,

        -0.57740479f, -0.40817118f, -0.70710678f,
        -0.57740479f, -0.40817118f,  0.70710678f,
        -0.57724122f,  0.81657368f, -3.15465886e-17f
    };

    Tensor<float> expected_projections({2, 2, 3, 3});
    expected_projections =
    {
         1.20562378f,  0.41574624f,  5.33591709e-16f,
        -1.90547864f,  0.08070810f, -6.96923002e-16f,
         0.69985486f, -0.49645434f,  8.95887736e-17f,

         1.23552187f,  0.03055077f, -5.14148336e-16f,
        -1.97937101f,  0.00551589f, -3.56189375e-16f,
         0.74384913f, -0.03606666f,  5.70337711e-16f,

         1.20562378f,  0.41574624f,  5.33591709e-16f,
        -1.90547864f,  0.08070810f, -6.96923002e-16f,
         0.69985486f, -0.49645434f,  8.95887736e-17f,

         1.23552187f,  0.03055077f, -5.14148336e-16f,
        -1.97937101f,  0.00551589f, -3.56189375e-16f,
         0.74384913f, -0.03606666f,  5.70337711e-16f
    };

    const double tol = 5e-3;

    // Check explained variance.
    {
        auto it_a = result.explained_variance.begin();
        auto it_b = expected_expv.begin();
        for (; it_a != result.explained_variance.end() &&
            it_b != expected_expv.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(*it_a, *it_b, tol);
        }
    }

    // Check loadings.
    {
        auto it_a = result.loadings.begin();
        auto it_b = expected_loadings.begin();
        for (; it_a != result.loadings.end() &&
            it_b != expected_loadings.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }

    // Check projections.
    {
        auto it_a = result.projections.begin();
        auto it_b = expected_projections.begin();
        for (; it_a != result.projections.end() &&
            it_b != expected_projections.end(); ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }
}

/**
 * @test PCA.pca_4d_k2
 * @brief PCA on 4D input with n_components=2; verifies truncated eigenvalues,
 * loadings, and projections.
 */

TEST(PCA, pca_4d_k2)
{
    Tensor<float> t({2, 2, 3, 3});
    t = {

        1, 2, 3,
        4, 5, 6,
        2, 3, 2,


        1, 3, 5,
        7, 9, 45,
        2, 4, 10,


        1, 2, 3,
        4, 5, 6,
        2, 3, 2,


        1, 3, 5,
        7, 9, 45,
        2, 4, 10
    };

    ml::PCAResult result = ml::pca(t, 2, true);

    Tensor<float> expected_expv({2, 2, 1, 2});
    expected_expv =
    {
        2.78708718f, 0.21291282f,

        2.99886771f, 0.00113229f,

        2.78708718f, 0.21291282f,

        2.99886771f, 0.00113229f
    };

    Tensor<float> expected_loadings({2, 2, 3, 2});
    expected_loadings =
    {
        -0.58916765f, -0.39100062f,
        -0.58916765f, -0.39100062f,
        -0.55295837f,  0.83320888f,

        -0.57740479f, -0.40817118f,
        -0.57740479f, -0.40817118f,
        -0.57724122f,  0.81657368f,

        -0.58916765f, -0.39100062f,
        -0.58916765f, -0.39100062f,
        -0.55295837f,  0.83320888f,

        -0.57740479f, -0.40817118f,
        -0.57740479f, -0.40817118f,
        -0.57724122f,  0.81657368f
    };

    Tensor<float> expected_proj({2, 2, 3, 2});
    expected_proj =
    {
         1.20562378f,  0.41574624f,
        -1.90547864f,  0.08070810f,
         0.69985486f, -0.49645434f,

         1.23552187f,  0.03055077f,
        -1.97937101f,  0.00551589f,
         0.74384913f, -0.03606666f,

         1.20562378f,  0.41574624f,
        -1.90547864f,  0.08070810f,
         0.69985486f, -0.49645434f,

         1.23552187f,  0.03055077f,
        -1.97937101f,  0.00551589f,
         0.74384913f, -0.03606666f
    };

    const double tol = 5e-3;

    // Check explained variance.
    {
        auto a = result.explained_variance.begin();
        auto b = expected_expv.begin();
        for (; a != result.explained_variance.end() &&
               b != expected_expv.end(); ++a, ++b)
        {
            EXPECT_NEAR(*a, *b, tol);
        }
    }

    // Check loadings.
    {
        auto a = result.loadings.begin();
        auto b = expected_loadings.begin();
        for (; a != result.loadings.end() &&
               b != expected_loadings.end(); ++a, ++b)
        {
            EXPECT_NEAR(std::fabs(*a), std::fabs(*b), tol);
        }
    }

    // Check projections.
    {
        auto a = result.projections.begin();
        auto b = expected_proj.begin();
        for (; a != result.projections.end() &&
               b != expected_proj.end(); ++a, ++b)
        {
            EXPECT_NEAR(std::fabs(*a), std::fabs(*b), tol);
        }
    }
}

/**
 * @test PCA.pca_4d_alias_view_strided
 * @brief PCA on a non-contiguous strided alias view of a 4D tensor; ensures
 * batching and stride traversal work correctly.
 */

TEST(PCA, pca_4d_alias_view_strided)
{
    const std::vector<uint64_t> owner_dims = {2, 2, 3, 3};
    const uint64_t b0 = owner_dims[0];
    const uint64_t b1 = owner_dims[1];
    const uint64_t rows = owner_dims[2];
    const uint64_t cols = owner_dims[3];
    ASSERT_EQ(rows, cols);
    const uint64_t elems_per_batch = rows * cols;
    const uint64_t owner_total = b0 * b1 * elems_per_batch;

    std::vector<float> owner_flat(owner_total, 0.0f);

    {
        const uint64_t batch_base = (0 * owner_dims[1] + 0) * elems_per_batch;
        owner_flat[batch_base + 0 * cols + 0] = 1.0f;
        owner_flat[batch_base + 0 * cols + 1] = 2.0f;
        owner_flat[batch_base + 0 * cols + 2] = 3.0f;
        owner_flat[batch_base + 1 * cols + 0] = 4.0f;
        owner_flat[batch_base + 1 * cols + 1] = 5.0f;
        owner_flat[batch_base + 1 * cols + 2] = 6.0f;
        owner_flat[batch_base + 2 * cols + 0] = 2.0f;
        owner_flat[batch_base + 2 * cols + 1] = 3.0f;
        owner_flat[batch_base + 2 * cols + 2] = 2.0f;
    }

    {
        const uint64_t batch_base = (1 * owner_dims[1] + 0) * elems_per_batch;
        owner_flat[batch_base + 0 * cols + 0] = 1.0f;
        owner_flat[batch_base + 0 * cols + 1] = 3.0f;
        owner_flat[batch_base + 0 * cols + 2] = 5.0f;
        owner_flat[batch_base + 1 * cols + 0] = 7.0f;
        owner_flat[batch_base + 1 * cols + 1] = 9.0f;
        owner_flat[batch_base + 1 * cols + 2] = 45.0f;
        owner_flat[batch_base + 2 * cols + 0] = 2.0f;
        owner_flat[batch_base + 2 * cols + 1] = 4.0f;
        owner_flat[batch_base + 2 * cols + 2] = 10.0f;
    }

    Tensor<float> owner(owner_dims, MemoryLocation::DEVICE);
    owner = owner_flat;

    std::vector<uint64_t> start_indices = {0, 0, 0, 0};
    std::vector<uint64_t> view_dims = {2, 1, rows, cols};

    std::vector<uint64_t> view_strides = {
        elems_per_batch * 2,
        elems_per_batch,
        static_cast<uint64_t>(cols),
        1
    };

    Tensor<float> view(owner, start_indices, view_dims, view_strides);

    ml::PCAResult<float> result = ml::pca(view, std::nullopt, true);

    Tensor<float> expected_expv({2, 1, 3});
    expected_expv =
    {
        2.78708718f, 0.21291282f, 7.03016493e-16f,
        2.99886771f, 0.00113229f, 7.05596321e-16f
    };

    Tensor<float> expected_loadings({2, 1, 3, 3});
    expected_loadings =
    {
        -0.58916765f, -0.39100062f, -0.70710678f,
        -0.58916765f, -0.39100062f,  0.70710678f,
        -0.55295837f,  0.83320888f, -1.22124533e-15f,

        -0.57740479f, -0.40817118f, -0.70710678f,
        -0.57740479f, -0.40817118f,  0.70710678f,
        -0.57724122f,  0.81657368f, -1.33559830e-13f
    };

    Tensor<float> expected_projections({2, 1, 3, 3});
    expected_projections =
    {
         1.20562378f,  0.41574624f, -7.20766725e-16f,
        -1.90547864f,  0.08070810f, -4.94177148e-17f,
         0.69985486f, -0.49645434f,  7.24197496e-16f,

         1.23552187f,  0.03055077f, -5.14396459e-15f,
        -1.97937101f,  0.00551589f, -3.04875468e-16f,
         0.74384913f, -0.03606666f,  5.87388085e-15f
    };

    const double tol = 5e-3;

    // Check explained variance.
    {
        auto it_a = result.explained_variance.begin();
        auto it_b = expected_expv.begin();
        for (; it_a != result.explained_variance.end() &&
            it_b != expected_expv.end();
            ++it_a, ++it_b)
        {
            EXPECT_NEAR(*it_a, *it_b, tol);
        }
    }

    // Check loadings.
    {
        auto it_a = result.loadings.begin();
        auto it_b = expected_loadings.begin();
        for (; it_a != result.loadings.end() && it_b != expected_loadings.end();
            ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }

    // Check projections.
    {
        auto it_a = result.projections.begin();
        auto it_b = expected_projections.begin();
        for (; it_a != result.projections.end() &&
            it_b != expected_projections.end();
            ++it_a, ++it_b)
        {
            EXPECT_NEAR(std::fabs(*it_a), std::fabs(*it_b), tol);
        }
    }
}

/**
 * @test PCA.pca_rank_less_than_two_throws
 * @brief PCA on rank-1 tensor should throw std::invalid_argument.
 */
TEST(PCA, pca_rank_less_than_two_throws)
{
    Tensor<float> t({5});

    EXPECT_THROW({
        ml::pca(t, std::nullopt, true);
    }, std::invalid_argument);
}

/**
 * @test PCA.pca_n_components_too_large_throws
 * @brief Requesting more components than the feature dimension should throw.
 */
TEST(PCA, pca_n_components_too_large_throws)
{
    Tensor<float> t({4, 3});

    EXPECT_THROW({
        ml::pca(t, 4, true);
    }, std::invalid_argument);
}

/**
 * @test PCA.pca_n_components_zero_throws
 * @brief Requesting zero components should throw std::invalid_argument.
 */
TEST(PCA, pca_n_components_zero_throws)
{
    Tensor<float> t({4, 3});

    EXPECT_THROW({
        ml::pca(t, 0, true);
    }, std::invalid_argument);
}

} // namespace Test