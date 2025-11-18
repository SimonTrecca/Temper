/**
 * @file ut_Stats.cpp
 * @brief Google Test suite for statistical distributions utilities.
 *
 * Contains unit tests ensuring correctness of functions implemented
 * in the Stats module.
 */

#include <gtest/gtest.h>

#define private public
#define protected public
#include "temper/Stats.hpp"
#undef private
#undef protected

using namespace temper;

namespace Test
{
/**
* @test NORM.randn_deterministic_and_shape
* @brief Verify that `stats::randn` produces deterministic output when a
* non-zero seed is provided and that the produced tensor has the
* requested shape and number of elements.
*/
TEST(RANDN, randn_deterministic_and_shape)
{
    const std::vector<uint64_t> out_shape = {3, 4};
    const uint64_t total = 3 * 4;
    const uint64_t seed = 2025ULL;

    Tensor<float> a = stats::randn<float>
        (out_shape, MemoryLocation::DEVICE, seed);
    Tensor<float> b = stats::randn<float>
        (out_shape, MemoryLocation::DEVICE, seed);

    std::vector<float> host_a(total), host_b(total);
    g_sycl_queue.memcpy(host_a.data(),
        a.m_p_data.get(), sizeof(float) * total).wait();
    g_sycl_queue.memcpy(host_b.data(),
        b.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(host_a[i], host_b[i]);
    }

    EXPECT_EQ(a.get_num_elements(), total);
}

/**
 * @test NORM.pdf_basic_values
 * @brief Check that pdf returns known values for simple 1-D inputs.
 */
TEST(NORM, pdf_basic_values)
{
    Tensor<float> x({3}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {0.0f, 1.0f, -1.0f};
    x = x_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    Tensor<float> out = stats::norm::pdf<float>(x, loc, scale);
    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();
    std::vector<float> expected = {
        0.3989422804014327f,
        0.24197072451914337f,
        0.24197072451914337f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.pdf_broadcast_loc_scalar_scale_vector_x_vector
 * @brief Verify broadcasting: scalar loc with vector scale and vector x.
 */
TEST(NORM, pdf_broadcast_loc_scalar_scale_vector_x_vector)
{
    Tensor<float> x({4}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {-1.0f, 0.0f, 0.5f, 2.0f};
    x = x_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({4}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 2.0f, 0.5f, 1.5f};
    scale = scale_vals;
    Tensor<float> out = stats::norm::pdf<float>(x, loc, scale);
    std::vector<float> host_out(4);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 4).wait();
    std::vector<float> expected = {
        0.24197072451914337f,
        0.19947114020071635f,
        0.48394144903828673f,
        0.10934004978399577f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.pdf_broadcast_x_scalar_loc_vector_scale_vector
 * @brief Verify broadcasting: scalar x with vector loc and vector scale.
 */
TEST(NORM, pdf_broadcast_x_scalar_loc_vector_scale_vector)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{1.0f};
    Tensor<float> loc({3}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {0.0f, 1.0f, 2.0f};
    loc = loc_vals;
    Tensor<float> scale({3}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 0.5f, 2.0f};
    scale = scale_vals;
    Tensor<float> out = stats::norm::pdf<float>(x, loc, scale);
    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();
    std::vector<float> expected = {
        0.24197072451914337f,
        0.7978845608028654f,
        0.17603266338214976f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.pdf_broadcast_2d_x_loc_vector_scale_scalar
 * @brief Verify broadcasting for 2D x against vector loc and scalar scale.
 */
TEST(NORM, pdf_broadcast_2d_x_loc_vector_scale_scalar)
{
    Tensor<float> x({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {0.0f, 1.0f, 2.0f, 3.0f};
    x = x_vals;
    Tensor<float> loc({2}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {0.0f, 1.0f};
    loc = loc_vals;
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    Tensor<float> out = stats::norm::pdf<float>(x, loc, scale);
    std::vector<float> host_out(4);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 4).wait();
    std::vector<float> expected = {
        0.3989422804014327f,
        0.3989422804014327f,
        0.05399096651318806f,
        0.05399096651318806f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.pdf_broadcast_mixed_shapes_all_operands
 * @brief Verify pdf with mixed-shaped operands that exercise broadcasting rules.
 */
TEST(NORM, pdf_broadcast_mixed_shapes_all_operands)
{
    Tensor<float> x({2, 1}, MemoryLocation::DEVICE);
    x = std::vector<float>{0.0f, 1.0f};
    Tensor<float> loc({1, 3}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f, 1.0f, 2.0f};
    Tensor<float> scale({1, 3}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f, 0.5f, 2.0f};
    Tensor<float> out = stats::norm::pdf<float>(x, loc, scale);
    std::vector<float> host_out(6);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 6).wait();
    std::vector<float> expected = {
        0.3989422804014327f,
        0.10798193302637613f,
        0.12098536225957168f,
        0.24197072451914337f,
        0.7978845608028654f,
        0.17603266338214976f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.pdf_view_with_weird_strides
 * @brief Ensure pdf accepts non-contiguous/alias views for x, loc, and scale.
 */
TEST(NORM, pdf_view_with_weird_strides)
{
    const std::vector<uint64_t> out_shape = {2, 3};
    const uint64_t total = 6;

    Tensor<float> loc_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {
        10.0f, 99.0f, 99.0f, 20.0f, 99.0f, 99.0f
    };
    loc_owner = loc_vals;
    Tensor<float> loc_alias(
        loc_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 1ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    Tensor<float> scale_owner({5}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {
        1.0f, 99.0f, 1.0f, 99.0f, 1.0f
    };
    scale_owner = scale_vals;
    Tensor<float> scale_alias(
        scale_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{1ull, 3ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    Tensor<float> x_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> x_owner_vals = {
        10.0f, 20.0f, 10.0f, 20.0f, 10.0f, 20.0f
    };
    x_owner = x_owner_vals;
    Tensor<float> x_alias(
        x_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 3ull},
        std::vector<uint64_t>{1ull, 2ull}
    );

    Tensor<float> out = stats::norm::pdf<float>(x_alias, loc_alias, scale_alias);

    std::vector<float> host_out(total);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected(total, 0.3989422804014327f);
    const double tol = 1e-6;
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.pdf_throws_on_nonpositive_scale
 * @brief pdf should throw std::invalid_argument when any scale element <= 0.
 */
TEST(NORM, pdf_throws_on_nonpositive_scale)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{0.0f};
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{0.0f};
    EXPECT_THROW(stats::norm::pdf<float>(x, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.pdf_throws_on_nan_input
 * @brief pdf should throw std::invalid_argument when x contains NaN.
 */
TEST(NORM, pdf_throws_on_nan_input)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    x = nanf;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    EXPECT_THROW(stats::norm::pdf<float>(x, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.logpdf_basic_values
 * @brief Check that logpdf returns the log of known pdf values for inputs.
 */
TEST(NORM, logpdf_basic_values)
{
    Tensor<float> x({3}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {0.0f, 1.0f, -1.0f};
    x = x_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    Tensor<float> out = stats::norm::logpdf<float>(x, loc, scale);
    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();
    std::vector<float> expected = {
        -0.9189385332046727f,
        -1.4189385332046727f,
        -1.4189385332046727f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.logpdf_broadcast_loc_scalar_scale_vector_x_vector
 * @brief Verify broadcasting: scalar loc with vector scale and vector x.
 */
TEST(NORM, logpdf_broadcast_loc_scalar_scale_vector_x_vector)
{
    Tensor<float> x({4}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {-1.0f, 0.0f, 0.5f, 2.0f};
    x = x_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({4}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 2.0f, 0.5f, 1.5f};
    scale = scale_vals;
    Tensor<float> out = stats::norm::logpdf<float>(x, loc, scale);
    std::vector<float> host_out(4);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 4).wait();
    std::vector<float> expected = {
        -1.4189385332046727f,
        -1.612085713764618f,
        -0.7257913526447274f,
        -2.2132925302017257f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.logpdf_broadcast_2d_x_loc_vector_scale_scalar
 * @brief Verify broadcasting for 2D x against vector loc and scalar scale.
 */
TEST(NORM, logpdf_broadcast_2d_x_loc_vector_scale_scalar)
{
    Tensor<float> x({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {0.0f, 1.0f, 2.0f, 3.0f};
    x = x_vals;
    Tensor<float> loc({2}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {0.0f, 1.0f};
    loc = loc_vals;
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    Tensor<float> out = stats::norm::logpdf<float>(x, loc, scale);
    std::vector<float> host_out(4);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 4).wait();
    std::vector<float> expected = {
        -0.9189385332046727f,
        -0.9189385332046727f,
        -2.9189385332046727f,
        -2.9189385332046727f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.logpdf_broadcast_mixed_shapes_all_operands
 * @brief Verify logpdf with mixed-shaped operands exercising broadcasting.
 */
TEST(NORM, logpdf_broadcast_mixed_shapes_all_operands)
{
    Tensor<float> x({2, 1}, MemoryLocation::DEVICE);
    x = std::vector<float>{0.0f, 1.0f};
    Tensor<float> loc({1, 3}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f, 1.0f, 2.0f};
    Tensor<float> scale({1, 3}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f, 0.5f, 2.0f};
    Tensor<float> out = stats::norm::logpdf<float>(x, loc, scale);
    std::vector<float> host_out(6);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 6).wait();
    std::vector<float> expected = {
        -0.9189385332046727f,
        -2.2257913526447273f,
        -2.112085713764618f,
        -1.4189385332046727f,
        -0.22579135264472738f,
        -1.7370857137646178f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.logpdf_view_with_weird_strides
 * @brief Ensure logpdf accepts x/loc/scale provided as alias-views.
 */
TEST(NORM, logpdf_view_with_weird_strides)
{
    const std::vector<uint64_t> out_shape = {2, 3};
    const uint64_t total = 6;
    Tensor<float> loc_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {
        10.0f, 99.0f, 99.0f, 20.0f, 99.0f, 99.0f
    };
    loc_owner = loc_vals;
    Tensor<float> loc_alias(
        loc_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 1ull},
        std::vector<uint64_t>{3ull, 2ull}
    );
    Tensor<float> scale_owner({5}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {
        1.0f, 99.0f, 1.0f, 99.0f, 1.0f
    };
    scale_owner = scale_vals;
    Tensor<float> scale_alias(
        scale_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{1ull, 3ull},
        std::vector<uint64_t>{3ull, 2ull}
    );
    Tensor<float> x_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> x_owner_vals = {
        10.0f, 20.0f, 10.0f, 20.0f, 10.0f, 20.0f
    };
    x_owner = x_owner_vals;
    Tensor<float> x_alias(
        x_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 3ull},
        std::vector<uint64_t>{1ull, 2ull}
    );
    Tensor<float> out = stats::norm::logpdf<float>(x_alias, loc_alias,
        scale_alias);
    std::vector<float> host_out(total);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * total).wait();
    std::vector<float> expected(total, -0.9189385332046727f);
    const double tol = 1e-6;
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.logpdf_throws_on_nonpositive_scale
 * @brief logpdf should throw std::invalid_argument when scale <= 0.
 */
TEST(NORM, logpdf_throws_on_nonpositive_scale)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{0.0f};
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{0.0f};
    EXPECT_THROW(stats::norm::logpdf<float>(x, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.logpdf_throws_on_nan_input
 * @brief logpdf should throw std::invalid_argument when x contains NaN.
 */
TEST(NORM, logpdf_throws_on_nan_input)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    x = nanf;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    EXPECT_THROW(stats::norm::logpdf<float>(x, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.cdf_basic_values
 * @brief Check that cdf returns known values for simple 1-D inputs.
 */
TEST(NORM, cdf_basic_values)
{
    Tensor<float> x({3}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {0.0f, 1.0f, -1.0f};
    x = x_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    Tensor<float> out = stats::norm::cdf<float>(x, loc, scale);
    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();
    std::vector<float> expected = {
        0.5f,
        0.8413447460685429f,
        0.15865525393145707f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.cdf_broadcast_loc_scalar_scale_vector_x_vector
 * @brief Verify broadcasting: scalar loc with vector scale and vector x for cdf.
 */
TEST(NORM, cdf_broadcast_loc_scalar_scale_vector_x_vector)
{
    Tensor<float> x({4}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {-1.0f, 0.0f, 0.5f, 2.0f};
    x = x_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({4}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 2.0f, 0.5f, 1.5f};
    scale = scale_vals;
    Tensor<float> out = stats::norm::cdf<float>(x, loc, scale);
    std::vector<float> host_out(4);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 4).wait();
    std::vector<float> expected = {
        0.15865525393145707f,
        0.5f,
        0.8413447460685429f,
        0.9087887802741321f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.cdf_broadcast_x_scalar_loc_vector_scale_vector
 * @brief Verify broadcasting: scalar x with vector loc and vector scale for cdf.
 */
TEST(NORM, cdf_broadcast_x_scalar_loc_vector_scale_vector)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{1.0f};
    Tensor<float> loc({3}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {0.0f, 1.0f, 2.0f};
    loc = loc_vals;
    Tensor<float> scale({3}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 0.5f, 2.0f};
    scale = scale_vals;
    Tensor<float> out = stats::norm::cdf<float>(x, loc, scale);
    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();
    std::vector<float> expected = {
        0.8413447460685429f,
        0.5f,
        0.3085375387259869f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.cdf_broadcast_2d_x_loc_vector_scale_scalar
 * @brief Verify cdf broadcasting for 2D x against vector loc and scalar scale.
 */
TEST(NORM, cdf_broadcast_2d_x_loc_vector_scale_scalar)
{
    Tensor<float> x({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> x_vals = {0.0f, 1.0f, 2.0f, 3.0f};
    x = x_vals;
    Tensor<float> loc({2}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {0.0f, 1.0f};
    loc = loc_vals;
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    Tensor<float> out = stats::norm::cdf<float>(x, loc, scale);
    std::vector<float> host_out(4);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 4).wait();
    std::vector<float> expected = {
        0.5f,
        0.5f,
        0.9772498680518208f,
        0.9772498680518208f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.cdf_broadcast_mixed_shapes_all_operands
 * @brief Verify cdf with mixed-shaped operands that exercise broadcasting rules.
 */
TEST(NORM, cdf_broadcast_mixed_shapes_all_operands)
{
    Tensor<float> x({2, 1}, MemoryLocation::DEVICE);
    x = std::vector<float>{0.0f, 1.0f};
    Tensor<float> loc({1, 3}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f, 1.0f, 2.0f};
    Tensor<float> scale({1, 3}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f, 0.5f, 2.0f};
    Tensor<float> out = stats::norm::cdf<float>(x, loc, scale);
    std::vector<float> host_out(6);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 6).wait();
    std::vector<float> expected = {
        0.5f,
        0.02275013194817921f,
        0.15865525393145707f,
        0.8413447460685429f,
        0.5f,
        0.3085375387259869f
    };
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.cdf_view_with_weird_strides
 * @brief Ensure cdf accepts non-contiguous/alias views for x, loc, and scale.
 */
TEST(NORM, cdf_view_with_weird_strides)
{
    const std::vector<uint64_t> out_shape = {2, 3};
    const uint64_t total = 6;

    Tensor<float> loc_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {
        10.0f, 99.0f, 99.0f, 20.0f, 99.0f, 99.0f
    };
    loc_owner = loc_vals;
    Tensor<float> loc_alias(
        loc_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 1ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    Tensor<float> scale_owner({5}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {
        1.0f, 99.0f, 1.0f, 99.0f, 1.0f
    };
    scale_owner = scale_vals;
    Tensor<float> scale_alias(
        scale_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{1ull, 3ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    Tensor<float> x_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> x_owner_vals = {
        10.0f, 20.0f, 10.0f, 20.0f, 10.0f, 20.0f
    };
    x_owner = x_owner_vals;
    Tensor<float> x_alias(
        x_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 3ull},
        std::vector<uint64_t>{1ull, 2ull}
    );

    Tensor<float> out = stats::norm::cdf<float>(x_alias, loc_alias, scale_alias);

    std::vector<float> host_out(total);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected(total, 0.5f);
    const double tol = 1e-6;
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.cdf_throws_on_nonpositive_scale
 * @brief cdf should throw std::invalid_argument when any scale element <= 0.
 */
TEST(NORM, cdf_throws_on_nonpositive_scale)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{0.0f};
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{0.0f};
    EXPECT_THROW(stats::norm::cdf<float>(x, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.cdf_throws_on_nan_input
 * @brief cdf should throw std::invalid_argument when x contains NaN.
 */
TEST(NORM, cdf_throws_on_nan_input)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    x = nanf;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    EXPECT_THROW(stats::norm::cdf<float>(x, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.ppf_basic_quantiles
 * @brief ppf returns known quantiles for the standard normal (loc=0, scale=1)
 */
TEST(NORM, ppf_basic_quantiles)
{
    Tensor<float> q({3}, MemoryLocation::DEVICE);
    std::vector<float> q_vals = {0.5f, 0.841344746f, 0.158655254f};
    q = q_vals;

    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};

    Tensor<float> out = stats::norm::ppf<float>(q, loc, scale);

    const double tol = 1e-5;
    EXPECT_NEAR(out[0], 0.0, tol);
    EXPECT_NEAR(out[1], 1.0, tol);
    EXPECT_NEAR(out[2], -1.0, tol);
}

/**
 * @test NORM.ppf_view_and_alias
 * @brief ppf accepts loc/scale provided as views and alias-views and
 * produces results consistent with broadcasting rules.
 */
TEST(NORM, ppf_view_and_alias)
{
    const std::vector<uint64_t> q_shape = {2, 3};
    Tensor<float> q(q_shape, MemoryLocation::DEVICE);
    std::vector<float> q_vals(6, 0.5f);
    q = q_vals;

    Tensor<float> loc_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {10.0f, 99.0f, 99.0f, 20.0f, 99.0f, 99.0f};
    loc_owner = loc_vals;

    Tensor<float> loc_alias(
        loc_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 1ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    Tensor<float> scale_owner({5}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 99.0f, 1.0f, 99.0f, 1.0f};
    scale_owner = scale_vals;

    Tensor<float> scale_alias(
        scale_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{1ull, 3ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    Tensor<float> out = stats::norm::ppf<float>(q, loc_alias, scale_alias);

    std::vector<float> host_out(6);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 6).wait();

    EXPECT_FLOAT_EQ(host_out[0], 10.0f);
    EXPECT_FLOAT_EQ(host_out[1], 10.0f);
    EXPECT_FLOAT_EQ(host_out[2], 10.0f);
    EXPECT_FLOAT_EQ(host_out[3], 20.0f);
    EXPECT_FLOAT_EQ(host_out[4], 20.0f);
    EXPECT_FLOAT_EQ(host_out[5], 20.0f);
}

/**
* @test NORM.ppf_throws_on_q_out_of_range
* @brief ppf should throw invalid_argument when q contains values outside [0,1].
*/
TEST(NORM, ppf_throws_on_q_out_of_range)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    std::vector<float> q_vals = {1.5f};
    q = q_vals;

    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};

    EXPECT_THROW(stats::norm::ppf<float>(q, loc, scale), std::invalid_argument);
}

/**
* @test NORM.ppf_throws_on_nonpositive_scale
* @brief ppf should throw invalid_argument when scale <= 0.
*/
TEST(NORM, ppf_throws_on_nonpositive_scale)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    std::vector<float> q_vals = {0.5f};
    q = q_vals;

    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{0.0f};


    EXPECT_THROW(stats::norm::ppf<float>(q, loc, scale), std::invalid_argument);
}

/**
* @test NORM.ppf_throws_on_nan_input
* @brief ppf should throw runtime_error when inputs contain NaN.
*/
TEST(NORM, ppf_throws_on_nan_input)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    q = nanf;

    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};

    EXPECT_THROW(stats::norm::ppf<float>(q, loc, scale), std::invalid_argument);
}

/**
 * @test NORM.isf_basic_quantiles
 * @brief isf returns known quantiles via relation isf(q) == ppf(1 - q)
 */
TEST(NORM, isf_basic_quantiles)
{
    Tensor<float> q({3}, MemoryLocation::DEVICE);
    std::vector<float> q_vals = {0.5f, 0.841344746f, 0.158655254f};
    q = q_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    Tensor<float> out = stats::norm::isf<float>(q, loc, scale);
    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();
    const double tol = 1e-5;
    EXPECT_NEAR(static_cast<double>(host_out[0]), 0.0, tol);
    EXPECT_NEAR(static_cast<double>(host_out[1]), -1.0, tol);
    EXPECT_NEAR(static_cast<double>(host_out[2]), 1.0, tol);
}

/**
 * @test NORM.isf_matches_ppf_with_complement
 * @brief isf(q) should equal ppf(1 - q) elementwise (broadcasting too).
 */
TEST(NORM, isf_matches_ppf_with_complement)
{
    const std::vector<uint64_t> out_shape = {4, 5};
    const uint64_t total = 4 * 5;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{2.5f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{0.75f};
    std::vector<float> host_uniforms; host_uniforms.reserve(total);
    for (uint64_t flat = 0; flat < total; ++flat)
    {
        uint64_t s = 123456789ULL ^
            (flat + 0x9e3779b97f4a7c15ULL);
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        uint64_t rnd = s * 2685821657736338717ULL;
        double u = static_cast<double>(rnd) /
            18446744073709551616.0;
        if (u < 1e-16) u = 1e-16;
        if (u > 1.0 - 1e-16) u = 1.0 - 1e-16;
        host_uniforms.push_back(static_cast<float>(u));
    }
    Tensor<float> q(out_shape, MemoryLocation::DEVICE);
    q = host_uniforms;
    std::vector<float> host_q_comp(total);
    for (uint64_t i = 0; i < total; ++i)
        host_q_comp[i] = 1.0f - host_uniforms[i];
    Tensor<float> q_comp(out_shape, MemoryLocation::DEVICE);
    q_comp = host_q_comp;
    Tensor<float> expected = stats::norm::ppf<float>(q_comp, loc, scale);
    Tensor<float> actual = stats::norm::isf<float>(q, loc, scale);
    std::vector<float> host_expected(total);
    std::vector<float> host_actual(total);
    g_sycl_queue.memcpy(host_expected.data(),
        expected.m_p_data.get(), sizeof(float) * total).wait();
    g_sycl_queue.memcpy(host_actual.data(),
        actual.m_p_data.get(), sizeof(float) * total).wait();
    const double tol = 1e-6;
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_expected[i]),
            static_cast<double>(host_actual[i]), tol);
    }
}

/**
 * @test NORM.isf_view_and_alias
 * @brief isf accepts q/loc/scale provided as alias-views with weird strides.
 */
TEST(NORM, isf_view_and_alias)
{
    const std::vector<uint64_t> q_shape = {2, 3};
    Tensor<float> q_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> q_vals(6, 0.5f);
    q_owner = q_vals;
    Tensor<float> q_alias(
        q_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 1ull},
        std::vector<uint64_t>{3ull, 2ull}
    );
    Tensor<float> loc_owner({6}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {
        10.0f, 99.0f, 99.0f, 20.0f, 99.0f, 99.0f
    };
    loc_owner = loc_vals;
    Tensor<float> loc_alias(
        loc_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{2ull, 1ull},
        std::vector<uint64_t>{3ull, 2ull}
    );
    Tensor<float> scale_owner({5}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 99.0f, 1.0f, 99.0f, 1.0f};
    scale_owner = scale_vals;
    Tensor<float> scale_alias(
        scale_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{1ull, 3ull},
        std::vector<uint64_t>{3ull, 2ull}
    );
    Tensor<float> out = stats::norm::isf<float>(q_alias, loc_alias,
        scale_alias);
    std::vector<float> host_out(6);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 6).wait();
    EXPECT_FLOAT_EQ(host_out[0], 10.0f);
    EXPECT_FLOAT_EQ(host_out[1], 10.0f);
    EXPECT_FLOAT_EQ(host_out[2], 10.0f);
    EXPECT_FLOAT_EQ(host_out[3], 20.0f);
    EXPECT_FLOAT_EQ(host_out[4], 20.0f);
    EXPECT_FLOAT_EQ(host_out[5], 20.0f);
}

/**
 * @test NORM.isf_throws_on_q_out_of_range
 * @brief isf should throw invalid_argument when q contains values outside [0,1].
 */
TEST(NORM, isf_throws_on_q_out_of_range)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    std::vector<float> q_vals = {1.5f};
    q = q_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    EXPECT_THROW(stats::norm::isf<float>(q, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.isf_throws_on_nonpositive_scale
 * @brief isf should throw invalid_argument when scale <= 0.
 */
TEST(NORM, isf_throws_on_nonpositive_scale)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    std::vector<float> q_vals = {0.5f};
    q = q_vals;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{0.0f};
    EXPECT_THROW(stats::norm::isf<float>(q, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.isf_throws_on_nan_input
 * @brief isf should throw std::invalid_argument when q contains NaN.
 */
TEST(NORM, isf_throws_on_nan_input)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    q = nanf;
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{1.0f};
    EXPECT_THROW(stats::norm::isf<float>(q, loc, scale),
        std::invalid_argument);
}

/**
 * @test NORM.rvs_matches_ppf_with_seed
 * @brief rvs(seed) should equal
 * ppf(uniforms_generated_with_same_seed, loc, scale).
 */
TEST(NORM, rvs_matches_ppf_with_seed)
{
    const std::vector<uint64_t> out_shape = {4, 5};
    const uint64_t total = 4 * 5;
    const uint64_t seed = 123456789ULL;

    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{2.5f};
    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{0.75f};

    std::vector<float> host_uniforms; host_uniforms.reserve(total);
    for (uint64_t flat = 0; flat < total; ++flat)
    {
        uint64_t s = seed ^ (flat + 0x9e3779b97f4a7c15ULL);
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        uint64_t rnd = s * 2685821657736338717ULL;
        double u = static_cast<double>(rnd) / 18446744073709551616.0;
        if (u < 1e-16) u = 1e-16;
        if (u > 1.0 - 1e-16) u = 1.0 - 1e-16;
        host_uniforms.push_back(static_cast<float>(u));
    }

    Tensor<float> q(out_shape, MemoryLocation::DEVICE);
    q = host_uniforms;
    Tensor<float> expected = stats::norm::ppf<float>(q, loc, scale);

    Tensor<float> actual = stats::norm::rvs<float>
        (loc, scale, out_shape, MemoryLocation::DEVICE, seed);

    std::vector<float> host_expected(total);
    std::vector<float> host_actual(total);
    g_sycl_queue.memcpy(host_expected.data(),
        expected.m_p_data.get(), sizeof(float) * total).wait();
    g_sycl_queue.memcpy(host_actual.data(),
        actual.m_p_data.get(), sizeof(float) * total).wait();

    const double tol = 1e-6;
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_expected[i]),
            static_cast<double>(host_actual[i]), tol);
    }
}

/**
 * @test NORM.rvs_view_and_alias
 * @brief rvs works when loc and scale are provided as views/alias views.
 */
TEST(NORM, rvs_view_and_alias)
{
    const std::vector<uint64_t> out_shape = {3, 3};
    const uint64_t total = 3 * 3;
    const uint64_t seed = 4242424242ULL;

    Tensor<float> loc_owner({7}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {0.0f, 99.0f, 99.0f, 1.0f, 99.0f, 99.0f, 2.0f};
    loc_owner = loc_vals;
    Tensor<float> loc_alias(
        loc_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{3ull, 1ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    Tensor<float> scale_owner({5}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    scale_owner = scale_vals;

    Tensor<float> scale_alias(
        scale_owner,
        std::vector<uint64_t>{0ull},
        std::vector<uint64_t>{1ull, 3ull},
        std::vector<uint64_t>{3ull, 2ull}
    );

    std::vector<float> host_uniforms; host_uniforms.reserve(total);
    for (uint64_t flat = 0; flat < total; ++flat)
    {
        uint64_t s = seed ^ (flat + 0x9e3779b97f4a7c15ULL);
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        uint64_t rnd = s * 2685821657736338717ULL;
        double u = static_cast<double>(rnd) / 18446744073709551616.0;
        if (u < 1e-16) u = 1e-16;
        if (u > 1.0 - 1e-16) u = 1.0 - 1e-16;
        host_uniforms.push_back(static_cast<float>(u));
    }

    Tensor<float> q(out_shape, MemoryLocation::DEVICE);
    q = host_uniforms;
    Tensor<float> expected = stats::norm::ppf<float>(q, loc_alias, scale_alias);

    Tensor<float> actual = stats::norm::rvs<float>
        (loc_alias, scale_alias, out_shape, MemoryLocation::DEVICE, seed);

    std::vector<float> host_expected(total);
    std::vector<float> host_actual(total);
    g_sycl_queue.memcpy(host_expected.data(),
        expected.m_p_data.get(), sizeof(float) * total).wait();
    g_sycl_queue.memcpy(host_actual.data(),
        actual.m_p_data.get(), sizeof(float) * total).wait();

    const double tol = 1e-6;
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_expected[i]),
            static_cast<double>(host_actual[i]), tol);
    }
}

/**
* @test NORM.rvs_throws_when_scale_nonpositive
* @brief rvs should propagate ppf errors: when scale <= 0, rvs must throw.
*/
TEST(NORM, rvs_throws_when_scale_nonpositive)
{
    const std::vector<uint64_t> out_shape = {2, 2};
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{1.0f};

    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{-0.5f};

    EXPECT_THROW(stats::norm::rvs<float>
        (loc, scale, out_shape, MemoryLocation::DEVICE, 42ULL),
        std::invalid_argument);
}

/**
 * @test NORM.mean_returns_loc
 * @brief mean(loc, scale) should return loc (ignoring scale).
 */
TEST(NORM, mean_returns_loc)
{
    Tensor<float> loc({3}, MemoryLocation::DEVICE);
    std::vector<float> loc_vals = {1.0f, 2.0f, -3.5f};
    loc = loc_vals;

    Tensor<float> scale({1}, MemoryLocation::DEVICE);
    scale = std::vector<float>{2.0f};

    Tensor<float> out = stats::norm::mean<float>(loc, scale);

    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();

    for (size_t i = 0; i < loc_vals.size(); ++i)
    {
        EXPECT_FLOAT_EQ(host_out[i], loc_vals[i]);
    }
}

/**
 * @test NORM.var_scale_squared
 * @brief var(loc, scale) should return scale ** 2 (elementwise).
 */
TEST(NORM, var_scale_squared)
{
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};

    Tensor<float> scale({3}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {1.0f, 2.0f, 0.5f};
    scale = scale_vals;

    Tensor<float> out = stats::norm::var<float>(loc, scale);

    std::vector<float> host_out(3);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();

    std::vector<float> expected = {1.0f, 4.0f, 0.25f};
    const double tol = 1e-6;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(expected[i]), tol);
    }
}

/**
 * @test NORM.stddev_returns_scale
 * @brief stddev(loc, scale) should return scale (ignoring loc).
 */
TEST(NORM, stddev_returns_scale)
{
    Tensor<float> loc({1}, MemoryLocation::DEVICE);
    loc = std::vector<float>{0.0f};

    Tensor<float> scale({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> scale_vals = {0.1f, 1.0f, 2.5f, 3.0f};
    scale = scale_vals;

    Tensor<float> out = stats::norm::stddev<float>(loc, scale);

    const uint64_t total = 4;
    std::vector<float> host_out(total);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float) * total).wait();

    const double tol = 1e-6;
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_NEAR(static_cast<double>(host_out[i]),
            static_cast<double>(scale_vals[i]), tol);
    }
}

#include "stats/chisquare_pdf.cpp"

/**
 * @test CHISQUARE.pdf_throws_on_empty_x
 * @brief cdf should throw std::invalid_argument when x is empty.
 */
TEST(CHISQUARE, pdf_throws_on_empty_x)
{
    Tensor<float> x;
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    EXPECT_THROW(stats::chisquare::pdf<float>(x, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.pdf_throws_on_empty_k
 * @brief pdf should throw std::invalid_argument when k is empty.
 */
TEST(CHISQUARE, pdf_throws_on_empty_k)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{1.0f};
    Tensor<float> k;

    EXPECT_THROW(stats::chisquare::pdf<float>(x, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.pdf_throws_on_negative_x
 * @brief pdf should throw std::invalid_argument when k contains values <= 0.
 */
TEST(CHISQUARE, pdf_throws_on_negative_x)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{-1.0f};
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    EXPECT_THROW(stats::chisquare::pdf<float>(x, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.pdf_throws_on_negative_or_zero_k
 * @brief pdf should throw std::invalid_argument when x contains negative values.
 */
TEST(CHISQUARE, pdf_throws_on_negative_or_zero_k)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{1.0f};
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    Tensor<float> k2({1}, MemoryLocation::DEVICE);
    k2 = std::vector<float>{-1.0f};

    EXPECT_THROW(stats::chisquare::pdf<float>(x, k), std::invalid_argument);
    EXPECT_THROW(stats::chisquare::pdf<float>(x, k2), std::invalid_argument);
}

/**
 * @test CHISQUARE.pdf_throws_on_nan_input
 * @brief pdf should throw std::invalid_argument when x contains NaN.
 */
TEST(CHISQUARE, pdf_throws_on_nan_input)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    x = nanf;
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{2.0f};

    EXPECT_THROW(stats::chisquare::pdf<float>(x, k), std::invalid_argument);
}

#include "stats/chisquare_cdf.cpp"

/**
 * @test CHISQUARE.cdf_throws_on_empty_x
 * @brief cdf should throw std::invalid_argument when x is empty.
 */
TEST(CHISQUARE, cdf_throws_on_empty_x)
{
    Tensor<float> x;
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    EXPECT_THROW(stats::chisquare::cdf<float>(x, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.cdf_throws_on_empty_k
 * @brief cdf should throw std::invalid_argument when k is empty.
 */
TEST(CHISQUARE, cdf_throws_on_empty_k)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{1.0f};
    Tensor<float> k;

    EXPECT_THROW(stats::chisquare::cdf<float>(x, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.cdf_throws_on_negative_x
 * @brief cdf should throw std::invalid_argument when k contains values <= 0.
 */
TEST(CHISQUARE, cdf_throws_on_negative_x)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{-1.0f};
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    EXPECT_THROW(stats::chisquare::cdf<float>(x, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.cdf_throws_on_negative_or_zero_k
 * @brief cdf should throw std::invalid_argument when x contains negative values.
 */
TEST(CHISQUARE, cdf_throws_on_negative_or_zero_k)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    x = std::vector<float>{1.0f};
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    Tensor<float> k2({1}, MemoryLocation::DEVICE);
    k2 = std::vector<float>{-1.0f};

    EXPECT_THROW(stats::chisquare::cdf<float>(x, k), std::invalid_argument);
    EXPECT_THROW(stats::chisquare::cdf<float>(x, k2), std::invalid_argument);
}

/**
 * @test CHISQUARE.cdf_throws_on_nan_input
 * @brief cdf should throw std::invalid_argument when x contains NaN.
 */
TEST(CHISQUARE, cdf_throws_on_nan_input)
{
    Tensor<float> x({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    x = nanf;
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{2.0f};

    EXPECT_THROW(stats::chisquare::cdf<float>(x, k), std::invalid_argument);
}

#include "stats/chisquare_ppf.cpp"

/**
 * @test CHISQUARE.ppf_throws_on_empty_q
 * @brief ppf should throw std::invalid_argument when q is empty.
 */
TEST(CHISQUARE, ppf_throws_on_empty_q)
{
    Tensor<float> q;
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    EXPECT_THROW(stats::chisquare::ppf<float>(q, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.ppf_throws_on_empty_k
 * @brief cdf should throw std::invalid_argument when k is empty.
 */
TEST(CHISQUARE, ppf_throws_on_empty_k)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    q = std::vector<float>{1.0f};
    Tensor<float> k;

    EXPECT_THROW(stats::chisquare::ppf<float>(q, k), std::invalid_argument);
}

/**
* @test CHISQUARE.ppf_throws_on_q_out_of_range
* @brief ppf should throw invalid_argument when q contains values outside [0,1].
*/
TEST(CHISQUARE, ppf_throws_on_q_out_of_range)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    std::vector<float> q_vals = {1.5f};
    q = q_vals;

    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    EXPECT_THROW(stats::chisquare::ppf<float>(q, k), std::invalid_argument);
}

/**
 * @test CHISQUARE.ppf_throws_on_negative_or_zero_k
 * @brief ppf should throw std::invalid_argument when x contains negative values.
 */
TEST(CHISQUARE, ppf_throws_on_negative_or_zero_k)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    q = std::vector<float>{0.5f};
    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    Tensor<float> k2({1}, MemoryLocation::DEVICE);
    k2 = std::vector<float>{-1.0f};

    EXPECT_THROW(stats::chisquare::ppf<float>(q, k), std::invalid_argument);
    EXPECT_THROW(stats::chisquare::ppf<float>(q, k2), std::invalid_argument);
}

/**
* @test CHISQUARE.ppf_throws_on_nan_input
* @brief ppf should throw runtime_error when inputs contain NaN.
*/
TEST(CHISQUARE, ppf_throws_on_nan_input)
{
    Tensor<float> q({1}, MemoryLocation::DEVICE);
    float nanf = std::numeric_limits<float>::quiet_NaN();
    q = nanf;

    Tensor<float> k({1}, MemoryLocation::DEVICE);
    k = std::vector<float>{0.0f};

    EXPECT_THROW(stats::chisquare::ppf<float>(q, k), std::invalid_argument);
}

} // namespace Test