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

} // namespace Test