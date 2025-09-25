/**
 * @file ut_Math.cpp
 * @brief Google Test suite for mathematical tensor operations.
 *
 * Contains unit tests ensuring correctness of functions implemented
 * in the Math module.
 */

#include <gtest/gtest.h>

#define private public
#define protected public
#include "temper/Math.hpp"
#undef private
#undef protected

using namespace temper;

namespace Test
{
/**
 * @test MATMUL.basic_2d
 * @brief Basic 2x3 @ 3x2 = 2x2 check.
 */
TEST(MATMUL, basic_2d)
{
    Tensor<float> A({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> a_vals = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    A = a_vals;

    Tensor<float> B({3, 2}, MemoryLocation::DEVICE);
    std::vector<float> b_vals = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    B = b_vals;

    Tensor<float> R = math::matmul<float>(A, B);

    std::vector<float> expected(4, 0.0f);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            float s = 0.0f;
            for (uint64_t k = 0; k < 3; ++k)
            {
                s += a_vals[i * 3 + k] * b_vals[k * 2 + j];
            }
            expected[i * 2 + j] = s;
        }
    }

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), R.m_p_data.get(), host.size() * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], expected[0]);
    EXPECT_FLOAT_EQ(host[1], expected[1]);
    EXPECT_FLOAT_EQ(host[2], expected[2]);
    EXPECT_FLOAT_EQ(host[3], expected[3]);
}

/**
 * @test MATMUL.vec_vec_dot
 * @brief 1D x 1D -> scalar (dot product).
 */
TEST(MATMUL, vec_vec_dot)
{
    Tensor<float> a({3}, MemoryLocation::DEVICE);
    std::vector<float> av = {1.0f, 2.0f, 3.0f};
    a = av;

    Tensor<float> b({3}, MemoryLocation::DEVICE);
    std::vector<float> bv = {4.0f, 5.0f, 6.0f};
    b = bv;

    Tensor<float> r = math::matmul<float>(a, b);

    float expect = 0.0f;
    for (size_t t = 0; t < av.size(); ++t)
    {
        expect += av[t] * bv[t];
    }

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), r.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host[0], expect);
}

/**
 * @test MATMUL.vec_mat
 * @brief 1D x 2D -> vector.
 */
TEST(MATMUL, vec_mat)
{
    Tensor<float> a({2}, MemoryLocation::DEVICE);
    std::vector<float> av = {1.0f, 2.0f};
    a = av;

    Tensor<float> B({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> bv = {
        3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f
    };
    B = bv;

    Tensor<float> R = math::matmul<float>(a, B);

    std::vector<float> expected(3, 0.0f);
    for (uint64_t j = 0; j < 3; ++j)
    {
        float s = 0.0f;
        for (uint64_t t = 0; t < 2; ++t)
        {
            s += av[t] * bv[t * 3 + j];
        }
        expected[j] = s;
    }

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), R.m_p_data.get(), 3 * sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host[0], expected[0]);
    EXPECT_FLOAT_EQ(host[1], expected[1]);
    EXPECT_FLOAT_EQ(host[2], expected[2]);
}

/**
 * @test MATMUL.mat_vec
 * @brief 2D x 1D -> vector.
 */
TEST(MATMUL, mat_vec)
{
    Tensor<float> A({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> a_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    A = a_vals;

    Tensor<float> v({3}, MemoryLocation::DEVICE);
    std::vector<float> vv = {7.0f, 8.0f, 9.0f};
    v = vv;

    Tensor<float> R = math::matmul<float>(A, v);

    std::vector<float> expected(2, 0.0f);
    for (uint64_t i = 0; i < 2; ++i)
    {
        float s = 0.0f;
        for (uint64_t k = 0; k < 3; ++k)
        {
            s += a_vals[i * 3 + k] * vv[k];
        }
        expected[i] = s;
    }

    std::vector<float> host(2);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(), 2 * sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host[0], expected[0]);
    EXPECT_FLOAT_EQ(host[1], expected[1]);
}

/**
 * @test MATMUL.batched_equal_batches
 * @brief Batched matmul with matching batch dims: (2,3,4) @ (2,4,5) -> (2,3,5)
 */
TEST(MATMUL, batched_equal_batches)
{
    const uint64_t B = 2, M = 3, K = 4, N = 5;
    Tensor<float> A({B, M, K}, MemoryLocation::DEVICE);
    Tensor<float> Bt({B, K, N}, MemoryLocation::DEVICE);

    std::vector<float> a_vals(B * M * K);
    std::vector<float> b_vals(B * K * N);

    for (uint64_t i = 0; i < a_vals.size(); ++i)
    {
        a_vals[i] = static_cast<float>(i + 1);
    }
    for (uint64_t i = 0; i < b_vals.size(); ++i)
    {
        b_vals[i] = static_cast<float>(i + 1 + a_vals.size());
    }

    A = a_vals;
    Bt = b_vals;

    Tensor<float> R = math::matmul<float>(A, Bt);

    std::vector<float> expected(B * M * N, 0.0f);
    for (uint64_t b = 0; b < B; ++b)
    {
        for (uint64_t i = 0; i < M; ++i)
        {
            for (uint64_t j = 0; j < N; ++j)
            {
                float s = 0.0f;
                for (uint64_t t = 0; t < K; ++t)
                {
                    uint64_t a_idx = ((b * M) + i) * K + t;
                    uint64_t b_idx = ((b * K) + t) * N + j;
                    s += a_vals[a_idx] * b_vals[b_idx];
                }
                uint64_t r_idx = ((b * M) + i) * N + j;
                expected[r_idx] = s;
            }
        }
    }

    std::vector<float> host(B * M * N);
    g_sycl_queue.memcpy
        (host.data(), R.m_p_data.get(), host.size() * sizeof(float)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test MATMUL.batched_broadcast_batches
 * @brief Batched matmul with broadcasting: A:(2,1,2,3) @B:(1,3,3,2)
 * ->out:(2,3,2,2)
 */
TEST(MATMUL, batched_broadcast_batches)
{
    const std::vector<uint64_t> a_shape = {2, 1, 2, 3};
    const std::vector<uint64_t> b_shape = {1, 3, 3, 2};

    Tensor<float> A(a_shape, MemoryLocation::DEVICE);
    Tensor<float> Bt(b_shape, MemoryLocation::DEVICE);

    uint64_t a_elems = A.get_num_elements();
    uint64_t b_elems = Bt.get_num_elements();
    std::vector<float> a_vals(a_elems);
    std::vector<float> b_vals(b_elems);

    for (uint64_t i = 0; i < a_elems; ++i)
    {
        a_vals[i] = static_cast<float>(i + 1);
    }
    for (uint64_t i = 0; i < b_elems; ++i)
    {
        b_vals[i] = static_cast<float>(i + 101);
    }

    A = a_vals;
    Bt = b_vals;

    Tensor<float> R = math::matmul<float>(A, Bt);

    const uint64_t B0 = 2, B1 = 3, M = 2, K = 3, N = 2;
    std::vector<float> expected(B0 * B1 * M * N, 0.0f);

    for (uint64_t b0 = 0; b0 < B0; ++b0)
    {
        for (uint64_t b1 = 0; b1 < B1; ++b1)
        {
            for (uint64_t i = 0; i < M; ++i)
            {
                for (uint64_t j = 0; j < N; ++j)
                {
                    float s = 0.0f;
                    for (uint64_t t = 0; t < K; ++t)
                    {
                        uint64_t a_idx = (b0 * 2 + i) * 3 + t;
                        uint64_t b_idx = ((b1) * 3 + t) * 2 + j;
                        s += a_vals[a_idx] * b_vals[b_idx];
                    }
                    uint64_t out_idx = (((b0 * B1) + b1) * M + i) * N + j;
                    expected[out_idx] = s;
                }
            }
        }
    }

    std::vector<float> host(B0 * B1 * M * N);
    g_sycl_queue.memcpy
        (host.data(), R.m_p_data.get(), host.size() * sizeof(float)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test MATMUL.nan_throws
 * @brief matmul should throw std::runtime_error if inputs contain NaN.
 */
TEST(MATMUL, nan_throws)
{
    Tensor<float> A({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> a_vals = {1.0f, std::numeric_limits<float>::quiet_NaN(),
                                 3.0f, 4.0f};
    A = a_vals;

    Tensor<float> B({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> b_vals = {1.0f, 2.0f, 3.0f, 4.0f};
    B = b_vals;

    EXPECT_THROW(math::matmul<float>(A, B), std::runtime_error);
}

/**
 * @test MATMUL.inf_throws
 * @brief matmul should throw std::runtime_error if result is non-finite (Inf).
 */
TEST(MATMUL, inf_throws)
{
    Tensor<float> A({1, 2}, MemoryLocation::DEVICE);
    std::vector<float> a_vals = {std::numeric_limits<float>::infinity(), 2.0f};
    A = a_vals;

    Tensor<float> B({2, 1}, MemoryLocation::DEVICE);
    std::vector<float> b_vals = {3.0f, 4.0f};
    B = b_vals;

    EXPECT_THROW(math::matmul<float>(A, B), std::runtime_error);
}

} // namespace Test