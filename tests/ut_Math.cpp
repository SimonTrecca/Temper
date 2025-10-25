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
 * @brief Batched matmul with broadcasting: A:(2,1,2,3) * B:(1,3,3,2)
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

/**
 * @test RESHAPE.reshape_free_function_basic
 * @brief Free reshape() creates an owning clone, reshapes its dimensions,
 * and leaves the source tensor unchanged.
 */
TEST(RESHAPE, reshape_free_function_basic)
{
    Tensor<float> src({2,3}, MemoryLocation::HOST);
    src = {1,2,3,4,5,6};

    Tensor<float> r = math::reshape(src, {3,2});

    EXPECT_EQ(src.get_dimensions(), (std::vector<uint64_t>{2,3}));

    EXPECT_EQ(r.get_dimensions(), (std::vector<uint64_t>{3,2}));

    for (uint64_t i = 0; i < src.get_num_elements(); ++i)
    {
        EXPECT_FLOAT_EQ(r.get_data()[i], src.get_data()[i]);
    }

    EXPECT_NE(r.get_data(), src.get_data());
}

/**
 * @test RESHAPE.reshape_free_function_flat
 * @brief Reshaping to 1D with free reshape() preserves contents.
 */
TEST(RESHAPE, reshape_free_function_flat)
{
    Tensor<float> src({2,3}, MemoryLocation::HOST);
    src = {10,11,12,13,14,15};

    Tensor<float> r = math::reshape(src, {6});

    EXPECT_EQ(r.get_dimensions(), (std::vector<uint64_t>{6}));
    for (uint64_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(r[i], src.get_data()[i]);
    }
}

/**
 * @test RESHAPE.reshape_free_function_invalid_size
 * @brief Free reshape() must throw if new dimensions don't match element count.
 */
TEST(RESHAPE, reshape_free_function_invalid_size)
{
    Tensor<float> src({2,3}, MemoryLocation::HOST);
    src = {1,2,3,4,5,6};

    EXPECT_THROW(
        { auto r = math::reshape(src, {4,2}); },
        std::invalid_argument
    );
}

/**
 * @test RESHAPE.reshape_free_function_from_view
 * @brief Reshaping a view via free reshape() is valid (clone first),
 * produces an owning tensor.
 */
TEST(RESHAPE, reshape_free_function_from_view)
{
    Tensor<float> base({4}, MemoryLocation::HOST);
    base = {0,1,2,3};

    Tensor<float> v(base, {0}, {4}, {1});
    EXPECT_FALSE(v.get_owns_data());

    Tensor<float> r = math::reshape(v, {2,2});

    EXPECT_TRUE(r.get_owns_data());
    EXPECT_EQ(r.get_dimensions(), (std::vector<uint64_t>{2,2}));

    for (uint64_t i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(r.get_data()[i], base.get_data()[i]);
    }
}

/**
 * @test RESHAPE.reshape_free_function_empty_tensor
 * @brief Reshaping an empty tensor through free reshape() throws.
 */
TEST(RESHAPE, reshape_free_function_empty_tensor)
{
    Tensor<float> empty;
    EXPECT_THROW(
        { auto r = math::reshape(empty, {1}); },
        std::invalid_argument
    );
}

/**
 * @test SORT.sort_function_independence
 * @brief Free function sort should not mutate the input tensor.
 */
TEST(SORT, sort_function_independence)
{
    Tensor<float> t({5}, MemoryLocation::HOST);
    std::vector<float> vals = {3,1,2,5,4};
    t = vals;

    Tensor<float> out = math::sort(t, 0);

    for (size_t i = 0; i < vals.size(); i++)
    {
        EXPECT_FLOAT_EQ(t[i], vals[i]);
    }

    std::vector<float> expected = {1,2,3,4,5};
    for (size_t i = 0; i < vals.size(); i++)
    {
        EXPECT_FLOAT_EQ(out[i], expected[i]);
    }
}

/**
 * @test SORT.sort_function_axis1
 * @brief Sorting a 2D tensor along axis 1 via free function.
 */
TEST(SORT, sort_function_axis1)
{
    Tensor<float> t({2,3}, MemoryLocation::HOST);
    // [[3,1,2],
    //  [0,-1,5]]
    t = std::vector<float>{3,1,2, 0,-1,5};

    Tensor<float> out = math::sort(t, 1);

    EXPECT_FLOAT_EQ(out[0][0], 1.0f);
    EXPECT_FLOAT_EQ(out[0][1], 2.0f);
    EXPECT_FLOAT_EQ(out[0][2], 3.0f);
    EXPECT_FLOAT_EQ(out[1][0], -1.0f);
    EXPECT_FLOAT_EQ(out[1][1], 0.0f);
    EXPECT_FLOAT_EQ(out[1][2], 5.0f);

    EXPECT_FLOAT_EQ(t[0][0], 3.0f);
    EXPECT_FLOAT_EQ(t[0][1], 1.0f);
    EXPECT_FLOAT_EQ(t[0][2], 2.0f);
}

/**
 * @test SORT.sort_function_axis_out_of_bounds
 * @brief Free function sort should throw for invalid axis.
 */
TEST(SORT, sort_function_axis_out_of_bounds)
{
    Tensor<float> t({3}, MemoryLocation::HOST);
    EXPECT_THROW(math::sort(t, 1), std::invalid_argument);
    EXPECT_THROW(math::sort(t, -2), std::invalid_argument);
}

/**
 * @test SUM.sum_all_elements
 * @brief Sum all elements (axis = -1) on a device tensor and return
 * a scalar with the correct total value.
 */
TEST(SUM, sum_all_elements)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    t = vals;

    Tensor<float> res = math::sum(t);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host[0], 6.0f);
}

/**
 * @test SUM.sum_axis0
 * @brief Sum along axis 0 for a 2x3 tensor stored on device
 * nd verify per-column sums.
 */
TEST(SUM, sum_axis0)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = math::sum(t, 0);

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f + 5.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f + 6.0f);
}

/**
 * @test SUM.sum_axis1
 * @brief Sum along axis 1 for a 2x3 device tensor and verify per-row sums.
 */
TEST(SUM, sum_axis1)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = math::sum(t, 1);

    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 2 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 2.0f + 3.0f);
    EXPECT_FLOAT_EQ(host[1], 4.0f + 5.0f + 6.0f);
}

/**
 * @test SUM.sum_axis0_3D
 * @brief Sum along axis 0 for a 2x2x2 device tensor and verify resulting values.
 */
TEST(SUM, sum_axis0_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::sum(t, 0);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 5.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f + 6.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f + 7.0f);
    EXPECT_FLOAT_EQ(host[3], 4.0f + 8.0f);
}

/**
 * @test SUM.sum_axis_negative
 * @brief Sum along axis -3 for a 2x2x2 device tensor and verify resulting values.
 */
TEST(SUM, sum_axis_negative)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::sum(t, -3);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 5.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f + 6.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f + 7.0f);
    EXPECT_FLOAT_EQ(host[3], 4.0f + 8.0f);
}

/**
 * @test SUM.sum_axis1_3D
 * @brief Sum along axis 1 for a 2x2x2 device tensor and verify resulting values.
 */
TEST(SUM, sum_axis1_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::sum(t, 1);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 3.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[2], 5.0f + 7.0f);
    EXPECT_FLOAT_EQ(host[3], 6.0f + 8.0f);
}

/**
 * @test SUM.sum_axis2_3D
 * @brief Sum along axis 2 for a 2x2x2 device tensor and verify resulting values.
 */
TEST(SUM, sum_axis2_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::sum(t, 2);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 2.0f);
    EXPECT_FLOAT_EQ(host[1], 3.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[2], 5.0f + 6.0f);
    EXPECT_FLOAT_EQ(host[3], 7.0f + 8.0f);
}

/**
 * @test SUM.sum_view_tensor
 * @brief Sum all elements (axis = -1) of a view into a device tensor and
 * verify the scalar result.
 */
TEST(SUM, sum_view_tensor)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 0ull};
    std::vector<uint64_t> view_shape = {3ull};

    Tensor<float> view(t, start_indices, view_shape);

    Tensor<float> res = math::sum(view);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 2.0f + 3.0f);
}

/**
 * @test SUM.sum_alias_view_tensor
 * @brief Sum all elements (axis = -1) of an alias view
 * with non-unit stride and verify result.
 */
TEST(SUM, sum_alias_view_tensor)
{
    Tensor<float> t({6}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {0ull};
    std::vector<uint64_t> dims = {3ull};
    std::vector<uint64_t> strides = {2ull};

    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = math::sum(alias_view);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 3.0f + 5.0f);
}

/**
 * @test SUM.sum_view_tensor_3d_axis1
 * @brief Sum along axis 1 on a 3D view and verify the produced values.
 */
TEST(SUM, sum_view_tensor_3d_axis1)
{
    Tensor<float> t({3, 4, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 1ull, 0ull};
    std::vector<uint64_t> view_shape    = {2ull, 2ull};
    Tensor<float> view(t, start_indices, view_shape);

    Tensor<float> res = math::sum(view, 1);

    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0], 23.0f);
    EXPECT_FLOAT_EQ(host[1], 27.0f);
}

/**
 * @test SUM.sum_alias_view_tensor_2d_strided
 * @brief Sum along axis 0 on a 2D alias view with custom strides and verify
 * each output element.
 */
TEST(SUM, sum_alias_view_tensor_2d_strided)
{
    Tensor<float> t({4, 5}, MemoryLocation::DEVICE);
    std::vector<float> vals(20);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 1ull};
    std::vector<uint64_t> dims          = {2ull, 3ull};
    std::vector<uint64_t> strides       = {5ull, 2ull};
    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = math::sum(alias_view, 0);

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0], 9.0f);
    EXPECT_FLOAT_EQ(host[1], 13.0f);
    EXPECT_FLOAT_EQ(host[2], 17.0f);
}

/**
 * @test SUM.sum_alias_view_tensor_overlapping_stride_zero
 * @brief Sum along axis 0 on an alias view that contains overlapping elements
 * via a zero stride and verify the sums account for repeated elements.
 */
TEST(SUM, sum_alias_view_tensor_overlapping_stride_zero)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 0ull};
    std::vector<uint64_t> dims          = {2ull, 2ull};
    std::vector<uint64_t> strides       = {0ull, 1ull};
    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = math::sum(alias_view, 0);

    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0], 8.0f);
    EXPECT_FLOAT_EQ(host[1], 10.0f);
}

/**
 * @test SUM.sum_nan_throws
 * @brief Tests that sum throws std::runtime_error
 * when the tensor contains NaN values.
 */
TEST(SUM, sum_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals =
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    t = vals;

    EXPECT_THROW(math::sum(t, -1), std::runtime_error);
}

/**
 * @test SUM.sum_non_finite_throws
 * @brief Tests that sum throws std::runtime_error when
 * the tensor contains non-finite values (infinity).
 */
TEST(SUM, sum_non_finite_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {std::numeric_limits<float>::infinity(), 1.0f};
    t = vals;

    EXPECT_THROW(math::sum(t), std::runtime_error);
}

/**
 * @test SUM.sum_empty
 * @brief Summing an empty tensor returns a scalar tensor containing 0.0.
 */
TEST(SUM, sum_empty)
{
    Tensor<float> t;

    Tensor<float> res({1}, MemoryLocation::DEVICE);
    res = math::sum(t);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 0.0f);
}

/**
 * @test CUMSUM.cumsum_all_elements_flatten
 * @brief Tests cumsum on a 1D tensor, flattening all elements.
 */
TEST(CUMSUM, cumsum_all_elements_flatten)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    t = vals;

    Tensor<float> res = t.cumsum();

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 1.0f + 2.0f);
    EXPECT_FLOAT_EQ(host[2], 1.0f + 2.0f + 3.0f);
}

/**
 * @test CUMSUM.cumsum_axis0_2D
 * @brief Tests cumsum along axis 0 of a 2D tensor.
 */
TEST(CUMSUM, cumsum_axis0_2D)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = math::cumsum(t, 0);

    std::vector<float> host(6);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 6 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f);
    EXPECT_FLOAT_EQ(host[3], 1.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[4], 2.0f + 5.0f);
    EXPECT_FLOAT_EQ(host[5], 3.0f + 6.0f);
}

/**
 * @test CUMSUM.cumsum_axis_negative
 * @brief Tests cumsum along axis -2 of a 2D tensor.
 */
TEST(CUMSUM, cumsum_axis_negative)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = math::cumsum(t, -2);

    std::vector<float> host(6);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 6 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f);
    EXPECT_FLOAT_EQ(host[3], 1.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[4], 2.0f + 5.0f);
    EXPECT_FLOAT_EQ(host[5], 3.0f + 6.0f);
}

/**
 * @test CUMSUM.cumsum_axis1_2D
 * @brief Tests cumsum along axis 1 of a 2D tensor.
 */
TEST(CUMSUM, cumsum_axis1_2D)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = math::cumsum(t, 1);

    std::vector<float> host(6);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 6 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 3.0f);
    EXPECT_FLOAT_EQ(host[2], 6.0f);
    EXPECT_FLOAT_EQ(host[3], 4.0f);
    EXPECT_FLOAT_EQ(host[4], 9.0f);
    EXPECT_FLOAT_EQ(host[5], 15.0f);
}

/**
 * @test CUMSUM.cumsum_flatten_3D
 * @brief Tests cumsum on a 3D tensor flattened along the last axis.
 */
TEST(CUMSUM, cumsum_flatten_3D)
{
    Tensor<float> t({2,2,2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f};
    t = vals;
    Tensor<float> res = math::cumsum(t);

    std::vector<float> host(8);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 8 * sizeof(float)).wait();

    std::vector<float> expected =
        {1.0f, 1.0f+2.0f, 1.0f+2.0f+3.0f, 1.0f+2.0f+3.0f+4.0f,
        1.0f+2.0f+3.0f+4.0f+5.0f, 1.0f+2.0f+3.0f+4.0f+5.0f+6.0f,
        1.0f+2.0f+3.0f+4.0f+5.0f+6.0f+7.0f,
        1.0f+2.0f+3.0f+4.0f+5.0f+6.0f+7.0f+8.0f};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test CUMSUM.cumsum_view_flatten
 * @brief Tests cumsum on a view of a 3D tensor flattened along the last axis.
 */
TEST(CUMSUM, cumsum_view_flatten)
{
    Tensor<float> t({3,4,2}, MemoryLocation::DEVICE);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start = {1ull, 1ull, 0ull};
    std::vector<uint64_t> view_shape = {2ull, 2ull};
    Tensor<float> view(t, start, view_shape);

    Tensor<float> res = math::cumsum(view);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 11.0f);
    EXPECT_FLOAT_EQ(host[1], 23.0f);
    EXPECT_FLOAT_EQ(host[2], 36.0f);
    EXPECT_FLOAT_EQ(host[3], 50.0f);
}

/**
 * @test CUMSUM.cumsum_alias_view_strided
 * @brief Tests cumsum on an alias view with a stride on a 1D tensor.
 */
TEST(CUMSUM, cumsum_alias_view_strided)
{
    Tensor<float> t({6}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t = vals;
    std::vector<uint64_t> start = {0ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> alias_view(t, start, dims, strides);

    Tensor<float> res = math::cumsum(alias_view);

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 1.0f + 3.0f);
    EXPECT_FLOAT_EQ(host[2], 1.0f + 3.0f + 5.0f);
}

/**
 * @test CUMSUM.cumsum_alias_view_overlapping_stride_zero
 * @brief Tests cumsum on an alias view with
 * overlapping stride of zero on a 2D tensor.
 */
TEST(CUMSUM, cumsum_alias_view_overlapping_stride_zero)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start   = {1ull, 0ull};
    std::vector<uint64_t> dims    = {2ull, 2ull};
    std::vector<uint64_t> strides = {0ull, 1ull};
    Tensor<float> alias_view(t, start, dims, strides);

    Tensor<float> res = math::cumsum(alias_view, 0);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 4.0f);
    EXPECT_FLOAT_EQ(host[1], 5.0f);
    EXPECT_FLOAT_EQ(host[2], 8.0f);
    EXPECT_FLOAT_EQ(host[3], 10.0f);
}

/**
 * @test CUMSUM.cumsum_alias_view_weird_strides
 * @brief Sorting an alias view with non-trivial strides (e.g. 13,4).
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Cumsum along axis 1.
 */
TEST(CUMSUM, cumsum_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::HOST);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});

    Tensor<float> view2 = math::cumsum(view, 1);
    EXPECT_EQ(view2.m_dimensions, (std::vector<uint64_t>{3,4}));
    EXPECT_EQ(view2.m_strides, (std::vector<uint64_t>{4,1}));

    Tensor<float> host = view2.clone();

    std::vector<float> out(12);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float)*12).wait();

    std::vector<float> expected =
    {
        0.f,  4.f,  12.f,  24.f,
        13.f, 30.f, 51.f, 76.f,
        26.f, 56.f, 90.f, 128.f
    };

    for (uint64_t k = 0; k < out.size(); ++k)
    {
        EXPECT_FLOAT_EQ(out[k], expected[k]);
    }
}

/**
 * @test CUMSUM.cumsum_axis_out_of_bounds
 * @brief Tests that cumsum throws std::invalid_argument
 * when the axis is out of bounds.
 */
TEST(CUMSUM, cumsum_axis_out_of_bounds)
{
    Tensor<float> t({2,2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
    t = vals;

    EXPECT_THROW(math::cumsum(t, 2), std::invalid_argument);
    EXPECT_THROW(math::cumsum(t, -3), std::invalid_argument);
}

/**
 * @test CUMSUM.cumsum_nan_throws
 * @brief Tests that cumsum throws std::runtime_error
 * when the tensor contains NaN values.
 */
TEST(CUMSUM, cumsum_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals =
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    t = vals;

    EXPECT_THROW(math::cumsum(t, -1), std::runtime_error);
}

/**
 * @test CUMSUM.cumsum_non_finite_throws
 * @brief Tests that cumsum throws std::runtime_error when
 * the tensor contains non-finite values (infinity).
 */
TEST(CUMSUM, cumsum_non_finite_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {std::numeric_limits<float>::infinity(), 1.0f};
    t = vals;

    EXPECT_THROW(math::cumsum(t, -1), std::runtime_error);
}

/**
 * @test CUMSUM.cumsum_empty
 * @brief Tests cumsum on an empty tensor returns a tensor
 * with a single zero element.
 */
TEST(CUMSUM, cumsum_empty)
{
    Tensor<float> t;

    Tensor<float> res({1}, MemoryLocation::DEVICE);
    res = math::cumsum(t, -1);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 0.0f);
}

/**
 * @test TRANSPOSE.transpose_noargs_reverse_axes
 * @brief Tests that transpose() with no arguments reverses all axes.
 */
TEST(TRANSPOSE, transpose_noargs_reverse_axes)
{
    Tensor<float> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    t = vals;

    Tensor<float> t_rev = math::transpose(t);

    EXPECT_EQ(t_rev.m_dimensions, (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(t_rev.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1], t.m_strides[0]}));

    Tensor<float> host = t_rev.clone();

    std::vector<float> out(24);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 24).wait();

    EXPECT_FLOAT_EQ(out[0], vals[0]);
    EXPECT_FLOAT_EQ(out[23], vals[23]);
}

/**
 * @test TRANSPOSE.transpose_explicit_axes
 * @brief Tests transpose with explicit axis permutation.
 */
TEST(TRANSPOSE, transpose_explicit_axes)
{
    Tensor<float> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    t = vals;

    Tensor<float> perm = math::transpose(t, {2, 1, 0});

    EXPECT_EQ(perm.m_dimensions, (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(perm.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1], t.m_strides[0]}));

    Tensor<float> host = perm.clone();
    std::vector<float> out(24);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 24).wait();

    EXPECT_FLOAT_EQ(out[0], vals[0]);
    EXPECT_FLOAT_EQ(out[23], vals[23]);
}

/**
 * @test TRANSPOSE.transpose_explicit_axes_negative
 * @brief Tests transpose with explicit negative axis permutation.
 */
TEST(TRANSPOSE, transpose_explicit_axes_negative)
{
    Tensor<float> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    t = vals;

    Tensor<float> perm = math::transpose(t, {-1, 1, -3});

    EXPECT_EQ(perm.m_dimensions, (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(perm.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1], t.m_strides[0]}));

    Tensor<float> host = perm.clone();
    std::vector<float> out(24);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 24).wait();

    EXPECT_FLOAT_EQ(out[0], vals[0]);
    EXPECT_FLOAT_EQ(out[23], vals[23]);
}

/**
 * @test TRANSPOSE.transpose_2d
 * @brief Tests transpose on a 2D tensor (matrix).
 */
TEST(TRANSPOSE, transpose_2d)
{
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    t = {1,2,3,4,5,6};

    Tensor<float> t_T = math::transpose(t);

    EXPECT_EQ(t_T.m_dimensions, (std::vector<uint64_t>{3,2}));
    EXPECT_EQ(t_T.m_strides,
        (std::vector<uint64_t>{t.m_strides[1], t.m_strides[0]}));

    Tensor<float> host = t_T.clone();
    std::vector<float> out(6);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 6).wait();

    EXPECT_FLOAT_EQ(out[0], 1.f);
    EXPECT_FLOAT_EQ(out[1], 4.f);
    EXPECT_FLOAT_EQ(out[2], 2.f);
    EXPECT_FLOAT_EQ(out[3], 5.f);
    EXPECT_FLOAT_EQ(out[4], 3.f);
    EXPECT_FLOAT_EQ(out[5], 6.f);
}

/**
 * @test TRANSPOSE.transpose_mutation_reflects
 * @brief Ensure that modifying the transposed alias updates the original tensor.
 */
TEST(TRANSPOSE, transpose_mutation_reflects)
{
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    t = {0,1,2,3,4,5};

    Tensor<float> t_T = math::transpose(t);
    t_T[0][0] = 100.f;
    t_T[2][1] = 200.f;

    Tensor<float> host = t.clone();
    std::vector<float> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*6).wait();

    EXPECT_FLOAT_EQ(out[0], 100.f);
    EXPECT_FLOAT_EQ(out[5], 200.f);
}

/**
 * @test TRANSPOSE.transpose_invalid_axes
 * @brief Transpose throws when axes permutation is invalid.
 */
TEST(TRANSPOSE, transpose_invalid_axes)
{
    Tensor<float> t({2,3,4}, MemoryLocation::HOST);
    t = std::vector<float>(24, 1.f);

    EXPECT_THROW(math::transpose(t, {0,1}), std::invalid_argument);

    EXPECT_THROW(math::transpose(t, {0,1,1}), std::invalid_argument);

    EXPECT_THROW(math::transpose(t, {0,1,3}), std::invalid_argument);
}

/**
 * @test TRANSPOSE.transpose_1d
 * @brief Transpose a 1D tensor should return a 1D alias (no change).
 */
TEST(TRANSPOSE, transpose_1d)
{
    Tensor<float> t({5}, MemoryLocation::HOST);
    t = {0,1,2,3,4};

    Tensor<float> t_tr = math::transpose(t);
    EXPECT_EQ(t_tr.m_dimensions, t.m_dimensions);
    EXPECT_EQ(t_tr.m_strides, t.m_strides);

    Tensor<float> host = t_tr.clone();
    std::vector<float> out(5);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*5).wait();

    for (uint64_t i = 0; i < 5; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i));
    }
}

/**
 * @test TRANSPOSE.transpose_empty
 * @brief Transpose of an empty tensor throws.
 */
TEST(TRANSPOSE, transpose_empty)
{
    Tensor<float> t;
    EXPECT_THROW(math::transpose(t), std::runtime_error);
}

/**
 * @test PAD.pad_correct_result_shape
 * @brief Verify output dimensions equal input dims plus specified paddings.
 */
TEST(PAD, pad_correct_result_shape)
{
    Tensor<float> t({10, 10});

    Tensor<float> result = math::pad(t, 1, 2, 3, 4, 0.0f);
    const std::vector<uint64_t> res_shape = result.get_dimensions();
    EXPECT_EQ(res_shape[0], 13);
    EXPECT_EQ(res_shape[1], 17);
}

/**
 * @test PAD.pad_values_are_correct
 * @brief Pad a 2x2 tensor with zeros on all sides and check element positions.
 */
TEST(PAD, pad_values_are_correct)
{
    Tensor<float> t({2,2});
    std::vector<float> host_vals = { 1.0f, 2.0f,
                           3.0f, 4.0f };
    t = host_vals;

    Tensor<float> out = math::pad(t, 1, 1, 1, 1, 0.0f);
    EXPECT_EQ(out.get_dimensions()[0], 4u);
    EXPECT_EQ(out.get_dimensions()[1], 4u);

    std::vector<float> expected = {
        0, 0, 0, 0,
        0, 1, 2, 0,
        0, 3, 4, 0,
        0, 0, 0, 0
    };
    for (uint64_t r = 0; r < 4; ++r)
    {
        for (uint64_t c = 0; c < 4; ++c)
        {
            EXPECT_EQ(out[r][c], expected[r*4 + c]);
        }
    }
}

/**
 * @test PAD.pad_asymmetric_positions
 * @brief Verify asymmetric top/bottom/left/right paddings place original
 * elements correctly and fill others with the given value.
 */
TEST(PAD, pad_asymmetric_positions)
{
    Tensor<float> t({2, 2});
    t = std::vector<float>{ 1.0f, 2.0f,
                            3.0f, 4.0f };

    const uint64_t top    = 3;
    const uint64_t bottom = 4;
    const uint64_t left   = 1;
    const uint64_t right  = 2;
    const float fill      = 0.0f;

    Tensor<float> out = math::pad(t, top, bottom, left, right, fill);

    const uint64_t out_rows = top + 2 + bottom;
    const uint64_t out_cols = left + 2 + right;

    const std::vector<uint64_t> dims = out.get_dimensions();
    EXPECT_EQ(dims[0], out_rows);
    EXPECT_EQ(dims[1], out_cols);

    for (uint64_t r = 0; r < out_rows; ++r)
    {
        for (uint64_t c = 0; c < out_cols; ++c)
        {
            const bool in_orig_r = (r >= top) && (r < top + 2);
            const bool in_orig_c = (c >= left) && (c < left + 2);

            if (in_orig_r && in_orig_c)
            {
                const uint64_t or_r = r - top;
                const uint64_t or_c = c - left;
                const float expected = t[or_r][or_c];
                EXPECT_FLOAT_EQ(out[r][c], expected)
                    << "orig mismatch at [" << r << "][" << c << "]";
            }
            else
            {
                EXPECT_FLOAT_EQ(out[r][c], fill)
                    << "pad mismatch at [" << r << "][" << c << "]";
            }
        }
    }
}

/**
 * @test PAD.pad_nonzero_fill_value
 * @brief Ensure non-zero fill value is used and the original element
 * appears at the expected coordinates.
 */
TEST(PAD, pad_nonzero_fill_value)
{
    Tensor<float> t({1, 1});
    t = std::vector<float>{ 9.0f };

    const uint64_t top    = 1;
    const uint64_t bottom = 0;
    const uint64_t left   = 2;
    const uint64_t right  = 1;
    const float fill      = -7.5f;

    Tensor<float> out = math::pad(t, top, bottom, left, right, fill);

    EXPECT_EQ(out.get_dimensions()[0], top + 1 + bottom);
    EXPECT_EQ(out.get_dimensions()[1], left + 1 + right);

    EXPECT_FLOAT_EQ(out[top][left], 9.0f);

    for (uint64_t r = 0; r < out.get_dimensions()[0]; ++r)
    {
        for (uint64_t c = 0; c < out.get_dimensions()[1]; ++c)
        {
            if (r == top && c == left) continue;
            EXPECT_FLOAT_EQ(out[r][c], fill);
        }
    }
}

/**
 * @test PAD.pad_on_view_keeps_positions_and_owner
 * @brief Padding a view copies the correct elements from the owner and
 * leaves the owner tensor unchanged.
 */
TEST(PAD, pad_on_view_keeps_positions_and_owner)
{
    Tensor<float> owner({4, 4});
    std::vector<float> vals;
    vals.reserve(16);
    for (int i = 0; i < 16; ++i)
    {
        vals.push_back(static_cast<float>(i + 1));
    }
    owner = vals;

    std::vector<uint64_t> start = {1, 1};
    std::vector<uint64_t> shape = {2, 2};
    Tensor<float> view(owner, start, shape);

    const uint64_t top = 1, bottom = 0, left = 2, right = 1;
    const float fill = -1.0f;

    Tensor<float> out = math::pad(view, top, bottom, left, right, fill);

    EXPECT_EQ(out.get_dimensions()[0], top + shape[0] + bottom);
    EXPECT_EQ(out.get_dimensions()[1], left + shape[1] + right);

    const uint64_t R = out.get_dimensions()[0];
    const uint64_t C = out.get_dimensions()[1];

    for (uint64_t r = 0; r < R; ++r)
    {
        for (uint64_t c = 0; c < C; ++c)
        {
            const bool in_orig_r = (r >= top) && (r < top + shape[0]);
            const bool in_orig_c = (c >= left) && (c < left + shape[1]);
            if (in_orig_r && in_orig_c)
            {
                const uint64_t or_r = r - top;
                const uint64_t or_c = c - left;
                const uint64_t owner_r = start[0] + or_r;
                const uint64_t owner_c = start[1] + or_c;
                const float expected = owner[owner_r][owner_c];
                EXPECT_FLOAT_EQ(out[r][c], expected);
            }
            else
            {
                EXPECT_FLOAT_EQ(out[r][c], fill);
            }
        }
    }

    EXPECT_FLOAT_EQ(owner[1][1], 6.0f);
    EXPECT_FLOAT_EQ(owner[1][2], 7.0f);
    EXPECT_FLOAT_EQ(owner[2][1], 10.0f);
    EXPECT_FLOAT_EQ(owner[2][2], 11.0f);
}

/**
 * @test PAD.pad_on_alias_view_respects_strides
 * @brief Padding an alias view with custom strides should respect those
 * strides when mapping values into the output.
 */
TEST(PAD, pad_on_alias_view_respects_strides)
{
    Tensor<float> owner({3, 4});
    std::vector<float> vals;
    vals.reserve(12);
    for (int i = 0; i < 12; ++i)
    {
        vals.push_back(static_cast<float>(i + 1));
    }
    owner = vals;

    std::vector<uint64_t> start = {0, 0};
    std::vector<uint64_t> dims = {2, 2};
    std::vector<uint64_t> strides = {4, 2};

    Tensor<float> aview(owner, start, dims, strides);

    const uint64_t top = 0, bottom = 1, left = 1, right = 0;
    const float fill = 0.0f;

    Tensor<float> out = math::pad(aview, top, bottom, left, right, fill);

    EXPECT_EQ(out.get_dimensions()[0], top + dims[0] + bottom);
    EXPECT_EQ(out.get_dimensions()[1], left + dims[1] + right);
    const uint64_t R = out.get_dimensions()[0];
    const uint64_t C = out.get_dimensions()[1];

    for (uint64_t r = 0; r < R; ++r)
    {
        for (uint64_t c = 0; c < C; ++c)
        {
            const bool in_orig_r = (r >= top) && (r < top + dims[0]);
            const bool in_orig_c = (c >= left) && (c < left + dims[1]);
            if (in_orig_r && in_orig_c)
            {
                const uint64_t or_r = r - top;
                const uint64_t or_c = c - left;
                const float expected = aview[or_r][or_c];
                EXPECT_FLOAT_EQ(out[r][c], expected);
            }
            else
            {
                EXPECT_FLOAT_EQ(out[r][c], fill);
            }
        }
    }

    EXPECT_FLOAT_EQ(aview[0][0], 1.0f);
    EXPECT_FLOAT_EQ(aview[0][1], 3.0f);
    EXPECT_FLOAT_EQ(aview[1][0], 5.0f);
    EXPECT_FLOAT_EQ(aview[1][1], 7.0f);

    EXPECT_FLOAT_EQ(owner[0][0], 1.0f);
    EXPECT_FLOAT_EQ(owner[0][2], 3.0f);
    EXPECT_FLOAT_EQ(owner[1][0], 5.0f);
    EXPECT_FLOAT_EQ(owner[1][2], 7.0f);
}

/**
 * @test PAD.pad_4d_tensor_preserves_batches
 * @brief Pad a 4D tensor (batch0, batch1, height, width).
 *
 * Verifies that padding only affects the last two spatial dimensions,
 * that batch dimensions are preserved without reordering, and that
 * original values are copied in-place while padded regions are filled
 * with the specified value.
 */
TEST(PAD, pad_4d_tensor_preserves_batches)
{
    const uint64_t B0 = 2, B1 = 3, H = 2, W = 2;
    Tensor<float> t({B0, B1, H, W});

    std::vector<float> vals;
    vals.reserve(B0 * B1 * H * W);
    for (uint64_t i = 0; i < B0 * B1 * H * W; ++i)
    {
        vals.push_back(static_cast<float>(i + 1));
    }
    t = vals;

    const uint64_t top = 1, bottom = 1, left = 2, right = 1;
    const float fill = 0.5f;

    Tensor<float> out = math::pad(t, top, bottom, left, right, fill);

    const uint64_t out_H = top + H + bottom;
    const uint64_t out_W = left + W + right;

    const uint64_t out_elems = B0 * B1 * out_H * out_W;
    std::vector<float> expected(out_elems, fill);

    for (uint64_t b0 = 0; b0 < B0; ++b0)
    {
        for (uint64_t b1 = 0; b1 < B1; ++b1)
        {
            for (uint64_t r = 0; r < out_H; ++r)
            {
                for (uint64_t c = 0; c < out_W; ++c)
                {
                    const bool in_r = (r >= top) && (r < top + H);
                    const bool in_c = (c >= left) && (c < left + W);
                    uint64_t out_idx =
                        (((b0 * B1 + b1) * out_H + r) * out_W + c);
                    if (in_r && in_c)
                    {
                        const uint64_t in_r = r - top;
                        const uint64_t in_c = c - left;
                        uint64_t in_idx =
                            (((b0 * B1 + b1) * H + in_r) * W + in_c);
                        expected[out_idx] = vals[in_idx];
                    }
                    else
                    {
                        expected[out_idx] = fill;
                    }
                }
            }
        }
    }

    std::vector<float> host(out_elems);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        host.size() * sizeof(float)).wait();

    for (uint64_t i = 0; i < out_elems; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }

    const std::vector<uint64_t> dims = out.get_dimensions();
    EXPECT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], B0);
    EXPECT_EQ(dims[1], B1);
    EXPECT_EQ(dims[2], out_H);
    EXPECT_EQ(dims[3], out_W);
}

/**
 * @test PAD.pad_empty
 * @brief Calling pad on an empty tensor must throw std::invalid_argument.
 */
TEST(PAD, pad_empty)
{
    Tensor<float> t;
    EXPECT_THROW(math::pad(t, 1, 2, 3, 4, 0.0f), std::invalid_argument);
}

/**
 * @test PAD.pad_rank1
 * @brief Calling pad on a rank-1 tensor must throw std::invalid_argument.
 */
TEST(PAD, pad_rank1)
{
    Tensor<float> t ({10});
    EXPECT_THROW(math::pad(t, 1, 2, 3, 4, 0.0f), std::invalid_argument);
}

/**
 * @test PAD.pad_padwidth_overflow
 * @brief Passing pad widths that cause uint64_t overflow must throw
 * std::overflow_error.
 */
TEST(PAD, pad_padwidth_overflow)
{
    Tensor<float> t({10, 10});
    const uint64_t big = std::numeric_limits<uint64_t>::max();

    EXPECT_THROW(math::pad(t, 1, 2, big, 1, 0.0f), std::overflow_error);
}

/**
 * @test PAD.pad_padheight_overflow
 * @brief Passing pad heights that cause uint64_t overflow must throw
 * std::overflow_error.
 */
TEST(PAD, pad_padheight_overflow)
{
    Tensor<float> t({10, 10});
    const uint64_t big = std::numeric_limits<uint64_t>::max();

    EXPECT_THROW(math::pad(t, big, 1, 3, 4, 0.0f), std::overflow_error);
}

/**
 * @test PAD.pad_symmetric
 * @brief Symmetric padding helper pads both sides and preserves original
 * element ordering; verifies pad values and owner unchanged.
 */
TEST(PAD, pad_symmetric)
{
    Tensor<float> owner({2, 2});
    std::vector<float> vals = {1, 2, 3, 4};
    owner = vals;

    const uint64_t pad_h = 1;
    const uint64_t pad_w = 2;
    const float pad_val = 7.5f;

    Tensor<float> out = math::pad(owner, pad_h, pad_w, pad_val);

    EXPECT_EQ(out.get_dimensions()[0], pad_h + 2 + pad_h);
    EXPECT_EQ(out.get_dimensions()[1], pad_w + 2 + pad_w);

    const uint64_t R = out.get_dimensions()[0];
    const uint64_t C = out.get_dimensions()[1];

    for (uint64_t r = 0; r < R; ++r)
    {
        for (uint64_t c = 0; c < C; ++c)
        {
            const bool in_orig_r = (r >= pad_h) && (r < pad_h + 2);
            const bool in_orig_c = (c >= pad_w) && (c < pad_w + 2);
            if (in_orig_r && in_orig_c)
            {
                const uint64_t or_r = r - pad_h;
                const uint64_t or_c = c - pad_w;
                const float expected = owner[or_r][or_c];
                EXPECT_FLOAT_EQ(out[r][c], expected);
            }
            else
            {
                EXPECT_FLOAT_EQ(out[r][c], pad_val);
            }
        }
    }

    EXPECT_FLOAT_EQ(owner[0][0], 1.0f);
    EXPECT_FLOAT_EQ(owner[0][1], 2.0f);
    EXPECT_FLOAT_EQ(owner[1][0], 3.0f);
    EXPECT_FLOAT_EQ(owner[1][1], 4.0f);
}

/**
 * @test ARGMAX.argmax_flattened
 * @brief argmax with axis = -1 (flattened) returns index of global max,
 * validated via at().
 */
TEST(ARGMAX, argmax_flattened)
{
    Tensor<float> t({5}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 5.0f, 3.0f, 5.0f, 2.0f};
    t = vals;
    std::vector<uint64_t> res = math::argmax(t);

    ASSERT_EQ(res.size(), 1u);

    EXPECT_EQ(t.at(res[0]), 5.0f);
}

/**
 * @test ARGMAX.argmax_axis0_2d
 * @brief argmax along axis 0 of a 2x3 matrix (per-column argmax),
 * verified via at().
 */
TEST(ARGMAX, argmax_axis0_2d)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = {1.0f,4.0f,3.0f,
         2.0f,0.0f,5.0f};

    std::vector<uint64_t> res = math::argmax(t, 0);

    ASSERT_EQ(res.size(), 3u);
    for (uint64_t col = 0; col < 3; ++col)
    {
        uint64_t row = res[col];
        float val = t.at(row * 3 + col);
        EXPECT_FLOAT_EQ(val, std::max(t.at(col), t.at(3 + col)));
    }
}

/**
 * @test ARGMAX.argmax_axis1_2d
 * @brief argmax along axis 1 of a 2x3 matrix (per-row argmax),
 * verified via at().
 */
TEST(ARGMAX, argmax_axis1_2d)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = {1.0f,4.0f,3.0f,
         2.0f,0.0f,5.0f};

    std::vector<uint64_t> res = math::argmax(t, 1);

    ASSERT_EQ(res.size(), 2u);
    EXPECT_FLOAT_EQ(t.at(0*3 + res[0]), 4.0f);
    EXPECT_FLOAT_EQ(t.at(1*3 + res[1]), 5.0f);
}

/**
 * @test ARGMAX.argmax_axis_negative
 * @brief argmax along axis -1 of a 2x3 matrix (per-row argmax),
 * verified via at().
 */
TEST(ARGMAX, argmax_axis_negative)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = {1.0f,4.0f,3.0f,
         2.0f,0.0f,5.0f};

    std::vector<uint64_t> res = math::argmax(t, -1);

    ASSERT_EQ(res.size(), 2u);
    EXPECT_FLOAT_EQ(t.at(0*3 + res[0]), 4.0f);
    EXPECT_FLOAT_EQ(t.at(1*3 + res[1]), 5.0f);
}

/**
 * @test ARGMAX.argmax_tie_prefers_first
 * @brief argmax should prefer the first occurrence on ties.
 */
TEST(ARGMAX, argmax_tie_prefers_first)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {2.0f, 2.0f, 1.0f,
                               0.0f, 5.0f, 5.0f};
    t = vals;

    std::vector<uint64_t> res = math::argmax(t, 1);

    ASSERT_EQ(res.size(), 2u);
    EXPECT_EQ(res[0], 0u);
    EXPECT_EQ(res[1], 1u);
}

/**
 * @test ARGMAX.argmax_axis_out_of_range
 * @brief argmax should throw std::invalid_argument for invalid axes.
 */
TEST(ARGMAX, argmax_axis_out_of_range)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_THROW(math::argmax(t, 2), std::invalid_argument);
    EXPECT_THROW(math::argmax(t, -3), std::invalid_argument);
}

/**
 * @test ARGMAX.argmax_nan_throws
 * @brief argmax should throw std::runtime_error when input contains NaN.
 */
TEST(ARGMAX, argmax_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals =
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f};
    t = vals;

    EXPECT_THROW(math::argmax(t), std::runtime_error);
}

/**
 * @test ARGMAX.argmax_alias_view
 * @brief argmax on a 1D alias view (non-contiguous) returns index
 * relative to view, verified via at().
 */
TEST(ARGMAX, argmax_alias_view)
{
    Tensor<float> owner({6}, MemoryLocation::DEVICE);
    owner = {1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 5.0f};

    Tensor<float> v(owner, {1}, {3}, {2});

    std::vector<uint64_t> res = math::argmax(v);

    ASSERT_EQ(res.size(), 1u);
    EXPECT_FLOAT_EQ(v.at(res[0]), 6.0f);
    EXPECT_FLOAT_EQ(owner.at(3), 6.0f);
}

/**
 * @test ARGMAX.argmax_3d_view_flatten
 * @brief argmax on a 3D sub-tensor view (flattened) returns the correct index.
 */
TEST(ARGMAX, argmax_3d_view_flatten)
{
    Tensor<float> t({2,2,3}, MemoryLocation::DEVICE);
    t = {1,5,2,0,3,4,6,2,0,1,2,3};

    Tensor<float> v(t, {1,0,0}, {1,2,3}, {6,3,1});
    std::vector<uint64_t> res = math::argmax(v);

    ASSERT_EQ(res.size(), 1u);
    EXPECT_FLOAT_EQ(v.at(res[0]), 6.0f);
}

/**
 * @test ARGMAX.argmax_on_alias_view_strided
 * @brief argmax on a 1D alias view with non-unit stride returns index relative
 * to the view (checks correct handling of strides).
 */
TEST(ARGMAX, argmax_on_alias_view_strided)
{
    Tensor<float> owner({6}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 5.0f};
    owner = vals;

    std::vector<uint64_t> start = {1ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> v(owner, start, dims, strides);

    std::vector<uint64_t> res = math::argmax(v);

    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0], 1u);
}

/**
 * @test LINSPACE.scalar_endpoint_true
 * @brief Linspace with 1-element start/stop (shape {1}), endpoint=true,
 * produces expected evenly spaced values when inserting axis 0.
 */
TEST(LINSPACE, scalar_endpoint_true)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    Tensor<float> stop({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{0.0f};
    stop  = std::vector<float>{4.0f};

    const uint64_t num = 5;
    Tensor<float> out =
        math::linspace(start, stop, num, MemoryLocation::DEVICE, 0, true);

    const uint64_t total = out.get_num_elements();
    ASSERT_EQ(total, num);

    std::vector<float> host(total);
    g_sycl_queue.memcpy
        (host.data(), out.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t i = 0; i < num; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], static_cast<float>(i));
    }
}

/**
 * @test LINSPACE.num_one_returns_start_and_zero_step
 * @brief num == 1 should return start values and step tensor of zeros.
 */
TEST(LINSPACE, num_one_returns_start_and_zero_step)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    Tensor<float> stop({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{7.5f};
    stop  = std::vector<float>{123.4f};

    const uint64_t num = 1;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace
        (start, stop, num, MemoryLocation::DEVICE, 0, true, &step_out);

    std::vector<float> host_out(1);
    g_sycl_queue.memcpy
        (host_out.data(), out.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host_out[0], 7.5f);

    ASSERT_EQ(step_out.get_num_elements(), 1u);
    std::vector<float> host_step(1);
    g_sycl_queue.memcpy
        (host_step.data(), step_out.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host_step[0], 0.0f);
}

/**
 * @test LINSPACE.broadcast_2x1_and_1x3_axis2_endpoint_true
 * @brief Broadcast start {2,1} and stop {1,3} -> S_shape {2,3}.
 * Insert new axis at the end (axis = 2) with num = 3 and endpoint=true.
 * For each (a,b) the sequence is [a, (a+b)/2, b].
 */
TEST(LINSPACE, broadcast_2x1_and_1x3_axis2_endpoint_true)
{
    Tensor<float> start({2,1}, MemoryLocation::DEVICE);
    start = std::vector<float>{0.0f, 10.0f};

    Tensor<float> stop({1,3}, MemoryLocation::DEVICE);
    stop = std::vector<float>{3.0f, 4.0f, 5.0f};

    const uint64_t num = 3;
    Tensor<float> out =
        math::linspace(start, stop, num, MemoryLocation::DEVICE, 2, true);

    const uint64_t B0 = 2, B1 = 3, N = 3;
    ASSERT_EQ(out.get_num_elements(), B0 * B1 * N);

    std::vector<float> host(B0 * B1 * N);
    g_sycl_queue.memcpy
        (host.data(), out.m_p_data.get(), host.size() * sizeof(float)).wait();

    std::vector<float> expected;
    expected.reserve(B0 * B1 * N);

    const std::array<float, 2> start_vals = {0.0f, 10.0f};
    const std::array<float, 3> stop_vals  = {3.0f, 4.0f, 5.0f};

    for (uint64_t i = 0; i < B0; ++i)
    {
        for (uint64_t j = 0; j < B1; ++j)
        {
            float a = start_vals[i];
            float b = stop_vals[j];
            float step = (b - a) / static_cast<float>(num - 1);
            for (uint64_t p = 0; p < N; ++p)
            {
                expected.push_back(a + step * static_cast<float>(p));
            }
        }
    }

    for (size_t idx = 0; idx < host.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(host[idx], expected[idx]);
    }
}

/**
 * @test LINSPACE.linspace_axis_negative
 * @brief Broadcast start {2,1} and stop {1,3} -> S_shape {2,3}.
 * Insert new axis at the end (axis = -1) with num = 3 and endpoint=true.
 * For each (a,b) the sequence is [a, (a+b)/2, b].
 */
TEST(LINSPACE, linspace_axis_negative)
{
    Tensor<float> start({2,1}, MemoryLocation::DEVICE);
    start = std::vector<float>{0.0f, 10.0f};

    Tensor<float> stop({1,3}, MemoryLocation::DEVICE);
    stop = std::vector<float>{3.0f, 4.0f, 5.0f};

    const uint64_t num = 3;
    Tensor<float> out =
        math::linspace(start, stop, num, MemoryLocation::DEVICE, -1, true);

    const uint64_t B0 = 2, B1 = 3, N = 3;
    ASSERT_EQ(out.get_num_elements(), B0 * B1 * N);

    std::vector<float> host(B0 * B1 * N);
    g_sycl_queue.memcpy
        (host.data(), out.m_p_data.get(), host.size() * sizeof(float)).wait();

    std::vector<float> expected;
    expected.reserve(B0 * B1 * N);

    const std::array<float, 2> start_vals = {0.0f, 10.0f};
    const std::array<float, 3> stop_vals  = {3.0f, 4.0f, 5.0f};

    for (uint64_t i = 0; i < B0; ++i)
    {
        for (uint64_t j = 0; j < B1; ++j)
        {
            float a = start_vals[i];
            float b = stop_vals[j];
            float step = (b - a) / static_cast<float>(num - 1);
            for (uint64_t p = 0; p < N; ++p)
            {
                expected.push_back(a + step * static_cast<float>(p));
            }
        }
    }

    for (size_t idx = 0; idx < host.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(host[idx], expected[idx]);
    }
}

/**
 * @test LINSPACE.nan_inputs_throw
 * @brief If start or stop contains NaN the function should throw.
 */
TEST(LINSPACE, nan_inputs_throw)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    Tensor<float> stop({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{std::numeric_limits<float>::quiet_NaN()};
    stop  = std::vector<float>{1.0f};

    EXPECT_THROW(
        {
            auto r = math::linspace
                (start, stop, 3, MemoryLocation::DEVICE, 0, true);
        },
        std::runtime_error
    );
}

/**
 * @test LINSPACE.step_out_broadcast_matches_values
 * @brief When start {2,1} and stop {1,3} are broadcast to S_shape {2,3},
 * the step_out produced when passing step_out pointer must contain
 * the per-(i,j) steps (i.e. (b-a)/(num-1) for endpoint=true).
 */
TEST(LINSPACE, step_out_broadcast_matches_values)
{
    Tensor<float> start({2,1}, MemoryLocation::DEVICE);
    start = std::vector<float>{0.0f, 10.0f};

    Tensor<float> stop({1,3}, MemoryLocation::DEVICE);
    stop = std::vector<float>{3.0f, 4.0f, 5.0f};

    const uint64_t num = 3;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace
        (start, stop, num, MemoryLocation::DEVICE, 2, true, &step_out);

    ASSERT_EQ(step_out.get_num_elements(), 6u);

    std::vector<float> host_steps(6);
    g_sycl_queue.memcpy(host_steps.data(),
        step_out.m_p_data.get(), sizeof(float) * host_steps.size()).wait();

    std::vector<float> expected_steps;
    expected_steps.reserve(6);

    const std::array<float, 2> start_vals = {0.0f, 10.0f};
    const std::array<float, 3> stop_vals  = {3.0f, 4.0f, 5.0f};

    for (uint64_t i = 0; i < start_vals.size(); ++i) {
        for (uint64_t j = 0; j < stop_vals.size(); ++j) {
            float a = start_vals[i];
            float b = stop_vals[j];
            expected_steps.push_back((b - a) / static_cast<float>(num - 1));
        }
    }

    for (size_t idx = 0; idx < host_steps.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(host_steps[idx], expected_steps[idx]);
    }
}

/**
 * @test LINSPACE.axis_out_of_range_throws
 * @brief Axis values outside [ -out_rank, out_rank-1 ] must throw.
 */
TEST(LINSPACE, axis_out_of_range_throws)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    Tensor<float> stop({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{0.0f};
    stop  = std::vector<float>{1.0f};

    EXPECT_THROW(
        {
            math::linspace(start, stop, 3, MemoryLocation::DEVICE, 5, true);
        },
        std::invalid_argument
    );
}

/**
 * @test LINSPACE.inf_inputs_throw
 * @brief If start or stop contains +Inf/-Inf the function should report an error.
 */
TEST(LINSPACE, inf_inputs_throw)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    Tensor<float> stop({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{std::numeric_limits<float>::infinity()};
    stop  = std::vector<float>{1.0f};

    EXPECT_THROW(
        {
            auto r =
                math::linspace(start, stop, 3, MemoryLocation::DEVICE, 0, true);
        },
        std::runtime_error
    );
}

/**
 * @test LINSPACE.scalar_vs_array_broadcast_front_axis
 * @brief start shape {1} (value 2) and stop shape {2} (values {3,7}) broadcast
 * to S_shape {2}. Insert axis at front (axis=0) with num=5 -> out shape {5,2}.
 */
TEST(LINSPACE, scalar_vs_array_broadcast_front_axis)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{2.0f};

    Tensor<float> stop({2}, MemoryLocation::DEVICE);
    stop = std::vector<float>{3.0f, 7.0f};

    const uint64_t num = 5;
    Tensor<float> out =
        math::linspace(start, stop, num, MemoryLocation::DEVICE, 0, true);

    ASSERT_EQ(out.get_num_elements(), num * 2u);

    std::vector<float> host(num * 2);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * host.size()).wait();

    std::vector<float> expected;
    expected.reserve(num * 2);

    const std::array<float, 2> stop_vals = {3.0f, 7.0f};
    const float a = 2.0f;

    for (uint64_t i = 0; i < num; ++i) {
        for (uint64_t j = 0; j < stop_vals.size(); ++j) {
            float b = stop_vals[j];
            float step = (b - a) / static_cast<float>(num - 1);
            expected.push_back(a + step * static_cast<float>(i));
        }
    }

    for (size_t idx = 0; idx < host.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(host[idx], expected[idx]);
    }
}

/**
 * @test LINSPACE.num_zero_returns_scalar_like
 * @brief When num == 0 the function should return a scalar-like output (1
 * element) and a step_out tensor with 1 element. The returned value must be
 * finite (not NaN) and the step must be finite as well.
 */
TEST(LINSPACE, num_zero_returns_scalar_like)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    Tensor<float> stop({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{2.0f};
    stop  = std::vector<float>{3.0f};

    const uint64_t num = 0;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace(start, stop, num,
       MemoryLocation::DEVICE, 0, true, &step_out);

    ASSERT_EQ(out.get_num_elements(), 1u);
    ASSERT_EQ(step_out.get_num_elements(), 1u);

    std::vector<float> host_out(1), host_step(1);
    g_sycl_queue.memcpy(host_out.data(),
        out.m_p_data.get(), sizeof(float)).wait();
    g_sycl_queue.memcpy(host_step.data(),
        step_out.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FALSE(std::isnan(host_out[0]));
    EXPECT_TRUE(std::isfinite(host_step[0]));
}

/**
 * @test LINSPACE.scalar_endpoint_false
 * @brief Linspace with scalar start/stop, endpoint=false and num>1 should
 * produce evenly spaced values that exclude the stop value. Example:
 * start=0, stop=1, num=4 -> [0.0, 0.25, 0.50, 0.75].
 */
TEST(LINSPACE, scalar_endpoint_false)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    Tensor<float> stop({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{0.0f};
    stop  = std::vector<float>{1.0f};

    const uint64_t num = 4;
    Tensor<float> out = math::linspace(start, stop, num,
                                       MemoryLocation::DEVICE, 0, false);

    ASSERT_EQ(out.get_num_elements(), num);
    std::vector<float> host(num);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(), sizeof(float) * num).wait();

    EXPECT_FLOAT_EQ(host[0], 0.0f);
    EXPECT_FLOAT_EQ(host[1], 0.25f);
    EXPECT_FLOAT_EQ(host[2], 0.5f);
    EXPECT_FLOAT_EQ(host[3], 0.75f);
}

/**
 * @test LINSPACE.decreasing_range_step_negative
 * @brief Linspace must handle decreasing ranges correctly: when start > stop
 * the generated sequence must decrease and the produced step_out must be
 * negative. Example: start=5, stop=1, num=5 -> [5,4,3,2,1] and step = -1.
 */
TEST(LINSPACE, decreasing_range_step_negative)
{
    Tensor<float> start({1}, MemoryLocation::DEVICE);
    start = std::vector<float>{5.0f};
    Tensor<float> stop({1},  MemoryLocation::DEVICE);
    stop  = std::vector<float>{1.0f};
    const uint64_t num = 5;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace
        (start, stop, num, MemoryLocation::DEVICE, 0, true, &step_out);

    std::vector<float> host(out.get_num_elements());
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float)*host.size()).wait();
    for (uint64_t i = 0; i < num; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], 5.0f - static_cast<float>(i));
    }

    ASSERT_EQ(step_out.get_num_elements(), 1u);
    std::vector<float> hs(1);
    g_sycl_queue.memcpy(hs.data(),
        step_out.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(hs[0], -1.0f);
}

/**
 * @test LINSPACE.start_equals_stop_zero_step
 * @brief When start and stop are equal elementwise the output sequences must
 * contain the same constant value and the corresponding step_out entries must
 * be zero.
 */
TEST(LINSPACE, start_equals_stop_zero_step)
{
    Tensor<float> start({2,1}, MemoryLocation::DEVICE);
    start = std::vector<float>{2.0f, 2.0f};
    Tensor<float> stop({1,3}, MemoryLocation::DEVICE);
    stop = std::vector<float>{2.0f, 2.0f, 2.0f};
    const uint64_t num = 4;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace
        (start, stop, num, MemoryLocation::DEVICE, 2, true, &step_out);

    std::vector<float> host(out.get_num_elements());
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), host.size()*sizeof(float)).wait();
    for (auto v : host)
    {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }

    std::vector<float> hs(step_out.get_num_elements());
    g_sycl_queue.memcpy(hs.data(),
        step_out.m_p_data.get(), hs.size()*sizeof(float)).wait();
    for (auto s : hs)
    {
        EXPECT_FLOAT_EQ(s, 0.0f);
    }
}

/**
 * @test LINSPACE.view_constructor_broadcast
 * @brief Verify linspace when start and stop are constructed as alias/views
 * that require broadcasting. Ensures:
 *  - correct broadcasting of view-constructed start/stop to S_shape,
 *  - correct output values for each broadcasted pair,
 *  - step_out has one step per broadcasted element (shape == S_shape).
 */
TEST(LINSPACE, view_constructor_broadcast)
{
    Tensor<float> owner_start({2,2}, MemoryLocation::DEVICE);
    owner_start = std::vector<float>{0.0f, 999.0f, 10.0f, 999.0f};

    Tensor<float> start
        (owner_start, std::vector<uint64_t>{0,0}, std::vector<uint64_t>{2,1});

    Tensor<float> owner_stop({1,4}, MemoryLocation::DEVICE);
    owner_stop = std::vector<float>{3.0f, 4.0f, 5.0f, 999.0f};

    Tensor<float> stop
        (owner_stop, std::vector<uint64_t>{0,0}, std::vector<uint64_t>{1,3});

    const uint64_t num = 3;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace
        (start, stop, num, MemoryLocation::DEVICE, 2, true, &step_out);

    const uint64_t B0 = 2, B1 = 3, N = 3;
    ASSERT_EQ(out.get_num_elements(), B0 * B1 * N);
    ASSERT_EQ(step_out.get_num_elements(), B0 * B1);

    std::vector<float> host(out.get_num_elements());
    g_sycl_queue.memcpy
        (host.data(), out.m_p_data.get(), host.size() * sizeof(float)).wait();

    std::vector<float> host_steps(step_out.get_num_elements());
    g_sycl_queue.memcpy(host_steps.data(), step_out.m_p_data.get(),
        host_steps.size() * sizeof(float)).wait();

    const std::array<float,2> start_vals = {0.0f, 10.0f};
    const std::array<float,3> stop_vals  = {3.0f, 4.0f, 5.0f};

    std::vector<float> expected;
    expected.reserve(B0 * B1 * N);
    for (uint64_t i = 0; i < B0; ++i)
    {
        for (uint64_t j = 0; j < B1; ++j)
        {
            float a = start_vals[i];
            float b = stop_vals[j];
            float step = (b - a) / static_cast<float>(num - 1);
            for (uint64_t p = 0; p < N; ++p) {
                expected.push_back(a + step * static_cast<float>(p));
            }
        }
    }

    ASSERT_EQ(host.size(), expected.size());
    for (size_t k = 0; k < host.size(); ++k)
    {
        EXPECT_FLOAT_EQ(host[k], expected[k]);
    }

    std::vector<float> expected_steps;
    expected_steps.reserve(B0 * B1);
    for (uint64_t i = 0; i < start_vals.size(); ++i)
    {
        for (uint64_t j = 0; j < stop_vals.size(); ++j)
        {
            expected_steps.push_back
                ((stop_vals[j] - start_vals[i]) / static_cast<float>(num - 1));
        }
    }

    ASSERT_EQ(host_steps.size(), expected_steps.size());
    for (size_t k = 0; k < host_steps.size(); ++k)
    {
        EXPECT_FLOAT_EQ(host_steps[k], expected_steps[k]);
    }
}

/**
 * @test LINSPACE.alias_view_stride_every_other
 * @brief Linspace where `start` is an alias view that picks every-other
 * element (non-unit stride). Ensures correct reading of alias view values,
 * correct broadcasting with `stop`, correct output ordering and that step_out
 * contains the per-entry steps.
 */
TEST(LINSPACE, alias_view_stride_every_other)
{
    Tensor<float> owner({6}, MemoryLocation::DEVICE);
    owner = std::vector<float>{0.f,1.f,2.f,3.f,4.f,5.f};

    Tensor<float> start_alias(owner,
        std::vector<uint64_t>{0},
        std::vector<uint64_t>{3},
        std::vector<uint64_t>{2});

    Tensor<float> stop({3}, MemoryLocation::DEVICE);
    stop = std::vector<float>{10.0f, 20.0f, 30.0f};

    const uint64_t num = 2;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace
        (start_alias, stop, num, MemoryLocation::DEVICE, 0, true, &step_out);

    ASSERT_EQ(out.get_num_elements(), num * 3u);
    ASSERT_EQ(step_out.get_num_elements(), 3u);

    std::vector<float> host(out.get_num_elements());
    g_sycl_queue.memcpy
        (host.data(), out.m_p_data.get(), host.size() * sizeof(float)).wait();

    std::vector<float> host_steps(3);
    g_sycl_queue.memcpy(host_steps.data(),
        step_out.m_p_data.get(), sizeof(float) * host_steps.size()).wait();

    std::vector<float> expected = {0.0f, 2.0f, 4.0f, 10.0f, 20.0f, 30.0f};
    for (size_t k = 0; k < host.size(); ++k)
    {
        EXPECT_FLOAT_EQ(host[k], expected[k]);
    }

    std::vector<float> expected_steps = {10.0f, 18.0f, 26.0f};
    for (size_t k = 0; k < host_steps.size(); ++k)
    {
        EXPECT_FLOAT_EQ(host_steps[k], expected_steps[k]);
    }
}

/**
 * @test LINSPACE.scalar_overload_endpoint_true_with_step_out
 * @brief Scalar overload taking float start/stop with endpoint=true and step_out.
 * Should produce the evenly spaced sequence and the correct step value.
 */
TEST(LINSPACE, scalar_overload_endpoint_true_with_step_out)
{
    float start = 2.0f;
    float stop  = 6.0f;

    const uint64_t num = 5;
    Tensor<float> step_out;
    Tensor<float> out = math::linspace
        (start, stop, num, MemoryLocation::DEVICE, 0, true, &step_out);

    ASSERT_EQ(out.get_num_elements(), num);

    std::vector<float> host(num);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * num).wait();

    for (uint64_t i = 0; i < num; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], start + static_cast<float>(i));
    }

    ASSERT_EQ(step_out.get_num_elements(), 1u);
    std::vector<float> host_step(1);
    g_sycl_queue.memcpy(host_step.data(),
        step_out.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host_step[0], (stop - start) / static_cast<float>(num - 1));
}

/**
 * @test ARANGE.basic_positive_step
 * @brief arange with positive step generates the correct sequence.
 * Example: start=0, stop=5, step=1 -> [0,1,2,3,4]
 */
TEST(ARANGE, basic_positive_step)
{
    Tensor<float> out = math::arange(0.0f, 5.0f, 1.0f, MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 5u);

    std::vector<float> host(5);
    g_sycl_queue.memcpy(host.data(), out.get_data(), 5 * sizeof(float)).wait();

    std::vector<float> expected = {0,1,2,3,4};
    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test ARANGE.basic_negative_step
 * @brief arange with negative step generates the correct decreasing sequence.
 * Example: start=5, stop=0, step=-1 -> [5,4,3,2,1]
 */
TEST(ARANGE, basic_negative_step)
{
    Tensor<float> out = math::arange(5.0f, 0.0f, -1.0f, MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 5u);

    std::vector<float> host(5);
    g_sycl_queue.memcpy(host.data(), out.get_data(), 5 * sizeof(float)).wait();

    std::vector<float> expected = {5,4,3,2,1};
    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test ARANGE.zero_step_throws
 * @brief arange must throw when step is zero.
 */
TEST(ARANGE, zero_step_throws)
{
    EXPECT_THROW(
        {
            auto out = math::arange(0.0f, 5.0f, 0.0f, MemoryLocation::DEVICE);
        },
        std::invalid_argument
    );
}

/**
 * @test ARANGE.non_finite_inputs_throw
 * @brief arange must throw if start, stop or step is NaN or Inf.
 */
TEST(ARANGE, non_finite_inputs_throw)
{
    EXPECT_THROW(math::arange
        (std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f,
            MemoryLocation::DEVICE), std::runtime_error);
    EXPECT_THROW(math::arange
        (0.0f, std::numeric_limits<float>::infinity(), 1.0f,
            MemoryLocation::DEVICE), std::runtime_error);
    EXPECT_THROW(math::arange
        (0.0f, 1.0f, std::numeric_limits<float>::quiet_NaN(),
            MemoryLocation::DEVICE), std::runtime_error);
}

/**
 * @test ARANGE_stop_only
 * @brief arange(stop) variant generates 0..stop-1 sequence.
 * Example: stop=4 -> [0,1,2,3]
 */
TEST(ARANGE, stop_only)
{
    Tensor<float> out = math::arange(4.0f, MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 4u);

    std::vector<float> host(4);
    g_sycl_queue.memcpy(host.data(), out.get_data(), 4 * sizeof(float)).wait();

    std::vector<float> expected = {0,1,2,3};
    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test ARANGE_empty_result
 * @brief arange returns empty tensor with a single element set to 0
 * when range cannot produce elements.
 * Example: start=0, stop=0 or start=5, stop=0 with positive step.
 */
TEST(ARANGE, empty_result)
{
    Tensor<float> out1 = math::arange(0.0f, 0.0f, 1.0f, MemoryLocation::DEVICE);
    ASSERT_EQ(out1.get_num_elements(), 1u);
    ASSERT_EQ(out1[0], 0.0f);

    Tensor<float> out2 = math::arange(5.0f, 0.0f, 1.0f, MemoryLocation::DEVICE);
    ASSERT_EQ(out2.get_num_elements(), 1u);
    ASSERT_EQ(out2[0], 0.0f);

}

/**
 * @test ARANGE_floating_point_step
 * @brief arange with non-integer step produces correct values.
 * Example: start=0, stop=1, step=0.2 -> [0.0,0.2,0.4,0.6,0.8]
 */
TEST(ARANGE, floating_point_step)
{
    Tensor<float> out = math::arange(0.0f, 1.0f, 0.2f, MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 5u);

    std::vector<float> host(5);
    g_sycl_queue.memcpy(host.data(), out.get_data(), 5 * sizeof(float)).wait();

    std::vector<float> expected = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f};
    for (size_t i = 0; i < host.size(); ++i) {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test ZEROS.device
 * @brief zeros<float> returns a device tensor that is zero-initialized.
 */
TEST(ZEROS, device)
{
    std::vector<uint64_t> shape = {2, 3};
    Tensor<float> z = math::zeros<float>(shape, MemoryLocation::DEVICE);

    ASSERT_NE(z.m_p_data.get(), nullptr);

    uint64_t total = z.get_num_elements();

    std::vector<float> host(total, -1.0f);
    g_sycl_queue.memcpy(host.data(),
        z.m_p_data.get(), total * sizeof(float)).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], 0.0f);
    }
}

/**
 * @test ZEROS.host
 * @brief zeros<float> returns a host tensor that is zero-initialized.
 */
TEST(ZEROS, host)
{
    std::vector<uint64_t> shape = {5};
    Tensor<float> z = math::zeros<float>(shape, MemoryLocation::HOST);

    ASSERT_NE(z.m_p_data.get(), nullptr);

    uint64_t total = z.get_num_elements();

    std::vector<float> host(total, -1.0f);
    g_sycl_queue.memcpy(host.data(),
        z.m_p_data.get(), total * sizeof(float)).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], 0.0f);
    }
}

/**
 * @test INTEGRAL.polynomial_exact
 * @brief Integrate f(x) = x^2 on [0,1]. Exact result = 1/3.
 */
TEST(INTEGRAL, polynomial_exact)
{
    auto f = std::function<float(float)>([](float x) -> float { return x * x; });

    float a = 0.0f;
    float b = 1.0f;
    uint64_t n_bins = 4;

    float result = math::integral<float>(f, a, b, n_bins);

    const float expected = 1.0f / 3.0f;
    EXPECT_NEAR(result, expected, 1e-6f);
}

/**
 * @test INTEGRAL.convergence
 * @brief Check that the error for integrating sin(x) on [0, pi] decreases
 * overall as n_bins increases and final (finest) result
 * meets a reasonable tolerance.
 */
TEST(INTEGRAL, convergence)
{
    auto f = std::function<float(float)>
        ([](float x) -> float { return std::sin(x); });

    const float pi = 3.14159265358979323846f;
    const float a = 0.0f;
    const float b = pi;
    const float expected = 2.0f;

    std::vector<uint64_t> n_bins_list =
        {4u, 8u, 16u, 32u, 64u, 128u, 256u, 1024u, 4096u};

    std::vector<float> errors;
    errors.reserve(n_bins_list.size());

    for (uint64_t n_bins : n_bins_list)
    {
        float result = math::integral<float>(f, a, b, n_bins);
        float err = std::fabs(result - expected);
        errors.push_back(err);
    }

    EXPECT_GT(errors.front(), errors.back());

    const float finest_tolerance = 5e-6f;
    EXPECT_LE(errors.back(), finest_tolerance);


    const float reduction_factor = 10.0f;
    EXPECT_GE(errors.front() / (errors.back() + 1e-30f), reduction_factor);
}

/**
 * @test INTEGRAL.invalid_bins
 * @brief integral should throw when n_bins < 1
 */
TEST(INTEGRAL, invalid_bins)
{
    auto f = std::function<float(float)>([](float) -> float { return 1.0f; });

    EXPECT_THROW(math::integral<float>
        (f, 0.0f, 1.0f, 0), std::invalid_argument);
}

/**
 * @test INTEGRAL.reverse_interval
 * @brief Integrating over [1,0] should produce the negative of
 * integrating over [0,1].
 */
TEST(INTEGRAL, reverse_interval)
{
    auto f = std::function<float(float)>([](float x) -> float { return x * x; });

    const float a = 1.0f;
    const float b = 0.0f;
    const uint64_t n_bins = 8;

    const float result = math::integral<float>(f, a, b, n_bins);

    const float expected = - (1.0f / 3.0f);
    EXPECT_NEAR(result, expected, 1e-6f);
}

/**
 * @test FACTORIAL.basic_values
 * @brief factorial on a simple 1D tensor (0..5) returns expected values.
 */
TEST(FACTORIAL, basic_values)
{
    Tensor<float> t({6}, MemoryLocation::DEVICE);
    t = std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    Tensor<float> out = math::factorial(t);

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 6u);

    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    const std::vector<float> expected = {1.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0f};
    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(host[i], expected[i]);
    }
}

/**
 * @test FACTORIAL.alias_view_strided
 * @brief factorial on a 1D alias view with non-unit stride uses view indexing.
 */
TEST(FACTORIAL, alias_view_strided)
{
    Tensor<float> owner({6}, MemoryLocation::DEVICE);
    owner = std::vector<float>{1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 5.0f};

    std::vector<uint64_t> start = {1ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> v(owner, start, dims, strides);

    Tensor<float> out = math::factorial(v);

    ASSERT_EQ(out.get_num_elements(), 3u);
    std::vector<float> host(3);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();

    EXPECT_FLOAT_EQ(host[0], 6.0f);
    EXPECT_FLOAT_EQ(host[1], 720.0f);
    EXPECT_FLOAT_EQ(host[2], 120.0f);

    EXPECT_FLOAT_EQ(owner.at(1), 3.0f);
    EXPECT_FLOAT_EQ(owner.at(3), 6.0f);
    EXPECT_FLOAT_EQ(owner.at(5), 5.0f);
}

/**
 * @test FACTORIAL.nan_throws
 * @brief factorial should throw std::runtime_error if input contains NaN.
 */
TEST(FACTORIAL, nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f};

    EXPECT_THROW(math::factorial(t), std::runtime_error);
}

/**
 * @test FACTORIAL.inf_throws
 * @brief factorial should throw std::runtime_error if input contains +Inf/-Inf.
 */
TEST(FACTORIAL, inf_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{std::numeric_limits<float>::infinity(), 1.0f};

    EXPECT_THROW(math::factorial(t), std::runtime_error);
}

/**
 * @test FACTORIAL_negative_and_noninteger_throws
 * @brief negative or non-integer inputs should throw std::invalid_argument.
 */
TEST(FACTORIAL, negative_and_noninteger_throws)
{
    Tensor<float> neg({1}, MemoryLocation::DEVICE);
    neg = std::vector<float>{-1.0f};
    EXPECT_THROW(math::factorial(neg), std::invalid_argument);

    Tensor<float> nonint({2}, MemoryLocation::DEVICE);
    nonint = std::vector<float>{2.5f, 3.14159f};
    EXPECT_THROW(math::factorial(nonint), std::invalid_argument);
}

/**
 * @test FACTORIAL_overflow_throws
 * @brief factorial of sufficiently large n should overflow float
 * and provoke runtime_error.
 *
 * 35! > float max (≈3.4e38), so 35 should produce non-finite and raise.
 */
TEST(FACTORIAL, overflow_throws)
{
    Tensor<float> t({1}, MemoryLocation::DEVICE);
    t = std::vector<float>{35.0f};

    EXPECT_THROW(math::factorial(t), std::runtime_error);
}

/**
 * @test FACTORIAL_zero_and_one
 * @brief factorial(0) == 1 and factorial(1) == 1.
 */
TEST(FACTORIAL, zero_and_one)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{0.0f, 1.0f};

    Tensor<float> out = math::factorial(t);

    ASSERT_EQ(out.get_num_elements(), 2u);
    std::vector<float> host(2);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 2).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 1.0f);
}

/**
 * @test LOG.basic_values
 * @brief log on simple positive values returns expected natural logs.
 */
TEST(LOG, basic_values)
{
    Tensor<float> t({4}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, static_cast<float>(M_E), 10.0f, 0.5f};

    Tensor<float> out = math::log(t);

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 4u);

    std::vector<float> host(N);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * N).wait();

    EXPECT_FLOAT_EQ(host[0], std::log(1.0f));
    EXPECT_FLOAT_EQ(host[1], std::log(static_cast<float>(M_E)));
    EXPECT_FLOAT_EQ(host[2], std::log(10.0f));
    EXPECT_FLOAT_EQ(host[3], std::log(0.5f));
}

/**
 * @test LOG.alias_view_strided
 * @brief log on a 1D alias view with non-unit stride reads view elements.
 */
TEST(LOG, alias_view_strided)
{
    Tensor<float> owner({6}, MemoryLocation::DEVICE);
    owner = std::vector<float>{1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 5.0f};

    std::vector<uint64_t> start = {0ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> v(owner, start, dims, strides);

    Tensor<float> out = math::log(v);

    ASSERT_EQ(out.get_num_elements(), 3u);
    std::vector<float> host(3);
    g_sycl_queue.memcpy(host.data(),
        out.m_p_data.get(), sizeof(float) * 3).wait();

    EXPECT_FLOAT_EQ(host[0], std::log(1.0f));
    EXPECT_FLOAT_EQ(host[1], std::log(2.0f));
    EXPECT_FLOAT_EQ(host[2], std::log(4.0f));
}

/**
 * @test LOG.nan_throws
 * @brief log should throw std::runtime_error if input contains NaN.
 */
TEST(LOG, nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f};

    EXPECT_THROW(math::log(t), std::runtime_error);
}

/**
 * @test LOG_inf_throws
 * @brief log should throw std::runtime_error if input contains +Inf/-Inf.
 */
TEST(LOG, inf_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{std::numeric_limits<float>::infinity(), 1.0f};

    EXPECT_THROW(math::log(t), std::runtime_error);
}

/**
 * @test LOG_zero_throws
 * @brief log(0) produces -inf which the implementation treats as non-finite
 * and should therefore throw std::runtime_error.
 */
TEST(LOG, zero_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{0.0f, 1.0f};

    EXPECT_THROW(math::log(t), std::runtime_error);
}

/**
 * @test LOG_negative_throws
 * @brief log of negative real values produces NaN/invalid result and should
 * throw std::runtime_error in this implementation.
 */
TEST(LOG, negative_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{-1.0f, 2.0f};

    EXPECT_THROW(math::log(t), std::runtime_error);
}

/**
 * @test LOG_scalar
 * @brief log of a scalar-like 1-element tensor returns a single-element
 * tensor with the expected value.
 */
TEST(LOG, scalar)
{
    Tensor<float> s({1}, MemoryLocation::DEVICE);
    s = std::vector<float>{2.7182818f};

    Tensor<float> out = math::log(s);
    ASSERT_EQ(out.get_num_elements(), 1u);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(), sizeof(float)).wait();
    EXPECT_NEAR(host[0], 1.0f, 1e-6f);
}

/**
 * @test MEAN.mean_all_elements
 * @brief Mean all elements (axis = nullopt) on a device tensor returns scalar mean.
 */
TEST(MEAN, mean_all_elements)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    t = vals;

    Tensor<float> res = math::mean(t);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host[0], 2.0f);
}

/**
 * @test MEAN.mean_axis0
 * @brief Mean along axis 0 (per-column) for a 2x3 device tensor.
 */
TEST(MEAN, mean_axis0)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = math::mean(t, 0);

    std::vector<float> host(3);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 2.5f);
    EXPECT_FLOAT_EQ(host[1], 3.5f);
    EXPECT_FLOAT_EQ(host[2], 4.5f);
}

/**
 * @test MEAN.mean_axis1
 * @brief Mean along axis 1 (per-row) for a 2x3 device tensor.
 */
TEST(MEAN, mean_axis1)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = math::mean(t, 1);

    std::vector<float> host(2);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 2 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 2.0f);
    EXPECT_FLOAT_EQ(host[1], 5.0f);
}

/**
 * @test MEAN.mean_axis0_3D
 * @brief Mean along axis 0 for a 2x2x2 tensor (verify outputs).
 */
TEST(MEAN, mean_axis0_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::mean(t, 0);

    std::vector<float> host(4);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], (1.0f + 5.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[1], (2.0f + 6.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[2], (3.0f + 7.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[3], (4.0f + 8.0f) / 2.0f);
}

/**
 * @test MEAN.mean_axis1_3D
 * @brief Mean along axis 1 for a 2x2x2 tensor (verify outputs).
 */
TEST(MEAN, mean_axis1_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::mean(t, 1);

    std::vector<float> host(4);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], (1.0f + 3.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[1], (2.0f + 4.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[2], (5.0f + 7.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[3], (6.0f + 8.0f) / 2.0f);
}

/**
 * @test MEAN.mean_axis2_3D
 * @brief Mean along axis 2 for a 2x2x2 tensor (verify outputs).
 */
TEST(MEAN, mean_axis2_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::mean(t, 2);

    std::vector<float> host(4);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], (1.0f + 2.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[1], (3.0f + 4.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[2], (5.0f + 6.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[3], (7.0f + 8.0f) / 2.0f);
}

/**
 * @test MEAN.mean_axis_negative
 * @brief Mean along axis -1 for a 2x2x2 tensor (verify outputs).
 */
TEST(MEAN, mean_axis_negative)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = math::mean(t, -1);

    std::vector<float> host(4);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], (1.0f + 2.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[1], (3.0f + 4.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[2], (5.0f + 6.0f) / 2.0f);
    EXPECT_FLOAT_EQ(host[3], (7.0f + 8.0f) / 2.0f);
}

/**
 * @test MEAN.mean_view_tensor
 * @brief Mean all elements (axis = -1) of a view into a device tensor.
 */
TEST(MEAN, mean_view_tensor)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 0ull};
    std::vector<uint64_t> view_shape = {3ull}; // view of 3 elements

    Tensor<float> view(t, start_indices, view_shape);

    Tensor<float> res = math::mean(view);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], (1.0f + 2.0f + 3.0f) / 3.0f);
}

/**
 * @test MEAN.mean_alias_view_tensor
 * @brief Mean all elements of an alias view (non-unit stride).
 */
TEST(MEAN, mean_alias_view_tensor)
{
    Tensor<float> t({6}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {0ull};
    std::vector<uint64_t> dims = {3ull};
    std::vector<uint64_t> strides = {2ull};

    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = math::mean(alias_view);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    // elements: 1,3,5 -> mean = 3.0
    EXPECT_FLOAT_EQ(host[0], 3.0f);
}

/**
 * @test MEAN.mean_view_tensor_3d_axis1
 * @brief Mean along axis 1 on a 3D view (verify outputs).
 */
TEST(MEAN, mean_view_tensor_3d_axis1)
{
    Tensor<float> t({3, 4, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 1ull, 0ull};
    std::vector<uint64_t> view_shape    = {2ull, 2ull};
    Tensor<float> view(t, start_indices, view_shape);

    Tensor<float> res = math::mean(view, 1);

    std::vector<float> host(2);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0], 23.0f / 2.0f);
    EXPECT_FLOAT_EQ(host[1], 27.0f / 2.0f);
}

/**
 * @test MEAN.mean_alias_view_tensor_2d_strided
 * @brief Mean along axis 0 on a 2D alias view with custom strides.
 */
TEST(MEAN, mean_alias_view_tensor_2d_strided)
{
    Tensor<float> t({4, 5}, MemoryLocation::DEVICE);
    std::vector<float> vals(20);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 1ull};
    std::vector<uint64_t> dims          = {2ull, 3ull};
    std::vector<uint64_t> strides       = {5ull, 2ull};
    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = math::mean(alias_view, 0);

    std::vector<float> host(3);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), sizeof(float) * host.size()).wait();

    // previous sum expected [9,13,17] -> divide by axis0 length (2)
    EXPECT_FLOAT_EQ(host[0], 9.0f / 2.0f);
    EXPECT_FLOAT_EQ(host[1], 13.0f / 2.0f);
    EXPECT_FLOAT_EQ(host[2], 17.0f / 2.0f);
}

/**
 * @test MEAN.mean_alias_view_tensor_overlapping_stride_zero
 * @brief Mean accounts for overlapping elements when a stride is zero.
 */
TEST(MEAN, mean_alias_view_tensor_overlapping_stride_zero)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 0ull};
    std::vector<uint64_t> dims          = {2ull, 2ull};
    std::vector<uint64_t> strides       = {0ull, 1ull};
    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = math::mean(alias_view, 0);

    std::vector<float> host(2);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), sizeof(float) * host.size()).wait();

    // previous sum expected [8,10] -> divide by axis0 length (2)
    EXPECT_FLOAT_EQ(host[0], 8.0f / 2.0f);
    EXPECT_FLOAT_EQ(host[1], 10.0f / 2.0f);
}

/**
 * @test MEAN.mean_nan_throws
 * @brief mean() throws when tensor contains NaN values.
 */
TEST(MEAN, mean_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals =
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    t = vals;

    EXPECT_THROW(math::mean(t), std::runtime_error);
}

/**
 * @test MEAN.mean_non_finite_throws
 * @brief mean() throws when tensor contains non-finite values (Inf).
 */
TEST(MEAN, mean_non_finite_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {std::numeric_limits<float>::infinity(), 1.0f};
    t = vals;

    EXPECT_THROW(math::mean(t), std::runtime_error);
}

/**
 * @test MEAN.mean_empty
 * @brief mean() on an empty tensor throws std::invalid_argument.
 */
TEST(MEAN, mean_empty)
{
    Tensor<float> t;

    EXPECT_THROW(math::mean(t), std::invalid_argument);
}

/**
 * @test VAR.var_all_elements
 * @brief Variance of all elements (axis = -1) returns scalar.
 */
TEST(VAR, var_all_elements)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> res = math::var(t, std::nullopt, 0);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 2.0f/3.0f);
}

/**
 * @test VAR.var_ddof_1_sample
 * @brief Variance with ddof=1 (sample variance).
 */
TEST(VAR, var_ddof_1_sample)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> res = math::var(t, std::nullopt, 1);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
}

/**
 * @test VAR.var_axis0
 * @brief Variance along axis 0 for 2x3 tensor (per-column).
 */
TEST(VAR, var_axis0)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1,2,3, 4,5,6};

    Tensor<float> res = math::var(t, 0, 0);
    std::vector<float> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), 3*sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 2.25f);
    EXPECT_FLOAT_EQ(host[1], 2.25f);
    EXPECT_FLOAT_EQ(host[2], 2.25f);
}

/**
 * @test VAR.var_axis_negative
 * @brief Variance along axis -2 for 2x3 tensor (per-column).
 */
TEST(VAR, var_axis_negative)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1,2,3, 4,5,6};

    Tensor<float> res = math::var(t, -2, 0);
    std::vector<float> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), 3*sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 2.25f);
    EXPECT_FLOAT_EQ(host[1], 2.25f);
    EXPECT_FLOAT_EQ(host[2], 2.25f);
}

/**
 * @test VAR.var_view_and_alias
 * @brief Variance on view and alias view (flattened).
 */
TEST(VAR, var_view_and_alias)
{
    Tensor<float> owner({2,3}, MemoryLocation::DEVICE);
    owner = std::vector<float>{1,2,3,4,5,6};

    Tensor<float> view(owner,
                       std::vector<uint64_t>{0ull, 0ull},
                       std::vector<uint64_t>{3ull},
                       std::vector<uint64_t>{1ull});

    Tensor<float> v1 = math::var(view, std::nullopt, 0);
    std::vector<float> h1(1);
    g_sycl_queue.memcpy(h1.data(), v1.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(h1[0], 2.0f/3.0f);

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> alias(owner, start, dims, strides);
    Tensor<float> v2 = math::var(alias, std::nullopt, 0);
    std::vector<float> h2(1);
    g_sycl_queue.memcpy(h2.data(), v2.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(h2[0], 8.0f/3.0f);
}

/**
 * @test VAR.var_nan_throws
 * @brief var() throws when inputs contain NaN.
 */
TEST(VAR, var_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    EXPECT_THROW(math::var(t, std::nullopt, 0), std::runtime_error);
}

/**
 * @test VAR.var_non_finite_throws
 * @brief var() throws when inputs contain Inf.
 */
TEST(VAR, var_non_finite_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{std::numeric_limits<float>::infinity(), 1.0f};
    EXPECT_THROW(math::var(t, std::nullopt, 0), std::runtime_error);
}

/**
 * @test VAR.var_empty
 * @brief var() on empty tensor throws std::invalid_argument.
 */
TEST(VAR, var_empty)
{
    Tensor<float> t;
    EXPECT_THROW(math::var(t, std::nullopt, 0), std::invalid_argument);
}

/**
 * @test COV.cov_empty
 * @brief Verify cov() throws when called on an empty tensor.
 */
TEST(COV, cov_empty)
{
    Tensor<float> t;
    EXPECT_THROW(math::cov(t, {1}, {1}), std::invalid_argument);
}

/**
 * @test COV.cov_axis_empty
 * @brief Verify cov() rejects empty axis vectors.
 */
TEST(COV, cov_axis_empty)
{
    Tensor<float> t({2, 3});
    EXPECT_THROW(math::cov(t, {}, {}), std::invalid_argument);
}

/**
 * @test COV.cov_ddof_negative
 * @brief Verify cov() rejects negative ddof.
 */
TEST(COV, cov_ddof_negative)
{
    Tensor<float> t({1});
    EXPECT_THROW(math::cov(t, {1}, {1}, -45), std::invalid_argument);
}


/**
 * @test COV.cov_axis_out_of_range
 * @brief Verify cov() rejects out-of-range axis indices.
 */
TEST(COV, cov_axis_out_of_range)
{
    Tensor<float> t({2, 3});
    EXPECT_THROW(math::cov(t, {15, 16, 17}, {2, 18}), std::invalid_argument);
}

/**
 * @test COV.cov_rank_lower_than_2
 * @brief Verify cov() requires tensor rank >= 2.
 */
TEST(COV, cov_rank_lower_than_2)
{
    Tensor<float> t({2});
    EXPECT_THROW(math::cov(t, {1}, {0}), std::invalid_argument);
}

/**
 * @test COV.cov_duplicate_axes
 * @brief Verify cov() rejects duplicate axes.
 */
TEST(COV, cov_duplicate_axes)
{
    Tensor<float> t({2, 3});
    EXPECT_THROW(math::cov(t, {0}, {0}), std::invalid_argument);
}


/**
 * @test COV.cov_ddof_too_high
 * @brief Verify cov() rejects ddof >= sample count.
 */
TEST(COV, cov_ddof_too_high)
{
    Tensor<float> t({2, 3});
    EXPECT_THROW(math::cov(t, {0}, {1}, 2), std::invalid_argument);
}

/**
 * @test TENSOR.cov_basic
 * @brief Basic 2D covariance computation (non-view).
 *
 * Creates a 3×2 tensor, computes sample covariance (ddof=1)
 * and verifies matrix entries against precomputed values.
 */
TEST(COV, cov_basic)
{
    Tensor<float> t({3, 2});
    std::vector<float> values = { 2.1f, 8.0f,
                                2.5f, 12.0f,
                                4.0f, 14.0f};
    t = values;

    Tensor<float> t_cov = math::cov(t, {0}, {1}, 1);

    std::vector<float> expected = { 1.0033f, 2.6665f,
                                    2.6665f, 9.333f};
    const float tol = 1e-3;

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(t_cov[i][j], expected[i*2 + j], tol);
        }
    }
}

/**
 * @test COV.cov_negative_axis
 * @brief Basic 2D covariance computation using negative indexing.
 *
 * Creates a 3×2 tensor, computes sample covariance (ddof=1)
 * and verifies matrix entries against precomputed values.
 */
TEST(COV, cov_negative_axis)
{
    Tensor<float> t({3, 2});
    std::vector<float> values = { 2.1f, 8.0f,
                                2.5f, 12.0f,
                                4.0f, 14.0f};
    t = values;

    Tensor<float> t_cov = math::cov(t, {-2}, {-1}, 1);

    std::vector<float> expected = { 1.0033f, 2.6665f,
                                    2.6665f, 9.333f};
    const float tol = 1e-3;

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(t_cov[i][j], expected[i*2 + j], tol);
        }
    }
}

/**
 * @test COV.cov_basic_view
 * @brief Basic covariance on a contiguous sub-tensor view.
 *
 * Constructs a non-owning view (contiguous layout) and verifies
 * cov(sample_axes={0}, event_axes={1}, ddof=1) values.
 */
TEST(COV, cov_basic_view)
{
    Tensor<float> t({3, 3});
    std::vector<float> values = { 0.4f, 2.1f, 8.0f,
                                1.1f, 2.5f, 12.0f,
                                15.0f, 4.0f, 14.0f};
    t = values;

    Tensor<float> t_view(t, {0, 1}, {3, 2});

    Tensor<float> t_cov = math::cov(t_view, {0}, {1}, 1);

    std::vector<float> expected = { 1.0033f, 2.6665f,
                                    2.6665f, 9.333f};
    const float tol = 1e-3;

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(t_cov[i][j], expected[i*2 + j], tol);
        }
    }
}

/**
 * @test COV.cov_basic_alias_view
 * @brief Covariance on a non-contiguous alias view (custom strides).
 *
 * Uses the alias-view constructor with non-trivial strides and
 * verifies covariance results against precomputed constants.
 */
TEST(COV, cov_basic_alias_view)
{
    Tensor<float> t({3, 3});
    std::vector<float> values =
    {
        0.4f, 2.1f, 8.0f,
        1.1f, 2.5f, 12.0f,
        15.0f, 4.0f, 14.0f
    };
    t = values;

    Tensor<float> t_alias(t, {0, 0}, {3, 2}, {1, 3});

    Tensor<float> t_cov = math::cov(t_alias, {0}, {1}, 1);

    std::vector<float> expected =
    {
        15.91f,
        23.545f,
        23.545f,
        35.17f
    };

    const float tol = 1e-3;

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(t_cov[i][j], expected[i*2 + j], tol);
        }
    }
}

/**
 * @test COV.cov_higherdim_batch
 * @brief Batched covariance on a 3D tensor (one batch axis).
 *
 * Tests cov on shape {2,3,2} producing one 2×2 covariance per batch
 * and verifies expected (scaled) results for both batches.
 */
TEST(COV, cov_higherdim_batch)
{
    Tensor<float> t({2, 3, 2});

    std::vector<float> values =
    {
        1.0f, 2.0f,
        2.0f, 1.0f,
        3.0f, 4.0f,

        10.0f, 20.0f,
        20.0f, 10.0f,
        30.0f, 40.0f
    };
    t = values;

    Tensor<float> t_cov = math::cov(t, {1}, {2}, 1);

    const double tol = 1e-3;

    std::vector<double> expected_b0 =
    {
        1.0, 1.0,
        1.0, 2.333333333333333
    };

    std::vector<double> expected_b1 =
    {
        100.0, 100.0,
        100.0, 233.33333333333331
    };

    for (uint64_t b = 0; b < 2; ++b)
    {
        for (uint64_t i = 0; i < 2; ++i)
        {
            for (uint64_t j = 0; j < 2; ++j)
            {
                double got = static_cast<double>(t_cov[b][i][j]);
                double exp;
                if (b == 0)
                {
                    exp = expected_b0[i*2 + j];
                }
                else
                {
                    exp = expected_b1[i*2 + j];
                }
                EXPECT_NEAR(got, exp, tol);
            }
        }
    }
}

/**
 * @test COV.cov_4d_batched
 * @brief Covariance on a 4D tensor with two batch axes.
 *
 * Exercises shape {2,2,3,2} with sample axis 2 and event axis 3,
 * verifying one 2×2 covariance per batch element.
 */

TEST(COV, cov_4d_batched)
{
    Tensor<float> t({2, 2, 3, 2});
    std::vector<float> values;
    values.reserve(2 * 2 * 3 * 2);

    std::vector<std::vector<float>> base_samples =
    {
        {1.0f, 2.0f},
        {2.0f, 1.0f},
        {3.0f, 4.0f}
    };

    for (uint64_t i0 = 0; i0 < 2; ++i0)
    {
        for (uint64_t i1 = 0; i1 < 2; ++i1)
        {
            uint64_t batch_index = i0 * 2 + i1;
            float k = static_cast<float>(batch_index + 1);
            for (const auto &s : base_samples)
            {
                values.push_back(k * s[0]);
                values.push_back(k * s[1]);
            }
        }
    }

    ASSERT_EQ(values.size(), 2 * 2 * 3 * 2);
    t = values;

    Tensor<float> t_cov = math::cov(t, {2}, {3}, 1);

    const double tol = 1e-3;
    std::vector<std::vector<double>> expected_for_k =
    {
        {1.0, 1.0, 1.0, 2.333333333333333},
        {4.0, 4.0, 4.0, 9.333333333333334},
        {9.0, 9.0, 9.0, 21.0},
        {16.0, 16.0, 16.0, 37.333333333333336}
    };

    for (uint64_t i0 = 0; i0 < 2; ++i0)
    {
        for (uint64_t i1 = 0; i1 < 2; ++i1)
        {
            uint64_t b = i0 * 2 + i1;
            for (uint64_t i = 0; i < 2; ++i)
            {
                for (uint64_t j = 0; j < 2; ++j)
                {
                    double got = static_cast<double>(t_cov[i0][i1][i][j]);
                    double exp = expected_for_k[b][i*2 + j];
                    EXPECT_NEAR(got, exp, tol);
                }
            }
        }
    }
}

/**
 * @test COV.cov_alias_view_higherdim
 * @brief Covariance on a higher-dim alias view with strides.
 *
 * Builds a 3×3×3 owner, creates an alias view with non-trivial
 * strides, and verifies the resulting 4×4 covariance.
 */
TEST(COV, cov_alias_view_higherdim)
{
    Tensor<float> owner({3, 3, 3});
    {
        std::vector<float> vals;
        vals.reserve(27);
        for (int i = 0; i < 27; ++i)
        {
            vals.push_back(static_cast<float>(i));
        }
        owner = vals;
    }

    Tensor<float> talias(owner, {0, 0, 0}, {3, 2, 2}, {1, 3, 9});

    Tensor<float> t_cov = math::cov(talias, {0}, {1, 2}, 1);

    const double tol = 1e-6;
    for (uint64_t i = 0; i < 4; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            double got = static_cast<double>(t_cov[i][j]);
            double exp = 1.0;
            EXPECT_NEAR(got, exp, tol);
        }
    }
}


/**
 * @test COV.cov_scattered_no_batch
 * @brief Covariance with scattered sample/event axes (no batch).
 *
 * Tests cov on a 4D tensor using sample_axes={0,2} and
 * event_axes={1,3} (resulting 6×6 covariance) and checks values.
 */
TEST(COV, cov_scattered_no_batch)
{
    Tensor<float> t({2, 3, 2, 2});
    const std::size_t N = 2ull * 3ull * 2ull * 2ull;
    std::vector<float> vals;
    vals.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        vals.push_back(static_cast<float>(i));
    }
    t = vals;

    Tensor<float> cov = math::cov(t, {0, 2}, {1, 3}, 1);

    const double expected = 49.33333333333333;
    const double tol = 1e-4;

    for (uint64_t i = 0; i < 6; ++i)
    {
        for (uint64_t j = 0; j < 6; ++j)
        {
            EXPECT_NEAR(static_cast<double>(cov[i][j]), expected, tol);
        }
    }
}

/**
 * @test COV.cov_scattered_batched
 * @brief Covariance with scattered axes producing a batch axis.
 *
 * Verifies cov on a 5D tensor with sample_axes and event_axes in
 * non-contiguous positions producing per-batch 4×4 covariances.
 */
TEST(COV, cov_scattered_batched)
{
    Tensor<float> t({2, 2, 3, 2, 2});
    const std::size_t N = 2ull * 2ull * 3ull * 2ull * 2ull;
    std::vector<float> vals;
    vals.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        vals.push_back(static_cast<float>(i));
    }
    t = vals;

    Tensor<float> cov = math::cov(t, {1, 3}, {0, 4}, 1);

    const double expected = 49.33333333333333;
    const double tol = 1e-4;

    for (uint64_t b = 0; b < 3; ++b)
    {
        for (uint64_t i = 0; i < 4; ++i)
        {
            for (uint64_t j = 0; j < 4; ++j)
            {
                EXPECT_NEAR(static_cast<double>(cov[b][i][j]), expected, tol);
            }
        }
    }
}

/**
 * @test COV.cov_alias_view_scattered
 * @brief Covariance on an alias view with scattered axes.
 *
 * Constructs a non-contiguous alias view on a 4×3×2 owner and
 * verifies cov(sample_axes={0,2}, event_axes={1}, ddof=1).
 */
TEST(COV, cov_alias_view_scattered)
{
    Tensor<float> owner({4, 3, 2});
    std::vector<float> owner_vals;
    owner_vals.reserve(4 * 3 * 2);
    for (std::size_t i = 0; i < 4 * 3 * 2; ++i)
    {
        owner_vals.push_back(static_cast<float>(i));
    }
    owner = owner_vals;

    Tensor<float> talias(owner, {0, 0, 1},
                         {2, 2, 2},
                         {1, 6, 2});
    Tensor<float> cov = math::cov(talias, {0, 2}, {1}, 1);

    const double expected = 1.6666666666666665;
    const double tol = 1e-6;

    ASSERT_EQ(cov.get_rank(), 2u);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(static_cast<double>(cov[i][j]), expected, tol);
        }
    }
}

/**
 * @test COV.cov_scattered_highdim
 * @brief High-dim scattered axes covariance (6D tensor).
 *
 * Exercises cov with multiple scattered sample and event axes on a
 * 6D tensor and verifies per-batch 4×4 covariance values.
 */
TEST(COV, cov_scattered_highdim)
{
    Tensor<float> t({2,2,2,2,2,2});
    const std::size_t N = 64;
    std::vector<float> vals;
    vals.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        vals.push_back(static_cast<float>(i));
    }
    t = vals;

    Tensor<float> cov = math::cov(t, {0, 3, 5}, {1, 4}, 1);

    const double expected = 297.4285714285714;
    const double tol = 1e-4;

    for (uint64_t b = 0; b < 2; ++b)
    {
        for (uint64_t i = 0; i < 4; ++i)
        {
            for (uint64_t j = 0; j < 4; ++j)
            {
                EXPECT_NEAR(static_cast<double>(cov[b][i][j]), expected, tol);
            }
        }
    }
}

/**
 * @test COV.cov_ddof_default_2d
 * @brief Test shorthand cov(ddof) for a 2D tensor.
 *
 * Verifies cov(ddof) equals cov({rank-2},{rank-1},ddof) and checks
 * numeric correctness on a small 2D example.
 */
TEST(COV, cov_ddof_default_2d)
{
    Tensor<float> t({3, 2});
    std::vector<float> vals = {
        1.0f, 2.0f,
        2.0f, 1.0f,
        3.0f, 4.0f
    };
    t = vals;

    Tensor<float> cov_shorthand = math::cov(t, 1);

    Tensor<float> cov_explicit = math::cov(t, {0}, {1}, 1);

    const double tol = 1e-4;

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            double a = static_cast<double>(cov_shorthand[i][j]);
            double b = static_cast<double>(cov_explicit[i][j]);
            EXPECT_NEAR(a, b, tol);
            double expected;
            if (i == 0 && j == 0)
            {
                expected = 1.0;
            }
            else if ((i == 0 && j == 1) || (i == 1 && j == 0))
            {
                expected = 1.0;
            }
            else
            {
                expected = 2.333333333333333;
            }
            EXPECT_NEAR(a, expected, tol);
        }
    }
}

/**
 * @test COV.cov_ddof_default_batched3d
 * @brief Test shorthand cov(ddof) for a batched 3D tensor.
 *
 * Ensures cov(ddof) behaves like explicit axes selection on a
 * {batch, samples, events} tensor and matches expected results.
 */
TEST(COV, cov_ddof_default_batched3d)
{
    Tensor<float> t({2, 3, 2});
    std::vector<float> vals;
    vals.reserve(2 * 3 * 2);

    vals.push_back(1.0f); vals.push_back(2.0f);
    vals.push_back(2.0f); vals.push_back(1.0f);
    vals.push_back(3.0f); vals.push_back(4.0f);

    vals.push_back(10.0f); vals.push_back(20.0f);
    vals.push_back(20.0f); vals.push_back(10.0f);
    vals.push_back(30.0f); vals.push_back(40.0f);

    t = vals;

    Tensor<float> cov_shorthand = math::cov(t, 1);

    Tensor<float> cov_explicit = math::cov(t, {1}, {2}, 1);

    const double tol = 1e-4;

    for (uint64_t b = 0; b < 2; ++b)
    {
        for (uint64_t i = 0; i < 2; ++i)
        {
            for (uint64_t j = 0; j < 2; ++j)
            {
                double a = static_cast<double>(cov_shorthand[b][i][j]);
                double bval = static_cast<double>(cov_explicit[b][i][j]);
                EXPECT_NEAR(a, bval, tol);

                double expected_base;
                if (i == 0 && j == 0)
                {
                    expected_base = 1.0;
                }
                else if ((i == 0 && j == 1) || (i == 1 && j == 0))
                {
                    expected_base = 1.0;
                }
                else
                {
                    expected_base = 2.333333333333333;
                }

                double expected;
                if (b == 0)
                {
                    expected = expected_base;
                }
                else
                {
                    expected = expected_base * 100.0;
                }
                EXPECT_NEAR(a, expected, tol);
            }
        }
    }
}

/**
 * @test STDDEV.std_basic
 * @brief math::std returns sqrt(var) for a 1D device tensor.
 */
TEST(STDDEV, stddev_basic)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> res = math::stddev(t, std::nullopt, 0);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], std::sqrt(2.0f/3.0f));
}

/**
 * @test STDDEV.stddev_ddof1
 * @brief math::std with ddof=1 (sample std).
 */
TEST(STDDEV, stddev_ddof1)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> res = math::stddev(t, std::nullopt, 1);
    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
}

/**
 * @test STDDEV.stddev_axis0_2D
 * @brief std along axis 0 produces per-column standard deviations.
 */
TEST(STDDEV, stddev_axis0_2D)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f,2.0f,3.0f, 4.0f,5.0f,6.0f};

    Tensor<float> res = math::stddev(t, 0, 0);
    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.5f);
    EXPECT_FLOAT_EQ(host[1], 1.5f);
    EXPECT_FLOAT_EQ(host[2], 1.5f);
}

/**
 * @test STDDEV.stddev_axis1_2D
 * @brief std along axis 1 produces per-row standard deviations.
 */
TEST(STDDEV, stddev_axis1_2D)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f,2.0f,3.0f, 4.0f,5.0f,6.0f};

    Tensor<float> res = math::stddev(t, 1, 0);
    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 2 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], std::sqrt(2.0f/3.0f));
    EXPECT_FLOAT_EQ(host[1], std::sqrt(2.0f/3.0f));
}

/**
 * @test STDDEV.stddev_axis_negative
 * @brief Standard deviation along axis -3 for a 2x2x2 tensor.
 */
TEST(STDDEV, stddev_axis_negative)
{
    Tensor<float> t({2,2,2}, MemoryLocation::DEVICE);
    t = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    Tensor<float> res = math::stddev(t, -3, 0);
    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 2.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 2.0f);
    EXPECT_FLOAT_EQ(host[3], 2.0f);
}

/**
 * @test STDDEV.stddev_alias_stride14
 * @brief Tests math::std on alias views with non-contiguous stride = 14.
 */
TEST(STDDEV, stddev_alias_stride14)
{
    std::vector<uint64_t> owner_shape;
    owner_shape.push_back(5ull);
    owner_shape.push_back(20ull);

    Tensor<float> owner(owner_shape, MemoryLocation::DEVICE);

    std::vector<float> vals;
    vals.resize(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    g_sycl_queue.memcpy
        (owner.m_p_data.get(), vals.data(), sizeof(float) * 100).wait();

    std::vector<uint64_t> start;
    start.push_back(0ull);
    start.push_back(0ull);

    std::vector<uint64_t> dims;
    dims.push_back(3ull);
    dims.push_back(4ull);

    std::vector<uint64_t> strides;
    strides.push_back(14ull);
    strides.push_back(4ull);

    Tensor<float> alias(owner, start, dims, strides);

    Tensor<float> r = math::stddev(alias, std::nullopt, 0);
    std::vector<float> host_val(1);
    g_sycl_queue.memcpy(host_val.data(), r.m_p_data.get(), sizeof(float)).wait();

    std::vector<float> data;
    data.resize(12);
    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            data[i * 4 + j] = static_cast<float>(i * 14ull + j * 4ull);
        }
    }

    double mean = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        mean += data[i];
    }
    mean /= static_cast<double>(data.size());

    double var = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        double diff = data[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(data.size());

    float expected_std = static_cast<float>(std::sqrt(var));

    EXPECT_FLOAT_EQ(host_val[0], expected_std);
}

/**
 * @test STDDEV.stddev_nan_throws
 * @brief math::std throws when inputs contain NaN.
 */
TEST(STDDEV, stddev_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};

    EXPECT_THROW(math::stddev(t, std::nullopt, 0), std::runtime_error);
}

/**
 * @test STDDEV.stddev_non_finite_throws
 * @brief math::std throws when inputs contain infinity.
 */
TEST(STDDEV, stddev_non_finite_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{std::numeric_limits<float>::infinity(), 1.0f};

    EXPECT_THROW(math::stddev(t, std::nullopt, 0), std::runtime_error);
}

/**
 * @test STDDEV.stddev_empty
 * @brief math::std on an empty tensor throws std::invalid_argument.
 */
TEST(STDDEV, stddev_empty)
{
    Tensor<float> t;
    EXPECT_THROW(math::stddev(t, std::nullopt, 0), std::invalid_argument);
}

/**
 * @test STDDEV.stddev_ddof_invalid_throws
 * @brief math::std throws when ddof >= axis length (invalid denominator).
 */
TEST(STDDEV, stddev_ddof_invalid_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, 2.0f, 3.0f};

    EXPECT_THROW(math::stddev(t, std::nullopt, 3), std::invalid_argument);
    EXPECT_THROW(math::stddev(t, std::nullopt, 4), std::invalid_argument);
}

/**
 * @test SQRT.sqrt_basic_positive
 * @brief Elementwise sqrt of positive elements on device.
 */
TEST(SQRT, sqrt_basic_positive)
{
    Tensor<float> t({5}, MemoryLocation::DEVICE);
    t = std::vector<float>{0.0f, 1.0f, 4.0f, 9.0f, 16.0f};

    Tensor<float> res = math::sqrt(t);

    std::vector<float> host(5);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 5 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 0.0f);
    EXPECT_FLOAT_EQ(host[1], 1.0f);
    EXPECT_FLOAT_EQ(host[2], 2.0f);
    EXPECT_FLOAT_EQ(host[3], 3.0f);
    EXPECT_FLOAT_EQ(host[4], 4.0f);
}

/**
 * @test EIG.eig_empty
 * @brief Verify eig() throws when called on an empty tensor.
 */
TEST(EIG, eig_empty)
{
    Tensor<float> t;
    EXPECT_THROW(math::eig(t, 100, 1e-6f), std::invalid_argument);
}

/**
 * @test EIG.eig_rank_lower_than_2
 * @brief Verify eig() requires tensor rank >= 2.
 */
TEST(EIG, eig_rank_lower_than_2)
{
    Tensor<float> t({2});
    EXPECT_THROW(math::eig(t, 100, 1e-6f), std::invalid_argument);
}

/**
 * @test EIG.eig_last_two_not_square
 * @brief Verify eig() rejects tensors whose last two dims are not square.
 */
TEST(EIG, eig_last_two_not_square)
{
    Tensor<float> t({2, 3, 4});
    EXPECT_THROW(math::eig(t, 100, 1e-6f), std::invalid_argument);
}

/**
 * @test EIG.eig_diagonal
 * @brief eig() returns expected eigenvalues for a diagonal device matrix.
 *
 * Creates a 4×4 diagonal matrix allocated on device, runs eig(), reads back
 * eigenvalues and checks they match the diagonal entries (within tolerance).
 */
TEST(EIG, eig_diagonal)
{
    const uint64_t n = 4;
    Tensor<float> A({n, n}, MemoryLocation::DEVICE);
    std::vector<float> flat(n * n, 0.0f);
    for (uint64_t i = 0; i < n; ++i)
    {
        flat[i * n + i] = static_cast<float>(i + 1);
    }
    A = flat;

    auto result = math::eig(A, 80, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<double> got;
    for (uint64_t i = 0; i < n; ++i)
    {
        Tensor<float> v_view(vals, std::vector<uint64_t>{i},
            std::vector<uint64_t>{1});
        got.push_back(static_cast<double>(static_cast<float>(v_view)));
    }
    std::sort(got.begin(), got.end());

    std::vector<double> expected = {1.0, 2.0, 3.0, 4.0};
    std::sort(expected.begin(), expected.end());

    const double tol = 5e-4;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(got[i], expected[i], tol);
    }
}

/**
 * @test EIG.eig_batched
 * @brief eig() on a small batched tensor yields per-batch eigenvalues.
 *
 * Builds a 2×3×3 device tensor containing two diagonal batches, runs eig()
 * with increased iterations, and checks each batch's computed eigenvalues
 * against expected diagonals (sorted, with loose tolerance for device runs).
 */
TEST(EIG, eig_batched)
{
    Tensor<float> T({2,3,3}, MemoryLocation::DEVICE);
    std::vector<float> data(2 * 3 * 3, 0.0f);

    data[0*9 + 0*3 + 0] = 1.0f;
    data[0*9 + 1*3 + 1] = 2.0f;
    data[0*9 + 2*3 + 2] = 3.0f;

    data[1*9 + 0*3 + 0] = 4.0f;
    data[1*9 + 1*3 + 1] = 5.0f;
    data[1*9 + 2*3 + 2] = 6.0f;

    T = data;

    auto result = math::eig(T, 200, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<std::vector<double>> expected =
    {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    const double tol = 5e-3;

    for (uint64_t b = 0; b < 2; ++b)
    {
        std::vector<double> got;
        got.reserve(3);
        for (uint64_t j = 0; j < 3; ++j)
        {
            Tensor<float> v_view(vals, std::vector<uint64_t>{b, j},
                std::vector<uint64_t>{1});
            got.push_back(static_cast<double>(static_cast<float>(v_view)));
        }
        std::sort(got.begin(), got.end());
        std::sort(expected[b].begin(), expected[b].end());

        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(got[i], expected[b][i], tol);
        }
    }
}

/**
 * @test EIG.eig_alias_view_strided
 * @brief eig() works on a strided alias/view and preserves owner memory.
 *
 * Creates a 10×10 owner matrix on device, builds a non-contiguous view that
 * selects every-other row/column (5×5) with a diagonal, computes eig() on the
 * view and verifies eigenvalues match the diagonal. Also verifies the owner
 * buffer is unmodified after the operation by copying back to host.
 */
TEST(EIG, eig_alias_view_strided)
{
    const uint64_t N = 10;
    const uint64_t n = 5;
    std::vector<float> owner_flat(N * N, 0.0f);

    uint64_t start_row = 0;
    uint64_t start_col = 1;
    for (uint64_t i = 0; i < n; ++i)
    {
        for (uint64_t j = 0; j < n; ++j)
        {
            size_t owner_index = static_cast<size_t>
                ((start_row + i * 2) * N + (start_col + j * 2));
            if (i == j)
            {
                owner_flat[owner_index] = static_cast<float>(10 * (i + 1));
            }
            else
            {
                owner_flat[owner_index] = 0.0f;
            }

        }
    }

    Tensor<float> owner({N, N}, MemoryLocation::DEVICE);
    owner = owner_flat;

    std::vector<uint64_t> start_indices = { start_row, start_col };
    std::vector<uint64_t> view_dims = { n, n };
    std::vector<uint64_t> view_strides = { N * 2, 2 };
    Tensor<float> view(owner, start_indices, view_dims, view_strides);

    EXPECT_NE(view.get_strides(), owner.get_strides());
    auto result = math::eig(view, 250, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<double> expected = {10.0, 20.0, 30.0, 40.0, 50.0};
    std::sort(expected.begin(), expected.end());

    std::vector<double> got;
    for (uint64_t i = 0; i < n; ++i)
    {
        Tensor<float> v_view(vals, std::vector<uint64_t>{i},
            std::vector<uint64_t>{1});
        got.push_back(static_cast<double>(static_cast<float>(v_view)));
    }
    std::sort(got.begin(), got.end());

    const double tol = 1e-4;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(got[i], expected[i], tol);
    }

    owner.to(MemoryLocation::HOST);
    const float* host_ptr = owner.get_data();
    for (size_t idx = 0; idx < owner_flat.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(owner_flat[idx], host_ptr[idx]);
    }
}

/**
 * @test EIG.eig_noncontig_batch_strides_device
 * @brief eig() on a non-contiguous batched view selects the correct batches.
 *
 * Constructs a 4-batch owner but creates a view that picks batches {0,2}
 * via nonstandard batch stride. Runs eig() on the view and compares per-batch
 * eigenvalues with expected values. Also ensures owner memory is unchanged.
 */
TEST(EIG, eig_noncontig_batch_strides_device)
{
    const uint64_t owner_batches = 4;
    const uint64_t batch_n = 3;
    const uint64_t owner_elems_per_batch = batch_n * batch_n;
    const uint64_t owner_total = owner_batches * owner_elems_per_batch;

    std::vector<float> owner_flat(owner_total, 0.0f);

    owner_flat[0 * owner_elems_per_batch + 0 * batch_n + 0] = 1.0f;
    owner_flat[0 * owner_elems_per_batch + 1 * batch_n + 1] = 2.0f;
    owner_flat[0 * owner_elems_per_batch + 2 * batch_n + 2] = 3.0f;

    owner_flat[2 * owner_elems_per_batch + 0 * batch_n + 0] = 7.0f;
    owner_flat[2 * owner_elems_per_batch + 1 * batch_n + 1] = 8.0f;
    owner_flat[2 * owner_elems_per_batch + 2 * batch_n + 2] = 9.0f;

    Tensor<float> owner({owner_batches, batch_n, batch_n},
        MemoryLocation::DEVICE);
    owner = owner_flat;

    std::vector<uint64_t> start_indices = { 0, 0, 0 };
    std::vector<uint64_t> view_dims = { 2, batch_n, batch_n };
    std::vector<uint64_t> view_strides =
        { owner_elems_per_batch * 2, batch_n, 1 };

    Tensor<float> view(owner, start_indices, view_dims, view_strides);

    EXPECT_NE(view.get_strides(), owner.get_strides());

    auto result = math::eig(view, 200, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<std::vector<double>> expected = {
        {1.0, 2.0, 3.0},
        {7.0, 8.0, 9.0}
    };

    const double tol = 5e-3;

    for (uint64_t b = 0; b < 2; ++b)
    {
        std::vector<double> got;
        got.reserve(batch_n);
        for (uint64_t j = 0; j < batch_n; ++j)
        {
            Tensor<float> v_view(vals, std::vector<uint64_t>{b, j},
                std::vector<uint64_t>{1});
            got.push_back(static_cast<double>(static_cast<float>(v_view)));
        }
        std::sort(got.begin(), got.end());
        std::sort(expected[b].begin(), expected[b].end());

        for (size_t i = 0; i < batch_n; ++i)
        {
            EXPECT_NEAR(got[i], expected[b][i], tol);
        }
    }

    owner.to(MemoryLocation::HOST);
    const float* host_ptr = owner.get_data();
    for (size_t idx = 0; idx < owner_flat.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(owner_flat[idx], host_ptr[idx]);
    }
}

/**
 * @test EIG.eig_5d_noncontig_device
 * @brief eig() handles higher-rank non-contiguous views with singleton dims.
 *
 * Uses a 5-D owner tensor `{4,1,1,3,3}`, selects two batches with custom
 * strides producing a non-contiguous view, computes eig() on the view and
 * checks per-batch eigenvalues match the expected diagonals (sorted, device
 * tolerance). Finally verifies owner memory integrity by copying to host.
 */
TEST(EIG, eig_5d_noncontig_device)
{
    const std::vector<uint64_t> owner_dims = {4, 1, 1, 3, 3};
    const uint64_t batch_count = owner_dims[0];
    const uint64_t rows = owner_dims[3];
    const uint64_t cols = owner_dims[4];
    ASSERT_EQ(rows, cols);

    const uint64_t elems_per_batch = rows * cols;
    const uint64_t owner_total = batch_count * elems_per_batch;

    std::vector<float> owner_flat(owner_total, 0.0f);

    {
        uint64_t batch_base = 0 * elems_per_batch;
        for (uint64_t d = 0; d < rows; ++d)
        {
            owner_flat[batch_base + d * cols + d] = static_cast<float>(d + 1);
        }
    }

    {
        uint64_t batch_base = 2 * elems_per_batch;
        for (uint64_t d = 0; d < rows; ++d)
        {
            owner_flat[batch_base + d * cols + d] = static_cast<float>(7 + d);
        }
    }

    Tensor<float> owner(owner_dims, MemoryLocation::DEVICE);
    owner = owner_flat;

    std::vector<uint64_t> start_indices = { 0, 0, 0, 0, 0 };
    std::vector<uint64_t> view_dims     = { 2, 1, 1, rows, cols };
    std::vector<uint64_t> view_strides  = { elems_per_batch * 2,
        elems_per_batch,
        elems_per_batch,
        static_cast<uint64_t>(cols),
        1 };

    Tensor<float> view(owner, start_indices, view_dims, view_strides);

    EXPECT_NE(view.get_strides(), owner.get_strides());

    auto result = math::eig(view, 200, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<std::vector<double>> expected = {
        {1.0, 2.0, 3.0},
        {7.0, 8.0, 9.0}
    };

    const double tol = 5e-3;

    for (uint64_t b = 0; b < 2; ++b)
    {
        std::vector<double> got;
        got.reserve(rows);
        for (uint64_t j = 0; j < rows; ++j)
        {
            const uint64_t vals_rank = vals.get_rank();
            std::vector<uint64_t> start(vals_rank, 0);
            start[0] = b;
            start[vals_rank - 1] = j;

            Tensor<float> v_view(vals, start, std::vector<uint64_t>{1});
            got.push_back(static_cast<double>(static_cast<float>(v_view)));
        }
        std::sort(got.begin(), got.end());
        std::sort(expected[b].begin(), expected[b].end());

        for (size_t i = 0; i < rows; ++i)
        {
            EXPECT_NEAR(got[i], expected[b][i], tol);
        }
    }

    owner.to(MemoryLocation::HOST);
    const float* host_ptr = owner.get_data();
    for (size_t idx = 0; idx < owner_flat.size(); ++idx)
    {
        EXPECT_FLOAT_EQ(owner_flat[idx], host_ptr[idx]);
    }
}

/**
 * @test SQRT.sqrt_zero_and_subnormal
 * @brief Sqrt handles zero and very small positive numbers (portable).
 */
TEST(SQRT, sqrt_zero_and_subnormal)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{0.0f, 1e-38f, 1e-20f};

    Tensor<float> res = math::sqrt(t);

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 0.0f);

    EXPECT_TRUE(std::isfinite(host[1]));
    EXPECT_GE(host[1], 0.0f);

    EXPECT_TRUE(std::isfinite(host[2]));
    EXPECT_GE(host[2], 0.0f);

    if (host[2] != 0.0f)
    {
        EXPECT_NEAR(host[2], std::sqrt(1e-20f), 1e-10f);
    }
}

/**
 * @test SQRT.sqrt_large_values
 * @brief Sqrt of large values (sanity check for finite result).
 */
TEST(SQRT, sqrt_large_values)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{1e30f, 1e20f};

    Tensor<float> res = math::sqrt(t);

    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 2 * sizeof(float)).wait();

    EXPECT_TRUE(std::isfinite(host[0]));
    EXPECT_TRUE(std::isfinite(host[1]));
}

/**
 * @test SQRT.sqrt_negative_throws
 * @brief Negative inputs produce NaN and should cause std::runtime_error.
 */
TEST(SQRT, sqrt_negative_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{4.0f, -1.0f, 9.0f};

    EXPECT_THROW(math::sqrt(t), std::runtime_error);
}

/**
 * @test SQRT.sqrt_nan_throws
 * @brief NaN elements in input cause std::runtime_error.
 */
TEST(SQRT, sqrt_nan_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN()};

    EXPECT_THROW(math::sqrt(t), std::runtime_error);
}

/**
 * @test SQRT.sqrt_inf_throws
 * @brief +Inf in input causes non-finite output and should throw.
 */
TEST(SQRT, sqrt_inf_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{std::numeric_limits<float>::infinity(), 4.0f};

    EXPECT_THROW(math::sqrt(t), std::runtime_error);
}

/**
 * @test SQRT.sqrt_empty_throws
 * @brief sqrt() on an empty tensor should throw std::invalid_argument.
 */
TEST(SQRT, sqrt_empty_throws)
{
    Tensor<float> t;
    EXPECT_THROW(math::sqrt(t), std::invalid_argument);
}

/**
 * @test SQRT.sqrt_view_and_alias
 * @brief Elementwise sqrt on views and alias views (strided/contiguous).
 */
TEST(SQRT, sqrt_view_and_alias)
{
    Tensor<float> owner({2,3}, MemoryLocation::DEVICE);
    owner = std::vector<float>{1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

    Tensor<float> view(owner,
                       std::vector<uint64_t>{0ull, 0ull},
                       std::vector<uint64_t>{3ull},
                       std::vector<uint64_t>{1ull});
    Tensor<float> r1 = math::sqrt(view);
    std::vector<float> h1(3);
    g_sycl_queue.memcpy(h1.data(), r1.m_p_data.get(), 3 * sizeof(float)).wait();
    EXPECT_FLOAT_EQ(h1[0], 1.0f);
    EXPECT_FLOAT_EQ(h1[1], 2.0f);
    EXPECT_FLOAT_EQ(h1[2], 3.0f);

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> alias(owner, start, dims, strides);
    Tensor<float> r2 = math::sqrt(alias);
    std::vector<float> h2(3);
    g_sycl_queue.memcpy(h2.data(), r2.m_p_data.get(), 3 * sizeof(float)).wait();
    EXPECT_FLOAT_EQ(h2[0], 1.0f);
    EXPECT_FLOAT_EQ(h2[1], 3.0f);
    EXPECT_FLOAT_EQ(h2[2], 5.0f);
}

/**
 * @test EXP.exp_basic
 * @brief Elementwise exp on simple values (0, 1, -1) on device.
 */
TEST(EXP, exp_basic)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    t = std::vector<float>{0.0f, 1.0f, -1.0f};

    Tensor<float> res = math::exp(t);

    std::vector<float> host(3);
    g_sycl_queue.memcpy(host.data(),
        res.m_p_data.get(), 3 * sizeof(float)).wait();

    const double tol = 1e-6;
    EXPECT_NEAR(static_cast<double>(host[0]),
        static_cast<double>(std::exp(0.0f)), tol);
    EXPECT_NEAR(static_cast<double>(host[1]),
        static_cast<double>(std::exp(1.0f)), tol);
    EXPECT_NEAR(static_cast<double>(host[2]),
        static_cast<double>(std::exp(-1.0f)), tol);
}

/**
 * @test EXP.exp_view_and_alias
 * @brief Elementwise exp on views and alias views (strided/contiguous).
 */
TEST(EXP, exp_view_and_alias)
{
    Tensor<float> owner({2,3}, MemoryLocation::DEVICE);
    owner = std::vector<float>{0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 3.0f};

    Tensor<float> view(owner,
                       std::vector<uint64_t>{0ull, 0ull},
                       std::vector<uint64_t>{3ull},
                       std::vector<uint64_t>{1ull});
    Tensor<float> r1 = math::exp(view);
    std::vector<float> h1(3);
    g_sycl_queue.memcpy(h1.data(), r1.m_p_data.get(), 3 * sizeof(float)).wait();
    EXPECT_NEAR(static_cast<double>(h1[0]),
        static_cast<double>(std::exp(0.0f)), 1e-6);
    EXPECT_NEAR(static_cast<double>(h1[1]),
        static_cast<double>(std::exp(1.0f)), 1e-6);
    EXPECT_NEAR(static_cast<double>(h1[2]),
        static_cast<double>(std::exp(2.0f)), 1e-6);

    std::vector<uint64_t> start = {0ull, 0ull};
    std::vector<uint64_t> dims = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> alias(owner, start, dims, strides);
    Tensor<float> r2 = math::exp(alias);
    std::vector<float> h2(3);
    g_sycl_queue.memcpy(h2.data(),
        r2.m_p_data.get(), 3 * sizeof(float)).wait();
    EXPECT_NEAR(static_cast<double>(h2[0]),
        static_cast<double>(std::exp(0.0f)), 1e-6);
    EXPECT_NEAR(static_cast<double>(h2[1]),
        static_cast<double>(std::exp(2.0f)), 1e-6);
    EXPECT_NEAR(static_cast<double>(h2[2]),
        static_cast<double>(std::exp(-2.0f)), 1e-6);
}

/**
 * @test EXP.exp_large_overflow_throws
 * @brief Very large inputs that overflow to +Inf should cause runtime error.
 */
TEST(EXP, exp_large_overflow_throws)
{
    Tensor<float> t({1}, MemoryLocation::DEVICE);
    t = std::vector<float>{1000.0f};

    EXPECT_THROW(math::exp(t), std::runtime_error);
}

/**
 * @test EXP.exp_nan_throws
 * @brief NaN in input should cause std::runtime_error.
 */
TEST(EXP, exp_nan_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    t = std::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN()};

    EXPECT_THROW(math::exp(t), std::runtime_error);
}

/**
 * @test EXP.exp_inf_input_throws
 * @brief +Inf in input leads to non-finite output and should throw.
 */
TEST(EXP, exp_inf_input_throws)
{
    Tensor<float> t({1}, MemoryLocation::DEVICE);
    t = std::vector<float>{std::numeric_limits<float>::infinity()};

    EXPECT_THROW(math::exp(t), std::runtime_error);
}

/**
 * @test EXP.exp_empty_throws
 * @brief exp() on an empty tensor should throw std::invalid_argument.
 */
TEST(EXP, exp_empty_throws)
{
    Tensor<float> t;
    EXPECT_THROW(math::exp(t), std::invalid_argument);
}

} // namespace Test