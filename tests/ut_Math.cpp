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

    Tensor<float> res = math::sum(t, -1);

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

    Tensor<float> res = math::sum(view, -1);

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

    Tensor<float> res = math::sum(alias_view, -1);

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

    EXPECT_THROW(math::sum(t, -1), std::runtime_error);
}

/**
 * @test SUM.sum_empty
 * @brief Summing an empty tensor returns a scalar tensor containing 0.0.
 */
TEST(SUM, sum_empty)
{
    Tensor<float> t;

    Tensor<float> res({1}, MemoryLocation::DEVICE);
    res = math::sum(t, -1);

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

    Tensor<float> res = t.cumsum(-1);

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
    Tensor<float> res = math::cumsum(t, -1);

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

    Tensor<float> res = math::cumsum(view, -1);

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

    Tensor<float> res = math::cumsum(alias_view, -1);

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
    EXPECT_THROW(math::cumsum(t, -2), std::invalid_argument);
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

} // namespace Test