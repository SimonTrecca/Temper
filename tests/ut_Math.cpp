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

    std::vector<uint64_t> res = math::argmax(t, -1);

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
    EXPECT_THROW(math::argmax(t, -2), std::invalid_argument);
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

    EXPECT_THROW(math::argmax(t, -1), std::runtime_error);
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

    std::vector<uint64_t> res = math::argmax(v, -1);

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
    std::vector<uint64_t> res = math::argmax(v, -1);

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

    std::vector<uint64_t> res = math::argmax(v, -1);

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

    for (uint64_t i = 0; i < B0; ++i) {
        for (uint64_t j = 0; j < B1; ++j) {
            float a = start_vals[i];
            float b = stop_vals[j];
            float step = (b - a) / static_cast<float>(num - 1);
            for (uint64_t p = 0; p < N; ++p) {
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


} // namespace Test