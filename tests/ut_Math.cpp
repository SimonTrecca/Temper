/**
 * @file ut_Math.cpp
 * @brief Google Test suite for mathematical tensor operations.
 *
 * Contains unit tests ensuring correctness of functions implemented
 * in the Math module.
 */

#include <gtest/gtest.h>
#include <type_traits>

#define private public
#define protected public
#include "temper/Math.hpp"
#undef private
#undef protected

using namespace temper;

namespace Test
{

template<typename T>
class TypedMatmul : public ::testing::Test {};

using MatmulTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedMatmul, MatmulTestTypes);

/**
 * @test TypedMatmul.basic_2d
 * @brief Basic 2x3 @ 3x2 = 2x2 check.
 */
TYPED_TEST(TypedMatmul, basic_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    A = a_vals;

    Tensor<value_t> B({3, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(7),  static_cast<value_t>(8),
        static_cast<value_t>(9),  static_cast<value_t>(10),
        static_cast<value_t>(11), static_cast<value_t>(12)
    };
    B = b_vals;

    Tensor<value_t> R = math::matmul<value_t>(A, B);

    std::vector<value_t> expected(4, static_cast<value_t>(0));
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            value_t s = static_cast<value_t>(0);
            for (uint64_t k = 0; k < 3; ++k)
            {
                s += a_vals[i * 3 + k] * b_vals[k * 2 + j];
            }
            expected[i * 2 + j] = s;
        }
    }

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
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
 * @test TypedMatmul.vec_vec_dot
 * @brief 1D x 1D -> scalar (dot product).
 */
TYPED_TEST(TypedMatmul, vec_vec_dot)
{
    using value_t = TypeParam;
    Tensor<value_t> a({3}, MemoryLocation::DEVICE);
    std::vector<value_t> av = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3)
    };
    a = av;

    Tensor<value_t> b({3}, MemoryLocation::DEVICE);
    std::vector<value_t> bv = {
        static_cast<value_t>(4), static_cast<value_t>(5),
        static_cast<value_t>(6)
    };
    b = bv;

    Tensor<value_t> r = math::matmul<value_t>(a, b);

    value_t expect = static_cast<value_t>(0);
    for (size_t idx = 0; idx < av.size(); ++idx)
        expect += av[idx] * bv[idx];

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), r.m_p_data.get(),
                        sizeof(value_t)).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(expect));
    }
    else
    {
        EXPECT_EQ(host[0], expect);
    }
}

/**
 * @test TypedMatmul.vec_mat
 * @brief 1D x 2D -> vector.
 */
TYPED_TEST(TypedMatmul, vec_mat)
{
    using value_t = TypeParam;
    Tensor<value_t> a({2}, MemoryLocation::DEVICE);
    std::vector<value_t> av = {
        static_cast<value_t>(1), static_cast<value_t>(2)
    };
    a = av;

    Tensor<value_t> B({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> bv = {
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(7), static_cast<value_t>(8)
    };
    B = bv;

    Tensor<value_t> R = math::matmul<value_t>(a, B);

    std::vector<value_t> expected(3, static_cast<value_t>(0));
    for (uint64_t j = 0; j < 3; ++j)
    {
        value_t s = static_cast<value_t>(0);
        for (uint64_t idx = 0; idx < 2; ++idx)
            s += av[idx] * bv[idx * 3 + j];
        expected[j] = s;
    }

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        3 * sizeof(value_t)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
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
 * @test TypedMatmul.mat_vec
 * @brief 2D x 1D -> vector.
 */
TYPED_TEST(TypedMatmul, mat_vec)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    A = a_vals;

    Tensor<value_t> v({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vv = {
        static_cast<value_t>(7), static_cast<value_t>(8),
        static_cast<value_t>(9)
    };
    v = vv;

    Tensor<value_t> R = math::matmul<value_t>(A, v);

    std::vector<value_t> expected(2, static_cast<value_t>(0));
    for (uint64_t i = 0; i < 2; ++i)
    {
        value_t s = static_cast<value_t>(0);
        for (uint64_t k = 0; k < 3; ++k)
            s += a_vals[i * 3 + k] * vv[k];
        expected[i] = s;
    }

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        2 * sizeof(value_t)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
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
 * @test TypedMatmul.batched_equal_batches
 * @brief Batched matmul with matching batch dims:
 *        (2,3,4) @ (2,4,5) -> (2,3,5)
 */
TYPED_TEST(TypedMatmul, batched_equal_batches)
{
    using value_t = TypeParam;
    const uint64_t B = 2, M = 3, K = 4, N = 5;
    Tensor<value_t> A({B, M, K}, MemoryLocation::DEVICE);
    Tensor<value_t> Bt({B, K, N}, MemoryLocation::DEVICE);

    std::vector<value_t> a_vals(B * M * K);
    std::vector<value_t> b_vals(B * K * N);

    for (uint64_t i = 0; i < a_vals.size(); ++i)
        a_vals[i] = static_cast<value_t>(i + 1);
    for (uint64_t i = 0; i < b_vals.size(); ++i)
        b_vals[i] = static_cast<value_t>(i + 1 + a_vals.size());

    A = a_vals;
    Bt = b_vals;

    Tensor<value_t> R = math::matmul<value_t>(A, Bt);

    std::vector<value_t> expected(B * M * N, static_cast<value_t>(0));
    for (uint64_t b = 0; b < B; ++b)
    {
        for (uint64_t i = 0; i < M; ++i)
        {
            for (uint64_t j = 0; j < N; ++j)
            {
                value_t s = static_cast<value_t>(0);
                for (uint64_t idx = 0; idx < K; ++idx)
                {
                    uint64_t a_idx = ((b * M) + i) * K + idx;
                    uint64_t b_idx = ((b * K) + idx) * N + j;
                    s += a_vals[a_idx] * b_vals[b_idx];
                }
                uint64_t r_idx = ((b * M) + i) * N + j;
                expected[r_idx] = s;
            }
        }
    }

    std::vector<value_t> host(B * M * N);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
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
 * @test TypedMatmul.batched_broadcast_batches
 * @brief Batched matmul with broadcasting:
 *        A:(2,1,2,3) * B:(1,3,3,2) -> out:(2,3,2,2)
 */
TYPED_TEST(TypedMatmul, batched_broadcast_batches)
{
    using value_t = TypeParam;
    const std::vector<uint64_t> a_shape = {2, 1, 2, 3};
    const std::vector<uint64_t> b_shape = {1, 3, 3, 2};

    Tensor<value_t> A(a_shape, MemoryLocation::DEVICE);
    Tensor<value_t> Bt(b_shape, MemoryLocation::DEVICE);

    uint64_t a_elems = A.get_num_elements();
    uint64_t b_elems = Bt.get_num_elements();
    std::vector<value_t> a_vals(a_elems);
    std::vector<value_t> b_vals(b_elems);

    for (uint64_t i = 0; i < a_elems; ++i)
        a_vals[i] = static_cast<value_t>(i + 1);
    for (uint64_t i = 0; i < b_elems; ++i)
        b_vals[i] = static_cast<value_t>(i + 101);

    A = a_vals;
    Bt = b_vals;

    Tensor<value_t> R = math::matmul<value_t>(A, Bt);

    const uint64_t B0 = 2, B1 = 3, M = 2, K = 3, N = 2;
    std::vector<value_t> expected(B0 * B1 * M * N,
                                  static_cast<value_t>(0));

    for (uint64_t b0 = 0; b0 < B0; ++b0)
    {
        for (uint64_t b1 = 0; b1 < B1; ++b1)
        {
            for (uint64_t i = 0; i < M; ++i)
            {
                for (uint64_t j = 0; j < N; ++j)
                {
                    value_t s = static_cast<value_t>(0);
                    for (uint64_t idx = 0; idx < K; ++idx)
                    {
                        uint64_t a_idx = (b0 * 2 + i) * 3 + idx;
                        uint64_t b_idx = ((b1) * 3 + idx) * 2 + j;
                        s += a_vals[a_idx] * b_vals[b_idx];
                    }
                    uint64_t out_idx = (((b0 * B1) + b1) * M + i) * N + j;
                    expected[out_idx] = s;
                }
            }
        }
    }

    std::vector<value_t> host(B0 * B1 * M * N);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
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
 * @test TypedMatmul.empty_throws
 * @brief matmul should throw std::invalid_argument if inputs are empty.
 */
TYPED_TEST(TypedMatmul, empty_throws)
{
    using value_t = TypeParam;

    Tensor<value_t> A;

    Tensor<value_t> B({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    B = b_vals;

    Tensor<value_t> B2;

    Tensor<value_t> A2({2, 2}, MemoryLocation::DEVICE);
    A2 = b_vals;

    EXPECT_THROW(math::matmul<value_t>(A, B), std::invalid_argument);
    EXPECT_THROW(math::matmul<value_t>(A2, B2), std::invalid_argument);

}

/**
 * @test TypedMatmul.nan_throws
 * @brief matmul should throw std::runtime_error if inputs contain NaN.
 */
TYPED_TEST(TypedMatmul, nan_throws)
{
    using value_t = TypeParam;
    if constexpr (!std::is_floating_point<value_t>::value)
    {
        // integral types cannot represent NaN; skip the test.
        return;
    }

    Tensor<value_t> A({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        static_cast<value_t>(1),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    A = a_vals;

    Tensor<value_t> B({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    B = b_vals;

    EXPECT_THROW(math::matmul<value_t>(A, B), std::runtime_error);
}

/**
 * @test TypedMatmul.inf_throws
 * @brief matmul should throw std::runtime_error if result is non-finite.
 */
TYPED_TEST(TypedMatmul, inf_throws)
{
    using value_t = TypeParam;
    if constexpr (!std::is_floating_point<value_t>::value)
    {
        // integral types cannot represent Inf; skip the test.
        return;
    }

    Tensor<value_t> A({1, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        std::numeric_limits<value_t>::infinity(),
        static_cast<value_t>(2)
    };
    A = a_vals;

    Tensor<value_t> B({2, 1}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    B = b_vals;

    EXPECT_THROW(math::matmul<value_t>(A, B), std::runtime_error);
}

template<typename T>
class TypedReshape : public ::testing::Test {};

using ReshapeTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedReshape, ReshapeTestTypes);

/**
 * @test TypedReshape.reshape_free_function_basic
 * @brief Free reshape() creates an owning clone, reshapes its
 *        dimensions, and leaves the source tensor unchanged.
 */
TYPED_TEST(TypedReshape, reshape_free_function_basic)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2, 3}, MemoryLocation::HOST);

    std::vector<value_t> src_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    src = src_vals;

    Tensor<value_t> r = math::reshape<value_t>(src, {3, 2});

    EXPECT_EQ(src.get_dimensions(),
              (std::vector<uint64_t>{2, 3}));

    EXPECT_EQ(r.get_dimensions(),
              (std::vector<uint64_t>{3, 2}));

    for (uint64_t i = 0; i < src.get_num_elements(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(r.get_data()[i]),
                            static_cast<double>(src.get_data()[i]));
        }
        else
        {
            EXPECT_EQ(r.get_data()[i], src.get_data()[i]);
        }
    }

    EXPECT_NE(r.get_data(), src.get_data());
}

/**
 * @test TypedReshape.reshape_free_function_flat
 * @brief Reshaping to 1D with free reshape() preserves contents.
 */
TYPED_TEST(TypedReshape, reshape_free_function_flat)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2, 3}, MemoryLocation::HOST);

    std::vector<value_t> src_vals = {
        static_cast<value_t>(10), static_cast<value_t>(11),
        static_cast<value_t>(12), static_cast<value_t>(13),
        static_cast<value_t>(14), static_cast<value_t>(15)
    };
    src = src_vals;

    Tensor<value_t> r = math::reshape<value_t>(src, {6});

    EXPECT_EQ(r.get_dimensions(),
              (std::vector<uint64_t>{6}));

    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(r[i]),
                            static_cast<double>(src.get_data()[i]));
        }
        else
        {
            EXPECT_EQ(r[i], src.get_data()[i]);
        }
    }
}

/**
 * @test TypedReshape.reshape_free_function_invalid_size
 * @brief Free reshape() must throw if new dimensions don't match
 *        element count.
 */
TYPED_TEST(TypedReshape, reshape_free_function_invalid_size)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2, 3}, MemoryLocation::HOST);

    std::vector<value_t> src_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    src = src_vals;

    EXPECT_THROW(
        { auto r = math::reshape<value_t>(src, {4, 2}); },
        std::invalid_argument
    );
}

/**
 * @test TypedReshape.reshape_free_function_from_view
 * @brief Reshaping a view via free reshape() is valid (clone first),
 *        produces an owning tensor.
 */
TYPED_TEST(TypedReshape, reshape_free_function_from_view)
{
    using value_t = TypeParam;
    Tensor<value_t> base({4}, MemoryLocation::HOST);

    std::vector<value_t> base_vals = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3)
    };
    base = base_vals;

    Tensor<value_t> v(base, {0}, {4}, {1});
    EXPECT_FALSE(v.get_owns_data());

    Tensor<value_t> r = math::reshape<value_t>(v, {2, 2});

    EXPECT_TRUE(r.get_owns_data());
    EXPECT_EQ(r.get_dimensions(),
              (std::vector<uint64_t>{2, 2}));

    for (uint64_t i = 0; i < 4; ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(r.get_data()[i]),
                            static_cast<double>(base.get_data()[i]));
        }
        else
        {
            EXPECT_EQ(r.get_data()[i], base.get_data()[i]);
        }
    }
}

/**
 * @test TypedReshape.reshape_free_function_empty_tensor
 * @brief Reshaping an empty tensor through free reshape() throws.
 */
TYPED_TEST(TypedReshape, reshape_free_function_empty_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> empty;

    EXPECT_THROW(
        { auto r = math::reshape<value_t>(empty, {1}); },
        std::invalid_argument
    );
}

template<typename T>
class TypedSort : public ::testing::Test {};

using SortTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedSort, SortTestTypes);

/**
 * @test TypedSort.sort_function_independence
 * @brief Free function sort should not mutate the input tensor.
 */
TYPED_TEST(TypedSort, sort_function_independence)
{
    using value_t = TypeParam;

    Tensor<value_t> t({5}, MemoryLocation::HOST);

    std::vector<value_t> vals = {
        static_cast<value_t>(3), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(5),
        static_cast<value_t>(4)
    };
    t = vals;

    Tensor<value_t> out = math::sort<value_t>(t, 0);

    for (size_t i = 0; i < vals.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(t[i]),
                            static_cast<double>(vals[i]));
        else
            EXPECT_EQ(t[i], vals[i]);
    }

    std::vector<value_t> expected = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5)
    };
    for (size_t i = 0; i < vals.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(out[i]),
                            static_cast<double>(expected[i]));
        else
            EXPECT_EQ(out[i], expected[i]);
    }
}

/**
 * @test TypedSort.sort_function_axis1
 * @brief Sorting a 2D tensor along axis 1 via free function.
 */
TYPED_TEST(TypedSort, sort_function_axis1)
{
    using value_t = TypeParam;

    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        // [[3,1,2],
        //  [0,-1,5]]
        t = std::vector<value_t>{
            static_cast<value_t>(3), static_cast<value_t>(1),
            static_cast<value_t>(2), static_cast<value_t>(0),
            static_cast<value_t>(-1), static_cast<value_t>(5)
        };
    }
    else
    {
        // avoid negative values for unsigned tests
        t = std::vector<value_t>{
            static_cast<value_t>(3), static_cast<value_t>(1),
            static_cast<value_t>(2), static_cast<value_t>(0),
            static_cast<value_t>(1), static_cast<value_t>(5)
        };
    }

    Tensor<value_t> out = math::sort<value_t>(t, 1);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(out[0][0], static_cast<value_t>(1.0));
        EXPECT_FLOAT_EQ(out[0][1], static_cast<value_t>(2.0));
        EXPECT_FLOAT_EQ(out[0][2], static_cast<value_t>(3.0));
        EXPECT_FLOAT_EQ(out[1][0], static_cast<value_t>(-1.0));
        EXPECT_FLOAT_EQ(out[1][1], static_cast<value_t>(0.0));
        EXPECT_FLOAT_EQ(out[1][2], static_cast<value_t>(5.0));
    }
    else
    {
        EXPECT_EQ(out[0][0], static_cast<value_t>(1));
        EXPECT_EQ(out[0][1], static_cast<value_t>(2));
        EXPECT_EQ(out[0][2], static_cast<value_t>(3));
        EXPECT_EQ(out[1][0], static_cast<value_t>(0));
        EXPECT_EQ(out[1][1], static_cast<value_t>(1));
        EXPECT_EQ(out[1][2], static_cast<value_t>(5));
    }

    // original must remain unchanged
    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(t[0][0], static_cast<value_t>(3.0));
        EXPECT_FLOAT_EQ(t[0][1], static_cast<value_t>(1.0));
        EXPECT_FLOAT_EQ(t[0][2], static_cast<value_t>(2.0));
    }
    else
    {
        EXPECT_EQ(t[0][0], static_cast<value_t>(3));
        EXPECT_EQ(t[0][1], static_cast<value_t>(1));
        EXPECT_EQ(t[0][2], static_cast<value_t>(2));
    }
}

/**
 * @test TypedSort.sort_function_axis_out_of_bounds
 * @brief Free function sort should throw for invalid axis.
 */
TYPED_TEST(TypedSort, sort_function_axis_out_of_bounds)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::HOST);

    EXPECT_THROW(math::sort<value_t>(t, 1), std::invalid_argument);
    EXPECT_THROW(math::sort<value_t>(t, -2), std::invalid_argument);
}

template<typename T>
class TypedSum : public ::testing::Test {};

using SumTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedSum, SumTestTypes);

/**
 * @test TypedSum.sum_all_elements
 * @brief Sum all elements (axis = -1) on a device tensor and return
 * a scalar with the correct total value.
 */
TYPED_TEST(TypedSum, sum_all_elements)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3)
    };
    t = vals;

    Tensor<value_t> res = math::sum<value_t>(t);

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t)).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(static_cast<value_t>(6)));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(6));
    }
}

/**
 * @test TypedSum.sum_axis0
 * @brief Sum along axis 0 for a 2x3 tensor stored on device and
 * verify per-column sums.
 */
TYPED_TEST(TypedSum, sum_axis0)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    Tensor<value_t> res = math::sum<value_t>(t, 0);

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        3 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(1) + static_cast<value_t>(4);
    auto e1 = static_cast<value_t>(2) + static_cast<value_t>(5);
    auto e2 = static_cast<value_t>(3) + static_cast<value_t>(6);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
    }
}

/**
 * @test TypedSum.sum_axis1
 * @brief Sum along axis 1 for a 2x3 device tensor and verify per-row
 * sums.
 */
TYPED_TEST(TypedSum, sum_axis1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    Tensor<value_t> res = math::sum<value_t>(t, 1);

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        2 * sizeof(value_t)).wait();

    auto r0 = static_cast<value_t>(1) + static_cast<value_t>(2) +
              static_cast<value_t>(3);
    auto r1 = static_cast<value_t>(4) + static_cast<value_t>(5) +
              static_cast<value_t>(6);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(r0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(r1));
    }
    else
    {
        EXPECT_EQ(host[0], r0);
        EXPECT_EQ(host[1], r1);
    }
}

/**
 * @test TypedSum.sum_axis0_3D
 * @brief Sum along axis 0 for a 2x2x2 device tensor and verify resulting
 * values.
 */
TYPED_TEST(TypedSum, sum_axis0_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(7), static_cast<value_t>(8)
    };
    t = vals;

    Tensor<value_t> res = math::sum<value_t>(t, 0);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        4 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(1) + static_cast<value_t>(5);
    auto e1 = static_cast<value_t>(2) + static_cast<value_t>(6);
    auto e2 = static_cast<value_t>(3) + static_cast<value_t>(7);
    auto e3 = static_cast<value_t>(4) + static_cast<value_t>(8);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(e3));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
        EXPECT_EQ(host[3], e3);
    }
}

/**
 * @test TypedSum.sum_axis_negative
 * @brief Sum along axis -3 for a 2x2x2 device tensor and verify
 * resulting values.
 */
TYPED_TEST(TypedSum, sum_axis_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(7), static_cast<value_t>(8)
    };
    t = vals;

    Tensor<value_t> res = math::sum<value_t>(t, -3);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        4 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(1) + static_cast<value_t>(5);
    auto e1 = static_cast<value_t>(2) + static_cast<value_t>(6);
    auto e2 = static_cast<value_t>(3) + static_cast<value_t>(7);
    auto e3 = static_cast<value_t>(4) + static_cast<value_t>(8);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(e3));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
        EXPECT_EQ(host[3], e3);
    }
}

/**
 * @test TypedSum.sum_axis1_3D
 * @brief Sum along axis 1 for a 2x2x2 device tensor and verify
 * resulting values.
 */
TYPED_TEST(TypedSum, sum_axis1_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(7), static_cast<value_t>(8)
    };
    t = vals;

    Tensor<value_t> res = math::sum<value_t>(t, 1);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        4 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(1) + static_cast<value_t>(3);
    auto e1 = static_cast<value_t>(2) + static_cast<value_t>(4);
    auto e2 = static_cast<value_t>(5) + static_cast<value_t>(7);
    auto e3 = static_cast<value_t>(6) + static_cast<value_t>(8);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(e3));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
        EXPECT_EQ(host[3], e3);
    }
}

/**
 * @test TypedSum.sum_axis2_3D
 * @brief Sum along axis 2 for a 2x2x2 device tensor and verify
 * resulting values.
 */
TYPED_TEST(TypedSum, sum_axis2_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(7), static_cast<value_t>(8)
    };
    t = vals;

    Tensor<value_t> res = math::sum<value_t>(t, 2);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        4 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(1) + static_cast<value_t>(2);
    auto e1 = static_cast<value_t>(3) + static_cast<value_t>(4);
    auto e2 = static_cast<value_t>(5) + static_cast<value_t>(6);
    auto e3 = static_cast<value_t>(7) + static_cast<value_t>(8);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(e3));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
        EXPECT_EQ(host[3], e3);
    }
}

/**
 * @test TypedSum.sum_view_tensor
 * @brief Sum all elements (axis = -1) of a view into a device tensor
 * and verify the scalar result.
 */
TYPED_TEST(TypedSum, sum_view_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 0ull};
    std::vector<uint64_t> view_shape    = {3ull};

    Tensor<value_t> view(t, start_indices, view_shape);

    Tensor<value_t> res = math::sum<value_t>(view);

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t)).wait();

    auto expect = static_cast<value_t>(1) + static_cast<value_t>(2) +
                  static_cast<value_t>(3);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(expect));
    }
    else
    {
        EXPECT_EQ(host[0], expect);
    }
}

/**
 * @test TypedSum.sum_alias_view_tensor
 * @brief Sum all elements (axis = -1) of an alias view with non-unit
 * stride and verify result.
 */
TYPED_TEST(TypedSum, sum_alias_view_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t({6}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    std::vector<uint64_t> start_indices = {0ull};
    std::vector<uint64_t> dims          = {3ull};
    std::vector<uint64_t> strides       = {2ull};

    Tensor<value_t> alias_view(t, start_indices, dims, strides);

    Tensor<value_t> res = math::sum<value_t>(alias_view);

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t)).wait();

    auto expect = static_cast<value_t>(1) + static_cast<value_t>(3) +
                  static_cast<value_t>(5);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(expect));
    }
    else
    {
        EXPECT_EQ(host[0], expect);
    }
}

/**
 * @test TypedSum.sum_view_tensor_3d_axis1
 * @brief Sum along axis 1 on a 3D view and verify the produced values.
 */
TYPED_TEST(TypedSum, sum_view_tensor_3d_axis1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 4, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<value_t>(i + 1);
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 1ull, 0ull};
    std::vector<uint64_t> view_shape    = {2ull, 2ull};
    Tensor<value_t> view(t, start_indices, view_shape);

    Tensor<value_t> res = math::sum<value_t>(view, 1);

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t) * host.size()).wait();

    auto e0 = static_cast<value_t>(23);
    auto e1 = static_cast<value_t>(27);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
    }
}

/**
 * @test TypedSum.sum_alias_view_tensor_2d_strided
 * @brief Sum along axis 0 on a 2D alias view with custom strides and
 * verify each output element.
 */
TYPED_TEST(TypedSum, sum_alias_view_tensor_2d_strided)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 5}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(20);
    for (uint64_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<value_t>(i + 1);
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 1ull};
    std::vector<uint64_t> dims          = {2ull, 3ull};
    std::vector<uint64_t> strides       = {5ull, 2ull};
    Tensor<value_t> alias_view(t, start_indices, dims, strides);

    Tensor<value_t> res = math::sum<value_t>(alias_view, 0);

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t) * host.size()).wait();

    auto e0 = static_cast<value_t>(9);
    auto e1 = static_cast<value_t>(13);
    auto e2 = static_cast<value_t>(17);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
    }
}

/**
 * @test TypedSum.sum_alias_view_tensor_overlapping_stride_zero
 * @brief Sum along axis 0 on an alias view that contains overlapping
 * elements via a zero stride and verify the sums account for repeated
 * elements.
 */
TYPED_TEST(TypedSum, sum_alias_view_tensor_overlapping_stride_zero)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 0ull};
    std::vector<uint64_t> dims          = {2ull, 2ull};
    std::vector<uint64_t> strides       = {0ull, 1ull};
    Tensor<value_t> alias_view(t, start_indices, dims, strides);

    Tensor<value_t> res = math::sum<value_t>(alias_view, 0);

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t) * host.size()).wait();

    auto e0 = static_cast<value_t>(8);
    auto e1 = static_cast<value_t>(10);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
    }
}

/**
 * @test TypedSum.sum_nan_throws
 * @brief Tests that sum throws std::runtime_error when the tensor
 * contains NaN values.
 */
TYPED_TEST(TypedSum, sum_nan_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
        return;

    Tensor<value_t> t({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(3)
    };
    t = vals;

    EXPECT_THROW(math::sum<value_t>(t, -1), std::runtime_error);
}

/**
 * @test TypedSum.sum_non_finite_throws
 * @brief Tests that sum throws std::runtime_error when the tensor
 * contains non-finite values (infinity).
 */
TYPED_TEST(TypedSum, sum_non_finite_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
        return;

    Tensor<value_t> t({2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        std::numeric_limits<value_t>::infinity(),
        static_cast<value_t>(1)
    };
    t = vals;

    EXPECT_THROW(math::sum<value_t>(t), std::runtime_error);
}

/**
 * @test TypedSum.sum_empty
 * @brief Summing an empty tensor returns a scalar tensor containing 0.0.
 */
TYPED_TEST(TypedSum, sum_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;

    Tensor<value_t> res({1}, MemoryLocation::DEVICE);
    res = math::sum<value_t>(t);

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t)).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(static_cast<value_t>(0)));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(0));
    }
}

template<typename T>
class TypedCumsum : public ::testing::Test {};

using CumsumTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedCumsum, CumsumTestTypes);

/**
 * @test TypedCumsum.cumsum_all_elements_flatten
 * @brief Tests cumsum on a 1D tensor, flattening all elements.
 */
TYPED_TEST(TypedCumsum, cumsum_all_elements_flatten)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3)
    };
    t = vals;

    Tensor<value_t> res = math::cumsum<value_t>(t);

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        3 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(1);
    auto e1 = static_cast<value_t>(1) + static_cast<value_t>(2);
    auto e2 = static_cast<value_t>(1) + static_cast<value_t>(2) +
              static_cast<value_t>(3);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
    }
}

/**
 * @test TypedCumsum.cumsum_axis0_2D
 * @brief Tests cumsum along axis 0 of a 2D tensor.
 */
TYPED_TEST(TypedCumsum, cumsum_axis0_2D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    Tensor<value_t> res = math::cumsum<value_t>(t, 0);

    std::vector<value_t> host(6);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        6 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(1.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(2.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(3.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(1.0 + 4.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[4]),
                        static_cast<double>(2.0 + 5.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[5]),
                        static_cast<double>(3.0 + 6.0));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(2));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(1 + 4));
        EXPECT_EQ(host[4], static_cast<value_t>(2 + 5));
        EXPECT_EQ(host[5], static_cast<value_t>(3 + 6));
    }
}

/**
 * @test TypedCumsum.cumsum_axis_negative
 * @brief Tests cumsum along axis -2 of a 2D tensor.
 */
TYPED_TEST(TypedCumsum, cumsum_axis_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    Tensor<value_t> res = math::cumsum<value_t>(t, -2);

    std::vector<value_t> host(6);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        6 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(1.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(2.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(3.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(1.0 + 4.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[4]),
                        static_cast<double>(2.0 + 5.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[5]),
                        static_cast<double>(3.0 + 6.0));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(2));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(1 + 4));
        EXPECT_EQ(host[4], static_cast<value_t>(2 + 5));
        EXPECT_EQ(host[5], static_cast<value_t>(3 + 6));
    }
}

/**
 * @test TypedCumsum.cumsum_axis1_2D
 * @brief Tests cumsum along axis 1 of a 2D tensor.
 */
TYPED_TEST(TypedCumsum, cumsum_axis1_2D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    Tensor<value_t> res = math::cumsum<value_t>(t, 1);

    std::vector<value_t> host(6);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        6 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(1.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(3.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(6.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(4.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[4]),
                        static_cast<double>(9.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[5]),
                        static_cast<double>(15.0));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(3));
        EXPECT_EQ(host[2], static_cast<value_t>(6));
        EXPECT_EQ(host[3], static_cast<value_t>(4));
        EXPECT_EQ(host[4], static_cast<value_t>(9));
        EXPECT_EQ(host[5], static_cast<value_t>(15));
    }
}

/**
 * @test TypedCumsum.cumsum_flatten_3D
 * @brief Tests cumsum on a 3D tensor flattened along the last axis.
 */
TYPED_TEST(TypedCumsum, cumsum_flatten_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(7), static_cast<value_t>(8)
    };
    t = vals;

    Tensor<value_t> res = math::cumsum<value_t>(t);

    std::vector<value_t> host(8);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        8 * sizeof(value_t)).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(1),
        static_cast<value_t>(1) + static_cast<value_t>(2),
        static_cast<value_t>(1) + static_cast<value_t>(2) +
            static_cast<value_t>(3),
        static_cast<value_t>(1) + static_cast<value_t>(2) +
            static_cast<value_t>(3) + static_cast<value_t>(4),
        static_cast<value_t>(1) + static_cast<value_t>(2) +
            static_cast<value_t>(3) + static_cast<value_t>(4) +
            static_cast<value_t>(5),
        static_cast<value_t>(1) + static_cast<value_t>(2) +
            static_cast<value_t>(3) + static_cast<value_t>(4) +
            static_cast<value_t>(5) + static_cast<value_t>(6),
        static_cast<value_t>(1) + static_cast<value_t>(2) +
            static_cast<value_t>(3) + static_cast<value_t>(4) +
            static_cast<value_t>(5) + static_cast<value_t>(6) +
            static_cast<value_t>(7),
        static_cast<value_t>(1) + static_cast<value_t>(2) +
            static_cast<value_t>(3) + static_cast<value_t>(4) +
            static_cast<value_t>(5) + static_cast<value_t>(6) +
            static_cast<value_t>(7) + static_cast<value_t>(8)
    };

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
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
 * @test TypedCumsum.cumsum_view_flatten
 * @brief Tests cumsum on a view of a 3D tensor flattened along the last
 * axis.
 */
TYPED_TEST(TypedCumsum, cumsum_view_flatten)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 4, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < vals.size(); ++i)
        vals[i] = static_cast<value_t>(i + 1);
    t = vals;

    std::vector<uint64_t> start = {1ull, 1ull, 0ull};
    std::vector<uint64_t> view_shape = {2ull, 2ull};
    Tensor<value_t> view(t, start, view_shape);

    Tensor<value_t> res = math::cumsum<value_t>(view);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        4 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(11);
    auto e1 = static_cast<value_t>(23);
    auto e2 = static_cast<value_t>(36);
    auto e3 = static_cast<value_t>(50);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(e3));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
        EXPECT_EQ(host[3], e3);
    }
}

/**
 * @test TypedCumsum.cumsum_alias_view_strided
 * @brief Tests cumsum on an alias view with a stride on a 1D tensor.
 */
TYPED_TEST(TypedCumsum, cumsum_alias_view_strided)
{
    using value_t = TypeParam;
    Tensor<value_t> t({6}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    std::vector<uint64_t> start = {0ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<value_t> alias_view(t, start, dims, strides);

    Tensor<value_t> res = math::cumsum<value_t>(alias_view);

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        3 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(1);
    auto e1 = static_cast<value_t>(1) + static_cast<value_t>(3);
    auto e2 = static_cast<value_t>(1) + static_cast<value_t>(3) +
              static_cast<value_t>(5);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
    }
}

/**
 * @test TypedCumsum.cumsum_alias_view_overlapping_stride_zero
 * @brief Tests cumsum on an alias view with overlapping stride of zero
 * on a 2D tensor.
 */
TYPED_TEST(TypedCumsum, cumsum_alias_view_overlapping_stride_zero)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    std::vector<uint64_t> start   = {1ull, 0ull};
    std::vector<uint64_t> dims    = {2ull, 2ull};
    std::vector<uint64_t> strides = {0ull, 1ull};
    Tensor<value_t> alias_view(t, start, dims, strides);

    Tensor<value_t> res = math::cumsum<value_t>(alias_view, 0);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        4 * sizeof(value_t)).wait();

    auto e0 = static_cast<value_t>(4);
    auto e1 = static_cast<value_t>(5);
    auto e2 = static_cast<value_t>(8);
    auto e3 = static_cast<value_t>(10);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(e0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(e1));
        EXPECT_FLOAT_EQ(static_cast<double>(host[2]),
                        static_cast<double>(e2));
        EXPECT_FLOAT_EQ(static_cast<double>(host[3]),
                        static_cast<double>(e3));
    }
    else
    {
        EXPECT_EQ(host[0], e0);
        EXPECT_EQ(host[1], e1);
        EXPECT_EQ(host[2], e2);
        EXPECT_EQ(host[3], e3);
    }
}

/**
 * @test TypedCumsum.cumsum_alias_view_weird_strides
 * @brief Sorting an alias view with non-trivial strides (e.g. 13,4).
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Cumsum along axis 1.
 */
TYPED_TEST(TypedCumsum, cumsum_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5, 20}, MemoryLocation::HOST);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
        vals[i] = static_cast<value_t>(i);
    owner = vals;

    Tensor<value_t> view(owner, {0, 0}, {3, 4}, {13, 4});

    Tensor<value_t> view2 = math::cumsum<value_t>(view, 1);
    EXPECT_EQ(view2.m_dimensions, (std::vector<uint64_t>{3, 4}));
    EXPECT_EQ(view2.m_strides, (std::vector<uint64_t>{4, 1}));

    Tensor<value_t> host = view2.clone();

    std::vector<value_t> out(12);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                        sizeof(value_t) * 12).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(0),  static_cast<value_t>(4),
        static_cast<value_t>(12), static_cast<value_t>(24),
        static_cast<value_t>(13), static_cast<value_t>(30),
        static_cast<value_t>(51), static_cast<value_t>(76),
        static_cast<value_t>(26), static_cast<value_t>(56),
        static_cast<value_t>(90), static_cast<value_t>(128)
    };

    for (uint64_t k = 0; k < out.size(); ++k)
    {
        if constexpr (std::is_floating_point<value_t>::value)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(out[k]),
                            static_cast<double>(expected[k]));
        }
        else
        {
            EXPECT_EQ(out[k], expected[k]);
        }
    }
}

/**
 * @test TypedCumsum.cumsum_axis_out_of_bounds
 * @brief Tests that cumsum throws std::invalid_argument when the axis
 * is out of bounds.
 */
TYPED_TEST(TypedCumsum, cumsum_axis_out_of_bounds)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    t = vals;

    EXPECT_THROW(math::cumsum<value_t>(t, 2), std::invalid_argument);
    EXPECT_THROW(math::cumsum<value_t>(t, -3), std::invalid_argument);
}

/**
 * @test TypedCumsum.cumsum_nan_throws
 * @brief Tests that cumsum throws std::runtime_error when the tensor
 * contains NaN values.
 */
TYPED_TEST(TypedCumsum, cumsum_nan_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
        return;

    Tensor<value_t> t({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(3)
    };
    t = vals;

    EXPECT_THROW(math::cumsum<value_t>(t, -1), std::runtime_error);
}

/**
 * @test TypedCumsum.cumsum_non_finite_throws
 * @brief Tests that cumsum throws std::runtime_error when the tensor
 * contains non-finite values (infinity).
 */
TYPED_TEST(TypedCumsum, cumsum_non_finite_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
        return;

    Tensor<value_t> t({2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        std::numeric_limits<value_t>::infinity(),
        static_cast<value_t>(1)
    };
    t = vals;

    EXPECT_THROW(math::cumsum<value_t>(t, -1), std::runtime_error);
}

/**
 * @test TypedCumsum.cumsum_empty
 * @brief Tests cumsum on an empty tensor returns a tensor with a single
 * zero element.
 */
TYPED_TEST(TypedCumsum, cumsum_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;

    Tensor<value_t> res({1}, MemoryLocation::DEVICE);
    res = math::cumsum<value_t>(t, -1);

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(value_t)).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(static_cast<value_t>(0)));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(0));
    }
}

template<typename T>
class TypedTranspose : public ::testing::Test {};

using TransposeTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedTranspose, TransposeTestTypes);

/**
 * @test TypedTranspose.transpose_noargs_reverse_axes
 * @brief Tests that transpose() with no arguments reverses all axes.
 */
TYPED_TEST(TypedTranspose, transpose_noargs_reverse_axes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
        vals[i] = static_cast<value_t>(i);
    t = vals;

    Tensor<value_t> t_rev = math::transpose<value_t>(t);

    EXPECT_EQ(t_rev.m_dimensions,
              (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(t_rev.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1],
                               t.m_strides[0]}));

    Tensor<value_t> host = t_rev.clone();
    std::vector<value_t> out(24);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                        sizeof(value_t) * 24).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(out[0]),
                        static_cast<double>(vals[0]));
        EXPECT_FLOAT_EQ(static_cast<double>(out[23]),
                        static_cast<double>(vals[23]));
    }
    else
    {
        EXPECT_EQ(out[0], vals[0]);
        EXPECT_EQ(out[23], vals[23]);
    }
}

/**
 * @test TypedTranspose.transpose_explicit_axes
 * @brief Tests transpose with explicit axis permutation.
 */
TYPED_TEST(TypedTranspose, transpose_explicit_axes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < 24; ++i) vals[i] = static_cast<value_t>(i);
    t = vals;

    Tensor<value_t> perm = math::transpose<value_t>(t, {2, 1, 0});

    EXPECT_EQ(perm.m_dimensions,
              (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(perm.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1],
                               t.m_strides[0]}));

    Tensor<value_t> host = perm.clone();
    std::vector<value_t> out(24);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                        sizeof(value_t) * 24).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(out[0]),
                        static_cast<double>(vals[0]));
        EXPECT_FLOAT_EQ(static_cast<double>(out[23]),
                        static_cast<double>(vals[23]));
    }
    else
    {
        EXPECT_EQ(out[0], vals[0]);
        EXPECT_EQ(out[23], vals[23]);
    }
}

/**
 * @test TypedTranspose.transpose_explicit_axes_negative
 * @brief Tests transpose with explicit negative axis permutation.
 */
TYPED_TEST(TypedTranspose, transpose_explicit_axes_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < 24; ++i) vals[i] = static_cast<value_t>(i);
    t = vals;

    Tensor<value_t> perm = math::transpose<value_t>(t, {-1, 1, -3});

    EXPECT_EQ(perm.m_dimensions,
              (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(perm.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1],
                               t.m_strides[0]}));

    Tensor<value_t> host = perm.clone();
    std::vector<value_t> out(24);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                        sizeof(value_t) * 24).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(out[0]),
                        static_cast<double>(vals[0]));
        EXPECT_FLOAT_EQ(static_cast<double>(out[23]),
                        static_cast<double>(vals[23]));
    }
    else
    {
        EXPECT_EQ(out[0], vals[0]);
        EXPECT_EQ(out[23], vals[23]);
    }
}

/**
 * @test TypedTranspose.transpose_2d
 * @brief Tests transpose on a 2D tensor (matrix).
 */
TYPED_TEST(TypedTranspose, transpose_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = init;

    Tensor<value_t> t_T = math::transpose<value_t>(t);

    EXPECT_EQ(t_T.m_dimensions, (std::vector<uint64_t>{3, 2}));
    EXPECT_EQ(t_T.m_strides,
        (std::vector<uint64_t>{t.m_strides[1], t.m_strides[0]}));

    Tensor<value_t> host = t_T.clone();
    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                        sizeof(value_t) * 6).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(2), static_cast<value_t>(5),
        static_cast<value_t>(3), static_cast<value_t>(6)
    };

    if constexpr (std::is_floating_point<value_t>::value)
    {
        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(out[i]),
                            static_cast<double>(expected[i]));
        }
    }
    else
    {
        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_EQ(out[i], expected[i]);
        }
    }
}

/**
 * @test TypedTranspose.transpose_mutation_reflects
 * @brief Ensure that modifying the transposed alias updates original.
 */
TYPED_TEST(TypedTranspose, transpose_mutation_reflects)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);

    std::vector<value_t> init = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    t = init;

    Tensor<value_t> t_T = math::transpose<value_t>(t);

    t_T[0][0] = static_cast<value_t>(100);
    t_T[2][1] = static_cast<value_t>(200);

    Tensor<value_t> host = t.clone();
    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                        sizeof(value_t) * 6).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(out[0]),
                        static_cast<double>(100.0));
        EXPECT_FLOAT_EQ(static_cast<double>(out[5]),
                        static_cast<double>(200.0));
    }
    else
    {
        EXPECT_EQ(out[0], static_cast<value_t>(100));
        EXPECT_EQ(out[5], static_cast<value_t>(200));
    }
}

/**
 * @test TypedTranspose.transpose_invalid_axes
 * @brief Transpose throws when axes permutation is invalid.
 */
TYPED_TEST(TypedTranspose, transpose_invalid_axes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    t = std::vector<value_t>(24, static_cast<value_t>(1));

    EXPECT_THROW(math::transpose<value_t>(t, {0, 1}),
                 std::invalid_argument);
    EXPECT_THROW(math::transpose<value_t>(t, {0, 1, 1}),
                 std::invalid_argument);
    EXPECT_THROW(math::transpose<value_t>(t, {0, 1, 3}),
                 std::invalid_argument);
}

/**
 * @test TypedTranspose.transpose_1d
 * @brief Transpose a 1D tensor should return a 1D alias (no change).
 */
TYPED_TEST(TypedTranspose, transpose_1d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::HOST);

    std::vector<value_t> init = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4)
    };
    t = init;

    Tensor<value_t> t_tr = math::transpose<value_t>(t);
    EXPECT_EQ(t_tr.m_dimensions, t.m_dimensions);
    EXPECT_EQ(t_tr.m_strides, t.m_strides);

    Tensor<value_t> host = t_tr.clone();
    std::vector<value_t> out(5);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                        sizeof(value_t) * 5).wait();

    for (uint64_t i = 0; i < 5; ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(out[i]),
                            static_cast<double>(i));
        }
        else
        {
            EXPECT_EQ(out[i], static_cast<value_t>(i));
        }
    }
}

/**
 * @test TypedTranspose.transpose_empty
 * @brief Transpose of an empty tensor throws.
 */
TYPED_TEST(TypedTranspose, transpose_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    EXPECT_THROW(math::transpose<value_t>(t), std::runtime_error);
}

template<typename T>
class TypedPad : public ::testing::Test {};

using PadTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedPad, PadTestTypes);

/**
 * @test TypedPad.pad_correct_result_shape
 * @brief Verify output dimensions equal input dims plus paddings.
 */
TYPED_TEST(TypedPad, pad_correct_result_shape)
{
    using value_t = TypeParam;
    Tensor<value_t> t({10, 10});

    value_t fill = static_cast<value_t>(0);
    Tensor<value_t> result =
        math::pad<value_t>(t, 1, 2, 3, 4, fill);

    const std::vector<uint64_t> res_shape = result.get_dimensions();
    EXPECT_EQ(res_shape[0], 13);
    EXPECT_EQ(res_shape[1], 17);
}

/**
 * @test TypedPad.pad_values_are_correct
 * @brief Pad a 2x2 tensor with zeros on all sides and check positions.
 */
TYPED_TEST(TypedPad, pad_values_are_correct)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2});

    std::vector<value_t> host_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    t = host_vals;

    value_t fill = static_cast<value_t>(0);
    Tensor<value_t> out = math::pad<value_t>(t, 1, 1, 1, 1, fill);

    EXPECT_EQ(out.get_dimensions()[0], 4u);
    EXPECT_EQ(out.get_dimensions()[1], 4u);

    std::vector<value_t> expected = {
        static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(0)
    };

    for (uint64_t r = 0; r < 4; ++r)
    {
        for (uint64_t c = 0; c < 4; ++c)
        {
            if constexpr (std::is_floating_point<value_t>::value)
            {
                EXPECT_FLOAT_EQ(out[r][c],
                                static_cast<value_t>(expected[r*4 + c]));
            }
            else
            {
                EXPECT_EQ(out[r][c], expected[r*4 + c]);
            }
        }
    }
}

/**
 * @test TypedPad.pad_asymmetric_positions
 * @brief Verify asymmetric paddings place original elements correctly.
 */
TYPED_TEST(TypedPad, pad_asymmetric_positions)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2});

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    t = init;

    const uint64_t top    = 3;
    const uint64_t bottom = 4;
    const uint64_t left   = 1;
    const uint64_t right  = 2;
    value_t fill = static_cast<value_t>(100);

    Tensor<value_t> out =
        math::pad<value_t>(t, top, bottom, left, right, fill);

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
                const value_t expected = t[or_r][or_c];
                if constexpr (std::is_floating_point<value_t>::value)
                    EXPECT_FLOAT_EQ(out[r][c], expected);
                else
                    EXPECT_EQ(out[r][c], expected);
            }
            else
            {
                if constexpr (std::is_floating_point<value_t>::value)
                    EXPECT_FLOAT_EQ(out[r][c], fill);
                else
                    EXPECT_EQ(out[r][c], fill);
            }
        }
    }
}

/**
 * @test TypedPad.pad_nonzero_fill_value
 * @brief Ensure non-zero fill value is used and original element placed.
 */
TYPED_TEST(TypedPad, pad_nonzero_fill_value)
{
    using value_t = TypeParam;
    Tensor<value_t> t({1, 1});
    t = std::vector<value_t>{ static_cast<value_t>(9) };

    const uint64_t top    = 1;
    const uint64_t bottom = 0;
    const uint64_t left   = 2;
    const uint64_t right  = 1;
    value_t fill;
    if constexpr (std::is_floating_point<value_t>::value)
        fill = static_cast<value_t>(-7.5);
    else
        fill = static_cast<value_t>(75);

    Tensor<value_t> out =
        math::pad<value_t>(t, top, bottom, left, right, fill);

    EXPECT_EQ(out.get_dimensions()[0], top + 1 + bottom);
    EXPECT_EQ(out.get_dimensions()[1], left + 1 + right);

    if constexpr (std::is_floating_point<value_t>::value)
        EXPECT_FLOAT_EQ(out[top][left], static_cast<value_t>(9));
    else
        EXPECT_EQ(out[top][left], static_cast<value_t>(9));

    for (uint64_t r = 0; r < out.get_dimensions()[0]; ++r)
    {
        for (uint64_t c = 0; c < out.get_dimensions()[1]; ++c)
        {
            if (r == top && c == left) continue;
            if constexpr (std::is_floating_point<value_t>::value)
            {
                EXPECT_FLOAT_EQ(out[r][c], fill);
            }
            else
            {
                EXPECT_EQ(out[r][c], fill);
            }
        }
    }
}

/**
 * @test TypedPad.pad_on_view_keeps_positions_and_owner
 * @brief Padding a view copies correct elements and leaves owner unchanged.
 */
TYPED_TEST(TypedPad, pad_on_view_keeps_positions_and_owner)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({4, 4});
    std::vector<value_t> vals;
    vals.reserve(16);
    for (int i = 0; i < 16; ++i) vals.push_back(static_cast<value_t>(i+1));
    owner = vals;

    std::vector<uint64_t> start = {1, 1};
    std::vector<uint64_t> shape = {2, 2};
    Tensor<value_t> view(owner, start, shape);

    const uint64_t top = 1, bottom = 0, left = 2, right = 1;
    value_t fill;
    if constexpr (std::is_floating_point<value_t>::value)
    {
        fill = static_cast<value_t>(-1.0);
    }
    else
    {
        fill = static_cast<value_t>(100);
    }

    Tensor<value_t> out =
        math::pad<value_t>(view, top, bottom, left, right, fill);

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
                const value_t expected = owner[owner_r][owner_c];
                if constexpr (std::is_floating_point<value_t>::value)
                {
                    EXPECT_FLOAT_EQ(out[r][c], expected);
                }
                else
                {
                    EXPECT_EQ(out[r][c], expected);
                }
            }
            else
            {
                if constexpr (std::is_floating_point<value_t>::value)
                {
                    EXPECT_FLOAT_EQ(out[r][c], fill);
                }
                else
                {
                    EXPECT_EQ(out[r][c], fill);
                }
            }
        }
    }

    // verify owner unchanged (values chosen fit both types)
    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(owner[1][1], static_cast<value_t>(6.0));
        EXPECT_FLOAT_EQ(owner[1][2], static_cast<value_t>(7.0));
        EXPECT_FLOAT_EQ(owner[2][1], static_cast<value_t>(10.0));
        EXPECT_FLOAT_EQ(owner[2][2], static_cast<value_t>(11.0));
    }
    else
    {
        EXPECT_EQ(owner[1][1], static_cast<value_t>(6));
        EXPECT_EQ(owner[1][2], static_cast<value_t>(7));
        EXPECT_EQ(owner[2][1], static_cast<value_t>(10));
        EXPECT_EQ(owner[2][2], static_cast<value_t>(11));
    }
}

/**
 * @test TypedPad.pad_on_alias_view_respects_strides
 * @brief Padding an alias view with custom strides should respect strides.
 */
TYPED_TEST(TypedPad, pad_on_alias_view_respects_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({3, 4});
    std::vector<value_t> vals;
    vals.reserve(12);
    for (int i = 0; i < 12; ++i) vals.push_back(static_cast<value_t>(i+1));
    owner = vals;

    std::vector<uint64_t> start = {0, 0};
    std::vector<uint64_t> dims = {2, 2};
    std::vector<uint64_t> strides = {4, 2};
    Tensor<value_t> aview(owner, start, dims, strides);

    const uint64_t top = 0, bottom = 1, left = 1, right = 0;
    value_t fill = static_cast<value_t>(1);

    Tensor<value_t> out =
        math::pad<value_t>(aview, top, bottom, left, right, fill);

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
                const value_t expected = aview[or_r][or_c];
                if constexpr (std::is_floating_point<value_t>::value)
                    EXPECT_FLOAT_EQ(out[r][c], expected);
                else
                    EXPECT_EQ(out[r][c], expected);
            }
            else
            {
                if constexpr (std::is_floating_point<value_t>::value)
                    EXPECT_FLOAT_EQ(out[r][c], fill);
                else
                    EXPECT_EQ(out[r][c], fill);
            }
        }
    }

    // verify alias and owner values (fits both types)
    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(aview[0][0], static_cast<value_t>(1.0));
        EXPECT_FLOAT_EQ(aview[0][1], static_cast<value_t>(3.0));
        EXPECT_FLOAT_EQ(aview[1][0], static_cast<value_t>(5.0));
        EXPECT_FLOAT_EQ(aview[1][1], static_cast<value_t>(7.0));
        EXPECT_FLOAT_EQ(owner[0][0], static_cast<value_t>(1.0));
        EXPECT_FLOAT_EQ(owner[0][2], static_cast<value_t>(3.0));
        EXPECT_FLOAT_EQ(owner[1][0], static_cast<value_t>(5.0));
        EXPECT_FLOAT_EQ(owner[1][2], static_cast<value_t>(7.0));
    }
    else
    {
        EXPECT_EQ(aview[0][0], static_cast<value_t>(1));
        EXPECT_EQ(aview[0][1], static_cast<value_t>(3));
        EXPECT_EQ(aview[1][0], static_cast<value_t>(5));
        EXPECT_EQ(aview[1][1], static_cast<value_t>(7));
        EXPECT_EQ(owner[0][0], static_cast<value_t>(1));
        EXPECT_EQ(owner[0][2], static_cast<value_t>(3));
        EXPECT_EQ(owner[1][0], static_cast<value_t>(5));
        EXPECT_EQ(owner[1][2], static_cast<value_t>(7));
    }
}

/**
 * @test TypedPad.pad_4d_tensor_preserves_batches
 * @brief Pad a 4D tensor (batch0, batch1, height, width).
 */
TYPED_TEST(TypedPad, pad_4d_tensor_preserves_batches)
{
    using value_t = TypeParam;
    const uint64_t B0 = 2, B1 = 3, H = 2, W = 2;
    Tensor<value_t> t({B0, B1, H, W});

    std::vector<value_t> vals;
    vals.reserve(B0 * B1 * H * W);
    for (uint64_t i = 0; i < B0 * B1 * H * W; ++i)
        vals.push_back(static_cast<value_t>(i + 1));
    t = vals;

    const uint64_t top = 1, bottom = 1, left = 2, right = 1;
    value_t fill;
    if constexpr (std::is_floating_point<value_t>::value)
        fill = static_cast<value_t>(0.5);
    else
        fill = static_cast<value_t>(1);

    Tensor<value_t> out =
        math::pad<value_t>(t, top, bottom, left, right, fill);

    const uint64_t out_H = top + H + bottom;
    const uint64_t out_W = left + W + right;
    const uint64_t out_elems = B0 * B1 * out_H * out_W;
    std::vector<value_t> expected(out_elems, fill);

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

    std::vector<value_t> host(out_elems);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    for (uint64_t i = 0; i < out_elems; ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        else
            EXPECT_EQ(host[i], expected[i]);
    }

    const std::vector<uint64_t> dims = out.get_dimensions();
    EXPECT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], B0);
    EXPECT_EQ(dims[1], B1);
    EXPECT_EQ(dims[2], out_H);
    EXPECT_EQ(dims[3], out_W);
}

/**
 * @test TypedPad.pad_empty
 * @brief Calling pad on an empty tensor must throw std::invalid_argument.
 */
TYPED_TEST(TypedPad, pad_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    value_t fill = static_cast<value_t>(0);
    EXPECT_THROW(math::pad<value_t>(t, 1, 2, 3, 4, fill),
        std::invalid_argument);
}

/**
 * @test TypedPad.pad_rank1
 * @brief Calling pad on a rank-1 tensor must throw std::invalid_argument.
 */
TYPED_TEST(TypedPad, pad_rank1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({10});
    value_t fill = static_cast<value_t>(0);
    EXPECT_THROW(math::pad<value_t>(t, 1, 2, 3, 4, fill),
                 std::invalid_argument);
}

/**
 * @test TypedPad.pad_padwidth_overflow
 * @brief Passing pad widths that cause uint64_t overflow must throw.
 */
TYPED_TEST(TypedPad, pad_padwidth_overflow)
{
    using value_t = TypeParam;
    Tensor<value_t> t({10, 10});
    const uint64_t big = std::numeric_limits<uint64_t>::max();
    value_t fill = static_cast<value_t>(0);

    EXPECT_THROW(math::pad<value_t>(t, 1, 2, big, 1, fill),
                 std::overflow_error);
}

/**
 * @test TypedPad.pad_padheight_overflow
 * @brief Passing pad heights that cause uint64_t overflow must throw.
 */
TYPED_TEST(TypedPad, pad_padheight_overflow)
{
    using value_t = TypeParam;
    Tensor<value_t> t({10, 10});
    const uint64_t big = std::numeric_limits<uint64_t>::max();
    value_t fill = static_cast<value_t>(0);

    EXPECT_THROW(math::pad<value_t>(t, big, 1, 3, 4, fill),
                 std::overflow_error);
}

/**
 * @test TypedPad.pad_symmetric
 * @brief Symmetric padding helper pads both sides and preserves ordering.
 */
TYPED_TEST(TypedPad, pad_symmetric)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2, 2});
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    owner = vals;

    const uint64_t pad_h = 1;
    const uint64_t pad_w = 2;
    value_t pad_val;
    if constexpr (std::is_floating_point<value_t>::value)
        pad_val = static_cast<value_t>(7.5);
    else
        pad_val = static_cast<value_t>(7);

    Tensor<value_t> out =
        math::pad<value_t>(owner, pad_h, pad_w, pad_val);

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
                const value_t expected = owner[or_r][or_c];
                if constexpr (std::is_floating_point<value_t>::value)
                {
                    EXPECT_FLOAT_EQ(out[r][c], expected);
                }
                else
                {
                    EXPECT_EQ(out[r][c], expected);
                }
            }
            else
            {
                if constexpr (std::is_floating_point<value_t>::value)
                {
                    EXPECT_FLOAT_EQ(out[r][c], pad_val);
                }
                else
                {
                    EXPECT_EQ(out[r][c], pad_val);
                }
            }
        }
    }

    // Owner must be unchanged
    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(owner[0][0], static_cast<value_t>(1.0));
        EXPECT_FLOAT_EQ(owner[0][1], static_cast<value_t>(2.0));
        EXPECT_FLOAT_EQ(owner[1][0], static_cast<value_t>(3.0));
        EXPECT_FLOAT_EQ(owner[1][1], static_cast<value_t>(4.0));
    }
    else
    {
        EXPECT_EQ(owner[0][0], static_cast<value_t>(1));
        EXPECT_EQ(owner[0][1], static_cast<value_t>(2));
        EXPECT_EQ(owner[1][0], static_cast<value_t>(3));
        EXPECT_EQ(owner[1][1], static_cast<value_t>(4));
    }
}

template<typename T>
class TypedArgmax : public ::testing::Test {};

using ArgmaxTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedArgmax, ArgmaxTestTypes);

/**
 * @test TypedArgmax.argmax_flattened
 * @brief argmax with axis = -1 (flattened) returns index of global max,
 * validated via at().
 */
TYPED_TEST(TypedArgmax, argmax_flattened)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(5),
        static_cast<value_t>(3), static_cast<value_t>(5),
        static_cast<value_t>(2)
    };
    t = vals;

    Tensor<uint64_t> res = math::argmax<value_t>(t);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(1));

    uint64_t idx = res.at(0);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(t.at(idx)),
                        static_cast<double>(static_cast<value_t>(5)));
    }
    else
    {
        EXPECT_EQ(t.at(idx), static_cast<value_t>(5));
    }
}

/**
 * @test TypedArgmax.argmax_axis0_2d
 * @brief argmax along axis 0 of a 2x3 matrix (per-column argmax),
 * verified via at().
 */
TYPED_TEST(TypedArgmax, argmax_axis0_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(5)
    };
    t = init;

    Tensor<uint64_t> res = math::argmax<value_t>(t, 0);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(3));
    for (uint64_t col = 0; col < 3; ++col)
    {
        uint64_t row = res.at(col);
        value_t v0 = t.at(col);
        value_t v1 = t.at(3 + col);
        value_t expected = (v0 > v1) ? v0 : v1;

        if constexpr (std::is_floating_point<value_t>::value)
        {
            EXPECT_FLOAT_EQ(static_cast<double>(t.at(row * 3 + col)),
                            static_cast<double>(expected));
        }
        else
        {
            EXPECT_EQ(t.at(row * 3 + col), expected);
        }
    }
}

/**
 * @test TypedArgmax.argmax_axis1_2d
 * @brief argmax along axis 1 of a 2x3 matrix (per-row argmax),
 * verified via at().
 */
TYPED_TEST(TypedArgmax, argmax_axis1_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(5)
    };
    t = init;

    Tensor<uint64_t> res = math::argmax<value_t>(t, 1);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(2));

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(t.at(0 * 3 + res.at(0))),
                        static_cast<double>(static_cast<value_t>(4)));
        EXPECT_FLOAT_EQ(static_cast<double>(t.at(1 * 3 + res.at(1))),
                        static_cast<double>(static_cast<value_t>(5)));
    }
    else
    {
        EXPECT_EQ(t.at(0 * 3 + res.at(0)),
                  static_cast<value_t>(4));
        EXPECT_EQ(t.at(1 * 3 + res.at(1)),
                  static_cast<value_t>(5));
    }
}

/**
 * @test TypedArgmax.argmax_axis_negative
 * @brief argmax along axis -1 of a 2x3 matrix (per-row argmax),
 * verified via at().
 */
TYPED_TEST(TypedArgmax, argmax_axis_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(5)
    };
    t = init;
    Tensor<uint64_t> res = math::argmax<value_t>(t, -1);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(2));

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(t.at(0 * 3 + res.at(0))),
                        static_cast<double>(static_cast<value_t>(4)));
        EXPECT_FLOAT_EQ(static_cast<double>(t.at(1 * 3 + res.at(1))),
                        static_cast<double>(static_cast<value_t>(5)));
    }
    else
    {
        EXPECT_EQ(t.at(0 * 3 + res.at(0)),
                  static_cast<value_t>(4));
        EXPECT_EQ(t.at(1 * 3 + res.at(1)),
                  static_cast<value_t>(5));
    }
}

/**
 * @test TypedArgmax.argmax_tie_prefers_first
 * @brief argmax should prefer the first occurrence on ties.
 */
TYPED_TEST(TypedArgmax, argmax_tie_prefers_first)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(2), static_cast<value_t>(2),
        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(5), static_cast<value_t>(5)
    };
    t = vals;

    Tensor<uint64_t> res = math::argmax<value_t>(t, 1);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(2));
    EXPECT_EQ(res.at(0), static_cast<uint64_t>(0));
    EXPECT_EQ(res.at(1), static_cast<uint64_t>(1));
}

/**
 * @test TypedArgmax.argmax_axis_out_of_range
 * @brief argmax should throw std::invalid_argument for invalid axes.
 */
TYPED_TEST(TypedArgmax, argmax_axis_out_of_range)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    t = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };

    EXPECT_THROW(math::argmax<value_t>(t, 2), std::invalid_argument);
    EXPECT_THROW(math::argmax<value_t>(t, -3), std::invalid_argument);
}

/**
 * @test TypedArgmax.argmax_nan_throws
 * @brief argmax should throw std::runtime_error when input contains NaN.
 */
TYPED_TEST(TypedArgmax, argmax_nan_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
    {
        return;
    }

    Tensor<value_t> t({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(0)
    };
    t = vals;

    EXPECT_THROW(math::argmax<value_t>(t), std::runtime_error);
}

/**
 * @test TypedArgmax.argmax_alias_view
 * @brief argmax on a 1D alias view (non-contiguous) returns index
 * relative to view, verified via at().
 */
TYPED_TEST(TypedArgmax, argmax_alias_view)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({6}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(3),
        static_cast<value_t>(2), static_cast<value_t>(6),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    owner = init;

    Tensor<value_t> v(owner, {1}, {3}, {2});

    Tensor<uint64_t> res = math::argmax<value_t>(v);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(1));

    uint64_t idx = res.at(0);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(v.at(idx)),
                        static_cast<double>(static_cast<value_t>(6)));
    }
    else
    {
        EXPECT_EQ(v.at(idx), static_cast<value_t>(6));
    }

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(owner.at(3)),
                        static_cast<double>(static_cast<value_t>(6)));
    }
    else
    {
        EXPECT_EQ(owner.at(3), static_cast<value_t>(6));
    }
}

/**
 * @test TypedArgmax.argmax_3d_view_flatten
 * @brief argmax on a 3D sub-tensor view (flattened) returns the correct
 * index.
 */
TYPED_TEST(TypedArgmax, argmax_3d_view_flatten)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 3}, MemoryLocation::DEVICE);

    t = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(5),
        static_cast<value_t>(2), static_cast<value_t>(0),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(6), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3)
    };

    Tensor<value_t> v(t, {1, 0, 0}, {1, 2, 3}, {6, 3, 1});
    Tensor<uint64_t> res = math::argmax<value_t>(v);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(1));

    uint64_t idx = res.at(0);

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(v.at(idx)),
                        static_cast<double>(static_cast<value_t>(6)));
    }
    else
    {
        EXPECT_EQ(v.at(idx), static_cast<value_t>(6));
    }
}

/**
 * @test TypedArgmax.argmax_3d_view_axis0_sorted
 * @brief argmax along axis 0 of a 3D view returns per-(i,j) layer indices.
 */
TYPED_TEST(TypedArgmax, argmax_3d_view_axis0_sorted)
{
    using value_t = TypeParam;

    Tensor<value_t> owner({3, 2, 2}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        // layer 0
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(2), static_cast<value_t>(0),
        // layer 1
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(5), static_cast<value_t>(1),
        // layer 2
        static_cast<value_t>(2), static_cast<value_t>(6),
        static_cast<value_t>(0), static_cast<value_t>(7)
    };
    owner = vals;

    Tensor<uint64_t> res = math::argmax<value_t>(owner, 0);

    // Should produce 2*2 = 4 outputs (one per (axis1, axis2) position).
    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(4));

    // Expected per-(axis1,axis2) results in row-major order over shape [2,2]:
    // flat order: (0,0), (0,1), (1,0), (1,1).
    std::array<uint64_t,4> expected = {1u, 2u, 1u, 2u};

    for (uint64_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(res.at(i), expected[i]);
    }


    for (uint64_t j = 0; j < 2; ++j)
    {
        for (uint64_t k = 0; k < 2; ++k)
        {
            uint64_t out_flat = j * 2 + k;
            uint64_t chosen_layer = res.at(out_flat);
            uint64_t flat_index_in_v = chosen_layer * 4 + j * 2 + k;
            value_t chosen_val = owner.at(flat_index_in_v);

            for (uint64_t layer = 0; layer < 3; ++layer)
            {
                uint64_t idx = layer * 4 + j * 2 + k;
                if (layer == chosen_layer) continue;
                if constexpr (std::is_floating_point<value_t>::value)
                {
                    EXPECT_GE(static_cast<double>(chosen_val),
                              static_cast<double>(owner.at(idx)));
                }
                else
                {
                    EXPECT_GE(chosen_val, owner.at(idx));
                }
            }
        }
    }
}

/**
 * @test TypedArgmax.argmax_on_alias_view_strided
 * @brief argmax on a 1D alias view with non-unit stride returns index
 * relative to the view (checks correct handling of strides).
 */
TYPED_TEST(TypedArgmax, argmax_on_alias_view_strided)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({6}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(3),
        static_cast<value_t>(2), static_cast<value_t>(6),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    owner = vals;

    std::vector<uint64_t> start = {1ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<value_t> v(owner, start, dims, strides);

    Tensor<uint64_t> res = math::argmax<value_t>(v);

    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(1));
    EXPECT_EQ(res.at(0), static_cast<uint64_t>(1));
}

template<typename T>
class TypedArgsort : public ::testing::Test {};

using ArgsortTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedArgsort, ArgsortTestTypes);

/**
 * @test TypedArgsort.argsort_flattened_ascending
 * @brief Verify flattened ascending argsort returns a valid perm.
 */
TYPED_TEST(TypedArgsort, argsort_flattened_ascending)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(5),
        static_cast<value_t>(3), static_cast<value_t>(5),
        static_cast<value_t>(2)
    };
    t = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(t, std::nullopt, false);
    const uint64_t N = res.get_num_elements();
    ASSERT_EQ(N, static_cast<uint64_t>(5));

    std::vector<uint64_t> host(N);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * N).wait();

    std::vector<bool> seen(N, false);
    for (uint64_t i = 0; i < N; ++i)
    {
        uint64_t idx = host[i];
        ASSERT_LT(idx, N);
        seen[idx] = true;
        value_t v = t.at(idx);
        if (i > 0)
        {
            value_t prev = t.at(host[i - 1]);
            if constexpr (std::is_floating_point<value_t>::value)
                EXPECT_LE(static_cast<double>(prev),
                          static_cast<double>(v));
            else
                EXPECT_LE(prev, v);
        }
    }
    for (uint64_t i = 0; i < N; ++i) EXPECT_TRUE(seen[i]);
}

/**
 * @test TypedArgsort.argsort_flattened_descending
 * @brief Verify flattened descending argsort orders values.
 */
TYPED_TEST(TypedArgsort, argsort_flattened_descending)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(5),
        static_cast<value_t>(3), static_cast<value_t>(5),
        static_cast<value_t>(2)
    };
    t = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(t, std::nullopt, true);
    const uint64_t N = res.get_num_elements();
    ASSERT_EQ(N, static_cast<uint64_t>(5));

    std::vector<uint64_t> host(N);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * N).wait();

    for (uint64_t i = 1; i < N; ++i)
    {
        value_t prev = t.at(host[i - 1]);
        value_t cur  = t.at(host[i]);
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_GE(static_cast<double>(prev), static_cast<double>(cur));
        else
            EXPECT_GE(prev, cur);
    }

    std::vector<bool> seen(N, false);
    for (uint64_t i = 0; i < N; ++i)
    {
        ASSERT_LT(host[i], N);
        seen[host[i]] = true;
    }
    for (uint64_t i = 0; i < N; ++i) EXPECT_TRUE(seen[i]);
}

/**
 * @test TypedArgsort.argsort_axis0_2d
 * @brief Argsort along axis 0 for a 2x3 matrix, ascending order.
 */
TYPED_TEST(TypedArgsort, argsort_axis0_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(5)
    };
    t = init;

    Tensor<uint64_t> res = math::argsort<value_t>(t, 0, false);

    const uint64_t axis_size = 2;
    const uint64_t slices = 3;
    ASSERT_EQ(res.get_num_elements(), axis_size * slices);

    std::vector<uint64_t> host(axis_size * slices);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * host.size()).wait();

    const auto strides = res.get_strides();

    for (uint64_t col = 0; col < slices; ++col)
    {
        std::vector<bool> seen(axis_size, false);
        for (uint64_t r = 0; r < axis_size; ++r)
        {
            // multi-index is [r, col] because axis==0 is the first dim
            uint64_t offset = r * strides[0] + col * strides[1];
            uint64_t idx = host[offset];
            ASSERT_LT(idx, axis_size);
            seen[idx] = true;

            value_t v = t.at(idx * 3 + col);
            if (r > 0)
            {
                uint64_t prev_offset = (r - 1) * strides[0] + col * strides[1];
                uint64_t prev_idx = host[prev_offset];
                value_t prev_v = t.at(prev_idx * 3 + col);
                if constexpr (std::is_floating_point<value_t>::value)
                    EXPECT_LE(static_cast<double>(prev_v),
                              static_cast<double>(v));
                else
                    EXPECT_LE(prev_v, v);
            }
        }
        for (uint64_t s = 0; s < axis_size; ++s) EXPECT_TRUE(seen[s]);
    }
}

/**
 * @test TypedArgsort.argsort_axis1_2d
 * @brief Argsort along axis 1 for a 2x3 matrix, ascending order.
 */
TYPED_TEST(TypedArgsort, argsort_axis1_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(0), static_cast<value_t>(5)
    };
    t = init;

    Tensor<uint64_t> res = math::argsort<value_t>(t, 1, false);

    const uint64_t axis_size = 3;
    const uint64_t slices = 2;
    ASSERT_EQ(res.get_num_elements(), axis_size * slices);

    std::vector<uint64_t> host(axis_size * slices);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * host.size()).wait();

    const auto strides = res.get_strides();

    for (uint64_t row = 0; row < slices; ++row)
    {
        std::vector<uint64_t> idxs(axis_size);
        for (uint64_t r = 0; r < axis_size; ++r)
        {
            uint64_t offset = row * strides[0] + r * strides[1];
            idxs[r] = host[offset];
            ASSERT_LT(idxs[r], axis_size);
        }

        value_t v0 = t.at(row * 3 + idxs[0]);
        value_t v1 = t.at(row * 3 + idxs[1]);
        value_t v2 = t.at(row * 3 + idxs[2]);

        if constexpr (std::is_floating_point<value_t>::value)
        {
            EXPECT_LE(static_cast<double>(v0), static_cast<double>(v1));
            EXPECT_LE(static_cast<double>(v1), static_cast<double>(v2));
        }
        else
        {
            EXPECT_LE(v0, v1);
            EXPECT_LE(v1, v2);
        }
        EXPECT_NE(idxs[0], idxs[1]);
        EXPECT_NE(idxs[1], idxs[2]);
        EXPECT_NE(idxs[0], idxs[2]);
    }
}

/**
 * @test TypedArgsort.argsort_axis_out_of_range
 * @brief Axis values outside valid range must throw an exception.
 */
TYPED_TEST(TypedArgsort, argsort_axis_out_of_range)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    t = std::vector<value_t>{ static_cast<value_t>(1),
                              static_cast<value_t>(2),
                              static_cast<value_t>(3),
                              static_cast<value_t>(4),
                              static_cast<value_t>(5),
                              static_cast<value_t>(6) };

    EXPECT_THROW(math::argsort<value_t>(t, 2, false), std::invalid_argument);
    EXPECT_THROW(math::argsort<value_t>(t, -3, false), std::invalid_argument);
}

/**
 * @test TypedArgsort.argsort_nan_behavior
 * @brief For floats, NaNs should be placed after finite values.
 */
TYPED_TEST(TypedArgsort, argsort_nan_behavior)
{
    using value_t = TypeParam;
    if constexpr (!std::is_floating_point<value_t>::value) return;

    Tensor<value_t> t({4}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(0.0),
        static_cast<value_t>(2.0)
    };
    t = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(t, std::nullopt, false);
    const uint64_t N = res.get_num_elements();
    std::vector<uint64_t> host(N);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * N).wait();

    EXPECT_EQ(host[N - 1], 1u);
}

/**
 * @test TypedArgsort.argsort_tie_stability
 * @brief Stable tie-breaking preserves first-occurrence order.
 */
TYPED_TEST(TypedArgsort, argsort_tie_stability)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(2), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(1),
        static_cast<value_t>(2)
    };
    t = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(t, std::nullopt, false);
    const uint64_t N = res.get_num_elements();
    ASSERT_EQ(N, static_cast<uint64_t>(5));

    std::vector<uint64_t> host(N);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * N).wait();

    std::vector<uint64_t> expected = {1u, 3u, 0u, 2u, 4u};
    for (uint64_t i = 0; i < N; ++i) EXPECT_EQ(host[i], expected[i]);
}

/**
 * @test TypedArgsort.argsort_alias_view_strided
 * @brief Argsort on a strided 1D alias view returns indices in view.
 */
TYPED_TEST(TypedArgsort, argsort_alias_view_strided)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({6}, MemoryLocation::DEVICE);
    owner = std::vector<value_t>{ static_cast<value_t>(1),
                                  static_cast<value_t>(3),
                                  static_cast<value_t>(2),
                                  static_cast<value_t>(6),
                                  static_cast<value_t>(4),
                                  static_cast<value_t>(5) };

    Tensor<value_t> v(owner, {1}, {3}, {2});

    Tensor<uint64_t> res = math::argsort<value_t>(v, std::nullopt, false);
    ASSERT_EQ(res.get_num_elements(), static_cast<uint64_t>(3));

    std::vector<uint64_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * 3).wait();

    EXPECT_EQ(host[0], 0u);
    EXPECT_EQ(host[1], 2u);
    EXPECT_EQ(host[2], 1u);
}

/**
 * @test TypedArgsort.argsort_3d_axis0
 * @brief Argsort along axis 0 for a 3x2x2 tensor, ascending order.
 */
TYPED_TEST(TypedArgsort, argsort_3d_axis0)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({3,2,2}, MemoryLocation::DEVICE);

    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(4),
        static_cast<value_t>(2), static_cast<value_t>(0),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(5), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(6),
        static_cast<value_t>(0), static_cast<value_t>(7)
    };
    owner = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(owner, 0, false);

    const uint64_t axis_size = 3;
    const uint64_t slice_count = 4;
    ASSERT_EQ(res.get_num_elements(), axis_size * slice_count);

    std::vector<uint64_t> host(axis_size * slice_count);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * host.size()).wait();

    // get strides for res (rank = 3)
    const auto strides = res.get_strides();

    for (uint64_t j = 0; j < 2; ++j)
    {
        for (uint64_t k = 0; k < 2; ++k)
        {
            std::vector<bool> seen(axis_size, false);
            for (uint64_t r = 0; r < axis_size; ++r)
            {
                uint64_t offset = r * strides[0]
                    + j * strides[1] + k * strides[2];
                uint64_t idx = host[offset];
                ASSERT_LT(idx, axis_size);
                seen[idx] = true;
                uint64_t flat = idx * 4 + j * 2 + k;
                value_t v = owner.at(flat);
                if (r > 0)
                {
                    uint64_t prev_offset = (r - 1) * strides[0]
                        + j * strides[1] + k * strides[2];
                    uint64_t prev_idx = host[prev_offset];
                    value_t prev_v = owner.at(prev_idx * 4 + j * 2 + k);
                    if constexpr (std::is_floating_point<value_t>::value)
                    {
                        EXPECT_LE(static_cast<double>(prev_v),
                                  static_cast<double>(v));
                    }
                    else
                    {
                        EXPECT_LE(prev_v, v);
                    }
                }
            }
            for (uint64_t s = 0; s < axis_size; ++s) EXPECT_TRUE(seen[s]);
        }
    }
}

/**
 * @test TypedArgsort.argsort_3d_axis1
 * @brief Argsort along axis 1 for a 2x3x2 tensor, ascending order.
 */
TYPED_TEST(TypedArgsort, argsort_3d_axis1)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,3,2}, MemoryLocation::DEVICE);

    std::vector<value_t> vals;
    vals.reserve(2*3*2);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            for (uint64_t k = 0; k < 2; ++k)
            {
                vals.push_back(static_cast<value_t>(i * 100 + j * 10 + k));
            }
        }
    }
    owner = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(owner, 1, false);

    const uint64_t axis_size = 3;
    const uint64_t slice_count = 4;
    ASSERT_EQ(res.get_num_elements(), axis_size * slice_count);

    std::vector<uint64_t> host(axis_size * slice_count);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * host.size()).wait();

    const auto strides = res.get_strides();

    uint64_t slice_idx = 0;
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t k = 0; k < 2; ++k)
        {
            for (uint64_t r = 0; r < axis_size; ++r)
            {
                uint64_t offset = i * strides[0] +
                    r * strides[1] + k * strides[2];
                ASSERT_EQ(host[offset], r);
            }
            ++slice_idx;
        }
    }

    ASSERT_EQ(slice_idx, slice_count);
}

/**
 * @test TypedArgsort.argsort_3d_axis_flattened
 * @brief Argsort over the entire tensor for a 2x3x2 tensor, descending order.
 * When axis == std::nullopt the tensor is flattened and
 * argsort operates over the whole 1-D view of the tensor.
 */
TYPED_TEST(TypedArgsort, argsort_3d_axis_flattened)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,3,2}, MemoryLocation::DEVICE);

    std::vector<value_t> vals;
    vals.reserve(2*3*2);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            for (uint64_t k = 0; k < 2; ++k)
            {
                vals.push_back(static_cast<value_t>(i * 100 + j * 10 + k));
            }
        }
    }
    owner = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(owner, std::nullopt, true);

    const uint64_t axis_size = owner.get_num_elements();
    const uint64_t slice_count = 1;
    ASSERT_EQ(res.get_num_elements(), axis_size * slice_count);

    std::vector<uint64_t> host(axis_size);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                        sizeof(uint64_t) * host.size()).wait();

    for (uint64_t r = 0; r < axis_size; ++r)
    {
        ASSERT_EQ(host[r], axis_size - 1 - r);
    }
}

template<typename T>
class TypedGather : public ::testing::Test {};

using GatherTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedGather, GatherTestTypes);

/**
 * @test TypedGather.gather_2d_last_axis
 * @brief Argsort each row (last axis) of a 2x3 tensor and gather using the
 * produced ordering. Verifies per-row ascending ordering of the last axis.
 */
TYPED_TEST(TypedGather, gather_2d_last_axis)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    t = {1, 3, 2, 15, 7, 6};

    Tensor<uint64_t> order = math::argsort(t, -1);

    Tensor<value_t> sorted = math::gather(t, order, -1);

    Tensor<value_t> expected({2, 3});
    expected =
    {
        1, 2, 3,
        6, 7, 15
    };

    EXPECT_EQ(sorted, expected);
}

/**
 * @test TypedGather.gather_3d_axis_flattened
 * @brief Compute argsort over the entire 3D tensor (flattened) and use the
 * resulting flattened indices to gather into a new tensor. Verifies that a
 * flattened index ordering is applied across all elements.
 */
TYPED_TEST(TypedGather, gather_3d_axis_flattened)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,3,2}, MemoryLocation::DEVICE);

    std::vector<value_t> vals;
    vals.reserve(2*3*2);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            for (uint64_t k = 0; k < 2; ++k)
            {
                vals.push_back(static_cast<value_t>(i * 100 + j * 10 + k));
            }
        }
    }
    owner = vals;

    Tensor<uint64_t> res = math::argsort<value_t>(owner, std::nullopt, true);

    Tensor<value_t> sorted = math::gather(owner, res, std::nullopt);

    Tensor<value_t> expected({2, 3, 2});
    expected =
    {
        121, 120,
        111, 110,
        101, 100,

        21, 20,
        11, 10,
        1, 0
    };

    EXPECT_EQ(sorted, expected);
}

/**
 * @test TypedGather.gather_2d_first_axis
 * @brief Argsort along axis 0 for a 2x3 tensor (per-column ordering) and
 * gather to produce rows sorted by column-wise ordering.
 */
TYPED_TEST(TypedGather, gather_2d_first_axis)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    t = {5, 4, 3, 1, 2, 0};

    Tensor<uint64_t> order = math::argsort(t, 0);

    Tensor<value_t> sorted = math::gather(t, order, 0);

    Tensor<value_t> expected({2, 3});
    expected =
    {
        1, 2, 0,
        5, 4, 3
    };

    EXPECT_EQ(sorted, expected);
}

/**
 * @test TypedGather.gather_negative_axis_middle
 * @brief Use a negative axis (-2) to select the middle axis of a 2x2x2 tensor,
 * argsort along that axis and gather accordingly. Verifies correct handling of
 * negative axis values and axis-relative gathering.
 */
TYPED_TEST(TypedGather, gather_negative_axis_middle)
{
    using value_t = TypeParam;

    Tensor<value_t> t({2,2,2}, MemoryLocation::DEVICE);
    t =
    {
        0,3,
        2,1,

        4,7,
        6,5
    };

    Tensor<uint64_t> order = math::argsort(t, -2, true);

    Tensor<value_t> sorted = math::gather(t, order, -2);

    Tensor<value_t> expected({2,2,2});
    expected =
    {
        2,3,
        0,1,

        6,7,
        4,5
    };
    EXPECT_EQ(sorted, expected);
}

/**
 * @test TypedGather.gather_strided_input_axis0
 * @brief Gather from a non-contiguous (strided) 1D view derived from an
 * owner tensor. Verifies gather correctly indexes into alias/view tensors.
 */
TYPED_TEST(TypedGather, gather_strided_input_axis0)
{
    using value_t = TypeParam;

    Tensor<value_t> owner({6}, MemoryLocation::DEVICE);
    owner = std::vector<value_t>
    {
        static_cast<value_t>(10),
        static_cast<value_t>(11),
        static_cast<value_t>(20),
        static_cast<value_t>(21),
        static_cast<value_t>(30),
        static_cast<value_t>(31)
    };

    Tensor<value_t> v(owner, std::vector<uint64_t>{0},
        std::vector<uint64_t>{3}, std::vector<uint64_t>{2});

    Tensor<uint64_t> idx({3}, MemoryLocation::DEVICE);
    idx = std::vector<uint64_t>{2, 0, 1};

    Tensor<value_t> res = math::gather(v, idx, 0);

    Tensor<value_t> expected({3}, MemoryLocation::DEVICE);
    expected = std::vector<value_t>
    {
        static_cast<value_t>(30),
        static_cast<value_t>(10),
        static_cast<value_t>(20)
    };

    EXPECT_EQ(res, expected);
}

/**
 * @test TypedGather.gather_flattened_with_strided_indexes
 * @brief Flattened gather (axis = nullopt) using an indexes view with stride>1.
 * The index view selects every-other entry from a backing array; verifies the
 * gather correctly reads logical index values from a non-contiguous index view.
 */
TYPED_TEST(TypedGather, gather_flattened_with_strided_indexes)
{
    using value_t = TypeParam;

    Tensor<value_t> owner({2,3,2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals;
    vals.reserve(12);
    for (uint64_t i = 0; i < 12; ++i)
    {
        vals.push_back(static_cast<value_t>(i));
    }
    owner = vals;

    const uint64_t tot = owner.get_num_elements();

    std::vector<uint64_t> idx_owner(24, 0);
    for (uint64_t i = 0; i < tot; ++i)
    {
        idx_owner[2 * i] = (tot - 1) - i;
    }

    Tensor<uint64_t> idx_owner_tensor({24}, MemoryLocation::DEVICE);
    idx_owner_tensor = idx_owner;

    Tensor<uint64_t> idx_view(idx_owner_tensor, std::vector<uint64_t>{0},
        std::vector<uint64_t>{tot}, std::vector<uint64_t>{2});

    Tensor<value_t> out = math::gather(owner, idx_view, std::nullopt);

    Tensor<value_t> expected({2,3,2}, MemoryLocation::DEVICE);
    std::vector<value_t> exp_vals(tot);
    for (uint64_t i = 0; i < tot; ++i)
    {
        exp_vals[i] = static_cast<value_t>( (tot - 1) - i );
    }
    expected = exp_vals;

    EXPECT_EQ(out, expected);
}

/**
 * @test TypedGather.gather_indexes_view_non_flattened
 * @brief Use a non-contiguous multi-dimensional index view (derived from a
 * larger backing array) to gather along the last axis of the data tensor.
 * Verifies correct logical indexing through a strided index view in the
 * non-flattened (axis-specified) path.
 */
TYPED_TEST(TypedGather, gather_indexes_view_non_flattened)
{
    using value_t = TypeParam;

    Tensor<value_t> owner({2,3,2}, MemoryLocation::DEVICE);
    std::vector<value_t> owner_vals;
    owner_vals.reserve(12);
    for (uint64_t i = 0; i < 12; ++i)
    {
        owner_vals.push_back(static_cast<value_t>(i));
    }
    owner = owner_vals;

    Tensor<uint64_t> idx_owner({2,3,4}, MemoryLocation::DEVICE);
    std::vector<uint64_t> idx_owner_vals(2 * 3 * 4, 999);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            idx_owner_vals[(i * 3 + j) * 4 + 0] = 1;
            idx_owner_vals[(i * 3 + j) * 4 + 2] = 0;
        }
    }
    idx_owner = idx_owner_vals;

    Tensor<uint64_t> idx_view(idx_owner,
        std::vector<uint64_t>{0,0,0},
        std::vector<uint64_t>{2,3,2},
        std::vector<uint64_t>{12,4,2});

    Tensor<value_t> out = math::gather(owner, idx_view, 2);

    std::vector<value_t> expected_host;
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            uint64_t base = i * (3 * 2) + j * 2;
            expected_host.push_back(owner_vals[base + 1]);
            expected_host.push_back(owner_vals[base + 0]);
        }
    }

    Tensor<value_t> expected({2, 3, 2});

    expected = expected_host;

    EXPECT_EQ(out, expected);
}

/**
 * @test TypedGather.gather_both_views
 * @brief Gather where both the source data and the index tensors are alias
 * views (non-contiguous). Ensures the gather logic correctly maps view
 * coordinates of both tensors when gathering along axis 0.
 */
TYPED_TEST(TypedGather, gather_both_views)
{
    using value_t = TypeParam;

    Tensor<value_t> data_owner({4,3,2}, MemoryLocation::DEVICE);
    std::vector<value_t> data_owner_vals(4 * 3 * 2);
    for (uint64_t i = 0; i < data_owner_vals.size(); ++i)
    {
        data_owner_vals[i] = static_cast<value_t>(i);
    }
    data_owner = data_owner_vals;

    Tensor<value_t> data_view(data_owner,
        std::vector<uint64_t>{0,0,0},
        std::vector<uint64_t>{2,3,2},
        std::vector<uint64_t>{12,2,1});

    Tensor<uint64_t> idx_owner({2,3,4}, MemoryLocation::DEVICE);
    std::vector<uint64_t> idx_owner_vals(2 * 3 * 4, 0);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            idx_owner_vals[(i * 3 + j) * 4 + 0] = 1;
            idx_owner_vals[(i * 3 + j) * 4 + 2] = 1;
        }
    }
    idx_owner = idx_owner_vals;

    Tensor<uint64_t> idx_view(idx_owner,
        std::vector<uint64_t>{0,0,0},
        std::vector<uint64_t>{2,3,2},
        std::vector<uint64_t>{12,4,2});

    Tensor<value_t> out = math::gather(data_view, idx_view, 0);

    std::vector<value_t> expected_host;
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            for (uint64_t k = 0; k < 2; ++k)
            {
                uint64_t chosen = 1;
                uint64_t owner_idx = chosen * 12 + j * 2 + k;
                expected_host.push_back(data_owner_vals[owner_idx]);
            }
        }
    }

    Tensor<value_t> expected({2, 3, 2});
    expected = expected_host;

    EXPECT_EQ(out, expected);
}

/**
 * @test TypedGather.axis_out_of_range_throws
 * @brief Passing an axis outside the valid range must
 * throw std::invalid_argument.
 */
TYPED_TEST(TypedGather, axis_out_of_range_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;
    Tensor<uint64_t> idx = math::argsort(t, 0);

    EXPECT_THROW(math::gather(t, idx, 5), std::invalid_argument);
}

/**
 * @test TypedGather.indexes_rank_higher_than_input_throws
 * @brief If indexes tensor has rank greater than the input tensor, gather must
 * throw std::invalid_argument.
 */
TYPED_TEST(TypedGather, indexes_rank_higher_than_input_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    t = vals;

    Tensor<uint64_t> idx({1,2,2}, MemoryLocation::DEVICE);
    std::vector<uint64_t> dummy(1*2*2, 0);
    idx = dummy;

    EXPECT_THROW(math::gather(t, idx, 0), std::invalid_argument);
}

/**
 * @test TypedGather.flattened_indexes_not_1d_throws
 * @brief For flattened gather (axis == nullopt) the indexes tensor must be
 * 1-D; otherwise std::invalid_argument must be thrown.
 */
TYPED_TEST(TypedGather, flattened_indexes_not_1d_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals =
    {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    t = vals;

    Tensor<uint64_t> idx({2,1}, MemoryLocation::DEVICE);
    idx = std::vector<uint64_t>{0,1};

    EXPECT_THROW(math::gather(t, idx, std::nullopt), std::invalid_argument);
}

/**
 * @test TypedGather.flattened_indexes_wrong_length_throws
 * @brief For flattened gather the 1-D indexes length must equal the total
 * number of input elements; otherwise std::invalid_argument must be thrown.
 */
TYPED_TEST(TypedGather, flattened_indexes_wrong_length_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals =
    {
        static_cast<value_t>(0), static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4), static_cast<value_t>(5)
    };
    t = vals;

    Tensor<uint64_t> idx({5}, MemoryLocation::DEVICE);
    idx = std::vector<uint64_t>{0,1,2,3,4};

    EXPECT_THROW(math::gather(t, idx, std::nullopt), std::invalid_argument);
}

/**
 * @test TypedGather.index_value_out_of_range_flattened_throws
 * @brief Flattened gather with any index >= total_input_elems must throw
 * std::out_of_range.
 */
TYPED_TEST(TypedGather, index_value_out_of_range_flattened_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals =
    {
        static_cast<value_t>(10), static_cast<value_t>(20),
        static_cast<value_t>(30), static_cast<value_t>(40)
    };
    t = vals;

    Tensor<uint64_t> idx({4}, MemoryLocation::DEVICE);
    idx = std::vector<uint64_t>{3,2,1,4};

    EXPECT_THROW(math::gather(t, idx, std::nullopt), std::out_of_range);
}

/**
 * @test TypedGather.index_value_out_of_range_axis_throws
 * @brief In axis-based gather, any index value that is >= size of the chosen
 * axis must cause std::out_of_range to be thrown.
 */
TYPED_TEST(TypedGather, index_value_out_of_range_axis_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals =
    {
        static_cast<value_t>(1), static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    Tensor<uint64_t> idx({2,3}, MemoryLocation::DEVICE);
    idx = std::vector<uint64_t>{0,1,2, 0,5,1};

    EXPECT_THROW(math::gather(t, idx, 1), std::out_of_range);
}

/**
 * @test TypedGather.incompatible_broadcast_throws
 * @brief If indexes cannot be broadcast to the input shape (for axis-based
 * gather), std::invalid_argument must be thrown.
 */
TYPED_TEST(TypedGather, incompatible_broadcast_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    Tensor<uint64_t> idx({2,2}, MemoryLocation::DEVICE);
    idx = std::vector<uint64_t>{0,1, 1,0};

    EXPECT_THROW(math::gather(t, idx, 0), std::invalid_argument);
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
        (start, stop, num, MemoryLocation::DEVICE, true, &step_out);

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

template<typename T>
class TypedArange : public ::testing::Test {};

using ArangeTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedArange, ArangeTestTypes);

/**
 * @test TypedArange.basic_positive_step
 * @brief arange with positive step generates the correct sequence.
 * Example: start=0, stop=5, step=1 -> [0,1,2,3,4]
 */
TYPED_TEST(TypedArange, basic_positive_step)
{
    using value_t = TypeParam;

    Tensor<value_t> out = math::arange<value_t>(
        static_cast<value_t>(0), static_cast<value_t>(5),
        static_cast<value_t>(1), MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 5u);

    std::vector<value_t> host(5);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        5 * sizeof(value_t)).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4)
    };

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        else
            EXPECT_EQ(host[i], expected[i]);
    }
}

/**
 * @test TypedArange.basic_negative_step
 * @brief arange with negative step generates correct decreasing seq.
 * Example: start=5, stop=0, step=-1 -> [5,4,3,2,1]
 */
TYPED_TEST(TypedArange, basic_negative_step)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
        return;

    Tensor<value_t> out = math::arange<value_t>(
        static_cast<value_t>(5.0), static_cast<value_t>(0.0),
        static_cast<value_t>(-1.0), MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 5u);

    std::vector<value_t> host(5);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        5 * sizeof(value_t)).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(5.0), static_cast<value_t>(4.0),
        static_cast<value_t>(3.0), static_cast<value_t>(2.0),
        static_cast<value_t>(1.0)
    };

    for (size_t i = 0; i < host.size(); ++i)
        EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                        static_cast<double>(expected[i]));
}

/**
 * @test TypedArange.zero_step_throws
 * @brief arange must throw when step is zero.
 */
TYPED_TEST(TypedArange, zero_step_throws)
{
    using value_t = TypeParam;

    EXPECT_THROW(
        {
            auto out = math::arange<value_t>(
                static_cast<value_t>(0), static_cast<value_t>(5),
                static_cast<value_t>(0), MemoryLocation::DEVICE);
        },
        std::invalid_argument);
}

/**
 * @test TypedArange.non_finite_inputs_throw
 * @brief arange must throw if start/stop/step is NaN or Inf.
 */
TYPED_TEST(TypedArange, non_finite_inputs_throw)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
        return;

    EXPECT_THROW(math::arange<value_t>(
                     std::numeric_limits<value_t>::quiet_NaN(),
                     static_cast<value_t>(1), static_cast<value_t>(1),
                     MemoryLocation::DEVICE),
                 std::runtime_error);

    EXPECT_THROW(math::arange<value_t>(
                     static_cast<value_t>(0),
                     std::numeric_limits<value_t>::infinity(),
                     static_cast<value_t>(1), MemoryLocation::DEVICE),
                 std::runtime_error);

    EXPECT_THROW(math::arange<value_t>(
                     static_cast<value_t>(0), static_cast<value_t>(1),
                     std::numeric_limits<value_t>::quiet_NaN(),
                     MemoryLocation::DEVICE),
                 std::runtime_error);
}

/**
 * @test TypedArange.stop_only
 * @brief arange(stop) variant generates 0..stop-1 sequence.
 */
TYPED_TEST(TypedArange, stop_only)
{
    using value_t = TypeParam;

    Tensor<value_t> out = math::arange<value_t>(
        static_cast<value_t>(4), MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 4u);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        4 * sizeof(value_t)).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3)
    };

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        else
            EXPECT_EQ(host[i], expected[i]);
    }
}

/**
 * @test TypedArange.empty_result
 * @brief arange returns empty tensor with single 0 when no elements.
 */
TYPED_TEST(TypedArange, empty_result)
{
    using value_t = TypeParam;

    Tensor<value_t> out1 = math::arange<value_t>(
        static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(1), MemoryLocation::DEVICE);
    ASSERT_EQ(out1.get_num_elements(), 1u);

    if constexpr (std::is_floating_point<value_t>::value)
        EXPECT_FLOAT_EQ(static_cast<double>(out1[0]),
                        static_cast<double>(static_cast<value_t>(0)));
    else
        EXPECT_EQ(out1[0], static_cast<value_t>(0));

    Tensor<value_t> out2 = math::arange<value_t>(
        static_cast<value_t>(5), static_cast<value_t>(0),
        static_cast<value_t>(1), MemoryLocation::DEVICE);
    ASSERT_EQ(out2.get_num_elements(), 1u);

    if constexpr (std::is_floating_point<value_t>::value)
        EXPECT_FLOAT_EQ(static_cast<double>(out2[0]),
                        static_cast<double>(static_cast<value_t>(0)));
    else
        EXPECT_EQ(out2[0], static_cast<value_t>(0));
}

/**
 * @test TypedArange.floating_point_step
 * @brief arange with non-integer step produces correct values.
 */
TYPED_TEST(TypedArange, floating_point_step)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
        return;

    Tensor<value_t> out = math::arange<value_t>(
        static_cast<value_t>(0.0), static_cast<value_t>(1.0),
        static_cast<value_t>(0.2), MemoryLocation::DEVICE);
    ASSERT_EQ(out.get_num_elements(), 5u);

    std::vector<value_t> host(5);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        5 * sizeof(value_t)).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(0.0), static_cast<value_t>(0.2),
        static_cast<value_t>(0.4), static_cast<value_t>(0.6),
        static_cast<value_t>(0.8)
    };

    for (size_t i = 0; i < host.size(); ++i)
        EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                        static_cast<double>(expected[i]));
}

template<typename T>
class TypedZeros : public ::testing::Test {};

using ZerosTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedZeros, ZerosTestTypes);

/**
 * @test TypedZeros.device
 * @brief zeros<T> returns a device tensor that is zero-initialized.
 */
TYPED_TEST(TypedZeros, device)
{
    using value_t = TypeParam;
    std::vector<uint64_t> shape = {2, 3};

    Tensor<value_t> z = math::zeros<value_t>(shape, MemoryLocation::DEVICE);

    ASSERT_NE(z.m_p_data.get(), nullptr);

    uint64_t total = z.get_num_elements();

    std::vector<value_t> host(total);
    g_sycl_queue.memcpy(host.data(), z.m_p_data.get(),
                        total * sizeof(value_t)).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(static_cast<value_t>(0)));
        else
            EXPECT_EQ(host[i], static_cast<value_t>(0));
    }
}

/**
 * @test TypedZeros.host
 * @brief zeros<T> returns a host tensor that is zero-initialized.
 */
TYPED_TEST(TypedZeros, host)
{
    using value_t = TypeParam;
    std::vector<uint64_t> shape = {5};

    Tensor<value_t> z = math::zeros<value_t>(shape, MemoryLocation::HOST);

    ASSERT_NE(z.m_p_data.get(), nullptr);

    uint64_t total = z.get_num_elements();

    std::vector<value_t> host(total);
    g_sycl_queue.memcpy(host.data(), z.m_p_data.get(),
                        total * sizeof(value_t)).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(static_cast<value_t>(0)));
        else
            EXPECT_EQ(host[i], static_cast<value_t>(0));
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

template<typename T>
class TypedFactorial : public ::testing::Test {};

using FactorialTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedFactorial, FactorialTestTypes);

/**
 * @test TypedFactorial.basic_values
 * @brief factorial on a simple 1D tensor (0..5) returns expected values.
 */
TYPED_TEST(TypedFactorial, basic_values)
{
    using value_t = TypeParam;

    Tensor<value_t> t({6}, MemoryLocation::DEVICE);
    t = std::vector<value_t>{
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };

    Tensor<value_t> out = math::factorial<value_t>(t);

    const uint64_t N = out.get_num_elements();
    ASSERT_EQ(N, 6u);

    std::vector<value_t> host(N);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
        sizeof(value_t) * N).wait();

    const std::vector<value_t> expected = {
        static_cast<value_t>(1), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(6),
        static_cast<value_t>(24), static_cast<value_t>(120)
    };

    for (uint64_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(host[i], expected[i]);
    }
}

/**
 * @test TypedFactorial.alias_view_strided
 * @brief factorial on a 1D alias view with non-unit stride uses view
 * indexing.
 */
TYPED_TEST(TypedFactorial, alias_view_strided)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({6}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(1), static_cast<value_t>(3),
        static_cast<value_t>(2), static_cast<value_t>(6),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    owner = init;

    std::vector<uint64_t> start = {1ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<value_t> v(owner, start, dims, strides);

    Tensor<value_t> out = math::factorial<value_t>(v);

    ASSERT_EQ(out.get_num_elements(), 3u);
    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * 3).wait();

    const std::array<uint64_t,3> expected_i = {6u, 720u, 120u};

    for (size_t i = 0; i < 3; ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected_i[i]));
        else
            EXPECT_EQ(host[i], static_cast<value_t>(expected_i[i]));
    }

    // owner must be unchanged
    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(owner.at(1)),
                        static_cast<double>(3.0));
        EXPECT_FLOAT_EQ(static_cast<double>(owner.at(3)),
                        static_cast<double>(6.0));
        EXPECT_FLOAT_EQ(static_cast<double>(owner.at(5)),
                        static_cast<double>(5.0));
    }
    else
    {
        EXPECT_EQ(owner.at(1), static_cast<value_t>(3));
        EXPECT_EQ(owner.at(3), static_cast<value_t>(6));
        EXPECT_EQ(owner.at(5), static_cast<value_t>(5));
    }
}

/**
 * @test TypedFactorial.nan_throws
 * @brief factorial should throw std::runtime_error if input contains NaN.
 */
TYPED_TEST(TypedFactorial, nan_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
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

    EXPECT_THROW(math::factorial<value_t>(t), std::runtime_error);
}

/**
 * @test TypedFactorial.inf_throws
 * @brief factorial should throw std::runtime_error if input contains Inf.
 */
TYPED_TEST(TypedFactorial, inf_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
    {
        return;
    }

    Tensor<value_t> t({2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        std::numeric_limits<value_t>::infinity(),
        static_cast<value_t>(1)
    };
    t = vals;

    EXPECT_THROW(math::factorial<value_t>(t), std::runtime_error);
}

/**
 * @test TypedFactorial.negative_and_noninteger_throws
 * @brief negative or non-integer inputs should throw std::invalid_arg.
 */
TYPED_TEST(TypedFactorial, negative_and_noninteger_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_signed_v<value_t>)
    {
        return;
    }

    // Negative input should throw for all signed types
    Tensor<value_t> neg({1}, MemoryLocation::DEVICE);
    neg = std::vector<value_t>{ static_cast<value_t>(-1) };
    EXPECT_THROW(math::factorial<value_t>(neg), std::invalid_argument);

    if constexpr (!std::is_floating_point_v<value_t>)
    {
        return;
    }

    Tensor<value_t> nonint({2}, MemoryLocation::DEVICE);
    nonint = std::vector<value_t>{
        static_cast<value_t>(2.5), static_cast<value_t>(3.14159)
    };
    EXPECT_THROW(math::factorial<value_t>(nonint), std::invalid_argument);
}

/**
 * @test TypedFactorial.overflow_throws
 * @brief factorial of sufficiently large n should overflow float and
 * provoke runtime_error.
 *
 * 35! > float max (≈3.4e38), so test only for floating types.
 */
TYPED_TEST(TypedFactorial, overflow_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
    {
        return;
    }

    Tensor<value_t> t({1}, MemoryLocation::DEVICE);
    t = std::vector<value_t>{ static_cast<value_t>(35.0) };

    EXPECT_THROW(math::factorial<value_t>(t), std::runtime_error);
}

/**
 * @test TypedFactorial.zero_and_one
 * @brief factorial(0) == 1 and factorial(1) == 1.
 */
TYPED_TEST(TypedFactorial, zero_and_one)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2}, MemoryLocation::DEVICE);

    std::vector<value_t> init = {
        static_cast<value_t>(0), static_cast<value_t>(1)
    };
    t = init;

    Tensor<value_t> out = math::factorial<value_t>(t);

    ASSERT_EQ(out.get_num_elements(), 2u);
    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), out.m_p_data.get(),
                        sizeof(value_t) * 2).wait();

    if constexpr (std::is_floating_point<value_t>::value)
    {
        EXPECT_FLOAT_EQ(static_cast<double>(host[0]),
                        static_cast<double>(1.0));
        EXPECT_FLOAT_EQ(static_cast<double>(host[1]),
                        static_cast<double>(1.0));
    }
    else
    {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(1));
    }
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

template<typename T>
class TypedPow : public ::testing::Test {};

using PowTestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedPow, PowTestTypes);

/**
 * @test TypedPow.basic_elementwise
 * @brief elementwise pow on two same-shaped tensors.
 */
TYPED_TEST(TypedPow, basic_elementwise)
{
    using value_t = TypeParam;

    Tensor<value_t> A({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    A = a_vals;

    Tensor<value_t> B({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(2), static_cast<value_t>(0),
        static_cast<value_t>(1), static_cast<value_t>(3)
    };
    B = b_vals;

    Tensor<value_t> R = math::pow<value_t>(A, B);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    std::vector<value_t> expected(4);
    for (size_t i = 0; i < expected.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
        {
            expected[i] = static_cast<value_t>(
                std::pow(static_cast<double>(a_vals[i]),
                         static_cast<double>(b_vals[i])));
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(expected[i]));
        }
        else
        {
            value_t base = a_vals[i];
            value_t exp  = b_vals[i];
            value_t r = static_cast<value_t>(1);
            for (value_t e = 0; e < exp; ++e) r *= base;
            expected[i] = r;
            EXPECT_EQ(host[i], expected[i]);
        }
    }
}

/**
 * @test TypedPow.scalar_broadcast
 * @brief broadcasting a scalar exponent across a tensor.
 */
TYPED_TEST(TypedPow, scalar_broadcast)
{
    using value_t = TypeParam;

    Tensor<value_t> A({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5), static_cast<value_t>(6)
    };
    A = a_vals;

    Tensor<value_t> B({1}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = { static_cast<value_t>(2) };
    B = b_vals;

    Tensor<value_t> R = math::pow<value_t>(A, B);

    std::vector<value_t> host(6);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_floating_point<value_t>::value)
        {
            value_t exp_v = static_cast<value_t>(
                std::pow(static_cast<double>(a_vals[i]),
                         static_cast<double>(b_vals[0])));
            EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                            static_cast<double>(exp_v));
        }
        else
        {
            value_t r = static_cast<value_t>(1);
            for (uint64_t e = 0; e < static_cast<uint64_t>(b_vals[0]); ++e)
                r *= a_vals[i];
            EXPECT_EQ(host[i], r);
        }
    }
}

/**
 * @test TypedPow.broadcast_dims
 * @brief broadcasting across compatible shapes (A:(2,1,3), B:(1,3)).
 */
TYPED_TEST(TypedPow, broadcast_dims)
{
    using value_t = TypeParam;

    const std::vector<uint64_t> a_shape = {2, 1, 3};
    const std::vector<uint64_t> b_shape = {1, 3};

    Tensor<value_t> A(a_shape, MemoryLocation::DEVICE);
    Tensor<value_t> B(b_shape, MemoryLocation::DEVICE);

    std::vector<value_t> a_vals(A.get_num_elements());
    for (uint64_t i = 0; i < a_vals.size(); ++i)
        a_vals[i] = static_cast<value_t>(i + 1);
    A = a_vals;

    std::vector<value_t> b_vals(B.get_num_elements());
    for (uint64_t i = 0; i < b_vals.size(); ++i)
        b_vals[i] = static_cast<value_t>(i + 2);
    B = b_vals;

    Tensor<value_t> R = math::pow<value_t>(A, B);

    const uint64_t out_elems = R.get_num_elements();
    std::vector<value_t> host(out_elems);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    for (uint64_t b0 = 0; b0 < 2; ++b0)
    {
        for (uint64_t k = 0; k < 3; ++k)
        {
            uint64_t out_idx = ((b0 * 1) + 0) * 3 + k;
            value_t av = a_vals[b0 * 3 + k];
            value_t bv = b_vals[k];
            if constexpr (std::is_floating_point<value_t>::value)
            {
                value_t exp_v = static_cast<value_t>(
                    std::pow(static_cast<double>(av),
                             static_cast<double>(bv)));
                EXPECT_FLOAT_EQ(static_cast<double>(host[out_idx]),
                                static_cast<double>(exp_v));
            }
            else
            {
                value_t r = static_cast<value_t>(1);
                for (uint64_t e = 0; e < static_cast<uint64_t>(bv); ++e) r *= av;
                EXPECT_EQ(host[out_idx], r);
            }
        }
    }
}

/**
 * @test TypedPow.nan_throws
 * @brief pow should throw std::runtime_error if inputs contain NaN
 * (floats only).
 */
TYPED_TEST(TypedPow, nan_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
    {
        // integral types cannot represent NaN; skip the test.
        return;
    }

    Tensor<value_t> A({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        static_cast<value_t>(1),
        std::numeric_limits<value_t>::quiet_NaN(),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    A = a_vals;

    Tensor<value_t> B({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    B = b_vals;

    EXPECT_THROW(math::pow<value_t>(A, B), std::runtime_error);
}

/**
 * @test TypedPow.inf_throws
 * @brief pow should throw std::runtime_error if result is non-finite
 * (floats only)
 */
TYPED_TEST(TypedPow, inf_throws)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
    {
        // integral types cannot represent Inf; skip the test.
        return;
    }

    Tensor<value_t> A({1, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        std::numeric_limits<value_t>::infinity(),
        static_cast<value_t>(2)
    };
    A = a_vals;

    Tensor<value_t> B({2, 1}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    B = b_vals;

    EXPECT_THROW(math::pow<value_t>(A, B), std::runtime_error);
}

/**
 * @test TypedPow.negative_exponent_float
 * @brief negative exponent only meaningful for floating-point types
 */
TYPED_TEST(TypedPow, negative_exponent_float)
{
    using value_t = TypeParam;

    if constexpr (!std::is_floating_point<value_t>::value)
    {
        // integral types don't represent negative exponents; skip.
        return;
    }

    Tensor<value_t> A({2}, MemoryLocation::DEVICE);
    std::vector<value_t> a_vals = {
        static_cast<value_t>(2.0), static_cast<value_t>(4.0)
    };
    A = a_vals;

    Tensor<value_t> B({2}, MemoryLocation::DEVICE);
    std::vector<value_t> b_vals = {
        static_cast<value_t>(-1.0), static_cast<value_t>(0.5)
    };
    B = b_vals;

    Tensor<value_t> R = math::pow<value_t>(A, B);

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    for (size_t i = 0; i < host.size(); ++i)
    {
        value_t expected = static_cast<value_t>(
            std::pow(static_cast<double>(a_vals[i]),
                     static_cast<double>(b_vals[i])));
        EXPECT_FLOAT_EQ(static_cast<double>(host[i]),
                        static_cast<double>(expected));
    }
}

/**
 * @test TypedPow.alias_views_noncontiguous_strides
 * @brief both inputs are alias views with non-contiguous, positive strides
 */
TYPED_TEST(TypedPow, alias_views_noncontiguous_strides)
{
    using value_t = TypeParam;

    Tensor<value_t> A_base({5, 6}, MemoryLocation::DEVICE);
    Tensor<value_t> B_base({5, 6}, MemoryLocation::DEVICE);

    std::vector<value_t> a_vals(A_base.get_num_elements());
    for (uint64_t i = 0; i < a_vals.size(); ++i)
        a_vals[i] = static_cast<value_t>(i + 1);
    A_base = a_vals;

    std::vector<value_t> b_vals(B_base.get_num_elements());
    for (uint64_t i = 0; i < b_vals.size(); ++i)
        b_vals[i] = static_cast<value_t>((i % 3) + 1);
    B_base = b_vals;

    std::vector<uint64_t> A_start   = {1ull, 1ull};
    std::vector<uint64_t> B_start   = {0ull, 2ull};
    std::vector<uint64_t> view_dims = {2ull, 3ull};
    std::vector<uint64_t> A_strides = {3ull, 2ull};
    std::vector<uint64_t> B_strides = {2ull, 3ull};

    Tensor<value_t> A_view(A_base, A_start, view_dims, A_strides);
    Tensor<value_t> B_view(B_base, B_start, view_dims, B_strides);

    Tensor<value_t> R = math::pow<value_t>(A_view, B_view);

    std::vector<value_t> host(R.get_num_elements());
    g_sycl_queue.memcpy(host.data(), R.m_p_data.get(),
                        host.size() * sizeof(value_t)).wait();

    const uint64_t base_cols = 6;
    for (uint64_t i = 0; i < view_dims[0]; ++i)
    {
        for (uint64_t j = 0; j < view_dims[1]; ++j)
        {
            uint64_t out_idx = i * view_dims[1] + j;

            uint64_t a_linear = (A_start[0] * base_cols + A_start[1]) +
                                i * A_strides[0] + j * A_strides[1];
            uint64_t b_linear = (B_start[0] * base_cols + B_start[1]) +
                                i * B_strides[0] + j * B_strides[1];

            value_t av = a_vals[a_linear];
            value_t bv = b_vals[b_linear];

            if constexpr (std::is_floating_point<value_t>::value)
            {
                value_t expected = static_cast<value_t>(
                    std::pow(static_cast<double>(av),
                             static_cast<double>(bv)));
                EXPECT_FLOAT_EQ(static_cast<double>(host[out_idx]),
                                static_cast<double>(expected));
            }
            else
            {
                value_t r = static_cast<value_t>(1);
                for (uint64_t e = 0; e < static_cast<uint64_t>(bv); ++e)
                    r *= av;
                EXPECT_EQ(host[out_idx], r);
            }
        }
    }
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
 * @brief eig() returns expected eigenvalues and eigenvectors
 * for a diagonal device matrix.
 *
 * Creates a 4×4 diagonal matrix allocated on device, runs eig(), reads back
 * eigenvalues and eigenvectors and checks they match the
 * diagonal entries (within tolerance).
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

    Tensor<float> expected({4});
    expected = {1.0, 2.0, 3.0, 4.0};
    const double tol = 5e-4;

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);

    }

    Tensor<float> expected_eigvecs({4, 4});
    expected_eigvecs =
    {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

    }
}

/**
 * @test EIG.eig_batched
 * @brief eig() on a small batched tensor yields per-batch eigenvalues
 * and eigenvectors.
 *
 * Builds a 2×3×3 device tensor containing two diagonal batches, runs eig()
 * with increased iterations, and checks each batch's computed eigenvalues and
 * eigenvectors against expected diagonals.
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

    Tensor<float> expected({2, 3});

    expected =
    {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    const double tol = 5e-3;

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);
    }

    Tensor<float> expected_eigvecs({2, 3, 3});
    expected_eigvecs =
    {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,

        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

    }
}

/**
 * @test EIG.eig_alias_view_strided
 * @brief eig() works on a strided alias/view.
 *
 * Creates a 10×10 owner matrix on device, builds a non-contiguous view that
 * selects every-other row/column (5×5) with a diagonal, computes eig() on the
 * view and verifies eigenvalues and eigenvectors match the diagonal.
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
        }
    }

    Tensor<float> owner({N, N}, MemoryLocation::DEVICE);
    owner = owner_flat;

    std::vector<uint64_t> start_indices = { start_row, start_col };
    std::vector<uint64_t> view_dims = { n, n };
    std::vector<uint64_t> view_strides = { N * 2, 2 };
    Tensor<float> view(owner, start_indices, view_dims, view_strides);

    auto result = math::eig(view, 250, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<float> expected = {10.0, 20.0, 30.0, 40.0, 50.0};

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    const double tol = 5e-3;

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);
    }

    Tensor<float> expected_eigvecs({5, 5});
    expected_eigvecs =
    {
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

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

    auto result = math::eig(view, 200, 1e-6f);
    Tensor<float> vals = result.first;

    Tensor<float> expected({2, 3});
    expected =
    {
        1.0, 2.0, 3.0,
        7.0, 8.0, 9.0
    };

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    const double tol = 5e-3;

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);
    }

    Tensor<float> expected_eigvecs({2, 3, 3});
    expected_eigvecs =
    {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,

        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

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

    auto result = math::eig(view, 200, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<float> expected =
    {
        1.0, 2.0, 3.0,
        7.0, 8.0, 9.0
    };

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    const double tol = 5e-3;

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);
    }

    Tensor<float> expected_eigvecs({2, 1, 1, 3, 3});
    expected_eigvecs =
    {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,

        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

    }
}

/**
 * @test EIG.eig_3x3_known
 * @brief eig() on a 3×3 symmetric matrix yields correct eigenpairs.
 *
 * Builds a 3×3 test matrix with known eigenvalues and eigenvectors,
 * computes eig() on device, and checks results against expected
 * analytical values within tolerance.
 */
TEST(EIG, eig_3x3_known)
{
    Tensor<float> A({3, 3}, MemoryLocation::DEVICE);
    std::vector<float> data = {
        2, 0, 0,
        0, 3, 4,
        0, 4, 9
    };
    A = data;
    auto result = math::eig(A, 200, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<float> expected =
    {
        2, 1, 11
    };

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    const double tol = 5e-3;

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);
    }

    Tensor<float> expected_eigvecs({3, 3});
    expected_eigvecs =
    {
        1, 0, 0,
        0, 0.894427, 0.447214,
        0, -0.447214, 0.894427
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

    }
}

/**
 * @test EIG.eig_3x3_harder
 * @brief eig() handles a nontrivial 3×3 symmetric case accurately.
 *
 * Runs eig() on a harder 3×3 matrix with off-diagonal structure,
 * verifying computed eigenvalues and normalized eigenvectors match
 * the reference up to floating-point tolerance.
 */
TEST(EIG, eig_3x3_harder)
{
    Tensor<float> A({3, 3}, MemoryLocation::DEVICE);
    std::vector<float> data = {
        4, 2, -2,
        2, 7, 3,
        -2, 3, 9
    };
    A = data;
    auto result = math::eig(A, 200, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<float> expected =
    {
        1.58338673, 7.21995753, 11.19665574
    };

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    const double tol = 5e-3;

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);
    }

    Tensor<float> expected_eigvecs({3, 3});
    expected_eigvecs =
    {
        0.758483, 0.647291, -0.0756199,
        -0.506903, 0.658908, 0.555779,
        0.409577, -0.383217, 0.827884
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

    }
}

/**
 * @test EIG.eig_3x3_combined
 * @brief eig() processes multiple 3×3 batches in one tensor.
 *
 * Combines two distinct 3×3 matrices into a batched tensor and
 * confirms eig() returns correct per-batch eigenvalues and
 * eigenvectors for both cases on device.
 */
TEST(EIG, eig_3x3_combined)
{
    Tensor<float> A({2, 3, 3}, MemoryLocation::DEVICE);
    std::vector<float> data = {
        2, 0, 0,
        0, 3, 4,
        0, 4, 9,

        4, 2, -2,
        2, 7, 3,
        -2, 3, 9
    };
    A = data;
    auto result = math::eig(A, 200, 1e-6f);
    Tensor<float> vals = result.first;

    std::vector<float> expected =
    {
        2, 1, 11,

        1.58338673, 7.21995753, 11.19665574
    };

    auto it_a = vals.begin();
    auto it_b = expected.begin();

    const double tol = 5e-3;

    for (; it_a != vals.end() && it_b != expected.end(); ++it_a, ++it_b)
    {
        EXPECT_NEAR(*it_a, *it_b, tol);
    }

    Tensor<float> expected_eigvecs({2, 3, 3});
    expected_eigvecs =
    {
        1, 0, 0,
        0, 0.894427, 0.447214,
        0, -0.447214, 0.894427,

        0.758483, 0.647291, -0.0756199,
        -0.506903, 0.658908, 0.555779,
        0.409577, -0.383217, 0.827884
    };

    Tensor<float> eigvecs = result.second;

    auto it_c = eigvecs.begin();
    auto it_d = expected_eigvecs.begin();

    for (; it_c != eigvecs.end() && it_d != expected_eigvecs.end();
        ++it_c, ++it_d)
    {
        EXPECT_NEAR(*it_c, *it_d, tol);

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