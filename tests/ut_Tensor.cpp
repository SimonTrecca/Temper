/**
 * @file ut_Tensor.cpp
 * @brief Google Test suite for proper Tensor class functionality.
 *
 * This file declares the Tensor class which handles multi-dimensional
 * arrays with row-major memory layout.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <type_traits>

#include "temper/Errors.hpp"

#define private public
#define protected public
#include "temper/Tensor.hpp"
#undef private
#undef protected

using namespace temper;

namespace Test
{

// Helper function for stride-aware copy
/**
 * @brief Copies data from a source tensor to a destination tensor,
 *        respecting the source/dest memory strides and tensor shape.
 * @note  Uses proper shape-based strides to map linear indices
 *        back to multi-dimensional coords.
 */
template <typename value_t>
void copy_tensor_data(Tensor<value_t>& dest, const Tensor<value_t>& src)
{
    ASSERT_EQ(dest.m_dimensions, src.m_dimensions);

    uint64_t total_elements = 1;
    for (uint64_t d : src.m_dimensions)
    {
        total_elements *= d;
    }
    if (total_elements == 0)
    {
        return;
    }

    uint64_t rank = static_cast<uint64_t>(src.m_dimensions.size());

    std::vector<uint64_t> shape_strides(rank, 1);
    if (rank >= 2)
    {
        for (uint64_t i = rank - 2; i == 0; --i)
        {
            shape_strides[i] = shape_strides[i + 1] * src.m_dimensions[i + 1];
        }
    }

    uint64_t* dims         = sycl::malloc_shared<uint64_t>(rank, g_sycl_queue);
    uint64_t* src_strides  = sycl::malloc_shared<uint64_t>(rank, g_sycl_queue);
    uint64_t* dest_strides = sycl::malloc_shared<uint64_t>(rank, g_sycl_queue);
    uint64_t* shape_str    = sycl::malloc_shared<uint64_t>(rank, g_sycl_queue);

    std::memcpy(dims,         src.m_dimensions.data(), rank * sizeof(uint64_t));
    std::memcpy(src_strides,  src.m_strides.data(),     rank * sizeof(uint64_t));
    std::memcpy(dest_strides, dest.m_strides.data(),    rank * sizeof(uint64_t));
    std::memcpy(shape_str,    shape_strides.data(),     rank * sizeof(uint64_t));

    value_t* src_data  = src.m_p_data.get();
    value_t* dest_data = dest.m_p_data.get();

    if (!src_data || !dest_data)
    {
        throw std::runtime_error("Null data pointer in copy_tensor_data.");
    }

    g_sycl_queue.parallel_for(sycl::range<1>(total_elements),
        [=](sycl::id<1> idx)
        {
            uint64_t linear = idx[0];
            uint64_t src_offset  = 0;
            uint64_t dest_offset = 0;

            for (uint64_t i = 0; i < rank; ++i)
            {
                uint64_t coord = (linear / shape_str[i]) % dims[i];
                src_offset  += coord * src_strides[i];
                dest_offset += coord * dest_strides[i];
            }

            dest_data[dest_offset] = src_data[src_offset];
        }
    ).wait();

    sycl::free(dims,         g_sycl_queue);
    sycl::free(src_strides,  g_sycl_queue);
    sycl::free(dest_strides, g_sycl_queue);
    sycl::free(shape_str,    g_sycl_queue);
}



template<typename T>
class TypedTensor : public ::testing::Test {};

using TestTypes = ::testing::Types<float, uint64_t>;
TYPED_TEST_SUITE(TypedTensor, TestTypes);

/**
 * @test TypedTensor.compute_strides_empty_dimensions
 * @brief Tests compute_strides() with no dimensions.
 *
 * If the dimensions vector is empty, strides should also remain empty.
 */
TYPED_TEST(TypedTensor, compute_strides_empty_dimensions)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    t.m_dimensions.clear();
    t.compute_strides();
    EXPECT_TRUE(t.m_strides.empty());
}

/**
 * @test TypedTensor.compute_strides_one_dimension
 * @brief Tests compute_strides() with a single dimension.
 *
 * A 1D tensor should always have a single stride of 1.
 */
TYPED_TEST(TypedTensor, compute_strides_one_dimension)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    t.m_dimensions = { 7 };
    t.compute_strides();

    // Single-dim stride should always be 1.
    ASSERT_EQ(t.m_strides.size(), 1u);
    EXPECT_EQ(t.m_strides[0], 1u);
}

/**
 * @test TypedTensor.compute_strides_larger_tensor
 * @brief Tests compute_strides() with four dimensions.
 *
 * For dims = [4, 1, 6, 2], expected strides = [12, 12, 2, 1].
 */
TYPED_TEST(TypedTensor, compute_strides_larger_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    t.m_dimensions = { 4, 1, 6, 2 };
    t.compute_strides();

    // Strides: [1*6*2, 6*2, 2, 1] = [12,12,2,1].
    std::vector<uint64_t> expected = { 12, 12, 2, 1 };
    ASSERT_EQ(t.m_strides, expected);
}

/**
 * @test TypedTensor.compute_strides_overflow_throws
 * @brief compute_strides() should throw if stride multiplication would overflow.
 *
 * Choose suffix dims so their product exceeds uint64_t and triggers overflow.
 */
TYPED_TEST(TypedTensor, compute_strides_overflow_throws)
{
    using value_t = TypeParam;
    const uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    uint64_t dim1 = (U64_MAX / 2) + 1;
    uint64_t dim2 = 2;

    Tensor<value_t> t;
    t.m_dimensions = { 1, dim1, dim2 };

    EXPECT_THROW(t.compute_strides(), temper::bounds_error);
}

/**
 * @test TypedTensor.iterator_constructor
 * @brief Tests that the Tensor::iterator constructor correctly
 * stores the owner pointer and the flat index.
 */
TYPED_TEST(TypedTensor, iterator_constructor)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    typename Tensor<value_t>::iterator it(&t, 0);

    EXPECT_EQ(it.m_p_owner, &t);
    EXPECT_EQ(it.m_flat_idx, 0);

    typename Tensor<value_t>::iterator it_end(&t, 5);
    EXPECT_EQ(it_end.m_p_owner, &t);
    EXPECT_EQ(it_end.m_flat_idx, 5);
}

/**
 * @test TypedTensor.iterator_dereference_returns_view
 * @brief operator* returns a Tensor view equivalent to at(flat).
 */
TYPED_TEST(TypedTensor, iterator_dereference_returns_view)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    t = vals;

    typename Tensor<value_t>::iterator it(&t, 2);
    Tensor<value_t> view = *it;

    EXPECT_EQ(view.get_num_elements(), uint64_t{1});
    EXPECT_FALSE(view.get_owns_data());
    EXPECT_EQ(view.m_p_data.get(), t.at(2).m_p_data.get());
}

/**
 * @test TypedTensor.iterator_pre_post_increment
 * @brief Tests pre-increment and post-increment semantics.
 */
TYPED_TEST(TypedTensor, iterator_pre_post_increment)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::iterator it(&t, 1);

    auto old = it++;
    EXPECT_EQ(old.m_flat_idx, 1);
    EXPECT_EQ(it.m_flat_idx, 2);

    typename Tensor<value_t>::iterator &ref = ++it;
    EXPECT_EQ(ref.m_flat_idx, 3);
    EXPECT_EQ(it.m_flat_idx, 3);
}

/**
 * @test TypedTensor.iterator_pre_post_decrement
 * @brief Tests pre-decrement and post-decrement semantics.
 */
TYPED_TEST(TypedTensor, iterator_pre_post_decrement)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 3}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::iterator it(&t, 5);

    auto old = it--;
    EXPECT_EQ(old.m_flat_idx, 5);
    EXPECT_EQ(it.m_flat_idx, 4);

    typename Tensor<value_t>::iterator &ref = --it;
    EXPECT_EQ(ref.m_flat_idx, 3);
    EXPECT_EQ(it.m_flat_idx, 3);
}

/**
 * @test TypedTensor.iterator_arithmetic_and_compound
 * @brief Tests operator+=, operator-=, operator+ and operator-.
 */
TYPED_TEST(TypedTensor, iterator_arithmetic_and_compound)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 4}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::iterator it(&t, 2);

    using diff_t = typename Tensor<value_t>::iterator::difference_type;

    it += static_cast<diff_t>(3);
    EXPECT_EQ(it.m_flat_idx, 5u);

    it -= static_cast<diff_t>(2);
    EXPECT_EQ(it.m_flat_idx, 3u);

    auto it2 = it + static_cast<diff_t>(4);
    EXPECT_EQ(it2.m_flat_idx, 7u);

    auto it3 = it2 - static_cast<diff_t>(5);
    EXPECT_EQ(it3.m_flat_idx, 2u);
}

/**
 * @test TypedTensor.iterator_difference_and_distance
 * @brief Tests iterator - iterator and std::distance.
 */
TYPED_TEST(TypedTensor, iterator_difference_and_distance)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    auto b = t.begin();
    auto e = t.end();

    auto diff = e - b;
    EXPECT_EQ(diff, static_cast<typename decltype(b)::difference_type>(6));

    auto sd = std::distance(b, e);
    EXPECT_EQ(sd, static_cast<std::ptrdiff_t>(6));
}

/**
 * @test TypedTensor.iterator_comparisons
 * @brief Tests relational comparisons between iterators.
 */
TYPED_TEST(TypedTensor, iterator_comparisons)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 3}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::iterator a(&t, 2);
    typename Tensor<value_t>::iterator b(&t, 5);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    typename Tensor<value_t>::iterator c(&t, 2);
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a != c);
}

/**
 * @test TypedTensor.iterator_roundtrip_arithmetic
 * @brief Ensure combining arithmetic yields consistent flat index.
 */
TYPED_TEST(TypedTensor, iterator_roundtrip_arithmetic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 2}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::iterator it(&t, 1);

    auto it_plus = it + 5;
    it_plus -= 3;
    it_plus += 1;

    EXPECT_EQ(it_plus.m_flat_idx, 4u);

    auto back = it_plus - 3;
    EXPECT_EQ(back.m_flat_idx, 1u);
}

/**
 * @test TypedTensor.const_iterator_constructor
 * @brief Tests that the const_iterator constructor stores owner and index.
 */
TYPED_TEST(TypedTensor, const_iterator_constructor)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    typename Tensor<value_t>::const_iterator it(&t, 0);

    EXPECT_EQ(it.m_p_owner, &t);
    EXPECT_EQ(it.m_flat_idx, 0u);

    typename Tensor<value_t>::const_iterator it_end(&t, 5);
    EXPECT_EQ(it_end.m_p_owner, &t);
    EXPECT_EQ(it_end.m_flat_idx, 5u);
}

/**
 * @test TypedTensor.const_iterator_dereference_returns_view
 * @brief operator* returns a Tensor view equivalent to at(flat).
 */
TYPED_TEST(TypedTensor, const_iterator_dereference_returns_view)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    t = vals;

    typename Tensor<value_t>::const_iterator it(&t, 2);
    Tensor<value_t> view = *it;

    EXPECT_EQ(view.get_num_elements(), uint64_t{1});
    EXPECT_FALSE(view.get_owns_data());
    EXPECT_EQ(view.m_p_data.get(), t.at(2).m_p_data.get());
}

/**
 * @test TypedTensor.const_iterator_pre_post_increment
 * @brief Tests pre-increment and post-increment semantics.
 */
TYPED_TEST(TypedTensor, const_iterator_pre_post_increment)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::const_iterator it(&t, 1);

    auto old = it++;
    EXPECT_EQ(old.m_flat_idx, 1u);
    EXPECT_EQ(it.m_flat_idx, 2u);

    typename Tensor<value_t>::const_iterator &ref = ++it;
    EXPECT_EQ(ref.m_flat_idx, 3u);
    EXPECT_EQ(it.m_flat_idx, 3u);
}

/**
 * @test TypedTensor.const_iterator_pre_post_decrement
 * @brief Tests pre-decrement and post-decrement semantics.
 */
TYPED_TEST(TypedTensor, const_iterator_pre_post_decrement)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 3}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::const_iterator it(&t, 5);

    auto old = it--;
    EXPECT_EQ(old.m_flat_idx, 5u);
    EXPECT_EQ(it.m_flat_idx, 4u);

    typename Tensor<value_t>::const_iterator &ref = --it;
    EXPECT_EQ(ref.m_flat_idx, 3u);
    EXPECT_EQ(it.m_flat_idx, 3u);
}

/**
 * @test TypedTensor.const_iterator_arithmetic_and_compound
 * @brief Tests operator+=, operator-=, operator+ and operator-.
 */
TYPED_TEST(TypedTensor, const_iterator_arithmetic_and_compound)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 4}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::const_iterator it(&t, 2);

    using diff_t =
        typename Tensor<value_t>::const_iterator::difference_type;

    it += static_cast<diff_t>(3);
    EXPECT_EQ(it.m_flat_idx, 5u);

    it -= static_cast<diff_t>(2);
    EXPECT_EQ(it.m_flat_idx, 3u);

    auto it2 = it + static_cast<diff_t>(4);
    EXPECT_EQ(it2.m_flat_idx, 7u);

    auto it3 = it2 - static_cast<diff_t>(5);
    EXPECT_EQ(it3.m_flat_idx, 2u);
}

/**
 * @test TypedTensor.const_iterator_difference_and_distance
 * @brief Tests const_iterator - const_iterator and std::distance.
 */
TYPED_TEST(TypedTensor, const_iterator_difference_and_distance)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    const uint64_t total = t.get_num_elements();

    typename Tensor<value_t>::const_iterator b(&t, 0);
    typename Tensor<value_t>::const_iterator e(&t, total);

    auto diff = e - b;
    EXPECT_EQ(diff,
        static_cast<typename decltype(b)::difference_type>(total));

    auto sd = std::distance(b, e);
    EXPECT_EQ(sd, static_cast<std::ptrdiff_t>(total));
}

/**
 * @test TypedTensor.const_iterator_comparisons
 * @brief Tests relational comparisons between const_iterators.
 */
TYPED_TEST(TypedTensor, const_iterator_comparisons)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 3}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::const_iterator a(&t, 2);
    typename Tensor<value_t>::const_iterator b(&t, 5);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    typename Tensor<value_t>::const_iterator c(&t, 2);
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a != c);
}

/**
 * @test TypedTensor.const_iterator_roundtrip_arithmetic
 * @brief Ensure combining arithmetic yields consistent flat index.
 */
TYPED_TEST(TypedTensor, const_iterator_roundtrip_arithmetic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 2}, MemoryLocation::DEVICE);
    typename Tensor<value_t>::const_iterator it(&t, 1);

    auto it_plus = it + 5;
    it_plus -= 3;
    it_plus += 1;

    EXPECT_EQ(it_plus.m_flat_idx, 4u);

    auto back = it_plus - 3;
    EXPECT_EQ(back.m_flat_idx, 1u);
}

/**
 * @test TypedTensor.stdlib_range_based_for_and_read
 * @brief Range-based for works: iterates all elements and reads values.
 */
TYPED_TEST(TypedTensor, stdlib_range_based_for_and_read)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> expected = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    t = expected;

    size_t idx = 0;
    for (auto view : t) {
        value_t val = static_cast<value_t>(view);
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<double>(val),
                            static_cast<double>(expected[idx++]));
        } else {
            EXPECT_EQ(val, expected[idx++]);
        }
    }
    EXPECT_EQ(idx, expected.size());
}

/**
 * @test TypedTensor.stdlib_distance_next_prev_advance
 * @brief std::distance, std::next, std::prev and std::advance work.
 */
TYPED_TEST(TypedTensor, stdlib_distance_next_prev_advance)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(8);
    std::iota(vals.begin(), vals.end(), static_cast<value_t>(0));
    t = vals;

    auto b = t.begin();
    auto e = t.end();
    EXPECT_EQ(std::distance(b, e), static_cast<std::ptrdiff_t>(8));

    auto it4 = std::next(b, 4);
    EXPECT_EQ(it4.m_flat_idx, 4u);

    auto it2 = std::prev(it4, 2);
    EXPECT_EQ(it2.m_flat_idx, 2u);

    auto it = b;
    std::advance(it, 6);
    EXPECT_EQ(it.m_flat_idx, 6u);
}

/**
 * @test TypedTensor.stdlib_find_transform_accumulate_for_each
 * @brief std::find_if, std::transform, std::accumulate and std::for_each
 * work.
 */
TYPED_TEST(TypedTensor, stdlib_find_transform_accumulate_for_each)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(10), static_cast<value_t>(20),
        static_cast<value_t>(30), static_cast<value_t>(40),
        static_cast<value_t>(50), static_cast<value_t>(60)
    };
    t = vals;

    auto found = std::find_if(t.begin(), t.end(),
        [](const Tensor<value_t>& v) {
            return static_cast<value_t>(v) ==
                   static_cast<value_t>(40);
        });
    ASSERT_NE(found, t.end());
    EXPECT_EQ(found.m_flat_idx, 3u);

    std::vector<value_t> out(6);
    std::transform(t.begin(), t.end(), out.begin(),
        [](const Tensor<value_t>& v) { return static_cast<value_t>(v); });
    EXPECT_EQ(out, vals);

    using acc_t =
        std::conditional_t<std::is_floating_point_v<value_t>, double,
                           uint64_t>;

    acc_t sum = std::accumulate(t.begin(), t.end(), acc_t{0},
        [](acc_t acc, const Tensor<value_t>& v) {
            return acc + static_cast<acc_t>(static_cast<value_t>(v));
        });

    acc_t expected_sum = std::accumulate(vals.begin(), vals.end(),
        acc_t{0}, [](acc_t acc, const value_t &x) {
            return acc + static_cast<acc_t>(x);
        });

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_DOUBLE_EQ(sum, expected_sum);
    } else {
        EXPECT_EQ(sum, expected_sum);
    }

    acc_t sum2 = 0;
    std::for_each(t.begin(), t.end(), [&](const Tensor<value_t>& v){
        sum2 += static_cast<acc_t>(static_cast<value_t>(v));
    });
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_DOUBLE_EQ(sum2, expected_sum);
    } else {
        EXPECT_EQ(sum2, expected_sum);
    }
}

/**
 * @test TypedTensor.stdlib_count_if_and_find_if_not
 * @brief std::count_if and std::find_if (negated) work.
 */
TYPED_TEST(TypedTensor, stdlib_count_if_and_find_if_not)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 1}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(5),
        static_cast<value_t>(10)
    };
    t = vals;

    auto greater_than_two = [](const Tensor<value_t>& v) {
        return static_cast<value_t>(v) > static_cast<value_t>(2);
    };

    auto cnt = std::count_if(t.begin(), t.end(), greater_than_two);
    EXPECT_EQ(cnt, 2);

    auto it_not_gt2 = std::find_if(t.begin(), t.end(),
        [&](const Tensor<value_t>& v){ return !greater_than_two(v); });
    ASSERT_NE(it_not_gt2, t.end());
    EXPECT_EQ(it_not_gt2.m_flat_idx, 0u);
}

/**
 * @test TypedTensor.main_constructor_sets_dimensions_and_strides
 * @brief Tests that the Tensor constructor
 * correctly sets dimensions and computes strides.
 */
TYPED_TEST(TypedTensor, main_constructor_sets_dimensions_and_strides)
{
    using value_t = TypeParam;
    std::vector<uint64_t> dims = { 2, 3, 4 };
    Tensor<value_t> t(dims, MemoryLocation::DEVICE);

    EXPECT_EQ(t.m_dimensions, dims);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<uint64_t> expected_strides = { 12, 4, 1 };
    EXPECT_EQ(t.m_strides, expected_strides);
}

/**
 * @test TypedTensor.main_constructor_zero_initializes_data
 * @brief Tests that the Tensor constructor allocates the
 * correct amount of memory and initializes it to zero.
 */
TYPED_TEST(TypedTensor, main_constructor_zero_initializes_data)
{
    using value_t = TypeParam;
    std::vector<uint64_t> dims = { 2, 3 };
    Tensor<value_t> t(dims, MemoryLocation::HOST);

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : dims)
    {
        total_size *= d;
    }

    std::vector<value_t> host_data(total_size);
    sycl::event e = g_sycl_queue.memcpy(
        host_data.data(),
        t.m_p_data.get(),
        sizeof(value_t) * total_size
    );
    e.wait();

    for (value_t v : host_data)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(v), 0.0f);
        } else {
            EXPECT_EQ(v, static_cast<value_t>(0));
        }
    }
}

/**
 * @test TypedTensor.main_constructor_autograd_defaults
 * @brief Default-constructed and main-constructed owning tensors must have
 * cleared Autograd meta (fn==nullptr, grad==nullptr, requires_grad==false).
 */
TYPED_TEST(TypedTensor, main_constructor_autograd_defaults)
{
    using value_t = TypeParam;
    Tensor<value_t> def;
    EXPECT_FALSE(def.m_meta.requires_grad);
    EXPECT_EQ(def.m_meta.grad, nullptr);
    EXPECT_EQ(def.m_meta.fn, nullptr);

    Tensor<value_t> own({2,2}, MemoryLocation::HOST);
    EXPECT_FALSE(own.m_meta.requires_grad);
    EXPECT_EQ(own.m_meta.grad, nullptr);
    EXPECT_EQ(own.m_meta.fn, nullptr);
}

/**
 * @test TypedTensor.main_constructor_memory_location_and_access
 * @brief Tests correct memory location assignment.
 */
TYPED_TEST(TypedTensor, main_constructor_memory_location_and_access)
{
    using value_t = TypeParam;
    Tensor<value_t> t_device(std::vector<uint64_t>{1, 1},
                             MemoryLocation::DEVICE);
    EXPECT_EQ(t_device.m_mem_loc, MemoryLocation::DEVICE);

    // Launch kernel to set element to 42.
    g_sycl_queue.submit([&](sycl::handler & cgh)
    {
        value_t * ptr = t_device.m_p_data.get();
        cgh.single_task([=]()
        {
            ptr[0] = static_cast<value_t>(42);
        });
    }).wait();

    // Copy back to host and check.
    value_t host_val = static_cast<value_t>(0);
    g_sycl_queue.memcpy(&host_val, t_device.m_p_data.get(),
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host_val),
                        static_cast<float>(42.0f));
    } else {
        EXPECT_EQ(host_val, static_cast<value_t>(42));
    }

    // HOST tensor test: write directly on host memory and read back.
    Tensor<value_t> t_host({1, 1}, MemoryLocation::HOST);
    EXPECT_EQ(t_host.m_mem_loc, MemoryLocation::HOST);

    // Direct write on host pointer.
    t_host.m_p_data.get()[0] = static_cast<value_t>(24);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t_host.m_p_data.get()[0]),
                        static_cast<float>(24.0f));
    } else {
        EXPECT_EQ(t_host.m_p_data.get()[0], static_cast<value_t>(24));
    }
}

/**
 * @test TypedTensor.main_constructor_empty_dimensions
 * @brief Throws temper::validation_error when dimensions vector is empty.
 */
TYPED_TEST(TypedTensor, main_constructor_empty_dimensions)
{
    using value_t = TypeParam;
    std::vector<uint64_t> dims = {};
    EXPECT_THROW(
        Tensor<value_t> t(dims, MemoryLocation::HOST),
        temper::validation_error
    );
}

/**
 * @test TypedTensor.main_constructor_zero_dimension
 * @brief Throws temper::validation_error when any dimension is zero.
 */
TYPED_TEST(TypedTensor, main_constructor_zero_dimension)
{
    using value_t = TypeParam;
    std::vector<uint64_t> dims = { 2, 0, 3 };

    EXPECT_THROW(
        Tensor<value_t> t(dims, MemoryLocation::HOST),
        temper::validation_error
    );
}

/**
 * @test TypedTensor.main_constructor_element_count_overflow
 * @brief Throws temper::bounds_error when total_size would overflow uint64_t.
 */
TYPED_TEST(TypedTensor, main_constructor_element_count_overflow)
{
    using value_t = TypeParam;
    std::vector<uint64_t> dims = {
        std::numeric_limits<uint64_t>::max(), 2
    };

    EXPECT_THROW(
        Tensor<value_t> t(dims, MemoryLocation::HOST),
        temper::bounds_error
    );
}

/**
 * @test TypedTensor.main_constructor_allocation_bytes_overflow
 * @brief Throws overflow_error when allocation size exceeds uint64_t.
 */
TYPED_TEST(TypedTensor, main_constructor_allocation_bytes_overflow)
{
    using value_t = TypeParam;
    std::vector<uint64_t> dims =
        { std::numeric_limits<uint64_t>::max() / sizeof(value_t) + 1 };

    EXPECT_THROW(
        Tensor<value_t> t(dims, MemoryLocation::HOST),
        temper::bounds_error
    );
}

/**
 * @test TypedTensor.main_constructor_exceeds_device_max_alloc
 * @brief Throws temper::device_error if requested allocation exceeds
 * device max_mem_alloc_size.
 */
TYPED_TEST(TypedTensor, main_constructor_exceeds_device_max_alloc)
{
    using value_t = TypeParam;
    auto dev = g_sycl_queue.get_device();
    uint64_t dev_max_alloc =
        dev.get_info<sycl::info::device::max_mem_alloc_size>();

    std::vector<uint64_t> dims = std::vector<uint64_t>{
        (dev_max_alloc / sizeof(value_t)) + 1
    };
    EXPECT_THROW(
        Tensor<value_t> t(dims, MemoryLocation::DEVICE),
        temper::device_error
    );
}

/**
 * @test TypedTensor.copy_constructor
 * @brief Tests copy constructor.
 */
TYPED_TEST(TypedTensor, copy_constructor)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> values = {
        static_cast<value_t>(1.0),
        static_cast<value_t>(2.0),
        static_cast<value_t>(3.0),
        static_cast<value_t>(4.0)
    };
    t1 = values;

    Tensor<value_t> t2(t1);

    EXPECT_EQ(t2.m_mem_loc, t1.m_mem_loc);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), t2.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        static_cast<float>(1.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        static_cast<float>(2.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        static_cast<float>(3.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]),
                        static_cast<float>(4.0f));
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(2));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(4));
    }
}

/**
 * @test TypedTensor.copy_constructor_default
 * @brief Tests that copying a default-constructed tensor
 * results in another empty tensor with nullptr data.
 */
TYPED_TEST(TypedTensor, copy_constructor_on_default_constructed)
{
    using value_t = TypeParam;
    Tensor<value_t> t1;

    Tensor<value_t> t2(t1);

    EXPECT_TRUE(t2.m_dimensions.empty());
    EXPECT_EQ(t2.m_p_data, nullptr);
    EXPECT_TRUE(t2.m_own_data);
}

/**
 * @test TypedTensor.copy_constructor_autograd_clears_meta
 * @brief Copying an owning tensor should clear autograd metadata on the copy.
 */
TYPED_TEST(TypedTensor, copy_constructor_autograd_clears_meta)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(1), static_cast<value_t>(2),
                              static_cast<value_t>(3), static_cast<value_t>(4) };

    t.m_meta.requires_grad = true;
    t.m_meta.grad = t.m_p_data;

    Tensor<value_t> cp(t);

    // Original should still indicate gradients required and have grad ptr.
    EXPECT_TRUE(t.m_meta.requires_grad);
    ASSERT_NE(t.m_meta.grad, nullptr);

    // Copy constructed owning tensor must have fresh (cleared) autograd meta.
    EXPECT_FALSE(cp.m_meta.requires_grad);
    EXPECT_EQ(cp.m_meta.grad, nullptr);
}

/**
 * @test TypedTensor.copy_constructor_autograd_from_view_preserves_meta
 * @brief Copy-constructing from a non-owning view should preserve the
 * Autograd meta (fn, requires_grad and aliased grad pointer with offset).
 */
TYPED_TEST(TypedTensor, copy_constructor_autograd_from_view_preserves_meta)
{
    using value_t = TypeParam;
    // Prepare owner with grad and a dummy FunctionEdge
    Tensor<value_t> owner({2,3}, MemoryLocation::HOST);
    owner = std::vector<value_t>{1,2,3,4,5,6};

    struct DummyEdge : public temper::FunctionEdge<value_t>
    {
        DummyEdge() : FunctionEdge<value_t>("dummy") {}
        void forward() override {}
        void backward(const Tensor<value_t>&) override {}
        std::vector<std::shared_ptr<Tensor<value_t>>> inputs() const override
        { return {}; }
        std::shared_ptr<Tensor<value_t>> output() const override
        { return nullptr; }
    };

    owner.m_meta.requires_grad = true;
    owner.m_meta.grad = owner.m_p_data;
    owner.m_meta.fn = std::make_shared<DummyEdge>();

    // Create a view and then copy-construct from the view
    Tensor<value_t> view(owner,
        std::vector<uint64_t>{1,1},
        std::vector<uint64_t>{1,2});
    Tensor<value_t> cp(view);

    EXPECT_TRUE(cp.m_meta.requires_grad);
    ASSERT_NE(cp.m_meta.grad, nullptr);
    // grad should be an alias into same control block (no offset here is 1*stride0+1*stride1)
    uint64_t offset = 1 * owner.m_strides[0] + 1 * owner.m_strides[1];
    EXPECT_EQ(cp.m_meta.grad.get(), owner.m_meta.grad.get() + offset);
    ASSERT_NE(cp.m_meta.fn, nullptr);
    EXPECT_EQ(cp.m_meta.fn.get(), owner.m_meta.fn.get());
}

/**
 * @test TypedTensor.copy_constructor_host
 * @brief Tests copy constructor with a tensor allocated in HOST memory.
 * Ensures contents are copied correctly.
 */
TYPED_TEST(TypedTensor, copy_constructor_host)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> values = {
        static_cast<value_t>(10.0),
        static_cast<value_t>(20.0),
        static_cast<value_t>(30.0),
        static_cast<value_t>(40.0)
    };
    t1 = values;

    Tensor<value_t> t2(t1);

    std::vector<value_t> host(4);
    std::memcpy(host.data(), t2.m_p_data.get(),
                sizeof(value_t) * 4);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        static_cast<float>(10.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        static_cast<float>(20.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        static_cast<float>(30.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]),
                        static_cast<float>(40.0f));
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(10));
        EXPECT_EQ(host[1], static_cast<value_t>(20));
        EXPECT_EQ(host[2], static_cast<value_t>(30));
        EXPECT_EQ(host[3], static_cast<value_t>(40));
    }
}

/**
 * @test TypedTensor.copy_constructor_view
 * @brief Tests copy constructor on a view tensor (non-owning).
 * Copying a view must produce another view, sharing the same memory.
 */
TYPED_TEST(TypedTensor, copy_constructor_view)
{
    using value_t = TypeParam;
    Tensor<value_t> t1(std::vector<uint64_t>{4},
                       MemoryLocation::DEVICE);
    std::vector<value_t> values = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    t1 = values;

    Tensor<value_t> view(t1, {2}, {2});
    ASSERT_FALSE(view.m_own_data);

    Tensor<value_t> copy(view);

    EXPECT_EQ(copy.m_dimensions, view.m_dimensions);
    EXPECT_FALSE(copy.m_own_data);
    EXPECT_EQ(copy.m_p_data.get(), view.m_p_data.get());
}

/**
 * @test TypedTensor.move_constructor
 * @brief Tests move constructor.
 */
TYPED_TEST(TypedTensor, move_constructor)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> values = {
        static_cast<value_t>(5.0),
        static_cast<value_t>(6.0),
        static_cast<value_t>(7.0),
        static_cast<value_t>(8.0)
    };
    t1 = values;

    value_t* original_ptr = t1.m_p_data.get();
    MemoryLocation original_loc = t1.m_mem_loc;

    Tensor<value_t> t2(std::move(t1));

    EXPECT_EQ(t2.m_p_data.get(), original_ptr);
    EXPECT_EQ(t2.m_mem_loc, original_loc);
    EXPECT_EQ(t1.m_p_data.get(), nullptr);
}

/**
 * @test TypedTensor.move_constructor_autograd_transfers_meta
 * @brief Move-construction should transfer Autograd meta and clear the source.
 */
TYPED_TEST(TypedTensor, move_constructor_autograd_transfers_meta)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2,2}, MemoryLocation::HOST);
    src = std::vector<value_t>{1,2,3,4};

    struct DummyEdge : public temper::FunctionEdge<value_t>
    {
        DummyEdge() : FunctionEdge<value_t>("move") {}
        void forward() override {}
        void backward(const Tensor<value_t>&) override {}
        std::vector<std::shared_ptr<Tensor<value_t>>> inputs() const override
        { return {}; }
        std::shared_ptr<Tensor<value_t>> output() const override
        { return nullptr; }
    };

    src.m_meta.requires_grad = true;
    src.m_meta.grad = src.m_p_data;
    src.m_meta.fn = std::make_shared<DummyEdge>();

    value_t* raw_grad_ptr = src.m_meta.grad.get();
    auto fn_ptr = src.m_meta.fn.get();

    Tensor<value_t> moved(std::move(src));

    // moved should have the metadata
    EXPECT_TRUE(moved.m_meta.requires_grad);
    ASSERT_NE(moved.m_meta.grad, nullptr);
    EXPECT_EQ(moved.m_meta.grad.get(), raw_grad_ptr);
    ASSERT_NE(moved.m_meta.fn, nullptr);
    EXPECT_EQ(moved.m_meta.fn.get(), fn_ptr);

    // source must be cleared
    EXPECT_FALSE(src.m_meta.requires_grad);
    EXPECT_EQ(src.m_meta.grad, nullptr);
    EXPECT_EQ(src.m_meta.fn, nullptr);
}

/**
 * @test TypedTensor.scalar_constructor_host
 * @brief Construction from a scalar (default HOST).
 */
TYPED_TEST(TypedTensor, scalar_constructor_host)
{
    using value_t = TypeParam;
    Tensor<value_t> t(static_cast<value_t>(3.14),
                      MemoryLocation::HOST);

    EXPECT_EQ(t.m_dimensions, std::vector<uint64_t>({1}));
    EXPECT_EQ(t.m_strides, std::vector<uint64_t>({1}));
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);
    EXPECT_TRUE(t.m_own_data);

    value_t host_val = t.m_p_data.get()[0];
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host_val),
                        static_cast<float>(3.14f));
    } else {
        EXPECT_EQ(host_val, static_cast<value_t>(3));
    }
}

/**
 * @test TypedTensor.scalar_constructor_device
 * @brief Construction from a scalar into DEVICE memory.
 */
TYPED_TEST(TypedTensor, scalar_constructor_device)
{
    using value_t = TypeParam;
    Tensor<value_t> t(static_cast<value_t>(2.718),
                      MemoryLocation::DEVICE);

    EXPECT_EQ(t.m_dimensions, std::vector<uint64_t>({1}));
    EXPECT_EQ(t.m_strides, std::vector<uint64_t>({1}));
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);
    EXPECT_TRUE(t.m_own_data);

    value_t host_val = static_cast<value_t>(0);
    g_sycl_queue.memcpy(&host_val, t.m_p_data.get(),
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host_val),
                        static_cast<float>(2.718f));
    } else {
        EXPECT_EQ(host_val, static_cast<value_t>(2));
    }
}

/**
 * @test TypedTensor.scalar_constructor_autograd_defaults
 * @brief Scalar constructor must initialize autograd meta to defaults.
 */
TYPED_TEST(TypedTensor, scalar_constructor_autograd_defaults)
{
    using value_t = TypeParam;
    Tensor<value_t> s(static_cast<value_t>(3.14), MemoryLocation::HOST);
    EXPECT_FALSE(s.m_meta.requires_grad);
    EXPECT_EQ(s.m_meta.grad, nullptr);
    EXPECT_EQ(s.m_meta.fn, nullptr);
}

/**
 * @test TypedTensor.scalar_constructor_used_for_parameter_passing
 * @brief Ensure explicit conversion allows passing a float
 * where Tensor is expected.
 */
TYPED_TEST(TypedTensor, scalar_constructor_used_for_parameter_passing)
{
    using value_t = TypeParam;
    auto read_scalar = [&](Tensor<value_t> x) -> value_t
    {
        value_t host_val = static_cast<value_t>(0);
        g_sycl_queue.memcpy(&host_val, x.m_p_data.get(),
                           sizeof(value_t)).wait();
        return host_val;
    };

    value_t got = read_scalar(Tensor<value_t>(static_cast<value_t>(9.81)));
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(got),
                        static_cast<float>(9.81f));
    } else {
        EXPECT_EQ(got, static_cast<value_t>(9));
    }
}

/**
 * @test TypedTensor.view_constructor_preserves_strides_and_data
 * @brief Tests that slicing a CHW-format tensor creates
 * a view with correct strides and verifies that the data
 * accessed via the view matches expected values.
 */
TYPED_TEST(TypedTensor, view_constructor_preserves_strides_and_data)
{
    using value_t = TypeParam;
    Tensor<value_t> img({3, 4, 5}, MemoryLocation::DEVICE);
    EXPECT_EQ(img.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> vals(3 * 4 * 5);

    for (uint64_t c = 0; c < 3; ++c)
    {
        for (uint64_t i = 0; i < 4; ++i)
        {
            for (uint64_t j = 0; j < 5; ++j)
            {
                vals[c * 20 + i * 5 + j] =
                    static_cast<value_t>(c * 100 + i * 10 + j);
            }
        }
    }

    img = vals;

    Tensor<value_t> patch(img, {1, 0, 0}, {2, 3});

    EXPECT_EQ(patch.m_mem_loc, MemoryLocation::DEVICE);

    EXPECT_EQ(patch.m_dimensions, std::vector<uint64_t>({2, 3}));
    EXPECT_EQ(patch.m_strides[0], img.m_strides[1]);
    EXPECT_EQ(patch.m_strides[1], img.m_strides[2]);

    Tensor<value_t> host({2, 3}, MemoryLocation::HOST);
    copy_tensor_data(host, patch);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    for (int ii = 0; ii < 2; ++ii)
    {
        for (int jj = 0; jj < 3; ++jj)
        {
            value_t expected =
                static_cast<value_t>(100 + ii * 10 + jj);
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(out[ii * 3 + jj]),
                                static_cast<float>(expected));
            } else {
                EXPECT_EQ(out[ii * 3 + jj], expected);
            }
        }
    }
}

/**
 * @test TypedTensor.view_constructor_identity_preserves_layout
 * @brief Verifies that slicing a tensor without dropping
 * any axes returns a view with identical dimensions,
 * strides, and values.
 */
TYPED_TEST(TypedTensor, view_constructor_identity_preserves_layout)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> v = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    t = v;

    Tensor<value_t> view(t, {0, 0}, {2, 3});

    EXPECT_EQ(view.m_mem_loc, MemoryLocation::DEVICE);

    EXPECT_EQ(view.m_dimensions, t.m_dimensions);
    EXPECT_EQ(view.m_strides, t.m_strides);

    Tensor<value_t> host({2, 3}, MemoryLocation::HOST);
    copy_tensor_data(host, view);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[i]),
                            static_cast<float>(v[i]));
        } else {
            EXPECT_EQ(out[i], v[i]);
        }
    }
}

/**
 * @test TypedTensor.view_constructor_autograd
 * @brief Ensure a view aliases the owner's autograd gradient buffer with offset.
 */
TYPED_TEST(TypedTensor, view_constructor_autograd)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5), static_cast<value_t>(6)
    };
    owner = vals;

    // Simulate gradient buffer owned by the tensor by reusing the data control
    // block as a gradient buffer for test purposes.
    owner.m_meta.requires_grad = true;
    owner.m_meta.grad = owner.m_p_data;

    std::vector<uint64_t> start = {1, 1};
    std::vector<uint64_t> shape = {1, 2};
    Tensor<value_t> view(owner, start, shape);

    uint64_t offset = start[0] * owner.m_strides[0]
                    + start[1] * owner.m_strides[1];

    EXPECT_TRUE(view.m_meta.requires_grad);
    ASSERT_NE(view.m_meta.grad, nullptr);
    EXPECT_EQ(view.m_meta.grad.get(), owner.m_meta.grad.get() + offset);
}

/**
 * @test TypedTensor.view_constructor_invalid_arguments_throw
 * @brief Ensures that invalid slice arguments
 * (e.g., mismatched ranks, out-of-bounds access)
 * correctly throw exceptions.
 */
TYPED_TEST(TypedTensor, view_constructor_invalid_arguments_throw)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2});

    // Too few shape dimensions test.
    EXPECT_THROW((Tensor<value_t>(t, {0, 0}, {1})),
        temper::validation_error);

    // Too many shape dimensions test.
    EXPECT_THROW((Tensor<value_t>(t, {0, 0, 0}, {1, 1, 1, 1})),
        temper::validation_error);

    // Zero-size dimension test.
    EXPECT_THROW((Tensor<value_t>(t, {0, 0, 0}, {0, 1})),
        temper::bounds_error);

    // Out-of-bounds shape test.
    EXPECT_THROW((Tensor<value_t>(t, {0, 0, 0}, {3, 1})),
        temper::bounds_error);
    EXPECT_THROW((Tensor<value_t>(t, {2, 0, 0}, {1, 1})),
        temper::bounds_error);
}

/**
 * @test TypedTensor.view_constructor_4d_drops_prefix_axes
 * @brief Tests slicing a 4D tensor while dropping the first two axes,
 * verifying correct shape, strides, and values in the resulting view.
 *
 * The original tensor has shape {2, 3, 4, 5} and is filled with
 * values from 1 to 120.
 * The slice extracts the {4, 5} sub-tensor at position (0, 0, :, :)
 */
TYPED_TEST(TypedTensor, view_constructor_4d_drops_prefix_axes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4, 5});
    std::vector<value_t> vals(120);

    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }

    t = vals;

    // Slice last two dimensions: drop first two axes.
    Tensor<value_t> slice(t, {0, 0, 0, 0}, {4, 5});

    EXPECT_EQ(slice.m_dimensions, std::vector<uint64_t>({4, 5}));
    EXPECT_EQ(slice.m_strides[0], t.m_strides[2]);
    EXPECT_EQ(slice.m_strides[1], t.m_strides[3]);

    Tensor<value_t> host({4, 5});
    copy_tensor_data(host, slice);

    std::vector<value_t> out(20);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 20).wait();

    for (uint64_t x = 0; x < 4; ++x)
    {
        for (uint64_t y = 0; y < 5; ++y)
        {
            value_t expected =
                static_cast<value_t>((0 * 60) + (0 * 20) + (x * 5) + y + 1);
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(out[x * 5 + y]),
                                static_cast<float>(expected));
            } else {
                EXPECT_EQ(out[x * 5 + y], expected);
            }
        }
    }
}

/**
 * @test TypedTensor.view_constructor_4d_extracts_3d_volume
 * @brief Extracts a 3D chunk from a 4D tensor by dropping the first axis
 * and verifies shape, stride, and copied values.
 */
TYPED_TEST(TypedTensor, view_constructor_4d_extracts_3d_volume)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4, 5});
    std::vector<value_t> vals(120);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }

    t = vals;

    Tensor<value_t> slice(t, {1, 0, 0, 0}, {3, 4, 5});

    EXPECT_EQ(slice.m_dimensions, std::vector<uint64_t>({3, 4, 5}));
    EXPECT_EQ(slice.m_strides[0], t.m_strides[1]);
    EXPECT_EQ(slice.m_strides[1], t.m_strides[2]);
    EXPECT_EQ(slice.m_strides[2], t.m_strides[3]);

    Tensor<value_t> host({3, 4, 5});
    copy_tensor_data(host, slice);

    std::vector<value_t> out(60);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 60).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            for (uint64_t k = 0; k < 5; ++k)
            {
                value_t expected =
                    static_cast<value_t>((1 * 60) + (i * 20) + (j * 5) + k + 1);
                if constexpr (std::is_floating_point_v<value_t>) {
                    EXPECT_FLOAT_EQ(static_cast<float>(out[i * 20 + j * 5 + k]),
                                    static_cast<float>(expected));
                } else {
                    EXPECT_EQ(out[i * 20 + j * 5 + k], expected);
                }
            }
        }
    }
}

/**
 * @test TypedTensor.view_constructor_4d_extracts_1d_row
 * @brief Slices a single row (1D) from the last dimension of a 4D tensor,
 * and verifies the extracted values.
 */
TYPED_TEST(TypedTensor, view_constructor_4d_extracts_1d_row)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4, 5});
    std::vector<value_t> vals(120);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    t = vals;

    Tensor<value_t> slice(t, {1, 2, 3, 0}, {5});

    EXPECT_EQ(slice.m_dimensions, std::vector<uint64_t>({5}));
    EXPECT_EQ(slice.m_strides[0], t.m_strides[3]);

    Tensor<value_t> host({5});
    copy_tensor_data(host, slice);

    std::vector<value_t> out(5);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 5).wait();

    for (uint64_t k = 0; k < 5; ++k)
    {
        value_t expected =
            static_cast<value_t>((1 * 60) + (2 * 20) + (3 * 5) + k + 1);
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[k]),
                            static_cast<float>(expected));
        } else {
            EXPECT_EQ(out[k], expected);
        }
    }
}

/**
 * @test TypedTensor.view_constructor_chw_extracts_large_patch
 * @brief Slices a 100x100 patch from a 3D CHW-format tensor
 * at a specified spatial location.
 * Verifies shape, stride, and content correctness.
 */
TYPED_TEST(TypedTensor, view_constructor_chw_extracts_large_patch)
{
    using value_t = TypeParam;
    Tensor<value_t> img({3, 256, 256});
    std::vector<value_t> vals(3 * 256 * 256);

    for (uint64_t c = 0; c < 3; ++c)
    {
        for (uint64_t h = 0; h < 256; ++h)
        {
            for (uint64_t w = 0; w < 256; ++w)
            {
                vals[c * 256 * 256 + h * 256 + w] =
                    static_cast<value_t>(c * 1000000 + h * 1000 + w);
            }
        }
    }

    img = vals;

    Tensor<value_t> patch(img, {0, 50, 70}, {100, 100});

    EXPECT_EQ(patch.m_dimensions, std::vector<uint64_t>({100, 100}));
    EXPECT_EQ(patch.m_strides[0], img.m_strides[1]);
    EXPECT_EQ(patch.m_strides[1], img.m_strides[2]);

    Tensor<value_t> host({100, 100});
    copy_tensor_data(host, patch);

    std::vector<value_t> out(100 * 100);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 100 * 100).wait();

    for (uint64_t i = 0; i < 100; ++i)
    {
        for (uint64_t j = 0; j < 100; ++j)
        {
            value_t expected =
                static_cast<value_t>((0 * 1000000) + (50 + i) * 1000
                                     + (70 + j));
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(out[i * 100 + j]),
                                static_cast<float>(expected));
            } else {
                EXPECT_EQ(out[i * 100 + j], expected);
            }
        }
    }
}

/**
 * @test TypedTensor.view_constructor_modification_reflects_in_original
 * @brief Tests that modifying a tensor view updates the original tensor's
 * memory.
 */
TYPED_TEST(TypedTensor, view_constructor_modification_reflects_in_original)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 4, 5});
    std::vector<value_t> vals(3 * 4 * 5, static_cast<value_t>(0));
    t = vals;

    Tensor<value_t> view(t, {1, 0, 0}, {4, 5});

    // Prepare values to write into the view.
    Tensor<value_t> host({4, 5});
    std::vector<value_t> patch_vals(4 * 5);

    for (uint64_t i = 0; i < 4; ++i)
    {
        for (uint64_t j = 0; j < 5; ++j)
        {
            patch_vals[i * 5 + j] =
                static_cast<value_t>(42 + i * 5 + j);
        }
    }
    host = patch_vals;

    // Write to view (which writes into t).
    copy_tensor_data(view, host);

    // Read again from the same region of t.
    Tensor<value_t> ref(t, {1, 0, 0}, {4, 5});
    Tensor<value_t> readback({4, 5});
    copy_tensor_data(readback, ref);

    std::vector<value_t> out(4 * 5);
    g_sycl_queue.memcpy(out.data(), readback.m_p_data.get(),
                       sizeof(value_t) * 4 * 5).wait();

    for (uint64_t i = 0; i < 4; ++i)
    {
        for (uint64_t j = 0; j < 5; ++j)
        {
            value_t expected = static_cast<value_t>(42 + i * 5 + j);
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(out[i * 5 + j]),
                                static_cast<float>(expected));
            } else {
                EXPECT_EQ(out[i * 5 + j], expected);
            }
        }
    }
}

/**
 * @test TypedTensor.view_constructor_owner_destroyed_before_view
 * @brief Ensure a view's aliasing shared_ptr keeps the underlying buffer
 * alive after the original owner goes out of scope.
 */
TYPED_TEST(TypedTensor, view_constructor_owner_destroyed_before_view)
{
    using value_t = TypeParam;
    std::weak_ptr<value_t> weak_data_ptr;
    Tensor<value_t> view;
    {
        Tensor<value_t> owner({2, 2});
        std::vector<value_t> vals = {
            static_cast<value_t>(1.1),
            static_cast<value_t>(2.2),
            static_cast<value_t>(3.3),
            static_cast<value_t>(4.4)
        };
        owner = vals;

        std::vector<uint64_t> start{0, 0};
        std::vector<uint64_t> shape{2, 2};
        view = Tensor<value_t>(owner, start, shape);
        weak_data_ptr = view.m_p_data;
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(view[0][0]),
                            static_cast<float>(1.1f));
        } else {
            EXPECT_EQ(static_cast<value_t>(view[0][0]),
                      static_cast<value_t>(1));
        }
    }
    EXPECT_FALSE(weak_data_ptr.expired());
}

/**
 * @test TypedTensor.view_constructor_view_destroyed_before_owner
 * @brief Ensure the buffer remains alive while the owner exists and is freed
 * after the owner releases ownership.
 */
TYPED_TEST(TypedTensor, view_constructor_view_destroyed_before_owner)
{
    using value_t = TypeParam;
    std::weak_ptr<value_t> weak_data_ptr;

    Tensor<value_t> owner({2, 2});
    std::vector<value_t> vals = {
        static_cast<value_t>(5.5),
        static_cast<value_t>(6.6),
        static_cast<value_t>(7.7),
        static_cast<value_t>(8.8)
    };
    owner = vals;

    {
        std::vector<uint64_t> start{0, 0};
        std::vector<uint64_t> shape{2, 2};
        Tensor<value_t> view(owner, start, shape);

        weak_data_ptr = view.m_p_data;

        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(view[0][0]),
                            static_cast<float>(5.5f));
        } else {
            EXPECT_EQ(static_cast<value_t>(view[0][0]),
                      static_cast<value_t>(5));
        }
    }
    EXPECT_FALSE(weak_data_ptr.expired());

    owner = Tensor<value_t>();
    EXPECT_TRUE(weak_data_ptr.expired());
}

/**
 * @test TypedTensor.view_constructor_from_uninitialized_throws
 * @brief Creating a view from a default-constructed/moved-from
 * tensor must throw.
 */
TYPED_TEST(TypedTensor, view_constructor_from_uninitialized_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> owner;

    std::vector<uint64_t> start = {0};
    std::vector<uint64_t> shape = {1};

    EXPECT_THROW(
        Tensor<value_t> view(owner, start, shape),
        temper::validation_error
    );

    Tensor<value_t> valid({2,2}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    valid = vals;

    Tensor<value_t> moved = std::move(valid);
    EXPECT_THROW(
        Tensor<value_t> view2(
            valid, std::vector<uint64_t>{0,0},
            std::vector<uint64_t>{1,1}
        ),
        temper::validation_error
    );
}

/**
 * @test TypedTensor.view_constructor_alias_pointer_offset
 * @brief View constructor must alias the owner's pointer at the correct
 * offset.
 */
TYPED_TEST(TypedTensor, view_constructor_alias_pointer_offset)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(10), static_cast<value_t>(11),
        static_cast<value_t>(12), static_cast<value_t>(20),
        static_cast<value_t>(21), static_cast<value_t>(22)
    };
    owner = vals;

    std::vector<uint64_t> start = {1, 1};
    std::vector<uint64_t> shape = {1, 2};

    Tensor<value_t> view(owner, start, shape);

    uint64_t offset =
        start[0] * owner.m_strides[0] + start[1] * owner.m_strides[1];

    EXPECT_EQ(view.m_p_data.get(), owner.m_p_data.get() + offset);

    std::vector<value_t> dst(2);
    g_sycl_queue.memcpy(dst.data(), view.m_p_data.get(),
                       sizeof(value_t) * 2).wait();
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(dst[0]),
                        static_cast<float>(vals[offset + 0]));
        EXPECT_FLOAT_EQ(static_cast<float>(dst[1]),
                        static_cast<float>(vals[offset + 1]));
    } else {
        EXPECT_EQ(dst[0], vals[offset + 0]);
        EXPECT_EQ(dst[1], vals[offset + 1]);
    }
}

/**
 * @test TypedTensor.view_constructor_from_alias
 * @brief Ensures that creating a view from an alias works correctly.
 */
TYPED_TEST(TypedTensor, view_constructor_from_alias)
{
    using value_t = TypeParam;
    // Owner tensor 2x3
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(0.f), static_cast<value_t>(1.f),
        static_cast<value_t>(2.f), static_cast<value_t>(3.f),
        static_cast<value_t>(4.f), static_cast<value_t>(5.f)
    };
    t = vals;

    Tensor<value_t> alias(t, {0,0}, {3,2}, {1,3});
    EXPECT_EQ(alias.m_dimensions, std::vector<uint64_t>({3,2}));
    EXPECT_EQ(alias.m_strides, std::vector<uint64_t>({1,3}));

    Tensor<value_t> subview(alias, {0,0}, {2,2});
    EXPECT_EQ(subview.m_dimensions, std::vector<uint64_t>({2,2}));
    EXPECT_EQ(subview.m_strides[0], alias.m_strides[0]);
    EXPECT_EQ(subview.m_strides[1], alias.m_strides[1]);

    Tensor<value_t> host({2,2}, MemoryLocation::HOST);
    copy_tensor_data(host, subview);

    std::vector<value_t> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    // Expected values from transposed alias:
    // subview = [ [0,3], [1,4] ].
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 0.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 3.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[3]), 4.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(0));
        EXPECT_EQ(out[1], static_cast<value_t>(3));
        EXPECT_EQ(out[2], static_cast<value_t>(1));
        EXPECT_EQ(out[3], static_cast<value_t>(4));
    }
}

/**
 * @test TypedTensor.alias_view_constructor_extracts_column
 * @brief Extracts a single column from a 2x4 tensor.
 * Verifies dimensions, strides, and content correctness.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_extracts_column)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(0.f), static_cast<value_t>(1.f),
        static_cast<value_t>(2.f), static_cast<value_t>(3.f),
        static_cast<value_t>(4.f), static_cast<value_t>(5.f),
        static_cast<value_t>(6.f), static_cast<value_t>(7.f)
    };
    t = vals;

    Tensor<value_t> col_view(t, {0,1}, {2}, {t.m_strides[0]});

    EXPECT_EQ(col_view.m_dimensions, std::vector<uint64_t>({2}));
    EXPECT_EQ(col_view.m_strides, std::vector<uint64_t>({4}));

    Tensor<value_t> host({2}, MemoryLocation::HOST);
    copy_tensor_data(host, col_view);

    std::vector<value_t> out(2);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 2).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 5.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(1));
        EXPECT_EQ(out[1], static_cast<value_t>(5));
    }
}

/**
 * @test TypedTensor.alias_view_constructor_extracts_row
 * @brief Extracts a row from a 2x4 tensor.
 * Verifies dimensions, strides, and content correctness.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_extracts_row)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,4}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5),
        static_cast<value_t>(6), static_cast<value_t>(7)
    };
    t = vals;

    Tensor<value_t> row_view(t, {1,0}, {4}, {t.m_strides[1]});

    EXPECT_EQ(row_view.m_dimensions, std::vector<uint64_t>({4}));
    EXPECT_EQ(row_view.m_strides, std::vector<uint64_t>({1}));

    Tensor<value_t> host({4}, MemoryLocation::HOST);
    copy_tensor_data(host, row_view);

    std::vector<value_t> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    for (uint64_t i = 0; i < 4; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[i]),
                            static_cast<float>(4.f + i));
        } else {
            EXPECT_EQ(out[i], static_cast<value_t>(4 + i));
        }
    }
}

/**
 * @test TypedTensor.alias_view_constructor_autograd
 * @brief Alias (strided) view constructor must preserve/alias Autograd meta.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_autograd)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({3,4}, MemoryLocation::HOST);
    std::vector<value_t> vals(12);
    for (uint64_t i = 0; i < 12; ++i) vals[i] = static_cast<value_t>(i + 1);
    owner = vals;

    struct DummyEdge : public temper::FunctionEdge<value_t>
    {
        DummyEdge() : FunctionEdge<value_t>("alias") {}
        void forward() override {}
        void backward(const Tensor<value_t>&) override {}
        std::vector<std::shared_ptr<Tensor<value_t>>> inputs() const override
        { return {}; }
        std::shared_ptr<Tensor<value_t>> output() const override
        { return nullptr; }
    };

    owner.m_meta.requires_grad = true;
    owner.m_meta.grad = owner.m_p_data;
    owner.m_meta.fn = std::make_shared<DummyEdge>();

    std::vector<uint64_t> start = {1, 2};
    std::vector<uint64_t> dims = {2, 2};
    std::vector<uint64_t> strides = { owner.m_strides[0], owner.m_strides[1] };

    Tensor<value_t> aview(owner, start, dims, strides);

    uint64_t offset = start[0] * owner.m_strides[0] + start[1] * owner.m_strides[1];

    EXPECT_TRUE(aview.m_meta.requires_grad);
    ASSERT_NE(aview.m_meta.fn, nullptr);
    EXPECT_EQ(aview.m_meta.fn.get(), owner.m_meta.fn.get());
    ASSERT_NE(aview.m_meta.grad, nullptr);
    EXPECT_EQ(aview.m_meta.grad.get(), owner.m_meta.grad.get() + offset);

    // If owner has no grad buffer, alias view should also have nullptr grad.
    owner.m_meta.grad = nullptr;
    Tensor<value_t> aview2(owner, start, dims, strides);
    EXPECT_EQ(aview2.m_meta.grad, nullptr);
}

/**
 * @test TypedTensor.alias_view_constructor_extracts_patch
 * @brief Extracts a 2x2 patch from a 4x4 tensor.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_extracts_patch)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4,4}, MemoryLocation::HOST);
    std::vector<value_t> vals(16);
    for (uint64_t i = 0; i < 16; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    t = vals;

    Tensor<value_t> patch(t, {1,1}, {2,2},
                         {t.m_strides[0], t.m_strides[1]});

    EXPECT_EQ(patch.m_dimensions, (std::vector<uint64_t>{2,2}));
    EXPECT_EQ(patch.m_strides, (std::vector<uint64_t>{4,1}));

    Tensor<value_t> host({2,2}, MemoryLocation::HOST);
    copy_tensor_data(host, patch);

    std::vector<value_t> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 5.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 6.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2]), 9.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[3]), 10.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(5));
        EXPECT_EQ(out[1], static_cast<value_t>(6));
        EXPECT_EQ(out[2], static_cast<value_t>(9));
        EXPECT_EQ(out[3], static_cast<value_t>(10));
    }
}

/**
 * @test TypedTensor.alias_view_constructor_mutation_reflects
 * @brief Mutating a view updates the underlying owner tensor.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_mutation_reflects)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };
    t = vals;

    Tensor<value_t> col1(t, {0,1}, {2}, {t.m_strides[0]});

    col1[0] = static_cast<value_t>(100);
    col1[1] = static_cast<value_t>(200);

    Tensor<value_t> host({2,3}, MemoryLocation::HOST);
    copy_tensor_data(host, t);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 100.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[4]), 200.f);
    } else {
        EXPECT_EQ(out[1], static_cast<value_t>(100));
        EXPECT_EQ(out[4], static_cast<value_t>(200));
    }
}

/**
 * @test TypedTensor.alias_view_constructor_broadcasting
 * @brief Create a broadcasted view (stride 0) of a 1D tensor.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_broadcasting)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::HOST);
    t = std::vector<value_t>{
        static_cast<value_t>(10),
        static_cast<value_t>(20),
        static_cast<value_t>(30)
    };

    Tensor<value_t> broadcast_view(t, {1}, {4}, {0});

    EXPECT_EQ(broadcast_view.m_dimensions, (std::vector<uint64_t>{4}));
    EXPECT_EQ(broadcast_view.m_strides, (std::vector<uint64_t>{0}));

    Tensor<value_t> host({4}, MemoryLocation::HOST);
    copy_tensor_data(host, broadcast_view);

    std::vector<value_t> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    for (uint64_t i = 0; i < 4; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[i]), 20.f);
        } else {
            EXPECT_EQ(out[i], static_cast<value_t>(20));
        }
    }
}

/**
 * @test TypedTensor.alias_view_constructor_reshaping
 * @brief Create a 2x3 view from a 1D 6-element tensor using strides.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_reshaping)
{
    using value_t = TypeParam;
    Tensor<value_t> t({6}, MemoryLocation::HOST);
    t = std::vector<value_t>{
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };

    Tensor<value_t> reshaped(t, {0}, {2,3}, {3,1});

    EXPECT_EQ(reshaped.m_dimensions, (std::vector<uint64_t>{2,3}));
    EXPECT_EQ(reshaped.m_strides, (std::vector<uint64_t>{3,1}));

    Tensor<value_t> host({2,3}, MemoryLocation::HOST);
    copy_tensor_data(host, reshaped);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[i]),
                            static_cast<float>(i));
        } else {
            EXPECT_EQ(out[i], static_cast<value_t>(i));
        }
    }
}

/**
 * @test TypedTensor.alias_view_constructor_out_of_bounds
 * @brief Check exceptions on invalid start indices or dims.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_out_of_bounds)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(1),
                              static_cast<value_t>(2),
                              static_cast<value_t>(3),
                              static_cast<value_t>(4) };

    EXPECT_THROW(Tensor<value_t>(t,{2,0},{1},{2}),
        temper::bounds_error);
    EXPECT_THROW(Tensor<value_t>(t,{0,0},{3},{2}),
        temper::bounds_error);
    EXPECT_THROW(Tensor<value_t>(t,{0,0},{2},{2,1}),
        temper::validation_error);
}

/**
 * @test TypedTensor.alias_view_constructor_zero_rank
 * @brief Creating a view with zero rank should throw.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_zero_rank)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(1),
                              static_cast<value_t>(2),
                              static_cast<value_t>(3),
                              static_cast<value_t>(4) };

    EXPECT_THROW(Tensor<value_t>(t, {0,0}, {}, {}), temper::validation_error);
}

/**
 * @test TypedTensor.alias_view_constructor_zero_dim
 * @brief Creating a view with a zero dimension should throw.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_zero_dim)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(1),
                              static_cast<value_t>(2),
                              static_cast<value_t>(3),
                              static_cast<value_t>(4) };

    EXPECT_THROW(Tensor<value_t>(t, {0,0}, {2,0}, {1,1}),
                 temper::validation_error);
}

/**
 * @test TypedTensor.alias_view_constructor_stride_overflow
 * @brief Creating a view that triggers stride*(dim-1) overflow should throw.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_stride_overflow)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(1),
                              static_cast<value_t>(2),
                              static_cast<value_t>(3),
                              static_cast<value_t>(4) };

    std::vector<uint64_t> huge_stride =
        { std::numeric_limits<uint64_t>::max(), 1 };
    EXPECT_THROW(Tensor<value_t>(t, {0,0}, {2,2}, huge_stride),
        temper::bounds_error);
}

/**
 * @test TypedTensor.alias_view_constructor_uninitialized_owner
 * @brief Creating a view from an uninitialized tensor should throw.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_uninitialized_owner)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    EXPECT_THROW(Tensor<value_t>(t, {0,0}, {2,2}, {2,1}),
        temper::validation_error);
}

/**
 * @test TypedTensor.alias_view_constructor_multi_dim_broadcast
 * @brief Broadcasting a single row/column in 2D.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_multi_dim_broadcast)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(1),
                              static_cast<value_t>(2),
                              static_cast<value_t>(3),
                              static_cast<value_t>(4) };

    Tensor<value_t> broadcast_view(t, {0,0}, {3,2}, {0,1});

    EXPECT_EQ(broadcast_view.m_dimensions, (std::vector<uint64_t>{3,2}));
    EXPECT_EQ(broadcast_view.m_strides, (std::vector<uint64_t>{0,1}));

    Tensor<value_t> host({3,2}, MemoryLocation::HOST);
    copy_tensor_data(host, broadcast_view);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[i]),
                            static_cast<float>((i % 2) ? 2.f : 1.f));
        } else {
            EXPECT_EQ(out[i],
                      static_cast<value_t>((i % 2) ? 2 : 1));
        }
    }
}

/**
 * @test TypedTensor.alias_view_constructor_stride0_invalid
 * @brief Using stride 0 for non-broadcast dimension >1
 * should still work as broadcast.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_stride0_invalid)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(42),
                              static_cast<value_t>(99) };

    Tensor<value_t> view(t, {0}, {2}, {0});

    EXPECT_EQ(view.m_dimensions, (std::vector<uint64_t>{2}));
    EXPECT_EQ(view.m_strides, (std::vector<uint64_t>{0}));

    Tensor<value_t> host({2}, MemoryLocation::HOST);
    copy_tensor_data(host, view);

    std::vector<value_t> out(2);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 2).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 42.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 42.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(42));
        EXPECT_EQ(out[1], static_cast<value_t>(42));
    }
}

/**
 * @test TypedTensor.alias_view_constructor_from_view
 * @brief Verifies aliasing with custom multi-dimensional strides by
 * simulating a transpose.
 */
TYPED_TEST(TypedTensor, alias_view_constructor_from_view)
{
    using value_t = TypeParam;
    // Owner tensor 2x3
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(0.f), static_cast<value_t>(1.f),
        static_cast<value_t>(2.f), static_cast<value_t>(3.f),
        static_cast<value_t>(4.f), static_cast<value_t>(5.f)
    };
    t = vals;

    Tensor<value_t> transposed_alias(t, {0,0}, {3,2}, {1,3});

    EXPECT_EQ(transposed_alias.m_dimensions, std::vector<uint64_t>({3,2}));
    EXPECT_EQ(transposed_alias.m_strides, std::vector<uint64_t>({1,3}));

    Tensor<value_t> host({3,2}, MemoryLocation::HOST);
    copy_tensor_data(host, transposed_alias);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    // Expected transpose:
    // [ [0,3], [1,4], [2,5] ]
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0*2 + 0]), 0.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[0*2 + 1]), 3.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1*2 + 0]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1*2 + 1]), 4.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2*2 + 0]), 2.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2*2 + 1]), 5.f);
    } else {
        EXPECT_EQ(out[0*2 + 0], static_cast<value_t>(0));
        EXPECT_EQ(out[0*2 + 1], static_cast<value_t>(3));
        EXPECT_EQ(out[1*2 + 0], static_cast<value_t>(1));
        EXPECT_EQ(out[1*2 + 1], static_cast<value_t>(4));
        EXPECT_EQ(out[2*2 + 0], static_cast<value_t>(2));
        EXPECT_EQ(out[2*2 + 1], static_cast<value_t>(5));
    }
}

/**
 * @test TypedTensor.operator_equals_copy_assignment
 * @brief Tests copy assignment operator.
 */
TYPED_TEST(TypedTensor, operator_equals_copy_assignment)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> values = {
        static_cast<value_t>(9.0),
        static_cast<value_t>(10.0),
        static_cast<value_t>(11.0),
        static_cast<value_t>(12.0)
    };
    t1 = values;

    Tensor<value_t> t2;
    t2 = t1;

    EXPECT_EQ(t2.m_mem_loc, t1.m_mem_loc);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), t2.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        static_cast<float>(9.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        static_cast<float>(10.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        static_cast<float>(11.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]),
                        static_cast<float>(12.0f));
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(9));
        EXPECT_EQ(host[1], static_cast<value_t>(10));
        EXPECT_EQ(host[2], static_cast<value_t>(11));
        EXPECT_EQ(host[3], static_cast<value_t>(12));
    }
}

/**
 * @test TypedTensor.operator_equals_copy_from_default
 * @brief Assigning from a default-constructed tensor yields an empty owner.
 */
TYPED_TEST(TypedTensor, operator_equals_copy_from_default)
{
    using value_t = TypeParam;
    Tensor<value_t> dst({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> v = {
        static_cast<value_t>(7.0),
        static_cast<value_t>(8.0),
        static_cast<value_t>(9.0),
        static_cast<value_t>(10.0)
    };
    dst = v;

    Tensor<value_t> src;

    dst = src;

    EXPECT_TRUE(dst.m_dimensions.empty());
    EXPECT_TRUE(dst.m_strides.empty());
    EXPECT_TRUE(dst.m_own_data);
    EXPECT_EQ(dst.m_p_data, nullptr);
}

/**
 * @test TypedTensor.operator_equals_copy_self_assignment
 * @brief Self-assignment must be safe and preserve data.
 */
TYPED_TEST(TypedTensor, operator_equals_copy_self_assignment)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::DEVICE);
    std::vector<value_t> v = {
        static_cast<value_t>(1.0),
        static_cast<value_t>(2.0),
        static_cast<value_t>(3.0),
        static_cast<value_t>(4.0)
    };
    t = v;

    Tensor<value_t> tmp = t;
    t = tmp;

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), t.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        static_cast<float>(1.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        static_cast<float>(2.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        static_cast<float>(3.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]),
                        static_cast<float>(4.0f));
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(2));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(4));
    }
}

/**
 * @test TypedTensor.operator_equals_copy_from_view
 * @brief Assigning from a view yields a non-owning alias.
 */
TYPED_TEST(TypedTensor, operator_equals_copy_from_view)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(10), static_cast<value_t>(11),
        static_cast<value_t>(12), static_cast<value_t>(20),
        static_cast<value_t>(21), static_cast<value_t>(22)
    };
    owner = vals;

    Tensor<value_t> view(owner, {1,1}, {1,2});
    ASSERT_FALSE(view.m_own_data);

    Tensor<value_t> dst;
    dst = view;

    EXPECT_FALSE(dst.m_own_data);
    EXPECT_EQ(dst.m_dimensions, view.m_dimensions);
    EXPECT_EQ(dst.m_strides, view.m_strides);
    EXPECT_EQ(dst.m_p_data.get(), view.m_p_data.get());

    std::vector<value_t> out(2);
    g_sycl_queue.memcpy(out.data(), dst.m_p_data.get(),
                       sizeof(value_t) * 2).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]),
                        static_cast<float>(vals[4]));
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]),
                        static_cast<float>(vals[5]));
    } else {
        EXPECT_EQ(out[0], vals[4]);
        EXPECT_EQ(out[1], vals[5]);
    }
}

/**
 * @test TypedTensor.operator_equals_copy_autograd
 * @brief Copy-assignment should clear meta when assigning from an owning
 * tensor, and copy meta when assigning from a view.
 */
TYPED_TEST(TypedTensor, operator_equals_copy_autograd)
{
    using value_t = TypeParam;
    // Source owning tensor with autograd meta
    Tensor<value_t> src({2,2}, MemoryLocation::HOST);
    src = std::vector<value_t>{1,2,3,4};
    src.m_meta.requires_grad = true;
    src.m_meta.grad = src.m_p_data;

    Tensor<value_t> dst;
    dst = src; // dst should be owning and have cleared meta

    EXPECT_FALSE(dst.m_meta.requires_grad);
    EXPECT_EQ(dst.m_meta.grad, nullptr);

    // Now assignment from a view should copy meta
    src.m_meta.requires_grad = true;
    src.m_meta.grad = src.m_p_data;

    Tensor<value_t> view(src, std::vector<uint64_t>{0,0}, std::vector<uint64_t>{2,1});
    Tensor<value_t> dst2;
    dst2 = view;

    EXPECT_TRUE(dst2.m_meta.requires_grad);
    ASSERT_NE(dst2.m_meta.grad, nullptr);
}

/**
 * @test TypedTensor.operator_equals_move
 * @brief Basic move-assignment: resources transferred, source emptied.
 */
TYPED_TEST(TypedTensor, operator_equals_move)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> values = {
        static_cast<value_t>(13.0),
        static_cast<value_t>(14.0),
        static_cast<value_t>(15.0),
        static_cast<value_t>(16.0)
    };
    src = values;

    value_t* original_ptr = src.m_p_data.get();
    MemoryLocation original_loc = src.m_mem_loc;

    Tensor<value_t> dst;
    dst = std::move(src);

    EXPECT_EQ(dst.m_p_data.get(), original_ptr);
    EXPECT_EQ(dst.m_mem_loc, original_loc);

    EXPECT_EQ(src.m_p_data.get(), nullptr);

    std::vector<value_t> host(4);
    std::memcpy(host.data(), dst.m_p_data.get(),
                sizeof(value_t) * 4);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        static_cast<float>(13.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        static_cast<float>(14.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        static_cast<float>(15.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]),
                        static_cast<float>(16.0f));
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(13));
        EXPECT_EQ(host[1], static_cast<value_t>(14));
        EXPECT_EQ(host[2], static_cast<value_t>(15));
        EXPECT_EQ(host[3], static_cast<value_t>(16));
    }
}

/**
 * @test TypedTensor.operator_equals_move_from_default
 * @brief Moving from a default tensor makes destination empty.
 */
TYPED_TEST(TypedTensor, operator_equals_move_from_default)
{
    using value_t = TypeParam;
    Tensor<value_t> dst({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(7.0), static_cast<value_t>(8.0),
        static_cast<value_t>(9.0), static_cast<value_t>(10.0)
    };
    dst = vals;

    Tensor<value_t> src;

    dst = std::move(src);

    EXPECT_TRUE(dst.m_dimensions.empty());
    EXPECT_TRUE(dst.m_strides.empty());
    EXPECT_TRUE(dst.m_own_data);
    EXPECT_EQ(dst.m_p_data, nullptr);

    EXPECT_TRUE(src.m_dimensions.empty());
    EXPECT_TRUE(src.m_own_data);
    EXPECT_EQ(src.m_p_data, nullptr);
}

/**
 * @test TypedTensor.operator_equals_move_self_assignment
 * @brief Self move-assignment must be safe and preserve contents.
 */
TYPED_TEST(TypedTensor, operator_equals_move_self_assignment)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    std::vector<value_t> v = {
        static_cast<value_t>(1.0), static_cast<value_t>(2.0),
        static_cast<value_t>(3.0), static_cast<value_t>(4.0)
    };
    t = v;

    t = std::move(t);

    std::vector<value_t> host(4);
    std::memcpy(host.data(), t.m_p_data.get(), sizeof(value_t) * 4);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        static_cast<float>(1.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        static_cast<float>(2.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        static_cast<float>(3.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]),
                        static_cast<float>(4.0f));
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(2));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(4));
    }
}

/**
 * @test TypedTensor.operator_equals_move_from_view
 * @brief Moving a non-owning view transfers aliasing shared_ptr.
 */
TYPED_TEST(TypedTensor, operator_equals_move_from_view)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(10.0), static_cast<value_t>(11.0),
        static_cast<value_t>(12.0), static_cast<value_t>(20.0),
        static_cast<value_t>(21.0), static_cast<value_t>(22.0)
    };
    owner = vals;

    Tensor<value_t> view(owner, {1,1}, {1,2});
    ASSERT_FALSE(view.m_own_data);

    value_t* view_ptr = view.m_p_data.get();

    Tensor<value_t> dst;
    dst = std::move(view);

    EXPECT_FALSE(dst.m_own_data);
    EXPECT_EQ(dst.m_p_data.get(), view_ptr);

    EXPECT_EQ(view.m_p_data.get(), nullptr);
    EXPECT_TRUE(view.m_own_data);

    std::vector<value_t> out(2);
    std::memcpy(out.data(), dst.m_p_data.get(), sizeof(value_t) * 2);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]),
                        static_cast<float>(vals[4]));
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]),
                        static_cast<float>(vals[5]));
    } else {
        EXPECT_EQ(out[0], vals[4]);
        EXPECT_EQ(out[1], vals[5]);
    }
}

/**
 * @test TypedTensor.operator_equals_move_autograd
 * @brief Move-assignment should transfer Autograd meta and clear the source.
 */
TYPED_TEST(TypedTensor, operator_equals_move_autograd)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2,2}, MemoryLocation::HOST);
    src = std::vector<value_t>{1,2,3,4};
    src.m_meta.requires_grad = true;
    src.m_meta.grad = src.m_p_data;

    Tensor<value_t> dst;
    dst = std::move(src);

    EXPECT_TRUE(dst.m_meta.requires_grad);
    ASSERT_NE(dst.m_meta.grad, nullptr);
    EXPECT_FALSE(src.m_meta.requires_grad);
    EXPECT_EQ(src.m_meta.grad, nullptr);
}

/**
 * @test TypedTensor.operator_equals_vector_assignment
 * @brief Tests assignment from flat std::vector.
 */
TYPED_TEST(TypedTensor, operator_equals_vector_assignment)
{
    using value_t = TypeParam;
    std::vector<value_t> values = {
        static_cast<value_t>(3.3),
        static_cast<value_t>(3.4),
        static_cast<value_t>(3.5),
        static_cast<value_t>(3.6)
    };

    Tensor<value_t> t({2, 2});
    t = values;

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), t.m_p_data.get(),
                       sizeof(value_t) * 4).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        static_cast<float>(3.3f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        static_cast<float>(3.4f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        static_cast<float>(3.5f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]),
                        static_cast<float>(3.6f));
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(3));
        EXPECT_EQ(host[1], static_cast<value_t>(3));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(3));
    }
}

/**
 * @test TypedTensor.operator_equals_vector_size_mismatch_throws
 * @brief Throws if assigning flat vector with incorrect size.
 */
TYPED_TEST(TypedTensor, operator_equals_vector_size_mismatch_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2});
    std::vector<value_t> values = {
        static_cast<value_t>(1.0), static_cast<value_t>(2.0)
    };

    EXPECT_THROW({
        t = values;
    }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_equals_vector_assign_to_default_throws
 * @brief Assigning a vector to a default tensor must throw.
 */
TYPED_TEST(TypedTensor, operator_equals_vector_assign_to_default_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t_default;
    std::vector<value_t> values = { static_cast<value_t>(1.0) };

    EXPECT_THROW({
        t_default = values;
    }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_equals_vector_assignment_to_view_column
 * @brief Assigning a std::vector to a column view writes into owner.
 */
TYPED_TEST(TypedTensor, operator_equals_vector_assignment_to_view_column)
{
    using value_t = TypeParam;
    // Owner 2x3
    Tensor<value_t> owner({2,3}, MemoryLocation::HOST);
    std::vector<value_t> base_vals = {
        static_cast<value_t>(0.f), static_cast<value_t>(1.f),
        static_cast<value_t>(2.f), static_cast<value_t>(3.f),
        static_cast<value_t>(4.f), static_cast<value_t>(5.f)
    };
    owner = base_vals;

    Tensor<value_t> col_view(owner, {0,1}, {2},
                             { owner.m_strides[0] });

    std::vector<value_t> new_col = {
        static_cast<value_t>(100.f), static_cast<value_t>(200.f)
    };
    col_view = new_col;

    EXPECT_FALSE(col_view.m_own_data);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), owner.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]),
                        static_cast<float>(100.f));
        EXPECT_FLOAT_EQ(static_cast<float>(out[4]),
                        static_cast<float>(200.f));
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]),
                        static_cast<float>(0.f));
        EXPECT_FLOAT_EQ(static_cast<float>(out[2]),
                        static_cast<float>(2.f));
    } else {
        EXPECT_EQ(out[1], static_cast<value_t>(100));
        EXPECT_EQ(out[4], static_cast<value_t>(200));
        EXPECT_EQ(out[0], static_cast<value_t>(0));
        EXPECT_EQ(out[2], static_cast<value_t>(2));
    }
}

/**
 * @test TypedTensor.operator_equals_vector_assignment_to_view_patch
 * @brief Assigning a std::vector to a strided patch view writes correctly.
 */
TYPED_TEST(TypedTensor, operator_equals_vector_assignment_to_view_patch)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({3,3}, MemoryLocation::HOST);
    owner = std::vector<value_t>(9, static_cast<value_t>(0.f));

    Tensor<value_t> patch(owner, {1,1}, {2,2},
        { owner.m_strides[0], owner.m_strides[1] });

    std::vector<value_t> vals = {
        static_cast<value_t>(1.f), static_cast<value_t>(2.f),
        static_cast<value_t>(3.f), static_cast<value_t>(4.f)
    };
    patch = vals;

    std::vector<value_t> out(9);
    g_sycl_queue.memcpy(out.data(), owner.m_p_data.get(),
                       sizeof(value_t) * 9).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[1 * 3 + 1]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1 * 3 + 2]), 2.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2 * 3 + 1]), 3.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2 * 3 + 2]), 4.f);

        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 0.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[3]), 0.f);
    } else {
        EXPECT_EQ(out[1 * 3 + 1], static_cast<value_t>(1));
        EXPECT_EQ(out[1 * 3 + 2], static_cast<value_t>(2));
        EXPECT_EQ(out[2 * 3 + 1], static_cast<value_t>(3));
        EXPECT_EQ(out[2 * 3 + 2], static_cast<value_t>(4));

        EXPECT_EQ(out[0], static_cast<value_t>(0));
        EXPECT_EQ(out[3], static_cast<value_t>(0));
    }
}

/**
 * @test TypedTensor.operator_equals_vector_assignment_to_weird_strides
 * @brief Assigns to a view with irregular strides and verifies placement.
 */
TYPED_TEST(TypedTensor, operator_equals_vector_assignment_to_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({10,10}, MemoryLocation::HOST);
    owner = std::vector<value_t>(100, static_cast<value_t>(0.f));

    Tensor<value_t> view(owner, {1,1}, {2,3}, {3,14});

    std::vector<value_t> vals = {
        static_cast<value_t>(1.f), static_cast<value_t>(2.f),
        static_cast<value_t>(3.f), static_cast<value_t>(4.f),
        static_cast<value_t>(5.f), static_cast<value_t>(6.f)
    };
    view = vals;

    std::vector<value_t> host(100);
    g_sycl_queue.memcpy(host.data(), owner.m_p_data.get(),
                       sizeof(value_t) * 100).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[1*10+1]),
                        static_cast<float>(1.f));
        EXPECT_FLOAT_EQ(static_cast<float>(host[1*10+15]),
                        static_cast<float>(2.f));
    } else {
        EXPECT_EQ(host[1*10+1], static_cast<value_t>(1));
        EXPECT_EQ(host[1*10+15], static_cast<value_t>(2));
    }
}

/**
 * @test TypedTensor.operator_equals_vector_assignment_view_size_mismatch_throws
 * @brief Assigning a wrong-size vector to a view should throw.
 */
TYPED_TEST(TypedTensor, operator_equals_vector_assignment_view_size_mismatch_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2,2}, MemoryLocation::HOST);
    owner = std::vector<value_t>{
        static_cast<value_t>(1.f), static_cast<value_t>(2.f),
        static_cast<value_t>(3.f), static_cast<value_t>(4.f)
    };

    Tensor<value_t> col(owner, {0,1}, {2}, { owner.m_strides[0] });

    std::vector<value_t> wrong = { static_cast<value_t>(9.f) };

    EXPECT_THROW({
        col = wrong;
    }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_equals_scalar_assignment_and_conversion
 * @brief Tests operator=(value_t) and operator value_t() on 1-element.
 */
TYPED_TEST(TypedTensor, operator_equals_scalar_assignment_and_conversion)
{
    using value_t = TypeParam;
    Tensor<value_t> s({1}, MemoryLocation::HOST);
    s = static_cast<value_t>(5.5);
    value_t sval = static_cast<value_t>(s);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(sval), 5.5f);
    } else {
        EXPECT_EQ(sval, static_cast<value_t>(5));
    }

    Tensor<value_t> t({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0), static_cast<value_t>(2.0),
        static_cast<value_t>(3.0), static_cast<value_t>(4.0)
    };
    t = vals;

    t[1][1] = static_cast<value_t>(99.25);
    value_t read_back = static_cast<value_t>(t[1][1]);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(read_back), 99.25f);
    } else {
        EXPECT_EQ(read_back, static_cast<value_t>(99));
    }
}

/**
 * @test TypedTensor.operator_equals_scalar_assignment_to_default_constructed_tensor
 * @brief Assign scalar to default tensor initializes scalar shape {1}.
 */
TYPED_TEST(TypedTensor,
           operator_equals_scalar_assignment_to_default_constructed_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    t = static_cast<value_t>(42.5);

    ASSERT_EQ(t.m_dimensions.size(), 1);
    EXPECT_EQ(t.m_dimensions[0], 1u);

    value_t val = static_cast<value_t>(t);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(val), 42.5f);
    } else {
        EXPECT_EQ(val, static_cast<value_t>(42));
    }

    Tensor<value_t> big({2, 2});
    EXPECT_THROW(big = static_cast<value_t>(1.0),
                 temper::validation_error);
}

/**
 * @test TypedTensor.operator_equals_scalar_assignment_wrong_size_throws
 * @brief Assigning a scalar to a non-1-element tensor must throw.
 */
TYPED_TEST(TypedTensor, operator_equals_scalar_assignment_wrong_size_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2}, MemoryLocation::HOST);
    EXPECT_THROW({
        t = static_cast<value_t>(3.14);
    }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_brackets_index_chain_assign_and_read
 * @brief Chained operator[] returns views and allows assignment/read.
 */
TYPED_TEST(TypedTensor, operator_brackets_index_chain_assign_and_read)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(2 * 3 * 4);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    t = vals;

    t[1][2][3] = static_cast<value_t>(420.0);

    value_t v = static_cast<value_t>(t[1][2][3]);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(v), 420.0f);
    } else {
        EXPECT_EQ(v, static_cast<value_t>(420));
    }

    value_t host_val = static_cast<value_t>(0);
    uint64_t offset =
        1 * t.m_strides[0] + 2 * t.m_strides[1] + 3 * t.m_strides[2];
    g_sycl_queue.memcpy(&host_val, t.m_p_data.get() + offset,
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host_val), 420.0f);
    } else {
        EXPECT_EQ(host_val, static_cast<value_t>(420));
    }
}

/**
 * @test TypedTensor.operator_brackets_index_chain_const_access
 * @brief Tests that const Tensor supports chained operator[] reading.
 */
TYPED_TEST(TypedTensor, operator_brackets_index_chain_const_access)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::HOST);
    std::vector<value_t> vals(8);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 10);
    }
    t = vals;

    const Tensor<value_t>& ct = t;

    value_t a = static_cast<value_t>(ct[1][1][1]);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(a),
                        static_cast<float>(vals[1 * 4 + 1 * 2 + 1]));
    } else {
        EXPECT_EQ(a, vals[1 * 4 + 1 * 2 + 1]);
    }
}

/**
 * @test TypedTensor.operator_brackets_autograd
 * @brief operator[] must create a view that inherits fn and requires_grad
 * and aliases the grad buffer at the correct offset.
 */
TYPED_TEST(TypedTensor, operator_brackets_autograd)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({3,2}, MemoryLocation::HOST);
    std::vector<value_t> vals(6);
    for (uint64_t i = 0; i < 6; ++i) vals[i] = static_cast<value_t>(i+1);
    owner = vals;

    struct DummyEdge : public temper::FunctionEdge<value_t>
    {
        DummyEdge() : FunctionEdge<value_t>("idx") {}
        void forward() override {}
        void backward(const Tensor<value_t>&) override {}
        std::vector<std::shared_ptr<Tensor<value_t>>> inputs() const override
        { return {}; }
        std::shared_ptr<Tensor<value_t>> output() const override
        { return nullptr; }
    };

    owner.m_meta.requires_grad = true;
    owner.m_meta.grad = owner.m_p_data;
    owner.m_meta.fn = std::make_shared<DummyEdge>();

    Tensor<value_t> row = owner[1];
    EXPECT_TRUE(row.m_meta.requires_grad);
    ASSERT_NE(row.m_meta.fn, nullptr);
    EXPECT_EQ(row.m_meta.fn.get(), owner.m_meta.fn.get());

    // offset should be index * stride0
    uint64_t offset = 1 * owner.m_strides[0];
    ASSERT_NE(row.m_meta.grad, nullptr);
    EXPECT_EQ(row.m_meta.grad.get(), owner.m_meta.grad.get() + offset);
}

/**
 * @test TypedTensor.operator_brackets_index_out_of_bounds_throws
 * @brief operator[] should throw when index is out of range.
 */
TYPED_TEST(TypedTensor, operator_brackets_index_out_of_bounds_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::HOST);

    EXPECT_THROW({ (void)t[2]; }, temper::bounds_error);
    EXPECT_THROW({ (void)t[1][2]; }, temper::bounds_error);
    EXPECT_THROW({ (void)t[1][1][2]; }, temper::bounds_error);
}

/**
 * @test TypedTensor.operator_brackets_view_constructor_middle_chunk_write_and_read
 * @brief Take a chunk, write via chained operator[] and verify positions.
 */
TYPED_TEST(TypedTensor,
           operator_brackets_view_constructor_middle_chunk_write_and_read)
{
    using value_t = TypeParam;
    Tensor<value_t> img({3, 6, 7}, MemoryLocation::HOST);

    std::vector<value_t> vals(3 * 6 * 7);
    for (uint64_t c = 0; c < 3; ++c)
    {
        for (uint64_t h = 0; h < 6; ++h)
        {
            for (uint64_t w = 0; w < 7; ++w)
            {
                uint64_t idx = c * (6 * 7) + h * 7 + w;
                vals[idx] = static_cast<value_t>(c * 10000 + h * 100 + w);
            }
        }
    }
    img = vals;

    std::vector<uint64_t> start = { 1, 2, 3 };
    std::vector<uint64_t> view_shape = { 3, 2 };
    Tensor<value_t> chunk(img, start, view_shape);

    EXPECT_EQ(chunk.m_dimensions, std::vector<uint64_t>({3, 2}));

    chunk[0][0] = static_cast<value_t>(9999.5);
    chunk[2][1] = static_cast<value_t>(8888.25);

    value_t a = static_cast<value_t>(chunk[0][0]);
    value_t b = static_cast<value_t>(chunk[2][1]);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(a), 9999.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(b), 8888.25f);
    } else {
        EXPECT_EQ(a, static_cast<value_t>(9999));
        EXPECT_EQ(b, static_cast<value_t>(8888));
    }

    uint64_t off_a =
        start[0] * img.m_strides[0] + (start[1] + 0) * img.m_strides[1] +
        (start[2] + 0) * img.m_strides[2];

    uint64_t off_b =
        start[0] * img.m_strides[0] + (start[1] + 2) * img.m_strides[1] +
        (start[2] + 1) * img.m_strides[2];

    value_t host_a = static_cast<value_t>(0);
    value_t host_b = static_cast<value_t>(0);
    g_sycl_queue.memcpy(&host_a, img.m_p_data.get() + off_a,
                       sizeof(value_t)).wait();

    g_sycl_queue.memcpy(&host_b, img.m_p_data.get() + off_b,
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host_a), 9999.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(host_b), 8888.25f);
    } else {
        EXPECT_EQ(host_a, static_cast<value_t>(9999));
        EXPECT_EQ(host_b, static_cast<value_t>(8888));
    }

    value_t cval = static_cast<value_t>(chunk[1][0]);

    uint64_t off_c =
        start[0] * img.m_strides[0] + (start[1] + 1) * img.m_strides[1] +
        (start[2] + 0) * img.m_strides[2];

    value_t host_c = static_cast<value_t>(0);
    g_sycl_queue.memcpy(&host_c, img.m_p_data.get() + off_c,
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(cval),
                        static_cast<float>(host_c));
    } else {
        EXPECT_EQ(cval, host_c);
    }
}

/**
 * @test TypedTensor.operator_float_valid_scalar
 * @brief Tests implicit conversion to scalar for a 1-element tensor.
 */
TYPED_TEST(TypedTensor, operator_float_valid_scalar)
{
    using value_t = TypeParam;
    Tensor<value_t> t({1}, MemoryLocation::HOST);
    value_t val = static_cast<value_t>(42.5);
    t = val;  // uses scalar assignment

    value_t converted = static_cast<value_t>(t);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(converted), 42.5f);
    } else {
        EXPECT_EQ(converted, static_cast<value_t>(42));
    }
}

/**
 * @test TypedTensor.operator_value_throws_no_dimensions
 * @brief Converting a moved-from tensor throws (no dimensions).
 */
TYPED_TEST(TypedTensor, operator_value_throws_no_dimensions)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({1}, MemoryLocation::HOST);
    t1 = static_cast<value_t>(3.14);

    Tensor<value_t> t2 = std::move(t1);
    EXPECT_THROW({
        value_t converted = static_cast<value_t>(t1);
        (void)converted;
    }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_value_throws_multiple_elements_rank1
 * @brief Conversion throws for rank-1 tensor with size > 1.
 */
TYPED_TEST(TypedTensor, operator_value_throws_multiple_elements_rank1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0), static_cast<value_t>(2.0),
        static_cast<value_t>(3.0)
    };
    t = vals;

    EXPECT_THROW({
        value_t converted = static_cast<value_t>(t);
        (void)converted;
    }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_value_throws_multi_dimensional
 * @brief Conversion throws for rank > 1 tensor.
 */
TYPED_TEST(TypedTensor, operator_value_throws_multi_dimensional)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };
    t = vals;

    EXPECT_THROW({
        value_t converted = static_cast<value_t>(t);
        (void)converted;
    }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_addition
 * @brief Verifies element-wise addition on device memory.
 *
 * Creates two 2x3 device tensors A and B with known values, computes
 * C = A + B, copies the result back to host and checks every element
 * equals avals[i] + bvals[i].
 */
TYPED_TEST(TypedTensor, operator_addition)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2,3}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f),
        static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f),
        static_cast<value_t>(6.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(6.0f),
        static_cast<value_t>(5.0f),
        static_cast<value_t>(4.0f),
        static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(1.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> C = A + B;

    const uint64_t total = 6;
    std::vector<value_t> ch(total);
    g_sycl_queue.memcpy(ch.data(), C.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(ch[i]),
                            static_cast<float>(avals[i] + bvals[i]));
        } else {
            EXPECT_EQ(ch[i], avals[i] + bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_addition_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for addition.
 */
TYPED_TEST(TypedTensor, operator_addition_broadcasting_1d_to_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };

    Tensor<value_t> R = A + B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(11.0f), static_cast<value_t>(22.0f),
        static_cast<value_t>(33.0f), static_cast<value_t>(41.0f),
        static_cast<value_t>(52.0f), static_cast<value_t>(63.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_addition_broadcasting_scalar
 * @brief Verifies addition broadcasting with a scalar operand.
 */
TYPED_TEST(TypedTensor, operator_addition_broadcasting_scalar)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::HOST);
    Tensor<value_t> B({1}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(5.0f) };

    Tensor<value_t> R = A + B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(6.0f), static_cast<value_t>(7.0f),
        static_cast<value_t>(8.0f), static_cast<value_t>(9.0f),
        static_cast<value_t>(10.0f), static_cast<value_t>(11.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_addition_with_views
 * @brief Verifies element-wise addition between a row view and a tensor.
 */
TYPED_TEST(TypedTensor, operator_addition_with_views)
{
    using value_t = TypeParam;
    Tensor<value_t> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };

    Tensor<value_t> row0 = T[0];
    Tensor<value_t> addend({3}, MemoryLocation::DEVICE);
    addend = std::vector<value_t>{
        static_cast<value_t>(10.0f),
        static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f)
    };

    Tensor<value_t> R = row0 + addend;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(11.0f),
        static_cast<value_t>(22.0f),
        static_cast<value_t>(33.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }

    // Sanity: original parent should remain unchanged.
    std::vector<value_t> parent_row(total);
    g_sycl_queue.memcpy(parent_row.data(), T.m_p_data.get(),
                       sizeof(value_t) * total).wait();
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(parent_row[0]),
                        static_cast<float>(1.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(parent_row[1]),
                        static_cast<float>(2.0f));
        EXPECT_FLOAT_EQ(static_cast<float>(parent_row[2]),
                        static_cast<float>(3.0f));
    } else {
        EXPECT_EQ(parent_row[0], static_cast<value_t>(1));
        EXPECT_EQ(parent_row[1], static_cast<value_t>(2));
        EXPECT_EQ(parent_row[2], static_cast<value_t>(3));
    }
}

/**
 * @test TypedTensor.operator_addition_incompatible_shapes
 * @brief Addition throws when operand shapes are incompatible.
 */
TYPED_TEST(TypedTensor, operator_addition_incompatible_shapes)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::HOST);
    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };

    EXPECT_THROW({ Tensor<value_t> R = A + B; }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_addition_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE.
 */
TYPED_TEST(TypedTensor, operator_addition_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A + B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TypedTensor.operator_addition_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST.
 */
TYPED_TEST(TypedTensor, operator_addition_both_host_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A + B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TypedTensor.operator_addition_nan_inputs_throws
 * @brief Addition detects NaN inputs and triggers a temper::nan_error,
 * as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_addition_nan_inputs_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2}, MemoryLocation::DEVICE);

    const float nanf = std::numeric_limits<float>::quiet_NaN();
    A = std::vector<value_t>{
        static_cast<value_t>(1.0f),
        static_cast<value_t>(nanf)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };

    if constexpr (!std::is_floating_point_v<value_t>)
    {
        // Do nothing.
    } else {
        EXPECT_THROW({ Tensor<value_t> R = A + B; }, temper::nan_error);
    }
}

/**
 * @test TypedTensor.operator_addition_non_finite_result_throws
 * @brief Addition that overflows to Inf should trigger a
 * temper::nonfinite_error, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_addition_non_finite_result_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({1}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    const float large = std::numeric_limits<float>::max();
    A = std::vector<value_t>{ static_cast<value_t>(large) };
    B = std::vector<value_t>{ static_cast<value_t>(large) };

    if constexpr (!std::is_floating_point_v<value_t>) {
        // Do nothing.
    } else {
        EXPECT_THROW({ Tensor<value_t> R = A + B; }, temper::nonfinite_error);
    }
}

/**
 * @test TypedTensor.operator_addition_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} + B{3,1}.
 */
TYPED_TEST(TypedTensor, operator_addition_broadcasting_complex_alignment)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3,4}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3,1}, MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<value_t> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<value_t>(i);
    }
    std::vector<value_t> bvals = {
        static_cast<value_t>(10.0f),
        static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> R = A + B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            value_t offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] + offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[idx]),
                            static_cast<float>(expected[idx]));
        } else {
            EXPECT_EQ(rh[idx], expected[idx]);
        }
    }
}

/**
 * @test TypedTensor.operator_addition_alias_view_noncontiguous
 * @brief Element-wise addition where the right operand is a non-contig
 * 1D alias view (stride 2).
 */
TYPED_TEST(TypedTensor, operator_addition_alias_view_noncontiguous)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({6}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> v(B, {0}, {3}, {2});

    Tensor<value_t> R = A + v;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(11.0f),
        static_cast<value_t>(32.0f),
        static_cast<value_t>(53.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }

    std::vector<value_t> b_host(6);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(),
                       sizeof(value_t) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(b_host[i]),
                            static_cast<float>(bvals[i]));
        } else {
            EXPECT_EQ(b_host[i], bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_addition_alias_view_broadcast
 * @brief Addition with a broadcasted alias view (stride 0).
 */
TYPED_TEST(TypedTensor, operator_addition_alias_view_broadcast)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(5.0f) };

    Tensor<value_t> vb(B, {0}, {3}, {0});

    Tensor<value_t> R = A + vb;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(6.0f), static_cast<value_t>(7.0f),
        static_cast<value_t>(8.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_addition_alias_view_weird_strides
 * @brief Addition of a dense small tensor with a 2D alias view with odd
 * strides.
 */
TYPED_TEST(TypedTensor, operator_addition_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    owner = vals;

    Tensor<value_t> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<value_t> add({3,4}, MemoryLocation::DEVICE);

    std::vector<value_t> add_vals(12, static_cast<value_t>(3.0f));
    add = add_vals;

    Tensor<value_t> R = view + add;

    Tensor<value_t> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<value_t> rh(12);
    g_sycl_queue.memcpy(rh.data(), hostR.m_p_data.get(),
                       sizeof(value_t) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(rh[k]),
                                static_cast<float>(vals[flat] + 3.0f));
            } else {
                EXPECT_EQ(rh[k], vals[flat] + static_cast<value_t>(3));
            }
        }
    }
}

/**
 * @test TypedTensor.operator_subtraction
 * @brief Verifies element-wise subtraction on device memory.
 */
TYPED_TEST(TypedTensor, operator_subtraction)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2,3}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(6.0f), static_cast<value_t>(5.0f),
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> D = A - B;

    const uint64_t total = 6;
    std::vector<value_t> dh(total);
    g_sycl_queue.memcpy(dh.data(), D.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(dh[i]),
                            static_cast<float>(avals[i] - bvals[i]));
        } else {
            EXPECT_EQ(dh[i], avals[i] - bvals[i]);
        }
    }
}
/**
 * @test TypedTensor.operator_subtraction_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for subtraction.
 */
TYPED_TEST(TypedTensor, operator_subtraction_broadcasting_1d_to_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };

    Tensor<value_t> R = A - B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(9.0f), static_cast<value_t>(18.0f),
        static_cast<value_t>(27.0f), static_cast<value_t>(39.0f),
        static_cast<value_t>(48.0f), static_cast<value_t>(57.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_subtraction_broadcasting_scalar
 * @brief Verifies subtraction broadcasting with a scalar operand.
 */
TYPED_TEST(TypedTensor, operator_subtraction_broadcasting_scalar)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::HOST);
    Tensor<value_t> B({1}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(5.0f) };

    Tensor<value_t> R = A - B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(-4), static_cast<value_t>(-3),
        static_cast<value_t>(-2), static_cast<value_t>(-1),
        static_cast<value_t>(0), static_cast<value_t>(1)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_subtraction_with_views
 * @brief Verifies element-wise subtraction between a row view and a tensor.
 */
TYPED_TEST(TypedTensor, operator_subtraction_with_views)
{
    using value_t = TypeParam;
    Tensor<value_t> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };

    Tensor<value_t> row0 = T[0];
    Tensor<value_t> subtrahend({3}, MemoryLocation::DEVICE);
    subtrahend = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };

    Tensor<value_t> R = row0 - subtrahend;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(9.0f), static_cast<value_t>(18.0f),
        static_cast<value_t>(27.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_subtraction_incompatible_shapes
 * @brief Subtraction throws when operand shapes are incompatible.
 */
TYPED_TEST(TypedTensor, operator_subtraction_incompatible_shapes)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::HOST);
    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };

    EXPECT_THROW({ Tensor<value_t> R = A - B; }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_subtraction_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE.
 */
TYPED_TEST(TypedTensor, operator_subtraction_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A - B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TypedTensor.operator_subtraction_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST.
 */
TYPED_TEST(TypedTensor, operator_subtraction_both_host_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A - B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TypedTensor.operator_subtraction_nan_inputs_throws
 * @brief Subtraction detects NaN inputs and triggers a temper::nan_error,
 * as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_subtraction_nan_inputs_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2}, MemoryLocation::DEVICE);

    const float nanf = std::numeric_limits<float>::quiet_NaN();
    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(nanf)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(2.0f), static_cast<value_t>(3.0f)
    };

    if constexpr (!std::is_floating_point_v<value_t>) {
        // Do nothing.
    } else {
        EXPECT_THROW({ Tensor<value_t> R = A - B; }, temper::nan_error);
    }
}

/**
 * @test TypedTensor.operator_subtraction_non_finite_result_throws
 * @brief Subtraction overflow to Inf should trigger a
 * temper::nonfinite_error, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_subtraction_non_finite_result_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({1}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    const float large = std::numeric_limits<float>::max();
    A = std::vector<value_t>{ static_cast<value_t>(large) };
    B = std::vector<value_t>{ static_cast<value_t>(-large) };

    if constexpr (!std::is_floating_point_v<value_t>) {
        // Do nothing.
    } else {
        EXPECT_THROW({ Tensor<value_t> R = A - B; }, temper::nonfinite_error);
    }
}

/**
 * @test TypedTensor.operator_subtraction_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} - B{3,1}.
 */
TYPED_TEST(TypedTensor, operator_subtraction_broadcasting_complex_alignment)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3, 4}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3, 1},   MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<value_t> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<value_t>(i);
    }

    std::vector<value_t> bvals = {
        static_cast<value_t>(10.0f),
        static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> R = A - B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            value_t offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] - offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[idx]),
                            static_cast<float>(expected[idx]));
        } else {
            EXPECT_EQ(rh[idx], expected[idx]);
        }
    }
}

/**
 * @test TypedTensor.operator_subtraction_alias_view_noncontiguous
 * @brief Element-wise subtraction where the right operand is a non-contig
 * 1D alias view (stride 2).
 */
TYPED_TEST(TypedTensor, operator_subtraction_alias_view_noncontiguous)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({6}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> v(B, {0}, {3}, {2});

    Tensor<value_t> R = A - v;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(-9), static_cast<value_t>(-28),
        static_cast<value_t>(-47)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }

    std::vector<value_t> b_host(6);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(),
                       sizeof(value_t) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(b_host[i]),
                            static_cast<float>(bvals[i]));
        } else {
            EXPECT_EQ(b_host[i], bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_subtraction_alias_view_broadcast
 * @brief Subtraction with a broadcasted alias view (stride 0).
 */
TYPED_TEST(TypedTensor, operator_subtraction_alias_view_broadcast)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(5.0f) };

    Tensor<value_t> vb(B, {0}, {3}, {0});

    Tensor<value_t> R = A - vb;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(-4), static_cast<value_t>(-3),
        static_cast<value_t>(-2)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_subtraction_alias_view_weird_strides
 * @brief Subtraction of a dense small tensor with a 2D alias view with odd
 * strides.
 */
TYPED_TEST(TypedTensor, operator_subtraction_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    owner = vals;

    Tensor<value_t> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<value_t> sub({3,4}, MemoryLocation::DEVICE);

    std::vector<value_t> sub_vals(12, static_cast<value_t>(3.0f));
    sub = sub_vals;

    Tensor<value_t> R = view - sub;

    Tensor<value_t> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<value_t> rh(12);
    g_sycl_queue.memcpy(rh.data(), hostR.m_p_data.get(),
                       sizeof(value_t) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(rh[k]),
                                static_cast<float>(vals[flat] - 3.0f));
            } else {
                EXPECT_EQ(rh[k], vals[flat] - static_cast<value_t>(3));
            }
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication
 * @brief Verifies element-wise multiplication on device memory.
 */
TYPED_TEST(TypedTensor, operator_multiplication)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2,3}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(6.0f), static_cast<value_t>(5.0f),
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> E = A * B;

    const uint64_t total = 6;
    std::vector<value_t> eh(total);
    g_sycl_queue.memcpy(eh.data(), E.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(eh[i]),
                            static_cast<float>(avals[i] * bvals[i]));
        } else {
            EXPECT_EQ(eh[i], avals[i] * bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for multiplication.
 */
TYPED_TEST(TypedTensor, operator_multiplication_broadcasting_1d_to_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };

    Tensor<value_t> R = A * B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(10.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(90.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(100.0f), static_cast<value_t>(180.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication_broadcasting_scalar
 * @brief Verifies multiplication broadcasting with a scalar operand.
 */
TYPED_TEST(TypedTensor, operator_multiplication_broadcasting_scalar)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::HOST);
    Tensor<value_t> B({1}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(5.0f) };

    Tensor<value_t> R = A * B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(5.0f), static_cast<value_t>(10.0f),
        static_cast<value_t>(15.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(25.0f), static_cast<value_t>(30.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication_with_views
 * @brief Verifies element-wise multiplication between a row view and a
 * tensor.
 */
TYPED_TEST(TypedTensor, operator_multiplication_with_views)
{
    using value_t = TypeParam;
    Tensor<value_t> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<value_t>{
        static_cast<value_t>(2.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(4.0f), static_cast<value_t>(5.0f),
        static_cast<value_t>(6.0f), static_cast<value_t>(7.0f)
    };

    Tensor<value_t> row0 = T[0];
    Tensor<value_t> multiplier({3}, MemoryLocation::DEVICE);
    multiplier = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f)
    };

    Tensor<value_t> R = row0 * multiplier;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(20.0f), static_cast<value_t>(60.0f),
        static_cast<value_t>(120.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication_incompatible_shapes
 * @brief Multiplication throws when operand shapes are incompatible.
 */
TYPED_TEST(TypedTensor, operator_multiplication_incompatible_shapes)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::HOST);
    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };

    EXPECT_THROW({ Tensor<value_t> R = A * B; }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_multiplication_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE.
 */
TYPED_TEST(TypedTensor, operator_multiplication_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A * B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TypedTensor.operator_multiplication_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST.
 */
TYPED_TEST(TypedTensor,
           operator_multiplication_both_host_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::HOST);
    Tensor<value_t> B({2,2}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A * B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TypedTensor.operator_multiplication_nan_inputs_throws
 * @brief Multiplication detects NaN inputs and triggers a temper::nan_error,
 * as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_multiplication_nan_inputs_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2}, MemoryLocation::DEVICE);

    const float nanf = std::numeric_limits<float>::quiet_NaN();

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(nanf)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(2.0f), static_cast<value_t>(3.0f)
    };

    if constexpr (!std::is_floating_point_v<value_t>) {
        // Do nothing for not floating point types.
    } else {
        EXPECT_THROW({ Tensor<value_t> R = A * B; }, temper::nan_error);
    }
}

/**
 * @test TypedTensor.operator_multiplication_non_finite_result_throws
 * @brief Non-finite multiplication result should trigger a
 * temper::nonfinite_error, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_multiplication_non_finite_result_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({1}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    const float large = std::numeric_limits<float>::max();
    A = std::vector<value_t>{ static_cast<value_t>(large) };
    B = std::vector<value_t>{ static_cast<value_t>(2.0f) };

    if constexpr (!std::is_floating_point_v<value_t>) {
        // Do nothing for not floating point types.
    } else {
        EXPECT_THROW({ Tensor<value_t> R = A * B; }, temper::nonfinite_error);
    }
}

/**
 * @test TypedTensor.operator_multiplication_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} * B{3,1}.
 */
TYPED_TEST(TypedTensor,
    operator_multiplication_broadcasting_complex_alignment)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3, 4}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3, 1},   MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<value_t> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<value_t>(i);
    }

    std::vector<value_t> bvals = {
        static_cast<value_t>(10.0f),
        static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> R = A * B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            value_t offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] * offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[idx]),
                            static_cast<float>(expected[idx]));
        } else {
            EXPECT_EQ(rh[idx], expected[idx]);
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication_alias_view_noncontiguous
 * @brief Element-wise multiplication where the right operand is a non-contig
 * 1D alias view (stride 2).
 */
TYPED_TEST(TypedTensor, operator_multiplication_alias_view_noncontiguous)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({6}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> v(B, {0}, {3}, {2});

    Tensor<value_t> R = A * v;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(10.0f), static_cast<value_t>(60.0f),
        static_cast<value_t>(150.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }

    std::vector<value_t> b_host(6);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(),
                       sizeof(value_t) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(b_host[i]),
                            static_cast<float>(bvals[i]));
        } else {
            EXPECT_EQ(b_host[i], bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication_alias_view_broadcast
 * @brief Multiplication with a broadcasted alias view (stride 0).
 */
TYPED_TEST(TypedTensor, operator_multiplication_alias_view_broadcast)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(5.0f) };

    Tensor<value_t> vb(B, {0}, {3}, {0});

    Tensor<value_t> R = A * vb;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(5.0f), static_cast<value_t>(10.0f),
        static_cast<value_t>(15.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_multiplication_alias_view_weird_strides
 * @brief Multiplication of a dense small tensor with a 2D alias view
 * with odd strides.
 */
TYPED_TEST(TypedTensor,
           operator_multiplication_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    owner = vals;

    Tensor<value_t> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<value_t> mul({3,4}, MemoryLocation::DEVICE);

    std::vector<value_t> mul_vals(12, static_cast<value_t>(3.0f));
    mul = mul_vals;

    Tensor<value_t> R = view * mul;

    Tensor<value_t> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<value_t> rh(12);
    g_sycl_queue.memcpy(rh.data(), hostR.m_p_data.get(),
                       sizeof(value_t) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(rh[k]),
                                static_cast<float>(vals[flat] * 3.0f));
            } else {
                EXPECT_EQ(rh[k], vals[flat] * static_cast<value_t>(3));
            }
        }
    }
}

/**
 * @test TypedTensor.operator_division
 * @brief Verifies element-wise division on device memory.
 */
TYPED_TEST(TypedTensor, operator_division)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2,3}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(6.0f), static_cast<value_t>(5.0f),
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> F = A / B;

    const uint64_t total = 6;
    std::vector<value_t> fh(total);
    g_sycl_queue.memcpy(fh.data(), F.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(fh[i]),
                            static_cast<float>(avals[i] / bvals[i]));
        } else {
            EXPECT_EQ(fh[i], avals[i] / bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_division_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for division.
 */
TYPED_TEST(TypedTensor, operator_division_broadcasting_1d_to_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };

    Tensor<value_t> R = A / B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(10.0f / 1.0f),
        static_cast<value_t>(20.0f / 2.0f),
        static_cast<value_t>(30.0f / 3.0f),
        static_cast<value_t>(40.0f / 1.0f),
        static_cast<value_t>(50.0f / 2.0f),
        static_cast<value_t>(60.0f / 3.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_division_broadcasting_scalar
 * @brief Verifies division broadcasting with a scalar operand.
 */
TYPED_TEST(TypedTensor, operator_division_broadcasting_scalar)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3}, MemoryLocation::HOST);
    Tensor<value_t> B({1}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(5.0f) };

    Tensor<value_t> R = A / B;

    uint64_t total = 6;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(1.0f / 5.0f),
        static_cast<value_t>(2.0f / 5.0f),
        static_cast<value_t>(3.0f / 5.0f),
        static_cast<value_t>(4.0f / 5.0f),
        static_cast<value_t>(5.0f / 5.0f),
        static_cast<value_t>(6.0f / 5.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_division_with_views
 * @brief Verifies element-wise division between a row view and a tensor.
 */
TYPED_TEST(TypedTensor, operator_division_with_views)
{
    using value_t = TypeParam;
    Tensor<value_t> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(50.0f), static_cast<value_t>(60.0f)
    };

    Tensor<value_t> row0 = T[0];
    Tensor<value_t> divisor({3}, MemoryLocation::DEVICE);
    divisor = std::vector<value_t>{
        static_cast<value_t>(2.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f)
    };

    Tensor<value_t> R = row0 / divisor;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(5.0f), static_cast<value_t>(5.0f),
        static_cast<value_t>(6.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_division_incompatible_shapes
 * @brief Division throws when operand shapes are incompatible.
 */
TYPED_TEST(TypedTensor, operator_division_incompatible_shapes)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3}, MemoryLocation::HOST);
    Tensor<value_t> B({2, 2}, MemoryLocation::HOST);
    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };

    EXPECT_THROW({ Tensor<value_t> R = A / B; }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_division_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE.
 */
TYPED_TEST(TypedTensor, operator_division_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 2}, MemoryLocation::HOST);
    Tensor<value_t> B({2, 2}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A / B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TypedTensor.operator_division_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST.
 */
TYPED_TEST(TypedTensor,
           operator_division_both_host_result_mem_location)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 2}, MemoryLocation::HOST);
    Tensor<value_t> B({2, 2}, MemoryLocation::HOST);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(4.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f), static_cast<value_t>(1.0f)
    };

    Tensor<value_t> R = A / B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TypedTensor.operator_division_by_zero_throws
 * @brief Division by zero in device kernel should trigger a runtime_error.
 */
TYPED_TEST(TypedTensor, operator_division_by_zero_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(-2.0f)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(0.0f), static_cast<value_t>(1.0f)
    };

    EXPECT_THROW({
        Tensor<value_t> R = A / B;
    }, temper::computation_error);
}

/**
 * @test TypedTensor.operator_division_nan_inputs_throws
 * @brief Division detects NaN inputs and triggers a temper::nan_error,
 * as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_division_nan_inputs_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2}, MemoryLocation::DEVICE);
    Tensor<value_t> B({2}, MemoryLocation::DEVICE);

    const float nanf = std::numeric_limits<float>::quiet_NaN();

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(nanf)
    };
    B = std::vector<value_t>{
        static_cast<value_t>(2.0f), static_cast<value_t>(3.0f)
    };

    if constexpr (!std::is_floating_point_v<value_t>)
    {
        // Do nothing.
    }
    else
    {
        EXPECT_THROW({ Tensor<value_t> R = A / B; }, temper::nan_error);
    }
}

/**
 * @test TypedTensor.operator_division_non_finite_result_throws
 * @brief Division that overflows to Inf should trigger a
 * temper::nonfinite_error, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_division_non_finite_result_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({1}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    float large = std::numeric_limits<float>::max();
    float tiny = std::numeric_limits<float>::min();
    A = std::vector<value_t>{ static_cast<value_t>(large) };
    B = std::vector<value_t>{ static_cast<value_t>(tiny) };

    if constexpr (!std::is_floating_point_v<value_t>)
    {
        // Do nothing.
    }
    else
    {
        EXPECT_THROW({ Tensor<value_t> R = A / B; }, temper::nonfinite_error);
    }
}

/**
 * @test TypedTensor.operator_division_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} / B{3,1}.
 */
TYPED_TEST(TypedTensor,
           operator_division_broadcasting_complex_alignment)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2, 3, 4}, MemoryLocation::DEVICE);
    Tensor<value_t> B({3, 1},   MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<value_t> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<value_t>(i);
    }

    std::vector<value_t> bvals = {
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> R = A / B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            value_t offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] / offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[idx]),
                            static_cast<float>(expected[idx]));
        } else {
            EXPECT_EQ(rh[idx], expected[idx]);
        }
    }
}

/**
 * @test TypedTensor.operator_division_alias_view_noncontiguous
 * @brief Element-wise division where the right operand is a non-contig
 * 1D alias view (no zero divisors at sampled indices).
 */
TYPED_TEST(TypedTensor,
           operator_division_alias_view_noncontiguous)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({6}, MemoryLocation::DEVICE);

    std::vector<value_t> avals = {
        static_cast<value_t>(100.0f), static_cast<value_t>(200.0f),
        static_cast<value_t>(300.0f)
    };
    std::vector<value_t> bvals = {
        static_cast<value_t>(2.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(4.0f), static_cast<value_t>(40.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(50.0f)
    };

    A = avals;
    B = bvals;

    Tensor<value_t> v(B, {0}, {3}, {2});

    Tensor<value_t> R = A / v;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(50.0f), static_cast<value_t>(50.0f),
        static_cast<value_t>(60.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }

    std::vector<value_t> b_host(6);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(),
                       sizeof(value_t) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(b_host[i]),
                            static_cast<float>(bvals[i]));
        } else {
            EXPECT_EQ(b_host[i], bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_division_alias_view_broadcast
 * @brief Division with a broadcasted alias view (stride 0).
 */
TYPED_TEST(TypedTensor, operator_division_alias_view_broadcast)
{
    using value_t = TypeParam;
    Tensor<value_t> A({3}, MemoryLocation::DEVICE);
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);

    A = std::vector<value_t>{
        static_cast<value_t>(10.0f), static_cast<value_t>(20.0f),
        static_cast<value_t>(30.0f)
    };
    B = std::vector<value_t>{ static_cast<value_t>(10.0f) };

    Tensor<value_t> vb(B, {0}, {3}, {0});

    Tensor<value_t> R = A / vb;

    const uint64_t total = 3;
    std::vector<value_t> rh(total);
    g_sycl_queue.memcpy(rh.data(), R.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(rh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(rh[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_division_alias_view_weird_strides
 * @brief Division of a dense small tensor by a 2D alias view with odd
 * strides.
 */
TYPED_TEST(TypedTensor, operator_division_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    owner = vals;

    Tensor<value_t> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<value_t> div({3,4}, MemoryLocation::DEVICE);

    std::vector<value_t> div_vals(12, static_cast<value_t>(2.0f));
    div = div_vals;

    Tensor<value_t> R = view / div;

    Tensor<value_t> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<value_t> rh(12);
    g_sycl_queue.memcpy(rh.data(), hostR.m_p_data.get(),
                       sizeof(value_t) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(rh[k]),
                                static_cast<float>(vals[flat] / 2.0f));
            } else {
                EXPECT_EQ(rh[k], vals[flat] / static_cast<value_t>(2));
            }
        }
    }
}

/**
 * @test TypedTensor.operator_unary_negation
 * @brief Verifies element-wise unary negation.
 */
TYPED_TEST(TypedTensor, operator_unary_negation)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::DEVICE);
    A = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(-2),
        static_cast<value_t>(3), static_cast<value_t>(0)
    };

    Tensor<value_t> N = -A;

    uint64_t total = 4;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<float> expected = {-1, 2, -3, -0};
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_EQ(static_cast<float>(nh[i]), expected[i]);
        } else {
            EXPECT_EQ(nh[i], static_cast<value_t>(expected[i]));
        }
    }
}

/**
 * @test TypedTensor.operator_unary_negation_result_mem_location_device
 * @brief Result memory follows input memory (DEVICE case).
 */
TYPED_TEST(TypedTensor, operator_unary_negation_result_mem_location_device)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::DEVICE);
    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(-2.0f),
        static_cast<value_t>(3.5f), static_cast<value_t>(0.0f)
    };
    A = avals;

    Tensor<value_t> N = -A;
    EXPECT_EQ(N.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TypedTensor.operator_unary_negation_result_mem_location_host
 * @brief Result memory follows input memory (HOST case).
 */
TYPED_TEST(TypedTensor, operator_unary_negation_result_mem_location_host)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::HOST);
    std::vector<value_t> avals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(-2.0f),
        static_cast<value_t>(3.5f), static_cast<value_t>(0.0f)
    };
    A = avals;

    Tensor<value_t> N = -A;
    EXPECT_EQ(N.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TypedTensor.operator_unary_negation_nan_input_throws
 * @brief Unary negation operator detects NaN inputs and triggers a
 * temper::nan_error, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, operator_unary_negation_nan_input_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2}, MemoryLocation::DEVICE);
    const float nanf = std::numeric_limits<float>::quiet_NaN();

    A = std::vector<value_t>{
        static_cast<value_t>(1.0f), static_cast<value_t>(nanf)
    };

    if constexpr (!std::is_floating_point_v<value_t>) {
        // Do nothing for not floating types.
    } else {
        EXPECT_THROW({ Tensor<value_t> N = -A; }, temper::nan_error);
    }
}

/**
 * @test TypedTensor.operator_unary_negation_empty_tensor_throws
 * @brief Negation on a rank-0 tensor must throw temper::validation_error.
 */
TYPED_TEST(TypedTensor,
           operator_unary_negation_empty_tensor_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> T;
    EXPECT_THROW({ Tensor<value_t> N = -T; }, temper::validation_error);
}

/**
 * @test TypedTensor.operator_unary_negation_with_view
 * @brief Negating a view returns correct values and does not modify parent.
 */
TYPED_TEST(TypedTensor, operator_unary_negation_with_view)
{
    using value_t = TypeParam;
    Tensor<value_t> T({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> tvals = {
        static_cast<value_t>(1), static_cast<value_t>(-2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(-5), static_cast<value_t>(6)
    };
    T = tvals;

    Tensor<value_t> row0 = T[0];
    Tensor<value_t> N = -row0;

    const uint64_t total = 3;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(-1), static_cast<value_t>(2),
        static_cast<value_t>(-3)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(nh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(nh[i], expected[i]);
        }
    }

    // Sanity: parent must remain unchanged.
    std::vector<value_t> parent_buf(6);
    g_sycl_queue.memcpy(parent_buf.data(), T.m_p_data.get(),
                       sizeof(value_t) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(parent_buf[i]),
                            static_cast<float>(tvals[i]));
        } else {
            EXPECT_EQ(parent_buf[i], tvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_unary_negation_sign_of_zero
 * @brief Check that negation preserves sign for zero (e.g. -0.0).
 */
TYPED_TEST(TypedTensor, operator_unary_negation_sign_of_zero)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::DEVICE);
    A = std::vector<value_t>{
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(-2), static_cast<value_t>(0)
    };

    Tensor<value_t> N = -A;

    const uint64_t total = 4;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(nh[0]), -0.0f);
        EXPECT_TRUE(std::signbit(static_cast<float>(nh[0])));
        EXPECT_FLOAT_EQ(static_cast<float>(nh[3]), -0.0f);
        EXPECT_TRUE(std::signbit(static_cast<float>(nh[3])));
        EXPECT_FLOAT_EQ(static_cast<float>(nh[1]), -1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(nh[2]), 2.0f);
    } else {
        // For integer types signbit not applicable; check expected ints
        EXPECT_EQ(nh[0], static_cast<value_t>(0));
        EXPECT_EQ(nh[3], static_cast<value_t>(0));
        EXPECT_EQ(nh[1], static_cast<value_t>(-1));
        EXPECT_EQ(nh[2], static_cast<value_t>(2));
    }
}

/**
 * @test TypedTensor.operator_unary_negation_non_contiguous_view_columns
 * @brief Negating a non-contiguous 2D view produces correct contiguous
 * result and does not modify the parent tensor.
 */
TYPED_TEST(TypedTensor, operator_unary_negation_non_contiguous_view_columns)
{
    using value_t = TypeParam;
    Tensor<value_t> T({3,4}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(12);
    for (uint64_t i = 0; i < 12; ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    T = vals;

    std::vector<uint64_t> start_indices = {0, 2};
    std::vector<uint64_t> view_shape = {3, 1};
    Tensor<value_t> col = Tensor<value_t>(T, start_indices, view_shape);

    Tensor<value_t> N = -col;

    const uint64_t total = 3 * 1;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    for (uint64_t r = 0; r < 3; ++r)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(nh[r]),
                            -static_cast<float>(vals[r * 4 + 2]));
        } else {
            EXPECT_EQ(nh[r], -vals[r * 4 + 2]);
        }
    }

    std::vector<value_t> parent_buf(12);
    g_sycl_queue.memcpy(parent_buf.data(), T.m_p_data.get(),
                       sizeof(value_t) * 12).wait();
    for (uint64_t i = 0; i < 12; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(parent_buf[i]),
                            static_cast<float>(vals[i]));
        } else {
            EXPECT_EQ(parent_buf[i], vals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_unary_negation_nan_outside_view
 * @brief If the parent contains a NaN outside the view region, negating
 * the view should succeed and not throw.
 */
TYPED_TEST(TypedTensor, operator_unary_negation_nan_outside_view)
{
    using value_t = TypeParam;
    Tensor<value_t> T({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f),
        static_cast<value_t>(std::numeric_limits<float>::quiet_NaN())
    };
    T = vals;
    std::vector<uint64_t> start_indices = {0, 0};
    std::vector<uint64_t> view_shape = {1, 3};
    Tensor<value_t> row0 = Tensor<value_t>(T, start_indices, view_shape);

    EXPECT_NO_THROW({ Tensor<value_t> N = -row0; });

    Tensor<value_t> N = -row0;
    const uint64_t total = 1 * 3;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(nh[i]),
                            -static_cast<float>(vals[i]));
        } else {
            EXPECT_EQ(nh[i], -vals[i]);
        }
    }

    std::vector<value_t> parent_buf(6);
    g_sycl_queue.memcpy(parent_buf.data(), T.m_p_data.get(),
                       sizeof(value_t) * 6).wait();
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_TRUE(std::isnan(static_cast<float>(parent_buf[5])));
    } else {
        // If not float, NaN isn't representable; skip check.
    }
}

/**
 * @test TypedTensor.operator_unary_negation_view_of_view
 * @brief Negating a view-of-view works and does not change the parent.
 */
TYPED_TEST(TypedTensor, operator_unary_negation_view_of_view)
{
    using value_t = TypeParam;
    Tensor<value_t> T({2,4}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f),
        static_cast<value_t>(7.0f), static_cast<value_t>(8.0f)
    };
    T = vals;

    Tensor<value_t> row1 = Tensor<value_t>(T, {1,0}, {4});
    Tensor<value_t> sub = Tensor<value_t>(row1, {1}, {2});
    Tensor<value_t> N = -sub;

    const uint64_t total = 2;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(nh[0]),
                        -static_cast<float>(vals[1 * 4 + 1]));
        EXPECT_FLOAT_EQ(static_cast<float>(nh[1]),
                        -static_cast<float>(vals[1 * 4 + 2]));
    } else {
        EXPECT_EQ(nh[0], -vals[1 * 4 + 1]);
        EXPECT_EQ(nh[1], -vals[1 * 4 + 2]);
    }

    std::vector<value_t> parent_buf(8);
    g_sycl_queue.memcpy(parent_buf.data(), T.m_p_data.get(),
                       sizeof(value_t) * 8).wait();
    for (uint64_t i = 0; i < 8; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(parent_buf[i]),
                            static_cast<float>(vals[i]));
        } else {
            EXPECT_EQ(parent_buf[i], vals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_unary_negation_alias_view_noncontiguous
 * @brief Unary negation applied to a non-contiguous 1D alias view returns
 * correct values and does not modify the parent tensor.
 */
TYPED_TEST(TypedTensor, operator_unary_negation_alias_view_noncontiguous)
{
    using value_t = TypeParam;
    Tensor<value_t> B({6}, MemoryLocation::DEVICE);
    std::vector<value_t> bvals = {
        static_cast<value_t>(1), static_cast<value_t>(-2),
        static_cast<value_t>(3), static_cast<value_t>(-4),
        static_cast<value_t>(5), static_cast<value_t>(-6)
    };
    B = bvals;

    Tensor<value_t> v(B, {0}, {3}, {2});

    Tensor<value_t> N = -v;

    const uint64_t total = 3;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(-1), static_cast<value_t>(-3),
        static_cast<value_t>(-5)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(nh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(nh[i], expected[i]);
        }
    }

    // parent remains unchanged
    std::vector<value_t> b_host(6);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(),
                       sizeof(value_t) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(b_host[i]),
                            static_cast<float>(bvals[i]));
        } else {
            EXPECT_EQ(b_host[i], bvals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_unary_negation_alias_view_broadcast
 * @brief Unary negation applied to a broadcasted alias view (stride 0).
 */
TYPED_TEST(TypedTensor, operator_unary_negation_alias_view_broadcast)
{
    using value_t = TypeParam;
    Tensor<value_t> B({1}, MemoryLocation::DEVICE);
    B = std::vector<value_t>{ static_cast<value_t>(5) };

    Tensor<value_t> vb(B, {0}, {3}, {0});

    Tensor<value_t> N = -vb;

    const uint64_t total = 3;
    std::vector<value_t> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(),
                       sizeof(value_t) * total).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(-5), static_cast<value_t>(-5),
        static_cast<value_t>(-5)
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(nh[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(nh[i], expected[i]);
        }
    }

    std::vector<value_t> b_host(1);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(),
                       sizeof(value_t) * 1).wait();
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(b_host[0]),
                        static_cast<float>(5.0f));
    } else {
        EXPECT_EQ(b_host[0], static_cast<value_t>(5));
    }
}

/**
 * @test TypedTensor.operator_unary_negation_alias_view_weird_strides
 * @brief Unary negation applied to a non-contiguous 2D alias view with odd
 * strides.
 */
TYPED_TEST(TypedTensor, operator_unary_negation_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> T({5,20}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i) vals[i] =
        static_cast<value_t>(i + 1);
    T = vals;

    Tensor<value_t> v(T, {1,2}, {2,3}, {13,4});

    Tensor<value_t> N = -v;

    Tensor<value_t> hostN({2,3}, MemoryLocation::HOST);
    copy_tensor_data(hostN, N);
    std::vector<value_t> nh(6);
    g_sycl_queue.memcpy(nh.data(), hostN.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            uint64_t start_flat = 1 * 20 + 2;
            uint64_t flat = start_flat + i * 13 + j * 4;
            uint64_t k = i * 3 + j;
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(nh[k]),
                                -static_cast<float>(vals[flat]));
            } else {
                EXPECT_EQ(nh[k], -vals[flat]);
            }
        }
    }

    std::vector<value_t> parent_buf(100);
    g_sycl_queue.memcpy(parent_buf.data(), T.m_p_data.get(),
                       sizeof(value_t) * 100).wait();
    for (uint64_t i = 0; i < 100; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(parent_buf[i]),
                            static_cast<float>(vals[i]));
        } else {
            EXPECT_EQ(parent_buf[i], vals[i]);
        }
    }
}

/**
 * @test TypedTensor.operator_compare_wrong_dimensions
 * @brief Using operator== on two tensors with different shapes
 * should return false.
 */
TYPED_TEST(TypedTensor, operator_compare_wrong_dimensions)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 4});
    Tensor<value_t> t2({2, 2});

    EXPECT_NE(t1, t2);
}

/**
 * @test TypedTensor.operator_compare_both_empty
 * @brief Using operator== on two empty tensors should return true.
 */
TYPED_TEST(TypedTensor, operator_compare_both_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t1;
    Tensor<value_t> t2;

    EXPECT_EQ(t1, t2);
}

/**
 * @test TypedTensor.operator_compare_matching_values
 * @brief Using operator== on two tensors with matching values should return true.
 */
TYPED_TEST(TypedTensor, operator_compare_matching_values)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 2});
    Tensor<value_t> t2({2, 2});

    std::vector<value_t> vals = {1, 2, 3 ,4};

    t1 = vals;
    t2 = vals;

    EXPECT_EQ(t1, t2);
}

/**
 * @test TypedTensor.operator_compare_mismatching_values
 * @brief Using operator== on two tensors with mismatching values
 * should return false.
 */
TYPED_TEST(TypedTensor, operator_compare_mismatching_values)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 2});
    Tensor<value_t> t2({2, 2});

    std::vector<value_t> vals1 = {1, 2, 3 ,4};
    std::vector<value_t> vals2 = {4, 4, 4 ,4};

    t1 = vals1;
    t2 = vals2;

    EXPECT_NE(t1, t2);
}

/**
 * @test TypedTensor.operator_compare_matching_values_wrong_shapes
 * @brief Using operator== on two tensors with matching values but different
 * shapes should return true.
 */
TYPED_TEST(TypedTensor, operator_compare_matching_values_wrong_shapes)
{
    using value_t = TypeParam;
    Tensor<value_t> t1({2, 1, 2});
    Tensor<value_t> t2({2, 2});

    std::vector<value_t> vals = {1, 2, 3 ,4};

    t1 = vals;
    t2 = vals;

    EXPECT_NE(t1, t2);
}

/**
 * @test TypedTensor.clone_empty
 * @brief Cloning an empty tensor should throw.
 */
TYPED_TEST(TypedTensor, clone_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;

    EXPECT_TRUE(t.m_dimensions.empty());
    EXPECT_EQ(t.get_num_elements(), 0u);
    EXPECT_EQ(t.m_p_data, nullptr);

    EXPECT_THROW(t.clone(), temper::validation_error);
}

/**
 * @test TypedTensor.clone_1d
 * @brief Cloning a 1D tensor should copy data and allocate new storage.
 */
TYPED_TEST(TypedTensor, clone_1d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::HOST);
    t = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5)
    };

    Tensor<value_t> c = t.clone();

    EXPECT_EQ(c.m_dimensions, t.m_dimensions);
    EXPECT_EQ(c.get_num_elements(), t.get_num_elements());

    uint64_t n = t.get_num_elements();
    for (uint64_t i = 0; i < n; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(c[i]),
                            static_cast<float>(t[i]));
        } else {
            EXPECT_EQ(c[i], t[i]);
        }
    }

    EXPECT_NE(c.m_p_data, t.m_p_data);
}

/**
 * @test TypedTensor.clone_2d
 * @brief Cloning a 2D tensor should preserve both dimensions and values.
 */
TYPED_TEST(TypedTensor, clone_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    t = std::vector<value_t>{
        static_cast<value_t>(1), static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5), static_cast<value_t>(6)
    };

    Tensor<value_t> c = t.clone();

    EXPECT_EQ(c.get_dimensions(), (std::vector<uint64_t>{2, 3}));

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(c[i][j]),
                                static_cast<float>(t[i][j]));
            } else {
                EXPECT_EQ(c[i][j], t[i][j]);
            }
        }
    }

    EXPECT_NE(c.get_data(), t.get_data());
}

/**
 * @test TypedTensor.clone_view
 * @brief Cloning a view tensor should yield an independent full copy.
 */
TYPED_TEST(TypedTensor, clone_view)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 4}, MemoryLocation::HOST);

    for (uint64_t i = 0; i < 16; ++i)
    {
        uint64_t r = i / 4;
        uint64_t cidx = i % 4;
        t[r][cidx] = static_cast<value_t>(i);
    }

    Tensor<value_t> v(t, {1, 1}, {2, 2});
    Tensor<value_t> c = v.clone();

    EXPECT_EQ(c.get_dimensions(), v.get_dimensions());

    for (uint64_t i = 0; i < v.get_dimensions()[0]; ++i)
    {
        for (uint64_t j = 0; j < v.get_dimensions()[1]; ++j)
        {
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(c[i][j]),
                                static_cast<float>(v[i][j]));
            } else {
                EXPECT_EQ(c[i][j], v[i][j]);
            }
        }
    }

    EXPECT_NE(c.m_p_data, v.m_p_data);
}

/**
 * @test TypedTensor.clone_alias_view
 * @brief Tests that cloning an alias view produces an independent owning tensor.
 */
TYPED_TEST(TypedTensor, clone_alias_view)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 4}, MemoryLocation::HOST);
    for (uint64_t i = 0; i < 16; ++i)
    {
        uint64_t r = i / 4;
        uint64_t cidx = i % 4;
        t[r][cidx] = static_cast<value_t>(i);
    }

    Tensor<value_t> view(
        t,
        {0, 0},
        {2, 2},
        {13, 2}
    );

    Tensor<value_t> clone = view.clone();

    ASSERT_EQ(clone.get_dimensions(), std::vector<uint64_t>({2, 2}));

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            uint64_t owner_index = i * 13 + j * 2;
            float expected = static_cast<float>(t.get_data()[owner_index]);
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(clone[i][j]), expected);
            } else {
                EXPECT_EQ(clone[i][j], static_cast<value_t>(expected));
            }
        }
    }

    EXPECT_NE(clone.m_p_data, view.m_p_data);

    t[0][0] = static_cast<value_t>(999.0f);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_NE(static_cast<float>(clone[0][0]), 999.0f);
    } else {
        EXPECT_NE(clone[0][0], static_cast<value_t>(999));
    }
}

/**
 * @test TypedTensor.clone_const
 * @brief Ensure clone() can be called on const Tensor.
 */
TYPED_TEST(TypedTensor, clone_const)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::HOST);
    t = std::vector<value_t>{
        static_cast<value_t>(10), static_cast<value_t>(20),
        static_cast<value_t>(30)
    };

    const Tensor<value_t>& ct = t;
    Tensor<value_t> c = ct.clone();

    EXPECT_EQ(c.m_dimensions, t.m_dimensions);

    uint64_t n = t.get_num_elements();
    for (uint64_t i = 0; i < n; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(c[i]),
                            static_cast<float>(t[i]));
        } else {
            EXPECT_EQ(c[i], t[i]);
        }
    }

    EXPECT_NE(c.m_p_data, t.m_p_data);
}

/**
 * @test TypedTensor.copy_from_identical_contiguous
 * @brief copy_from fast-path for identical contiguous shapes (memcpy path).
 */
TYPED_TEST(TypedTensor, copy_from_identical_contiguous)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2,3}, MemoryLocation::HOST);
    Tensor<value_t> dst({2,3}, MemoryLocation::HOST);

    std::vector<value_t> vals = {
        static_cast<value_t>(1.f), static_cast<value_t>(2.f),
        static_cast<value_t>(3.f), static_cast<value_t>(4.f),
        static_cast<value_t>(5.f), static_cast<value_t>(6.f)
    };
    src = vals;
    dst = std::vector<value_t>{ static_cast<value_t>(0), static_cast<value_t>(0),
                                static_cast<value_t>(0), static_cast<value_t>(0),
                                static_cast<value_t>(0), static_cast<value_t>(0) };

    dst.copy_from(src);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), dst.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    for (size_t i = 0; i < out.size(); ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[i]),
                            static_cast<float>(vals[i]));
        } else {
            EXPECT_EQ(out[i], vals[i]);
        }
    }
}

/**
 * @test TypedTensor.copy_from_scalar_to_owner_and_view
 * @brief copy_from with scalar source writes the scalar to owner and to a view.
 */
TYPED_TEST(TypedTensor, copy_from_scalar_to_owner_and_view)
{
    using value_t = TypeParam;
    // scalar -> owner
    Tensor<value_t> scalar({1}, MemoryLocation::HOST);
    scalar = static_cast<value_t>(7.5f);

    Tensor<value_t> owner({2,2}, MemoryLocation::HOST);
    owner = std::vector<value_t>{ static_cast<value_t>(0), static_cast<value_t>(0),
                                  static_cast<value_t>(0), static_cast<value_t>(0) };

    owner.copy_from(scalar);

    std::vector<value_t> out_owner(4);
    g_sycl_queue.memcpy(out_owner.data(),
        owner.m_p_data.get(), sizeof(value_t) * 4).wait();
    for (value_t v : out_owner) {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(v), 7.5f);
        } else {
            EXPECT_EQ(v, static_cast<value_t>(7));
        }
    }

    Tensor<value_t> base({3,3}, MemoryLocation::HOST);
    std::vector<value_t> base_vals(9);
    for (uint64_t i = 0; i < 9; ++i)
    {
        base_vals[i] = static_cast<value_t>(i + 1);
    }
    base = base_vals;

    Tensor<value_t> col_view(base, {0,1}, {3}, { base.m_strides[0] });

    col_view.copy_from(scalar);

    // read back base
    std::vector<value_t> base_out(9);
    g_sycl_queue.memcpy(base_out.data(),
        base.m_p_data.get(), sizeof(value_t) * 9).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[1]), 7.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[4]), 7.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[7]), 7.5f);

        EXPECT_FLOAT_EQ(static_cast<float>(base_out[0]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[2]), 3.f);
    } else {
        EXPECT_EQ(base_out[1], static_cast<value_t>(7));
        EXPECT_EQ(base_out[4], static_cast<value_t>(7));
        EXPECT_EQ(base_out[7], static_cast<value_t>(7));

        EXPECT_EQ(base_out[0], static_cast<value_t>(1));
        EXPECT_EQ(base_out[2], static_cast<value_t>(3));
    }
}

/**
 * @test TypedTensor.copy_from_broadcast_1d_to_2d
 * @brief copy_from broadcasting from 1-D into 2-D destination.
 */
TYPED_TEST(TypedTensor, copy_from_broadcast_1d_to_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> src({3}, MemoryLocation::HOST);
    src = std::vector<value_t>{
        static_cast<value_t>(1.f), static_cast<value_t>(2.f),
        static_cast<value_t>(3.f)
    };

    Tensor<value_t> dst({2,3}, MemoryLocation::HOST);
    dst = std::vector<value_t>{
        static_cast<value_t>(0), static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(0), static_cast<value_t>(0)
    };

    dst.copy_from(src);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), dst.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 2.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2]), 3.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[3]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[4]), 2.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[5]), 3.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(1));
        EXPECT_EQ(out[1], static_cast<value_t>(2));
        EXPECT_EQ(out[2], static_cast<value_t>(3));
        EXPECT_EQ(out[3], static_cast<value_t>(1));
        EXPECT_EQ(out[4], static_cast<value_t>(2));
        EXPECT_EQ(out[5], static_cast<value_t>(3));
    }
}

/**
 * @test TypedTensor.copy_from_from_view_into_owner
 * @brief copy_from where the source is a non-owning view.
 */
TYPED_TEST(TypedTensor, copy_from_from_view_into_owner)
{
    using value_t = TypeParam;
    Tensor<value_t> owner_src({2,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(10.f), static_cast<value_t>(11.f), static_cast<value_t>(12.f),
        static_cast<value_t>(20.f), static_cast<value_t>(21.f), static_cast<value_t>(22.f)
    };
    owner_src = vals;

    Tensor<value_t> src_view(owner_src, {1,0}, {3});

    Tensor<value_t> dst({2,3}, MemoryLocation::HOST);
    dst = std::vector<value_t>{
        static_cast<value_t>(0), static_cast<value_t>(0), static_cast<value_t>(0),
        static_cast<value_t>(0), static_cast<value_t>(0), static_cast<value_t>(0)
    };

    dst.copy_from(src_view);

    std::vector<value_t> out(6);
    g_sycl_queue.memcpy(out.data(), dst.m_p_data.get(),
                       sizeof(value_t) * 6).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 20.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 21.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2]), 22.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[3]), 20.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[4]), 21.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[5]), 22.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(20));
        EXPECT_EQ(out[1], static_cast<value_t>(21));
        EXPECT_EQ(out[2], static_cast<value_t>(22));
        EXPECT_EQ(out[3], static_cast<value_t>(20));
        EXPECT_EQ(out[4], static_cast<value_t>(21));
        EXPECT_EQ(out[5], static_cast<value_t>(22));
    }
}

/**
 * @test TypedTensor.copy_from_to_noncontiguous_dst
 * @brief copy_from into a non-contiguous (strided) destination.
 */
TYPED_TEST(TypedTensor, copy_from_to_noncontiguous_dst)
{
    using value_t = TypeParam;
    Tensor<value_t> src({2,2}, MemoryLocation::HOST);
    src = std::vector<value_t>{
        static_cast<value_t>(1.f), static_cast<value_t>(2.f),
        static_cast<value_t>(3.f), static_cast<value_t>(4.f)
    };

    Tensor<value_t> base({3,3}, MemoryLocation::HOST);
    base = std::vector<value_t>(9, static_cast<value_t>(0.f));

    Tensor<value_t> patch(base, {1,1}, {2,2},
        { base.m_strides[0], base.m_strides[1] });

    patch.copy_from(src);

    std::vector<value_t> base_out(9);
    g_sycl_queue.memcpy(base_out.data(),
        base.m_p_data.get(), sizeof(value_t) * 9).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[1 * 3 + 1]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[1 * 3 + 2]), 2.f);
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[2 * 3 + 1]), 3.f);
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[2 * 3 + 2]), 4.f);

        EXPECT_FLOAT_EQ(static_cast<float>(base_out[0]), 0.f);
        EXPECT_FLOAT_EQ(static_cast<float>(base_out[3]), 0.f);
    } else {
        EXPECT_EQ(base_out[1 * 3 + 1], static_cast<value_t>(1));
        EXPECT_EQ(base_out[1 * 3 + 2], static_cast<value_t>(2));
        EXPECT_EQ(base_out[2 * 3 + 1], static_cast<value_t>(3));
        EXPECT_EQ(base_out[2 * 3 + 2], static_cast<value_t>(4));

        EXPECT_EQ(base_out[0], static_cast<value_t>(0));
        EXPECT_EQ(base_out[3], static_cast<value_t>(0));
    }
}

/**
 * @test TypedTensor.copy_from_incompatible_shapes_throws
 * @brief copy_from should throw when source cannot be broadcast to destination.
 */
TYPED_TEST(TypedTensor, copy_from_incompatible_shapes_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> src({3}, MemoryLocation::HOST);
    src = std::vector<value_t>{
        static_cast<value_t>(1.f), static_cast<value_t>(2.f), static_cast<value_t>(3.f)
    };

    Tensor<value_t> dst({2,2}, MemoryLocation::HOST);
    dst = std::vector<value_t>{
        static_cast<value_t>(0.f), static_cast<value_t>(0.f),
        static_cast<value_t>(0.f), static_cast<value_t>(0.f)
    };

    EXPECT_THROW({
        dst.copy_from(src);
    }, temper::validation_error);
}

/**
 * @test TypedTensor.to_host_to_device
 * @brief Moves a tensor from HOST to DEVICE and verifies data integrity.
 */
TYPED_TEST(TypedTensor, to_host_to_device)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 5}, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : t.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<value_t> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<value_t>(i + 1);
    }

    t = values;

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    t.to(MemoryLocation::DEVICE);

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> host_data(total_size);
    g_sycl_queue.memcpy(host_data.data(), t.m_p_data.get(),
                       sizeof(value_t) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_data[i]),
                            static_cast<float>(i + 1));
        } else {
            EXPECT_EQ(host_data[i], static_cast<value_t>(i + 1));
        }
    }
}

/**
 * @test TypedTensor.to_device_to_host
 * @brief Moves a tensor from DEVICE to HOST and verifies data integrity.
 */
TYPED_TEST(TypedTensor, to_device_to_host)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 4}, MemoryLocation::DEVICE);

    uint64_t total_size = 1;
    for (uint64_t d : t.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<value_t> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<value_t>(i + 1);
    }

    t = values;

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    t.to(MemoryLocation::HOST);

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    std::vector<value_t> host_data(total_size);
    g_sycl_queue.memcpy(host_data.data(), t.m_p_data.get(),
                       sizeof(value_t) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_data[i]),
                            static_cast<float>(i + 1));
        } else {
            EXPECT_EQ(host_data[i], static_cast<value_t>(i + 1));
        }
    }
}

/**
 * @test TypedTensor.to_noop_when_already_in_target
 * @brief Calling to() with current memory location should do nothing.
 */
TYPED_TEST(TypedTensor, to_noop_when_already_in_target)
{
    using value_t = TypeParam;
    Tensor<value_t> t_host({2, 2}, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : t_host.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<value_t> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<value_t>(i + 1);
    }

    t_host = values;

    t_host.to(MemoryLocation::HOST);
    EXPECT_EQ(t_host.m_mem_loc, MemoryLocation::HOST);

    std::vector<value_t> host_data_host(total_size);
    g_sycl_queue.memcpy(host_data_host.data(),
        t_host.m_p_data.get(), sizeof(value_t) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_data_host[i]),
                            static_cast<float>(i + 1));
        } else {
            EXPECT_EQ(host_data_host[i], static_cast<value_t>(i + 1));
        }
    }

    Tensor<value_t> t_device({2, 2}, MemoryLocation::DEVICE);

    t_device = values;

    t_device.to(MemoryLocation::DEVICE);
    EXPECT_EQ(t_device.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<value_t> host_data_device(total_size);
    g_sycl_queue.memcpy(host_data_device.data(),
        t_device.m_p_data.get(), sizeof(value_t) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_data_device[i]),
                            static_cast<float>(i + 1));
        } else {
            EXPECT_EQ(host_data_device[i], static_cast<value_t>(i + 1));
        }
    }
}

/**
 * @test TypedTensor.to_throws_for_view
 * @brief Calling to() on a view (non-owning tensor) should throw.
 */
TYPED_TEST(TypedTensor, to_throws_for_view)
{
    using value_t = TypeParam;
    Tensor<value_t> t_owner({2, 2}, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : t_owner.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<value_t> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<value_t>(i + 1);
    }

    t_owner = values;

    Tensor<value_t> t_view(t_owner, {0, 0}, {2, 1});

    EXPECT_FALSE(t_view.m_own_data);

    EXPECT_THROW(t_view.to(MemoryLocation::DEVICE), temper::validation_error);
    EXPECT_THROW(t_view.to(MemoryLocation::HOST), temper::validation_error);
}

/**
 * @test TypedTensor.to_throws_for_empty_tensor
 * @brief Calling to() on a tensor with no elements should throw.
 */
TYPED_TEST(TypedTensor, to_throws_for_empty_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t_empty;
    EXPECT_THROW(t_empty.to(MemoryLocation::DEVICE), temper::validation_error);
}

/**
 * @test TypedTensor.to_host_to_device_and_back
 * @brief Moves tensor from HOST to DEVICE and back, verifying data integrity.
 */
TYPED_TEST(TypedTensor, to_host_to_device_and_back)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 4}, MemoryLocation::HOST);
    uint64_t total_size = 16;
    std::vector<value_t> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<value_t>(i + 10);
    }

    t = values;
    t.to(MemoryLocation::DEVICE);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    t.to(MemoryLocation::HOST);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    std::vector<value_t> host_data(total_size);
    g_sycl_queue.memcpy(host_data.data(), t.m_p_data.get(),
                       sizeof(value_t) * total_size).wait();
    for (uint64_t i = 0; i < total_size; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_data[i]),
                            static_cast<float>(i + 10));
        } else {
            EXPECT_EQ(host_data[i], static_cast<value_t>(i + 10));
        }
    }
}

/**
 * @test TypedTensor.reshape_preserves_linear_memory_and_strides
 * @brief Reshaping a tensor (2x3 -> 3x2) preserves linear memory and recomputes strides.
 */
TYPED_TEST(TypedTensor, reshape_preserves_linear_memory_and_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(4.0f), static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    A = vals;

    std::vector<uint64_t> new_dims = {3, 2};
    EXPECT_NO_THROW(A.reshape(new_dims));

    EXPECT_EQ(static_cast<uint64_t>(A.m_dimensions.size()), uint64_t{2});
    EXPECT_EQ(A.m_dimensions[0], uint64_t{3});
    EXPECT_EQ(A.m_dimensions[1], uint64_t{2});

    ASSERT_EQ(static_cast<uint64_t>(A.m_strides.size()), uint64_t{2});
    EXPECT_EQ(A.m_strides[0], uint64_t{2});
    EXPECT_EQ(A.m_strides[1], uint64_t{1});

    const uint64_t total = 6;
    std::vector<value_t> host_buf(total);
    g_sycl_queue.memcpy(host_buf.data(), A.m_p_data.get(),
                        sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_buf[i]),
                            static_cast<float>(vals[i]));
        } else {
            EXPECT_EQ(host_buf[i], vals[i]);
        }
    }
}

/**
 * @test TypedTensor.reshape_to_flat_vector_preserves_contents
 * @brief Reshape to a single-dimension tensor (1x6) preserves buffer and produces stride [1].
 */
TYPED_TEST(TypedTensor, reshape_to_flat_vector_preserves_contents)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(10.0f), static_cast<value_t>(11.0f), static_cast<value_t>(12.0f),
        static_cast<value_t>(13.0f), static_cast<value_t>(14.0f), static_cast<value_t>(15.0f)
    };
    A = vals;

    std::vector<uint64_t> new_dims = {6};
    EXPECT_NO_THROW(A.reshape(new_dims));

    EXPECT_EQ(static_cast<uint64_t>(A.m_dimensions.size()), uint64_t{1});
    EXPECT_EQ(A.m_dimensions[0], uint64_t{6});

    ASSERT_EQ(static_cast<uint64_t>(A.m_strides.size()), uint64_t{1});
    EXPECT_EQ(A.m_strides[0], uint64_t{1});

    const uint64_t total = 6;
    std::vector<value_t> host_buf(total);
    g_sycl_queue.memcpy(host_buf.data(), A.m_p_data.get(),
                        sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_buf[i]),
                            static_cast<float>(vals[i]));
        } else {
            EXPECT_EQ(host_buf[i], vals[i]);
        }
    }
}

/**
 * @test TypedTensor.reshape_invalid_size_throws
 * @brief Reshaping to dimensions whose product differs from the original must throw.
 */
TYPED_TEST(TypedTensor, reshape_invalid_size_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f), static_cast<value_t>(3.0f),
        static_cast<value_t>(4.0f), static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    A = vals;

    std::vector<uint64_t> bad_dims = {4, 2};
    EXPECT_THROW({ A.reshape(bad_dims); }, temper::validation_error);
}

/**
 * @test TypedTensor.reshape_empty_dimensions_throws
 * @brief Reshaping with an empty dimensions vector must throw.
 */
TYPED_TEST(TypedTensor, reshape_empty_dimensions_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    std::vector<uint64_t> bad_dims = {};
    EXPECT_THROW({ A.reshape(bad_dims); }, temper::validation_error);
}

/**
 * @test TypedTensor.reshape_new_dimensions_with_zero_throws
 * @brief Reshaping with new_dimensions containing zero must throw.
 */
TYPED_TEST(TypedTensor, reshape_new_dimensions_with_zero_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    std::vector<uint64_t> bad_dims = {2, 0};
    EXPECT_THROW({ A.reshape(bad_dims); }, temper::validation_error);
}

/**
 * @test TypedTensor.reshape_dimension_product_overflow_throws
 * @brief Reshaping with excessively large dimensions causing
 * overflow must throw.
 */
TYPED_TEST(TypedTensor, reshape_dimension_product_overflow_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,2}, MemoryLocation::DEVICE);
    std::vector<uint64_t> bad_dims = {UINT64_MAX, 2};
    EXPECT_THROW({ A.reshape(bad_dims); }, temper::bounds_error);
}

/**
 * @test TypedTensor.reshape_multiple_roundtrip_preserves_data
 * @brief Perform multiple reshapes and verify the linear buffer and
 * final shape return to original.
 */
TYPED_TEST(TypedTensor, reshape_multiple_roundtrip_preserves_data)
{
    using value_t = TypeParam;
    Tensor<value_t> A({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> orig = {
        static_cast<value_t>(7.0f), static_cast<value_t>(8.0f), static_cast<value_t>(9.0f),
        static_cast<value_t>(10.0f), static_cast<value_t>(11.0f), static_cast<value_t>(12.0f)
    };
    A = orig;

    EXPECT_NO_THROW(A.reshape({3,2}));
    EXPECT_NO_THROW(A.reshape({6}));
    EXPECT_NO_THROW(A.reshape({2,3}));

    ASSERT_EQ(static_cast<uint64_t>(A.m_dimensions.size()), uint64_t{2});
    EXPECT_EQ(A.m_dimensions[0], uint64_t{2});
    EXPECT_EQ(A.m_dimensions[1], uint64_t{3});

    const uint64_t total = 6;
    std::vector<value_t> host_buf(total);
    g_sycl_queue.memcpy(host_buf.data(), A.m_p_data.get(),
                        sizeof(value_t) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host_buf[i]),
                            static_cast<float>(orig[i]));
        } else {
            EXPECT_EQ(host_buf[i], orig[i]);
        }
    }
}

/**
 * @test TypedTensor.reshape_view_throws
 * @brief Attempting to reshape an alias/view must throw (non-owning).
 */
TYPED_TEST(TypedTensor, reshape_view_throws)
{
    using value_t = TypeParam;
    Tensor<value_t> base({2,3}, MemoryLocation::HOST);
    base = std::vector<value_t>{
        static_cast<value_t>(0), static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4), static_cast<value_t>(5)
    };

    Tensor<value_t> v(base, {0,0}, {2,3}, {3,1});
    EXPECT_FALSE(v.m_own_data);

    EXPECT_THROW(v.reshape(std::vector<uint64_t>{3,2}), temper::validation_error);
}

/**
 * @test TypedTensor.sort_empty
 * @brief Sorting an empty tensor should not throw.
 */
TYPED_TEST(TypedTensor, sort_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    EXPECT_NO_THROW(t.sort(0));
}

/**
 * @test TypedTensor.sort_axis_out_of_bounds
 * @brief Sorting with invalid axis should throw.
 */
TYPED_TEST(TypedTensor, sort_axis_out_of_bounds)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::HOST);
    EXPECT_THROW(t.sort(1), temper::bounds_error);
    EXPECT_THROW(t.sort(-2), temper::bounds_error);
}

/**
 * @test TypedTensor.sort_axis_size_one
 * @brief Sorting along axis with size <= 1 should not modify tensor.
 */
TYPED_TEST(TypedTensor, sort_axis_size_one)
{
    using value_t = TypeParam;
    Tensor<value_t> t({1}, MemoryLocation::HOST);
    t = static_cast<value_t>(123.0f);
    EXPECT_NO_THROW(t.sort(0));
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t),
                        static_cast<float>(123.0f));
    } else {
        EXPECT_EQ(t, static_cast<value_t>(123));
    }
}

/**
 * @test TypedTensor.sort_1D_basic
 * @brief Sorting a 1D tensor with random values.
 */
TYPED_TEST(TypedTensor, sort_1D_basic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(3.0f),
        static_cast<value_t>(-1.0f),
        static_cast<value_t>(2.5f),
        static_cast<value_t>(0.0f),
        static_cast<value_t>(10.0f)
    };
    t = vals;
    t.sort(0);
    std::vector<value_t> expected = {
        static_cast<value_t>(-1.0f),
        static_cast<value_t>(0.0f),
        static_cast<value_t>(2.5f),
        static_cast<value_t>(3.0f),
        static_cast<value_t>(10.0f)
    };
    for (uint64_t i = 0; i < 5; i++)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(t[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(t[i], expected[i]);
        }
    }
}

/**
 * @test TypedTensor.sort_2D_axis0
 * @brief Sorting a 2D tensor along axis 0 (rows).
 */
TYPED_TEST(TypedTensor, sort_2D_axis0)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 2}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(1), static_cast<value_t>(5),
        static_cast<value_t>(4), static_cast<value_t>(0)
    };
    t = vals;
    t.sort(0);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2][0]), 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2][1]), 5.0f);
    } else {
        EXPECT_EQ(t[0][0], static_cast<value_t>(1));
        EXPECT_EQ(t[0][1], static_cast<value_t>(0));
        EXPECT_EQ(t[1][0], static_cast<value_t>(3));
        EXPECT_EQ(t[1][1], static_cast<value_t>(2));
        EXPECT_EQ(t[2][0], static_cast<value_t>(4));
        EXPECT_EQ(t[2][1], static_cast<value_t>(5));
    }
}

/**
 * @test TypedTensor.sort_2D_axis1
 * @brief Sorting a 2D tensor along axis 1 (columns).
 */
TYPED_TEST(TypedTensor, sort_2D_axis1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(3), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(0),
        static_cast<value_t>(1), static_cast<value_t>(5)
    };
    t = vals;
    t.sort(1);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][2]), 5.0f);
    } else {
        EXPECT_EQ(t[0][0], static_cast<value_t>(1));
        EXPECT_EQ(t[0][1], static_cast<value_t>(2));
        EXPECT_EQ(t[0][2], static_cast<value_t>(3));
        EXPECT_EQ(t[1][0], static_cast<value_t>(0));
        EXPECT_EQ(t[1][1], static_cast<value_t>(1));
        EXPECT_EQ(t[1][2], static_cast<value_t>(5));
    }
}

/**
 * @test TypedTensor.sort_3D_axis2
 * @brief Sort the last axis of a 3D tensor: shape {2,2,4}.
 */
TYPED_TEST(TypedTensor, sort_3D_axis2)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2,4}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(4), static_cast<value_t>(1),
        static_cast<value_t>(3), static_cast<value_t>(2),

        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(5), static_cast<value_t>(2),

        static_cast<value_t>(9), static_cast<value_t>(7),
        static_cast<value_t>(8), static_cast<value_t>(6),

        static_cast<value_t>(3), static_cast<value_t>(3),
        static_cast<value_t>(1), static_cast<value_t>(2)
    };
    t = vals;
    t.sort(2);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][3]), 4.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][0]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][1]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][2]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][3]), 5.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][0]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][1]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][2]), 8.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][3]), 9.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][3]), 3.0f);
    } else {
        EXPECT_EQ(t[0][0][0], static_cast<value_t>(1));
        EXPECT_EQ(t[0][0][1], static_cast<value_t>(2));
        EXPECT_EQ(t[0][0][2], static_cast<value_t>(3));
        EXPECT_EQ(t[0][0][3], static_cast<value_t>(4));

        EXPECT_EQ(t[0][1][0], static_cast<value_t>(0));
        EXPECT_EQ(t[0][1][1], static_cast<value_t>(1));
        EXPECT_EQ(t[0][1][2], static_cast<value_t>(2));
        EXPECT_EQ(t[0][1][3], static_cast<value_t>(5));

        EXPECT_EQ(t[1][0][0], static_cast<value_t>(6));
        EXPECT_EQ(t[1][0][1], static_cast<value_t>(7));
        EXPECT_EQ(t[1][0][2], static_cast<value_t>(8));
        EXPECT_EQ(t[1][0][3], static_cast<value_t>(9));

        EXPECT_EQ(t[1][1][0], static_cast<value_t>(1));
        EXPECT_EQ(t[1][1][1], static_cast<value_t>(2));
        EXPECT_EQ(t[1][1][2], static_cast<value_t>(3));
        EXPECT_EQ(t[1][1][3], static_cast<value_t>(3));
    }
}

/**
 * @test TypedTensor.sort_3D_axis_negative
 * @brief Sort the last axis of a 3D tensor (through negative indexing).
 */
TYPED_TEST(TypedTensor, sort_3D_axis_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2,4}, MemoryLocation::HOST);

    std::vector<value_t> vals = {
        static_cast<value_t>(4), static_cast<value_t>(1),
        static_cast<value_t>(3), static_cast<value_t>(2),

        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(5), static_cast<value_t>(2),

        static_cast<value_t>(9), static_cast<value_t>(7),
        static_cast<value_t>(8), static_cast<value_t>(6),

        static_cast<value_t>(3), static_cast<value_t>(3),
        static_cast<value_t>(1), static_cast<value_t>(2)
    };
    t = vals;
    t.sort(-1);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][3]), 4.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][0]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][1]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][2]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][3]), 5.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][0]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][1]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][2]), 8.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][3]), 9.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][3]), 3.0f);
    } else {
        EXPECT_EQ(t[0][0][0], static_cast<value_t>(1));
        EXPECT_EQ(t[0][0][1], static_cast<value_t>(2));
        EXPECT_EQ(t[0][0][2], static_cast<value_t>(3));
        EXPECT_EQ(t[0][0][3], static_cast<value_t>(4));

        EXPECT_EQ(t[0][1][0], static_cast<value_t>(0));
        EXPECT_EQ(t[0][1][1], static_cast<value_t>(1));
        EXPECT_EQ(t[0][1][2], static_cast<value_t>(2));
        EXPECT_EQ(t[0][1][3], static_cast<value_t>(5));

        EXPECT_EQ(t[1][0][0], static_cast<value_t>(6));
        EXPECT_EQ(t[1][0][1], static_cast<value_t>(7));
        EXPECT_EQ(t[1][0][2], static_cast<value_t>(8));
        EXPECT_EQ(t[1][0][3], static_cast<value_t>(9));

        EXPECT_EQ(t[1][1][0], static_cast<value_t>(1));
        EXPECT_EQ(t[1][1][1], static_cast<value_t>(2));
        EXPECT_EQ(t[1][1][2], static_cast<value_t>(3));
        EXPECT_EQ(t[1][1][3], static_cast<value_t>(3));
    }
}

/**
 * @test TypedTensor.sort_3D_flatten
 * @brief Flatten-sort a 3D tensor (axis = -1) and confirm global ordering.
 */
TYPED_TEST(TypedTensor, sort_3D_flatten)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3,2}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(5), static_cast<value_t>(1),
        static_cast<value_t>(3), static_cast<value_t>(7),
        static_cast<value_t>(2), static_cast<value_t>(9),

        static_cast<value_t>(0), static_cast<value_t>(4),
        static_cast<value_t>(8), static_cast<value_t>(6),
        static_cast<value_t>(-1), static_cast<value_t>(10)
    };
    t = vals;
    t.sort();

    std::vector<value_t> got;
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            for (uint64_t k = 0; k < 2; ++k)
            {
                got.push_back(t[i][j][k]);
            }
        }
    }
    for (uint64_t i = 1; i < got.size(); ++i)
    {
        EXPECT_LE(got[i-1], got[i]);
    }
}

/**
 * @test TypedTensor.sort_with_nan
 * @brief NaNs should be placed last.
 */
TYPED_TEST(TypedTensor, sort_with_nan)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::HOST);
    std::vector<value_t> vals;
    if constexpr (std::is_floating_point_v<value_t>) {
        vals = {
            static_cast<value_t>(3.0f),
            static_cast<value_t>(NAN),
            static_cast<value_t>(1.0f),
            static_cast<value_t>(-2.0f),
            static_cast<value_t>(NAN)
        };
        t = vals;
        t.sort(0);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0]), -2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2]), 3.0f);
        EXPECT_TRUE(std::isnan(static_cast<float>(t[3])));
        EXPECT_TRUE(std::isnan(static_cast<float>(t[4])));
    } else {
        // Non-floating types cannot represent NaN; skip test.
    }
}

/**
 * @test TypedTensor.sort_with_inf
 * @brief -Inf should come first, +Inf after finite numbers.
 */
TYPED_TEST(TypedTensor, sort_with_inf)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::HOST);
    if constexpr (std::is_floating_point_v<value_t>) {
        std::vector<value_t> vals = {
            static_cast<value_t>(INFINITY),
            static_cast<value_t>(-1.0f),
            static_cast<value_t>(-INFINITY),
            static_cast<value_t>(5.0f),
            static_cast<value_t>(0.0f)
        };
        t = vals;
        t.sort(0);
        std::vector<value_t> expected = {
            static_cast<value_t>(-INFINITY),
            static_cast<value_t>(-1.0f),
            static_cast<value_t>(0.0f),
            static_cast<value_t>(5.0f),
            static_cast<value_t>(INFINITY)
        };
        for (uint64_t i = 0; i < 5; i++)
        {
            EXPECT_FLOAT_EQ(static_cast<float>(t[i]),
                            static_cast<float>(expected[i]));
        }
    } else {
        // Non-floating: infinities not representable; skip.
    }
}

/**
 * @test TypedTensor.sort_view_tensor
 * @brief Sorting a tensor view should only affect the view region.
 */
TYPED_TEST(TypedTensor, sort_view_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(4.0f),
        static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(1.0f)
    };
    t = vals;
    Tensor<value_t> v(t, {1}, {2});
    v.sort(0);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0]), 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[3]), 1.0f);
    } else {
        EXPECT_EQ(t[0], static_cast<value_t>(4));
        EXPECT_EQ(t[1], static_cast<value_t>(2));
        EXPECT_EQ(t[2], static_cast<value_t>(3));
        EXPECT_EQ(t[3], static_cast<value_t>(1));
    }
}

/**
 * @test TypedTensor.sort_view_tensor_2D_row
 * @brief Sorting a view of a row should only affect that row segment.
 */
TYPED_TEST(TypedTensor, sort_view_tensor_2D_row)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(4), static_cast<value_t>(3),
        static_cast<value_t>(2), static_cast<value_t>(1),

        static_cast<value_t>(8), static_cast<value_t>(7),
        static_cast<value_t>(6), static_cast<value_t>(5)
    };
    t = vals;

    Tensor<value_t> v(t, {0,1}, {1,2});
    v.sort(1);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0]), 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][3]), 1.0f);
    } else {
        EXPECT_EQ(t[0][0], static_cast<value_t>(4));
        EXPECT_EQ(t[0][1], static_cast<value_t>(2));
        EXPECT_EQ(t[0][2], static_cast<value_t>(3));
        EXPECT_EQ(t[0][3], static_cast<value_t>(1));
    }
    if constexpr (!std::is_floating_point_v<value_t>) {
        // second row unchanged for integer types too; assert explicitly
        EXPECT_EQ(t[1][0], static_cast<value_t>(8));
        EXPECT_EQ(t[1][1], static_cast<value_t>(7));
        EXPECT_EQ(t[1][2], static_cast<value_t>(6));
        EXPECT_EQ(t[1][3], static_cast<value_t>(5));
    } else {
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0]), 8.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][2]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][3]), 5.0f);
    }
}

/**
 * @test TypedTensor.sort_view_tensor_2D_col
 * @brief Sorting a view of a column should only affect that column segment.
 */
TYPED_TEST(TypedTensor, sort_view_tensor_2D_col)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(9), static_cast<value_t>(8),
        static_cast<value_t>(7), static_cast<value_t>(6),
        static_cast<value_t>(5), static_cast<value_t>(4),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(1)
    };
    t = vals;

    Tensor<value_t> v(t, {0,1}, {3,1});
    v.sort(0);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2][1]), 8.0f);
    } else {
        EXPECT_EQ(t[0][1], static_cast<value_t>(2));
        EXPECT_EQ(t[1][1], static_cast<value_t>(5));
        EXPECT_EQ(t[2][1], static_cast<value_t>(8));
    }
}

/**
 * @test TypedTensor.sort_view_tensor_3D_subcube
 * @brief Sorting a subcube view in 3D only changes inside that region.
 */
TYPED_TEST(TypedTensor, sort_view_tensor_3D_subcube)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(9), static_cast<value_t>(7),
        static_cast<value_t>(8), static_cast<value_t>(6),

        static_cast<value_t>(5), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(2),

        static_cast<value_t>(1), static_cast<value_t>(-1),
        static_cast<value_t>(0), static_cast<value_t>(-2),

        static_cast<value_t>(10), static_cast<value_t>(12),
        static_cast<value_t>(11), static_cast<value_t>(13)
    };
    t = vals;

    Tensor<value_t> v(t, {0,0,0}, {1,2,4});
    v.sort(2);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][0]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0][3]), 9.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][0]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1][3]), 5.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1][0]), 10.0f);
    } else {
        EXPECT_EQ(t[0][0][0], static_cast<value_t>(6));
        EXPECT_EQ(t[0][0][3], static_cast<value_t>(9));
        EXPECT_EQ(t[0][1][0], static_cast<value_t>(2));
        EXPECT_EQ(t[0][1][3], static_cast<value_t>(5));

        EXPECT_EQ(t[1][0][0], static_cast<value_t>(1));
        EXPECT_EQ(t[1][1][0], static_cast<value_t>(10));
    }
}

/**
 * @test TypedTensor.sort_view_non1D_flatten
 * @brief Sorting a non-1D view with axis = -1
 * should only flatten-sort the view region.
 */
TYPED_TEST(TypedTensor, sort_view_non1D_flatten)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3),

        static_cast<value_t>(13), static_cast<value_t>(14),
        static_cast<value_t>(15),

        static_cast<value_t>(7), static_cast<value_t>(8),
        static_cast<value_t>(9),

        static_cast<value_t>(10), static_cast<value_t>(11),
        static_cast<value_t>(12),

        static_cast<value_t>(16), static_cast<value_t>(17),
        static_cast<value_t>(18),

        static_cast<value_t>(4), static_cast<value_t>(5),
        static_cast<value_t>(6)
    };
    t = vals;

    Tensor<value_t> v(t, {0,1,0}, {2,2,3});
    v.sort();

    std::vector<value_t> got;
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 1; j < 3; ++j)
        {
            for (uint64_t k = 0; k < 3; ++k)
            {
                got.push_back(t[i][j][k]);
            }
        }
    }
    ASSERT_EQ(got.size(), 12u);
    for (uint64_t idx = 1; idx < got.size(); ++idx)
    {
        EXPECT_LE(got[idx-1], got[idx]);
    }

    // Ensure elements outside the view (j == 0) are unchanged.
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 1; ++j)
        {
            for (uint64_t k = 0; k < 3; ++k)
            {
                uint64_t flat = i * (3 * 3) + j * 3 + k;
                if constexpr (std::is_floating_point_v<value_t>) {
                    EXPECT_FLOAT_EQ(static_cast<float>(t[i][j][k]),
                                    static_cast<float>(vals[flat]));
                } else {
                    EXPECT_EQ(t[i][j][k], vals[flat]);
                }
            }
        }
    }
}

/**
 * @test TypedTensor.sort_alias_view_basic_1D
 * @brief Sorting a 1D alias view (contiguous) should only reorder
 * the view region in the owner.
 */
TYPED_TEST(TypedTensor, sort_alias_view_basic_1D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({6}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(6.0f),
        static_cast<value_t>(5.0f),
        static_cast<value_t>(4.0f),
        static_cast<value_t>(3.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(1.0f)
    };
    t = vals;

    Tensor<value_t> v(t, {1}, {3}, {1});
    v.sort(0);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2]), 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[3]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[4]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[5]), 1.0f);
    } else {
        EXPECT_EQ(t[0], static_cast<value_t>(6));
        EXPECT_EQ(t[1], static_cast<value_t>(3));
        EXPECT_EQ(t[2], static_cast<value_t>(4));
        EXPECT_EQ(t[3], static_cast<value_t>(5));
        EXPECT_EQ(t[4], static_cast<value_t>(2));
        EXPECT_EQ(t[5], static_cast<value_t>(1));
    }
}

/**
 * @test TypedTensor.sort_alias_view_noncontiguous_stride
 * @brief Sorting a non-contiguous 1D alias (stride > 1)
 * updates only the sampled elements.
 */
TYPED_TEST(TypedTensor, sort_alias_view_noncontiguous_stride)
{
    using value_t = TypeParam;
    Tensor<value_t> t({8}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(10.f), static_cast<value_t>(0.f),
        static_cast<value_t>(9.f), static_cast<value_t>(0.f),
        static_cast<value_t>(8.f), static_cast<value_t>(0.f),
        static_cast<value_t>(7.f), static_cast<value_t>(0.f)
    };
    t = vals;

    Tensor<value_t> v(t, {0}, {4}, {2});
    v.sort(0);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2]), 8.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[3]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[4]), 9.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[5]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[6]), 10.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[7]), 0.0f);
    } else {
        EXPECT_EQ(t[0], static_cast<value_t>(7));
        EXPECT_EQ(t[1], static_cast<value_t>(0));
        EXPECT_EQ(t[2], static_cast<value_t>(8));
        EXPECT_EQ(t[3], static_cast<value_t>(0));
        EXPECT_EQ(t[4], static_cast<value_t>(9));
        EXPECT_EQ(t[5], static_cast<value_t>(0));
        EXPECT_EQ(t[6], static_cast<value_t>(10));
        EXPECT_EQ(t[7], static_cast<value_t>(0));
    }
}

/**
 * @test TypedTensor.sort_alias_view_2D_submatrix_axis1
 * @brief Sorting a 2x3 submatrix view along its
 * last axis only sorts that submatrix.
 */
TYPED_TEST(TypedTensor, sort_alias_view_2D_submatrix_axis1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(9), static_cast<value_t>(8),
        static_cast<value_t>(7), static_cast<value_t>(6),

        static_cast<value_t>(5), static_cast<value_t>(4),
        static_cast<value_t>(3), static_cast<value_t>(2),

        static_cast<value_t>(1), static_cast<value_t>(0),
        static_cast<value_t>(-1), static_cast<value_t>(-2)
    };
    t = vals;

    Tensor<value_t> sub(t, {0,1}, {2,3}, {4,1});
    sub.sort(1);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0]), 9.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][2]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][3]), 8.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][3]), 4.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[2][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2][1]), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2][2]), -1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[2][3]), -2.0f);
    } else {
        EXPECT_EQ(t[0][0], static_cast<value_t>(9));
        EXPECT_EQ(t[0][1], static_cast<value_t>(6));
        EXPECT_EQ(t[0][2], static_cast<value_t>(7));
        EXPECT_EQ(t[0][3], static_cast<value_t>(8));

        EXPECT_EQ(t[1][0], static_cast<value_t>(5));
        EXPECT_EQ(t[1][1], static_cast<value_t>(2));
        EXPECT_EQ(t[1][2], static_cast<value_t>(3));
        EXPECT_EQ(t[1][3], static_cast<value_t>(4));

        EXPECT_EQ(t[2][0], static_cast<value_t>(1));
        EXPECT_EQ(t[2][1], static_cast<value_t>(0));
        EXPECT_EQ(t[2][2], static_cast<value_t>(-1));
        EXPECT_EQ(t[2][3], static_cast<value_t>(-2));
    }
}

/**
 * @test TypedTensor.sort_alias_view_flatten_subregion
 * @brief Flatten-sorting (axis = -1) a submatrix view
 * only orders elements within the view.
 */
TYPED_TEST(TypedTensor, sort_alias_view_flatten_subregion)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(5), static_cast<value_t>(1),
        static_cast<value_t>(3),

        static_cast<value_t>(4), static_cast<value_t>(2),
        static_cast<value_t>(0)
    };
    t = vals;

    Tensor<value_t> sub(t, {0,0}, {2,2}, {3,1});
    sub.sort();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[0][2]), 3.0f);

        EXPECT_FLOAT_EQ(static_cast<float>(t[1][0]), 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][1]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1][2]), 0.0f);
    } else {
        EXPECT_EQ(t[0][0], static_cast<value_t>(1));
        EXPECT_EQ(t[0][1], static_cast<value_t>(2));
        EXPECT_EQ(t[0][2], static_cast<value_t>(3));

        EXPECT_EQ(t[1][0], static_cast<value_t>(4));
        EXPECT_EQ(t[1][1], static_cast<value_t>(5));
        EXPECT_EQ(t[1][2], static_cast<value_t>(0));
    }
}

/**
 * @test TypedTensor.sort_alias_view_weird_strides
 * @brief Sorting an alias view with non-trivial strides (e.g. 13,4).
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Sort along axis 1.
 */
TYPED_TEST(TypedTensor, sort_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5,20}, MemoryLocation::HOST);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[99 - i] = static_cast<value_t>(i);
    }
    owner = vals;

    Tensor<value_t> view(owner, {0,0}, {3,4}, {13,4});
    view.sort(1);

    EXPECT_EQ(view.m_dimensions, (std::vector<uint64_t>{3,4}));
    EXPECT_EQ(view.m_strides, (std::vector<uint64_t>{13,4}));

    Tensor<value_t> host({3,4}, MemoryLocation::HOST);
    copy_tensor_data(host, view);
    std::vector<value_t> out(12);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 12).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(87.f),
        static_cast<value_t>(91.f),
        static_cast<value_t>(95.f),
        static_cast<value_t>(99.f),

        static_cast<value_t>(74.f),
        static_cast<value_t>(78.f),
        static_cast<value_t>(82.f),
        static_cast<value_t>(86.f),

        static_cast<value_t>(61.f),
        static_cast<value_t>(65.f),
        static_cast<value_t>(69.f),
        static_cast<value_t>(73.f)
    };

    for (uint64_t k = 0; k < out.size(); ++k)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[k]),
                            static_cast<float>(expected[k]));
        } else {
            EXPECT_EQ(out[k], expected[k]);
        }
    }
}

/**
 * @test TypedTensor.sort_alias_view_broadcast_noop
 * @brief Sorting a broadcasted view (stride 0) is a no-op
 * in terms of data reordering, but must not crash; owner must remain consistent.
 */
TYPED_TEST(TypedTensor, sort_alias_view_broadcast_noop)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2}, MemoryLocation::HOST);
    t = std::vector<value_t>{ static_cast<value_t>(42.0f),
                              static_cast<value_t>(99.0f) };

    Tensor<value_t> b(t, {1}, {4}, {0});
    EXPECT_NO_THROW(b.sort(0));

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t[0]), 42.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t[1]), 99.0f);
    } else {
        EXPECT_EQ(t[0], static_cast<value_t>(42));
        EXPECT_EQ(t[1], static_cast<value_t>(99));
    }
}

/**
 * @test TypedTensor.sort_idempotence
 * @brief Sorting twice should give same result.
 */
TYPED_TEST(TypedTensor, sort_idempotence)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(4), static_cast<value_t>(1),
        static_cast<value_t>(3), static_cast<value_t>(2),
        static_cast<value_t>(0)
    };
    t = vals;
    t.sort(0);
    std::vector<value_t> once(5);
    for (uint64_t i = 0; i < 5; i++)
    {
        once[i] = t[i];
    }
    t.sort(0);
    for (uint64_t i = 0; i < 5; i++)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(t[i]),
                            static_cast<float>(once[i]));
        } else {
            EXPECT_EQ(t[i], once[i]);
        }
    }
}

/**
 * @test TypedTensor.sum_all_elements
 * @brief Sum all elements (axis = -1) on a device tensor and return
 * a scalar with the correct total value.
 */
TYPED_TEST(TypedTensor, sum_all_elements)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    t = vals;

    Tensor<value_t> res = t.sum();

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 6.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(6));
    }
}

/**
 * @test TypedTensor.sum_axis0
 * @brief Sum along axis 0 for a 2x3 tensor stored on device
 * nd verify per-column sums.
 */
TYPED_TEST(TypedTensor, sum_axis0)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    Tensor<value_t> res = t.sum(0);

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       3 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        1.0f + 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        2.0f + 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]),
                        3.0f + 6.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 4));
        EXPECT_EQ(host[1], static_cast<value_t>(2 + 5));
        EXPECT_EQ(host[2], static_cast<value_t>(3 + 6));
    }
}

/**
 * @test TypedTensor.sum_axis1
 * @brief Sum along axis 1 for a 2x3 device tensor and verify per-row sums.
 */
TYPED_TEST(TypedTensor, sum_axis1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    Tensor<value_t> res = t.sum(1);

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       2 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        1.0f + 2.0f + 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]),
                        4.0f + 5.0f + 6.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 2 + 3));
        EXPECT_EQ(host[1], static_cast<value_t>(4 + 5 + 6));
    }
}

/**
 * @test TypedTensor.sum_axis0_3D
 * @brief Sum along axis 0 for a 2x2x2 device tensor and verify resulting values.
 */
TYPED_TEST(TypedTensor, sum_axis0_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f),
        static_cast<value_t>(7.0f), static_cast<value_t>(8.0f)
    };
    t = vals;

    Tensor<value_t> res = t.sum(0);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       4 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f + 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 2.0f + 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 3.0f + 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 4.0f + 8.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 5));
        EXPECT_EQ(host[1], static_cast<value_t>(2 + 6));
        EXPECT_EQ(host[2], static_cast<value_t>(3 + 7));
        EXPECT_EQ(host[3], static_cast<value_t>(4 + 8));
    }
}

/**
 * @test TypedTensor.sum_axis_negative
 * @brief Sum along axis -3 for a 2x2x2 device tensor and verify resulting values.
 */
TYPED_TEST(TypedTensor, sum_axis_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f),
        static_cast<value_t>(7.0f), static_cast<value_t>(8.0f)
    };
    t = vals;

    Tensor<value_t> res = t.sum(-3);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       4 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f + 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 2.0f + 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 3.0f + 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 4.0f + 8.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 5));
        EXPECT_EQ(host[1], static_cast<value_t>(2 + 6));
        EXPECT_EQ(host[2], static_cast<value_t>(3 + 7));
        EXPECT_EQ(host[3], static_cast<value_t>(4 + 8));
    }
}

/**
 * @test TypedTensor.sum_axis1_3D
 * @brief Sum along axis 1 for a 2x2x2 device tensor and verify resulting values.
 */
TYPED_TEST(TypedTensor, sum_axis1_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f),
        static_cast<value_t>(7.0f), static_cast<value_t>(8.0f)
    };
    t = vals;

    Tensor<value_t> res = t.sum(1);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       4 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f + 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 2.0f + 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 5.0f + 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 6.0f + 8.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 3));
        EXPECT_EQ(host[1], static_cast<value_t>(2 + 4));
        EXPECT_EQ(host[2], static_cast<value_t>(5 + 7));
        EXPECT_EQ(host[3], static_cast<value_t>(6 + 8));
    }
}

/**
 * @test TypedTensor.sum_axis2_3D
 * @brief Sum along axis 2 for a 2x2x2 device tensor and verify resulting values.
 */
TYPED_TEST(TypedTensor, sum_axis2_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f),
        static_cast<value_t>(7.0f), static_cast<value_t>(8.0f)
    };
    t = vals;

    Tensor<value_t> res = t.sum(2);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       4 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f + 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 3.0f + 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 5.0f + 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 7.0f + 8.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 2));
        EXPECT_EQ(host[1], static_cast<value_t>(3 + 4));
        EXPECT_EQ(host[2], static_cast<value_t>(5 + 6));
        EXPECT_EQ(host[3], static_cast<value_t>(7 + 8));
    }
}

/**
 * @test TypedTensor.sum_view_tensor
 * @brief Sum all elements (axis = -1) of a view into a device tensor and
 * verify the scalar result.
 */
TYPED_TEST(TypedTensor, sum_view_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 0ull};
    std::vector<uint64_t> view_shape = {3ull};
    Tensor<value_t> view(t, start_indices, view_shape);

    Tensor<value_t> res = view.sum();

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        1.0f + 2.0f + 3.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 2 + 3));
    }
}

/**
 * @test TypedTensor.sum_alias_view_tensor
 * @brief Sum all elements (axis = -1) of an alias view
 * with non-unit stride and verify result.
 */
TYPED_TEST(TypedTensor, sum_alias_view_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> t({6}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    std::vector<uint64_t> start_indices = {0ull};
    std::vector<uint64_t> dims = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<value_t> alias_view(t, start_indices, dims, strides);

    Tensor<value_t> res = alias_view.sum();

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]),
                        1.0f + 3.0f + 5.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1 + 3 + 5));
    }
}

/**
 * @test TypedTensor.sum_view_tensor_3d_axis1
 * @brief Sum along axis 1 on a 3D view and verify the produced values.
 */
TYPED_TEST(TypedTensor, sum_view_tensor_3d_axis1)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 4, 2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 1ull, 0ull};
    std::vector<uint64_t> view_shape    = {2ull, 2ull};
    Tensor<value_t> view(t, start_indices, view_shape);

    Tensor<value_t> res = view.sum(1);

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t) * host.size()).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 23.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 27.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(23));
        EXPECT_EQ(host[1], static_cast<value_t>(27));
    }
}

/**
 * @test TypedTensor.sum_alias_view_tensor_2d_strided
 * @brief Sum along axis 0 on a 2D alias view with custom strides and verify
 * each output element.
 */
TYPED_TEST(TypedTensor, sum_alias_view_tensor_2d_strided)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 5}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(20);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 1ull};
    std::vector<uint64_t> dims = {2ull, 3ull};
    std::vector<uint64_t> strides = {5ull, 2ull};
    Tensor<value_t> alias_view(t, start_indices, dims, strides);

    Tensor<value_t> res = alias_view.sum(0);

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t) * host.size()).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 9.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 13.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 17.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(9));
        EXPECT_EQ(host[1], static_cast<value_t>(13));
        EXPECT_EQ(host[2], static_cast<value_t>(17));
    }
}

/**
 * @test TypedTensor.sum_alias_view_tensor_overlapping_stride_zero
 * @brief Sum along axis 0 on an alias view that contains overlapping elements
 * via a zero stride and verify the sums account for repeated elements.
 */
TYPED_TEST(TypedTensor, sum_alias_view_tensor_overlapping_stride_zero)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 0ull};
    std::vector<uint64_t> dims = {2ull, 2ull};
    std::vector<uint64_t> strides = {0ull, 1ull};
    Tensor<value_t> alias_view(t, start_indices, dims, strides);

    Tensor<value_t> res = alias_view.sum(0);

    std::vector<value_t> host(2);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t) * host.size()).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 8.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 10.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(8));
        EXPECT_EQ(host[1], static_cast<value_t>(10));
    }
}

/**
 * @test TypedTensor.sum_nan_throws
 * @brief Tests that sum throws temper::nan_error when the tensor
 * contains NaN values, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, sum_nan_throws)
{
    using value_t = TypeParam;
    if constexpr (std::is_floating_point_v<value_t>) {
        Tensor<value_t> t({3}, MemoryLocation::DEVICE);
        std::vector<value_t> vals = {
            static_cast<value_t>(1.0f),
            static_cast<value_t>(std::numeric_limits<float>::quiet_NaN()),
            static_cast<value_t>(3.0f)
        };
        t = vals;
        EXPECT_THROW(t.sum(), temper::nan_error);
    } else {
        // Non-floating types cannot contain NaN; skip.
    }
}

/**
 * @test TypedTensor.sum_non_finite_throws
 * @brief Non-finite sum result should trigger a
 * temper::nonfinite_error, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, sum_non_finite_throws)
{
    using value_t = TypeParam;
    if constexpr (std::is_floating_point_v<value_t>) {
        Tensor<value_t> t({2}, MemoryLocation::DEVICE);
        std::vector<value_t> vals = {
            static_cast<value_t>(std::numeric_limits<float>::infinity()),
            static_cast<value_t>(1.0f)
        };
        t = vals;
        EXPECT_THROW(t.sum(), temper::nonfinite_error);
    }
    else
    {
        // Non-floating: skip.
    }
}

/**
 * @test TypedTensor.sum_empty
 * @brief Summing an empty tensor returns a scalar tensor containing 0.0.
 */
TYPED_TEST(TypedTensor, sum_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;

    Tensor<value_t> res({1}, MemoryLocation::DEVICE);
    res = t.sum();

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 0.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(0));
    }
}

/**
 * @test TypedTensor.cumsum_all_elements_flatten
 * @brief Tests cumsum on a 1D tensor, flattening all elements.
 */
TYPED_TEST(TypedTensor, cumsum_all_elements_flatten)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f),
        static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f)
    };
    t = vals;

    Tensor<value_t> res = t.cumsum(-1);

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       3 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 6.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(1 + 2));
        EXPECT_EQ(host[2], static_cast<value_t>(1 + 2 + 3));
    }
}

/**
 * @test TypedTensor.cumsum_axis0_2D
 * @brief Tests cumsum along axis 0 of a 2D tensor.
 */
TYPED_TEST(TypedTensor, cumsum_axis0_2D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    Tensor<value_t> res = t.cumsum(0);

    std::vector<value_t> host(6);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       6 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[4]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[5]), 9.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(2));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(1 + 4));
        EXPECT_EQ(host[4], static_cast<value_t>(2 + 5));
        EXPECT_EQ(host[5], static_cast<value_t>(3 + 6));
    }
}

/**
 * @test TypedTensor.cumsum_axis_negative
 * @brief Tests cumsum along axis -2 of a 2D tensor.
 */
TYPED_TEST(TypedTensor, cumsum_axis_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    Tensor<value_t> res = t.cumsum(-2);

    std::vector<value_t> host(6);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       6 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[4]), 7.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[5]), 9.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(2));
        EXPECT_EQ(host[2], static_cast<value_t>(3));
        EXPECT_EQ(host[3], static_cast<value_t>(1 + 4));
        EXPECT_EQ(host[4], static_cast<value_t>(2 + 5));
        EXPECT_EQ(host[5], static_cast<value_t>(3 + 6));
    }
}

/**
 * @test TypedTensor.cumsum_axis1_2D
 * @brief Tests cumsum along axis 1 of a 2D tensor.
 */
TYPED_TEST(TypedTensor, cumsum_axis1_2D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    Tensor<value_t> res = t.cumsum(1);

    std::vector<value_t> host(6);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       6 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[4]), 9.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[5]), 15.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(1 + 2));
        EXPECT_EQ(host[2], static_cast<value_t>(1 + 2 + 3));
        EXPECT_EQ(host[3], static_cast<value_t>(4));
        EXPECT_EQ(host[4], static_cast<value_t>(4 + 5));
        EXPECT_EQ(host[5], static_cast<value_t>(4 + 5 + 6));
    }
}

/**
 * @test TypedTensor.cumsum_flatten_3D
 * @brief Tests cumsum on a 3D tensor flattened.
 */
TYPED_TEST(TypedTensor, cumsum_flatten_3D)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2,2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f),
        static_cast<value_t>(7.0f), static_cast<value_t>(8.0f)
    };
    t = vals;
    Tensor<value_t> res = t.cumsum();

    std::vector<value_t> host(8);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       8 * sizeof(value_t)).wait();

    std::vector<float> expected = {
        1.0f,
        1.0f + 2.0f,
        1.0f + 2.0f + 3.0f,
        1.0f + 2.0f + 3.0f + 4.0f,
        1.0f + 2.0f + 3.0f + 4.0f + 5.0f,
        1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f,
        1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f + 7.0f,
        1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f + 7.0f + 8.0f
    };

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(host[i]),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(host[i], static_cast<value_t>(expected[i]));
        }
    }
}

/**
 * @test TypedTensor.cumsum_view_flatten
 * @brief Tests cumsum on a view of a 3D tensor flattened along the last axis.
 */
TYPED_TEST(TypedTensor, cumsum_view_flatten)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3,4,2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    t = vals;

    std::vector<uint64_t> start = {1ull, 1ull, 0ull};
    std::vector<uint64_t> view_shape = {2ull, 2ull};
    Tensor<value_t> view(t, start, view_shape);

    Tensor<value_t> res = view.cumsum();

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       4 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 11.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 23.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 36.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 50.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(11));
        EXPECT_EQ(host[1], static_cast<value_t>(23));
        EXPECT_EQ(host[2], static_cast<value_t>(36));
        EXPECT_EQ(host[3], static_cast<value_t>(50));
    }
}

/**
 * @test TypedTensor.cumsum_alias_view_strided
 * @brief Tests cumsum on an alias view with a stride on a 1D tensor.
 */
TYPED_TEST(TypedTensor, cumsum_alias_view_strided)
{
    using value_t = TypeParam;
    Tensor<value_t> t({6}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;
    std::vector<uint64_t> start = {0ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<value_t> alias_view(t, start, dims, strides);

    Tensor<value_t> res = alias_view.cumsum();

    std::vector<value_t> host(3);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       3 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 1.0f + 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 1.0f + 3.0f + 5.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(1));
        EXPECT_EQ(host[1], static_cast<value_t>(1 + 3));
        EXPECT_EQ(host[2], static_cast<value_t>(1 + 3 + 5));
    }
}

/**
 * @test TypedTensor.cumsum_alias_view_overlapping_stride_zero
 * @brief Tests cumsum on an alias view with
 * overlapping stride of zero on a 2D tensor.
 */
TYPED_TEST(TypedTensor, cumsum_alias_view_overlapping_stride_zero)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f),
        static_cast<value_t>(5.0f), static_cast<value_t>(6.0f)
    };
    t = vals;

    std::vector<uint64_t> start   = {1ull, 0ull};
    std::vector<uint64_t> dims    = {2ull, 2ull};
    std::vector<uint64_t> strides = {0ull, 1ull};
    Tensor<value_t> alias_view(t, start, dims, strides);

    Tensor<value_t> res = alias_view.cumsum(0);

    std::vector<value_t> host(4);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       4 * sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 4.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[1]), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[2]), 8.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(host[3]), 10.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(4));
        EXPECT_EQ(host[1], static_cast<value_t>(5));
        EXPECT_EQ(host[2], static_cast<value_t>(8));
        EXPECT_EQ(host[3], static_cast<value_t>(10));
    }
}

/**
 * @test TypedTensor.cumsum_alias_view_weird_strides
 * @brief Sorting an alias view with non-trivial strides (e.g. 13,4).
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Cumsum along axis 1.
 */
TYPED_TEST(TypedTensor, cumsum_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5,20}, MemoryLocation::HOST);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    owner = vals;

    Tensor<value_t> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<value_t> view2 = view.cumsum(1);

    EXPECT_EQ(view2.m_dimensions, (std::vector<uint64_t>{3,4}));
    EXPECT_EQ(view2.m_strides, (std::vector<uint64_t>{4,1}));

    Tensor<value_t> host({3,4}, MemoryLocation::HOST);
    copy_tensor_data(host, view2);

    std::vector<value_t> out(12);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
                       sizeof(value_t) * 12).wait();

    std::vector<value_t> expected = {
        static_cast<value_t>(0.f), static_cast<value_t>(4.f),
        static_cast<value_t>(12.f), static_cast<value_t>(24.f),

        static_cast<value_t>(13.f), static_cast<value_t>(30.f),
        static_cast<value_t>(51.f), static_cast<value_t>(76.f),

        static_cast<value_t>(26.f), static_cast<value_t>(56.f),
        static_cast<value_t>(90.f), static_cast<value_t>(128.f)
    };

    for (uint64_t k = 0; k < out.size(); ++k)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[k]),
                            static_cast<float>(expected[k]));
        } else {
            EXPECT_EQ(out[k], expected[k]);
        }
    }
}

/**
 * @test TypedTensor.cumsum_axis_out_of_bounds
 * @brief Tests that cumsum throws temper::validation_error
 * when the axis is out of bounds.
 */
TYPED_TEST(TypedTensor, cumsum_axis_out_of_bounds)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    t = vals;

    EXPECT_THROW(t.cumsum(2), temper::bounds_error);
    EXPECT_THROW(t.cumsum(-3), temper::bounds_error);
}

/**
 * @test TypedTensor.cumsum_nan_throws
 * @brief Tests that cumsum throws temper::nan_error when the tensor
 * contains NaN values, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, cumsum_nan_throws)
{
    using value_t = TypeParam;
    if constexpr (std::is_floating_point_v<value_t>) {
        Tensor<value_t> t({3}, MemoryLocation::DEVICE);
        std::vector<value_t> vals = {
            static_cast<value_t>(1.0f),
            static_cast<value_t>(std::numeric_limits<float>::quiet_NaN()),
            static_cast<value_t>(3.0f)
        };
        t = vals;
        EXPECT_THROW(t.cumsum(), temper::nan_error);
    } else {
        // Skip for non-floating.
    }
}

/**
 * @test TypedTensor.cumsum_non_finite_throws
 * @brief Non-finite cumsumsum result should trigger a
 * temper::nonfinite_error, as specified in the error handling policy.
 */
TYPED_TEST(TypedTensor, cumsum_non_finite_throws)
{
    using value_t = TypeParam;
    if constexpr (std::is_floating_point_v<value_t>) {
        Tensor<value_t> t({2}, MemoryLocation::DEVICE);
        std::vector<value_t> vals = {
            static_cast<value_t>(std::numeric_limits<float>::infinity()),
            static_cast<value_t>(1.0f)
        };
        t = vals;
        EXPECT_THROW(t.cumsum(), temper::nonfinite_error);
    }
    else
    {
        // Skip for non-floating.
    }
}

/**
 * @test TypedTensor.cumsum_empty
 * @brief Tests cumsum on an empty tensor returns a tensor
 * with a single zero element.
 */
TYPED_TEST(TypedTensor, cumsum_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;

    Tensor<value_t> res({1}, MemoryLocation::DEVICE);
    res = t.cumsum();

    std::vector<value_t> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(),
                       sizeof(value_t)).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(host[0]), 0.0f);
    } else {
        EXPECT_EQ(host[0], static_cast<value_t>(0));
    }
}

/**
 * @test TypedTensor.transpose_noargs_reverse_axes
 * @brief Tests that transpose() with no arguments reverses all axes.
 */
TYPED_TEST(TypedTensor, transpose_noargs_reverse_axes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    t = vals;

    Tensor<value_t> t_rev = t.transpose();

    EXPECT_EQ(t_rev.m_dimensions,
              (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(t_rev.m_strides,
              (std::vector<uint64_t>{
                  t.m_strides[2], t.m_strides[1], t.m_strides[0]
              }));

    Tensor<value_t> host({4, 3, 2}, MemoryLocation::HOST);
    copy_tensor_data(host, t_rev);

    std::vector<value_t> out(24);
    g_sycl_queue.memcpy
    (out.data(), host.m_p_data.get(), sizeof(value_t) * 24).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]),
                        static_cast<float>(vals[0]));
        EXPECT_FLOAT_EQ(static_cast<float>(out[23]),
                        static_cast<float>(vals[23]));
    } else {
        EXPECT_EQ(out[0], vals[0]);
        EXPECT_EQ(out[23], vals[23]);
    }
}

/**
 * @test TypedTensor.transpose_explicit_axes
 * @brief Tests transpose with explicit axis permutation.
 */
TYPED_TEST(TypedTensor, transpose_explicit_axes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    t = vals;

    Tensor<value_t> perm = t.transpose({2, 1, 0});

    EXPECT_EQ(perm.m_dimensions,
              (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(perm.m_strides,
              (std::vector<uint64_t>{
                  t.m_strides[2], t.m_strides[1], t.m_strides[0]
              }));

    Tensor<value_t> host({4, 3, 2}, MemoryLocation::HOST);
    copy_tensor_data(host, perm);
    std::vector<value_t> out(24);
    g_sycl_queue.memcpy
    (out.data(), host.m_p_data.get(), sizeof(value_t) * 24).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]),
                        static_cast<float>(vals[0]));
        EXPECT_FLOAT_EQ(static_cast<float>(out[23]),
                        static_cast<float>(vals[23]));
    } else {
        EXPECT_EQ(out[0], vals[0]);
        EXPECT_EQ(out[23], vals[23]);
    }
}

/**
 * @test TypedTensor.transpose_explicit_axes_negative
 * @brief Tests transpose with explicit negative axis permutation.
 */
TYPED_TEST(TypedTensor, transpose_explicit_axes_negative)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    t = vals;

    Tensor<value_t> perm = t.transpose({-1, -2, -3});

    EXPECT_EQ(perm.m_dimensions,
              (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(perm.m_strides,
              (std::vector<uint64_t>{
                  t.m_strides[2], t.m_strides[1], t.m_strides[0]
              }));

    Tensor<value_t> host({4, 3, 2}, MemoryLocation::HOST);
    copy_tensor_data(host, perm);
    std::vector<value_t> out(24);
    g_sycl_queue.memcpy
    (out.data(), host.m_p_data.get(), sizeof(value_t) * 24).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]),
                        static_cast<float>(vals[0]));
        EXPECT_FLOAT_EQ(static_cast<float>(out[23]),
                        static_cast<float>(vals[23]));
    } else {
        EXPECT_EQ(out[0], vals[0]);
        EXPECT_EQ(out[23], vals[23]);
    }
}

/**
 * @test TypedTensor.transpose_2d
 * @brief Tests transpose on a 2D tensor (matrix).
 */
TYPED_TEST(TypedTensor, transpose_2d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    t = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };

    Tensor<value_t> t_T = t.transpose();

    EXPECT_EQ(t_T.m_dimensions,
              (std::vector<uint64_t>{3, 2}));
    EXPECT_EQ(t_T.m_strides,
              (std::vector<uint64_t>{t.m_strides[1], t.m_strides[0]}));

    Tensor<value_t> host({3,2}, MemoryLocation::HOST);
    copy_tensor_data(host, t_T);
    std::vector<value_t> out(6);
    g_sycl_queue.memcpy
    (out.data(), host.m_p_data.get(), sizeof(value_t) * 6).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 1.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[1]), 4.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[2]), 2.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[3]), 5.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[4]), 3.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[5]), 6.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(1));
        EXPECT_EQ(out[1], static_cast<value_t>(4));
        EXPECT_EQ(out[2], static_cast<value_t>(2));
        EXPECT_EQ(out[3], static_cast<value_t>(5));
        EXPECT_EQ(out[4], static_cast<value_t>(3));
        EXPECT_EQ(out[5], static_cast<value_t>(6));
    }
}

/**
 * @test TypedTensor.transpose_mutation_reflects
 * @brief Ensure that modifying the transposed alias updates the original.
 */
TYPED_TEST(TypedTensor, transpose_mutation_reflects)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    t = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4), static_cast<value_t>(5)
    };

    Tensor<value_t> t_T = t.transpose();
    t_T[0][0] = static_cast<value_t>(100.f);
    t_T[2][1] = static_cast<value_t>(200.f);

    Tensor<value_t> host({2,3}, MemoryLocation::HOST);
    copy_tensor_data(host, t);
    std::vector<value_t> out(6);
    g_sycl_queue.memcpy
    (out.data(), host.m_p_data.get(), sizeof(value_t) * 6).wait();

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[0]), 100.f);
        EXPECT_FLOAT_EQ(static_cast<float>(out[5]), 200.f);
    } else {
        EXPECT_EQ(out[0], static_cast<value_t>(100));
        EXPECT_EQ(out[5], static_cast<value_t>(200));
    }
}

/**
 * @test TypedTensor.transpose_invalid_axes
 * @brief Transpose throws when axes permutation is invalid.
 */
TYPED_TEST(TypedTensor, transpose_invalid_axes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3,4}, MemoryLocation::HOST);
    t = std::vector<value_t>(24, static_cast<value_t>(1));

    EXPECT_THROW(t.transpose({0,1,1}), temper::bounds_error);

    EXPECT_THROW(t.transpose({0,1,3}), temper::bounds_error);
}

/**
 * @test TypedTensor.transpose_1d
 * @brief Transpose a 1D tensor should return a 1D alias (no change).
 */
TYPED_TEST(TypedTensor, transpose_1d)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5}, MemoryLocation::HOST);
    t = {
        static_cast<value_t>(0), static_cast<value_t>(1),
        static_cast<value_t>(2), static_cast<value_t>(3),
        static_cast<value_t>(4)
    };

    Tensor<value_t> t_tr = t.transpose();
    EXPECT_EQ(t_tr.m_dimensions, t.m_dimensions);
    EXPECT_EQ(t_tr.m_strides, t.m_strides);

    Tensor<value_t> host({5}, MemoryLocation::HOST);
    copy_tensor_data(host, t_tr);
    std::vector<value_t> out(5);
    g_sycl_queue.memcpy
    (out.data(), host.m_p_data.get(), sizeof(value_t) * 5).wait();

    for (uint64_t i = 0; i < 5; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(out[i]),
                            static_cast<float>(i));
        } else {
            EXPECT_EQ(out[i], static_cast<value_t>(i));
        }
    }
}

/**
 * @test TypedTensor.transpose_empty
 * @brief Transpose of an empty tensor throws.
 */
TYPED_TEST(TypedTensor, transpose_empty)
{
    using value_t = TypeParam;
    Tensor<value_t> t;
    EXPECT_THROW(t.transpose(), temper::validation_error);
}

/**
 * @test TypedTensor.print_tensor
 * @brief Checks that print correctly outputs a 2x2 tensor with values.
 */
TYPED_TEST(TypedTensor, print_tensor)
{
    using value_t = TypeParam;
    temper::Tensor<value_t> t({2, 2}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1.0f), static_cast<value_t>(2.0f),
        static_cast<value_t>(3.0f), static_cast<value_t>(4.0f)
    };
    t = vals;

    std::stringstream ss;
    t.print(ss);

    std::string expected = "[[1, 2],\n [3, 4]]\n";
    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TypedTensor.print_view_tensor
 * @brief Checks print on a 2x2 view of a 3x4 owner tensor.
 */
TYPED_TEST(TypedTensor, print_view_tensor)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({3,4}, MemoryLocation::HOST);
    std::vector<value_t> vals(12);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    owner = vals;

    std::vector<uint64_t> start = {1ull, 1ull};
    std::vector<uint64_t> view_shape = {2ull, 2ull};
    Tensor<value_t> view(owner, start, view_shape);

    std::stringstream ss;
    view.print(ss);

    std::string expected = "[[6, 7],\n [10, 11]]\n";
    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TypedTensor.print_alias_view_weird_strides
 * @brief Checks print on an alias view with non-trivial strides.
 */
TYPED_TEST(TypedTensor, print_alias_view_weird_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({5,20}, MemoryLocation::HOST);
    std::vector<value_t> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<value_t>(i);
    }
    owner = vals;

    Tensor<value_t> view(owner, {0,0}, {3,4}, {13,4});

    std::stringstream ss;
    view.print(ss);

    std::string expected =
        "[[0, 4, 8, 12],\n"
        " [13, 17, 21, 25],\n"
        " [26, 30, 34, 38]]\n";

    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TypedTensor.print_empty_tensor
 * @brief Checks that print correctly outputs an empty tensor.
 */
TYPED_TEST(TypedTensor, print_empty_tensor)
{
    using value_t = TypeParam;
    temper::Tensor<value_t> t;
    std::stringstream ss;
    t.print(ss);
    EXPECT_EQ(ss.str(), "[]\n");
}

/**
 * @test TypedTensor.print_shape_basic
 * @brief Checks that print_shape outputs the dimensions for a 3D tensor.
 */
TYPED_TEST(TypedTensor, print_shape_basic)
{
    using value_t = TypeParam;
    temper::Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);

    std::stringstream ss;
    t.print_shape(ss);

    std::string expected = "[2, 3, 4]\n";
    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TypedTensor.print_shape_empty
 * @brief Checks that print_shape correctly outputs an empty shape.
 */
TYPED_TEST(TypedTensor, print_shape_empty)
{
    using value_t = TypeParam;
    temper::Tensor<value_t> t;

    std::stringstream ss;
    t.print_shape(ss);

    EXPECT_EQ(ss.str(), "[]\n");
}

/**
 * @test TypedTensor.print_shape_view
 * @brief Checks that print_shape outputs the shape of a view tensor.
 */
TYPED_TEST(TypedTensor, print_shape_view)
{
    using value_t = TypeParam;
    temper::Tensor<value_t> owner({3, 4}, MemoryLocation::HOST);
    std::vector<value_t> vals(12);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<value_t>(i + 1);
    }
    owner = vals;

    std::vector<uint64_t> start = {1ull, 1ull};
    std::vector<uint64_t> view_shape = {2ull, 2ull};
    temper::Tensor<value_t> view(owner, start, view_shape);

    std::stringstream ss;
    view.print_shape(ss);

    std::string expected = "[2, 2]\n";
    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TypedTensor.get_data_const
 * @brief Verifies the const overload of get_data().
 */
TYPED_TEST(TypedTensor, get_data_const)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(0.f), static_cast<value_t>(1.f),
        static_cast<value_t>(2.f), static_cast<value_t>(3.f),
        static_cast<value_t>(4.f), static_cast<value_t>(5.f)
    };
    t = vals;

    const Tensor<value_t> ct = t;

    const value_t* cptr = ct.get_data();
    ASSERT_NE(cptr, nullptr);

    EXPECT_EQ(ct.get_dimensions(), t.get_dimensions());
    EXPECT_EQ(ct.get_strides(), t.get_strides());
    EXPECT_EQ(ct.get_num_elements(), t.get_num_elements());

    for (uint64_t i = 0; i < ct.get_num_elements(); ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(cptr[i]),
                            static_cast<float>(i));
            EXPECT_FLOAT_EQ(static_cast<float>(cptr[i]),
                            static_cast<float>(t.get_data()[i]));
        } else {
            EXPECT_EQ(cptr[i], t.get_data()[i]);
            EXPECT_EQ(cptr[i], static_cast<value_t>(i));
        }
    }
}

/**
 * @test TypedTensor.get_data_nonconst
 * @brief Verifies the non-const overload of get_data() is writable.
 */
TYPED_TEST(TypedTensor, get_data_nonconst)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(0.f), static_cast<value_t>(1.f),
        static_cast<value_t>(2.f), static_cast<value_t>(3.f),
        static_cast<value_t>(4.f), static_cast<value_t>(5.f)
    };
    t = vals;

    value_t* wptr = t.get_data();
    ASSERT_NE(wptr, nullptr);

    wptr[0] = static_cast<value_t>(42);
    wptr[5] = static_cast<value_t>(-3);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.get_data()[0]), 42.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t.get_data()[5]), -3.0f);
    } else {
        EXPECT_EQ(t.get_data()[0], static_cast<value_t>(42));
        EXPECT_EQ(t.get_data()[5], static_cast<value_t>(-3));
    }

    for (uint64_t i = 1; i < t.get_num_elements() - 1; ++i)
    {
        if (i != 5)
        {
            if constexpr (std::is_floating_point_v<value_t>) {
                EXPECT_FLOAT_EQ(static_cast<float>(t.get_data()[i]),
                                static_cast<float>(vals[i]));
            } else {
                EXPECT_EQ(t.get_data()[i], vals[i]);
            }
        }
    }
}

/**
 * @test TypedTensor.get_dimensions
 * @brief Validates that get_dimensions() returns the correct shape.
 */
TYPED_TEST(TypedTensor, get_dimensions)
{
    using value_t = TypeParam;
    Tensor<value_t> t({4, 5, 6}, MemoryLocation::HOST);

    EXPECT_EQ(t.get_dimensions(), t.m_dimensions);

    EXPECT_EQ(&t.get_shape(), &t.get_dimensions());

    const Tensor<value_t> ct = t;
    EXPECT_EQ(ct.get_dimensions(), t.get_dimensions());
    EXPECT_EQ(&ct.get_shape(), &ct.get_dimensions());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_dimensions()),
        const std::vector<uint64_t>&
    >, "get_dimensions() must return const std::vector<uint64_t>&");

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_shape()),
        const std::vector<uint64_t>&
    >, "get_shape() must return const std::vector<uint64_t>&");
}

/**
 * @test TypedTensor.get_strides
 * @brief Confirms that get_strides() returns the correct stride vector.
 */
TYPED_TEST(TypedTensor, get_strides)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 2}, MemoryLocation::HOST);

    EXPECT_EQ(t.get_strides(), t.m_strides);

    const Tensor<value_t> ct = t;
    EXPECT_EQ(ct.get_strides(), t.get_strides());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_strides()),
        const std::vector<uint64_t>&
    >, "get_strides() must return const std::vector<uint64_t>&");
}

/**
 * @test TypedTensor.get_rank
 * @brief Tests that get_rank() returns the correct number of dimensions.
 */
TYPED_TEST(TypedTensor, get_rank)
{
    using value_t = TypeParam;
    Tensor<value_t> t({7, 8, 9}, MemoryLocation::HOST);
    EXPECT_EQ(t.get_rank(), static_cast<int64_t>(3));

    const Tensor<value_t> ct = t;
    EXPECT_EQ(ct.get_rank(), t.get_rank());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_rank()),
        int64_t
    >, "get_rank() must return int64_t");
}

/**
 * @test TypedTensor.get_num_elements
 * @brief Ensures get_num_elements() returns the correct element count.
 */
TYPED_TEST(TypedTensor, get_num_elements)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 4, 3}, MemoryLocation::HOST);
    EXPECT_EQ(t.get_num_elements(), static_cast<uint64_t>(2 * 4 * 3));

    Tensor<value_t> empty;
    EXPECT_EQ(empty.get_num_elements(), static_cast<uint64_t>(0));

    const Tensor<value_t> ct = t;
    EXPECT_EQ(ct.get_num_elements(), t.get_num_elements());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_num_elements()),
        uint64_t
    >, "get_num_elements() must return uint64_t");
}

/**
 * @test TypedTensor.get_memory_location
 * @brief Validates that get_memory_location() reports allocation target.
 */
TYPED_TEST(TypedTensor, get_memory_location)
{
    using value_t = TypeParam;
    Tensor<value_t> host_t({2, 2}, MemoryLocation::HOST);
    Tensor<value_t> device_t({2, 2}, MemoryLocation::DEVICE);

    EXPECT_EQ(host_t.get_memory_location(), MemoryLocation::HOST);
    EXPECT_EQ(device_t.get_memory_location(), MemoryLocation::DEVICE);

    const Tensor<value_t> cht = host_t;
    EXPECT_EQ(cht.get_memory_location(), MemoryLocation::HOST);

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_memory_location()),
        MemoryLocation
    >, "get_memory_location() must return MemoryLocation");
}

/**
 * @test TypedTensor.get_owns_data
 * @brief Tests that get_owns_data() distinguishes owning tensors.
 */
TYPED_TEST(TypedTensor, get_owns_data)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({3, 3}, MemoryLocation::HOST);
    EXPECT_TRUE(owner.get_owns_data());

    Tensor<value_t> base({4, 4}, MemoryLocation::HOST);
    std::vector<uint64_t> start = {1, 1};
    std::vector<uint64_t> shape = {2, 2};
    Tensor<value_t> view(base, start, shape);
    EXPECT_FALSE(view.get_owns_data());

    const Tensor<value_t> const_owner = owner;
    EXPECT_TRUE(const_owner.get_owns_data());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_owns_data()),
        bool
    >, "get_owns_data() must return bool");
}

/**
 * @test TypedTensor.is_view
 * @brief Verifies that is_view() correctly identifies tensor views.
 */
TYPED_TEST(TypedTensor, is_view)
{
    using value_t = TypeParam;
    Tensor<value_t> owner({2, 2}, MemoryLocation::HOST);
    Tensor<value_t> base({4, 4}, MemoryLocation::HOST);
    Tensor<value_t> view(base, {1,1}, {2,2});

    EXPECT_FALSE(owner.is_view());
    EXPECT_TRUE(view.is_view());

    const Tensor<value_t> const_view = view;
    EXPECT_TRUE(const_view.is_view());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().is_view()),
        bool
    >, "is_view() must return bool");
}

/**
 * @test TypedTensor.get_element_size_bytes
 * @brief Tests that get_element_size_bytes() returns sizeof(type).
 */
TYPED_TEST(TypedTensor, get_element_size_bytes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({1}, MemoryLocation::HOST);
    EXPECT_EQ(t.get_element_size_bytes(),
              static_cast<uint64_t>(sizeof(value_t)));

    const Tensor<value_t> ct = t;
    EXPECT_EQ(ct.get_element_size_bytes(),
              static_cast<uint64_t>(sizeof(value_t)));

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>()
                 .get_element_size_bytes()),
        uint64_t
    >, "get_element_size_bytes() must return uint64_t");
}

/**
 * @test TypedTensor.get_total_bytes
 * @brief Ensures get_total_bytes() returns correct size in bytes.
 */
TYPED_TEST(TypedTensor, get_total_bytes)
{
    using value_t = TypeParam;
    Tensor<value_t> t({5, 6}, MemoryLocation::HOST);
    uint64_t expected_elems = 5 * 6;
    uint64_t expected_total =
        expected_elems * static_cast<uint64_t>(sizeof(value_t));
    EXPECT_EQ(t.get_total_bytes(), expected_total);

    Tensor<value_t> empty;
    EXPECT_EQ(empty.get_total_bytes(), static_cast<uint64_t>(0));

    const Tensor<value_t> ct = t;
    EXPECT_EQ(ct.get_total_bytes(), expected_total);

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<value_t>&>().get_total_bytes()),
        uint64_t
    >, "get_total_bytes() must return uint64_t");
}

/**
 * @test TypedTensor.index_to_coords_basic
 * @brief Verify index_to_coords maps flat indices to coordinates.
 */
TYPED_TEST(TypedTensor, index_to_coords_basic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3, 4}, MemoryLocation::HOST);

    std::vector<uint64_t> c0 = t.index_to_coords(0);
    ASSERT_EQ(c0.size(), 3u);
    EXPECT_EQ(c0[0], 0u);
    EXPECT_EQ(c0[1], 0u);
    EXPECT_EQ(c0[2], 0u);

    std::vector<uint64_t> c5 = t.index_to_coords(5);
    ASSERT_EQ(c5.size(), 3u);
    EXPECT_EQ(c5[0], 0u);
    EXPECT_EQ(c5[1], 1u);
    EXPECT_EQ(c5[2], 1u);

    std::vector<uint64_t> c23 = t.index_to_coords(23);
    ASSERT_EQ(c23.size(), 3u);
    EXPECT_EQ(c23[0], 1u);
    EXPECT_EQ(c23[1], 2u);
    EXPECT_EQ(c23[2], 3u);

    EXPECT_THROW(t.index_to_coords(24), temper::bounds_error);
}

/**
 * @test TypedTensor.coords_to_index_basic
 * @brief Verify coords_to_index maps coordinates to flat index.
 */
TYPED_TEST(TypedTensor, coords_to_index_basic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3, 4, 5}, MemoryLocation::HOST);

    std::vector<uint64_t> coords = {2u, 3u, 4u};
    uint64_t flat = t.coords_to_index(coords);
    EXPECT_EQ(flat, 59u);

    std::vector<uint64_t> bad_size = {1u, 2u};
    EXPECT_THROW(t.coords_to_index(bad_size), temper::validation_error);

    std::vector<uint64_t> out_of_range = {3u, 0u, 0u};
    EXPECT_THROW(t.coords_to_index(out_of_range), temper::bounds_error);
}

/**
 * @test TypedTensor.at_basic
 * @brief Test reading and writing elements using at() with flat indices.
 */
TYPED_TEST(TypedTensor, at_basic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::HOST);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(0)), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(5)), 6.0f);
    } else {
        EXPECT_EQ(t.at(0), static_cast<value_t>(1));
        EXPECT_EQ(t.at(5), static_cast<value_t>(6));
    }

    t.at(2) = static_cast<value_t>(42.0f);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(2)), 42.0f);
    } else {
        EXPECT_EQ(t.at(2), static_cast<value_t>(42));
    }

    t.at(5) = static_cast<value_t>(-3);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(5)), -3.0f);
    } else {
        EXPECT_EQ(t.at(5), static_cast<value_t>(-3));
    }

    std::vector<value_t> expected = {
        static_cast<value_t>(1),
        static_cast<value_t>(2),
        static_cast<value_t>(42),
        static_cast<value_t>(4),
        static_cast<value_t>(5),
        static_cast<value_t>(-3)
    };
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(t.at(i)),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(t.at(i), expected[i]);
        }
    }
}

/**
 * @test TypedTensor.at_basic_device
 * @brief Test at() on a device tensor.
 */
TYPED_TEST(TypedTensor, at_basic_device)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };
    t = vals;

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(0)), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(5)), 6.0f);
    } else {
        EXPECT_EQ(t.at(0), static_cast<value_t>(1));
        EXPECT_EQ(t.at(5), static_cast<value_t>(6));
    }

    t.at(2) = static_cast<value_t>(42.0f);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(2)), 42.0f);
    } else {
        EXPECT_EQ(t.at(2), static_cast<value_t>(42));
    }

    t.at(5) = static_cast<value_t>(-3);
    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(5)), -3.0f);
    } else {
        EXPECT_EQ(t.at(5), static_cast<value_t>(-3));
    }

    std::vector<value_t> expected = {
        static_cast<value_t>(1),
        static_cast<value_t>(2),
        static_cast<value_t>(42),
        static_cast<value_t>(4),
        static_cast<value_t>(5),
        static_cast<value_t>(-3)
    };
    for (uint64_t i = 0; i < 6; ++i)
    {
        if constexpr (std::is_floating_point_v<value_t>) {
            EXPECT_FLOAT_EQ(static_cast<float>(t.at(i)),
                            static_cast<float>(expected[i]));
        } else {
            EXPECT_EQ(t.at(i), expected[i]);
        }
    }
}

/**
 * @test TypedTensor.at_out_of_range
 * @brief Verify that at() throws temper::bounds_error for invalid indices.
 */
TYPED_TEST(TypedTensor, at_out_of_range)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::HOST);

    EXPECT_NO_THROW(t.at(0));
    EXPECT_NO_THROW(t.at(5));

    EXPECT_THROW(t.at(6), temper::bounds_error);
    EXPECT_THROW(t.at(100), temper::bounds_error);
}

/**
 * @test TypedTensor.at_const
 * @brief Verify that const tensors can read using at().
 */
TYPED_TEST(TypedTensor, at_const)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,3}, MemoryLocation::HOST);
    t = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6)
    };

    const Tensor<value_t>& ct = t;

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(ct.at(0)), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(ct.at(5)), 6.0f);
    } else {
        EXPECT_EQ(ct.at(0), static_cast<value_t>(1));
        EXPECT_EQ(ct.at(5), static_cast<value_t>(6));
    }
}

/**
 * @test TypedTensor.at_view
 * @brief Test at() on a sub-tensor view with contiguous strides.
 */
TYPED_TEST(TypedTensor, at_view)
{
    using value_t = TypeParam;
    Tensor<value_t> t({3,3}, MemoryLocation::HOST);
    t = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4),
        static_cast<value_t>(5), static_cast<value_t>(6),
        static_cast<value_t>(7), static_cast<value_t>(8),
        static_cast<value_t>(9)
    };

    Tensor<value_t> v(t, {1,1}, {2,2}, {3,1});

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(0)), 5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(1)), 6.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(2)), 8.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(3)), 9.0f);
    } else {
        EXPECT_EQ(v.at(0), static_cast<value_t>(5));
        EXPECT_EQ(v.at(1), static_cast<value_t>(6));
        EXPECT_EQ(v.at(2), static_cast<value_t>(8));
        EXPECT_EQ(v.at(3), static_cast<value_t>(9));
    }

    v.at(0) = static_cast<value_t>(-5);
    v.at(3) = static_cast<value_t>(-9);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(4)), -5.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(8)), -9.0f);
    } else {
        EXPECT_EQ(t.at(4), static_cast<value_t>(-5));
        EXPECT_EQ(t.at(8), static_cast<value_t>(-9));
    }
}

/**
 * @test TypedTensor.at_alias_view_noncontiguous
 * @brief Test at() on a non-contiguous alias view (stride > 1).
 */
TYPED_TEST(TypedTensor, at_alias_view_noncontiguous)
{
    using value_t = TypeParam;
    Tensor<value_t> t({1,8}, MemoryLocation::HOST);
    t = {
        static_cast<value_t>(10), static_cast<value_t>(0),
        static_cast<value_t>(20), static_cast<value_t>(0),
        static_cast<value_t>(30), static_cast<value_t>(0),
        static_cast<value_t>(40), static_cast<value_t>(0)
    };

    Tensor<value_t> v(t, {0,0}, {1,4}, {1,2});

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(0)), 10.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(1)), 20.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(2)), 30.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(v.at(3)), 40.0f);
    } else {
        EXPECT_EQ(v.at(0), static_cast<value_t>(10));
        EXPECT_EQ(v.at(1), static_cast<value_t>(20));
        EXPECT_EQ(v.at(2), static_cast<value_t>(30));
        EXPECT_EQ(v.at(3), static_cast<value_t>(40));
    }

    v.at(1) = static_cast<value_t>(99);
    v.at(3) = static_cast<value_t>(77);

    if constexpr (std::is_floating_point_v<value_t>) {
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(2)), 99.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(t.at(6)), 77.0f);
    } else {
        EXPECT_EQ(t.at(2), static_cast<value_t>(99));
        EXPECT_EQ(t.at(6), static_cast<value_t>(77));
    }
}

/**
 * @test TypedTensor.at_view_out_of_range
 * @brief Verify that at() throws temper::bounds_error
 * for invalid indices on views.
 */
TYPED_TEST(TypedTensor, at_view_out_of_range)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2,2}, MemoryLocation::HOST);
    t = {
        static_cast<value_t>(1), static_cast<value_t>(2),
        static_cast<value_t>(3), static_cast<value_t>(4)
    };

    Tensor<value_t> v(t, {0,0}, {2,2}, {2,1});

    EXPECT_NO_THROW(v.at(0));
    EXPECT_NO_THROW(v.at(3));
    EXPECT_THROW(v.at(4), temper::bounds_error);
    EXPECT_THROW(v.at(100), temper::bounds_error);
}

/**
 * @test TypedTensor.begin_end_basic
 * @brief begin() points at flat index 0, end() at num elements.
 */
TYPED_TEST(TypedTensor, begin_end_basic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);

    auto b = t.begin();
    auto e = t.end();

    EXPECT_EQ(b.m_flat_idx, 0u);
    EXPECT_EQ(e.m_flat_idx, t.get_num_elements());

    size_t cnt = 0;
    for (auto it = b; it != e; ++it)
    {
        ++cnt;
    }
    EXPECT_EQ(cnt, t.get_num_elements());
}

/**
 * @test TypedTensor.begin_end_range_for_equivalence
 * @brief Range-based for uses begin()/end() and reads values in order.
 */
TYPED_TEST(TypedTensor, begin_end_range_for_equivalence)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(6);
    std::iota(vals.begin(), vals.end(), static_cast<value_t>(0.0f));
    t = vals;

    EXPECT_EQ(t.begin().m_flat_idx, 0u);
    EXPECT_EQ(t.end().m_flat_idx, t.get_num_elements());

    size_t idx = 0;
    for (auto view : t) {
        float v = static_cast<float>(view);
        EXPECT_FLOAT_EQ(v, static_cast<float>(vals[idx++]));
    }
    EXPECT_EQ(idx, vals.size());
}

/**
 * @test TypedTensor.begin_end_const
 * @brief const overloads of begin()/end() point at flat index 0.
 */
TYPED_TEST(TypedTensor, begin_end_const)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(6);
    std::iota(vals.begin(), vals.end(), static_cast<value_t>(0.0f));
    t = vals;

    const Tensor<value_t>& ct = t;

    auto b = ct.begin();
    auto e = ct.end();

    EXPECT_EQ(b.m_flat_idx, 0u);
    EXPECT_EQ(e.m_flat_idx, ct.get_num_elements());

    size_t idx = 0;
    for (auto view : ct) {
        float v = static_cast<float>(view);
        EXPECT_FLOAT_EQ(v, static_cast<float>(vals[idx++]));
    }
    EXPECT_EQ(idx, vals.size());
}

/**
 * @test TypedTensor.cbegin_cend_basic
 * @brief cbegin()/cend() are equivalent to begin()/end().
 */
TYPED_TEST(TypedTensor, cbegin_cend_basic)
{
    using value_t = TypeParam;
    Tensor<value_t> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<value_t> vals(6);
    std::iota(vals.begin(), vals.end(), static_cast<value_t>(0.0f));
    t = vals;

    auto cb = t.cbegin();
    auto ce = t.cend();

    EXPECT_EQ(cb.m_flat_idx, t.begin().m_flat_idx);
    EXPECT_EQ(ce.m_flat_idx, t.end().m_flat_idx);

    size_t idx = 0;
    for (auto it = cb; it != ce; ++it) {
        auto view = *it;
        float v = static_cast<float>(view);
        EXPECT_FLOAT_EQ(v, static_cast<float>(vals[idx++]));
    }
    EXPECT_EQ(idx, vals.size());
}

} // namespace Test