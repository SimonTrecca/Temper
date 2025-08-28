/**
 * @file ut_Tensor.cpp
 * @brief Google Test suite for proper Tensor class functionality.
 *
 * This file declares the Tensor class which handles multi-dimensional
 * arrays with row-major memory layout.
 */

#include <gtest/gtest.h>
#include <cstdint>

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
template <typename float_t>
void copy_tensor_data(Tensor<float_t>& dest, const Tensor<float_t>& src)
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

    float_t* src_data  = src.m_p_data.get();
    float_t* dest_data = dest.m_p_data.get();

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

/**
 * @test TENSOR.compute_strides_empty_dimensions
 * @brief Tests compute_strides() with no dimensions.
 *
 * If the dimensions vector is empty, strides should also remain empty.
 */
TEST(TENSOR, compute_strides_empty_dimensions)
{
    Tensor<float> t;
    t.m_dimensions.clear();
    t.compute_strides();
    EXPECT_TRUE(t.m_strides.empty());
}

/**
 * @test TENSOR.compute_strides_one_dimension
 * @brief Tests compute_strides() with a single dimension.
 *
 * A 1D tensor should always have a single stride of 1.
 */
TEST(TENSOR, compute_strides_one_dimension)
{
    Tensor<float> t;
    t.m_dimensions = { 7 };
    t.compute_strides();

    // Single-dim stride should always be 1.
    ASSERT_EQ(t.m_strides.size(), 1u);
    EXPECT_EQ(t.m_strides[0], 1u);
}

/**
 * @test TENSOR.compute_strides_larger_tensor
 * @brief Tests compute_strides() with four dimensions.
 *
 * For dims = [4, 1, 6, 2], expected strides = [12, 12, 2, 1].
 */
TEST(TENSOR, compute_strides_larger_tensor)
{
    Tensor<float> t;
    t.m_dimensions = { 4, 1, 6, 2 };
    t.compute_strides();

    // Strides: [1*6*2, 6*2, 2, 1] = [12,12,2,1].
    std::vector<uint64_t> expected = { 12, 12, 2, 1 };
    ASSERT_EQ(t.m_strides, expected);
}

/**
 * @test TENSOR.compute_strides_zero_dimension_throws
 * @brief compute_strides() should throw if any dimension is zero.
 */
TEST(TENSOR, compute_strides_zero_dimension_throws)
{
    Tensor<float> t;
    t.m_dimensions = { 3, 0, 2 };
    EXPECT_THROW(t.compute_strides(), std::invalid_argument);
}

/**
 * @test TENSOR.compute_strides_overflow_throws
 * @brief compute_strides() should throw if stride multiplication would overflow.
 *
 * Choose suffix dims so their product exceeds uint64_t and triggers overflow.
 */
TEST(TENSOR, compute_strides_overflow_throws)
{
    const uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    uint64_t dim1 = (U64_MAX / 2) + 1;
    uint64_t dim2 = 2;

    Tensor<float> t;
    t.m_dimensions = { 1, dim1, dim2 };

    EXPECT_THROW(t.compute_strides(), std::overflow_error);
}


/**
 * @test TENSOR.main_constructor_sets_dimensions_and_strides
 * @brief Tests that the Tensor constructor
 * correctly sets dimensions and computes strides.
 */
TEST(TENSOR, main_constructor_sets_dimensions_and_strides)
{
    std::vector<uint64_t> dims = { 2, 3, 4 };
    Tensor<float> t(dims, MemoryLocation::DEVICE);

    EXPECT_EQ(t.m_dimensions, dims);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<uint64_t> expected_strides = { 12, 4, 1 };
    EXPECT_EQ(t.m_strides, expected_strides);
}

/**
 * @test TENSOR.main_constructor_zero_initializes_data
 * @brief Tests that the Tensor constructor allocates the
 * correct amount of memory and initializes it to zero.
 */
TEST(TENSOR, main_constructor_zero_initializes_data)
{
    std::vector<uint64_t> dims = { 2, 3 };
    Tensor<float> t(dims, MemoryLocation::HOST);

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : dims)
    {
        total_size *= d;
    }

    std::vector<float> host_data(total_size);
    sycl::event e = g_sycl_queue.memcpy(
        host_data.data(),
        t.m_p_data.get(),
        sizeof(float) * total_size
    );
    e.wait();

    for (float v : host_data)
    {
        EXPECT_EQ(v, 0.0f);
    }
}

/**
 * @test TENSOR.main_constructor_memory_location_and_access
 * @brief Tests correct memory location assignment.
 */
TEST(TENSOR, main_constructor_memory_location_and_access)
{
    // DEVICE tensor test: write via kernel, then copy back to host and verify.
    Tensor<float> t_device({1, 1}, MemoryLocation::DEVICE);
    EXPECT_EQ(t_device.m_mem_loc, MemoryLocation::DEVICE);

    // Launch kernel to set element to 42.0f.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        float * ptr = t_device.m_p_data.get();
        cgh.single_task([=]()
        {
            ptr[0] = 42.0f;
        });
    }).wait();

    // Copy back to host and check.
    float host_val = 0.f;
    g_sycl_queue.memcpy
        (&host_val, t_device.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host_val, 42.0f);


    // HOST tensor test: write directly on host memory and read back.
    Tensor<float> t_host({1, 1}, MemoryLocation::HOST);
    EXPECT_EQ(t_host.m_mem_loc, MemoryLocation::HOST);

    // Direct write on host pointer.
    t_host.m_p_data.get()[0] = 24.0f;
    EXPECT_FLOAT_EQ(t_host.m_p_data.get()[0], 24.0f);
}

/**
 * @test TENSOR.main_constructor_empty_dimensions
 * @brief Throws invalid_argument when dimensions vector is empty.
 */
TEST(TENSOR, main_constructor_empty_dimensions)
{
    std::vector<uint64_t> dims = {};
    EXPECT_THROW(
        Tensor<float> t(dims, MemoryLocation::HOST),
        std::invalid_argument
    );
}

/**
 * @test TENSOR.main_constructor_zero_dimension
 * @brief Throws invalid_argument when any dimension is zero.
 */
TEST(TENSOR, main_constructor_zero_dimension)
{
    std::vector<uint64_t> dims = { 2, 0, 3 };

    EXPECT_THROW(
        Tensor<float> t(dims, MemoryLocation::HOST),
        std::invalid_argument
    );
}

/**
 * @test TENSOR.main_constructor_element_count_overflow
 * @brief Throws overflow_error when total_size would overflow uint64_t.
 */
TEST(TENSOR, main_constructor_element_count_overflow)
{
    std::vector<uint64_t> dims = { std::numeric_limits<uint64_t>::max(), 2 };

    EXPECT_THROW(
        Tensor<float> t(dims, MemoryLocation::HOST),
        std::overflow_error
    );
}

/**
 * @test TENSOR.main_constructor_allocation_bytes_overflow
 * @brief Throws overflow_error when allocation size exceeds uint64_t.
 */
TEST(TENSOR, main_constructor_allocation_bytes_overflow)
{
    std::vector<uint64_t> dims =
        { std::numeric_limits<uint64_t>::max() / sizeof(float) + 1 };

    EXPECT_THROW(
        Tensor<float> t(dims, MemoryLocation::HOST),
        std::overflow_error
    );
}

/**
 * @test TENSOR.main_constructor_exceeds_device_max_alloc
 * @brief Throws runtime_error if requested allocation exceeds
 * device max_mem_alloc_size.
 */
TEST(TENSOR, main_constructor_exceeds_device_max_alloc)
{
    auto dev = g_sycl_queue.get_device();
    uint64_t dev_max_alloc =
        dev.get_info<sycl::info::device::max_mem_alloc_size>();

    std::vector<uint64_t> dims = { (dev_max_alloc / sizeof(float)) + 1 };
    EXPECT_THROW(
        Tensor<float> t(dims, MemoryLocation::DEVICE),
        std::runtime_error
    );
}

/**
 * @test TENSOR.main_constructor_exceeds_device_global_mem
 * @brief Throws runtime_error if requested allocation exceeds
 * device global_mem_size.
 */
TEST(TENSOR, main_constructor_exceeds_device_global_mem)
{
    auto dev = g_sycl_queue.get_device();
    uint64_t dev_global_mem =
        dev.get_info<sycl::info::device::global_mem_size>();

    std::vector<uint64_t> dims = { (dev_global_mem / sizeof(float)) + 1 };
    EXPECT_THROW(
        Tensor<float> t(dims, MemoryLocation::DEVICE),
        std::runtime_error
    );
}

/**
 * @test TENSOR.copy_constructor
 * @brief Tests copy constructor.
 */
TEST(TENSOR, copy_constructor)
{
    Tensor<float> t1({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    t1 = values;

    Tensor<float> t2(t1);

    EXPECT_EQ(t2.m_mem_loc, t1.m_mem_loc);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), t2.m_p_data.get(), sizeof(float) * 4).wait();

    EXPECT_EQ(host[0], 1.0f);
    EXPECT_EQ(host[1], 2.0f);
    EXPECT_EQ(host[2], 3.0f);
    EXPECT_EQ(host[3], 4.0f);
}

/**
 * @test TENSOR.copy_constructor_default
 * @brief Tests that copying a default-constructed tensor
 * results in another empty tensor with nullptr data.
 */
TEST(TENSOR, copy_constructor_on_default_constructed)
{
    Tensor<float> t1;

    Tensor<float> t2(t1);

    EXPECT_TRUE(t2.m_dimensions.empty());
    EXPECT_EQ(t2.m_p_data, nullptr);
    EXPECT_TRUE(t2.m_own_data);
}

/**
 * @test TENSOR.copy_constructor_host
 * @brief Tests copy constructor with a tensor allocated in HOST memory.
 * Ensures contents are copied correctly.
 */
TEST(TENSOR, copy_constructor_host)
{
    Tensor<float> t1({2, 2}, MemoryLocation::HOST);
    std::vector<float> values = {10.0f, 20.0f, 30.0f, 40.0f};
    t1 = values;

    Tensor<float> t2(t1);

    std::vector<float> host(4);
    std::memcpy(host.data(), t2.m_p_data.get(), sizeof(float) * 4);

    EXPECT_EQ(host[0], 10.0f);
    EXPECT_EQ(host[1], 20.0f);
    EXPECT_EQ(host[2], 30.0f);
    EXPECT_EQ(host[3], 40.0f);
}

/**
 * @test TENSOR.copy_constructor_view
 * @brief Tests copy constructor on a view tensor (non-owning).
 * Copying a view must produce another view, sharing the same memory.
 */
TEST(TENSOR, copy_constructor_view)
{
    Tensor<float> t1({4}, MemoryLocation::DEVICE);
    std::vector<float> values = {1, 2, 3, 4};
    t1 = values;

    Tensor<float> view(t1, {2}, {2});
    ASSERT_FALSE(view.m_own_data);

    Tensor<float> copy(view);

    EXPECT_EQ(copy.m_dimensions, view.m_dimensions);
    EXPECT_FALSE(copy.m_own_data);
    EXPECT_EQ(copy.m_p_data.get(), view.m_p_data.get());
}

/**
 * @test TENSOR.move_constructor
 * @brief Tests move constructor.
 */
TEST(TENSOR, move_constructor)
{
    Tensor<float> t1({2, 2}, MemoryLocation::HOST);
    std::vector<float> values = {5.0f, 6.0f, 7.0f, 8.0f};
    t1 = values;

    float* original_ptr = t1.m_p_data.get();
    MemoryLocation original_loc = t1.m_mem_loc;

    Tensor<float> t2(std::move(t1));

    EXPECT_EQ(t2.m_p_data.get(), original_ptr);
    EXPECT_EQ(t2.m_mem_loc, original_loc);
    EXPECT_EQ(t1.m_p_data.get(), nullptr);
}

/**
 * @test TENSOR.view_constructor_preserves_strides_and_data
 * @brief Tests that slicing a CHW-format tensor creates
 * a view with correct strides and verifies that the data
 * accessed via the view matches expected values.
 */
TEST(TENSOR, view_constructor_preserves_strides_and_data)
{
    Tensor<float> img({3, 4, 5}, MemoryLocation::DEVICE);
    EXPECT_EQ(img.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> vals(3 * 4 * 5);

    for (uint64_t c = 0; c < 3; ++c)
    {
        for (uint64_t i = 0; i < 4; ++i)
        {
            for (uint64_t j = 0; j < 5; ++j)
            {
                vals[c * 20 + i * 5 + j] =
                    static_cast<float>(c * 100 + i * 10 + j);
            }
        }
    }

    img = vals;

    Tensor<float> patch(img, {1, 0, 0}, {2, 3});

    EXPECT_EQ(patch.m_mem_loc, MemoryLocation::DEVICE);

    EXPECT_EQ(patch.m_dimensions, std::vector<uint64_t>({2, 3}));
    EXPECT_EQ(patch.m_strides[0], img.m_strides[1]);
    EXPECT_EQ(patch.m_strides[1], img.m_strides[2]);

    Tensor<float> host({2, 3}, MemoryLocation::HOST);
    copy_tensor_data(host, patch);

    std::vector<float> out(6);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 6).wait();

    for (int ii = 0; ii < 2; ++ii)
    {
        for (int jj = 0; jj < 3; ++jj)
        {
            float expected = static_cast<float>(100 + ii * 10 + jj);
            EXPECT_FLOAT_EQ(out[ii * 3 + jj], expected);
        }
    }
}

/**
 * @test TENSOR.view_constructor_identity_preserves_layout
 * @brief Verifies that slicing a tensor without dropping
 * any axes returns a view with identical dimensions, strides, and values.
 */
TEST(TENSOR, view_constructor_identity_preserves_layout)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> v = {0, 1, 2, 3, 4, 5};
    t = v;

    Tensor<float> view(t, {0, 0}, {2, 3});

    EXPECT_EQ(view.m_mem_loc, MemoryLocation::DEVICE);

    EXPECT_EQ(view.m_dimensions, t.m_dimensions);
    EXPECT_EQ(view.m_strides, t.m_strides);

    Tensor<float> host({2, 3}, MemoryLocation::HOST);
    copy_tensor_data(host, view);

    std::vector<float> out(6);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 6).wait();

    for (uint64_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], v[i]);
    }
}

/**
 * @test TENSOR.view_constructor_invalid_arguments_throw
 * @brief Ensures that invalid slice arguments
 * (e.g., mismatched ranks, out-of-bounds access)
 * correctly throw exceptions.
 */
TEST(TENSOR, view_constructor_invalid_arguments_throw)
{
    Tensor<float> t({2, 2, 2});

    // Too few shape dimensions test.
    EXPECT_THROW((Tensor<float>(t, {0, 0}, {1})), std::invalid_argument);

    // Too many shape dimensions test.
    EXPECT_THROW((Tensor<float>(t, {0, 0, 0}, {1, 1, 1, 1})),
        std::invalid_argument);

    // Zero-size dimension test.
    EXPECT_THROW((Tensor<float>(t, {0, 0, 0}, {0, 1})), std::out_of_range);

    // Out-of-bounds shape test.
    EXPECT_THROW((Tensor<float>(t, {0, 0, 0}, {3, 1})), std::out_of_range);
    EXPECT_THROW((Tensor<float>(t, {2, 0, 0}, {1, 1})), std::out_of_range);
}

/**
 * @test TENSOR.view_constructor_4d_drops_prefix_axes
 * @brief Tests slicing a 4D tensor while dropping the first two axes,
 * verifying correct shape, strides, and values in the resulting view.
 *
 * The original tensor has shape {2, 3, 4, 5} and is filled with
 * values from 1 to 120.
 * The slice extracts the {4, 5} sub-tensor at position (0, 0, :, :)
 */
TEST(TENSOR, view_constructor_4d_drops_prefix_axes)
{
    Tensor<float> t({2, 3, 4, 5});
    std::vector<float> vals(120);

    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }

    t = vals;

    // Slice last two dimensions: drop first two axes.
    Tensor<float> slice(t, {0, 0, 0, 0}, {4, 5});

    EXPECT_EQ(slice.m_dimensions, std::vector<uint64_t>({4, 5}));
    EXPECT_EQ(slice.m_strides[0], t.m_strides[2]);
    EXPECT_EQ(slice.m_strides[1], t.m_strides[3]);

    Tensor<float> host({4, 5});
    copy_tensor_data(host, slice);

    std::vector<float> out(20);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 20).wait();

    for (uint64_t x = 0; x < 4; ++x)
    {
        for (uint64_t y = 0; y < 5; ++y)
        {
            float expected = static_cast<float>
                ((0 * 60) + (0 * 20) + (x * 5) + y + 1);
            EXPECT_FLOAT_EQ(out[x * 5 + y], expected);
        }
    }
}

/**
 * @test TENSOR.view_constructor_4d_extracts_3d_volume
 * @brief Extracts a 3D chunk from a 4D tensor by dropping the first axis and
 * verifies shape, stride, and copied values.
 */
TEST(TENSOR, view_constructor_4d_extracts_3d_volume)
{
    Tensor<float> t({2, 3, 4, 5});
    std::vector<float> vals(120);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }

    t = vals;

    Tensor<float> slice(t, {1, 0, 0, 0}, {3, 4, 5});

    EXPECT_EQ(slice.m_dimensions, std::vector<uint64_t>({3, 4, 5}));
    EXPECT_EQ(slice.m_strides[0], t.m_strides[1]);
    EXPECT_EQ(slice.m_strides[1], t.m_strides[2]);
    EXPECT_EQ(slice.m_strides[2], t.m_strides[3]);

    Tensor<float> host({3, 4, 5});
    copy_tensor_data(host, slice);

    std::vector<float> out(60);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 60).wait();

    for (uint64_t i = 0; i < 3; ++i)
        for (uint64_t j = 0; j < 4; ++j)
            for (uint64_t k = 0; k < 5; ++k)
            {
                float expected = static_cast<float>
                    ((1 * 60) + (i * 20) + (j * 5) + k + 1);
                EXPECT_FLOAT_EQ(out[i * 20 + j * 5 + k], expected);
            }
}

/**
 * @test TENSOR.view_constructor_4d_extracts_1d_row
 * @brief Slices a single row (1D) from the last dimension of a 4D tensor,
 * and verifies the extracted values.
 */
TEST(TENSOR, view_constructor_4d_extracts_1d_row)
{
    Tensor<float> t({2, 3, 4, 5});
    std::vector<float> vals(120);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    t = vals;

    Tensor<float> slice(t, {1, 2, 3, 0}, {5});

    EXPECT_EQ(slice.m_dimensions, std::vector<uint64_t>({5}));
    EXPECT_EQ(slice.m_strides[0], t.m_strides[3]);

    Tensor<float> host({5});
    copy_tensor_data(host, slice);

    std::vector<float> out(5);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 5).wait();

    for (uint64_t k = 0; k < 5; ++k)
    {
        float expected = static_cast<float>
            ((1 * 60) + (2 * 20) + (3 * 5) + k + 1);
        EXPECT_FLOAT_EQ(out[k], expected);
    }
}

/**
 * @test TENSOR.view_constructor_chw_extracts_large_patch
 * @brief Slices a 100x100 patch from a 3D CHW-format tensor
 * at a specified spatial location.
 * Verifies shape, stride, and content correctness.
 */
TEST(TENSOR, view_constructor_chw_extracts_large_patch)
{
    Tensor<float> img({3, 256, 256});
    std::vector<float> vals(3 * 256 * 256);

    for (uint64_t c = 0; c < 3; ++c)
    {
        for (uint64_t h = 0; h < 256; ++h)
        {
            for (uint64_t w = 0; w < 256; ++w)
            {
                vals[c * 256 * 256 + h * 256 + w] = static_cast<float>
                    (c * 1000000 + h * 1000 + w);
            }
        }
    }

    img = vals;

    Tensor<float> patch(img, {0, 50, 70}, {100, 100});

    EXPECT_EQ(patch.m_dimensions, std::vector<uint64_t>({100, 100}));
    EXPECT_EQ(patch.m_strides[0], img.m_strides[1]);
    EXPECT_EQ(patch.m_strides[1], img.m_strides[2]);

    Tensor<float> host({100, 100});
    copy_tensor_data(host, patch);

    std::vector<float> out(100 * 100);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(),
        sizeof(float) * 100 * 100).wait();

    for (uint64_t i = 0; i < 100; ++i)
    {
        for (uint64_t j = 0; j < 100; ++j)
        {
            float expected = static_cast<float>
                ((0 * 1000000) + (50 + i) * 1000 + (70 + j));
            EXPECT_FLOAT_EQ(out[i * 100 + j], expected);
        }
    }
}

/**
 * @test TENSOR.view_constructor_modification_reflects_in_original
 * @brief Tests that modifying a tensor view updates the original tensor's memory.
 *
 * A view is created on a region of the tensor. Data is written via the view and
 * then read again from a second view on the original tensor. The test verifies
 * that the data matches, confirming memory is shared.
 */
TEST(TENSOR, view_constructor_modification_reflects_in_original)
{
    Tensor<float> t({3, 4, 5});
    std::vector<float> vals(3 * 4 * 5, 0.0f);
    t = vals;

    Tensor<float> view(t, {1, 0, 0}, {4, 5});

    // Prepare values to write into the view.
    Tensor<float> host({4, 5});
    std::vector<float> patch_vals(4 * 5);

    for (uint64_t i = 0; i < 4; ++i)
    {
        for (uint64_t j = 0; j < 5; ++j)
        {
            patch_vals[i * 5 + j] = static_cast<float>(42 + i * 5 + j);
        }
    }
    host = patch_vals;

    // Write to view (which writes into t).
    copy_tensor_data(view, host);

    // Read again from the same region of t.
    Tensor<float> ref(t, {1, 0, 0}, {4, 5});
    Tensor<float> readback({4, 5});
    copy_tensor_data(readback, ref);

    std::vector<float> out(4 * 5);
    g_sycl_queue.memcpy(out.data(), readback.m_p_data.get(),
        sizeof(float) * 4 * 5).wait();

    for (uint64_t i = 0; i < 4; ++i)
    {
        for (uint64_t j = 0; j < 5; ++j)
        {
            EXPECT_FLOAT_EQ(out[i * 5 + j], static_cast<float>(42 + i * 5 + j));
        }
    }
}

/**
 * @test TENSOR.view_constructor_owner_destroyed_before_view
 * @brief Ensure a view's aliasing shared_ptr keeps the underlying buffer alive
 * after the original owner goes out of scope.
 */

TEST(TENSOR, view_constructor_owner_destroyed_before_view)
{
    std::weak_ptr<float> weak_data_ptr;
    Tensor<float> view;
    {
        Tensor<float> owner({2, 2});
        std::vector<float> vals = {1.1f, 2.2f, 3.3f, 4.4f};
        owner = vals;

        std::vector<uint64_t> start{0, 0};
        std::vector<uint64_t> shape{2, 2};
        view = Tensor<float>(owner, start, shape);
        weak_data_ptr = view.m_p_data;
        EXPECT_FLOAT_EQ(static_cast<float>(view[0][0]), 1.1f);
    }
    EXPECT_FALSE(weak_data_ptr.expired());
}

/**
 * @test TENSOR.view_constructor_view_destroyed_before_owner
 * @brief Ensure the buffer remains alive while the owner exists and is freed
 * after the owner releases ownership.
 */

TEST(TENSOR, view_constructor_view_destroyed_before_owner)
{
    std::weak_ptr<float> weak_data_ptr;

    Tensor<float> owner({2, 2});
    std::vector<float> vals = {5.5f, 6.6f, 7.7f, 8.8f};
    owner = vals;

    {
        std::vector<uint64_t> start{0, 0};
        std::vector<uint64_t> shape{2, 2};
        Tensor<float> view(owner, start, shape);

        weak_data_ptr = view.m_p_data;

        EXPECT_FLOAT_EQ(static_cast<float>(view[0][0]), 5.5f);
    }
    EXPECT_FALSE(weak_data_ptr.expired());

    owner = Tensor<float>();
    EXPECT_TRUE(weak_data_ptr.expired());
}

/**
 * @test TENSOR.view_constructor_from_uninitialized_throws
 * @brief Creating a view from a default-constructed/moved-from
 * tensor must throw.
 */
TEST(TENSOR, view_constructor_from_uninitialized_throws)
{
    Tensor<float> owner;

    std::vector<uint64_t> start = {0};
    std::vector<uint64_t> shape = {1};

    EXPECT_THROW(
        Tensor<float> view(owner, start, shape),
        std::runtime_error
    );

    Tensor<float> valid({2,2}, MemoryLocation::HOST);
    std::vector<float> vals = {1,2,3,4};
    valid = vals;

    Tensor<float> moved = std::move(valid);
    EXPECT_THROW(
        Tensor<float> view2
            (valid, std::vector<uint64_t>{0,0}, std::vector<uint64_t>{1,1}),
        std::runtime_error
    );
}

/**
 * @test TENSOR.view_constructor_alias_pointer_offset
 * @brief View constructor must alias the owner's pointer at the correct offset.
 */
TEST(TENSOR, view_constructor_alias_pointer_offset)
{
    Tensor<float> owner({2,3}, MemoryLocation::HOST);
    std::vector<float> vals = { 10,11,12, 20,21,22 };
    owner = vals;

    std::vector<uint64_t> start = {1, 1};
    std::vector<uint64_t> shape = {1, 2};

    Tensor<float> view(owner, start, shape);

    uint64_t offset =
        start[0] * owner.m_strides[0] + start[1] * owner.m_strides[1];

    EXPECT_EQ(view.m_p_data.get(), owner.m_p_data.get() + offset);

    std::vector<float> dst(2);
    g_sycl_queue.memcpy
        (dst.data(), view.m_p_data.get(), sizeof(float) * 2).wait();
    EXPECT_FLOAT_EQ(dst[0], vals[offset + 0]);
    EXPECT_FLOAT_EQ(dst[1], vals[offset + 1]);
}

/**
 * @test TENSOR.view_constructor_from_alias
 * @brief Ensures that creating a view from an alias works correctly.
 *
 * Start with a 2x3 tensor:
 * [ [0, 1, 2],
 * [3, 4, 5] ]
 * First build an alias with shape (3,2) and strides (1,3)
 * (transposed view: [ [0,3], [1,4], [2,5] ]).
 * Then slice a view from that alias: take the first two rows
 * => shape (2,2). Verify dimensions, strides, and values.
 */
TEST(TENSOR, view_constructor_from_alias) {
    // Owner tensor 2x3
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    std::vector<float> vals = { 0.f, 1.f, 2.f,
                                3.f, 4.f, 5.f };
    t = vals;

    Tensor<float> alias(t, {0,0}, {3,2}, {1,3});
    EXPECT_EQ(alias.m_dimensions, std::vector<uint64_t>({3,2}));
    EXPECT_EQ(alias.m_strides, std::vector<uint64_t>({1,3}));

    Tensor<float> subview(alias, {0,0}, {2,2});
    EXPECT_EQ(subview.m_dimensions, std::vector<uint64_t>({2,2}));
    EXPECT_EQ(subview.m_strides[0], alias.m_strides[0]);
    EXPECT_EQ(subview.m_strides[1], alias.m_strides[1]);

    Tensor<float> host({2,2}, MemoryLocation::HOST);
    copy_tensor_data(host, subview);

    std::vector<float> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*4).wait();

    // Expected values from transposed alias:
    // subview = [ [0,3],
    //             [1,4] ].
    EXPECT_FLOAT_EQ(out[0], 0.f);
    EXPECT_FLOAT_EQ(out[1], 3.f);
    EXPECT_FLOAT_EQ(out[2], 1.f);
    EXPECT_FLOAT_EQ(out[3], 4.f);
}

/**
 * @test TENSOR.alias_view_constructor_extracts_column
 * @brief Extracts a single column from a 2x4 tensor.
 * Verifies dimensions, strides, and content correctness.
 */
TEST(TENSOR, alias_view_constructor_extracts_column)
{
    Tensor<float> t({2, 4}, MemoryLocation::HOST);
    std::vector<float> vals = {
        0.f, 1.f, 2.f, 3.f,
        4.f, 5.f, 6.f, 7.f
    };
    t = vals;

    Tensor<float> col_view(t, {0,1}, {2}, {t.m_strides[0]});

    EXPECT_EQ(col_view.m_dimensions, std::vector<uint64_t>({2}));
    EXPECT_EQ(col_view.m_strides, std::vector<uint64_t>({4}));

    Tensor<float> host({2}, MemoryLocation::HOST);
    copy_tensor_data(host, col_view);

    std::vector<float> out(2);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*2).wait();

    EXPECT_FLOAT_EQ(out[0], 1.f);
    EXPECT_FLOAT_EQ(out[1], 5.f);
}

/**
 * @test TENSOR.alias_view_constructor_extracts_row
 * @brief Extracts a row from a 2x4 tensor.
 * Verifies dimensions, strides, and content correctness.
 */
TEST(TENSOR, alias_view_constructor_extracts_row)
{
    Tensor<float> t({2,4}, MemoryLocation::HOST);
    std::vector<float> vals = {0,1,2,3,4,5,6,7};
    t = vals;

    Tensor<float> row_view(t, {1,0}, {4}, {t.m_strides[1]});

    EXPECT_EQ(row_view.m_dimensions, std::vector<uint64_t>({4}));
    EXPECT_EQ(row_view.m_strides, std::vector<uint64_t>({1}));

    Tensor<float> host({4}, MemoryLocation::HOST);
    copy_tensor_data(host, row_view);

    std::vector<float> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*4).wait();

    for (uint64_t i=0; i<4; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], 4.f+i);
    }
}

/**
 * @test TENSOR.alias_view_constructor_extracts_patch
 * @brief Extracts a 2x2 patch from a 4x4 tensor.
 */
TEST(TENSOR, alias_view_constructor_extracts_patch)
{
    Tensor<float> t({4,4}, MemoryLocation::HOST);
    std::vector<float> vals(16);
    for (uint64_t i=0; i<16; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    t = vals;

    Tensor<float> patch(t, {1,1}, {2,2}, {t.m_strides[0], t.m_strides[1]});

    EXPECT_EQ(patch.m_dimensions, (std::vector<uint64_t>{2,2}));
    EXPECT_EQ(patch.m_strides, (std::vector<uint64_t>{4,1}));

    Tensor<float> host({2,2}, MemoryLocation::HOST);
    copy_tensor_data(host, patch);

    std::vector<float> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*4).wait();

    EXPECT_FLOAT_EQ(out[0], 5.f);
    EXPECT_FLOAT_EQ(out[1], 6.f);
    EXPECT_FLOAT_EQ(out[2], 9.f);
    EXPECT_FLOAT_EQ(out[3], 10.f);
}

/**
 * @test TENSOR.alias_view_constructor_mutation_reflects
 * @brief Mutating a view updates the underlying owner tensor.
 */
TEST(TENSOR, alias_view_constructor_mutation_reflects)
{
    Tensor<float> t({2,3}, MemoryLocation::HOST);
    std::vector<float> vals = {0,1,2,3,4,5};
    t = vals;

    Tensor<float> col1(t, {0,1}, {2}, {t.m_strides[0]});

    col1[0] = 100.f;
    col1[1] = 200.f;

    Tensor<float> host({2,3}, MemoryLocation::HOST);
    copy_tensor_data(host, t);

    std::vector<float> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*6).wait();

    EXPECT_FLOAT_EQ(out[1], 100.f);
    EXPECT_FLOAT_EQ(out[4], 200.f);
}

/**
 * @test TENSOR.alias_view_constructor_broadcasting
 * @brief Create a broadcasted view (stride 0) of a 1D tensor.
 */
TEST(TENSOR, alias_view_constructor_broadcasting)
{
    Tensor<float> t({3}, MemoryLocation::HOST);
    t = std::vector<float>{10,20,30};

    Tensor<float> broadcast_view(t, {1}, {4}, {0});

    EXPECT_EQ(broadcast_view.m_dimensions, std::vector<uint64_t>{4});
    EXPECT_EQ(broadcast_view.m_strides, std::vector<uint64_t>{0});

    Tensor<float> host({4}, MemoryLocation::HOST);
    copy_tensor_data(host, broadcast_view);

    std::vector<float> out(4);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*4).wait();

    for (uint64_t i=0; i<4; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], 20.f);
    }
}

/**
 * @test TENSOR.alias_view_constructor_reshaping
 * @brief Create a 2x3 view from a 1D 6-element tensor using strides.
 */
TEST(TENSOR, alias_view_constructor_reshaping)
{
    Tensor<float> t({6}, MemoryLocation::HOST);
    t = std::vector<float>{0,1,2,3,4,5};

    Tensor<float> reshaped(t, {0}, {2,3}, {3,1});

    EXPECT_EQ(reshaped.m_dimensions, (std::vector<uint64_t>{2,3}));
    EXPECT_EQ(reshaped.m_strides, (std::vector<uint64_t>{3,1}));


    Tensor<float> host({2,3}, MemoryLocation::HOST);
    copy_tensor_data(host, reshaped);

    std::vector<float> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*6).wait();

    EXPECT_FLOAT_EQ(out[0], 0.f);
    EXPECT_FLOAT_EQ(out[1], 1.f);
    EXPECT_FLOAT_EQ(out[2], 2.f);
    EXPECT_FLOAT_EQ(out[3], 3.f);
    EXPECT_FLOAT_EQ(out[4], 4.f);
    EXPECT_FLOAT_EQ(out[5], 5.f);
}

/**
 * @test TENSOR.alias_view_constructor_out_of_bounds
 * @brief Check exceptions on invalid start indices or dims.
 */
TEST(TENSOR, alias_view_constructor_out_of_bounds)
{
    Tensor<float> t({2,2}, MemoryLocation::HOST);
    t = std::vector<float>{1,2,3,4};

    EXPECT_THROW(Tensor<float>(t,{2,0},{1},{2}), std::out_of_range);
    EXPECT_THROW(Tensor<float>(t,{0,0},{3},{2}), std::out_of_range);
    EXPECT_THROW(Tensor<float>(t,{0,0},{2},{2,1}), std::invalid_argument);
}

/**
 * @test TENSOR.alias_view_constructor_zero_rank
 * @brief Creating a view with zero rank should throw.
 */
TEST(TENSOR, alias_view_constructor_zero_rank)
{
    Tensor<float> t({2,2}, MemoryLocation::HOST);
    t = std::vector<float>{1,2,3,4};

    EXPECT_THROW(Tensor<float>(t, {0,0}, {}, {}), std::invalid_argument);
}

/**
 * @test TENSOR.alias_view_constructor_zero_dim
 * @brief Creating a view with a zero dimension should throw.
 */
TEST(TENSOR, alias_view_constructor_zero_dim)
{
    Tensor<float> t({2,2}, MemoryLocation::HOST);
    t = std::vector<float>{1,2,3,4};

    EXPECT_THROW(Tensor<float>(t, {0,0}, {2,0}, {1,1}), std::invalid_argument);
}

/**
 * @test TENSOR.alias_view_constructor_stride_overflow
 * @brief Creating a view that triggers stride*(dim-1) overflow should throw.
 */
TEST(TENSOR, alias_view_constructor_stride_overflow)
{
    Tensor<float> t({2,2}, MemoryLocation::HOST);
    t = std::vector<float>{1,2,3,4};

    std::vector<uint64_t> huge_stride =
        {std::numeric_limits<uint64_t>::max(), 1};
    EXPECT_THROW(Tensor<float>(t, {0,0}, {2,2}, huge_stride),
        std::overflow_error);
}

/**
 * @test TENSOR.alias_view_constructor_uninitialized_owner
 * @brief Creating a view from an uninitialized tensor should throw.
 */
TEST(TENSOR, alias_view_constructor_uninitialized_owner)
{
    Tensor<float> t;
    EXPECT_THROW(Tensor<float>(t, {0,0}, {2,2}, {2,1}), std::runtime_error);
}

/**
 * @test TENSOR.alias_view_constructor_multi_dim_broadcast
 * @brief Broadcasting a single row/column in 2D.
 */
TEST(TENSOR, alias_view_constructor_multi_dim_broadcast)
{
    Tensor<float> t({2,2}, MemoryLocation::HOST);
    t = std::vector<float>{1,2,3,4};

    Tensor<float> broadcast_view(t, {0,0}, {3,2}, {0,1});

    EXPECT_EQ(broadcast_view.m_dimensions, (std::vector<uint64_t>{3,2}));
    EXPECT_EQ(broadcast_view.m_strides, (std::vector<uint64_t>{0,1}));

    Tensor<float> host({3,2}, MemoryLocation::HOST);
    copy_tensor_data(host, broadcast_view);

    std::vector<float> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*6).wait();

    EXPECT_FLOAT_EQ(out[0], 1.f);
    EXPECT_FLOAT_EQ(out[1], 2.f);
    EXPECT_FLOAT_EQ(out[2], 1.f);
    EXPECT_FLOAT_EQ(out[3], 2.f);
    EXPECT_FLOAT_EQ(out[4], 1.f);
    EXPECT_FLOAT_EQ(out[5], 2.f);
}

/**
 * @test TENSOR.alias_view_constructor_stride0_invalid
 * @brief Using stride 0 for non-broadcast dimension >1
 * should still work as broadcast.
 */
TEST(TENSOR, alias_view_constructor_stride0_invalid)
{
    Tensor<float> t({2}, MemoryLocation::HOST);
    t = std::vector<float>{42, 99};

    Tensor<float> view(t, {0}, {2}, {0});

    EXPECT_EQ(view.m_dimensions, (std::vector<uint64_t>{2}));
    EXPECT_EQ(view.m_strides, (std::vector<uint64_t>{0}));

    Tensor<float> host({2}, MemoryLocation::HOST);
    copy_tensor_data(host, view);

    std::vector<float> out(2);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*2).wait();

    EXPECT_FLOAT_EQ(out[0], 42.f);
    EXPECT_FLOAT_EQ(out[1], 42.f);
}

/**
 * @test TENSOR.alias_view_constructor_from_view
 * @brief Verifies aliasing with custom multi-dimensional strides by simulating
 * a transpose. Starts from a 2x3 tensor (row-major), then creates
 * an alias with shape (3,2) and strides (1,3). This alias should
 * present the transposed view of the original tensor without
 * reordering memory.
 */
TEST(TENSOR, alias_view_constructor_from_view) {
    // Owner tensor 2x3
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    std::vector<float> vals = { 0.f, 1.f, 2.f,
                                3.f, 4.f, 5.f };
    t = vals;

    Tensor<float> transposed_alias(t, {0,0}, {3,2}, {1,3});

    EXPECT_EQ(transposed_alias.m_dimensions, std::vector<uint64_t>({3,2}));
    EXPECT_EQ(transposed_alias.m_strides, std::vector<uint64_t>({1,3}));

    Tensor<float> host({3,2}, MemoryLocation::HOST);
    copy_tensor_data(host, transposed_alias);

    std::vector<float> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*6).wait();

    // Expected transpose:
    // [ [0,3],
    //   [1,4],
    //   [2,5] ]
    EXPECT_FLOAT_EQ(out[0*2 + 0], 0.f);
    EXPECT_FLOAT_EQ(out[0*2 + 1], 3.f);
    EXPECT_FLOAT_EQ(out[1*2 + 0], 1.f);
    EXPECT_FLOAT_EQ(out[1*2 + 1], 4.f);
    EXPECT_FLOAT_EQ(out[2*2 + 0], 2.f);
    EXPECT_FLOAT_EQ(out[2*2 + 1], 5.f);
}

/**
 * @test TENSOR.operator_equals_copy_assignment
 * @brief Tests copy assignment operator.
 */
TEST(TENSOR, operator_equals_copy_assignment)
{
    Tensor<float> t1({2, 2}, MemoryLocation::DEVICE);
    std::vector<float> values = {9.0f, 10.0f, 11.0f, 12.0f};
    t1 = values;

    Tensor<float> t2;
    t2 = t1;

    EXPECT_EQ(t2.m_mem_loc, t1.m_mem_loc);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), t2.m_p_data.get(), sizeof(float) * 4).wait();

    EXPECT_EQ(host[0], 9.0f);
    EXPECT_EQ(host[1], 10.0f);
    EXPECT_EQ(host[2], 11.0f);
    EXPECT_EQ(host[3], 12.0f);
}

/**
 * @test TENSOR.operator_equals_copy_from_default
 * @brief Assigning from a default-constructed tensor
 * yields an empty owning tensor.
 */
TEST(TENSOR, operator_equals_copy_from_default)
{
    Tensor<float> dst({2, 2}, MemoryLocation::HOST);
    std::vector<float> v = {7.0f, 8.0f, 9.0f, 10.0f};
    dst = v;

    Tensor<float> src;

    dst = src;

    EXPECT_TRUE(dst.m_dimensions.empty());
    EXPECT_TRUE(dst.m_strides.empty());
    EXPECT_TRUE(dst.m_own_data);
    EXPECT_EQ(dst.m_p_data, nullptr);
}

/**
 * @test TENSOR.operator_equals_copy_self_assignment
 * @brief Self-assignment must be safe and preserve data.
 */
TEST(TENSOR, operator_equals_copy_self_assignment)
{
    Tensor<float> t({2,2}, MemoryLocation::DEVICE);
    std::vector<float> v = {1.0f,2.0f,3.0f,4.0f};
    t = v;

    t = t;

    std::vector<float> host(4);
    g_sycl_queue.memcpy(host.data(), t.m_p_data.get(), sizeof(float)*4).wait();
    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f);
    EXPECT_FLOAT_EQ(host[3], 4.0f);
}

/**
 * @test TENSOR.operator_equals_copy_from_view
 * @brief Assigning from a view (non-owning) yields
 * a non-owning alias to same buffer.
 */
TEST(TENSOR, operator_equals_copy_from_view)
{
    Tensor<float> owner({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = { 10,11,12, 20,21,22 };
    owner = vals;

    Tensor<float> view(owner, {1,1}, {1,2});
    ASSERT_FALSE(view.m_own_data);

    Tensor<float> dst;
    dst = view;

    EXPECT_FALSE(dst.m_own_data);
    EXPECT_EQ(dst.m_dimensions, view.m_dimensions);
    EXPECT_EQ(dst.m_strides, view.m_strides);
    EXPECT_EQ(dst.m_p_data.get(), view.m_p_data.get());

    std::vector<float> out(2);
    g_sycl_queue.memcpy(out.data(), dst.m_p_data.get(), sizeof(float)*2).wait();
    EXPECT_FLOAT_EQ(out[0], vals[4]);
    EXPECT_FLOAT_EQ(out[1], vals[5]);
}

/**
 * @test TENSOR.operator_equals_move
 * @brief Basic move-assignment: resources are transferred and source is emptied.
 */
TEST(TENSOR, operator_equals_move)
{
    Tensor<float> src({2, 2}, MemoryLocation::HOST);
    std::vector<float> values = {13.0f, 14.0f, 15.0f, 16.0f};
    src = values;

    float* original_ptr = src.m_p_data.get();
    MemoryLocation original_loc = src.m_mem_loc;

    Tensor<float> dst;
    dst = std::move(src);

    EXPECT_EQ(dst.m_p_data.get(), original_ptr);
    EXPECT_EQ(dst.m_mem_loc, original_loc);

    EXPECT_EQ(src.m_p_data.get(), nullptr);

    std::vector<float> host(4);
    std::memcpy(host.data(), dst.m_p_data.get(), sizeof(float) * 4);
    EXPECT_FLOAT_EQ(host[0], 13.0f);
    EXPECT_FLOAT_EQ(host[1], 14.0f);
    EXPECT_FLOAT_EQ(host[2], 15.0f);
    EXPECT_FLOAT_EQ(host[3], 16.0f);
}

/**
 * @test TENSOR.operator_equals_move_from_default
 * @brief Moving from a default-constructed (empty) tensor
 * makes destination empty.
 */
TEST(TENSOR, operator_equals_move_from_default)
{
    Tensor<float> dst({2, 2}, MemoryLocation::HOST);
    std::vector<float> vals = {7.0f, 8.0f, 9.0f, 10.0f};
    dst = vals;

    Tensor<float> src;

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
 * @test TENSOR.operator_equals_move_self_assignment
 * @brief Self move-assignment must be safe and preserve contents.
 */
TEST(TENSOR, operator_equals_move_self_assignment)
{
    Tensor<float> t({2,2}, MemoryLocation::HOST);
    std::vector<float> v = {1.0f,2.0f,3.0f,4.0f};
    t = v;

    t = std::move(t);

    std::vector<float> host(4);
    std::memcpy(host.data(), t.m_p_data.get(), sizeof(float) * 4);
    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f);
    EXPECT_FLOAT_EQ(host[3], 4.0f);
}

/**
 * @test TENSOR.operator_equals_move_from_view
 * @brief Moving a non-owning view transfers
 * the aliasing shared_ptr to destination.
 */
TEST(TENSOR, operator_equals_move_from_view)
{
    Tensor<float> owner({2,3}, MemoryLocation::HOST);
    std::vector<float> vals = { 10.0f,11.0f,12.0f, 20.0f,21.0f,22.0f };
    owner = vals;

    Tensor<float> view(owner, {1,1}, {1,2});
    ASSERT_FALSE(view.m_own_data);

    float* view_ptr = view.m_p_data.get();

    Tensor<float> dst;
    dst = std::move(view);

    EXPECT_FALSE(dst.m_own_data);
    EXPECT_EQ(dst.m_p_data.get(), view_ptr);

    EXPECT_EQ(view.m_p_data.get(), nullptr);
    EXPECT_TRUE(view.m_own_data);

    std::vector<float> out(2);
    std::memcpy(out.data(), dst.m_p_data.get(), sizeof(float) * 2);
    EXPECT_FLOAT_EQ(out[0], vals[4]);
    EXPECT_FLOAT_EQ(out[1], vals[5]);
}

/**
 * @test TENSOR.operator_equals_vector_assignment
 * @brief Tests assignment from flat std::vector.
 */
TEST(TENSOR, operator_equals_vector_assignment)
{
    std::vector<float> values = {3.3f, 3.4f, 3.5f, 3.6f};

    Tensor<float> t({2, 2});
    t = values;

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), t.m_p_data.get(), sizeof(float) * 4).wait();

    EXPECT_FLOAT_EQ(host[0], 3.3f);
    EXPECT_FLOAT_EQ(host[1], 3.4f);
    EXPECT_FLOAT_EQ(host[2], 3.5f);
    EXPECT_FLOAT_EQ(host[3], 3.6f);
}

/**
 * @test TENSOR.operator_equals_vector_size_mismatch_throws
 * @brief Throws if assigning flat vector with incorrect size.
 */
TEST(TENSOR, operator_equals_vector_size_mismatch_throws)
{
    Tensor<float> t({2, 2});
    std::vector<float> values = {1.0f, 2.0f};

    EXPECT_THROW({
        t = values;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_equals_vector_assign_to_default_throws
 * @brief Assigning a vector to a default-constructed tensor must throw.
 */
TEST(TENSOR, operator_equals_vector_assign_to_default_throws)
{
    Tensor<float> t_default;
    std::vector<float> values = { 1.0f };

    EXPECT_THROW({
        t_default = values;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_equals_scalar_assignment_and_conversion
 * @brief Tests operator=(float_t)
 * and operator float_t() on 1-element tensors / views.
 */
TEST(TENSOR, operator_equals_scalar_assignment_and_conversion)
{
    Tensor<float> s({1}, MemoryLocation::HOST);
    s = 5.5f;
    float sval = s;
    EXPECT_FLOAT_EQ(sval, 5.5f);

    Tensor<float> t({2, 2}, MemoryLocation::HOST);
    std::vector<float> vals = { 1.0f, 2.0f, 3.0f, 4.0f };
    t = vals;

    t[1][1] = 99.25f;
    float read_back = t[1][1];
    EXPECT_FLOAT_EQ(read_back, 99.25f);
}

/**
 * @test TENSOR.operator_equals_scalar_assignment_to_default_constructed_tensor
 * @brief Tests that assigning a scalar to a default-constructed tensor
 * automatically initializes it as a scalar tensor of shape {1} and stores
 * the assigned value correctly.
 *
 * Also verifies that assigning a scalar to a tensor with more than one element
 * throws std::invalid_argument.
 */
TEST(TENSOR, operator_equals_scalar_assignment_to_default_constructed_tensor) {
    Tensor<float> t;
    t = 42.5f;

    ASSERT_EQ(t.m_dimensions.size(), 1);
    EXPECT_EQ(t.m_dimensions[0], 1);

    float val = static_cast<float>(t);
    EXPECT_FLOAT_EQ(val, 42.5f);

    Tensor<float> big({2, 2});
    EXPECT_THROW(big = 1.0f, std::invalid_argument);
}

/**
 * @test TENSOR.operator_equals_scalar_assignment_wrong_size_throws
 * @brief Assigning a scalar to a non-1-element tensor must throw.
 */
TEST(TENSOR, operator_equals_scalar_assignment_wrong_size_throws)
{
    Tensor<float> t({2, 2}, MemoryLocation::HOST);
    EXPECT_THROW(
    {
        t = 3.14f;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_brackets_index_chain_assign_and_read
 * @brief Tests that chained operator[] returns views
 * and allows assignment and reading.
 */
TEST(TENSOR, operator_brackets_index_chain_assign_and_read)
{
    Tensor<float> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<float> vals(2 * 3 * 4);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    t = vals;

    t[1][2][3] = 420.0f;

    float v = t[1][2][3];
    EXPECT_FLOAT_EQ(v, 420.0f);

    float host_val = 0.0f;
    uint64_t offset =
        1 * t.m_strides[0] + 2 * t.m_strides[1] + 3 * t.m_strides[2];
    g_sycl_queue.memcpy
        (&host_val, t.m_p_data.get() + offset, sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host_val, 420.0f);
}

/**
 * @test TENSOR.operator_brackets_index_chain_const_access
 * @brief Tests that const Tensor supports chained operator[] reading.
 */
TEST(TENSOR, operator_brackets_index_chain_const_access)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::HOST);
    std::vector<float> vals(8);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 10);
    }
    t = vals;

    const Tensor<float>& ct = t;

    float a = ct[1][1][1];
    EXPECT_FLOAT_EQ(a, vals[1 * 4 + 1 * 2 + 1]);
}

/**
 * @test TENSOR.operator_brackets_index_out_of_bounds_throws
 * @brief operator[] should throw when index is out of range.
 */
TEST(TENSOR, operator_brackets_index_out_of_bounds_throws)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::HOST);

    EXPECT_THROW(
    {
        (void)t[2];
    }, std::out_of_range);

    EXPECT_THROW(
    {
        (void)t[1][2];
    }, std::out_of_range);

    EXPECT_THROW(
    {
        (void)t[1][1][2];
    }, std::out_of_range);
}

/**
 * @test TENSOR.operator_brackets_view_constructor_middle_chunk_write_and_read
 * @brief Create a 3D tensor, take a chunk in the middle of a channel
 * with the view constructor, write via chained operator[] on the view
 * and verify reads from both the view and the original tensor
 * at the expected positions.
 */
TEST(TENSOR, operator_brackets_view_constructor_middle_chunk_write_and_read)
{
    Tensor<float> img({3, 6, 7}, MemoryLocation::HOST);

    std::vector<float> vals(3 * 6 * 7);
    for (uint64_t c = 0; c < 3; ++c)
    {
        for (uint64_t h = 0; h < 6; ++h)
        {
            for (uint64_t w = 0; w < 7; ++w)
            {
                uint64_t idx = c * (6 * 7) + h * 7 + w;
                vals[idx] = static_cast<float>(c * 10000 + h * 100 + w);
            }
        }
    }
    img = vals;

    std::vector<uint64_t> start = { 1, 2, 3 };
    std::vector<uint64_t> view_shape = { 3, 2 };
    Tensor<float> chunk(img, start, view_shape);

    EXPECT_EQ(chunk.m_dimensions, std::vector<uint64_t>({3, 2}));

    chunk[0][0] = 9999.5f;
    chunk[2][1] = 8888.25f;

    float a = chunk[0][0];
    float b = chunk[2][1];

    EXPECT_FLOAT_EQ(a, 9999.5f);
    EXPECT_FLOAT_EQ(b, 8888.25f);

    uint64_t off_a =
        start[0] * img.m_strides[0] + (start[1] + 0) * img.m_strides[1] +
        (start[2] + 0) * img.m_strides[2];

    uint64_t off_b =
        start[0] * img.m_strides[0] + (start[1] + 2) * img.m_strides[1] +
        (start[2] + 1) * img.m_strides[2];

    float host_a = 0.0f;
    float host_b = 0.0f;
    g_sycl_queue.memcpy
        (&host_a, img.m_p_data.get() + off_a, sizeof(float)).wait();

    g_sycl_queue.memcpy
        (&host_b, img.m_p_data.get() + off_b, sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host_a, 9999.5f);
    EXPECT_FLOAT_EQ(host_b, 8888.25f);

    float cval = chunk[1][0];

    uint64_t off_c =
        start[0] * img.m_strides[0] + (start[1] + 1) * img.m_strides[1] +
        (start[2] + 0) * img.m_strides[2];

    float host_c = 0.0f;
    g_sycl_queue.memcpy
        (&host_c, img.m_p_data.get() + off_c, sizeof(float)).wait();

    EXPECT_FLOAT_EQ(cval, host_c);
}

/**
 * @test TENSOR.operator_float_valid_scalar
 * @brief Tests implicit conversion to scalar for a 1-element tensor.
 */
TEST(TENSOR, operator_float_valid_scalar)
{
    Tensor<float> t({1}, MemoryLocation::HOST);
    float val = 42.5f;
    t = val;  // uses scalar assignment

    float converted = static_cast<float>(t);
    EXPECT_FLOAT_EQ(converted, 42.5f);
}

/**
 * @test TENSOR.operator_float_throws_no_dimensions
 * @brief Tests that converting a moved-from tensor throws (no dimensions).
 */
TEST(TENSOR, operator_float_throws_no_dimensions)
{
    Tensor<float> t1({1}, MemoryLocation::HOST);
    t1 = 3.14f;

    Tensor<float> t2 = std::move(t1);
    EXPECT_THROW({
        float converted = static_cast<float>(t1);
        (void)converted;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_float_throws_multiple_elements_rank1
 * @brief Tests that conversion throws for rank-1 tensor with size > 1.
 */
TEST(TENSOR, operator_float_throws_multiple_elements_rank1)
{
    Tensor<float> t({3}, MemoryLocation::HOST);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    t = vals;

    EXPECT_THROW({
        float converted = static_cast<float>(t);
        (void)converted;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_float_throws_multi_dimensional
 * @brief Tests that conversion throws for rank > 1 tensor.
 */
TEST(TENSOR, operator_float_throws_multi_dimensional)
{
    Tensor<float> t({2, 2}, MemoryLocation::HOST);
    std::vector<float> vals = {1, 2, 3, 4};
    t = vals;

    EXPECT_THROW({
        float converted = static_cast<float>(t);
        (void)converted;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_addition
 * @brief Verifies element-wise addition on device memory.
 *
 * Creates two 2x3 device tensors A and B with known values, computes
 * C = A + B, copies the result back to host and checks every element
 * equals avals[i] + bvals[i].
 */
TEST(TENSOR, operator_addition)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    Tensor<float> B({2,3}, MemoryLocation::DEVICE);

    std::vector<float> avals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> bvals = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    A = avals;
    B = bvals;

    Tensor<float> C = A + B;

    const uint64_t total = 6;
    std::vector<float> ch(total);
    g_sycl_queue.memcpy
        (ch.data(), C.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(ch[i], avals[i] + bvals[i]);
    }
}

/**
 * @test TENSOR.operator_addition_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for addition.
 *
 * Creates A (2x3) and B (shape [3]) on device, computes R = A + B,
 * copies result to host and checks each element equals the expected
 * broadcasted sum.
 */
TEST(TENSOR, operator_addition_broadcasting_1d_to_2d)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    Tensor<float> B({3}, MemoryLocation::DEVICE);

    A = std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> R = A + B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {11.0f, 22.0f, 33.0f, 41.0f, 52.0f, 63.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_addition_broadcasting_scalar
 * @brief Verifies addition broadcasting with a scalar operand.
 *
 * Creates A (2x3) and B (shape [1]) on host, computes R = A + B,
 * copies result to host and checks each element equals the expected
 * scalar-added value.
 */
TEST(TENSOR, operator_addition_broadcasting_scalar)
{
    Tensor<float> A({2,3}, MemoryLocation::HOST);
    Tensor<float> B({1}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{5.0f};

    Tensor<float> R = A + B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_addition_with_views
 * @brief Verifies element-wise addition between a row view and a tensor.
 */
TEST(TENSOR, operator_addition_with_views)
{
    Tensor<float> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor<float> row0 = T[0];
    Tensor<float> addend({3}, MemoryLocation::DEVICE);
    addend = std::vector<float>{10.0f, 20.0f, 30.0f};

    Tensor<float> R = row0 + addend;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {11.0f, 22.0f, 33.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }

    // Sanity: original parent should remain unchanged.
    std::vector<float> parent_row(total);
    g_sycl_queue.memcpy
        (parent_row.data(), T.m_p_data.get(), sizeof(float) * total).wait();
    EXPECT_FLOAT_EQ(parent_row[0], 1.0f);
    EXPECT_FLOAT_EQ(parent_row[1], 2.0f);
    EXPECT_FLOAT_EQ(parent_row[2], 3.0f);
}

/**
 * @test TENSOR.operator_addition_incompatible_shapes
 * @brief Addition throws when operand shapes are incompatible.
 *
 * Creates A (2x3) and B (2x2) on host and expects an std::invalid_argument
 * when attempting to compute A + B.
 */
TEST(TENSOR, operator_addition_incompatible_shapes)
{
    Tensor<float> A({2,3}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::HOST);
    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_THROW({ Tensor<float> R = A + B; }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_addition_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE (addition).
 */
TEST(TENSOR, operator_addition_result_mem_location)
{
    Tensor<float> A({2,2}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A + B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TENSOR.operator_addition_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST (addition).
 */
TEST(TENSOR, operator_addition_both_host_result_mem_location)
{
    Tensor<float> A({2,2}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A + B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TENSOR.operator_addition_nan_inputs_throws
 * @brief Addition detects NaN inputs and triggers a runtime_error.
 *
 * Creates A and B on DEVICE with a NaN in A, then expects A + B to throw.
 */
TEST(TENSOR, operator_addition_nan_inputs_throws)
{
    Tensor<float> A({2}, MemoryLocation::DEVICE);
    Tensor<float> B({2}, MemoryLocation::DEVICE);

    float nan = std::numeric_limits<float>::quiet_NaN();
    A = std::vector<float>{1.0f, nan};
    B = std::vector<float>{2.0f, 3.0f};

    EXPECT_THROW({ Tensor<float> R = A + B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_addition_non_finite_result_throws
 * @brief Addition that overflows to Inf should trigger runtime_error.
 */
TEST(TENSOR, operator_addition_non_finite_result_throws)
{
    Tensor<float> A({1}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    float large = std::numeric_limits<float>::max();
    A = std::vector<float>{large};
    B = std::vector<float>{large};

    EXPECT_THROW({ Tensor<float> R = A + B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_addition_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} + B{3,1} (B aligned to {1,3,1}).
 *
 * Fills A with values 0..23 and B with {10,20,30}. Expects
 * R[i,j,k] == A[i,j,k] + B[j,0]. Result is expected to be on DEVICE.
 */
TEST(TENSOR, operator_addition_broadcasting_complex_alignment)
{
    Tensor<float> A({2,3,4}, MemoryLocation::DEVICE);
    Tensor<float> B({3,1},   MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<float> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<float>(i);
    }

    std::vector<float> bvals = {10.0f, 20.0f, 30.0f};

    A = avals;
    B = bvals;

    Tensor<float> R = A + B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            float offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] + offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        EXPECT_FLOAT_EQ(rh[idx], expected[idx]);
    }
}

/**
 * @test TENSOR.operator_addition_alias_view_noncontiguous
 * @brief Element-wise addition where the right operand
 * is a non-contiguous 1D alias view.
 *
 * owner B: {6} -> values {10,20,30,40,50,60}
 * A: {3} -> values {1,2,3}
 * view on B: start {0}, dims {3}, strides {2} -> maps to B[0],B[2],B[4]
 */
TEST(TENSOR, operator_addition_alias_view_noncontiguous)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({6}, MemoryLocation::DEVICE);

    std::vector<float> avals = {1.0f, 2.0f, 3.0f};
    std::vector<float> bvals = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    A = avals;
    B = bvals;

    Tensor<float> v(B, {0}, {3}, {2});

    Tensor<float> R = A + v;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {11.0f, 32.0f, 53.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }

    std::vector<float> b_host(6);
    g_sycl_queue.memcpy
        (b_host.data(), B.m_p_data.get(), sizeof(float) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(b_host[i], bvals[i]);
    }
}

/**
 * @test TENSOR.operator_addition_alias_view_broadcast
 * @brief Addition with a broadcasted alias view (stride 0).
 */
TEST(TENSOR, operator_addition_alias_view_broadcast)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, 2.0f, 3.0f};
    B = std::vector<float>{5.0f};

    Tensor<float> vb(B, {0}, {3}, {0});

    Tensor<float> R = A + vb;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {6.0f, 7.0f, 8.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_addition_alias_view_weird_strides
 * @brief Addition of a dense small tensor with a 2D alias view
 * with "weird" strides.
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Addend: {3,4} filled with scalar 3.0f => R = view + addend
 */
TEST(TENSOR, operator_addition_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<float> add({3,4}, MemoryLocation::DEVICE);

    std::vector<float> add_vals(12, 3.0f);
    add = add_vals;

    Tensor<float> R = view + add;

    Tensor<float> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<float> rh(12);
    g_sycl_queue.memcpy
        (rh.data(), hostR.m_p_data.get(), sizeof(float) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            EXPECT_FLOAT_EQ(rh[k], vals[flat] + 3.0f);
        }
    }
}


/**
 * @test TENSOR.operator_subtraction
 * @brief Verifies element-wise subtraction on device memory.
 *
 * Creates two 2x3 device tensors A and B with known values, computes
 * D = A - B, copies the result back to host and checks every element
 * equals avals[i] - bvals[i].
 */
TEST(TENSOR, operator_subtraction)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    Tensor<float> B({2,3}, MemoryLocation::DEVICE);

    std::vector<float> avals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> bvals = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    A = avals;
    B = bvals;

    Tensor<float> D = A - B;

    const uint64_t total = 6;
    std::vector<float> dh(total);
    g_sycl_queue.memcpy
        (dh.data(), D.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(dh[i], avals[i] - bvals[i]);
    }
}

/**
 * @test TENSOR.operator_subtraction_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for subtraction.
 *
 * Creates A (2x3) and B (shape [3]) on device, computes R = A - B,
 * copies result to host and checks each element equals the expected
 * broadcasted difference.
 */
TEST(TENSOR, operator_subtraction_broadcasting_1d_to_2d)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    Tensor<float> B({3}, MemoryLocation::DEVICE);

    A = std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> R = A - B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {9.0f, 18.0f, 27.0f, 39.0f, 48.0f, 57.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_subtraction_broadcasting_scalar
 * @brief Verifies addition broadcasting with a scalar operand for subtraction.
 *
 * Creates A (2x3) and B (shape [1]) on host, computes R = A - B,
 * copies result to host and checks each element equals the expected
 * scalar-subtracted value.
 */
TEST(TENSOR, operator_subtraction_broadcasting_scalar)
{
    Tensor<float> A({2,3}, MemoryLocation::HOST);
    Tensor<float> B({1}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{5.0f};

    Tensor<float> R = A - B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_subtraction_with_views
 * @brief Verifies element-wise subtraction between a row view and a tensor.
 */
TEST(TENSOR, operator_subtraction_with_views)
{
    Tensor<float> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    Tensor<float> row0 = T[0];
    Tensor<float> subtrahend({3}, MemoryLocation::DEVICE);
    subtrahend = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> R = row0 - subtrahend;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {9.0f, 18.0f, 27.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_subtraction_incompatible_shapes
 * @brief Subtraction throws when operand shapes are incompatible.
 *
 * Creates A (2x3) and B (2x2) on host and expects an std::invalid_argument
 * when attempting to compute A - B.
 */
TEST(TENSOR, operator_subtraction_incompatible_shapes)
{
    Tensor<float> A({2,3}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::HOST);
    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_THROW({ Tensor<float> R = A - B; }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_subtraction_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE (subtraction).
 */
TEST(TENSOR, operator_subtraction_result_mem_location)
{
    Tensor<float> A({2,2}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A - B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TENSOR.operator_subtraction_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST (subtraction).
 */
TEST(TENSOR, operator_subtraction_both_host_result_mem_location)
{
    Tensor<float> A({2,2}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A - B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TENSOR.operator_subtraction_nan_inputs_throws
 * @brief Subtraction detects NaN inputs and triggers a runtime_error.
 *
 * Creates A and B on DEVICE with a NaN in A, then expects A - B to throw.
 */
TEST(TENSOR, operator_subtraction_nan_inputs_throws)
{
    Tensor<float> A({2}, MemoryLocation::DEVICE);
    Tensor<float> B({2}, MemoryLocation::DEVICE);

    float nan = std::numeric_limits<float>::quiet_NaN();
    A = std::vector<float>{1.0f, nan};
    B = std::vector<float>{2.0f, 3.0f};

    EXPECT_THROW({ Tensor<float> R = A - B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_subtraction_non_finite_result_throws
 * @brief Subtraction that overflows to Inf should trigger runtime_error.
 */
TEST(TENSOR, operator_subtraction_non_finite_result_throws)
{
    Tensor<float> A({1}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    float large = std::numeric_limits<float>::max();
    A = std::vector<float>{ large };
    B = std::vector<float>{ -large };

    EXPECT_THROW({ Tensor<float> R = A - B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_subtraction_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} - B{3,1} (B aligned to {1,3,1}).
 *
 * Fills A with values 0..23 and B with {10,20,30}. Expects
 * R[i,j,k] == A[i,j,k] - B[j,0]. Result is expected to be on DEVICE.
 */
TEST(TENSOR, operator_subtraction_broadcasting_complex_alignment)
{
    Tensor<float> A({2, 3, 4}, MemoryLocation::DEVICE);
    Tensor<float> B({3, 1},   MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<float> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<float>(i);
    }

    std::vector<float> bvals = {10.0f, 20.0f, 30.0f};

    A = avals;
    B = bvals;

    Tensor<float> R = A - B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            float offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] - offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        EXPECT_FLOAT_EQ(rh[idx], expected[idx]);
    }
}

/**
 * @test TENSOR.operator_subtraction_alias_view_noncontiguous
 * @brief Element-wise subtraction where the right operand
 * is a non-contiguous 1D alias view.
 *
 * owner B: {6} -> values {10,20,30,40,50,60}
 * A: {3} -> values {1,2,3}
 * view on B: start {0}, dims {3}, strides {2} -> maps to B[0],B[2],B[4]
 */
TEST(TENSOR, operator_subtraction_alias_view_noncontiguous)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({6}, MemoryLocation::DEVICE);

    std::vector<float> avals = {1.0f, 2.0f, 3.0f};
    std::vector<float> bvals = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    A = avals;
    B = bvals;

    Tensor<float> v(B, {0}, {3}, {2});

    Tensor<float> R = A - v;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {-9.0f, -28.0f, -47.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }

    std::vector<float> b_host(6);
    g_sycl_queue.memcpy
        (b_host.data(), B.m_p_data.get(), sizeof(float) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(b_host[i], bvals[i]);
    }
}

/**
 * @test TENSOR.operator_subtraction_alias_view_broadcast
 * @brief Subtraction with a broadcasted alias view (stride 0).
 */
TEST(TENSOR, operator_subtraction_alias_view_broadcast)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, 2.0f, 3.0f};
    B = std::vector<float>{5.0f};

    Tensor<float> vb(B, {0}, {3}, {0});

    Tensor<float> R = A - vb;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {-4.0f, -3.0f, -2.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_subtraction_alias_view_weird_strides
 * @brief Subtraction of a dense small tensor with a 2D alias view
 * with "weird" strides.
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Subtrahend: {3,4} filled with scalar 3.0f => R = view - subtrahend
 */
TEST(TENSOR, operator_subtraction_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<float> sub({3,4}, MemoryLocation::DEVICE);

    std::vector<float> sub_vals(12, 3.0f);
    sub = sub_vals;

    Tensor<float> R = view - sub;

    Tensor<float> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<float> rh(12);
    g_sycl_queue.memcpy
        (rh.data(), hostR.m_p_data.get(), sizeof(float) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            EXPECT_FLOAT_EQ(rh[k], vals[flat] - 3.0f);
        }
    }
}

/**
 * @test TENSOR.operator_multiplication
 * @brief Verifies element-wise multiplication on device memory.
 *
 * Creates two 2x3 device tensors A and B with known values, computes
 * E = A * B, copies the result back to host and checks every element
 * equals avals[i] * bvals[i].
 */
TEST(TENSOR, operator_multiplication)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    Tensor<float> B({2,3}, MemoryLocation::DEVICE);

    std::vector<float> avals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> bvals = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    A = avals;
    B = bvals;

    Tensor<float> E = A * B;

    const uint64_t total = 6;
    std::vector<float> eh(total);
    g_sycl_queue.memcpy
        (eh.data(), E.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(eh[i], avals[i] * bvals[i]);
    }
}

/**
 * @test TENSOR.operator_multiplication_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for multiplication.
 *
 * Creates A (2x3) and B (shape [3]) on device, computes R = A * B,
 * copies result to host and checks each element equals the expected
 * broadcasted product.
 */
TEST(TENSOR, operator_multiplication_broadcasting_1d_to_2d)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    Tensor<float> B({3}, MemoryLocation::DEVICE);

    A = std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> R = A * B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {10.0f, 40.0f, 90.0f, 40.0f, 100.0f, 180.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_multiplication_broadcasting_scalar
 * @brief Verifies multiplication broadcasting with a scalar operand.
 *
 * Creates A (2x3) and B (shape [1]) on host, computes R = A * B,
 * copies result to host and checks each element equals the expected
 * scalar-multiplied value.
 */
TEST(TENSOR, operator_multiplication_broadcasting_scalar)
{
    Tensor<float> A({2,3}, MemoryLocation::HOST);
    Tensor<float> B({1}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{5.0f};

    Tensor<float> R = A * B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_multiplication_with_views
 * @brief Verifies element-wise multiplication between a row view and a tensor.
 */
TEST(TENSOR, operator_multiplication_with_views)
{
    Tensor<float> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    Tensor<float> row0 = T[0];
    Tensor<float> multiplier({3}, MemoryLocation::DEVICE);
    multiplier = std::vector<float>{10.0f, 20.0f, 30.0f};

    Tensor<float> R = row0 * multiplier;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {20.0f, 60.0f, 120.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_multiplication_incompatible_shapes
 * @brief Multiplication throws when operand shapes are incompatible.
 *
 * Creates A (2x3) and B (2x2) on host and expects an std::invalid_argument
 * when attempting to compute A * B.
 */
TEST(TENSOR, operator_multiplication_incompatible_shapes)
{
    Tensor<float> A({2,3}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::HOST);
    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_THROW({ Tensor<float> R = A * B; }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_multiplication_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE (multiplication).
 */
TEST(TENSOR, operator_multiplication_result_mem_location)
{
    Tensor<float> A({2,2}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A * B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TENSOR.operator_multiplication_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST (multiplication).
 */
TEST(TENSOR, operator_multiplication_both_host_result_mem_location)
{
    Tensor<float> A({2,2}, MemoryLocation::HOST);
    Tensor<float> B({2,2}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A * B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TENSOR.operator_multiplication_nan_inputs_throws
 * @brief Multiplication detects NaN inputs and triggers a runtime_error.
 *
 * Creates A and B on DEVICE with a NaN in A, then expects A * B to throw.
 */
TEST(TENSOR, operator_multiplication_nan_inputs_throws)
{
    Tensor<float> A({2}, MemoryLocation::DEVICE);
    Tensor<float> B({2}, MemoryLocation::DEVICE);

    float nan = std::numeric_limits<float>::quiet_NaN();
    A = std::vector<float>{1.0f, nan};
    B = std::vector<float>{2.0f, 3.0f};

    EXPECT_THROW({ Tensor<float> R = A * B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_multiplication_non_finite_result_throws
 * @brief Non-finite multiplication result (overflow -> Inf)
 * triggers runtime_error.
 *
 * Multiplies max float by 2 on DEVICE and expects the operation to throw due
 * to non-finite result detection in the kernel.
 */
TEST(TENSOR, operator_multiplication_non_finite_result_throws)
{
    Tensor<float> A({1}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    float large = std::numeric_limits<float>::max();
    A = std::vector<float>{large};
    B = std::vector<float>{2.0f};

    EXPECT_THROW({ Tensor<float> R = A * B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_multiplication_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} * B{3,1} (B aligned to {1,3,1}).
 *
 * Fills A with values 0..23 and B with {10,20,30}. Expects
 * R[i,j,k] == A[i,j,k] * B[j,0]. Result is expected to be on DEVICE.
 */
TEST(TENSOR, operator_multiplication_broadcasting_complex_alignment)
{
    Tensor<float> A({2, 3, 4}, MemoryLocation::DEVICE);
    Tensor<float> B({3, 1},   MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<float> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<float>(i);
    }

    std::vector<float> bvals = {10.0f, 20.0f, 30.0f};

    A = avals;
    B = bvals;

    Tensor<float> R = A * B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            float offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] * offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        EXPECT_FLOAT_EQ(rh[idx], expected[idx]);
    }
}

/**
 * @test TENSOR.operator_multiplication_alias_view_noncontiguous
 * @brief Element-wise multiplication where the right operand
 * is a non-contiguous 1D alias view.
 *
 * owner B: {6} -> values {10,20,30,40,50,60}
 * A: {3} -> values {1,2,3}
 * view on B: start {0}, dims {3}, strides {2} -> maps to B[0],B[2],B[4]
 */
TEST(TENSOR, operator_multiplication_alias_view_noncontiguous)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({6}, MemoryLocation::DEVICE);

    std::vector<float> avals = {1.0f, 2.0f, 3.0f};
    std::vector<float> bvals = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    A = avals;
    B = bvals;

    Tensor<float> v(B, {0}, {3}, {2});

    Tensor<float> R = A * v;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {10.0f, 60.0f, 150.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }

    std::vector<float> b_host(6);
    g_sycl_queue.memcpy
        (b_host.data(), B.m_p_data.get(), sizeof(float) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(b_host[i], bvals[i]);
    }
}

/**
 * @test TENSOR.operator_multiplication_alias_view_broadcast
 * @brief Multiplication with a broadcasted alias view (stride 0).
 */
TEST(TENSOR, operator_multiplication_alias_view_broadcast)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, 2.0f, 3.0f};
    B = std::vector<float>{5.0f};

    Tensor<float> vb(B, {0}, {3}, {0});

    Tensor<float> R = A * vb;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {5.0f, 10.0f, 15.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_multiplication_alias_view_weird_strides
 * @brief Multiplication of a dense small tensor with a 2D alias
 * view with "weird" strides.
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4} (maps to specific flat indices)
 * Multiplier: {3,4} filled with scalar 3.0f => R = view * multiplier
 */
TEST(TENSOR, operator_multiplication_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<float> mul({3,4}, MemoryLocation::DEVICE);

    std::vector<float> mul_vals(12, 3.0f);
    mul = mul_vals;

    Tensor<float> R = view * mul;

    Tensor<float> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<float> rh(12);
    g_sycl_queue.memcpy
        (rh.data(), hostR.m_p_data.get(), sizeof(float) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            EXPECT_FLOAT_EQ(rh[k], vals[flat] * 3.0f);
        }
    }
}


/**
 * @test TENSOR.operator_division
 * @brief Verifies element-wise division on device memory.
 *
 * Creates two 2x3 device tensors A and B with known non-zero divisor
 * values, computes F = A / B, copies the result back to host and checks
 * every element equals avals[i] / bvals[i].
 */
TEST(TENSOR, operator_division)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    Tensor<float> B({2,3}, MemoryLocation::DEVICE);

    std::vector<float> avals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> bvals = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    A = avals;
    B = bvals;

    Tensor<float> F = A / B;

    const uint64_t total = 6;
    std::vector<float> fh(total);
    g_sycl_queue.memcpy
        (fh.data(), F.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(fh[i], avals[i] / bvals[i]);
    }
}

/**
 * @test TENSOR.operator_division_broadcasting_1d_to_2d
 * @brief Verifies broadcasting from 1-D to 2-D for division.
 *
 * Creates A (2x3) and B (shape [3]) on device, computes R = A / B,
 * copies result to host and checks each element equals the expected
 * broadcasted quotient.
 */
TEST(TENSOR, operator_division_broadcasting_1d_to_2d)
{
    Tensor<float> A({2, 3}, MemoryLocation::DEVICE);
    Tensor<float> B({3}, MemoryLocation::DEVICE);

    A = std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f};

    Tensor<float> R = A / B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {
        10.0f / 1.0f, 20.0f / 2.0f, 30.0f / 3.0f,
        40.0f / 1.0f, 50.0f / 2.0f, 60.0f / 3.0f
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_division_broadcasting_scalar
 * @brief Verifies division broadcasting with a scalar operand.
 *
 * Creates A (2x3) and B (shape [1]) on host (non-zero scalar),
 * computes R = A / B, copies result to host and checks each element
 * equals the expected scalar-divided value.
 */
TEST(TENSOR, operator_division_broadcasting_scalar)
{
    Tensor<float> A({2, 3}, MemoryLocation::HOST);
    Tensor<float> B({1}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{5.0f};

    Tensor<float> R = A / B;

    uint64_t total = 6;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {
        1.0f / 5.0f, 2.0f / 5.0f, 3.0f / 5.0f,
        4.0f / 5.0f, 5.0f / 5.0f, 6.0f / 5.0f
    };
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_division_with_views
 * @brief Verifies element-wise division between a row view and a tensor.
 */
TEST(TENSOR, operator_division_with_views)
{
    Tensor<float> T({2,3}, MemoryLocation::DEVICE);
    T = std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    Tensor<float> row0 = T[0];
    Tensor<float> divisor({3}, MemoryLocation::DEVICE);
    divisor = std::vector<float>{2.0f, 4.0f, 5.0f};

    Tensor<float> R = row0 / divisor;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {5.0f, 5.0f, 6.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_division_incompatible_shapes
 * @brief Division throws when operand shapes are incompatible.
 *
 * Creates A (2x3) and B (2x2) on host and expects an std::invalid_argument
 * when attempting to compute A / B.
 */
TEST(TENSOR, operator_division_incompatible_shapes)
{
    Tensor<float> A({2, 3}, MemoryLocation::HOST);
    Tensor<float> B({2, 2}, MemoryLocation::HOST);
    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    B = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_THROW({ Tensor<float> R = A / B; }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_division_result_mem_location
 * @brief Result memory is DEVICE if either operand is DEVICE (division).
 */
TEST(TENSOR, operator_division_result_mem_location)
{
    Tensor<float> A({2, 2}, MemoryLocation::HOST);
    Tensor<float> B({2, 2}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A / B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TENSOR.operator_division_both_host_result_mem_location
 * @brief Result memory is HOST when both operands are HOST (division).
 */
TEST(TENSOR, operator_division_both_host_result_mem_location)
{
    Tensor<float> A({2, 2}, MemoryLocation::HOST);
    Tensor<float> B({2, 2}, MemoryLocation::HOST);

    A = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    B = std::vector<float>{4.0f, 3.0f, 2.0f, 1.0f};

    Tensor<float> R = A / B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TENSOR.operator_division_by_zero_throws
 * @brief Division by zero in device kernel should trigger a runtime_error.
 *
 * Creates A and B on DEVICE where B contains a zero element, then attempts
 * R = A / B and expects a std::runtime_error thrown due to division-by-zero.
 */
TEST(TENSOR, operator_division_by_zero_throws)
{
    Tensor<float> A({2}, MemoryLocation::DEVICE);
    Tensor<float> B({2}, MemoryLocation::DEVICE);

    A = std::vector<float>{1.0f, -2.0f};
    B = std::vector<float>{0.0f, 1.0f};

    EXPECT_THROW({
        Tensor<float> R = A / B;
    }, std::runtime_error);
}

/**
 * @test TENSOR.operator_division_nan_inputs_throws
 * @brief Division detects NaN inputs and triggers a runtime_error.
 *
 * Creates A and B on DEVICE with a NaN in A, then expects A / B to throw.
 */
TEST(TENSOR, operator_division_nan_inputs_throws)
{
    Tensor<float> A({2}, MemoryLocation::DEVICE);
    Tensor<float> B({2}, MemoryLocation::DEVICE);

    float nan = std::numeric_limits<float>::quiet_NaN();
    A = std::vector<float>{1.0f, nan};
    B = std::vector<float>{2.0f, 3.0f};

    EXPECT_THROW({ Tensor<float> R = A / B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_division_non_finite_result_throws
 * @brief Division that overflows to Inf should trigger runtime_error.
 */
TEST(TENSOR, operator_division_non_finite_result_throws)
{
    Tensor<float> A({1}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    float large = std::numeric_limits<float>::max();
    float tiny = std::numeric_limits<float>::min();
    A = std::vector<float>{ large };
    B = std::vector<float>{ tiny };

    EXPECT_THROW({ Tensor<float> R = A / B; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_division_broadcasting_complex_alignment
 * @brief Verifies broadcasting for A{2,3,4} / B{3,1} (B aligned to {1,3,1}).
 *
 * Fills A with values 0..23 and B with {10,20,30}. Expects
 * R[i,j,k] == A[i,j,k] / B[j,0]. Result is expected to be on DEVICE.
 */
TEST(TENSOR, operator_division_broadcasting_complex_alignment)
{
    Tensor<float> A({2, 3, 4}, MemoryLocation::DEVICE);
    Tensor<float> B({3, 1},   MemoryLocation::DEVICE);

    const uint64_t total = 2 * 3 * 4;
    std::vector<float> avals(total);
    for (uint64_t i = 0; i < total; ++i)
    {
        avals[i] = static_cast<float>(i);
    }

    std::vector<float> bvals = {10.0f, 20.0f, 30.0f};

    A = avals;
    B = bvals;

    Tensor<float> R = A / B;
    EXPECT_EQ(R.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected(total);
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            float offset = bvals[j];
            for (uint64_t k = 0; k < 4; ++k)
            {
                uint64_t idx = i * 3 * 4 + j * 4 + k;
                expected[idx] = avals[idx] / offset;
            }
        }
    }

    for (uint64_t idx = 0; idx < total; ++idx)
    {
        EXPECT_FLOAT_EQ(rh[idx], expected[idx]);
    }
}

/**
 * @test TENSOR.operator_division_alias_view_noncontiguous
 * @brief Element-wise division where the right operand
 * is a non-contiguous 1D alias view (no zero divisors at sampled indices).
 *
 * owner B: {6} -> values {2,20,4,40,5,50}
 * A: {3} -> values {100,200,300}
 * view on B: start {0}, dims {3}, strides {2} -> maps to B[0],B[2],B[4] => 2,4,5
 */
TEST(TENSOR, operator_division_alias_view_noncontiguous)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({6}, MemoryLocation::DEVICE);

    std::vector<float> avals = {100.0f, 200.0f, 300.0f};
    std::vector<float> bvals = {2.0f, 20.0f, 4.0f, 40.0f, 5.0f, 50.0f};

    A = avals;
    B = bvals;

    Tensor<float> v(B, {0}, {3}, {2});

    Tensor<float> R = A / v;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {50.0f, 50.0f, 60.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }

    std::vector<float> b_host(6);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(), sizeof(float) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(b_host[i], bvals[i]);
    }
}

/**
 * @test TENSOR.operator_division_alias_view_broadcast
 * @brief Division with a broadcasted alias view (stride 0).
 */
TEST(TENSOR, operator_division_alias_view_broadcast)
{
    Tensor<float> A({3}, MemoryLocation::DEVICE);
    Tensor<float> B({1}, MemoryLocation::DEVICE);

    A = std::vector<float>{10.0f, 20.0f, 30.0f};
    B = std::vector<float>{10.0f};

    Tensor<float> vb(B, {0}, {3}, {0});

    Tensor<float> R = A / vb;

    const uint64_t total = 3;
    std::vector<float> rh(total);
    g_sycl_queue.memcpy
        (rh.data(), R.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {1.0f, 2.0f, 3.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(rh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_division_alias_view_weird_strides
 * @brief Division of a dense small tensor by a 2D alias view
 * with "weird" strides.
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Divisor: {3,4} filled with scalar 2.0f (non-zero) => R = view / divisor
 */
TEST(TENSOR, operator_division_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::DEVICE);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});
    Tensor<float> div({3,4}, MemoryLocation::DEVICE);

    std::vector<float> div_vals(12, 2.0f);
    div = div_vals;

    Tensor<float> R = view / div;

    Tensor<float> hostR({3,4}, MemoryLocation::HOST);
    copy_tensor_data(hostR, R);
    std::vector<float> rh(12);
    g_sycl_queue.memcpy
        (rh.data(), hostR.m_p_data.get(), sizeof(float) * 12).wait();

    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
        {
            uint64_t k = i * 4 + j;
            uint64_t flat = i * 13 + j * 4;
            EXPECT_FLOAT_EQ(rh[k], vals[flat] / 2.0f);
        }
    }
}

/**
 * @test TENSOR.operator_unary_negation
 * @brief Verifies element-wise unary negation.
 *
 * Creates A (2x2) on device, computes N = -A, copies result to host and
 * checks each element equals the negated input (including sign of zero).
 */
TEST(TENSOR, operator_unary_negation)
{
    Tensor<float> A({2,2}, MemoryLocation::DEVICE);
    A = std::vector<float>{1.0f, -2.0f, 3.5f, 0.0f};

    Tensor<float> N = -A;

    uint64_t total = 4;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy
        (nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {-1.0f, 2.0f, -3.5f, -0.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(nh[i], expected[i]);
    }
}

/**
 * @test TENSOR.operator_unary_negation_result_mem_location_device
 * @brief Result memory follows input memory (DEVICE case).
 *
 * Verifies that negating a DEVICE tensor yields a result also on DEVICE.
 */
TEST(TENSOR, operator_unary_negation_result_mem_location_device)
{
    Tensor<float> A({2,2}, MemoryLocation::DEVICE);
    std::vector<float> avals = {1.0f, -2.0f, 3.5f, 0.0f};
    A = avals;

    Tensor<float> N = -A;
    EXPECT_EQ(N.m_mem_loc, MemoryLocation::DEVICE);
}

/**
 * @test TENSOR.operator_unary_negation_result_mem_location_host
 * @brief Result memory follows input memory (HOST case).
 *
 * Verifies that negating a HOST tensor yields a result also on HOST.
 */
TEST(TENSOR, operator_unary_negation_result_mem_location_host)
{
    Tensor<float> A({2,2}, MemoryLocation::HOST);
    std::vector<float> avals = {1.0f, -2.0f, 3.5f, 0.0f};
    A = avals;

    Tensor<float> N = -A;
    EXPECT_EQ(N.m_mem_loc, MemoryLocation::HOST);
}

/**
 * @test TENSOR.operator_unary_negation_nan_input_throws
 * @brief NaN in input should cause the operation to throw.
 *
 * Kernel marks NaN and host code throws std::runtime_error; this test
 * verifies that behavior.
 */
TEST(TENSOR, operator_unary_negation_nan_input_throws)
{
    Tensor<float> A({2}, MemoryLocation::DEVICE);
    float nan = std::numeric_limits<float>::quiet_NaN();
    A = std::vector<float>{1.0f, nan};

    EXPECT_THROW({ Tensor<float> N = -A; }, std::runtime_error);
}

/**
 * @test TENSOR.operator_unary_negation_empty_tensor_throws
 * @brief Negation on a rank-0 tensor must throw std::invalid_argument.
 */
TEST(TENSOR, operator_unary_negation_empty_tensor_throws)
{
    Tensor<float> T;
    EXPECT_THROW({ Tensor<float> N = -T; }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_unary_negation_with_view
 * @brief Negating a view returns correct values and does not modify parent.
 *
 * Create a parent T, take a row view, negate the view and verify results.
 */
TEST(TENSOR, operator_unary_negation_with_view)
{
    Tensor<float> T({2,3}, MemoryLocation::DEVICE);
    std::vector<float> tvals = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
    T = tvals;

    Tensor<float> row0 = T[0];
    Tensor<float> N = -row0;

    const uint64_t total = 3;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy
        (nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {-1.0f, 2.0f, -3.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(nh[i], expected[i]);
    }

    // Sanity: parent must remain unchanged.
    std::vector<float> parent_buf(6);
    g_sycl_queue.memcpy
        (parent_buf.data(), T.m_p_data.get(), sizeof(float) * 6).wait();
    EXPECT_FLOAT_EQ(parent_buf[0], 1.0f);
    EXPECT_FLOAT_EQ(parent_buf[1], -2.0f);
    EXPECT_FLOAT_EQ(parent_buf[2], 3.0f);
}

/**
 * @test TENSOR.operator_unary_negation_sign_of_zero
 * @brief Check that negation preserves sign for zero (e.g. -0.0).
 *
 * Uses std::signbit to assert that the zero element has expected sign.
 */
TEST(TENSOR, operator_unary_negation_sign_of_zero)
{
    Tensor<float> A({2,2}, MemoryLocation::DEVICE);
    A = std::vector<float>{0.0f, 1.0f, -2.0f, 0.0f};

    Tensor<float> N = -A;

    const uint64_t total = 4;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy
        (nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    // Check values and signbit for indices where we expect -0.0.
    EXPECT_FLOAT_EQ(nh[0], -0.0f);
    EXPECT_TRUE(std::signbit(nh[0]));
    EXPECT_FLOAT_EQ(nh[3], -0.0f);
    EXPECT_TRUE(std::signbit(nh[3]));

    //Sanity checks.
    EXPECT_FLOAT_EQ(nh[1], -1.0f);
    EXPECT_FLOAT_EQ(nh[2], 2.0f);
}

/**
 * @test TENSOR.operator_unary_negation_non_contiguous_view_columns
 * @brief Negating a non-contiguous 2D view (a single column) produces the
 * correct contiguous result and does not modify the parent tensor.
 */
TEST(TENSOR, operator_unary_negation_non_contiguous_view_columns)
{
    Tensor<float> T({3,4}, MemoryLocation::DEVICE);
    std::vector<float> vals(12);
    for (uint64_t i = 0; i < 12; ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    T = vals;

    std::vector<uint64_t> start_indices = {0, 2};
    std::vector<uint64_t> view_shape = {3, 1};
    Tensor<float> col = Tensor<float>(T, start_indices, view_shape);

    Tensor<float> N = -col;

    const uint64_t total = 3 * 1;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy
        (nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t r = 0; r < 3; ++r)
    {
        float expected = -vals[r * 4 + 2];
        EXPECT_FLOAT_EQ(nh[r], expected);
    }

    std::vector<float> parent_buf(12);
    g_sycl_queue.memcpy
        (parent_buf.data(), T.m_p_data.get(), sizeof(float) * 12).wait();
    for (uint64_t i = 0; i < 12; ++i)
    {
        EXPECT_FLOAT_EQ(parent_buf[i], vals[i]);
    }
}

/**
 * @test TENSOR.operator_unary_negation_nan_outside_view
 * @brief If the parent contains a NaN outside the view region, negating the
 * view should succeed and not throw.
 */
TEST(TENSOR, operator_unary_negation_nan_outside_view)
{
    Tensor<float> T({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                            4.0f, 5.0f,std::numeric_limits<float>::quiet_NaN()};
    T = vals;
    std::vector<uint64_t> start_indices = {0, 0};
    std::vector<uint64_t> view_shape = {1, 3};
    Tensor<float> row0 = Tensor<float>(T, start_indices, view_shape);

    EXPECT_NO_THROW({ Tensor<float> N = -row0; });

    Tensor<float> N = -row0;
    const uint64_t total = 1 * 3;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy
        (nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(nh[i], -vals[i]);
    }

    std::vector<float> parent_buf(6);
    g_sycl_queue.memcpy
        (parent_buf.data(), T.m_p_data.get(), sizeof(float) * 6).wait();
    EXPECT_TRUE(std::isnan(parent_buf[5]));
}

/**
 * @test TENSOR.operator_unary_negation_view_of_view
 * @brief Negating a view-of-view works and does not change the parent tensor.
 */
TEST(TENSOR, operator_unary_negation_view_of_view)
{
    Tensor<float> T({2,4}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f};
    T = vals;

    Tensor<float> row1 = Tensor<float>(T, {1,0}, {4});
    Tensor<float> sub = Tensor<float>(row1, {1}, {2});
    Tensor<float> N = -sub;

    const uint64_t total = 2;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy
        (nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    EXPECT_FLOAT_EQ(nh[0], -vals[1 * 4 + 1]);
    EXPECT_FLOAT_EQ(nh[1], -vals[1 * 4 + 2]);

    std::vector<float> parent_buf(8);
    g_sycl_queue.memcpy
        (parent_buf.data(), T.m_p_data.get(), sizeof(float) * 8).wait();
    for (uint64_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(parent_buf[i], vals[i]);
    }
}

/**
 * @test TENSOR.operator_unary_negation_alias_view_noncontiguous
 * @brief Unary negation applied to a non-contiguous 1D alias view returns correct values
 * and does not modify the parent tensor.
 *
 * owner B: {6} -> values {1, -2, 3, -4, 5, -6}
 * view on B: start {0}, dims {3}, strides {2} -> maps to B[0],B[2],B[4] => 1,3,5
 */
TEST(TENSOR, operator_unary_negation_alias_view_noncontiguous)
{
    Tensor<float> B({6}, MemoryLocation::DEVICE);
    std::vector<float> bvals = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    B = bvals;

    Tensor<float> v(B, {0}, {3}, {2});

    Tensor<float> N = -v;

    const uint64_t total = 3;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy(nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {-1.0f, -3.0f, -5.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(nh[i], expected[i]);
    }

    // parent remains unchanged
    std::vector<float> b_host(6);
    g_sycl_queue.memcpy(b_host.data(), B.m_p_data.get(), sizeof(float) * 6).wait();
    for (uint64_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(b_host[i], bvals[i]);
    }
}

/**
 * @test TENSOR.operator_unary_negation_alias_view_broadcast
 * @brief Unary negation applied to a broadcasted alias view (stride 0).
 */
TEST(TENSOR, operator_unary_negation_alias_view_broadcast)
{
    Tensor<float> B({1}, MemoryLocation::DEVICE);
    B = std::vector<float>{5.0f};

    Tensor<float> vb(B, {0}, {3}, {0});

    Tensor<float> N = -vb;

    const uint64_t total = 3;
    std::vector<float> nh(total);
    g_sycl_queue.memcpy
        (nh.data(), N.m_p_data.get(), sizeof(float) * total).wait();

    std::vector<float> expected = {-5.0f, -5.0f, -5.0f};
    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(nh[i], expected[i]);
    }

    std::vector<float> b_host(1);
    g_sycl_queue.memcpy
        (b_host.data(), B.m_p_data.get(), sizeof(float) * 1).wait();
    EXPECT_FLOAT_EQ(b_host[0], 5.0f);
}

/**
 * @test TENSOR.operator_unary_negation_alias_view_weird_strides
 * @brief Unary negation applied to a
 * non-contiguous 2D alias view with odd strides.
 */
TEST(TENSOR, operator_unary_negation_alias_view_weird_strides)
{
    Tensor<float> T({5,20}, MemoryLocation::DEVICE);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i) vals[i] = static_cast<float>(i + 1);
    T = vals;

    Tensor<float> v(T, {1,2}, {2,3}, {13,4});

    Tensor<float> N = -v;

    Tensor<float> hostN({2,3}, MemoryLocation::HOST);
    copy_tensor_data(hostN, N);
    std::vector<float> nh(6);
    g_sycl_queue.memcpy
        (nh.data(), hostN.m_p_data.get(), sizeof(float) * 6).wait();

    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
        {
            uint64_t idx = i * 13 + j * 4 + (1 * 20 + 2);

            uint64_t start_flat = 1 * 20 + 2;
            uint64_t flat = start_flat + i * 13 + j * 4;
            uint64_t k = i * 3 + j;
            EXPECT_FLOAT_EQ(nh[k], -vals[flat]);
        }
    }

    std::vector<float> parent_buf(100);
    g_sycl_queue.memcpy
        (parent_buf.data(), T.m_p_data.get(), sizeof(float) * 100).wait();
    for (uint64_t i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(parent_buf[i], vals[i]);
    }
}

/**
 * @test TENSOR.to_host_to_device
 * @brief Moves a tensor from HOST to DEVICE
 * and verifies data integrity and metadata.
 */
TEST(TENSOR, to_host_to_device)
{
    Tensor<float> t({3, 5}, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : t.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<float> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<float>(i + 1);
    }

    t = values;

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    t.to(MemoryLocation::DEVICE);

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> host_data(total_size);
    g_sycl_queue.memcpy
        (host_data.data(), t.m_p_data.get(), sizeof(float) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        EXPECT_FLOAT_EQ(host_data[i], static_cast<float>(i + 1));
    }
}

/**
 * @test TENSOR.to_device_to_host
 * @brief Moves a tensor from DEVICE to HOST
 * and verifies data integrity and metadata.
 */
TEST(TENSOR, to_device_to_host)
{
    Tensor<float> t({4, 4}, MemoryLocation::DEVICE);

    uint64_t total_size = 1;
    for (uint64_t d : t.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<float> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<float>(i + 1);
    }

    t = values;

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    t.to(MemoryLocation::HOST);

    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    std::vector<float> host_data(total_size);
    g_sycl_queue.memcpy
        (host_data.data(), t.m_p_data.get(), sizeof(float) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        EXPECT_FLOAT_EQ(host_data[i], static_cast<float>(i + 1));
    }
}

/**
 * @test TENSOR.to_noop_when_already_in_target
 * @brief Calling to() with current memory location should do nothing.
 */
TEST(TENSOR, to_noop_when_already_in_target)
{
    Tensor<float> t_host({2, 2}, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : t_host.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<float> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<float>(i + 1);
    }

    t_host = values;

    t_host.to(MemoryLocation::HOST);
    EXPECT_EQ(t_host.m_mem_loc, MemoryLocation::HOST);

    std::vector<float> host_data_host(total_size);
    g_sycl_queue.memcpy(host_data_host.data(),
        t_host.m_p_data.get(), sizeof(float) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        EXPECT_FLOAT_EQ(host_data_host[i], static_cast<float>(i + 1));
    }


    Tensor<float> t_device({2, 2}, MemoryLocation::DEVICE);

    t_device = values;

    t_device.to(MemoryLocation::DEVICE);
    EXPECT_EQ(t_device.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<float> host_data_device(total_size);
    g_sycl_queue.memcpy(host_data_device.data(),
        t_device.m_p_data.get(), sizeof(float) * total_size).wait();

    for (uint64_t i = 0; i < total_size; ++i)
    {
        EXPECT_FLOAT_EQ(host_data_device[i], static_cast<float>(i + 1));
    }

}

/**
 * @test TENSOR.to_throws_for_view
 * @brief Calling to() on a view (non-owning tensor) should throw.
 */
TEST(TENSOR, to_throws_for_view)
{
    Tensor<float> t_owner({2, 2}, MemoryLocation::HOST);

    uint64_t total_size = 1;
    for (uint64_t d : t_owner.m_dimensions)
    {
        total_size *= d;
    }

    std::vector<float> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<float>(i + 1);
    }

    t_owner = values;

    Tensor<float> t_view(t_owner, {0, 0}, {2, 1});

    EXPECT_FALSE(t_view.m_own_data);

    EXPECT_THROW(t_view.to(MemoryLocation::DEVICE), std::runtime_error);
    EXPECT_THROW(t_view.to(MemoryLocation::HOST), std::runtime_error);
}

/**
 * @test TENSOR.to_throws_for_empty_tensor
 * @brief Calling to() on a tensor with no elements should throw.
 */
TEST(TENSOR, to_throws_for_empty_tensor)
{
    Tensor<float> t_empty;
    EXPECT_THROW(t_empty.to(MemoryLocation::DEVICE), std::invalid_argument);
}

/**
 * @test TENSOR.to_host_to_device_and_back
 * @brief Moves tensor from HOST to DEVICE and back, verifying data integrity.
 */
TEST(TENSOR, to_host_to_device_and_back)
{
    Tensor<float> t({4, 4}, MemoryLocation::HOST);
    uint64_t total_size = 16;
    std::vector<float> values(total_size);
    for (uint64_t i = 0; i < total_size; ++i)
    {
        values[i] = static_cast<float>(i + 10);
    }

    t = values;
    t.to(MemoryLocation::DEVICE);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    t.to(MemoryLocation::HOST);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::HOST);

    std::vector<float> host_data(total_size);
    g_sycl_queue.memcpy
        (host_data.data(), t.m_p_data.get(), sizeof(float) * total_size).wait();
    for (uint64_t i = 0; i < total_size; ++i)
    {
        EXPECT_FLOAT_EQ(host_data[i], static_cast<float>(i + 10));
    }
}

/**
 * @test TENSOR.reshape_preserves_linear_memory_and_strides
 * @brief Reshaping a tensor (2x3 -> 3x2) preserves the linear memory layout
 * (raw buffer contents unchanged) and recomputes strides correctly.
 */
TEST(TENSOR, reshape_preserves_linear_memory_and_strides)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
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
    std::vector<float> host_buf(total);
    g_sycl_queue.memcpy(host_buf.data(), A.m_p_data.get(),
                        sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(host_buf[i], vals[i]);
    }
}

/**
 * @test TENSOR.reshape_to_flat_vector_preserves_contents
 * @brief Reshape to a single-dimension tensor (1x6) preserves buffer and
 * produces stride [1].
 */
TEST(TENSOR, reshape_to_flat_vector_preserves_contents)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {10.0f, 11.0f, 12.0f,
                               13.0f, 14.0f, 15.0f};
    A = vals;

    std::vector<uint64_t> new_dims = {6};
    EXPECT_NO_THROW(A.reshape(new_dims));

    EXPECT_EQ(static_cast<uint64_t>(A.m_dimensions.size()), uint64_t{1});
    EXPECT_EQ(A.m_dimensions[0], uint64_t{6});

    ASSERT_EQ(static_cast<uint64_t>(A.m_strides.size()), uint64_t{1});
    EXPECT_EQ(A.m_strides[0], uint64_t{1});

    const uint64_t total = 6;
    std::vector<float> host_buf(total);
    g_sycl_queue.memcpy(host_buf.data(), A.m_p_data.get(),
                        sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(host_buf[i], vals[i]);
    }
}

/**
 * @test TENSOR.reshape_invalid_size_throws
 * @brief Reshaping to dimensions whose product differs from the original
 * total size must throw std::invalid_argument.
 */
TEST(TENSOR, reshape_invalid_size_throws)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    A = vals;

    std::vector<uint64_t> bad_dims = {4, 2};
    EXPECT_THROW({ A.reshape(bad_dims); }, std::invalid_argument);
}

/**
 * @test TENSOR.reshape_empty_dimensions_throws
 * @brief Reshaping with an empty dimensions vector
 * must throw std::invalid_argument.
 */
TEST(TENSOR, reshape_empty_dimensions_throws)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    std::vector<uint64_t> bad_dims = {};
    EXPECT_THROW({ A.reshape(bad_dims); }, std::invalid_argument);
}

/**
 * @test TENSOR.reshape_new_dimensions_with_zero_throws
 * @brief Reshaping with new_dimensions containing zero
 * must throw std::invalid_argument.
 */
TEST(TENSOR, reshape_new_dimensions_with_zero_throws)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    std::vector<uint64_t> bad_dims = {2, 0};
    EXPECT_THROW({ A.reshape(bad_dims); }, std::invalid_argument);
}

/**
 * @test TENSOR.reshape_dimension_product_overflow_throws
 * @brief Reshaping with excessively large dimensions
 * causing uint64_t overflow must throw std::overflow_error.
 */
TEST(TENSOR, reshape_dimension_product_overflow_throws)
{
    Tensor<float> A({2,2}, MemoryLocation::DEVICE);
    std::vector<uint64_t> bad_dims = {UINT64_MAX, 2};
    EXPECT_THROW({ A.reshape(bad_dims); }, std::overflow_error);
}

/**
 * @test TENSOR.reshape_multiple_roundtrip_preserves_data
 * @brief Perform multiple reshapes (2x3 -> 3x2 -> 1x6 -> 2x3) and verify the
 * linear buffer and final shape return to original values.
 */
TEST(TENSOR, reshape_multiple_roundtrip_preserves_data)
{
    Tensor<float> A({2,3}, MemoryLocation::DEVICE);
    std::vector<float> orig = {7.0f, 8.0f, 9.0f,
                               10.0f, 11.0f, 12.0f};
    A = orig;

    EXPECT_NO_THROW(A.reshape({3,2}));
    EXPECT_NO_THROW(A.reshape({6}));
    EXPECT_NO_THROW(A.reshape({2,3}));

    ASSERT_EQ(static_cast<uint64_t>(A.m_dimensions.size()), uint64_t{2});
    EXPECT_EQ(A.m_dimensions[0], uint64_t{2});
    EXPECT_EQ(A.m_dimensions[1], uint64_t{3});

    const uint64_t total = 6;
    std::vector<float> host_buf(total);
    g_sycl_queue.memcpy(host_buf.data(), A.m_p_data.get(),
                        sizeof(float) * total).wait();

    for (uint64_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(host_buf[i], orig[i]);
    }
}

/**
 * @test TENSOR.reshape_view_throws
 * @brief Attempting to reshape an alias/view must throw (non-owning).
 */
TEST(TENSOR, reshape_view_throws)
{
    Tensor<float> base({2,3}, MemoryLocation::HOST);
    base = std::vector<float>{0,1,2,3,4,5};

    Tensor<float> v(base, {0,0}, {2,3}, {3,1});
    EXPECT_FALSE(v.m_own_data);

    EXPECT_THROW(v.reshape(std::vector<uint64_t>{3,2}), std::invalid_argument);
}

/**
 * @test TENSOR.sort_empty
 * @brief Sorting an empty tensor should not throw.
 */
TEST(TENSOR, sort_empty)
{
    Tensor<float> t;
    EXPECT_NO_THROW(t.sort(0));
}

/**
 * @test TENSOR.sort_axis_out_of_bounds
 * @brief Sorting with invalid axis should throw.
 */
TEST(TENSOR, sort_axis_out_of_bounds)
{
    Tensor<float> t({3}, MemoryLocation::HOST);
    EXPECT_THROW(t.sort(1), std::invalid_argument);
    EXPECT_THROW(t.sort(-2), std::invalid_argument);
}

/**
 * @test TENSOR.sort_axis_size_one
 * @brief Sorting along axis with size <= 1 should not modify tensor.
 */
TEST(TENSOR, sort_axis_size_one)
{
    Tensor<float> t({1}, MemoryLocation::HOST);
    t = 123.0f;
    EXPECT_NO_THROW(t.sort(0));
    EXPECT_FLOAT_EQ(static_cast<float>(t), 123.0f);
}

/**
 * @test TENSOR.sort_1D_basic
 * @brief Sorting a 1D tensor with random values.
 */
TEST(TENSOR, sort_1D_basic)
{
    Tensor<float> t({5}, MemoryLocation::HOST);
    std::vector<float> vals =  {3.0f, -1.0f, 2.5f, 0.0f, 10.0f};
    t = vals;
    t.sort(0);
    std::vector<float> expected = {-1.0f, 0.0f, 2.5f, 3.0f, 10.0f};
    for (uint64_t i = 0; i < 5; i++)
    {
        EXPECT_FLOAT_EQ(t[i], expected[i]);
    }
}

/**
 * @test TENSOR.sort_2D_axis0
 * @brief Sorting a 2D tensor along axis 0 (rows).
 */
TEST(TENSOR, sort_2D_axis0)
{
    Tensor<float> t({3, 2}, MemoryLocation::HOST);
    // [[3, 2],
    //  [1, 5],
    //  [4, 0]]
    std::vector<float> vals = {3, 2, 1, 5, 4, 0};
    t = vals;
    t.sort(0);
    // Sorted along rows => [[1,0],[3,2],[4,5]]
    EXPECT_FLOAT_EQ(t[0][0], 1.0f);
    EXPECT_FLOAT_EQ(t[0][1], 0.0f);
    EXPECT_FLOAT_EQ(t[1][0], 3.0f);
    EXPECT_FLOAT_EQ(t[1][1], 2.0f);
    EXPECT_FLOAT_EQ(t[2][0], 4.0f);
    EXPECT_FLOAT_EQ(t[2][1], 5.0f);
}

/**
 * @test TENSOR.sort_2D_axis1
 * @brief Sorting a 2D tensor along axis 1 (columns).
 */
TEST(TENSOR, sort_2D_axis1)
{
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    // [[3, 1, 2],
    //  [0, -1, 5]]
    std::vector<float> vals = {3,1,2,0,-1,5};
    t = vals;
    t.sort(1);
    // [[1,2,3], [-1,0,5]]
    EXPECT_FLOAT_EQ(t[0][0], 1.0f);
    EXPECT_FLOAT_EQ(t[0][1], 2.0f);
    EXPECT_FLOAT_EQ(t[0][2], 3.0f);
    EXPECT_FLOAT_EQ(t[1][0], -1.0f);
    EXPECT_FLOAT_EQ(t[1][1], 0.0f);
    EXPECT_FLOAT_EQ(t[1][2], 5.0f);
}

/**
 * @test TENSOR.sort_3D_axis2
 * @brief Sort the last axis of a 3D tensor: shape {2,2,4}.
 */
TEST(TENSOR, sort_3D_axis2)
{
    Tensor<float> t({2,2,4}, MemoryLocation::HOST);
    // slices:
    // t[0][0] = [4,1,3,2]  -> [1,2,3,4]
    // t[0][1] = [0,-1,5,2] -> [-1,0,2,5]
    // t[1][0] = [9,7,8,6]  -> [6,7,8,9]
    // t[1][1] = [3,3,1,2]  -> [1,2,3,3]
    std::vector<float> vals = {
        4,1,3,2,
        0,-1,5,2,
        9,7,8,6,
        3,3,1,2
    };
    t = vals;
    t.sort(2);

    EXPECT_FLOAT_EQ(t[0][0][0], 1.0f);
    EXPECT_FLOAT_EQ(t[0][0][1], 2.0f);
    EXPECT_FLOAT_EQ(t[0][0][2], 3.0f);
    EXPECT_FLOAT_EQ(t[0][0][3], 4.0f);

    EXPECT_FLOAT_EQ(t[0][1][0], -1.0f);
    EXPECT_FLOAT_EQ(t[0][1][1], 0.0f);
    EXPECT_FLOAT_EQ(t[0][1][2], 2.0f);
    EXPECT_FLOAT_EQ(t[0][1][3], 5.0f);

    EXPECT_FLOAT_EQ(t[1][0][0], 6.0f);
    EXPECT_FLOAT_EQ(t[1][0][1], 7.0f);
    EXPECT_FLOAT_EQ(t[1][0][2], 8.0f);
    EXPECT_FLOAT_EQ(t[1][0][3], 9.0f);

    EXPECT_FLOAT_EQ(t[1][1][0], 1.0f);
    EXPECT_FLOAT_EQ(t[1][1][1], 2.0f);
    EXPECT_FLOAT_EQ(t[1][1][2], 3.0f);
    EXPECT_FLOAT_EQ(t[1][1][3], 3.0f);
}

/**
 * @test TENSOR.sort_3D_flatten
 * @brief Flatten-sort a 3D tensor (axis = -1) and confirm global ordering.
 */
TEST(TENSOR, sort_3D_flatten)
{
    Tensor<float> t({2,3,2}, MemoryLocation::HOST);
    std::vector<float> vals = {
        5, 1,
        3, 7,
        2, 9,

        0, 4,
        8, 6,
        -1, 10
    };
    t = vals;
    t.sort(-1);

    std::vector<float> got;
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
 * @test TENSOR.sort_with_nan
 * @brief NaNs should be placed last.
 */
TEST(TENSOR, sort_with_nan)
{
    Tensor<float> t({5}, MemoryLocation::HOST);
    std::vector<float> vals = {3.0f, NAN, 1.0f, -2.0f, NAN};
    t = vals;
    t.sort(0);
    // Expect [-2.0, 1.0, 3.0, nan, nan]
    EXPECT_FLOAT_EQ(t[0], -2.0f);
    EXPECT_FLOAT_EQ(t[1], 1.0f);
    EXPECT_FLOAT_EQ(t[2], 3.0f);
    EXPECT_TRUE(std::isnan(t[3]));
    EXPECT_TRUE(std::isnan(t[4]));
}

/**
 * @test TENSOR.sort_with_inf
 * @brief -Inf should come first, +Inf after finite numbers.
 */
TEST(TENSOR, sort_with_inf)
{
    Tensor<float> t({5}, MemoryLocation::HOST);
    std::vector<float> vals = {INFINITY, -1.0f, -INFINITY, 5.0f, 0.0f};
    t = vals;
    t.sort(0);
    std::vector<float> expected = {-INFINITY, -1.0f, 0.0f, 5.0f, INFINITY};
    for (uint64_t i = 0; i < 5; i++)
    {
        EXPECT_FLOAT_EQ(t[i], expected[i]);
    }
}

/**
 * @test TENSOR.sort_view_tensor
 * @brief Sorting a tensor view should only affect the view region.
 */
TEST(TENSOR, sort_view_tensor)
{
    Tensor<float> t({4}, MemoryLocation::HOST);
    std::vector<float> vals = {4.0f, 3.0f, 2.0f, 1.0f};
    t = vals;
    Tensor<float> v(t, {1}, {2});
    v.sort(0);

    EXPECT_FLOAT_EQ(t[0], 4.0f);
    EXPECT_FLOAT_EQ(t[1], 2.0f);
    EXPECT_FLOAT_EQ(t[2], 3.0f);
    EXPECT_FLOAT_EQ(t[3], 1.0f);
}

/**
 * @test TENSOR.sort_view_tensor_2D_row
 * @brief Sorting a view of a row should only affect that row segment.
 */
TEST(TENSOR, sort_view_tensor_2D_row)
{
    Tensor<float> t({2, 4}, MemoryLocation::HOST);
    // [[4,3,2,1],
    //  [8,7,6,5]]
    std::vector<float> vals = {4,3,2,1, 8,7,6,5};
    t = vals;

    Tensor<float> v(t, {0,1}, {1,2});
    v.sort(1);

    // Expect sorted only inside the view
    // [[4,2,3,1],
    //  [8,7,6,5]]
    EXPECT_FLOAT_EQ(t[0][0], 4.0f);
    EXPECT_FLOAT_EQ(t[0][1], 2.0f);
    EXPECT_FLOAT_EQ(t[0][2], 3.0f);
    EXPECT_FLOAT_EQ(t[0][3], 1.0f);
    EXPECT_FLOAT_EQ(t[1][0], 8.0f);
    EXPECT_FLOAT_EQ(t[1][1], 7.0f);
    EXPECT_FLOAT_EQ(t[1][2], 6.0f);
    EXPECT_FLOAT_EQ(t[1][3], 5.0f);
}

/**
 * @test TENSOR.sort_view_tensor_2D_col
 * @brief Sorting a view of a column should only affect that column segment.
 */
TEST(TENSOR, sort_view_tensor_2D_col)
{
    Tensor<float> t({3, 3}, MemoryLocation::HOST);
    // [[9,8,7],
    //  [6,5,4],
    //  [3,2,1]]
    std::vector<float> vals = {9,8,7, 6,5,4, 3,2,1};
    t = vals;

    Tensor<float> v(t, {0,1}, {3,1});
    v.sort(0);

    // [[9,2,7],
    //  [6,5,4],
    //  [3,8,1]]
    EXPECT_FLOAT_EQ(t[0][1], 2.0f);
    EXPECT_FLOAT_EQ(t[1][1], 5.0f);
    EXPECT_FLOAT_EQ(t[2][1], 8.0f);
}

/**
 * @test TENSOR.sort_view_tensor_3D_subcube
 * @brief Sorting a subcube view in 3D only changes inside that region.
 */
TEST(TENSOR, sort_view_tensor_3D_subcube)
{
    Tensor<float> t({2, 2, 4}, MemoryLocation::HOST);
    // [[[9,7,8,6],
    //   [5,3,4,2]],
    //  [[1,-1,0,-2],
    //   [10,12,11,13]]]
    std::vector<float> vals = {
        9,7,8,6,  5,3,4,2,
        1,-1,0,-2, 10,12,11,13
    };
    t = vals;

    Tensor<float> v(t, {0,0,0}, {1,2,4});
    v.sort(2);

    // [[[6,7,8,9],
    //   [2,3,4,5]],
    //  [[1,-1,0,-2],
    //   [10,12,11,13]]]
    EXPECT_FLOAT_EQ(t[0][0][0], 6.0f);
    EXPECT_FLOAT_EQ(t[0][0][3], 9.0f);
    EXPECT_FLOAT_EQ(t[0][1][0], 2.0f);
    EXPECT_FLOAT_EQ(t[0][1][3], 5.0f);

    EXPECT_FLOAT_EQ(t[1][0][0], 1.0f);
    EXPECT_FLOAT_EQ(t[1][1][0], 10.0f);
}

/**
 * @test TENSOR.sort_view_non1D_flatten
 * @brief Sorting a non-1D view with axis = -1
 * should only flatten-sort the view region.
 */
TEST(TENSOR, sort_view_non1D_flatten)
{
    Tensor<float> t({2,3,3}, MemoryLocation::HOST);

    std::vector<float> vals = {
        1,2,3,   13,14,15,   7,8,9,
        10,11,12, 16,17,18,  4,5,6
    };
    t = vals;

    Tensor<float> v(t, {0,1,0}, {2,2,3});

    v.sort(-1);

    std::vector<float> got;
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

    // Ensure elements outside the view (here: j == 0 rows) are unchanged.
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 1; ++j)
        {
            for (uint64_t k = 0; k < 3; ++k)
            {
                uint64_t flat = i * (3 * 3) + j * 3 + k;
                EXPECT_FLOAT_EQ(t[i][j][k], vals[flat]);
            }
        }
    }
}

/**
 * @test TENSOR.sort_alias_view_basic_1D
 * @brief Sorting a 1D alias view (contiguous) should only reorder the view region in the owner.
 */
TEST(TENSOR, sort_alias_view_basic_1D)
{
    Tensor<float> t({6}, MemoryLocation::HOST);
    std::vector<float> vals = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    t = vals;

    Tensor<float> v(t, {1}, {3}, {1});

    v.sort(0);

    EXPECT_FLOAT_EQ(t[0], 6.0f);
    EXPECT_FLOAT_EQ(t[1], 3.0f);
    EXPECT_FLOAT_EQ(t[2], 4.0f);
    EXPECT_FLOAT_EQ(t[3], 5.0f);
    EXPECT_FLOAT_EQ(t[4], 2.0f);
    EXPECT_FLOAT_EQ(t[5], 1.0f);
}

/**
 * @test TENSOR.sort_alias_view_noncontiguous_stride
 * @brief Sorting a non-contiguous 1D alias (stride > 1)
 * updates only the sampled elements.
 */
TEST(TENSOR, sort_alias_view_noncontiguous_stride)
{
    Tensor<float> t({8}, MemoryLocation::HOST);
    std::vector<float> vals = {10.f, 0.f, 9.f, 0.f, 8.f, 0.f, 7.f, 0.f};
    t = vals;

    Tensor<float> v(t, {0}, {4}, {2});

    v.sort(0);

    EXPECT_FLOAT_EQ(t[0], 7.0f);
    EXPECT_FLOAT_EQ(t[1], 0.0f);
    EXPECT_FLOAT_EQ(t[2], 8.0f);
    EXPECT_FLOAT_EQ(t[3], 0.0f);
    EXPECT_FLOAT_EQ(t[4], 9.0f);
    EXPECT_FLOAT_EQ(t[5], 0.0f);
    EXPECT_FLOAT_EQ(t[6], 10.0f);
    EXPECT_FLOAT_EQ(t[7], 0.0f);
}

/**
 * @test TENSOR.sort_alias_view_2D_submatrix_axis1
 * @brief Sorting a 2x3 submatrix view along its
 * last axis only sorts that submatrix.
 */
TEST(TENSOR, sort_alias_view_2D_submatrix_axis1)
{
    Tensor<float> t({3, 4}, MemoryLocation::HOST);
    // matrix (row-major):
    // [ 9, 8, 7, 6 ]
    // [ 5, 4, 3, 2 ]
    // [ 1, 0,-1,-2 ]
    std::vector<float> vals = {9,8,7,6, 5,4,3,2, 1,0,-1,-2};
    t = vals;

    Tensor<float> sub(t, {0,1}, {2,3}, {4,1});

    // sort each row inside the submatrix (axis 1)
    sub.sort(1);

    // Expected owner after operation:
    // row0 original: [9, 8, 7, 6] -> sub part [8,7,6] sorted -> [6,7,8]
    // row1 original: [5, 4, 3, 2] -> sub part [4,3,2] sorted -> [2,3,4]
    EXPECT_FLOAT_EQ(t[0][0], 9.0f);
    EXPECT_FLOAT_EQ(t[0][1], 6.0f);
    EXPECT_FLOAT_EQ(t[0][2], 7.0f);
    EXPECT_FLOAT_EQ(t[0][3], 8.0f);

    EXPECT_FLOAT_EQ(t[1][0], 5.0f);
    EXPECT_FLOAT_EQ(t[1][1], 2.0f);
    EXPECT_FLOAT_EQ(t[1][2], 3.0f);
    EXPECT_FLOAT_EQ(t[1][3], 4.0f);

    // row2 unchanged
    EXPECT_FLOAT_EQ(t[2][0], 1.0f);
    EXPECT_FLOAT_EQ(t[2][1], 0.0f);
    EXPECT_FLOAT_EQ(t[2][2], -1.0f);
    EXPECT_FLOAT_EQ(t[2][3], -2.0f);
}

/**
 * @test TENSOR.sort_alias_view_flatten_subregion
 * @brief Flatten-sorting (axis = -1) a submatrix view
 * only orders elements within the view.
 */
TEST(TENSOR, sort_alias_view_flatten_subregion)
{
    Tensor<float> t({2,3}, MemoryLocation::HOST);
    // [ 5, 1, 3 ]
    // [ 4, 2, 0 ]
    std::vector<float> vals = {5,1,3, 4,2,0};
    t = vals;

    Tensor<float> sub(t, {0,0}, {2,2}, {3,1});

    sub.sort(-1);

    // After operation owner should be:
    // row0: [1,2,3]
    // row1: [4,5,0]  (note last element outside view unchanged).
    EXPECT_FLOAT_EQ(t[0][0], 1.0f);
    EXPECT_FLOAT_EQ(t[0][1], 2.0f);
    EXPECT_FLOAT_EQ(t[0][2], 3.0f);

    EXPECT_FLOAT_EQ(t[1][0], 4.0f);
    EXPECT_FLOAT_EQ(t[1][1], 5.0f);
    EXPECT_FLOAT_EQ(t[1][2], 0.0f);
}

/**
 * @test TENSOR.sort_alias_view_weird_strides
 * @brief Sorting an alias view with non-trivial strides (e.g. 13,4).
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Sort along axis 1.
 */
TEST(TENSOR, sort_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::HOST);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[99 - i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});
    view.sort(1);

    EXPECT_EQ(view.m_dimensions, (std::vector<uint64_t>{3,4}));
    EXPECT_EQ(view.m_strides, (std::vector<uint64_t>{13,4}));

    Tensor<float> host({3,4}, MemoryLocation::HOST);
    copy_tensor_data(host, view);

    std::vector<float> out(12);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float)*12).wait();

    std::vector<float> expected =
    {
        87.f,  91.f,  95.f,  99.f,
        74.f, 78.f, 82.f, 86.f,
        61.f, 65.f, 69.f, 73.f
    };

    for (uint64_t k = 0; k < out.size(); ++k)
    {
        EXPECT_FLOAT_EQ(out[k], expected[k]);
    }
}

/**
 * @test TENSOR.sort_alias_view_broadcast_noop
 * @brief Sorting a broadcasted view (stride 0) is a no-op
 * in terms of data reordering, but must not crash; owner must remain consistent.
 */
TEST(TENSOR, sort_alias_view_broadcast_noop)
{
    Tensor<float> t({2}, MemoryLocation::HOST);
    t = std::vector<float>{42.0f, 99.0f};

    Tensor<float> b(t, {1}, {4}, {0});

    EXPECT_NO_THROW(b.sort(0));
    EXPECT_FLOAT_EQ(t[0], 42.0f);
    EXPECT_FLOAT_EQ(t[1], 99.0f);
}

/**
 * @test TENSOR.sort_idempotence
 * @brief Sorting twice should give same result.
 */
TEST(TENSOR, sort_idempotence)
{
    Tensor<float> t({5}, MemoryLocation::HOST);
    std::vector<float> vals = {4,1,3,2,0};
    t = vals;
    t.sort(0);
    std::vector<float> once(5);
    for (uint64_t i=0;i<5;i++)
    {
        once[i] = t[i];
    }
    t.sort(0);
    for (uint64_t i=0;i<5;i++)
    {
        EXPECT_FLOAT_EQ(t[i], once[i]);
    }
}

/**
 * @test TENSOR.sum_all_elements
 * @brief Sum all elements (axis = -1) on a device tensor and return
 * a scalar with the correct total value.
 */
TEST(TENSOR, sum_all_elements)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    t = vals;

    Tensor<float> res = t.sum(-1);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();
    EXPECT_FLOAT_EQ(host[0], 6.0f);
}

/**
 * @test TENSOR.sum_axis0
 * @brief Sum along axis 0 for a 2x3 tensor stored on device
 * nd verify per-column sums.
 */
TEST(TENSOR, sum_axis0)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = t.sum(0);

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f + 5.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f + 6.0f);
}

/**
 * @test TENSOR.sum_axis1
 * @brief Sum along axis 1 for a 2x3 device tensor and verify per-row sums.
 */
TEST(TENSOR, sum_axis1)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);

    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = t.sum(1);

    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 2 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 2.0f + 3.0f);
    EXPECT_FLOAT_EQ(host[1], 4.0f + 5.0f + 6.0f);
}

/**
 * @test TENSOR.sum_axis0_3D
 * @brief Sum along axis 0 for a 2x2x2 device tensor and verify resulting values.
 */
TEST(TENSOR, sum_axis0_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = t.sum(0);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 5.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f + 6.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f + 7.0f);
    EXPECT_FLOAT_EQ(host[3], 4.0f + 8.0f);
}

/**
 * @test TENSOR.sum_axis1_3D
 * @brief Sum along axis 1 for a 2x2x2 device tensor and verify resulting values.
 */
TEST(TENSOR, sum_axis1_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = t.sum(1);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 3.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[2], 5.0f + 7.0f);
    EXPECT_FLOAT_EQ(host[3], 6.0f + 8.0f);
}

/**
 * @test TENSOR.sum_axis2_3D
 * @brief Sum along axis 2 for a 2x2x2 device tensor and verify resulting values.
 */
TEST(TENSOR, sum_axis2_3D)
{
    Tensor<float> t({2, 2, 2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    t = vals;

    Tensor<float> res = t.sum(2);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 2.0f);
    EXPECT_FLOAT_EQ(host[1], 3.0f + 4.0f);
    EXPECT_FLOAT_EQ(host[2], 5.0f + 6.0f);
    EXPECT_FLOAT_EQ(host[3], 7.0f + 8.0f);
}

/**
 * @test TENSOR.sum_view_tensor
 * @brief Sum all elements (axis = -1) of a view into a device tensor and
 * verify the scalar result.
 */
TEST(TENSOR, sum_view_tensor)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {0ull, 0ull};
    std::vector<uint64_t> view_shape = {3ull};

    Tensor<float> view(t, start_indices, view_shape);

    Tensor<float> res = view.sum(-1);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 2.0f + 3.0f);
}

/**
 * @test TENSOR.sum_alias_view_tensor
 * @brief Sum all elements (axis = -1) of an alias view
 * with non-unit stride and verify result.
 */
TEST(TENSOR, sum_alias_view_tensor)
{
    Tensor<float> t({6}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {0ull};
    std::vector<uint64_t> dims = {3ull};
    std::vector<uint64_t> strides = {2ull};

    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = alias_view.sum(-1);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f + 3.0f + 5.0f);
}

/**
 * @test TENSOR.sum_view_tensor_3d_axis1
 * @brief Sum along axis 1 on a 3D view and verify the produced values.
 */
TEST(TENSOR, sum_view_tensor_3d_axis1)
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

    Tensor<float> res = view.sum(1);

    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0], 23.0f);
    EXPECT_FLOAT_EQ(host[1], 27.0f);
}

/**
 * @test TENSOR.sum_alias_view_tensor_2d_strided
 * @brief Sum along axis 0 on a 2D alias view with custom strides and verify
 * each output element.
 */
TEST(TENSOR, sum_alias_view_tensor_2d_strided)
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

    Tensor<float> res = alias_view.sum(0);

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0], 9.0f);
    EXPECT_FLOAT_EQ(host[1], 13.0f);
    EXPECT_FLOAT_EQ(host[2], 17.0f);
}

/**
 * @test TENSOR.sum_alias_view_tensor_overlapping_stride_zero
 * @brief Sum along axis 0 on an alias view that contains overlapping elements
 * via a zero stride and verify the sums account for repeated elements.
 */
TEST(TENSOR, sum_alias_view_tensor_overlapping_stride_zero)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start_indices = {1ull, 0ull};
    std::vector<uint64_t> dims          = {2ull, 2ull};
    std::vector<uint64_t> strides       = {0ull, 1ull};
    Tensor<float> alias_view(t, start_indices, dims, strides);

    Tensor<float> res = alias_view.sum(0);

    std::vector<float> host(2);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), sizeof(float) * host.size()).wait();

    EXPECT_FLOAT_EQ(host[0], 8.0f);
    EXPECT_FLOAT_EQ(host[1], 10.0f);
}

/**
 * @test TENSOR.sum_nan_throws
 * @brief Tests that sum throws std::runtime_error
 * when the tensor contains NaN values.
 */
TEST(TENSOR, sum_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals =
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    t = vals;

    EXPECT_THROW(t.sum(-1), std::runtime_error);
}

/**
 * @test TENSOR.sum_non_finite_throws
 * @brief Tests that sum throws std::runtime_error when
 * the tensor contains non-finite values (infinity).
 */
TEST(TENSOR, sum_non_finite_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {std::numeric_limits<float>::infinity(), 1.0f};
    t = vals;

    EXPECT_THROW(t.sum(-1), std::runtime_error);
}

/**
 * @test TENSOR.sum_empty
 * @brief Summing an empty tensor returns a scalar tensor containing 0.0.
 */
TEST(TENSOR, sum_empty)
{
    Tensor<float> t;

    Tensor<float> res({1}, MemoryLocation::DEVICE);
    res = t.sum(-1);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 0.0f);
}

/**
 * @test TENSOR.cumsum_all_elements_flatten
 * @brief Tests cumsum on a 1D tensor, flattening all elements.
 */
TEST(TENSOR, cumsum_all_elements_flatten)
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
 * @test TENSOR.cumsum_axis0_2D
 * @brief Tests cumsum along axis 0 of a 2D tensor.
 */
TEST(TENSOR, cumsum_axis0_2D)
{
    Tensor<float> t({2, 3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = t.cumsum(0);

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
 * @test TENSOR.cumsum_axis1_2D
 * @brief Tests cumsum along axis 1 of a 2D tensor.
 */
TEST(TENSOR, cumsum_axis1_2D)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    Tensor<float> res = t.cumsum(1);

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
 * @test TENSOR.cumsum_flatten_3D
 * @brief Tests cumsum on a 3D tensor flattened along the last axis.
 */
TEST(TENSOR, cumsum_flatten_3D)
{
    Tensor<float> t({2,2,2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f};
    t = vals;
    Tensor<float> res = t.cumsum(-1);

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
 * @test TENSOR.cumsum_view_flatten
 * @brief Tests cumsum on a view of a 3D tensor flattened along the last axis.
 */
TEST(TENSOR, cumsum_view_flatten)
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

    Tensor<float> res = view.cumsum(-1);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 11.0f);
    EXPECT_FLOAT_EQ(host[1], 23.0f);
    EXPECT_FLOAT_EQ(host[2], 36.0f);
    EXPECT_FLOAT_EQ(host[3], 50.0f);
}

/**
 * @test TENSOR.cumsum_alias_view_strided
 * @brief Tests cumsum on an alias view with a stride on a 1D tensor.
 */
TEST(TENSOR, cumsum_alias_view_strided)
{
    Tensor<float> t({6}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t = vals;
    std::vector<uint64_t> start = {0ull};
    std::vector<uint64_t> dims  = {3ull};
    std::vector<uint64_t> strides = {2ull};
    Tensor<float> alias_view(t, start, dims, strides);

    Tensor<float> res = alias_view.cumsum(-1);

    std::vector<float> host(3);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 3 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 1.0f + 3.0f);
    EXPECT_FLOAT_EQ(host[2], 1.0f + 3.0f + 5.0f);
}

/**
 * @test TENSOR.cumsum_alias_view_overlapping_stride_zero
 * @brief Tests cumsum on an alias view with
 * overlapping stride of zero on a 2D tensor.
 */
TEST(TENSOR, cumsum_alias_view_overlapping_stride_zero)
{
    Tensor<float> t({2,3}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f};
    t = vals;

    std::vector<uint64_t> start   = {1ull, 0ull};
    std::vector<uint64_t> dims    = {2ull, 2ull};
    std::vector<uint64_t> strides = {0ull, 1ull};
    Tensor<float> alias_view(t, start, dims, strides);

    Tensor<float> res = alias_view.cumsum(0);

    std::vector<float> host(4);
    g_sycl_queue.memcpy
        (host.data(), res.m_p_data.get(), 4 * sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 4.0f);
    EXPECT_FLOAT_EQ(host[1], 5.0f);
    EXPECT_FLOAT_EQ(host[2], 8.0f);
    EXPECT_FLOAT_EQ(host[3], 10.0f);
}

/**
 * @test TENSOR.cumsum_alias_view_weird_strides
 * @brief Sorting an alias view with non-trivial strides (e.g. 13,4).
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Cumsum along axis 1.
 */
TEST(TENSOR, cumsum_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::HOST);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});

    Tensor<float> view2 = view.cumsum(1);
    EXPECT_EQ(view2.m_dimensions, (std::vector<uint64_t>{3,4}));
    EXPECT_EQ(view2.m_strides, (std::vector<uint64_t>{4,1}));

    Tensor<float> host({3,4}, MemoryLocation::HOST);
    copy_tensor_data(host, view2);

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
 * @test TENSOR.cumsum_axis_out_of_bounds
 * @brief Tests that cumsum throws std::invalid_argument
 * when the axis is out of bounds.
 */
TEST(TENSOR, cumsum_axis_out_of_bounds)
{
    Tensor<float> t({2,2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
    t = vals;

    EXPECT_THROW(t.cumsum(2), std::invalid_argument);
    EXPECT_THROW(t.cumsum(-2), std::invalid_argument);
}

/**
 * @test TENSOR.cumsum_nan_throws
 * @brief Tests that cumsum throws std::runtime_error
 * when the tensor contains NaN values.
 */
TEST(TENSOR, cumsum_nan_throws)
{
    Tensor<float> t({3}, MemoryLocation::DEVICE);
    std::vector<float> vals =
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    t = vals;

    EXPECT_THROW(t.cumsum(-1), std::runtime_error);
}

/**
 * @test TENSOR.cumsum_non_finite_throws
 * @brief Tests that cumsum throws std::runtime_error when
 * the tensor contains non-finite values (infinity).
 */
TEST(TENSOR, cumsum_non_finite_throws)
{
    Tensor<float> t({2}, MemoryLocation::DEVICE);
    std::vector<float> vals = {std::numeric_limits<float>::infinity(), 1.0f};
    t = vals;

    EXPECT_THROW(t.cumsum(-1), std::runtime_error);
}

/**
 * @test TENSOR.cumsum_empty
 * @brief Tests cumsum on an empty tensor returns a tensor
 * with a single zero element.
 */
TEST(TENSOR, cumsum_empty)
{
    Tensor<float> t;

    Tensor<float> res({1}, MemoryLocation::DEVICE);
    res = t.cumsum(-1);

    std::vector<float> host(1);
    g_sycl_queue.memcpy(host.data(), res.m_p_data.get(), sizeof(float)).wait();

    EXPECT_FLOAT_EQ(host[0], 0.0f);
}

/**
 * @test TENSOR.transpose_noargs_reverse_axes
 * @brief Tests that transpose() with no arguments reverses all axes.
 */
TEST(TENSOR, transpose_noargs_reverse_axes)
{
    Tensor<float> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    t = vals;

    Tensor<float> t_rev = t.transpose();

    EXPECT_EQ(t_rev.m_dimensions, (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(t_rev.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1], t.m_strides[0]}));

    Tensor<float> host({4, 3, 2}, MemoryLocation::HOST);
    copy_tensor_data(host, t_rev);

    std::vector<float> out(24);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 24).wait();

    EXPECT_FLOAT_EQ(out[0], vals[0]);
    EXPECT_FLOAT_EQ(out[23], vals[23]);
}

/**
 * @test TENSOR.transpose_explicit_axes
 * @brief Tests transpose with explicit axis permutation.
 */
TEST(TENSOR, transpose_explicit_axes)
{
    Tensor<float> t({2, 3, 4}, MemoryLocation::HOST);
    std::vector<float> vals(24);
    for (uint64_t i = 0; i < 24; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    t = vals;

    Tensor<float> perm = t.transpose({2, 1, 0});

    EXPECT_EQ(perm.m_dimensions, (std::vector<uint64_t>{4, 3, 2}));
    EXPECT_EQ(perm.m_strides,
        (std::vector<uint64_t>{t.m_strides[2], t.m_strides[1], t.m_strides[0]}));

    Tensor<float> host({4, 3, 2}, MemoryLocation::HOST);
    copy_tensor_data(host, perm);
    std::vector<float> out(24);
    g_sycl_queue.memcpy
        (out.data(), host.m_p_data.get(), sizeof(float) * 24).wait();

    EXPECT_FLOAT_EQ(out[0], vals[0]);
    EXPECT_FLOAT_EQ(out[23], vals[23]);
}

/**
 * @test TENSOR.transpose_2d
 * @brief Tests transpose on a 2D tensor (matrix).
 */
TEST(TENSOR, transpose_2d)
{
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    t = {1,2,3,4,5,6};

    Tensor<float> t_T = t.transpose();

    EXPECT_EQ(t_T.m_dimensions, (std::vector<uint64_t>{3,2}));
    EXPECT_EQ(t_T.m_strides,
        (std::vector<uint64_t>{t.m_strides[1], t.m_strides[0]}));

    Tensor<float> host({3,2}, MemoryLocation::HOST);
    copy_tensor_data(host, t_T);
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
 * @test TENSOR.transpose_mutation_reflects
 * @brief Ensure that modifying the transposed alias updates the original tensor.
 */
TEST(TENSOR, transpose_mutation_reflects)
{
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    t = {0,1,2,3,4,5};

    Tensor<float> t_T = t.transpose();
    t_T[0][0] = 100.f;
    t_T[2][1] = 200.f;

    Tensor<float> host({2,3}, MemoryLocation::HOST);
    copy_tensor_data(host, t);
    std::vector<float> out(6);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*6).wait();

    EXPECT_FLOAT_EQ(out[0], 100.f);
    EXPECT_FLOAT_EQ(out[5], 200.f);
}

/**
 * @test TENSOR.transpose_invalid_axes
 * @brief Transpose throws when axes permutation is invalid.
 */
TEST(TENSOR, transpose_invalid_axes)
{
    Tensor<float> t({2,3,4}, MemoryLocation::HOST);
    t = std::vector<float>(24, 1.f);

    EXPECT_THROW(t.transpose({0,1}), std::invalid_argument);

    EXPECT_THROW(t.transpose({0,1,1}), std::invalid_argument);

    EXPECT_THROW(t.transpose({0,1,3}), std::invalid_argument);
}

/**
 * @test TENSOR.transpose_1d
 * @brief Transpose a 1D tensor should return a 1D alias (no change).
 */
TEST(TENSOR, transpose_1d)
{
    Tensor<float> t({5}, MemoryLocation::HOST);
    t = {0,1,2,3,4};

    Tensor<float> t_tr = t.transpose();
    EXPECT_EQ(t_tr.m_dimensions, t.m_dimensions);
    EXPECT_EQ(t_tr.m_strides, t.m_strides);

    Tensor<float> host({5}, MemoryLocation::HOST);
    copy_tensor_data(host, t_tr);
    std::vector<float> out(5);
    g_sycl_queue.memcpy(out.data(), host.m_p_data.get(), sizeof(float)*5).wait();

    for (uint64_t i = 0; i < 5; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i));
    }
}

/**
 * @test TENSOR.transpose_empty
 * @brief Transpose of an empty tensor throws.
 */
TEST(TENSOR, transpose_empty)
{
    Tensor<float> t;
    EXPECT_THROW(t.transpose(), std::runtime_error);
}

/**
 * @test TENSOR.print_tensor
 * @brief Checks that print correctly outputs a 2x2 tensor with assigned values.
 *
 * Creates a 2x2 tensor, assigns values {1,2,3,4}, and verifies
 * that print outputs the expected nested format.
 */
TEST(TENSOR, print_tensor)
{
    temper::Tensor<float> t({2, 2}, temper::MemoryLocation::HOST);
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
    t = vals;

    std::stringstream ss;
    t.print(ss);

    std::string expected = "[[1, 2],\n [3, 4]]\n";

    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TENSOR.print_view_tensor
 * @brief Checks that print correctly outputs a 2x2 view of a 3x4 owner tensor.
 *
 * Owner shape: {3,4} values 1..12
 * View: start {1,1}, dims {2,2} -> should print [[6, 7], [10, 11]]
 */
TEST(TENSOR, print_view_tensor)
{
    Tensor<float> owner({3,4}, MemoryLocation::HOST);
    std::vector<float> vals(12);
    for (uint64_t i = 0; i < vals.size(); ++i)
    {
        vals[i] = static_cast<float>(i + 1);
    }
    owner = vals;

    std::vector<uint64_t> start = {1ull, 1ull};
    std::vector<uint64_t> view_shape = {2ull, 2ull};
    Tensor<float> view(owner, start, view_shape);

    std::stringstream ss;
    view.print(ss);

    std::string expected = "[[6, 7],\n [10, 11]]\n";
    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TENSOR.print_alias_view_weird_strides
 * @brief Checks print on an alias view with non-trivial strides (e.g. 13,4).
 *
 * Owner shape: {5,20} -> 100 elements [0..99]
 * View: start {0,0}, dims {3,4}, strides {13,4}
 * Printed matrix should reflect the stride mapping.
 */
TEST(TENSOR, print_alias_view_weird_strides)
{
    Tensor<float> owner({5,20}, MemoryLocation::HOST);
    std::vector<float> vals(100);
    for (uint64_t i = 0; i < 100; ++i)
    {
        vals[i] = static_cast<float>(i);
    }
    owner = vals;

    Tensor<float> view(owner, {0,0}, {3,4}, {13,4});

    std::stringstream ss;
    view.print(ss);

    std::string expected =
        "[[0, 4, 8, 12],\n"
        " [13, 17, 21, 25],\n"
        " [26, 30, 34, 38]]\n";

    EXPECT_EQ(ss.str(), expected);
}

/**
 * @test TENSOR.print_empty_tensor
 * @brief Checks that print correctly outputs an empty tensor.
 *
 * Creates a tensor with no dimensions and verifies that print
 * outputs the string "[]\n".
 */
TEST(TENSOR, print_empty_tensor)
{
    temper::Tensor<float> t;
    std::stringstream ss;
    t.print(ss);
    EXPECT_EQ(ss.str(), "[]\n");
}

/**
 * @test TENSOR.get_data
 * @brief Verifies that get_data() provides correct raw pointer access.
 *
 * Initializes a tensor with values, checks that get_data() returns
 * the expected sequence of floats, and ensures const access works.
 * Also validates that get_data() returns const float* on const tensors.
 */
TEST(TENSOR, get_data)
{
    Tensor<float> t({2, 3}, MemoryLocation::HOST);
    std::vector<float> vals = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    t = vals;

    for (uint64_t i = 0; i < t.get_num_elements(); ++i)
    {
        EXPECT_FLOAT_EQ(t.get_data()[i], static_cast<float>(i));
    }

    const Tensor<float> ct = t;
    EXPECT_EQ(ct.get_dimensions(), t.get_dimensions());
    EXPECT_EQ(ct.get_strides(), t.get_strides());
    EXPECT_EQ(ct.get_num_elements(), t.get_num_elements());

    for (uint64_t i = 0; i < ct.get_num_elements(); ++i)
    {
        EXPECT_FLOAT_EQ(ct.get_data()[i], t.get_data()[i]);
    }

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_data()),
        const float* >,
        "get_data() should return const float* called on const Tensor<float>&");
}

/**
 * @test TENSOR.get_dimensions
 * @brief Validates that get_dimensions() returns the correct shape.
 *
 * Checks consistency between get_dimensions() and internal storage,
 * verifies aliasing with get_shape(), and enforces const correctness.
 */
TEST(TENSOR, get_dimensions)
{
    Tensor<float> t({4, 5, 6}, MemoryLocation::HOST);

    EXPECT_EQ(t.get_dimensions(), t.m_dimensions);

    EXPECT_EQ(&t.get_shape(), &t.get_dimensions());

    const Tensor<float> ct = t;
    EXPECT_EQ(ct.get_dimensions(), t.get_dimensions());
    EXPECT_EQ(&ct.get_shape(), &ct.get_dimensions());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_dimensions()),
        const std::vector<uint64_t>&
    >, "get_dimensions() must return const std::vector<uint64_t>&");

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_shape()),
        const std::vector<uint64_t>&
    >, "get_shape() must return const std::vector<uint64_t>&");
}

/**
 * @test TENSOR.get_strides
 * @brief Confirms that get_strides() returns the correct stride vector.
 *
 * Ensures correct access for both mutable and const tensors, and checks
 * compile-time type correctness.
 */
TEST(TENSOR, get_strides)
{
    Tensor<float> t({3, 2}, MemoryLocation::HOST);

    EXPECT_EQ(t.get_strides(), t.m_strides);

    const Tensor<float> ct = t;
    EXPECT_EQ(ct.get_strides(), t.get_strides());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_strides()),
        const std::vector<uint64_t>&
    >, "get_strides() must return const std::vector<uint64_t>&");
}

/**
 * @test TENSOR.get_rank
 * @brief Tests that get_rank() returns the correct number of dimensions.
 *
 * Verifies rank for a tensor with shape {7, 8, 9}, const-correctness,
 * and enforces return type at compile time.
 */
TEST(TENSOR, get_rank)
{
    Tensor<float> t({7, 8, 9}, MemoryLocation::HOST);
    EXPECT_EQ(t.get_rank(), static_cast<uint64_t>(3));

    const Tensor<float> ct = t;
    EXPECT_EQ(ct.get_rank(), t.get_rank());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_rank()),
        uint64_t
    >, "get_rank() must return uint64_t");
}

/**
 * @test TENSOR.get_num_elements
 * @brief Ensures get_num_elements() returns the correct element count.
 *
 * Checks correctness for non-empty tensors and empty tensors, validates
 * const-correctness, and enforces return type.
 */
TEST(TENSOR, get_num_elements)
{
    Tensor<float> t({2, 4, 3}, MemoryLocation::HOST);
    EXPECT_EQ(t.get_num_elements(), static_cast<uint64_t>(2 * 4 * 3));

    Tensor<float> empty;
    EXPECT_EQ(empty.get_num_elements(), static_cast<uint64_t>(0));

    const Tensor<float> ct = t;
    EXPECT_EQ(ct.get_num_elements(), t.get_num_elements());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_num_elements()),
        uint64_t
    >, "get_num_elements() must return uint64_t");
}

/**
 * @test TENSOR.get_memory_location
 * @brief Validates that get_memory_location() reports correct allocation target.
 *
 * Ensures host and device tensors report expected MemoryLocation values
 * and verifies const-correctness and return type.
 */
TEST(TENSOR, get_memory_location)
{
    Tensor<float> host_t({2, 2}, MemoryLocation::HOST);
    Tensor<float> device_t({2, 2}, MemoryLocation::DEVICE);

    EXPECT_EQ(host_t.get_memory_location(), MemoryLocation::HOST);
    EXPECT_EQ(device_t.get_memory_location(), MemoryLocation::DEVICE);

    const Tensor<float> cht = host_t;
    EXPECT_EQ(cht.get_memory_location(), MemoryLocation::HOST);

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_memory_location()),
        MemoryLocation
    >, "get_memory_location() must return MemoryLocation");
}

/**
 * @test TENSOR.get_owns_data
 * @brief Tests that get_owns_data() distinguishes owning vs. non-owning tensors.
 *
 * Verifies that owning tensors report true, view tensors report false,
 * and checks const-correctness and return type.
 */
TEST(TENSOR, get_owns_data)
{
    Tensor<float> owner({3, 3}, MemoryLocation::HOST);
    EXPECT_TRUE(owner.get_owns_data());

    Tensor<float> base({4, 4}, MemoryLocation::HOST);
    std::vector<uint64_t> start = {1, 1};
    std::vector<uint64_t> shape = {2, 2};
    Tensor<float> view(base, start, shape);
    EXPECT_FALSE(view.get_owns_data());

    const Tensor<float> const_owner = owner;
    EXPECT_TRUE(const_owner.get_owns_data());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_owns_data()),
        bool
    >, "get_owns_data() must return bool");
}

/**
 * @test TENSOR.is_view
 * @brief Verifies that is_view() correctly identifies tensor views.
 *
 * Checks that owning tensors return false, sub-tensors return true,
 * validates const correctness, and enforces return type.
 */
TEST(TENSOR, is_view)
{
    Tensor<float> owner({2, 2}, MemoryLocation::HOST);
    Tensor<float> base({4, 4}, MemoryLocation::HOST);
    Tensor<float> view(base, {1,1}, {2,2});

    EXPECT_FALSE(owner.is_view());
    EXPECT_TRUE(view.is_view());

    const Tensor<float> const_view = view;
    EXPECT_TRUE(const_view.is_view());

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().is_view()),
        bool
    >, "is_view() must return bool");
}

/**
 * @test TENSOR.get_element_size_bytes
 * @brief Tests that get_element_size_bytes() returns sizeof(element type).
 *
 * Ensures correct behavior for float tensors, const-correctness, and
 * compile-time type verification.
 */
TEST(TENSOR, get_element_size_bytes)
{
    Tensor<float> t({1}, MemoryLocation::HOST);
    EXPECT_EQ(t.get_element_size_bytes(), static_cast<uint64_t>(sizeof(float)));

    const Tensor<float> ct = t;
    EXPECT_EQ(ct.get_element_size_bytes(), static_cast<uint64_t>(sizeof(float)));

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_element_size_bytes()),
        uint64_t
    >, "get_element_size_bytes() must return uint64_t");
}

/**
 * @test TENSOR.get_total_bytes
 * @brief Ensures get_total_bytes() returns correct total size in bytes.
 *
 * Validates correct computation for non-empty and empty tensors, enforces
 * const correctness, and checks return type at compile time.
 */
TEST(TENSOR, get_total_bytes)
{
    Tensor<float> t({5, 6}, MemoryLocation::HOST);
    uint64_t expected_elems = 5 * 6;
    uint64_t expected_total =
        expected_elems * static_cast<uint64_t>(sizeof(float));
    EXPECT_EQ(t.get_total_bytes(), expected_total);

    Tensor<float> empty;
    EXPECT_EQ(empty.get_total_bytes(), static_cast<uint64_t>(0));

    const Tensor<float> ct = t;
    EXPECT_EQ(ct.get_total_bytes(), expected_total);

    static_assert(std::is_same_v<
        decltype(std::declval<const Tensor<float>&>().get_total_bytes()),
        uint64_t
    >, "get_total_bytes() must return uint64_t");
}

} // namespace Test