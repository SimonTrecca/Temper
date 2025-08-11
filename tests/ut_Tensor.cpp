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
 * @test TENSOR.constructor_sets_dimensions_and_strides
 * @brief Tests that the Tensor constructor
 * correctly sets dimensions and computes strides.
 */
TEST(TENSOR, constructor_sets_dimensions_and_strides)
{
    std::vector<uint64_t> dims = { 2, 3, 4 };
    Tensor<float> t(dims, MemoryLocation::DEVICE);

    EXPECT_EQ(t.m_dimensions, dims);
    EXPECT_EQ(t.m_mem_loc, MemoryLocation::DEVICE);

    std::vector<uint64_t> expected_strides = { 12, 4, 1 };
    EXPECT_EQ(t.m_strides, expected_strides);
}

/**
 * @test TENSOR.constructor_zero_initializes_data
 * @brief Tests that the Tensor constructor allocates the
 * correct amount of memory and initializes it to zero.
 */
TEST(TENSOR, constructor_zero_initializes_data)
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
 * @test TENSOR.constructor_memory_location_and_access
 * @brief Tests correct memory location assignment.
 */
TEST(TENSOR, constructor_memory_location_and_access)
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
 * @test TENSOR.copy_assignment_operator
 * @brief Tests copy assignment operator.
 */
TEST(TENSOR, copy_assignment_operator)
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
 * @test TENSOR.move_assignment_operator
 * @brief Tests move assignment operator.
 */
TEST(TENSOR, move_assignment_operator)
{
    Tensor<float> t1({2, 2}, MemoryLocation::HOST);
    std::vector<float> values = {13.0f, 14.0f, 15.0f, 16.0f};
    t1 = values;

    float* original_ptr = t1.m_p_data.get();
    MemoryLocation original_loc = t1.m_mem_loc;

    Tensor<float> t2;
    t2 = std::move(t1);

    EXPECT_EQ(t2.m_p_data.get(), original_ptr);
    EXPECT_EQ(t2.m_mem_loc, original_loc);
    EXPECT_EQ(t1.m_p_data.get(), nullptr);
}

/**
 * @test TENSOR.assignment_from_flat_vector
 * @brief Tests assignment from flat std::vector.
 */
TEST(TENSOR, assignment_from_flat_vector)
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
 * @test TENSOR.assign_flat_vector_size_mismatch_throws
 * @brief Throws if assigning flat vector with incorrect size.
 */
TEST(TENSOR, assign_flat_vector_size_mismatch_throws)
{
    Tensor<float> t({2, 2});
    std::vector<float> values = {1.0f, 2.0f};

    EXPECT_THROW({
        t = values;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.slice_view_preserves_strides_and_data
 * @brief Tests that slicing a CHW-format tensor creates
 * a view with correct strides and verifies that the data
 * accessed via the view matches expected values.
 */
TEST(TENSOR, slice_view_preserves_strides_and_data)
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
 * @test TENSOR.slice_identity_preserves_layout
 * @brief Verifies that slicing a tensor without dropping
 * any axes returns a view with identical dimensions, strides, and values.
 */
TEST(TENSOR, slice_identity_preserves_layout)
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
 * @test TENSOR.slice_invalid_arguments_throw
 * @brief Ensures that invalid slice arguments
 * (e.g., mismatched ranks, out-of-bounds access)
 * correctly throw exceptions.
 */
TEST(TENSOR, slice_invalid_arguments_throw)
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
 * @test TENSOR.slice_4d_drops_prefix_axes
 * @brief Tests slicing a 4D tensor while dropping the first two axes,
 * verifying correct shape, strides, and values in the resulting view.
 *
 * The original tensor has shape {2, 3, 4, 5} and is filled with values from 1 to 120.
 * The slice extracts the {4, 5} sub-tensor at position (0, 0, :, :)
 */
TEST(TENSOR, slice_4d_drops_prefix_axes)
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
 * @test TENSOR.slice_4d_extracts_3d_volume
 * @brief Extracts a 3D chunk from a 4D tensor by dropping the first axis and verifies
 * shape, stride, and copied values.
 */
TEST(TENSOR, slice_4d_extracts_3d_volume)
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
 * @test TENSOR.slice_4d_extracts_1d_row
 * @brief Slices a single row (1D) from the last dimension of a 4D tensor,
 * and verifies the extracted values.
 */
TEST(TENSOR, slice_4d_extracts_1d_row)
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
 * @test TENSOR.slice_chw_extracts_large_patch
 * @brief Slices a 100x100 patch from a 3D CHW-format tensor
 * at a specified spatial location.
 * Verifies shape, stride, and content correctness.
 */
TEST(TENSOR, slice_chw_extracts_large_patch)
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
 * @test TENSOR.tensor_view_modification_reflects_in_original
 * @brief Tests that modifying a tensor view updates the original tensor's memory.
 *
 * A view is created on a region of the tensor. Data is written via the view and
 * then read again from a second view on the original tensor. The test verifies
 * that the data matches, confirming memory is shared.
 */
TEST(TENSOR, tensor_view_modification_reflects_in_original)
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
 * @test TENSOR.owner_destroyed_before_view
 * @brief Ensure a view's aliasing shared_ptr keeps the underlying buffer alive
 * after the original owner goes out of scope.
 */

TEST(TENSOR, owner_destroyed_before_view)
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
 * @test TENSOR.view_destroyed_before_owner
 * @brief Ensure the buffer remains alive while the owner exists and is freed
 * after the owner releases ownership.
 */

TEST(TENSOR, view_destroyed_before_owner)
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
 * @test TENSOR.operator_index_chain_assign_and_read
 * @brief Tests that chained operator[] returns views
 * and allows assignment and reading.
 */
TEST(TENSOR, operator_index_chain_assign_and_read)
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
 * @test TENSOR.operator_index_chain_const_access
 * @brief Tests that const Tensor supports chained operator[] reading.
 */
TEST(TENSOR, operator_index_chain_const_access)
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
 * @test TENSOR.scalar_assignment_and_conversion
 * @brief Tests operator=(float_t)
 * and operator float_t() on 1-element tensors / views.
 */
TEST(TENSOR, scalar_assignment_and_conversion)
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
 * @test TENSOR.scalar_assignment_to_default_constructed_tensor
 * @brief Tests that assigning a scalar to a default-constructed tensor
 * automatically initializes it as a scalar tensor of shape {1} and stores
 * the assigned value correctly.
 *
 * Also verifies that assigning a scalar to a tensor with more than one element
 * throws std::invalid_argument.
 */
TEST(TENSOR, scalar_assignment_to_default_constructed_tensor) {
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
 * @test TENSOR.scalar_assignment_wrong_size_throws
 * @brief Assigning a scalar to a non-1-element tensor must throw.
 */
TEST(TENSOR, scalar_assignment_wrong_size_throws)
{
    Tensor<float> t({2, 2}, MemoryLocation::HOST);
    EXPECT_THROW(
    {
        t = 3.14f;
    }, std::invalid_argument);
}

/**
 * @test TENSOR.operator_index_out_of_bounds_throws
 * @brief operator[] should throw when index is out of range.
 */
TEST(TENSOR, operator_index_out_of_bounds_throws)
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
 * @test TENSOR.view_constructor_middle_chunk_write_and_read
 * @brief Create a 3D tensor, take a chunk in the middle of a channel
 * with the view constructor, write via chained operator[] on the view
 * and verify reads from both the view and the original tensor
 * at the expected positions.
 */
TEST(TENSOR, view_constructor_middle_chunk_write_and_read)
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
 * @test TENSOR.print_empty_tensor
 * @brief Checks that print correctly outputs an empty tensor.
 *
 * Creates a tensor with no dimensions and verifies that print
 * outputs the string "[]\n".
 */
TEST(TENSOR, print_empty_tensor)
{
    temper::Tensor<float> t({}, temper::MemoryLocation::HOST);
    std::stringstream ss;
    t.print(ss);
    EXPECT_EQ(ss.str(), "[]\n");
}

} // namespace Test