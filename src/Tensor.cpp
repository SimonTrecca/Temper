/**
 * @file Tensor.cpp
 * @brief Tensor class function definitions.
 */

#include "temper/Tensor.hpp"
#include <iostream>

namespace temper{

template<typename float_t>
void Tensor<float_t>::compute_strides()
{
    m_strides.resize(m_dimensions.size());

    // Empty shape: nothing to do; callers should normally prevent this.
    if (m_dimensions.empty())
    {
        return;
    }

    // Use uint64_t limits for overflow guards below.
    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    m_strides.back() = 1;

    for (uint64_t i = m_dimensions.size() - 1; i > 0; --i)
    {
        uint64_t dim = m_dimensions[i];
        if (dim == 0)
        {
            // Zero-sized dims are invalid.
            throw std::invalid_argument
                (R"(Tensor(compute_strides):
                    zero-sized dimension encountered.)");
        }
        uint64_t next_stride = m_strides[i];

        if (next_stride > U64_MAX / dim)
        {
            // Prevent silent wraparound on multiplication.
            throw std::overflow_error
                (R"(Tensor(compute_strides):
                    stride multiplication overflow.)");
        }
        m_strides[i - 1] = next_stride * dim;
    }
}

template<typename float_t>
Tensor<float_t>::Tensor(const std::vector<uint64_t> & dimensions,
                        MemoryLocation loc)
    : m_dimensions(dimensions),
      m_strides(dimensions.size()),
      m_own_data(true),
      m_mem_loc(loc)
{

    if (m_dimensions.empty())
    {
        // Reject rank-0: caller must provide a non-empty shape.
        throw std::invalid_argument(R"(Tensor(main constructor):
            dims must not be empty (rank-0 not supported).)");
    }

    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    // Compute total element count with overflow guard.
    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        if (d == 0)
        {
            // Zero-sized dims are invalid.
            throw std::invalid_argument(R"(Tensor(main constructor):
                zero-sized dimension is not allowed.)");
        }
        if (total_size > U64_MAX / d)
        {
            // Prevent silent wraparound on multiplication.
            throw std::overflow_error(R"(Tensor(main constructor):
                total element count overflow (too many elements).)");
        }
        total_size *= d;
    }

    // Compute byte count safely and ensure it fits in size_t.
    const uint64_t elem_size_u64 = static_cast<uint64_t>(sizeof(float_t));

    if (total_size > U64_MAX / elem_size_u64)
    {
        // Total_size * elem_size_u64 must not overflow uint64_t.
        throw std::overflow_error(R"(Tensor(main constructor):
            allocation size (bytes) overflow (uint64_t).)");
    }

    // Compute byte count in uint64_t.
    const uint64_t alloc_bytes_u64 = total_size * elem_size_u64;

    const uint64_t max_size_t_u64 = static_cast<uint64_t>
        (std::numeric_limits<size_t>::max());
    if (alloc_bytes_u64 > max_size_t_u64)
    {
        // Byte count must fit in platform size_t.
        throw std::overflow_error(R"(Tensor(main constructor): allocation size
            (bytes) doesn't fit into size_t on this platform.)");
    }
    // Safe to narrow to size_t.
    const size_t allocation_bytes = static_cast<size_t>(alloc_bytes_u64);

    // Compute strides now that dimensions are validated.
    compute_strides();

    // Query device limits and fail early if the request is impossible.
    auto dev = g_sycl_queue.get_device();
    const uint64_t dev_max_alloc = static_cast<uint64_t>(
        dev.get_info<sycl::info::device::max_mem_alloc_size>());
    const uint64_t dev_global_mem = static_cast<uint64_t>(
        dev.get_info<sycl::info::device::global_mem_size>());
    if (alloc_bytes_u64 > dev_max_alloc)
    {
        throw std::runtime_error(R"(Tensor(main constructor):
            requested allocation exceeds device max_mem_alloc_size.)");
    }
    if (alloc_bytes_u64 > dev_global_mem)
    {
        throw std::runtime_error(R"(Tensor(main constructor):
            requested allocation exceeds device global_mem_size.)");
    }

    // Allocate USM (shared for HOST, device for DEVICE).
    float_t* raw_ptr = nullptr;
    if (m_mem_loc == MemoryLocation::HOST)
    {
        raw_ptr = static_cast<float_t*>(
            sycl::malloc_shared(allocation_bytes, g_sycl_queue));
    }
    else
    {
        raw_ptr = static_cast<float_t*>(
            sycl::malloc_device(allocation_bytes, g_sycl_queue));
    }
    if (!raw_ptr)
    {
        // Allocation failed; throw bad_alloc.
        throw std::bad_alloc();
    }
    m_p_data = std::shared_ptr<float_t>(raw_ptr,
        [](float_t* p)
        {
            if (p)
            {
                sycl::free(p, g_sycl_queue);
            }
        }
    );

    // Zero-initialize the buffer.
    g_sycl_queue.memset(m_p_data.get(), 0, allocation_bytes).wait();
}

template<typename float_t>
Tensor<float_t>::Tensor(const Tensor & other)
    : m_dimensions(other.m_dimensions),
      m_strides(other.m_strides),
      m_own_data(other.m_own_data),
      m_mem_loc(other.m_mem_loc)
{
    if (m_own_data)
    {
        // If other has been default constructed, build an empty ptr and return.
        if (m_dimensions.empty())
        {
            m_p_data = std::shared_ptr<float_t>(nullptr);
            return;
        }

        uint64_t total_size = 1;
        for (uint64_t d : m_dimensions)
        {
            total_size *= d;
        }

        const size_t alloc_bytes =
            static_cast<size_t>(total_size) * sizeof(float_t);

        // Allocate same kind of USM as other's mem_loc.
        float_t* raw_ptr = nullptr;
        if (m_mem_loc == MemoryLocation::HOST)
        {
            raw_ptr = static_cast<float_t*>
                (sycl::malloc_shared(alloc_bytes, g_sycl_queue));
        }
        else
        {
            raw_ptr = static_cast<float_t*>
                (sycl::malloc_device(alloc_bytes, g_sycl_queue));
        }

        if (!raw_ptr)
        {
            throw std::bad_alloc();
        }

        m_p_data = std::shared_ptr<float_t>(raw_ptr,
            [](float_t* p) { if (p) sycl::free(p, g_sycl_queue); });

        // Copy contents (assume other.m_p_data is valid).
        g_sycl_queue.memcpy
            (m_p_data.get(), other.m_p_data.get(), alloc_bytes).wait();
    }
    else
    {
        // Share control block (view).
        m_p_data = other.m_p_data;
    }
}

template<typename float_t>
Tensor<float_t>::Tensor(Tensor && other) noexcept
    : m_p_data(std::move(other.m_p_data)),
      m_dimensions(std::move(other.m_dimensions)),
      m_strides(std::move(other.m_strides)),
      m_own_data(other.m_own_data),
      m_mem_loc(other.m_mem_loc)
{
    other.m_dimensions.clear();
    other.m_strides.clear();
    other.m_p_data.reset();
    other.m_own_data = true;
}

template<typename float_t>
Tensor<float_t>::Tensor(Tensor & other,
                  const std::vector<uint64_t> & start_indices,
                  const std::vector<uint64_t> & view_shape)
    : m_own_data(false),
      m_mem_loc(other.m_mem_loc)
{
    const uint64_t original_rank = other.m_dimensions.size();
    const uint64_t view_rank = view_shape.size();

    if (!other.m_p_data) {
        throw std::runtime_error(R"(Tensor(view constructor):
            cannot create view from uninitialized tensor.)");
    }

    if (start_indices.size() != original_rank)
    {
        throw std::invalid_argument(R"(Tensor(view constructor):
            start_indices must match tensor rank.)");
    }

    if (view_rank == 0 || view_rank > original_rank)
    {
        throw std::invalid_argument
            (R"(Tensor(view constructor):
                view shape rank must be between 1 and tensor rank.)");
    }

    // Check bounds for start indices and view dimensions.
    for (uint64_t i = 0; i < original_rank; ++i)
    {
        if (start_indices[i] >= other.m_dimensions[i])
        {
            throw std::out_of_range(R"(Tensor(view constructor):
                start index out of bounds.)");
        }
    }
    for (uint64_t j = 0; j < view_rank; ++j)
    {
        uint64_t i = original_rank - view_rank + j;
        if (view_shape[j] == 0 ||
            start_indices[i] + view_shape[j] > other.m_dimensions[i])
        {
            throw std::out_of_range(R"(Tensor(view constructor):
                view shape out of bounds.)");
        }
    }

    uint64_t offset = 0;
    for (uint64_t i = 0; i < original_rank; ++i)
    {
        offset += start_indices[i] * other.m_strides[i];
    }

    m_p_data = std::shared_ptr<float_t>
        (other.m_p_data, other.m_p_data.get() + offset);

    // Set dimensions and strides for the view.
    m_dimensions.assign(view_shape.begin(), view_shape.end());
    m_strides.assign(other.m_strides.end() - view_rank, other.m_strides.end());
}

template<typename float_t>
Tensor<float_t> & Tensor<float_t>::operator=(const Tensor & other)
{
    if (this != &other)
    {
        m_dimensions = other.m_dimensions;
        m_strides = other.m_strides;
        m_own_data = other.m_own_data;
        m_mem_loc = other.m_mem_loc;

        if (m_own_data)
        {
            if (m_dimensions.empty())
            {
                m_p_data = std::shared_ptr<float_t>(nullptr);
                return *this;
            }

            uint64_t total_size = 1;
            for (uint64_t d : m_dimensions)
            {
                total_size *= d;
            }

            const size_t alloc_bytes =
                static_cast<size_t>(total_size) * sizeof(float_t);

            float_t* raw_ptr = nullptr;
            if (m_mem_loc == MemoryLocation::HOST)
            {
                raw_ptr = static_cast<float_t*>(sycl::malloc_shared
                    (alloc_bytes, g_sycl_queue));
            }
            else
            {
                raw_ptr = static_cast<float_t*>(sycl::malloc_device
                    (alloc_bytes, g_sycl_queue));
            }

            if (!raw_ptr){
                throw std::bad_alloc();
            }

            m_p_data = std::shared_ptr<float_t>(raw_ptr,
                [](float_t* p)
                {
                    if (p)
                    {
                        sycl::free(p, g_sycl_queue);
                    }
                }
            );

            g_sycl_queue.memcpy(m_p_data.get(), other.m_p_data.get(),
                                alloc_bytes).wait();
        }
        else
        {
            m_p_data = other.m_p_data;
        }
    }
    return *this;
}

template<typename float_t>
Tensor<float_t>& Tensor<float_t>::operator=(Tensor && other) noexcept
{
    if (this != &other)
    {
        m_p_data = std::move(other.m_p_data);
        m_dimensions = std::move(other.m_dimensions);
        m_strides = std::move(other.m_strides);
        m_own_data = other.m_own_data;
        m_mem_loc = other.m_mem_loc;

        other.m_p_data.reset();
        other.m_dimensions.clear();
        other.m_strides.clear();
        other.m_own_data = true;
    }
    return *this;
}


template<typename float_t>
Tensor<float_t> & Tensor<float_t>::operator=(const std::vector<float_t> & values)
{
    if (m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(values assignment):
            target tensor has no elements.)");
    }
    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        total_size *= d;
    }

    uint64_t values_size = static_cast<uint64_t>(values.size());
    if (values_size != total_size)
    {
        throw std::invalid_argument(R"(Tensor(values assignment):
            size mismatch in 1D vector assignment.)");
    }

    const size_t alloc_bytes =
                static_cast<size_t>(total_size) * sizeof(float_t);

    g_sycl_queue.memcpy
        (m_p_data.get(), values.data(), alloc_bytes).wait();

    return *this;
}

template<typename float_t>
Tensor<float_t> & Tensor<float_t>::operator=(float_t val)
{
    if (m_dimensions.empty())
    {
        m_dimensions = {1};
        compute_strides();

        float_t* raw_ptr = nullptr;
        if (m_mem_loc == MemoryLocation::HOST)
        {
            raw_ptr = static_cast<float_t*>(
                sycl::malloc_shared(sizeof(float_t), g_sycl_queue));
        }
        else
        {
            raw_ptr = static_cast<float_t*>(
                sycl::malloc_device(sizeof(float_t), g_sycl_queue));
        }

        m_p_data = std::shared_ptr<float_t>(raw_ptr,
            [](float_t* p)
            {
                if (p)
                {
                    sycl::free(p, g_sycl_queue);
                }
            }
        );

        m_own_data = true;
    }

    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        total_size *= d;
    }

    if (total_size != 1)
    {
        throw std::invalid_argument(R"(Tensor(single value assignment):
            scalar assignment only allowed for tensors with single element.)");
    }

    g_sycl_queue.memcpy(m_p_data.get(), &val, sizeof(float_t)).wait();
    return *this;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator[](uint64_t idx)
{
    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());
    if (rank == 0)
    {
        throw std::out_of_range(R"(Tensor(operator[]):
            tensor has no elements.)");
    }
    if (idx >= m_dimensions[0])
    {
        throw std::out_of_range(R"(Tensor(operator[]):
            Index out of bounds (operator[]).)");
    }

    if (rank == 1)
    {
        std::vector<uint64_t> start_indices{ idx };
        std::vector<uint64_t> view_shape{ 1 };
        return Tensor(*this, start_indices, view_shape);
    }
    else
    {
        std::vector<uint64_t> start_indices(rank, 0);
        start_indices[0] = idx;
        std::vector<uint64_t> view_shape
            (m_dimensions.begin() + 1, m_dimensions.end());
        return Tensor(*this, start_indices, view_shape);
    }
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator[](uint64_t idx) const
{
    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());
    if (rank == 0)
    {
        throw std::out_of_range(R"(Tensor(operator[]):
            tensor has no elements.)");
    }
    if (idx >= m_dimensions[0])
    {
        throw std::out_of_range(R"(Tensor(operator[]):
            Index out of bounds (operator[]).)");
    }

    if (rank == 1)
    {
        std::vector<uint64_t> start_indices{ idx };
        std::vector<uint64_t> view_shape{ 1 };
        return Tensor(const_cast<Tensor&>(*this), start_indices, view_shape);
    }
    else
    {
        std::vector<uint64_t> start_indices(rank, 0);
        start_indices[0] = idx;
        std::vector<uint64_t> view_shape
            (m_dimensions.begin() + 1, m_dimensions.end());
        return Tensor(const_cast<Tensor&>(*this), start_indices, view_shape);
    }
}

template<typename float_t>
Tensor<float_t>::operator float_t() const
{
    if (m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(implicit type conversion):
            tensor has no elements.)");
    }

    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        total_size *= d;
    }

    if (total_size != 1)
    {
        throw std::invalid_argument
            (R"(Tensor(implicit type conversion):
                scalar read only allowed for tensors with single element.)");
    }

    float_t tmp;

    g_sycl_queue.memcpy(&tmp, m_p_data.get(), sizeof(float_t)).wait();
    return tmp;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator+(const Tensor & other) const
{
    if (m_dimensions.empty() || other.m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(operator+):
            either tensor has no elements.)");
    }

    // Align the shapes for broadcasting.
    const uint64_t rank_a = static_cast<uint64_t>(m_dimensions.size());
    const uint64_t rank_b = static_cast<uint64_t>(other.m_dimensions.size());
    const uint64_t max_rank = std::max(rank_a, rank_b);

    std::vector<uint64_t> a_shape_aligned(max_rank, 1);
    std::vector<uint64_t> b_shape_aligned(max_rank, 1);

    std::vector<uint64_t> a_strides_aligned(max_rank, 0);
    std::vector<uint64_t> b_strides_aligned(max_rank, 0);

    for (uint64_t i = 0; i < rank_a; ++i)
    {
        a_shape_aligned[max_rank - rank_a + i] = m_dimensions[i];
        a_strides_aligned[max_rank - rank_a + i] = m_strides[i];
    }
    for (uint64_t i = 0; i < rank_b; ++i)
    {
        b_shape_aligned[max_rank - rank_b + i] = other.m_dimensions[i];
        b_strides_aligned[max_rank - rank_b + i] = other.m_strides[i];
    }

    std::vector<uint64_t> out_shape(max_rank);
    std::vector<uint64_t> a_strides_broadcasted(max_rank, 0);
    std::vector<uint64_t> b_strides_broadcasted(max_rank, 0);

    // Compute output shape and effective strides (for broadcasting).
    for (uint64_t d = 0; d < max_rank; ++d)
    {
        if (a_shape_aligned[d] == b_shape_aligned[d])
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (a_shape_aligned[d] == 1)
        {
            out_shape[d] = b_shape_aligned[d];
            a_strides_broadcasted[d] = 0;
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (b_shape_aligned[d] == 1)
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = 0;
        }
        else
        {
            throw std::invalid_argument(R"(Tensor(operator+):
                incompatible shapes for broadcasting.)");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        total_size *= dim;
    }

    MemoryLocation res_loc;

    if (m_mem_loc == MemoryLocation::DEVICE ||
        other.m_mem_loc == MemoryLocation::DEVICE)
    {
        res_loc = MemoryLocation::DEVICE;
    }
    else
    {
        res_loc = MemoryLocation::HOST;
    }

    Tensor result(out_shape, res_loc);

    uint64_t* p_result_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));

    g_sycl_queue.memcpy(p_result_strides,
        result.m_strides.data(), sizeof(uint64_t) * max_rank).wait();

    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    // Shared error flag (0 = OK, 1 = NaN in inputs, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    float_t* p_a_data = m_p_data.get();
    float_t* p_b_data = other.m_p_data.get();
    float_t* p_r_data = result.m_p_data.get();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            auto atomic_err = sycl::atomic_ref<int32_t,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*(p_error_flag));

            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);
            uint64_t remainder = flat_idx;
            uint64_t offset_a = 0;
            uint64_t offset_b = 0;

            for (uint64_t dim = 0; dim < max_rank; ++dim)
            {
                uint64_t stride = p_result_strides[dim];
                uint64_t coord = remainder / stride;
                remainder = remainder % stride;

                offset_a += coord * p_a_strides[dim];
                offset_b += coord * p_b_strides[dim];
            }

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            float_t res = a_val + b_val;

            if (std::isnan(a_val) || std::isnan(b_val) || std::isnan(res))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 1);

                if (std::numeric_limits<float_t>::has_quiet_NaN)
                {
                    p_r_data[flat_idx] =
                        std::numeric_limits<float_t>::quiet_NaN();
                }
                else
                {
                    p_r_data[flat_idx] = res;
                }
                return;
            }
            if (!std::isfinite(res))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 2);
                p_r_data[flat_idx] = res;
                return;
            }

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_result_strides, g_sycl_queue);
    sycl::free(p_a_strides, g_sycl_queue);
    sycl::free(p_b_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(Tensor(operator+):
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(Tensor(operator+):
                non-finite result (overflow or Inf).)");
        }
        throw std::runtime_error(R"(Tensor(operator+):
            numeric error during element-wise addition.)");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator-(const Tensor & other) const
{
    if (m_dimensions.empty() || other.m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(operator-):
            either tensor has no elements.)");
    }

    const uint64_t rank_a = static_cast<uint64_t>(m_dimensions.size());
    const uint64_t rank_b = static_cast<uint64_t>(other.m_dimensions.size());
    const uint64_t max_rank = std::max(rank_a, rank_b);

    std::vector<uint64_t> a_shape_aligned(max_rank, 1);
    std::vector<uint64_t> b_shape_aligned(max_rank, 1);

    std::vector<uint64_t> a_strides_aligned(max_rank, 0);
    std::vector<uint64_t> b_strides_aligned(max_rank, 0);

    for (uint64_t i = 0; i < rank_a; ++i)
    {
        a_shape_aligned[max_rank - rank_a + i] = m_dimensions[i];
        a_strides_aligned[max_rank - rank_a + i] = m_strides[i];
    }

    for (uint64_t i = 0; i < rank_b; ++i)
    {
        b_shape_aligned[max_rank - rank_b + i] = other.m_dimensions[i];
        b_strides_aligned[max_rank - rank_b + i] = other.m_strides[i];
    }

    std::vector<uint64_t> out_shape(max_rank);
    std::vector<uint64_t> a_strides_broadcasted(max_rank, 0);
    std::vector<uint64_t> b_strides_broadcasted(max_rank, 0);

    // Compute output shape and effective strides (for broadcasting).
    for (uint64_t d = 0; d < max_rank; ++d)
    {
        if (a_shape_aligned[d] == b_shape_aligned[d])
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (a_shape_aligned[d] == 1)
        {
            out_shape[d] = b_shape_aligned[d];
            a_strides_broadcasted[d] = 0;
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (b_shape_aligned[d] == 1)
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = 0;
        }
        else
        {
            throw std::invalid_argument(R"(Tensor(operator-):
                incompatible shapes for broadcasting.)");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        total_size *= dim;
    }

    MemoryLocation res_loc;

    if (m_mem_loc == MemoryLocation::DEVICE ||
        other.m_mem_loc == MemoryLocation::DEVICE)
    {
        res_loc = MemoryLocation::DEVICE;
    }
    else
    {
        res_loc = MemoryLocation::HOST;
    }

    Tensor result(out_shape, res_loc);

    uint64_t* p_result_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));

    g_sycl_queue.memcpy(p_result_strides,
        result.m_strides.data(), sizeof(uint64_t) * max_rank).wait();

    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    // Shared error flag (0 = OK, 1 = NaN in inputs, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    float_t* p_a_data = m_p_data.get();
    float_t* p_b_data = other.m_p_data.get();
    float_t* p_r_data = result.m_p_data.get();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            auto atomic_err = sycl::atomic_ref<int32_t,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*(p_error_flag));

            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);
            uint64_t remainder = flat_idx;
            uint64_t offset_a = 0;
            uint64_t offset_b = 0;

            // Generic loop to find the index given the coordinates.
            for (uint64_t dim = 0; dim < max_rank; ++dim)
            {
                uint64_t stride = p_result_strides[dim];
                uint64_t coord = remainder / stride;
                remainder = remainder % stride;

                offset_a += coord * p_a_strides[dim];
                offset_b += coord * p_b_strides[dim];
            }

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            float_t res = a_val - b_val;

            if (std::isnan(a_val) || std::isnan(b_val) || std::isnan(res))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 1);

                if (std::numeric_limits<float_t>::has_quiet_NaN)
                {
                    p_r_data[flat_idx] =
                        std::numeric_limits<float_t>::quiet_NaN();
                }
                else
                {
                    p_r_data[flat_idx] = res;
                }
                return;
            }
            if (!std::isfinite(res))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 2);
                p_r_data[flat_idx] = res;
                return;
            }

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_result_strides, g_sycl_queue);
    sycl::free(p_a_strides, g_sycl_queue);
    sycl::free(p_b_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(Tensor(operator-):
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(Tensor(operator-):
                non-finite result (overflow or Inf).)");
        }
        throw std::runtime_error(R"(Tensor(operator-):
            numeric error during element-wise addition.)");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator*(const Tensor & other) const
{
    if (m_dimensions.empty() || other.m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(operator*):
            either tensor has no elements.)");
    }

    const uint64_t rank_a = static_cast<uint64_t>(m_dimensions.size());
    const uint64_t rank_b = static_cast<uint64_t>(other.m_dimensions.size());
    const uint64_t max_rank = std::max(rank_a, rank_b);

    std::vector<uint64_t> a_shape_aligned(max_rank, 1);
    std::vector<uint64_t> b_shape_aligned(max_rank, 1);

    std::vector<uint64_t> a_strides_aligned(max_rank, 0);
    std::vector<uint64_t> b_strides_aligned(max_rank, 0);

    for (uint64_t i = 0; i < rank_a; ++i)
    {
        a_shape_aligned[max_rank - rank_a + i] = m_dimensions[i];
        a_strides_aligned[max_rank - rank_a + i] = m_strides[i];
    }

    for (uint64_t i = 0; i < rank_b; ++i)
    {
        b_shape_aligned[max_rank - rank_b + i] = other.m_dimensions[i];
        b_strides_aligned[max_rank - rank_b + i] = other.m_strides[i];
    }

    std::vector<uint64_t> out_shape(max_rank);
    std::vector<uint64_t> a_strides_broadcasted(max_rank, 0);
    std::vector<uint64_t> b_strides_broadcasted(max_rank, 0);

    // Compute output shape and effective strides (for broadcasting).
    for (uint64_t d = 0; d < max_rank; ++d)
    {
        if (a_shape_aligned[d] == b_shape_aligned[d])
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (a_shape_aligned[d] == 1)
        {
            out_shape[d] = b_shape_aligned[d];
            a_strides_broadcasted[d] = 0;
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (b_shape_aligned[d] == 1)
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = 0;
        }
        else
        {
            throw std::invalid_argument(R"(Tensor(operator*):
                incompatible shapes for broadcasting.)");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        total_size *= dim;
    }

    MemoryLocation res_loc;

    if (m_mem_loc == MemoryLocation::DEVICE ||
        other.m_mem_loc == MemoryLocation::DEVICE)
    {
        res_loc = MemoryLocation::DEVICE;
    }
    else
    {
        res_loc = MemoryLocation::HOST;
    }

    Tensor result(out_shape, res_loc);

    uint64_t* p_result_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));

    g_sycl_queue.memcpy(p_result_strides,
        result.m_strides.data(), sizeof(uint64_t) * max_rank).wait();

    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    // Shared error flag (0 = OK, 1 = NaN in inputs, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    float_t* p_a_data = m_p_data.get();
    float_t* p_b_data = other.m_p_data.get();
    float_t* p_r_data = result.m_p_data.get();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            auto atomic_err = sycl::atomic_ref<int32_t,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*(p_error_flag));

            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);
            uint64_t remainder = flat_idx;
            uint64_t offset_a = 0;
            uint64_t offset_b = 0;

            // Generic loop to find the index given the coordinates.
            for (uint64_t dim = 0; dim < max_rank; ++dim)
            {
                uint64_t stride = p_result_strides[dim];
                uint64_t coord = remainder / stride;
                remainder = remainder % stride;

                offset_a += coord * p_a_strides[dim];
                offset_b += coord * p_b_strides[dim];
            }

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            float_t res = a_val * b_val;

            if (std::isnan(a_val) || std::isnan(b_val) || std::isnan(res))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 1);

                if (std::numeric_limits<float_t>::has_quiet_NaN)
                {
                    p_r_data[flat_idx] =
                        std::numeric_limits<float_t>::quiet_NaN();
                }
                else
                {
                    p_r_data[flat_idx] = res;
                }
                return;
            }
            if (!std::isfinite(res))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 2);
                p_r_data[flat_idx] = res;
                return;
            }

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_result_strides, g_sycl_queue);
    sycl::free(p_a_strides, g_sycl_queue);
    sycl::free(p_b_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(Tensor(operator*):
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(Tensor(operator*):
                non-finite result (overflow or Inf).)");
        }
        throw std::runtime_error(R"(Tensor(operator*):
            numeric error during element-wise addition.)");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator/(const Tensor & other) const
{
    if (m_dimensions.empty() || other.m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(operator/):
            either tensor has no elements.)");
    }

    const uint64_t rank_a = static_cast<uint64_t>(m_dimensions.size());
    const uint64_t rank_b = static_cast<uint64_t>(other.m_dimensions.size());
    const uint64_t max_rank = std::max(rank_a, rank_b);

    std::vector<uint64_t> a_shape_aligned(max_rank, 1);
    std::vector<uint64_t> b_shape_aligned(max_rank, 1);

    std::vector<uint64_t> a_strides_aligned(max_rank, 0);
    std::vector<uint64_t> b_strides_aligned(max_rank, 0);

    for (uint64_t i = 0; i < rank_a; ++i)
    {
        a_shape_aligned[max_rank - rank_a + i] = m_dimensions[i];
        a_strides_aligned[max_rank - rank_a + i] = m_strides[i];
    }

    for (uint64_t i = 0; i < rank_b; ++i)
    {
        b_shape_aligned[max_rank - rank_b + i] = other.m_dimensions[i];
        b_strides_aligned[max_rank - rank_b + i] = other.m_strides[i];
    }

    std::vector<uint64_t> out_shape(max_rank);
    std::vector<uint64_t> a_strides_broadcasted(max_rank, 0);
    std::vector<uint64_t> b_strides_broadcasted(max_rank, 0);

    for (uint64_t d = 0; d < max_rank; ++d)
    {
        if (a_shape_aligned[d] == b_shape_aligned[d])
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (a_shape_aligned[d] == 1)
        {
            out_shape[d] = b_shape_aligned[d];
            a_strides_broadcasted[d] = 0;
            b_strides_broadcasted[d] = b_strides_aligned[d];
        }
        else if (b_shape_aligned[d] == 1)
        {
            out_shape[d] = a_shape_aligned[d];
            a_strides_broadcasted[d] = a_strides_aligned[d];
            b_strides_broadcasted[d] = 0;
        }
        else
        {
            throw std::invalid_argument(R"(Tensor(operator/):
                incompatible shapes for broadcasting.)");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        total_size *= dim;
    }

    MemoryLocation res_loc;

    if (m_mem_loc == MemoryLocation::DEVICE ||
        other.m_mem_loc == MemoryLocation::DEVICE)
    {
        res_loc = MemoryLocation::DEVICE;
    }
    else
    {
        res_loc = MemoryLocation::HOST;
    }

    Tensor result(out_shape, res_loc);

    uint64_t* p_result_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));

    g_sycl_queue.memcpy(p_result_strides,
        result.m_strides.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    /*
      Shared error flag:
      0 = OK,
      1 = NaN in inputs,
      2 = division by zero,
      3 = non-finite result
     */
    int32_t* p_error_flag = static_cast<int32_t*>
        (sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    float_t* p_a_data = m_p_data.get();
    float_t* p_b_data = other.m_p_data.get();
    float_t* p_r_data = result.m_p_data.get();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            auto atomic_err = sycl::atomic_ref<int32_t,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*(p_error_flag));

            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);
            uint64_t remainder = flat_idx;
            uint64_t offset_a = 0;
            uint64_t offset_b = 0;

            // Generic loop to find the index given the coordinates.
            for (uint64_t dim = 0; dim < max_rank; ++dim)
            {
                uint64_t stride = p_result_strides[dim];
                uint64_t coord = remainder / stride;
                remainder = remainder % stride;

                offset_a += coord * p_a_strides[dim];
                offset_b += coord * p_b_strides[dim];
            }

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            if (std::isnan(a_val) || std::isnan(b_val))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 1);

                if (std::numeric_limits<float_t>::has_quiet_NaN)
                {
                    p_r_data[flat_idx] =
                        std::numeric_limits<float_t>::quiet_NaN();
                }
                else
                {
                    p_r_data[flat_idx] = a_val / b_val;
                }
                return;
            }
            if (b_val == 0.0f)
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 2);

                if (a_val >= 0.0f)
                {
                    p_r_data[flat_idx] =
                        std::numeric_limits<float_t>::infinity();
                }
                else
                {
                    p_r_data[flat_idx] =
                        -std::numeric_limits<float_t>::infinity();
                }

                return;
            }

            float_t res = a_val / b_val;
            if (!std::isfinite(res))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 3);
                p_r_data[flat_idx] = res;
                return;
            }

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_result_strides, g_sycl_queue);
    sycl::free(p_a_strides, g_sycl_queue);
    sycl::free(p_b_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(Tensor(operator/):
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(Tensor(operator/):
                division by zero detected.)");
        }
        if (err == 3)
        {
            throw std::runtime_error(R"(Tensor(operator/):
                non-finite result detected.)");
        }
        throw std::runtime_error(R"(Tensor(operator/):
            numeric error during element-wise division.)");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator-() const
{
    if (m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(operator-):
            tensor has no elements.)");
    }

    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());

    uint64_t total_size = 1;
    for (uint64_t dim : m_dimensions)
    {
        total_size *= dim;
    }

    MemoryLocation res_loc = m_mem_loc;
    Tensor result(m_dimensions, res_loc);

    uint64_t* p_shape =static_cast<uint64_t*>
        (sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_strides = static_cast<uint64_t*>
        (sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));

    g_sycl_queue.memcpy
        (p_shape, m_dimensions.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy
        (p_strides, m_strides.data(), sizeof(uint64_t) * rank).wait();

    int32_t* p_error_flag = static_cast<int32_t*>
        (sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    float_t* p_src = m_p_data.get();
    float_t* p_dst = result.m_p_data.get();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            auto atomic_err = sycl::atomic_ref<int32_t,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*(p_error_flag));

            uint64_t flat_idx = idx[0];
            uint64_t remainder = flat_idx;
            uint64_t offset = 0;

            // Generic loop to find the index given the coordinates.
            for (uint64_t dim = 0; dim < rank; ++dim)
            {
                uint64_t divisor = 1;
                for (uint64_t k = dim + 1; k < rank; ++k)
                {
                    divisor *= p_shape[k];
                }

                uint64_t coord = remainder / divisor;
                remainder = remainder % divisor;
                offset += coord * p_strides[dim];
            }

            float_t val = p_src[offset];
            if (std::isnan(val))
            {
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 1);
                p_dst[flat_idx] = std::numeric_limits<float_t>::quiet_NaN();
                return;
            }

            p_dst[flat_idx] = -val;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_shape, g_sycl_queue);
    sycl::free(p_strides, g_sycl_queue);

    if (err != 0)
    {
        throw std::runtime_error(R"(Tensor(operator-):
            NaN detected in input.)");
    }

    return result;
}

template<typename float_t>
void Tensor<float_t>::to(MemoryLocation target_loc)
{
    if (m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(to):
            tensor has no elements.)");
    }
    if (!m_own_data)
    {
        throw std::runtime_error
            (R"(Tensor(to):
                cannot move memory of a Tensor view (non-owning).)");
    }

    if (m_mem_loc == target_loc)
    {
        return;
    }

    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        total_size *= d;
    }

    float_t* raw_ptr = nullptr;
    if (target_loc == MemoryLocation::HOST)
    {
        raw_ptr = static_cast<float_t*>(
            sycl::malloc_shared(total_size * sizeof(float_t), g_sycl_queue));
    }
    else
    {
        raw_ptr = static_cast<float_t*>(
            sycl::malloc_device(total_size * sizeof(float_t), g_sycl_queue));
    }
    if (!raw_ptr)
    {
        throw std::bad_alloc();
    }

    std::shared_ptr<float_t> new_ptr = std::shared_ptr<float_t>(raw_ptr,
        [](float_t* p)
        {
            if (p)
            {
                sycl::free(p, g_sycl_queue);
            }
        }
    );

    g_sycl_queue.memcpy
        (new_ptr.get(), m_p_data.get(), total_size * sizeof(float_t)).wait();

    m_p_data = std::move(new_ptr);
    m_mem_loc = target_loc;
}

template<typename float_t>
void Tensor<float_t>::reshape(const std::vector<uint64_t>& new_dimensions)
{
    if (new_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(reshape):
            new_dimensions cannot be empty.)");
    }

    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    uint64_t og_total_size = 1;
    for (uint64_t dim : m_dimensions)
    {
        og_total_size *= dim;
    }

    uint64_t new_total_size = 1;
    for (uint64_t dim : new_dimensions)
    {
        if (dim == 0)
        {
            throw std::invalid_argument
                ("Tensor(reshape): new_dimensions cannot contain zero.");
        }
        if (new_total_size > U64_MAX / dim)
        {
            throw std::overflow_error
                ("Tensor(reshape): dimension product overflow.");
        }
        new_total_size *= dim;
    }

    if (new_total_size != og_total_size)
    {
        throw std::invalid_argument
            (R"(Tensor(reshape):
                total number of elements must remain the same.)");
    }

    m_dimensions = new_dimensions;
    compute_strides();
}


template<typename float_t>
void Tensor<float_t>::print(std::ostream& os) const
{
    std::function<void(uint64_t, uint64_t)> recurse
        = [&](uint64_t dim, uint64_t offset)
    {
        if (dim == m_dimensions.size() - 1)
        {
            os << "[";
            for (uint64_t i = 0; i < m_dimensions[dim]; ++i)
            {
                float_t val;
                g_sycl_queue.memcpy
                    (&val, m_p_data.get() + offset + i, sizeof(float_t)).wait();
                os << val;
                if (i != m_dimensions[dim] - 1)
                {
                    os << ", ";
                }
            }
            os << "]";
        } else
        {
            os << "[";
            for (uint64_t i = 0; i < m_dimensions[dim]; ++i)
            {
                recurse(dim + 1, offset + i * m_strides[dim]);
                if (i != m_dimensions[dim] - 1)
                {
                    os << ",\n" << std::string(dim + 1, ' ');
                }
            }
            os << "]";
        }
    };

    if (m_dimensions.empty())
    {
        os << "[]\n";
        return;
    }
    recurse(0, 0);
    os << "\n";
}

template class Tensor<float>;

} // namespace temper