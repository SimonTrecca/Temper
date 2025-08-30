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

        uint64_t total_size = other.get_num_elements();

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
Tensor<float_t>::Tensor(const Tensor & owner,
                        const std::vector<uint64_t> & start_indices,
                        const std::vector<uint64_t> & view_shape)
    : m_own_data(false),
      m_mem_loc(owner.m_mem_loc)
{
    const uint64_t original_rank = owner.m_dimensions.size();
    const uint64_t view_rank = view_shape.size();

    if (!owner.m_p_data)
    {
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
        throw std::invalid_argument(R"(Tensor(view constructor):
            view shape rank must be between 1 and tensor rank.)");
    }

    // Check bounds for start indices and view dimensions.
    for (uint64_t i = 0; i < original_rank; ++i)
    {
        if (start_indices[i] >= owner.m_dimensions[i])
        {
            throw std::out_of_range(R"(Tensor(view constructor):
                start index out of bounds.)");
        }
    }
    for (uint64_t j = 0; j < view_rank; ++j)
    {
        uint64_t i = original_rank - view_rank + j;
        if (view_shape[j] == 0 ||
            start_indices[i] + view_shape[j] > owner.m_dimensions[i])
        {
            throw std::out_of_range(R"(Tensor(view constructor):
                view shape out of bounds.)");
        }
    }

    uint64_t offset = 0;
    for (uint64_t i = 0; i < original_rank; ++i)
    {
        offset += start_indices[i] * owner.m_strides[i];
    }

    m_p_data = std::shared_ptr<float_t>
        (owner.m_p_data, owner.m_p_data.get() + offset);

    // Set dimensions and strides for the view.
    m_dimensions.assign(view_shape.begin(), view_shape.end());
    m_strides.assign(owner.m_strides.end() - view_rank, owner.m_strides.end());
}

template<typename float_t>
Tensor<float_t>::Tensor(const Tensor & owner,
                        const std::vector<uint64_t> & start_indices,
                        const std::vector<uint64_t> & dims,
                        const std::vector<uint64_t> & strides)
    : m_dimensions(dims),
      m_strides(strides),
      m_own_data(false),
      m_mem_loc(owner.m_mem_loc)
{
    if (!owner.m_p_data)
    {
        throw std::runtime_error(R"(Tensor(alias view constructor):
            cannot create view from uninitialized tensor.)");
    }

    const uint64_t owner_rank = static_cast<uint64_t>(owner.m_dimensions.size());
    const uint64_t view_rank = static_cast<uint64_t>(dims.size());

    if (start_indices.size() != owner_rank)
    {
        throw std::invalid_argument(R"(Tensor(alias view constructor):
            start_indices must match owner's rank.)");
    }

    if (strides.size() != view_rank)
    {
        throw std::invalid_argument(R"(Tensor(alias view constructor):
            dims and strides must have the same rank.)");
    }

    if (view_rank == 0)
    {
        throw std::invalid_argument(R"(Tensor(alias view constructor):
            view rank must be >= 1.)");
    }

    for (uint64_t i = 0; i < owner_rank; ++i)
    {
        if (start_indices[i] >= owner.m_dimensions[i])
        {
            throw std::out_of_range(R"(Tensor(alias view constructor):
                start index out of bounds.)");
        }
    }

    for (uint64_t j = 0; j < view_rank; ++j)
    {
        if (dims[j] == 0)
        {
            throw std::invalid_argument(R"(Tensor(alias view constructor):
                view dimensions must be non-zero.)");
        }
    }

    uint64_t offset = 0;
    for (uint64_t i = 0; i < owner_rank; ++i)
    {
        offset += owner.m_strides[i] * start_indices[i];
    }

    uint64_t owner_total = owner.get_num_elements();

    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();


    uint64_t max_index = offset;
    for (uint64_t j = 0; j < view_rank; ++j)
    {
        uint64_t dimm1 = dims[j] - 1;
        uint64_t vstride = strides[j];

        if (vstride != 0 && dimm1 > 0)
        {
            if (vstride > U64_MAX / dimm1)
            {
                throw std::overflow_error(R"(Tensor(alias view constructor):
                    stride * (dim-1) overflow.)");
            }
            uint64_t add = vstride * dimm1;
            if (max_index > U64_MAX - add)
            {
                throw std::overflow_error(R"(Tensor(alias view constructor):
                    max index computation overflow.)");
            }
            max_index += add;
        }
    }

    if (max_index >= owner_total)
    {
        throw std::out_of_range(R"(Tensor(alias view constructor):
            view exceeds owner's bounds.)");
    }

    m_p_data =
        std::shared_ptr<float_t>(owner.m_p_data, owner.m_p_data.get() + offset);
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

            uint64_t total_size = get_num_elements();

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
    uint64_t total_size = get_num_elements();

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

    uint64_t total_size = get_num_elements();

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

    uint64_t total_size = get_num_elements();
    if (total_size != 1)
    {
        throw std::invalid_argument(R"(Tensor(implicit type conversion):
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

template <typename float_t>
Tensor<float_t> Tensor<float_t>::clone() const
{
    if (m_dimensions.empty())
    {
        throw std::invalid_argument(R"(Tensor(clone):
            tensor has no elements.)");
    }
    Tensor<float_t> result(m_dimensions, m_mem_loc);

    uint64_t total_elements = get_num_elements();

    uint64_t rank = get_rank();

    std::vector<uint64_t> shape_strides(rank, 1);
    if (rank >= 2)
    {
        for (uint64_t i = rank - 2; i < rank; --i)
        {
            shape_strides[i] = shape_strides[i + 1] * m_dimensions[i + 1];
            if (i == 0)
            {
                break;
            }
        }
    }

    uint64_t* p_dims = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);
    uint64_t* p_src_strides = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);
    uint64_t* p_dest_strides = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);
    uint64_t* p_shape_str = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);

    if (!p_dims || !p_src_strides || !p_dest_strides || !p_shape_str)
    {
        if (p_dims)
        {
            sycl::free(p_dims, g_sycl_queue);
        }
        if (p_src_strides)
        {
            sycl::free(p_src_strides, g_sycl_queue);
        }
        if (p_dest_strides)
        {
            sycl::free(p_dest_strides, g_sycl_queue);
        }
        if (p_shape_str)
        {
            sycl::free(p_shape_str, g_sycl_queue);
        }
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy
        (p_dims, m_dimensions.data(), rank * sizeof(uint64_t)).wait();
    g_sycl_queue.memcpy
        (p_src_strides, m_strides.data(), rank * sizeof(uint64_t)).wait();
    g_sycl_queue.memcpy
        (p_dest_strides, result.m_strides.data(), rank * sizeof(uint64_t)).wait();
    g_sycl_queue.memcpy
        (p_shape_str, shape_strides.data(), rank * sizeof(uint64_t)).wait();


    const float_t* p_src_data = m_p_data.get();
    float_t* p_dest_data = result.m_p_data.get();

    g_sycl_queue.parallel_for(sycl::range<1>(total_elements),
        [=](sycl::id<1> idx)
        {
            uint64_t linear = idx[0];
            uint64_t src_offset  = 0;
            uint64_t dest_offset = 0;

            for (uint64_t i = 0; i < rank; ++i)
            {
                uint64_t coord = (linear / p_shape_str[i]) % p_dims[i];
                src_offset  += coord * p_src_strides[i];
                dest_offset += coord * p_dest_strides[i];
            }

            p_dest_data[dest_offset] = p_src_data[src_offset];
        }
    ).wait();

    sycl::free(p_dims, g_sycl_queue);
    sycl::free(p_src_strides, g_sycl_queue);
    sycl::free(p_dest_strides, g_sycl_queue);
    sycl::free(p_shape_str, g_sycl_queue);

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
        throw std::runtime_error(R"(Tensor(to):
            cannot move memory of a Tensor view (non-owning).)");
    }

    if (m_mem_loc == target_loc)
    {
        return;
    }

    uint64_t total_size = get_num_elements();

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

    if (!m_own_data)
    {
        throw std::invalid_argument(R"(Tensor(reshape):
            cannot reshape an alias/view tensor.)");
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
            throw std::invalid_argument(R"(Tensor(reshape):
                new_dimensions cannot contain zero.)");
        }
        if (new_total_size > U64_MAX / dim)
        {
            throw std::overflow_error(R"(Tensor(reshape):
                dimension product overflow.)");
        }
        new_total_size *= dim;
    }

    if (new_total_size != og_total_size)
    {
        throw std::invalid_argument(R"(Tensor(reshape):
            total number of elements must remain the same.)");
    }

    m_dimensions = new_dimensions;
    compute_strides();
}

template<typename float_t>
void Tensor<float_t>::sort(int64_t axis)
{
    if (m_dimensions.empty())
    {
        // Return immediately if tensor has no dimensions.
        return;
    }

    // Compute rank and validate axis.
    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());
    if (axis != -1 && (axis < 0 ||
        static_cast<uint64_t>(axis) >= rank))
    {
        throw std::invalid_argument("Tensor(sort): axis out of bounds");
    }

    uint64_t total_size = get_num_elements();

    // Compute effective axis size, slice count and axis stride.
    uint64_t effective_axis_size = 0;
    uint64_t slice_count = 1;
    uint64_t axis_stride = 1;
    if (axis == -1)
    {
        effective_axis_size = total_size;
        slice_count = 1;
        axis_stride = 1;
    }
    else
    {
        const uint64_t u_axis = static_cast<uint64_t>(axis);
        slice_count = 1;
        for (uint64_t i = 0; i < rank; ++i)
        {
            if (i == u_axis)
            {
                effective_axis_size = m_dimensions[i];
                axis_stride = m_strides[i];
            }
            else
            {
                slice_count *= m_dimensions[i];
            }
        }
    }

    // Compute row-major divisors for mapping linear->coords.
    std::vector<uint64_t> divisors(static_cast<size_t>(rank), 1);
    if (rank >= 2)
    {
        for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i)
        {
            divisors[static_cast<size_t>(i)] =
                divisors[static_cast<size_t>(i + 1)] *
                m_dimensions[static_cast<size_t>(i + 1)];
        }
    }

    // Query device and derive a safe workgroup size cap.
    auto dev = g_sycl_queue.get_device();
    size_t device_max_wg =
        dev.get_info<sycl::info::device::max_work_group_size>();
    size_t preferred_cap = 256;
    size_t wg_cap = std::min(device_max_wg, preferred_cap);
    if (wg_cap == 0)
    {
        wg_cap = 1;
    }

    // Build per-slice base offsets on host for axis != -1.
    std::vector<uint64_t> host_slice_base(static_cast<size_t>(slice_count), 0);
    if (axis != -1)
    {
        std::vector<uint64_t> dims;
        std::vector<uint64_t> strides_for_dims;
        const uint64_t u_axis = static_cast<uint64_t>(axis);
        for (uint64_t i = 0; i < rank; ++i)
        {
            if (i == u_axis) continue;
            dims.push_back(m_dimensions[i]);
            strides_for_dims.push_back(m_strides[i]);
        }
        const uint64_t D = static_cast<uint64_t>(dims.size());

        std::vector<uint64_t> index_factors(static_cast<size_t>(D), 1);
        if (D >= 2)
        {
            for (int64_t j = static_cast<int64_t>(D) - 2; j >= 0; --j)
            {
                index_factors[static_cast<size_t>(j)] =
                    index_factors[static_cast<size_t>(j + 1)] *
                    dims[static_cast<size_t>(j + 1)];
            }
        }

        for (uint64_t s = 0; s < slice_count; ++s)
        {
            uint64_t base = 0;
            uint64_t rem = s;
            for (uint64_t j = 0; j < D; ++j)
            {
                uint64_t coord = rem / index_factors[j];
                rem = rem % index_factors[j];
                base += coord * strides_for_dims[j];
            }
            host_slice_base[s] = base;
        }
    }

    uint64_t* p_slice_base = static_cast<uint64_t*>(
        sycl::malloc_device(static_cast<size_t>(slice_count) *
                            sizeof(uint64_t), g_sycl_queue));
    if (!p_slice_base)
    {
        throw std::bad_alloc();
    }
    g_sycl_queue.memcpy(p_slice_base, host_slice_base.data(),
                        static_cast<size_t>(slice_count) *
                        sizeof(uint64_t)).wait();

    uint64_t* p_divisors = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    if (!p_divisors || !p_strides)
    {
        if (p_divisors)
        {
            sycl::free(p_divisors, g_sycl_queue);
        }
        if (p_strides)
        {
            sycl::free(p_strides, g_sycl_queue);
        }
        sycl::free(p_slice_base, g_sycl_queue);
        throw std::bad_alloc();
    }
    g_sycl_queue.memcpy(p_divisors, divisors.data(),
                        sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_strides, m_strides.data(),
                        sizeof(uint64_t) * rank).wait();

    float_t* tensor_data = m_p_data.get();
    float_t* merge_buffer = nullptr;
    if (m_mem_loc == MemoryLocation::DEVICE)
    {
        merge_buffer = static_cast<float_t*>(
            sycl::malloc_device(static_cast<size_t>(total_size) *
                                sizeof(float_t), g_sycl_queue));
    }
    else
    {
        merge_buffer = static_cast<float_t*>(
            sycl::malloc_shared(static_cast<size_t>(total_size) *
                                sizeof(float_t), g_sycl_queue));
    }
    if (!merge_buffer)
    {
        sycl::free(p_divisors, g_sycl_queue);
        sycl::free(p_strides, g_sycl_queue);
        sycl::free(p_slice_base, g_sycl_queue);
        throw std::bad_alloc();
    }

    float_t* merge_input = tensor_data;
    float_t* merge_output = merge_buffer;

    // Determine a power-of-two workgroup size within cap.
    size_t axis_size_for_wg = 0;
    if (effective_axis_size == 0)
    {
        axis_size_for_wg = 1;
    }
    else
    {
        axis_size_for_wg = static_cast<size_t>(effective_axis_size);
    }

    size_t max_allowed = std::min(wg_cap, axis_size_for_wg);
    size_t workgroup_size = 1;
    while (workgroup_size * 2 <= max_allowed)
    {
        workgroup_size *= 2;
    }

    // Iterative bottom-up merge passes.
    for (uint64_t width = 1; width < effective_axis_size; width *= 2)
    {
        const uint64_t chunks_per_slice =
            (effective_axis_size + 2 * width - 1) / (2 * width);
        const uint64_t total_merges =
            static_cast<uint64_t>(slice_count) * chunks_per_slice;
        if (total_merges == 0)
        {
            break;
        }

        size_t local_size = workgroup_size;
        size_t global_items = static_cast<size_t>(total_merges) * local_size;

        g_sycl_queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global_items),
                                  sycl::range<1>(local_size)),
                [=](sycl::nd_item<1> it)
            {
                const uint64_t group_id = it.get_group(0);
                const uint64_t local_id = it.get_local_id(0);
                const uint64_t local_range = it.get_local_range(0);

                const uint64_t slice_idx = group_id / chunks_per_slice;
                const uint64_t chunk_idx = group_id % chunks_per_slice;
                const uint64_t slice_base = p_slice_base[slice_idx];

                const uint64_t left = chunk_idx * 2 * width;
                if (left >= effective_axis_size)
                {
                    return;
                }
                uint64_t mid = left + width;
                if (mid > effective_axis_size)
                {
                    mid = effective_axis_size;
                }
                uint64_t right = left + 2 * width;
                if (right > effective_axis_size)
                {
                    right = effective_axis_size;
                }

                const uint64_t len_left = mid - left;
                const uint64_t len_right = right - mid;
                const uint64_t total_len = len_left + len_right;

                const uint64_t merge_start =
                    (total_len * local_id) / local_range;
                const uint64_t merge_end =
                    (total_len * (local_id + 1)) / local_range;
                if (merge_start >= merge_end) return;

                // Lambda: map logical index to physical offset.
                auto idx_of = [&](uint64_t logical_idx) -> uint64_t
                {
                    if (axis == -1)
                    {
                        uint64_t rem = logical_idx;
                        uint64_t off = 0;
                        for (uint64_t d = 0; d < rank; ++d)
                        {
                            const uint64_t div = p_divisors[d];
                            const uint64_t coord = rem / div;
                            rem = rem % div;
                            off += coord * p_strides[d];
                        }
                        return off;
                    }
                    else
                    {
                        return slice_base + logical_idx * axis_stride;
                    }
                };

                // Lambda: find partition for merge-path binary search.
                auto find_partition = [&](uint64_t k) -> uint64_t
                {
                    uint64_t i_min;
                    if (k > len_right) i_min = k - len_right;
                    else i_min = 0;
                    uint64_t i_max;
                    if (k < len_left) i_max = k;
                    else i_max = len_left;
                    while (i_min < i_max)
                    {
                        uint64_t i_mid = (i_min + i_max) / 2;
                        uint64_t j_mid = k - i_mid;
                        uint64_t a = idx_of(left + i_mid);
                        uint64_t b = idx_of(mid + j_mid - 1);
                        bool cmp;
                        if (!sycl::isnan(merge_input[a]) &&
                            sycl::isnan(merge_input[b]))
                        {
                            cmp = true;
                        }
                        else if (merge_input[a] < merge_input[b])
                        {
                            cmp = true;
                        }
                        else
                        {
                            cmp = false;
                        }
                        if (cmp)
                        {
                            i_min = i_mid + 1;
                        }
                        else
                        {
                            i_max = i_mid;
                        }
                    }
                    return i_min;
                };

                const uint64_t i_start = find_partition(merge_start);
                const uint64_t j_start = merge_start - i_start;
                const uint64_t i_end = find_partition(merge_end);
                const uint64_t j_end = merge_end - i_end;

                uint64_t i = i_start;
                uint64_t j = j_start;
                uint64_t out_k = merge_start;
                while (i < i_end && j < j_end)
                {
                    uint64_t a = idx_of(left + i);
                    uint64_t b = idx_of(mid + j);
                    float_t va = merge_input[a];
                    float_t vb = merge_input[b];
                    uint64_t out = idx_of(left + out_k);
                    if ((!sycl::isnan(va) && sycl::isnan(vb)) || va < vb)
                    {
                        merge_output[out] = va;
                        ++i;
                    }
                    else
                    {
                        merge_output[out] = vb;
                        ++j;
                    }
                    ++out_k;
                }
                while (i < i_end)
                {
                    merge_output[idx_of(left + out_k)] =
                        merge_input[idx_of(left + i)];
                    ++i;
                    ++out_k;
                }
                while (j < j_end)
                {
                    merge_output[idx_of(left + out_k)] =
                        merge_input[idx_of(mid + j)];
                    ++j;
                    ++out_k;
                }
            });
        }).wait();

        // Swap buffers for next merge pass.
        float_t* tmp = merge_input;
        merge_input = merge_output;
        merge_output = tmp;
    }

    // Copy back if final result is in temporary buffer.
    if (merge_input != tensor_data)
    {
        g_sycl_queue.memcpy(tensor_data, merge_input,
            static_cast<size_t>(total_size) * sizeof(float_t)).wait();
    }

    // Free device temporaries.
    sycl::free(p_slice_base, g_sycl_queue);
    sycl::free(merge_buffer, g_sycl_queue);
    sycl::free(p_divisors, g_sycl_queue);
    sycl::free(p_strides, g_sycl_queue);
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::sum(int64_t axis) const
{
    if (m_dimensions.empty())
    {
        // Return scalar tensor if there are no dimensions.
        const MemoryLocation res_loc = m_mem_loc;
        return Tensor<float_t>({1}, res_loc);
    }

    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());

    uint64_t total_size = get_num_elements();

    const MemoryLocation res_loc = m_mem_loc;

    // Compute linear-index divisors for row-major coordinate recovery.
    std::vector<uint64_t> divisors(static_cast<size_t>(rank), 1);
    if (rank >= 2)
    {
        for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i)
        {
            divisors[static_cast<size_t>(i)] =
                divisors[static_cast<size_t>(i + 1)] *
                m_dimensions[static_cast<size_t>(i + 1)];
        }
    }

    // Query device and derive a safe workgroup size cap.
    auto dev = g_sycl_queue.get_device();
    size_t device_max_wg =
        dev.get_info<sycl::info::device::max_work_group_size>();
    size_t preferred_cap = 256;
    size_t wg_cap = std::min(device_max_wg, preferred_cap);
    if (wg_cap == 0) wg_cap = 1;

    if (axis != -1 && (axis < 0 || static_cast<uint64_t>(axis) >= rank))
    {
        throw std::invalid_argument("Tensor(sum): axis is out of bounds.");
    }

    uint64_t ax = 0;
    if (axis != -1)
    {
        ax = static_cast<uint64_t>(axis);
    }

    uint64_t axis_size = 0;
    if (axis != -1)
    {
        axis_size = m_dimensions[ax];
    }

    // Build output dimensions with the reduced axis set to 1.
    std::vector<uint64_t> new_dimensions(m_dimensions);
    if (axis != -1)
    {
        new_dimensions[ax] = 1;
    }

    uint64_t output_size = 1;
    for (uint64_t d : new_dimensions)
    {
        output_size *= d;
    }

    // Effective sizes to unify axis == -1 and axis != -1 cases.
    uint64_t effective_axis_size = 0;
    if (axis == -1)
    {
        effective_axis_size = total_size;
    }
    else
    {
        effective_axis_size = axis_size;
    }

    uint64_t effective_output_size = 0;
    if (axis == -1)
    {
        effective_output_size = 1;
    }
    else
    {
        effective_output_size = output_size;
    }

    Tensor<float_t> result;
    if (axis == -1)
    {
        std::vector<uint64_t> one_dim;
        one_dim.push_back(1);
        result = Tensor<float_t>(one_dim, res_loc);
    }
    else
    {
        result = Tensor<float_t>(new_dimensions, res_loc);
    }

    float_t* p_out = result.m_p_data.get();
    const float_t* p_src = m_p_data.get();

    uint64_t* p_strides_dev = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    g_sycl_queue.memcpy(p_strides_dev, m_strides.data(),
                        sizeof(uint64_t) * rank).wait();

    uint64_t* p_divisors_dev = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    g_sycl_queue.memcpy(p_divisors_dev, divisors.data(),
                        sizeof(uint64_t) * rank).wait();

    std::vector<uint64_t> fixed_dims;
    if (axis != -1)
    {
        fixed_dims.reserve(rank - 1);
    }
    for (uint64_t i = 0; i < rank; ++i)
    {
        if (axis != -1 && i == ax)
        {
            continue;
        }
        fixed_dims.push_back(m_dimensions[i]);
    }
    const uint64_t fixed_count = fixed_dims.size();

    // Compute divisors for mapping slice index to fixed coordinates.
    std::vector<uint64_t> fixed_divisors(static_cast<size_t>(fixed_count), 1);
    if (fixed_count >= 2)
    {
        for (int64_t j = static_cast<int64_t>(fixed_count) - 2; j >= 0; --j)
        {
            fixed_divisors[static_cast<size_t>(j)] =
                fixed_divisors[static_cast<size_t>(j + 1)] *
                fixed_dims[static_cast<size_t>(j + 1)];
        }
    }

    uint64_t* p_fixed_divs_dev = nullptr;
    if (fixed_count > 0)
    {
        p_fixed_divs_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * fixed_count, g_sycl_queue));
        g_sycl_queue.memcpy(p_fixed_divs_dev, fixed_divisors.data(),
                            sizeof(uint64_t) * fixed_count).wait();
    }

    size_t axis_size_for_wg = 0;
    if (effective_axis_size == 0)
    {
        axis_size_for_wg = 1;
    }
    else
    {
        axis_size_for_wg = static_cast<size_t>(effective_axis_size);
    }

    size_t max_allowed = std::min(wg_cap, axis_size_for_wg);
    size_t workgroup_size = 1;
    while (workgroup_size * 2 <= max_allowed)
    {
        workgroup_size *= 2;
    }

    // Number of groups per output slice (ceil division).
    size_t num_groups_per_slice = 0;
    {
        size_t eff_axis = static_cast<size_t>(effective_axis_size);
        if (eff_axis == 0)
        {
            eff_axis = 1;
        }
        num_groups_per_slice = (eff_axis + workgroup_size - 1) / workgroup_size;
    }
    if (num_groups_per_slice == 0)
    {
        num_groups_per_slice = 1;
    }

    size_t total_groups = static_cast<size_t>(effective_output_size) *
                          num_groups_per_slice;
    size_t total_group_items = total_groups * workgroup_size;
    if (total_group_items == 0) total_group_items = workgroup_size;

    // Allocate and clear partials on device as contiguous per-slice blocks.
    size_t partial_count = static_cast<size_t>(effective_output_size) *
                           num_groups_per_slice;
    size_t alloc_partial_count = partial_count;
    if (alloc_partial_count == 0)
    {
        alloc_partial_count = 1;
    }

    float_t* p_partials = static_cast<float_t*>(
        sycl::malloc_device(sizeof(float_t) * alloc_partial_count,
                            g_sycl_queue));
    g_sycl_queue.memset(p_partials, 0,
                        sizeof(float_t) * alloc_partial_count).wait();

    /*
      Shared error flag:
      0 = OK,
      1 = NaN in inputs,
      2 = non-finite result
    */
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    // First kernel: compute partials for every slice.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(total_group_items),
                sycl::range<1>(workgroup_size)),
            [=](sycl::nd_item<1> it)
        {
            const size_t global_group_id = it.get_group(0);
            const size_t slice = global_group_id / num_groups_per_slice;
            const size_t group_in_slice =
                global_group_id % num_groups_per_slice;
            const size_t local_id = it.get_local_id(0);

            float_t local_sum = float_t{};

            if (axis == -1)
            {
                size_t start = group_in_slice * workgroup_size + local_id;
                size_t stride = workgroup_size * num_groups_per_slice;
                const size_t N = static_cast<size_t>(effective_axis_size);
                for (size_t linear = start; linear < N; linear += stride)
                {
                    uint64_t remainder = static_cast<uint64_t>(linear);
                    uint64_t offset = 0;
                    for (uint64_t d = 0; d < rank; ++d)
                    {
                        uint64_t div = p_divisors_dev[d];
                        uint64_t coord = remainder / div;
                        remainder = remainder % div;
                        offset += coord * p_strides_dev[d];
                    }
                    float_t v = p_src[offset];
                    if (std::isnan(v))
                    {
                        auto atomic_err = sycl::atomic_ref<int32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                                (*(p_error_flag));
                        int32_t expected = 0;
                        atomic_err.compare_exchange_strong(expected, 1);
                        if (std::numeric_limits<float_t>::has_quiet_NaN)
                        {
                            v = std::numeric_limits<float_t>::quiet_NaN();
                        }
                    }
                    local_sum += v;
                }
            }
            else
            {
                // Axis-specific mapping from slice index to base offset.
                uint64_t remaining = slice;
                uint64_t base_offset = 0;
                uint64_t counter = 0;
                for (uint64_t i = 0; i < rank; ++i)
                {
                    if (i == ax)
                    {
                        continue;
                    }
                    uint64_t div = 1;
                    if (p_fixed_divs_dev)
                    {
                        div = p_fixed_divs_dev[counter];
                    }
                    uint64_t idx = remaining / div;
                    remaining = remaining % div;
                    base_offset += idx * p_strides_dev[i];
                    ++counter;
                }

                size_t start = group_in_slice * workgroup_size + local_id;
                size_t stride = workgroup_size * num_groups_per_slice;
                for (size_t j = start;
                     j < static_cast<size_t>(effective_axis_size); j += stride)
                {
                    uint64_t offs = base_offset +
                        static_cast<uint64_t>(j) * p_strides_dev[ax];
                    float_t v = p_src[offs];
                    if (std::isnan(v))
                    {
                        auto atomic_err = sycl::atomic_ref<int32_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                                (*(p_error_flag));
                        int32_t expected = 0;
                        atomic_err.compare_exchange_strong(expected, 1);
                        if (std::numeric_limits<float_t>::has_quiet_NaN)
                        {
                            v = std::numeric_limits<float_t>::quiet_NaN();
                        }
                    }
                    local_sum += v;
                }
            }

            // Intra-group reduction to produce the partial sum.
            auto group = it.get_group();
            float_t group_sum =
                sycl::reduce_over_group(group, local_sum, sycl::plus<float_t>());

            if (!std::isfinite(group_sum))
            {
                auto atomic_err = sycl::atomic_ref<int32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(*(p_error_flag));
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 2);
            }

            if (local_id == 0)
            {
                size_t write_idx = slice * num_groups_per_slice + group_in_slice;
                p_partials[write_idx] = group_sum;
            }
        });
    }).wait();

    // Second kernel: reduce partials per slice to final outputs.
    size_t wg2 = 1;
    while (wg2 * 2 <= std::min(workgroup_size,
                                std::max<size_t>(1, num_groups_per_slice)))
    {
        wg2 = wg2 * 2;
    }

    size_t second_ndrange = static_cast<size_t>(effective_output_size) * wg2;
    if (second_ndrange == 0)
    {
        second_ndrange = wg2;
    }

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(second_ndrange),
                sycl::range<1>(wg2)),
            [=](sycl::nd_item<1> it)
        {
            const size_t slice = it.get_group(0);
            const size_t lid = it.get_local_id(0);
            const size_t local_range = it.get_local_range(0);

            // Each thread sums a strided subset of the partials.
            float_t v = float_t{};
            for (size_t idx = lid; idx < num_groups_per_slice; idx += local_range)
            {
                float_t pv = p_partials[slice * num_groups_per_slice + idx];
                if (std::isnan(pv))
                {
                    auto atomic_err = sycl::atomic_ref<int32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                            (*(p_error_flag));
                    int32_t expected = 0;
                    atomic_err.compare_exchange_strong(expected, 1);
                    if (std::numeric_limits<float_t>::has_quiet_NaN)
                    {
                        pv = std::numeric_limits<float_t>::quiet_NaN();
                    }
                }
                v += pv;
            }

            auto group = it.get_group();
            float_t total =
                sycl::reduce_over_group(group, v, sycl::plus<float_t>());

            // detect non-finite total
            if (!std::isfinite(total))
            {
                auto atomic_err = sycl::atomic_ref<int32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(*(p_error_flag));
                int32_t expected = 0;
                atomic_err.compare_exchange_strong(expected, 2);
            }

            if (lid == 0)
            {
                p_out[slice] = total;
            }
        });
    }).wait();

    sycl::free(p_partials, g_sycl_queue);
    sycl::free(p_divisors_dev, g_sycl_queue);
    sycl::free(p_strides_dev, g_sycl_queue);
    if (p_fixed_divs_dev)
    {
        sycl::free(p_fixed_divs_dev, g_sycl_queue);
    }

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(Tensor(sum):
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(Tensor(sum):
                non-finite result detected.)");
        }
        throw std::runtime_error(R"(Tensor(sum):
            numeric error during sum.)");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::cumsum(int64_t axis) const
{
    if (m_dimensions.empty())
    {
        const MemoryLocation res_loc = m_mem_loc;
        return Tensor<float_t>({1}, res_loc);
    }

    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());

    uint64_t total_size = get_num_elements();

    if (axis != -1 && (axis < 0 || static_cast<uint64_t>(axis) >= rank))
    {
        throw std::invalid_argument("Tensor(cumsum): axis is out of bounds.");
    }

    uint64_t ax = 0;
    uint64_t axis_size = 0;
    if (axis == -1)
    {
        axis_size = total_size;
        ax = 0;
    }
    else
    {
        ax = static_cast<uint64_t>(axis);
        axis_size = m_dimensions[ax];
    }

    std::vector<uint64_t> out_dims;

    if (axis == -1)
    {
        out_dims = std::vector<uint64_t>{ total_size };
    }
    else
    {
        out_dims = m_dimensions;
    }

    Tensor<float_t> result(out_dims, m_mem_loc);

    const float_t* p_src = m_p_data.get();
    float_t* p_out = result.m_p_data.get();

    std::vector<uint64_t> divisors(rank, 1);
    if (rank >= 2)
    {
        for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i)
        {
            divisors[static_cast<size_t>(i)] =
                divisors[static_cast<size_t>(i + 1)] *
                m_dimensions[static_cast<size_t>(i + 1)];
        }
    }

    uint64_t* p_divs_dev = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    g_sycl_queue.memcpy
        (p_divs_dev, divisors.data(), sizeof(uint64_t) * rank).wait();

    uint64_t* p_strides_dev = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    g_sycl_queue.memcpy
        (p_strides_dev, m_strides.data(), sizeof(uint64_t) * rank).wait();

    uint64_t* p_out_strides_dev = nullptr;
    if (axis != -1)
    {
        p_out_strides_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
        g_sycl_queue.memcpy(p_out_strides_dev, result.m_strides.data(),
                            sizeof(uint64_t) * rank).wait();
    }

    std::vector<uint64_t> fixed_dims;
    fixed_dims.reserve(rank - 1);

    for (uint64_t i = 0; i < rank; ++i) {
        if (axis != -1 && i == ax)
        {
            continue;
        }
        fixed_dims.push_back(m_dimensions[i]);
    }
    const uint64_t fixed_count = fixed_dims.size();

    std::vector<uint64_t> fixed_divisors(static_cast<size_t>(fixed_count), 1);
    if (fixed_count >= 2)
    {
        for (int64_t j = static_cast<int64_t>(fixed_count) - 2; j >= 0; --j)
        {
            fixed_divisors[static_cast<size_t>(j)] =
                fixed_divisors[static_cast<size_t>(j + 1)] *
                fixed_dims[static_cast<size_t>(j + 1)];
        }
    }

    uint64_t* p_fixed_divs_dev = nullptr;
    if (fixed_count > 0)
    {
        p_fixed_divs_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * fixed_count, g_sycl_queue));
        g_sycl_queue.memcpy(p_fixed_divs_dev, fixed_divisors.data(),
                            sizeof(uint64_t) * fixed_count).wait();
    }

    // Effective sizes unify axis == -1 and axis != -1 cases.
    uint64_t effective_axis_size;
    if (axis == -1)
    {
        effective_axis_size = total_size;
    }
    else
    {
        effective_axis_size = axis_size;
    }

    uint64_t effective_output_size = 0;
    if (axis == -1)
    {
        effective_output_size = 1;
    }
    else
    {
        uint64_t out_sz = 1;
        for (uint64_t i = 0; i < rank; ++i)
        {
            if (i == ax)
            {
                continue;
            }
            out_sz *= m_dimensions[i];
        }
        effective_output_size = out_sz;
    }

    /*
      Shared error flag:
      0 = OK,
      1 = NaN in inputs,
      2 = non-finite result
    */
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    *p_error_flag = 0;

    auto dev = g_sycl_queue.get_device();
    size_t device_max_wg =
        dev.get_info<sycl::info::device::max_work_group_size>();
    size_t preferred_cap = 256;
    size_t wg_cap = std::min(device_max_wg, preferred_cap);
    if (wg_cap == 0)
    {
        wg_cap = 1;
    }

    size_t workgroup_size = 1;
    while (workgroup_size * 2 <= wg_cap &&
           workgroup_size * 2 <= static_cast<size_t>(effective_axis_size))
    {
        workgroup_size *= 2;
    }

    size_t num_groups_per_slice =
        static_cast<size_t>((static_cast<size_t>(effective_axis_size)
            + workgroup_size - 1) / workgroup_size);
    if (num_groups_per_slice == 0)
    {
        num_groups_per_slice = 1;
    }

    size_t total_groups = static_cast<size_t>(effective_output_size) *
                            num_groups_per_slice;
    size_t total_group_items = total_groups * workgroup_size;
    if (total_group_items == 0)
    {
        total_group_items = workgroup_size;
    }

    // Block partials (one per (slice, group_in_slice))
    float_t* p_block_partials = static_cast<float_t*>(
        sycl::malloc_device(sizeof(float_t) * total_groups, g_sycl_queue));
    g_sycl_queue.memset
        (p_block_partials, 0, sizeof(float_t) * total_groups).wait();

    // First kernel: per-group inclusive scan + write block partial.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(total_group_items),
                              sycl::range<1>(workgroup_size)),
            [=](sycl::nd_item<1> it)
        {
            const size_t global_group_id = it.get_group(0);
            const size_t group_in_slice = global_group_id % num_groups_per_slice;
            const size_t slice = global_group_id / num_groups_per_slice;
            const size_t local_id = it.get_local_id(0);
            const size_t local_range = it.get_local_range(0);

            const size_t index_in_slice =
                group_in_slice * local_range + local_id;

            const size_t base_index = group_in_slice * local_range;
            size_t remaining = 0;
            if (effective_axis_size > base_index)
            {
                remaining = static_cast<size_t>(effective_axis_size) - base_index;
            }
            const size_t valid_count = std::min(remaining, local_range);

            bool active = false;
            if (index_in_slice < static_cast<size_t>(effective_axis_size))
            {
                active = true;
            }

            uint64_t base_src = 0, base_dst = 0;
            if (effective_output_size > 1)
            {
                uint64_t rem = slice;
                uint64_t counter = 0;
                for (uint64_t d = 0; d < rank; ++d) {
                    if (d == ax) continue;
                    uint64_t div = 1;
                    if (p_fixed_divs_dev)
                    {
                        div = p_fixed_divs_dev[counter];
                    }
                    uint64_t idx = rem / div;
                    rem = rem % div;
                    base_src += idx * p_strides_dev[d];
                    if (axis != -1)
                    {
                        base_dst += idx * p_out_strides_dev[d];
                    }
                    ++counter;
                }
            }

            // Element offsets (source and destination).
            uint64_t src_off = 0, dst_off = 0;
            if (axis == -1)
            {
                uint64_t linear = static_cast<uint64_t>(index_in_slice);
                uint64_t remainder = linear;
                for (uint64_t d = 0; d < rank; ++d)
                {
                    uint64_t div = p_divs_dev[d];
                    uint64_t coord = remainder / div;
                    remainder = remainder % div;
                    src_off += coord * p_strides_dev[d];
                }
                dst_off = static_cast<uint64_t>(index_in_slice);
            }
            else
            {
                src_off = base_src +
                    static_cast<uint64_t>(index_in_slice) * p_strides_dev[ax];
                dst_off = base_dst +
                    static_cast<uint64_t>(index_in_slice) * p_out_strides_dev[ax];
            }

            float_t x = float_t{0};
            if (active)
            {
                x = p_src[src_off];
                if (std::isnan(x))
                {
                    auto atomic_err = sycl::atomic_ref<int32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                            (*p_error_flag);
                    int32_t expected = 0;
                    atomic_err.compare_exchange_strong(expected, 1);
                    if (std::numeric_limits<float_t>::has_quiet_NaN)
                    {
                        x = std::numeric_limits<float_t>::quiet_NaN();
                    }
                }
            }

            float_t prefix = sycl::inclusive_scan_over_group
                (it.get_group(), x, sycl::plus<float_t>());

            if (active)
            {
                if (!std::isfinite(prefix))
                {
                    auto atomic_err = sycl::atomic_ref<int32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                            (*p_error_flag);
                    int32_t expected = 0;
                    atomic_err.compare_exchange_strong(expected, 2);
                }
                p_out[dst_off] = prefix;
            }

            size_t last_valid_lane = 0;
            if (valid_count > 0)
            {
                last_valid_lane = valid_count - 1;
            }
            if (valid_count > 0 && local_id == last_valid_lane)
            {
                size_t write_idx = slice * num_groups_per_slice + group_in_slice;
                p_block_partials[write_idx] = prefix;
            }
        });
    }).wait();

    // Second kernel: inclusive scan of block partials per slice.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>
            (effective_output_size)),
            [=](sycl::id<1> slice_id)
        {
            const size_t slice = slice_id[0];
            float_t running = float_t{0};
            size_t base = slice * num_groups_per_slice;
            for (size_t g = 0; g < num_groups_per_slice; ++g)
            {
                float_t v = p_block_partials[base + g];
                running += v;
                p_block_partials[base + g] = running;
            }
        });
    }).wait();

    // Third kernel: add previous-block prefix to each element (groups > 0).
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(total_group_items),
                              sycl::range<1>(workgroup_size)),
            [=](sycl::nd_item<1> it)
        {
            const size_t global_group_id = it.get_group(0);
            const size_t group_in_slice = global_group_id % num_groups_per_slice;
            const size_t slice = global_group_id / num_groups_per_slice;
            const size_t local_id = it.get_local_id(0);
            const size_t local_range = it.get_local_range(0);

            if (group_in_slice == 0)
            {
                return;
            }

            const size_t index_in_slice =
                group_in_slice * local_range + local_id;

            if (index_in_slice >= static_cast<size_t>(effective_axis_size))
            {
                return;
            }

            uint64_t base_dst = 0;
            if (effective_output_size > 1)
            {
                uint64_t rem = slice;
                uint64_t counter = 0;
                for (uint64_t d = 0; d < rank; ++d)
                {
                    if (d == ax)
                    {
                        continue;
                    }
                    uint64_t div = 1;
                    if (p_fixed_divs_dev)
                    {
                        div = p_fixed_divs_dev[counter];
                    }
                    uint64_t idx = rem / div;
                    rem = rem % div;
                    if (axis != -1)
                    {
                        base_dst += idx * p_out_strides_dev[d];
                    }
                    ++counter;
                }
            }

            uint64_t dst_off = 0;
            if (axis == -1)
            {
                dst_off = static_cast<uint64_t>(index_in_slice);
            }
            else
            {
                dst_off = base_dst +
                        static_cast<uint64_t>(index_in_slice) *
                        p_out_strides_dev[ax];
            }

            float_t add = p_block_partials
                [slice * num_groups_per_slice + (group_in_slice - 1)];

            p_out[dst_off] += add;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_block_partials, g_sycl_queue);
    sycl::free(p_divs_dev, g_sycl_queue);
    sycl::free(p_strides_dev, g_sycl_queue);
    if (p_out_strides_dev)
    {
        sycl::free(p_out_strides_dev, g_sycl_queue);
    }
    if (p_fixed_divs_dev)
    {
        sycl::free(p_fixed_divs_dev, g_sycl_queue);
    }
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(Tensor(cumsum):
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(Tensor(cumsum):
                non-finite result detected.)");
        }
        throw std::runtime_error(R"(Tensor(cumsum):
            numeric error during cumsum.)");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::transpose() const
{
    const uint64_t rank = m_dimensions.size();
    if (rank == 0)
    {
        throw std::runtime_error(R"(Tensor(transpose):
            cannot transpose an empty tensor.)");
    }

    std::vector<uint64_t> axes(rank);
    for (uint64_t i = 0; i < rank; ++i)
    {
        axes[i] = rank - 1 - i;
    }

    return transpose(axes);
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::transpose(const std::vector<uint64_t> & axes) const
{
    const uint64_t rank = m_dimensions.size();

    if (axes.size() != rank)
    {
        throw std::invalid_argument(R"(Tensor(transpose):
            axes vector must have same length as tensor rank.)"
        );
    }

    std::vector<bool> seen(rank, false);
    for (uint64_t ax : axes)
    {
        if (ax >= rank || seen[ax])
        {
            throw std::invalid_argument(R"(Tensor(transpose):
                axes must be a permutation of [0..rank-1].)"
            );
        }
        seen[ax] = true;
    }

    std::vector<uint64_t> new_dims(rank);
    std::vector<uint64_t> new_strides(rank);
    for (uint64_t i = 0; i < rank; ++i)
    {
        new_dims[i] = m_dimensions[axes[i]];
        new_strides[i] = m_strides[axes[i]];
    }

    std::vector<uint64_t> start_indices(rank, 0);

    return Tensor(*this, start_indices, new_dims, new_strides);
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

                uint64_t current_offset = offset + i * m_strides[dim];
                g_sycl_queue.memcpy(&val, m_p_data.get() + current_offset,
                    sizeof(float_t)).wait();

                os << val;
                if (i != m_dimensions[dim] - 1)
                {
                    os << ", ";
                }
            }
            os << "]";
        }
        else
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

template<typename float_t>
const float_t * Tensor<float_t>::get_data() const noexcept
{
    return m_p_data.get();
}

template<typename float_t>
const std::vector<uint64_t> & Tensor<float_t>::get_dimensions() const noexcept
{
    return m_dimensions;
}

template<typename float_t>
const std::vector<uint64_t> & Tensor<float_t>::get_strides() const noexcept
{
    return m_strides;
}

template<typename float_t>
const std::vector<uint64_t> & Tensor<float_t>::get_shape() const noexcept
{
    return m_dimensions;
}

template<typename float_t>
uint64_t Tensor<float_t>::get_rank() const noexcept
{
    return static_cast<uint64_t>(m_dimensions.size());
}

template<typename float_t>
uint64_t Tensor<float_t>::get_num_elements() const noexcept
{
    if (m_dimensions.empty())
    {
        return 0;
    }

    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        total_size *= d;
    }
    return total_size;
}

template<typename float_t>
MemoryLocation Tensor<float_t>::get_memory_location() const noexcept
{
    return m_mem_loc;
}

template<typename float_t>
bool Tensor<float_t>::get_owns_data() const noexcept
{
    return m_own_data;
}

template<typename float_t>
bool Tensor<float_t>::is_view() const noexcept
{
    return !m_own_data;
}

template<typename float_t>
uint64_t Tensor<float_t>::get_element_size_bytes() const noexcept
{
    return static_cast<uint64_t>(sizeof(float_t));
}

template<typename float_t>
uint64_t Tensor<float_t>::get_total_bytes() const noexcept
{
    const uint64_t elems = get_num_elements();
    const uint64_t elem_size = get_element_size_bytes();

    return elems * elem_size;
}

template class Tensor<float>;

} // namespace temper