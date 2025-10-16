/**
 * @file Tensor.cpp
 * @brief Tensor class function definitions.
 */

#include "temper/Tensor.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Utils.hpp"
#include "temper/Math.hpp"
#include "temper/Stats.hpp"

#include <iostream>

namespace temper
{

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
Tensor<float_t>::Tensor(const std::initializer_list<uint64_t> & dimensions,
                        MemoryLocation loc)
    : Tensor(std::vector<uint64_t>(dimensions), loc)
{
    // Constructor already delegated; no op.
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
Tensor<float_t>::Tensor(float_t val, MemoryLocation loc)
    : m_dimensions({1}),
      m_strides(1),
      m_own_data(true),
      m_mem_loc(loc)
{
    compute_strides();

    const size_t alloc_bytes = sizeof(float_t);

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
        [](float_t* p)
        {
            if (p)
            {
                sycl::free(p, g_sycl_queue);
            }
        }
    );

    if (m_mem_loc == MemoryLocation::HOST)
    {
        *raw_ptr = val;
    }
    else
    {
        g_sycl_queue.memcpy(raw_ptr, &val, sizeof(float_t)).wait();
    }
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

    // Compute offset from owner's base pointer (with basic overflow safety)
    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();
    uint64_t offset = 0;
    for (uint64_t i = 0; i < owner_rank; ++i)
    {
        uint64_t si = start_indices[i];
        uint64_t ostride = owner.m_strides[i];

        if (si != 0 && ostride != 0)
        {
            if (ostride > U64_MAX / si)
            {
                throw std::overflow_error(R"(Tensor(alias view constructor):
                    stride * start_index overflow while computing offset.)");
            }
            uint64_t add = ostride * si;
            if (offset > U64_MAX - add)
            {
                throw std::overflow_error(R"(Tensor(alias view constructor):
                    offset computation overflow.)");
            }
            offset += add;
        }
    }

    // Compute the maximum linear index that the new view may access
    // (relative to owner.m_p_data).
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

    // Compute the owner's maximum reachable linear index
    // (relative to owner.m_p_data).
    uint64_t owner_max_index = 0;
    for (uint64_t j = 0; j < owner_rank; ++j)
    {
        uint64_t odimm1 = owner.m_dimensions[j] - 1;
        uint64_t ostride = owner.m_strides[j];

        if (ostride != 0 && odimm1 > 0)
        {
            if (ostride > U64_MAX / odimm1)
            {
                throw std::overflow_error(R"(Tensor(alias view constructor):
                    stride * (dim-1) overflow while computing owner bounds.)");
            }
            uint64_t add = ostride * odimm1;
            if (owner_max_index > U64_MAX - add)
            {
                throw std::overflow_error(R"(Tensor(alias view constructor):
                    owner max index computation overflow.)");
            }
            owner_max_index += add;
        }
    }

    // The view's maximum index must be within the owner's reachable range.
    if (max_index > owner_max_index)
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

    const uint64_t rank = get_rank();
    bool dst_contig = true;
    if (rank > 0)
    {
        // Build canonical row-major strides for this shape
        std::vector<uint64_t> canon(rank);
        canon[rank - 1] = 1;
        for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i)
        {
            canon[i] = canon[i + 1] * m_dimensions[i + 1];
        }
        dst_contig = (m_strides == canon);
    }

    if (dst_contig)
    {
        const size_t alloc_bytes =
            static_cast<size_t>(total_size) * sizeof(float_t);
        g_sycl_queue.memcpy(m_p_data.get(), values.data(), alloc_bytes).wait();
        return *this;
    }

    std::vector<uint64_t> divisors(rank);
    for (uint64_t i = 0; i < rank; ++i)
    {
        uint64_t d = 1;
        for (uint64_t j = i + 1; j < rank; ++j)
        {
            d *= m_dimensions[j];
        }
        divisors[i] = d;
    }

    uint64_t* p_res_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_dst_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    float_t* p_vals_shared = static_cast<float_t*>(sycl::malloc_shared
        (sizeof(float_t) * static_cast<size_t>(total_size), g_sycl_queue));

    if (!p_res_divs || !p_dst_strides || !p_vals_shared)
    {
        if (p_res_divs) sycl::free(p_res_divs, g_sycl_queue);
        if (p_dst_strides) sycl::free(p_dst_strides, g_sycl_queue);
        if (p_vals_shared) sycl::free(p_vals_shared, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_res_divs,
        divisors.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_dst_strides,
        m_strides.data(), sizeof(uint64_t) * rank).wait();

    std::memcpy(p_vals_shared,
        values.data(), sizeof(float_t) * static_cast<size_t>(total_size));

    const float_t* p_src_flat = p_vals_shared;
    float_t* p_dst = get_data();

    g_sycl_queue.submit([&](sycl::handler & cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t off_dst = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_dst_strides, rank);

            p_dst[off_dst] = p_src_flat[flat_idx];
        });
    }).wait();

    sycl::free(p_res_divs, g_sycl_queue);
    sycl::free(p_dst_strides, g_sycl_queue);
    sycl::free(p_vals_shared, g_sycl_queue);

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
const Tensor<float_t> Tensor<float_t>::operator[](uint64_t idx) const
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

    const uint64_t rank_a = get_rank();
    const uint64_t rank_b = other.get_rank();
    const uint64_t max_rank = std::max(rank_a, rank_b);

    utils::TensorDesc a_desc{m_dimensions, m_strides, {}};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides, {}};

    utils::TensorDesc a_aligned = utils::align_tensor(a_desc, max_rank);
    utils::TensorDesc b_aligned = utils::align_tensor(b_desc, max_rank);

    utils::BroadcastResult br = utils::compute_broadcast(a_aligned, b_aligned);
    std::vector<uint64_t> out_shape = br.out.shape;
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.a_strides);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.b_strides);
    std::vector<uint64_t> res_divs = std::move(br.out.divisors);

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

    uint64_t* p_res_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    // Shared error flag
    // (0 = OK, 1 = NaN in inputs/result, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    if (!p_res_divs || !p_a_strides || !p_b_strides || !p_error_flag)
    {
        if (p_res_divs) sycl::free(p_res_divs, g_sycl_queue);
        if (p_a_strides) sycl::free(p_a_strides, g_sycl_queue);
        if (p_b_strides) sycl::free(p_b_strides, g_sycl_queue);
        if (p_error_flag) sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    *p_error_flag = 0;
    g_sycl_queue.memcpy(p_res_divs,
        res_divs.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    const float_t* p_a_data = get_data();
    const float_t* p_b_data = other.get_data();
    float_t* p_r_data = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t offset_a = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_a_strides, max_rank);

            uint64_t offset_b = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_b_strides, max_rank);

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            float_t res = a_val + b_val;

            sycl_utils::device_check_nan_and_set(a_val, p_error_flag);
            sycl_utils::device_check_nan_and_set(b_val, p_error_flag);
            sycl_utils::device_check_nan_and_set(res, p_error_flag);

            sycl_utils::device_check_finite_and_set(res, p_error_flag);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_res_divs, g_sycl_queue);
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

    const uint64_t rank_a = get_rank();
    const uint64_t rank_b = other.get_rank();
    const uint64_t max_rank = std::max(rank_a, rank_b);

    utils::TensorDesc a_desc{m_dimensions, m_strides, {}};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides, {}};

    utils::TensorDesc a_aligned = utils::align_tensor(a_desc, max_rank);
    utils::TensorDesc b_aligned = utils::align_tensor(b_desc, max_rank);

    utils::BroadcastResult br = utils::compute_broadcast(a_aligned, b_aligned);
    std::vector<uint64_t> out_shape = br.out.shape;
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.a_strides);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.b_strides);
    std::vector<uint64_t> res_divs = std::move(br.out.divisors);

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

    uint64_t* p_res_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    // Shared error flag
    // (0 = OK, 1 = NaN in inputs/result, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    if (!p_res_divs || !p_a_strides || !p_b_strides || !p_error_flag)
    {
        if (p_res_divs) sycl::free(p_res_divs, g_sycl_queue);
        if (p_a_strides) sycl::free(p_a_strides, g_sycl_queue);
        if (p_b_strides) sycl::free(p_b_strides, g_sycl_queue);
        if (p_error_flag) sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    *p_error_flag = 0;
    g_sycl_queue.memcpy(p_res_divs,
        res_divs.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    const float_t* p_a_data = get_data();
    const float_t* p_b_data = other.get_data();
    float_t* p_r_data = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t offset_a = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_a_strides, max_rank);

            uint64_t offset_b = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_b_strides, max_rank);

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            float_t res = a_val - b_val;

            sycl_utils::device_check_nan_and_set(a_val, p_error_flag);
            sycl_utils::device_check_nan_and_set(b_val, p_error_flag);
            sycl_utils::device_check_nan_and_set(res, p_error_flag);

            sycl_utils::device_check_finite_and_set(res, p_error_flag);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_res_divs, g_sycl_queue);
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

    const uint64_t rank_a = get_rank();
    const uint64_t rank_b = other.get_rank();
    const uint64_t max_rank = std::max(rank_a, rank_b);

    utils::TensorDesc a_desc{m_dimensions, m_strides, {}};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides, {}};

    utils::TensorDesc a_aligned = utils::align_tensor(a_desc, max_rank);
    utils::TensorDesc b_aligned = utils::align_tensor(b_desc, max_rank);

    utils::BroadcastResult br = utils::compute_broadcast(a_aligned, b_aligned);
    std::vector<uint64_t> out_shape = br.out.shape;
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.a_strides);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.b_strides);
    std::vector<uint64_t> res_divs = std::move(br.out.divisors);

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

    uint64_t* p_res_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    // Shared error flag
    // (0 = OK, 1 = NaN in inputs/result, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    if (!p_res_divs || !p_a_strides || !p_b_strides || !p_error_flag)
    {
        if (p_res_divs) sycl::free(p_res_divs, g_sycl_queue);
        if (p_a_strides) sycl::free(p_a_strides, g_sycl_queue);
        if (p_b_strides) sycl::free(p_b_strides, g_sycl_queue);
        if (p_error_flag) sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    *p_error_flag = 0;
    g_sycl_queue.memcpy(p_res_divs,
        res_divs.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    const float_t* p_a_data = get_data();
    const float_t* p_b_data = other.get_data();
    float_t* p_r_data = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t offset_a = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_a_strides, max_rank);

            uint64_t offset_b = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_b_strides, max_rank);

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            float_t res = a_val * b_val;

            sycl_utils::device_check_nan_and_set(a_val, p_error_flag);
            sycl_utils::device_check_nan_and_set(b_val, p_error_flag);
            sycl_utils::device_check_nan_and_set(res, p_error_flag);

            sycl_utils::device_check_finite_and_set(res, p_error_flag);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_res_divs, g_sycl_queue);
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

    const uint64_t rank_a = get_rank();
    const uint64_t rank_b = other.get_rank();
    const uint64_t max_rank = std::max(rank_a, rank_b);

    utils::TensorDesc a_desc{m_dimensions, m_strides, {}};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides, {}};

    utils::TensorDesc a_aligned = temper::utils::align_tensor(a_desc, max_rank);
    utils::TensorDesc b_aligned = temper::utils::align_tensor(b_desc, max_rank);

    utils::BroadcastResult br = utils::compute_broadcast(a_aligned, b_aligned);

    std::vector<uint64_t> out_shape = br.out.shape;
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.a_strides);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.b_strides);
    std::vector<uint64_t> res_divs = std::move(br.out.divisors);

    uint64_t total_size = 1;
    for (uint64_t d : out_shape)
    {
        total_size *= d;
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

    uint64_t* p_res_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    // Shared error flag
    // (0 = OK, 1 = NaN in inputs/result, 2 = non-finite result).
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    if (!p_res_divs || !p_a_strides || !p_b_strides || !p_error_flag)
    {
        if (p_res_divs) sycl::free(p_res_divs, g_sycl_queue);
        if (p_a_strides) sycl::free(p_a_strides, g_sycl_queue);
        if (p_b_strides) sycl::free(p_b_strides, g_sycl_queue);
        if (p_error_flag) sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    *p_error_flag = 0;
    g_sycl_queue.memcpy(p_res_divs,
        res_divs.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_a_strides,
        a_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_b_strides,
        b_strides_broadcasted.data(), sizeof(uint64_t) * max_rank).wait();

    const float_t* p_a_data = get_data();
    const float_t* p_b_data = other.get_data();
    float_t* p_r_data = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t offset_a = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_a_strides, max_rank);
            uint64_t offset_b = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_b_strides, max_rank);

            float_t a_val = p_a_data[offset_a];
            float_t b_val = p_b_data[offset_b];

            sycl_utils::device_check_nan_and_set(a_val, p_error_flag);
            sycl_utils::device_check_nan_and_set(b_val, p_error_flag);

            sycl_utils::device_check_divzero_and_set(b_val, p_error_flag);

            float_t res = a_val / b_val;
            sycl_utils::device_check_finite_and_set(res, p_error_flag);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);

    sycl::free(p_res_divs, g_sycl_queue);
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
                non-finite result detected.)");
        }
        if (err == 3)
        {
            throw std::runtime_error(R"(Tensor(operator/):
                division by zero detected.)");
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

    const uint64_t rank = get_rank();
    uint64_t total_size = get_num_elements();

    MemoryLocation res_loc = m_mem_loc;
    Tensor result(m_dimensions, res_loc);

    std::vector<uint64_t> divisors = utils::compute_divisors(m_dimensions);

    uint64_t* p_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    if (!p_divs || !p_strides || !p_error_flag)
    {
        if (p_divs) sycl::free(p_divs, g_sycl_queue);
        if (p_strides) sycl::free(p_strides, g_sycl_queue);
        if (p_error_flag) sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    *p_error_flag = 0;

    g_sycl_queue.memcpy(
        p_divs, divisors.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(
    p_strides, m_strides.data(), sizeof(uint64_t) * rank).wait();

    const float_t* p_src = get_data();
    float_t* p_dst = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t off = sycl_utils::idx_of(flat_idx, p_divs, p_strides, rank);
            float_t val = p_src[off];

            sycl_utils::device_check_nan_and_set(val, p_error_flag);

            p_dst[flat_idx] = -val;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_divs, g_sycl_queue);
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

    std::vector<uint64_t> shape_divs = utils::compute_divisors(m_dimensions);

    uint64_t* p_dims = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);
    uint64_t* p_src_strides = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);
    uint64_t* p_dest_strides = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);
    uint64_t* p_shape_divs = sycl::malloc_device<uint64_t>(rank, g_sycl_queue);

    if (!p_dims || !p_src_strides || !p_dest_strides || !p_shape_divs)
    {
        if (p_dims) sycl::free(p_dims, g_sycl_queue);
        if (p_src_strides) sycl::free(p_src_strides, g_sycl_queue);
        if (p_dest_strides) sycl::free(p_dest_strides, g_sycl_queue);
        if (p_shape_divs) sycl::free(p_shape_divs, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy
        (p_dims, m_dimensions.data(), rank * sizeof(uint64_t)).wait();
    g_sycl_queue.memcpy
        (p_src_strides,m_strides.data(), rank * sizeof(uint64_t)).wait();
    g_sycl_queue.memcpy(p_dest_strides,
        result.m_strides.data(), rank * sizeof(uint64_t)).wait();
    g_sycl_queue.memcpy
        (p_shape_divs,shape_divs.data(), rank * sizeof(uint64_t)).wait();

    const float_t* p_src_data = get_data();
    float_t* p_dest_data = result.get_data();

    g_sycl_queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(total_elements)),
        [=](sycl::id<1> idx)
    {
        uint64_t linear = static_cast<uint64_t>(idx[0]);
        uint64_t src_offset = sycl_utils::idx_of(
            linear, p_shape_divs, p_src_strides, rank);
        uint64_t dest_offset = sycl_utils::idx_of(
            linear, p_shape_divs, p_dest_strides, rank);
        p_dest_data[dest_offset] = p_src_data[src_offset];
    }).wait();

    sycl::free(p_dims, g_sycl_queue);
    sycl::free(p_src_strides, g_sycl_queue);
    sycl::free(p_dest_strides, g_sycl_queue);
    sycl::free(p_shape_divs, g_sycl_queue);

    return result;
}

template<typename float_t>
void Tensor<float_t>::copy_from(const Tensor & src)
{
    if (!m_p_data)
    {
        throw std::runtime_error(R"(Tensor(copy_from):
            target tensor has no storage (uninitialized view?).)");
    }

    if (!src.m_p_data)
    {
        throw std::runtime_error(R"(Tensor(copy_from):
            source tensor has no storage.)");
    }

    const uint64_t total_dst = get_num_elements();
    if (total_dst == 0)
    {
        throw std::invalid_argument(R"(Tensor(copy_from):
            target has zero elements.)");
    }

    const uint64_t rank_dst = get_rank();
    const uint64_t rank_src = src.get_rank();

    if (rank_src > rank_dst)
    {
        throw std::invalid_argument(R"(Tensor(copy_from):
            source rank > destination rank.)");
    }

    bool same_shape = (m_dimensions == src.m_dimensions);
    if (same_shape)
    {
        std::vector<uint64_t> canon(rank_dst);
        if (rank_dst > 0)
        {
            canon[rank_dst - 1] = 1;
            for (int64_t i = static_cast<int64_t>(rank_dst) - 2; i >= 0; --i)
            {
                canon[i] = canon[i + 1] * m_dimensions[i + 1];
            }
        }

        bool dst_contig = (m_strides == canon);
        bool src_contig = (src.m_strides == canon);

        if (dst_contig && src_contig)
        {
            const size_t bytes =
                static_cast<size_t>(total_dst) * sizeof(float_t);
            g_sycl_queue.memcpy(get_data(), src.get_data(), bytes).wait();
            return;
        }
    }

    const uint64_t pad = rank_dst - rank_src;
    const uint64_t max_rank = rank_dst;

    std::vector<uint64_t> src_aligned_dims(max_rank);
    std::vector<uint64_t> src_aligned_strides(max_rank);

    for (uint64_t i = 0; i < pad; ++i)
    {
        src_aligned_dims[i] = 1;
        src_aligned_strides[i] = 0;
    }

    for (uint64_t i = 0; i < rank_src; ++i)
    {
        src_aligned_dims[pad + i] = src.m_dimensions[i];
        src_aligned_strides[pad + i] = src.m_strides[i];
    }

    std::vector<uint64_t> dst_aligned_dims = m_dimensions;
    std::vector<uint64_t> dst_aligned_strides = m_strides;

    utils::TensorDesc a_desc{src_aligned_dims, src_aligned_strides, {}};
    utils::TensorDesc b_desc{dst_aligned_dims, dst_aligned_strides, {}};

    utils::BroadcastResult br = utils::compute_broadcast(a_desc, b_desc);

    if (br.out.shape != dst_aligned_dims)
    {
        throw std::invalid_argument(R"(Tensor(copy_from):
            source cannot be broadcast to destination shape.)");
    }

    // Prepare device arrays for divisors and strides (length = max_rank)
    uint64_t* p_res_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_a_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));
    uint64_t* p_b_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * max_rank, g_sycl_queue));

    if (!p_res_divs || !p_a_strides || !p_b_strides)
    {
        sycl::free(p_res_divs, g_sycl_queue);
        sycl::free(p_a_strides, g_sycl_queue);
        sycl::free(p_b_strides, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_res_divs, br.out.divisors.data(),
                        sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_a_strides, br.a_strides.data(),
                        sizeof(uint64_t) * max_rank).wait();
    g_sycl_queue.memcpy(p_b_strides, br.b_strides.data(),
                        sizeof(uint64_t) * max_rank).wait();

    const float_t* p_src_data = src.get_data();
    float_t* p_dst_data = get_data();

    g_sycl_queue.submit([&](sycl::handler & cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_dst)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t off_src = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_a_strides, max_rank);
            uint64_t off_dst = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_b_strides, max_rank);

            float_t val = p_src_data[off_src];

            p_dst_data[off_dst] = val;
        });
    }).wait();

    sycl::free(p_res_divs, g_sycl_queue);
    sycl::free(p_a_strides, g_sycl_queue);
    sycl::free(p_b_strides, g_sycl_queue);
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
        return;
    }

    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());
    if (axis != -1 && (axis < 0 || static_cast<uint64_t>(axis) >= rank))
    {
        throw std::invalid_argument("Tensor(sort): axis out of bounds");
    }

    uint64_t total_size = get_num_elements();

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

    std::vector<uint64_t> divisors = utils::compute_divisors(m_dimensions);

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
        std::vector<uint64_t> index_factors = utils::compute_divisors(dims);
        const uint64_t D = static_cast<uint64_t>(dims.size());

        for (uint64_t s = 0; s < slice_count; ++s)
        {
            host_slice_base[s] = sycl_utils::
                idx_of(s, index_factors.data(), strides_for_dims.data(), D);
        }
    }

    uint64_t* p_slice_base = static_cast<uint64_t*>(
        sycl::malloc_device(static_cast<size_t>(slice_count) * sizeof(uint64_t),
                            g_sycl_queue));
    uint64_t* p_divisors = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));

    if (!p_divisors || !p_strides || !p_slice_base)
    {
        if (p_divisors) sycl::free(p_divisors, g_sycl_queue);
        if (p_strides) sycl::free(p_strides, g_sycl_queue);
        if (p_slice_base) sycl::free(p_slice_base, g_sycl_queue);
        throw std::bad_alloc();
    }
    g_sycl_queue.memcpy(p_slice_base, host_slice_base.data(),
        static_cast<size_t>(slice_count) * sizeof(uint64_t)).wait();
    g_sycl_queue.memcpy(p_divisors, divisors.data(),
                        sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy(p_strides, m_strides.data(),
                        sizeof(uint64_t) * rank).wait();

    float_t* tensor_data = m_p_data.get();
    float_t* merge_buffer = nullptr;
    if (m_mem_loc == MemoryLocation::DEVICE)
    {
        merge_buffer = static_cast<float_t*>(sycl::malloc_device
            (static_cast<size_t>(total_size) * sizeof(float_t), g_sycl_queue));
    }
    else
    {
        merge_buffer = static_cast<float_t*>(sycl::malloc_shared
            (static_cast<size_t>(total_size) * sizeof(float_t), g_sycl_queue));
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

    size_t workgroup_size = temper::utils::compute_pow2_workgroup_size(
        g_sycl_queue, static_cast<size_t>(effective_axis_size));

    // Bottom-up merge passes.
    for (uint64_t width = 1; width < effective_axis_size; width *= 2)
    {
        const uint64_t chunks_per_slice =
            (effective_axis_size + 2 * width - 1) / (2 * width);
        const uint64_t total_merges =
            static_cast<uint64_t>(slice_count) * chunks_per_slice;
        if (total_merges == 0) break;

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
                if (merge_start >= merge_end)
                {
                    return;
                }

                auto idx_of_local = [&](uint64_t logical_idx) -> uint64_t
                {
                    if (axis == -1)
                    {
                        return sycl_utils::idx_of
                            (logical_idx, p_divisors, p_strides, rank);
                    }
                    else
                    {
                        return slice_base + logical_idx * axis_stride;
                    }
                };

                auto find_partition_local = [&](uint64_t k) -> uint64_t
                {
                    if (axis == -1)
                    {
                        return sycl_utils::merge_path_partition<float_t>(
                            k, left, mid, right, p_divisors, p_strides, rank,
                            merge_input);
                    }
                    else
                    {
                        uint64_t i_min;
                        if (k > len_right)
                        {
                            i_min = k - len_right;
                        }
                        else
                        {
                            i_min = 0;
                        }
                        uint64_t i_max;
                        if (k < len_left)
                        {
                            i_max = k;
                        }
                        else
                        {
                            i_max = len_left;
                        }
                        while (i_min < i_max)
                        {
                            uint64_t i_mid = (i_min + i_max) / 2;
                            uint64_t j_mid = k - i_mid;
                            uint64_t a_off = idx_of_local(left + i_mid);
                            uint64_t b_off = idx_of_local(mid + j_mid - 1);
                            float_t va = merge_input[a_off];
                            float_t vb = merge_input[b_off];
                            bool cmp;
                            if (!sycl::isnan(va) && sycl::isnan(vb))
                            {
                                cmp = true;
                            }
                            else if (va < vb)
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
                    }
                };

                const uint64_t i_start = find_partition_local(merge_start);
                const uint64_t j_start = merge_start - i_start;
                const uint64_t i_end = find_partition_local(merge_end);
                const uint64_t j_end = merge_end - i_end;

                uint64_t i = i_start;
                uint64_t j = j_start;
                uint64_t out_k = merge_start;
                while (i < i_end && j < j_end)
                {
                    uint64_t a_idx = idx_of_local(left + i);
                    uint64_t b_idx = idx_of_local(mid + j);
                    float_t va = merge_input[a_idx];
                    float_t vb = merge_input[b_idx];
                    uint64_t out_idx = idx_of_local(left + out_k);
                    if ((!sycl::isnan(va) && sycl::isnan(vb)) || va < vb)
                    {
                        merge_output[out_idx] = va;
                        ++i;
                    }
                    else
                    {
                        merge_output[out_idx] = vb;
                        ++j;
                    }
                    ++out_k;
                }
                while (i < i_end)
                {
                    merge_output[idx_of_local(left + out_k)] =
                        merge_input[idx_of_local(left + i)];
                    ++i; ++out_k;
                }
                while (j < j_end)
                {
                    merge_output[idx_of_local(left + out_k)] =
                        merge_input[idx_of_local(mid + j)];
                    ++j; ++out_k;
                }
            });
        }).wait();

        float_t* tmp = merge_input;
        merge_input = merge_output;
        merge_output = tmp;
    }

    if (merge_input != tensor_data)
    {
        g_sycl_queue.memcpy(tensor_data, merge_input,
            static_cast<size_t>(total_size) * sizeof(float_t)).wait();
    }

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
        const MemoryLocation res_loc = m_mem_loc;
        return Tensor<float_t>({1}, res_loc);
    }

    const uint64_t rank = get_rank();
    uint64_t total_size = get_num_elements();
    const MemoryLocation res_loc = m_mem_loc;

    std::vector<uint64_t> divisors = utils::compute_divisors(m_dimensions);

    // Validate axis
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

    std::vector<uint64_t> new_dimensions(m_dimensions);
    if (axis != -1)
    {
        new_dimensions[ax] = 1;
    }

    uint64_t output_size = 1;
    for (uint64_t d : new_dimensions) output_size *= d;

    uint64_t effective_axis_size = 0;
    uint64_t effective_output_size = 0;
    if (axis == -1)
    {
        effective_axis_size = total_size;
        effective_output_size = 1;
    }
    else
    {
        effective_axis_size = axis_size;
        effective_output_size = output_size;
    }

    Tensor<float_t> result;
    if (axis == -1)
    {
        result = Tensor<float_t>({1}, res_loc);
    }
    else
    {
        result = Tensor<float_t>(new_dimensions, res_loc);
    }

    float_t* p_out = result.get_data();
    const float_t* p_src = get_data();

    uint64_t* p_strides_dev = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    uint64_t* p_divisors_dev = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));

    std::vector<uint64_t> fixed_dims;
    if (rank > 0) fixed_dims.reserve(rank - 1);
    for (uint64_t i = 0; i < rank; ++i)
    {
        if (axis != -1 && i == ax) continue;
        fixed_dims.push_back(m_dimensions[i]);
    }
    const uint64_t fixed_count = fixed_dims.size();

    std::vector<uint64_t> fixed_divisors;
    if (fixed_count > 0)
    {
        fixed_divisors = temper::utils::compute_divisors(fixed_dims);
    }

    uint64_t* p_fixed_divs_dev = nullptr;

    auto [workgroup_size, num_groups_per_slice] =
        utils::compute_wg_and_groups(g_sycl_queue,
            static_cast<size_t>(effective_axis_size));

    size_t total_groups = static_cast<size_t>(effective_output_size) * num_groups_per_slice;
    size_t total_group_items = total_groups * workgroup_size;
    if (total_group_items == 0) total_group_items = workgroup_size;

    size_t partial_count = static_cast<size_t>(effective_output_size) * num_groups_per_slice;
    size_t alloc_partial_count = 0;
    if (partial_count == 0)
    {
        alloc_partial_count = 1;
    }
    else
    {
        alloc_partial_count = partial_count;
    }

    float_t* p_partials = nullptr;
    int32_t* p_error_flag = nullptr;

    if (fixed_count > 0)
    {
        p_fixed_divs_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * fixed_count, g_sycl_queue));
    }

    p_partials = static_cast<float_t*>(
        sycl::malloc_device(sizeof(float_t) * alloc_partial_count, g_sycl_queue));

    p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_failed = false;
    if (!p_strides_dev) alloc_failed = true;
    if (!p_divisors_dev) alloc_failed = true;
    if (fixed_count > 0 && !p_fixed_divs_dev) alloc_failed = true;
    if (!p_partials) alloc_failed = true;
    if (!p_error_flag) alloc_failed = true;

    if (alloc_failed)
    {
        if (p_strides_dev) sycl::free(p_strides_dev, g_sycl_queue);
        if (p_divisors_dev) sycl::free(p_divisors_dev, g_sycl_queue);
        if (p_fixed_divs_dev) sycl::free(p_fixed_divs_dev, g_sycl_queue);
        if (p_partials) sycl::free(p_partials, g_sycl_queue);
        if (p_error_flag) sycl::free(p_error_flag, g_sycl_queue);

        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_strides_dev, m_strides.data(),
                        sizeof(uint64_t) * rank).wait();

    g_sycl_queue.memcpy(p_divisors_dev, divisors.data(),
                        sizeof(uint64_t) * rank).wait();

    if (fixed_count > 0)
    {
        g_sycl_queue.memcpy(p_fixed_divs_dev, fixed_divisors.data(),
                            sizeof(uint64_t) * fixed_count).wait();
    }

    g_sycl_queue.memset
        (p_partials, 0, sizeof(float_t) * alloc_partial_count).wait();

    *p_error_flag = 0;

    // First kernel: compute partial sums.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(total_group_items),
                              sycl::range<1>(workgroup_size)),
            [=](sycl::nd_item<1> it)
        {
            const size_t global_group_id = it.get_group(0);
            const size_t slice = global_group_id / num_groups_per_slice;
            const size_t group_in_slice = global_group_id % num_groups_per_slice;
            const size_t local_id = it.get_local_id(0);

            float_t local_sum = float_t{};

            if (axis == -1)
            {
                size_t start = group_in_slice * workgroup_size + local_id;
                size_t stride = workgroup_size * num_groups_per_slice;
                const size_t N = static_cast<size_t>(effective_axis_size);
                for (size_t linear = start; linear < N; linear += stride)
                {
                    uint64_t offset = sycl_utils::idx_of(
                        static_cast<uint64_t>(linear),
                        p_divisors_dev, p_strides_dev, rank);

                    float_t v = p_src[offset];
                    sycl_utils::device_check_nan_and_set(v, p_error_flag);
                    local_sum += v;
                }
            }
            else
            {
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
                    j < static_cast<size_t>(effective_axis_size);
                    j += stride)
                {
                    uint64_t offs = base_offset + static_cast<uint64_t>(j) *
                        p_strides_dev[ax];
                    float_t v = p_src[offs];
                    sycl_utils::device_check_nan_and_set(v, p_error_flag);
                    local_sum += v;
                }
            }

            auto group = it.get_group();
            float_t group_sum = sycl::reduce_over_group
                (group, local_sum, sycl::plus<float_t>());

            sycl_utils::device_check_finite_and_set(group_sum, p_error_flag);

            if (local_id == 0)
            {
                size_t write_idx = slice * num_groups_per_slice + group_in_slice;
                p_partials[write_idx] = group_sum;
            }
        });
    }).wait();

    // Second kernel: reduce partials per slice to final outputs.
    size_t wg2 = 1;
    {
        size_t candidate_limit = workgroup_size;
        size_t candidate_min = 1;
        if (num_groups_per_slice > candidate_min) candidate_min = num_groups_per_slice;
        size_t chosen = 1;
        while (chosen * 2 <= std::min(candidate_limit, candidate_min))
        {
            chosen *= 2;
        }
        wg2 = chosen;
    }

    size_t second_ndrange = static_cast<size_t>(effective_output_size) * wg2;
    if (second_ndrange == 0) second_ndrange = wg2;

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(second_ndrange),
                              sycl::range<1>(wg2)),
            [=](sycl::nd_item<1> it)
        {
            const size_t slice = it.get_group(0);
            const size_t lid = it.get_local_id(0);
            const size_t local_range = it.get_local_range(0);

            float_t v = float_t{};
            for (size_t idx = lid; idx < num_groups_per_slice; idx += local_range)
            {
                float_t pv = p_partials[slice * num_groups_per_slice + idx];
                sycl_utils::device_check_nan_and_set(pv, p_error_flag);
                v += pv;
            }

            auto group = it.get_group();
            float_t total =
                sycl::reduce_over_group(group, v, sycl::plus<float_t>());
            sycl_utils::device_check_finite_and_set(total, p_error_flag);

            if (lid == 0)
            {
                p_out[slice] = total;
            }
        });
    }).wait();

    sycl::free(p_partials, g_sycl_queue);
    sycl::free(p_divisors_dev, g_sycl_queue);
    sycl::free(p_strides_dev, g_sycl_queue);
    if (p_fixed_divs_dev) sycl::free(p_fixed_divs_dev, g_sycl_queue);

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

    const uint64_t rank = get_rank();
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

    const float_t* p_src = get_data();
    float_t* p_out = result.get_data();
    std::vector<uint64_t> divisors = utils::compute_divisors(m_dimensions);

    std::vector<uint64_t> fixed_divisors;
    std::vector<uint64_t> fixed_strides;
    std::vector<uint64_t> fixed_out_strides;
    uint64_t fixed_count = 0;

    if (axis != -1)
    {
        std::vector<uint64_t> fixed_dims;
        fixed_dims.reserve(rank - 1);
        for (uint64_t i = 0; i < rank; ++i)
        {
            if (i == ax) continue;
            fixed_dims.push_back(m_dimensions[i]);
        }
        fixed_count = static_cast<uint64_t>(fixed_dims.size());
        fixed_divisors = temper::utils::compute_divisors(fixed_dims);

        fixed_strides.reserve(fixed_count);
        fixed_out_strides.reserve(fixed_count);
        for (uint64_t d = 0; d < rank; ++d)
        {
            if (d == ax) continue;
            fixed_strides.push_back(m_strides[d]);
            fixed_out_strides.push_back(result.m_strides[d]);
        }
    }
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
            if (i == ax) continue;
            out_sz *= m_dimensions[i];
        }
        effective_output_size = out_sz;
    }

    auto wg_and_groups = utils::compute_wg_and_groups(
        g_sycl_queue, static_cast<size_t>(effective_axis_size));
    size_t workgroup_size = wg_and_groups.first;
    size_t num_groups_per_slice = wg_and_groups.second;
    if (workgroup_size == 0) workgroup_size = 1;
    if (num_groups_per_slice == 0) num_groups_per_slice = 1;

    size_t total_groups =
        static_cast<size_t>(effective_output_size) * num_groups_per_slice;
    size_t total_group_items = total_groups * workgroup_size;
    if (total_group_items == 0)
    {
        total_group_items = workgroup_size;
    }

    uint64_t *p_divs_dev = nullptr;
    uint64_t *p_strides_dev = nullptr;
    uint64_t *p_out_strides_dev = nullptr;
    uint64_t *p_fixed_divs_dev = nullptr;
    uint64_t *p_fixed_strides_dev = nullptr;
    uint64_t *p_fixed_out_strides_dev = nullptr;
    int32_t  *p_error_flag = nullptr;
    float_t  *p_block_partials = nullptr;

    p_divs_dev = static_cast<uint64_t*>
        (sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    p_strides_dev = static_cast<uint64_t*>
        (sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));

    if (axis != -1)
    {
        p_out_strides_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * rank, g_sycl_queue));
    }

    if (fixed_count > 0)
    {
        p_fixed_divs_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * fixed_count, g_sycl_queue));
        p_fixed_strides_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * fixed_count, g_sycl_queue));
        p_fixed_out_strides_dev = static_cast<uint64_t*>(
            sycl::malloc_device(sizeof(uint64_t) * fixed_count, g_sycl_queue));
    }

    p_error_flag = static_cast<int32_t*>
        (sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));
    p_block_partials = static_cast<float_t*>
        (sycl::malloc_device(sizeof(float_t) * total_groups, g_sycl_queue));

    bool alloc_ok = true;
    if (!p_divs_dev || !p_strides_dev || !p_error_flag || !p_block_partials)
        alloc_ok = false;
    if (axis != -1 && !p_out_strides_dev) alloc_ok = false;
    if (fixed_count > 0 &&
        (!p_fixed_divs_dev || !p_fixed_strides_dev || !p_fixed_out_strides_dev))
        alloc_ok = false;

    if (!alloc_ok)
    {
        if (p_block_partials) sycl::free(p_block_partials, g_sycl_queue);
        if (p_fixed_out_strides_dev) sycl::free(p_fixed_out_strides_dev, g_sycl_queue);
        if (p_fixed_strides_dev) sycl::free(p_fixed_strides_dev, g_sycl_queue);
        if (p_fixed_divs_dev) sycl::free(p_fixed_divs_dev, g_sycl_queue);
        if (p_out_strides_dev) sycl::free(p_out_strides_dev, g_sycl_queue);
        if (p_strides_dev) sycl::free(p_strides_dev, g_sycl_queue);
        if (p_divs_dev) sycl::free(p_divs_dev, g_sycl_queue);
        if (p_error_flag) sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy
        (p_divs_dev, divisors.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy
        (p_strides_dev, m_strides.data(), sizeof(uint64_t) * rank).wait();

    if (axis != -1)
    {
        g_sycl_queue.memcpy(p_out_strides_dev, result.m_strides.data(),
                            sizeof(uint64_t) * rank).wait();
    }

    if (fixed_count > 0)
    {
        g_sycl_queue.memcpy(p_fixed_divs_dev, fixed_divisors.data(),
                            sizeof(uint64_t) * fixed_count).wait();
        g_sycl_queue.memcpy(p_fixed_strides_dev, fixed_strides.data(),
                            sizeof(uint64_t) * fixed_count).wait();
        g_sycl_queue.memcpy(p_fixed_out_strides_dev, fixed_out_strides.data(),
                            sizeof(uint64_t) * fixed_count).wait();
    }

    *p_error_flag = 0;
    if (total_groups > 0)
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

            const size_t index_in_slice = group_in_slice * local_range + local_id;
            const size_t base_index = group_in_slice * local_range;
            size_t remaining = 0;
            if (effective_axis_size > base_index)
            {
                remaining = static_cast<size_t>(effective_axis_size) - base_index;
            }
            const size_t valid_count = std::min(remaining, local_range);

            bool active = (index_in_slice < static_cast<size_t>(effective_axis_size));

            uint64_t base_src = 0;
            uint64_t base_dst = 0;

            if (effective_output_size > 1)
            {
                base_src = sycl_utils::idx_of(slice,
                    p_fixed_divs_dev,
                    p_fixed_strides_dev,
                    static_cast<uint64_t>(fixed_count));
                if (axis != -1)
                {
                    base_dst = sycl_utils::idx_of(slice,
                        p_fixed_divs_dev,
                        p_fixed_out_strides_dev,
                        static_cast<uint64_t>(fixed_count));
                }
            }

            uint64_t src_off = 0, dst_off = 0;
            if (axis == -1)
            {
                uint64_t linear = static_cast<uint64_t>(index_in_slice);
                src_off = sycl_utils::idx_of
                    (linear, p_divs_dev, p_strides_dev, rank);
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
                sycl_utils::device_check_nan_and_set(x, p_error_flag);
            }

            float_t prefix = sycl::inclusive_scan_over_group
                (it.get_group(), x, sycl::plus<float_t>());

            if (active)
            {
                sycl_utils::device_check_finite_and_set(prefix, p_error_flag);
                p_out[dst_off] = prefix;
            }

            size_t last_valid_lane = 0;
            if (valid_count > 0) last_valid_lane = valid_count - 1;
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
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(effective_output_size)),
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

            if (group_in_slice == 0) return;

            const size_t index_in_slice = group_in_slice * local_range + local_id;
            if (index_in_slice >= static_cast<size_t>(effective_axis_size)) return;

            uint64_t base_dst = 0;
            if (effective_output_size > 1)
            {
                base_dst = temper::sycl_utils::idx_of(slice,
                    p_fixed_divs_dev,
                    p_fixed_out_strides_dev,
                    static_cast<uint64_t>(fixed_count));
            }

            uint64_t dst_off = 0;
            if (axis == -1)
            {
                dst_off = static_cast<uint64_t>(index_in_slice);
            }
            else
            {
                dst_off = base_dst +
                    static_cast<uint64_t>(index_in_slice) * p_out_strides_dev[ax];
            }

            float_t add = p_block_partials
                [slice * num_groups_per_slice + (group_in_slice - 1)];
            p_out[dst_off] += add;
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_block_partials, g_sycl_queue);
    sycl::free(p_fixed_out_strides_dev, g_sycl_queue);
    sycl::free(p_fixed_strides_dev, g_sycl_queue);
    sycl::free(p_fixed_divs_dev, g_sycl_queue);
    sycl::free(p_out_strides_dev, g_sycl_queue);
    sycl::free(p_strides_dev, g_sycl_queue);
    sycl::free(p_divs_dev, g_sycl_queue);
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
Tensor<float_t> Tensor<float_t>::mean(int64_t axis) const
{
    const uint64_t total_elems = this->get_num_elements();
    if (total_elems == 0)
    {
        throw std::invalid_argument(R"(Tensor(mean):
            input tensor has no elements.)");
    }

    uint64_t denom_u = 1;
    if (axis == -1)
    {
        denom_u = total_elems;
    }
    else
    {
        const uint64_t rank = this->get_rank();
        if (axis < 0 || static_cast<uint64_t>(axis) >= rank)
        {
            throw std::invalid_argument(R"(Tensor(mean): axis out of range.)");
        }
        denom_u = this->get_dimensions()[static_cast<uint64_t>(axis)];
    }

    Tensor<float_t> s = this->sum(axis);

    float_t denom_val = static_cast<float_t>(denom_u);
    MemoryLocation loc = this->get_memory_location();
    Tensor<float_t> denom_t({1}, loc);
    denom_t = denom_val;

    Tensor<float_t> result = s / denom_t;
    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::var(int64_t axis, int64_t ddof) const
{
    const uint64_t total_elems = this->get_num_elements();
    if (total_elems == 0)
    {
        throw std::invalid_argument(R"(Tensor(var):
            input tensor has no elements.)");
    }

    if (ddof < 0)
    {
        throw std::invalid_argument(R"(Tensor(var):
            ddof must be non-negative.)");
    }

    uint64_t N = 0;
    if (axis == -1)
    {
        if (static_cast<uint64_t>(ddof) >= total_elems)
        {
            throw std::invalid_argument
                (R"(Tensor(var): ddof >= number of elements.)");
        }
        N = total_elems;
    }
    else
    {
        const uint64_t rank = this->get_rank();
        if (axis < 0 || static_cast<uint64_t>(axis) >= rank)
        {
            throw std::invalid_argument(R"(Tensor(var): axis out of range.)");
        }
        const uint64_t axis_u = static_cast<uint64_t>(axis);
        const uint64_t axis_len = this->get_dimensions()[axis_u];
        if (axis_len == 0)
        {
            throw std::invalid_argument(R"(Tensor(var):
                selected axis has zero length.)");
        }
        if (static_cast<uint64_t>(ddof) >= axis_len)
        {
            throw std::invalid_argument(R"(Tensor(var):
                ddof >= axis length.)");
        }
        N = axis_len;
    }

    uint64_t denom_u = N - static_cast<uint64_t>(ddof);
    Tensor<float_t> m = this->mean(axis);

    Tensor<float_t> diff = (*this) - m;
    Tensor<float_t> sq = diff * diff;
    Tensor<float_t> sumsq = sq.sum(axis);

    float_t denom_val = static_cast<float_t>(denom_u);
    MemoryLocation loc = this->get_memory_location();
    Tensor<float_t> denom_t({1}, loc);
    denom_t = denom_val;

    Tensor<float_t> result = sumsq / denom_t;
    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::cov(std::vector<uint64_t> sample_axes,
                        std::vector<uint64_t> event_axes,
                        int64_t ddof) const
{
    const uint64_t total_elems = this->get_num_elements();
    const std::vector<uint64_t> & original_shape = this->get_dimensions();
    const uint64_t rank = this->get_rank();
    if (total_elems == 0)
    {
        throw std::invalid_argument(R"(Tensor(cov):
            input tensor has no elements.)");
    }

    if(sample_axes.empty() || event_axes.empty())
    {
        throw std::invalid_argument(R"(Tensor(cov):
            axes arguments cannot be empty.)");
    }

    if (ddof < 0)
    {
        throw std::invalid_argument(R"(Tensor(cov):
            ddof must be non-negative.)");
    }

    if (rank < 2)
    {
        throw std::invalid_argument(R"(Tensor(cov):
            rank must be >= 2.)");
    }

    // Check if the sample axes are regular
    // (within tensor range, not replicated).
    std::vector<bool> seen(rank, false);
    for (uint64_t axis : sample_axes)
    {
        if (axis >= rank)
        {
            throw std::invalid_argument(R"(Tensor(cov):
                sample axis out of range)");
        }
        if (seen[axis])
        {
            throw std::invalid_argument(R"(Tensor(cov):
                the same axis cannot be used twice)");
        }
        seen[axis] = true;
    }

    // We do the same for event axes.
    for (uint64_t axis : event_axes)
    {
        if (axis >= rank)
        {
            throw std::invalid_argument(R"(Tensor(cov):
                event axis out of range)");
        }
        if (seen[axis])
        {
            throw std::invalid_argument(R"(Tensor(cov):
                the same axis cannot be used twice)");
        }
        seen[axis] = true;
    }

    // Build the tensor shape: batch axes -> sample axes -> event axes.
    std::vector<uint64_t> t_shape;
    t_shape.reserve(rank);
    for (size_t i = 0; i < rank; ++i)
    {
        if (seen[i])
        {
            continue;
        }
        t_shape.push_back(i);
    }
    t_shape.insert(t_shape.end(), sample_axes.begin(), sample_axes.end());
    t_shape.insert(t_shape.end(), event_axes.begin(), event_axes.end());

    Tensor<float_t> t_tensor = this->transpose(t_shape);

    // We clone the transposed tensor because reshape does not work on views.
    Tensor<float_t> t_tensor_clone = t_tensor.clone();

    uint64_t sample_total = 1, event_total = 1;
    // Compute the final shape of the tensor we need to operate on.
    for (uint64_t axis : sample_axes)
    {
        sample_total *= original_shape[axis];
    }
    if (static_cast<uint64_t>(ddof) >= sample_total)
    {
        throw std::invalid_argument(R"(Tensor(cov):
                not enough samples for ddof.)");
    }
    for (uint64_t axis : event_axes)
    {
        event_total *= original_shape[axis];
    }

    const std::vector<uint64_t> & transposed_shape =
        t_tensor_clone.get_dimensions();

    const size_t num_sample_axes = sample_axes.size();
    const size_t num_event_axes  = event_axes.size();
    const size_t batch_len = rank - (num_sample_axes + num_event_axes);

    std::vector<uint64_t> final_shape;
    final_shape.reserve(batch_len + 2);
    for (size_t i = 0; i < batch_len; ++i)
    {
        final_shape.push_back(transposed_shape[i]);
    }
    final_shape.push_back(sample_total);
    final_shape.push_back(event_total);

    t_tensor_clone.reshape(final_shape);

    uint64_t sample_axis_idx = static_cast<uint64_t>(batch_len);

    Tensor<float_t> mu = t_tensor_clone.mean(sample_axis_idx);
    Tensor<float_t> centered = t_tensor_clone - mu;

    Tensor<float_t> denom({1}, m_mem_loc);
    denom = 1.0f / (sample_total - ddof);

    std::vector<uint64_t> transpose_order;
    transpose_order.reserve(centered.get_rank());
    for (uint64_t i = 0; i < batch_len; ++i)
    {
        transpose_order.push_back(i);
    }
    transpose_order.push_back(batch_len + 1);
    transpose_order.push_back(batch_len);

    Tensor<float_t> centered_t = centered.transpose(transpose_order);

    Tensor<float> result = denom * math::matmul(centered_t, centered);

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::cov(int64_t ddof) const
{
    const uint64_t rank = this->get_rank();
    if (rank < 2)
    {
        throw std::invalid_argument(R"(Tensor(cov):
            rank must be >= 2.)");
    }

    return this->cov({rank - 2}, {rank - 1}, ddof);
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::std(int64_t axis, int64_t ddof) const
{
    Tensor<float_t> v = this->var(axis, ddof);

    const std::vector<uint64_t> & in_shape = v.get_dimensions();
    if (in_shape.empty())
    {
        throw std::invalid_argument(R"(Tensor(std):
            input tensor has no elements.)");
    }

    const uint64_t arr_len = static_cast<uint64_t>(in_shape.size());
    const uint64_t total_output_elems = v.get_num_elements();
    MemoryLocation res_loc = v.get_memory_location();
    Tensor<float_t> result(in_shape, res_loc);

    const std::vector<uint64_t> in_divs =
        temper::utils::compute_divisors(in_shape);
    const std::vector<uint64_t> in_strides = v.get_strides();

    uint64_t* p_in_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(arr_len),
                            g_sycl_queue));
    uint64_t* p_in_strides = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(arr_len),
                            g_sycl_queue));
    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_in_divs && p_in_strides && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_in_divs, g_sycl_queue);
        sycl::free(p_in_strides, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_in_divs, in_divs.data(),
        sizeof(uint64_t) * arr_len).wait();
    g_sycl_queue.memcpy(p_in_strides, in_strides.data(),
        sizeof(uint64_t) * arr_len).wait();
    *p_error_flag = 0;

    const float_t* p_in_data = v.get_data();
    float_t* p_out = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);
            uint64_t in_idx = temper::sycl_utils::idx_of(flat,
                                                    p_in_divs,
                                                    p_in_strides,
                                                    arr_len);
            float_t val = p_in_data[in_idx];
            temper::sycl_utils::device_check_nan_and_set<float_t>
                (val, p_error_flag);
            float_t outv = sycl::sqrt(val);
            temper::sycl_utils::device_check_finite_and_set<float_t>
                (outv, p_error_flag);
            p_out[flat] = outv;
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_error_flag, g_sycl_queue);
    sycl::free(p_in_divs, g_sycl_queue);
    sycl::free(p_in_strides, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::runtime_error(R"(Tensor(std): NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(Tensor(std):
                non-finite result (Inf/overflow/NaN) produced.)");
        }
        throw std::runtime_error(R"(Tensor(std):
            numeric error during sqrt computation.)");
    }

    return result;
}

template<typename float_t>
std::pair<Tensor<float_t>, Tensor<float_t>> Tensor<float_t>::eig
    (uint64_t max_iters, float_t tol) const
{
    const uint64_t rank = this->get_rank();
    if (rank < 2)
    {
        throw std::invalid_argument("Tensor(eig): rank must be >= 2.");
    }

    const std::vector<uint64_t> shape = this->get_dimensions();
    if (shape[rank - 1] != shape[rank - 2])
    {
        throw std::invalid_argument(R"(Tensor(eig):
            last two dims must be square.)");
    }
    const uint64_t n = shape[rank - 1];

    std::vector<uint64_t> batch_shape;
    for (uint64_t i = 0; i + 2 < rank; ++i)
    {
        batch_shape.push_back(shape[i]);
    }

    std::vector<uint64_t> eigvals_shape = batch_shape;
    eigvals_shape.push_back(n);
    std::vector<uint64_t> eigvecs_shape = batch_shape;
    eigvecs_shape.push_back(n);
    eigvecs_shape.push_back(n);

    MemoryLocation res_loc = this->get_memory_location();
    Tensor<float_t> eigvals_tensor(eigvals_shape, res_loc);
    Tensor<float_t> eigvecs_tensor(eigvecs_shape, res_loc);

    uint64_t batch_count = 1;
    for (uint64_t d : batch_shape)
    {
        batch_count *= d;
    }

    std::vector<uint64_t> batch_strides(batch_shape.size(), 1);
    if (!batch_shape.empty())
    {
        for (int i = static_cast<int>(batch_shape.size()) - 2; i >= 0; --i)
        {
            batch_strides[i] = batch_strides[i + 1] * batch_shape[i + 1];
        }
    }

    for (uint64_t b = 0; b < batch_count; ++b)
    {
        std::vector<uint64_t> start_indices(rank, 0);
        uint64_t rem = b;
        for (size_t k = 0; k < batch_shape.size(); ++k)
        {
            uint64_t idx = rem / batch_strides[k];
            rem = rem % batch_strides[k];
            start_indices[k] = idx;
        }
        start_indices[rank - 2] = 0;
        start_indices[rank - 1] = 0;

        Tensor<float_t> A = Tensor<float_t>
            (*this, start_indices, std::vector<uint64_t>({ n, n }));

        std::vector<float_t> eigvals; eigvals.reserve(n);
        std::vector<Tensor<float_t>> eigvecs; eigvecs.reserve(n);

        Tensor<float_t> eigvecs_mat({n, n}, res_loc);
        Tensor<float_t> eigvals_vec({n}, res_loc);

        // Iterate to find n eigenpairs by deflation.
        for (uint64_t j = 0; j < n; ++j)
        {
            Tensor<float_t> v = stats::randn<float_t>({n});

            float_t nv2 = (v * v).sum();
            if (nv2 == static_cast<float_t>(0))
            {
                throw std::runtime_error(R"(Tensor(eig):
                    initial vector is zero.)");
            }
            v = v / Tensor<float_t>(std::sqrt(nv2));

            for (uint64_t iter = 0; iter < max_iters; ++iter)
            {
                Tensor<float_t> w = math::matmul(A, v);

                float_t nw2 = (w * w).sum();
                if (nw2 == static_cast<float_t>(0))
                {
                    break;
                }
                Tensor<float_t> v_next = w / Tensor<float_t>(std::sqrt(nw2));

                Tensor<float_t> diff = v_next - v;
                float_t diff2 = (diff * diff).sum();
                if (std::sqrt(diff2) < tol)
                {
                    v = v_next;
                    break;
                }
                v = v_next;
            }

            Tensor<float_t> AT = A.transpose();
            Tensor<float_t> u = stats::randn<float_t>({n});
            float_t nu2 = (u * u).sum();
            if (nu2 == static_cast<float_t>(0))
            {
                throw std::runtime_error(R"(Tensor(eig):
                    initial left vector is zero.)");
            }
            u = u / Tensor<float_t>(std::sqrt(nu2));

            for (uint64_t iter = 0; iter < max_iters; ++iter)
            {
                Tensor<float_t> w = math::matmul(AT, u);

                float_t nw2 = (w * w).sum();
                if (nw2 == static_cast<float_t>(0))
                {
                    break;
                }
                Tensor<float_t> u_next = w / Tensor<float_t>(std::sqrt(nw2));

                Tensor<float_t> diff = u_next - u;
                float_t diff2 = (diff * diff).sum();
                if (std::sqrt(diff2) < tol)
                {
                    u = u_next;
                    break;
                }
                u = u_next;
            }

            float_t uv = (u * v).sum();
            if (uv == static_cast<float_t>(0))
            {
                throw std::runtime_error(R"(Tensor(eig):
                    left/right inner product is zero; cannot deflate.)");
            }
            u = u / Tensor<float_t>(uv);

            Tensor<float_t> Av = math::matmul(A, v);
            float_t lambda = (u * Av).sum();

            eigvecs.push_back(v);
            eigvals.push_back(lambda);

            std::vector<uint64_t> start_col_idx(2);
            start_col_idx[0] = 0;
            start_col_idx[1] = j;
            std::vector<uint64_t> view_shape = { n, 1 };
            Tensor<float_t> dst_col_view(eigvecs_mat, start_col_idx, view_shape);
            Tensor<float_t> src_col = v;
            src_col.reshape({ n, 1 });
            dst_col_view.copy_from(src_col);

            Tensor<float_t> v_col = v; v_col.reshape({ n, 1 });
            Tensor<float_t> u_row = u; u_row.reshape({ 1, n });
            Tensor<float_t> outer = math::matmul(v_col, u_row);
            Tensor<float_t> scaled = outer * Tensor<float_t>(lambda);
            A = A - scaled;
        }

        eigvals_vec = eigvals;

        // Write eigvals_vec into eigvals_tensor at the batch location.
        {
            std::vector<uint64_t> out_start;
            out_start.reserve(batch_shape.size() + 1);
            for (size_t i = 0; i < batch_shape.size(); ++i)
            {
                out_start.push_back(start_indices[i]);
            }
            out_start.push_back(0);
            std::vector<uint64_t> view_shape(out_start.size(), 1);
            view_shape.back() = n;
            Tensor<float_t> dst_vals_view(eigvals_tensor, out_start, view_shape);
            Tensor<float_t> src_vals = eigvals_vec;
            src_vals.reshape(view_shape);
            dst_vals_view.copy_from(src_vals);
        }

        // Write eigvecs_mat into eigvecs_tensor at the batch location.
        {
            std::vector<uint64_t> out_start;
            out_start.reserve(batch_shape.size() + 2);
            for (size_t i = 0; i < batch_shape.size(); ++i)
            {
                out_start.push_back(start_indices[i]);
            }
            out_start.push_back(0);
            out_start.push_back(0);
            std::vector<uint64_t> view_shape(out_start.size(), 1);
            view_shape[out_start.size() - 2] = n;
            view_shape[out_start.size() - 1] = n;
            Tensor<float_t> dst_vecs_view(eigvecs_tensor, out_start, view_shape);
            Tensor<float_t> src_vecs = eigvecs_mat;
            src_vecs.reshape(view_shape);
            dst_vecs_view.copy_from(src_vecs);
        }
    }

    return std::make_pair(eigvals_tensor, eigvecs_tensor);
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
void Tensor<float_t>::print_shape(std::ostream& os) const
{
    os << "[";
    for (size_t i = 0; i < m_dimensions.size(); ++i)
    {
        if (i > 0) os << ", ";
        os << m_dimensions[i];
    }
    os << "]\n";
}

template<typename float_t>
const float_t * Tensor<float_t>::get_data() const noexcept
{
    return m_p_data.get();
}

template<typename float_t>
float_t * Tensor<float_t>::get_data() noexcept
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

template<typename float_t>
std::vector<uint64_t> Tensor<float_t>::index_to_coords(uint64_t flat) const
{
    const size_t rank = m_dimensions.size();
    if (rank == 0)
        return {};

    uint64_t total = 1;
    for (uint64_t s : m_dimensions)
        total *= s;

    if (flat >= total)
        throw std::out_of_range("Tensor::index_to_coords: flat out of range");

    std::vector<uint64_t> divs = utils::compute_divisors(m_dimensions);
    std::vector<uint64_t> coords(rank, 0);

    for (size_t d = 0; d < rank; ++d)
    {
        if (divs[d] == 0)
        {
            coords[d] = 0;
        }
        else
        {
            coords[d] = (flat / divs[d]) % m_dimensions[d];
        }
    }

    return coords;
}

template<typename float_t>
uint64_t Tensor<float_t>::coords_to_index
    (const std::vector<uint64_t>& coords) const
{
    const size_t rank = m_dimensions.size();
    if (coords.size() != rank)
    {
        throw std::invalid_argument(R"(Tensor(coords_to_index):
            size mismatch)");
    }

    if (rank == 0)
    {
        return 0;
    }

    std::vector<uint64_t> divs = temper::utils::compute_divisors(m_dimensions);
    uint64_t flat = 0;

    for (size_t d = 0; d < rank; ++d)
    {
        if (coords[d] >= m_dimensions[d])
        {
            throw std::out_of_range(R"(Tensor(coords_to_index):
                coord out of range)");
        }

        flat += coords[d] * divs[d];
    }

    return flat;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::at(uint64_t flat)
{
    const size_t rank = m_dimensions.size();
    if (rank == 0)
    {
        throw std::out_of_range("Tensor(at): tensor has no elements.");
    }

    const uint64_t total = get_num_elements();
    if (flat >= total)
    {
        throw std::out_of_range("Tensor(at): flat index out of range.");
    }

    std::vector<uint64_t> coords = index_to_coords(flat);
    return Tensor(*this, coords, std::vector<uint64_t>{1});
}

template<typename float_t>
const Tensor<float_t> Tensor<float_t>::at(uint64_t flat) const
{
    const size_t rank = m_dimensions.size();
    if (rank == 0)
    {
        throw std::out_of_range("Tensor(at): tensor has no elements.");
    }

    const uint64_t total = get_num_elements();
    if (flat >= total)
    {
        throw std::out_of_range("Tensor(at): flat index out of range.");
    }

    std::vector<uint64_t> coords = index_to_coords(flat);
    return Tensor(*this, coords, std::vector<uint64_t>{1});
}

template class Tensor<float>;

} // namespace temper