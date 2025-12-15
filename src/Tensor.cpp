/**
 * @file Tensor.cpp
 * @brief Tensor class function definitions.
 */

#include "temper/Tensor.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Utils.hpp"
#include "temper/Errors.hpp"

#include <iostream>

namespace temper
{

template<typename value_t>
void Tensor<value_t>::compute_strides()
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

    for (int64_t i = this->get_rank() - 1; i > 0; --i)
    {
        uint64_t dim = m_dimensions[i];
        uint64_t next_stride = m_strides[i];

        TEMPER_CHECK(next_stride > U64_MAX / dim,
            bounds_error,
            R"(Tensor(compute_strides): stride multiplication overflow.)");

        m_strides[i - 1] = next_stride * dim;
    }
}

template<typename value_t>
Tensor<value_t>::Tensor(const std::vector<uint64_t> & dimensions,
                        MemoryLocation loc)
    : m_dimensions(dimensions),
      m_strides(dimensions.size()),
      m_own_data(true),
      m_mem_loc(loc)
{
    TEMPER_CHECK(m_dimensions.empty(),
        validation_error,
        R"(Tensor(main constructor):
            dims must not be empty (rank-0 not supported).)");

    // Ensure the number of dimensions itself fits in an int64_t.
    const size_t max_int64_size =
        static_cast<size_t>(std::numeric_limits<int64_t>::max());

    TEMPER_CHECK(m_dimensions.size() > max_int64_size,
        bounds_error,
        R"(Tensor(main constructor):
            number of dimensions doesn't fit in int64_t.)");

    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    // Compute total element count with overflow guard.
    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        TEMPER_CHECK(d == 0,
            validation_error,
            R"(Tensor(main constructor):
                zero-sized dimension is not allowed.)");

        TEMPER_CHECK(total_size > U64_MAX / d,
            bounds_error,
            R"(Tensor(main constructor):
                total element count overflow (too many elements).)");

        total_size *= d;
    }

    // Compute byte count safely and ensure it fits in size_t.
    const uint64_t elem_size_u64 = static_cast<uint64_t>(sizeof(value_t));

    TEMPER_CHECK(total_size > U64_MAX / elem_size_u64,
        bounds_error,
        R"(Tensor(main constructor):
            allocation size (bytes) overflow (uint64_t).)");

    // Compute byte count in uint64_t.
    const uint64_t alloc_bytes_u64 = total_size * elem_size_u64;

    const uint64_t max_size_t_u64 = static_cast<uint64_t>
        (std::numeric_limits<size_t>::max());

    TEMPER_CHECK(alloc_bytes_u64 > max_size_t_u64,
        bounds_error,
        R"(Tensor(main constructor): allocation size
            (bytes) doesn't fit into size_t on this platform.)");

    // Safe to narrow to size_t.
    const size_t allocation_bytes = static_cast<size_t>(alloc_bytes_u64);

    // Compute strides now that dimensions are validated.
    compute_strides();

    // Query device limits and fail early if the request is impossible.
    auto dev = g_sycl_queue.get_device();
    const uint64_t dev_max_alloc = static_cast<uint64_t>(
        dev.get_info<sycl::info::device::max_mem_alloc_size>());

    TEMPER_CHECK(alloc_bytes_u64 > dev_max_alloc,
        device_error,
        R"(Tensor(main constructor):
            requested allocation exceeds device max_mem_alloc_size.)");

    // Allocate USM (shared for HOST, device for DEVICE).
    value_t* raw_ptr = nullptr;
    if (m_mem_loc == MemoryLocation::HOST)
    {
        raw_ptr = static_cast<value_t*>(
            sycl::malloc_shared(allocation_bytes, g_sycl_queue));
    }
    else
    {
        raw_ptr = static_cast<value_t*>(
            sycl::malloc_device(allocation_bytes, g_sycl_queue));
    }

    TEMPER_CHECK(!raw_ptr,
        device_error,
        R"(Tensor(main constructor):
            error allocating tensor memory on device.)");

    m_p_data = std::shared_ptr<value_t>(raw_ptr,
        [](value_t* p)
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

template<typename value_t>
Tensor<value_t>::Tensor(const std::initializer_list<uint64_t> & dimensions,
                        MemoryLocation loc)
    : Tensor(std::vector<uint64_t>(dimensions), loc)
{
    // Constructor already delegated; no op.
}

template<typename value_t>
Tensor<value_t>::Tensor(const Tensor & other)
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
            m_p_data = std::shared_ptr<value_t>(nullptr);
            return;
        }

        uint64_t total_size = other.get_num_elements();

        const size_t alloc_bytes =
            static_cast<size_t>(total_size) * sizeof(value_t);

        // Allocate same kind of USM as other's mem_loc.
        value_t* raw_ptr = nullptr;
        if (m_mem_loc == MemoryLocation::HOST)
        {
            raw_ptr = static_cast<value_t*>
                (sycl::malloc_shared(alloc_bytes, g_sycl_queue));
        }
        else
        {
            raw_ptr = static_cast<value_t*>
                (sycl::malloc_device(alloc_bytes, g_sycl_queue));
        }

        TEMPER_CHECK(!raw_ptr,
            device_error,
            R"(Tensor(copy constructor):
                error allocating tensor memory on device.)");

        m_p_data = std::shared_ptr<value_t>(raw_ptr,
            [](value_t* p) { if (p) sycl::free(p, g_sycl_queue); });

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

template<typename value_t>
Tensor<value_t>::Tensor(Tensor && other) noexcept
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

template<typename value_t>
Tensor<value_t>::Tensor(value_t val, MemoryLocation loc)
    : m_dimensions({1}),
      m_strides(1),
      m_own_data(true),
      m_mem_loc(loc)
{
    compute_strides();

    const size_t alloc_bytes = sizeof(value_t);

    value_t* raw_ptr = nullptr;
    if (m_mem_loc == MemoryLocation::HOST)
    {
        raw_ptr = static_cast<value_t*>
            (sycl::malloc_shared(alloc_bytes, g_sycl_queue));
    }
    else
    {
        raw_ptr = static_cast<value_t*>
            (sycl::malloc_device(alloc_bytes, g_sycl_queue));
    }

    TEMPER_CHECK(!raw_ptr,
        device_error,
        R"(Tensor(scalar constructor):
            error allocating tensor memory on device.)");

    m_p_data = std::shared_ptr<value_t>(raw_ptr,
        [](value_t* p)
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
        g_sycl_queue.memcpy(raw_ptr, &val, sizeof(value_t)).wait();
    }
}

template<typename value_t>
Tensor<value_t>::Tensor(const Tensor & owner,
    const std::vector<uint64_t> & start_indices,
    const std::vector<uint64_t> & view_shape)
    : m_own_data(false),
      m_mem_loc(owner.m_mem_loc)
{
    const int64_t original_rank = owner.get_rank();

    const size_t max_int64_size =
        static_cast<size_t>(std::numeric_limits<int64_t>::max());

    TEMPER_CHECK(view_shape.size() > max_int64_size,
        bounds_error,
        R"(Tensor(view constructor):
            number of dimensions doesn't fit in int64_t.)");

    const int64_t view_rank = static_cast<int64_t>(view_shape.size());

    TEMPER_CHECK(!owner.m_p_data,
        validation_error,
        R"(Tensor(view constructor):
            cannot create view from uninitialized tensor.)");

    TEMPER_CHECK(start_indices.size() != static_cast<size_t>(original_rank),
        validation_error,
        R"(Tensor(view constructor):
            start_indices must match tensor rank.)");

    TEMPER_CHECK(view_rank == 0 || view_rank > original_rank,
        validation_error,
        R"(Tensor(view constructor):
            view shape rank must be between 1 and tensor rank.)");

    // Check bounds for start indices and view dimensions.
    for (int64_t i = 0; i < original_rank; ++i)
    {
        TEMPER_CHECK(start_indices[i] >= owner.m_dimensions[i],
            bounds_error,
            R"(Tensor(view constructor):
                start index out of bounds.)");
    }
    for (int64_t j = 0; j < view_rank; ++j)
    {
        int64_t i = original_rank - view_rank + j;
        TEMPER_CHECK(view_shape[j] == 0 ||
            start_indices[i] + view_shape[j] > owner.m_dimensions[i],
            bounds_error,
            R"(Tensor(view constructor):
                view shape out of bounds.)");
    }

    uint64_t offset = 0;
    for (int64_t i = 0; i < original_rank; ++i)
    {
        offset += start_indices[i] * owner.m_strides[i];
    }

    m_p_data = std::shared_ptr<value_t>
        (owner.m_p_data, owner.m_p_data.get() + offset);

    // Set dimensions and strides for the view.
    m_dimensions.assign(view_shape.begin(), view_shape.end());
    m_strides.assign(owner.m_strides.end() - view_rank, owner.m_strides.end());
}

template<typename value_t>
Tensor<value_t>::Tensor(const Tensor & owner,
    const std::vector<uint64_t> & start_indices,
    const std::vector<uint64_t> & dims,
    const std::vector<uint64_t> & strides)
    : m_dimensions(dims),
      m_strides(strides),
      m_own_data(false),
      m_mem_loc(owner.m_mem_loc)
{
    TEMPER_CHECK(!owner.m_p_data,
        validation_error,
        R"(Tensor(alias view constructor):
            cannot create view from uninitialized tensor.)");

    const int64_t owner_rank = owner.get_rank();

    const size_t max_int64_size =
        static_cast<size_t>(std::numeric_limits<int64_t>::max());

    TEMPER_CHECK(dims.size() > max_int64_size || strides.size() > max_int64_size,
        bounds_error,
        R"(Tensor(alias view constructor):
            number of dimensions or strides doesn't fit in int64_t.)");

    const int64_t view_rank = static_cast<int64_t>(dims.size());

    TEMPER_CHECK(static_cast<int64_t>(start_indices.size()) != owner_rank,
        validation_error,
        R"(Tensor(alias view constructor):
            start_indices must match owner's rank.)");

    TEMPER_CHECK(static_cast<int64_t>(strides.size()) != view_rank,
        validation_error,
        R"(Tensor(alias view constructor):
            dims and strides must have the same rank.)");

    TEMPER_CHECK(static_cast<int64_t>(view_rank) == 0,
        validation_error,
        R"(Tensor(alias view constructor):
            view rank must be >= 1.)");

    for (int64_t i = 0; i < owner_rank; ++i)
    {
        TEMPER_CHECK(start_indices[i] >= owner.m_dimensions[i],
            bounds_error,
            R"(Tensor(alias view constructor):
                start index out of bounds.)");
    }

    for (int64_t j = 0; j < view_rank; ++j)
    {
        TEMPER_CHECK(dims[j] == 0,
            validation_error,
            R"(Tensor(alias view constructor):
                view dimensions must be non-zero.)");
    }

    // Compute offset from owner's base pointer (with basic overflow safety)
    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();
    uint64_t offset = 0;
    for (int64_t i = 0; i < owner_rank; ++i)
    {
        uint64_t si = start_indices[i];
        uint64_t ostride = owner.m_strides[i];

        if (si != 0 && ostride != 0)
        {
            TEMPER_CHECK(ostride > U64_MAX / si,
                bounds_error,
                R"(Tensor(alias view constructor):
                    stride * start_index overflow while computing offset.)");

            uint64_t add = ostride * si;

            TEMPER_CHECK(offset > U64_MAX - add,
                bounds_error,
                R"(Tensor(alias view constructor):
                    offset computation overflow.)");

            offset += add;
        }
    }

    // Compute the maximum linear index that the new view may access
    // (relative to owner.m_p_data).
    uint64_t max_index = offset;
    for (int64_t j = 0; j < view_rank; ++j)
    {
        uint64_t dimm1 = dims[j] - 1;
        uint64_t vstride = strides[j];

        if (vstride != 0 && dimm1 > 0)
        {
            TEMPER_CHECK(vstride > U64_MAX / dimm1,
                bounds_error,
                R"(Tensor(alias view constructor):
                    stride * (dim-1) overflow.)");

            uint64_t add = vstride * dimm1;

            TEMPER_CHECK(max_index > U64_MAX - add,
                bounds_error,
                R"(Tensor(alias view constructor):
                    max index computation overflow.)");

            max_index += add;
        }
    }

    // Compute the owner's maximum reachable linear index
    // (relative to owner.m_p_data).
    uint64_t owner_max_index = 0;
    for (int64_t j = 0; j < owner_rank; ++j)
    {
        uint64_t odimm1 = owner.m_dimensions[j] - 1;
        uint64_t ostride = owner.m_strides[j];

        if (ostride != 0 && odimm1 > 0)
        {
            TEMPER_CHECK(ostride > U64_MAX / odimm1,
                bounds_error,
                R"(Tensor(alias view constructor):
                    stride * (dim-1) overflow while computing owner bounds.)");

            uint64_t add = ostride * odimm1;

            TEMPER_CHECK(owner_max_index > U64_MAX - add,
                bounds_error,
                R"(Tensor(alias view constructor):
                    owner max index computation overflow.)");

            owner_max_index += add;
        }
    }

    // The view's maximum index must be within the owner's reachable range.
    TEMPER_CHECK(max_index > owner_max_index,
        bounds_error,
        R"(Tensor(alias view constructor):
            view exceeds owner's bounds.)");

    m_p_data =
        std::shared_ptr<value_t>(owner.m_p_data, owner.m_p_data.get() + offset);
}

template<typename value_t>
Tensor<value_t> & Tensor<value_t>::operator=(const Tensor & other)
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
                m_p_data = std::shared_ptr<value_t>(nullptr);
                return *this;
            }

            uint64_t total_size = get_num_elements();

            const size_t alloc_bytes =
                static_cast<size_t>(total_size) * sizeof(value_t);

            value_t* raw_ptr = nullptr;
            if (m_mem_loc == MemoryLocation::HOST)
            {
                raw_ptr = static_cast<value_t*>(sycl::malloc_shared
                    (alloc_bytes, g_sycl_queue));
            }
            else
            {
                raw_ptr = static_cast<value_t*>(sycl::malloc_device
                    (alloc_bytes, g_sycl_queue));
            }

            TEMPER_CHECK(!raw_ptr,
                device_error,
                R"(Tensor(operator=):
                    error allocating tensor memory on device.)");

            m_p_data = std::shared_ptr<value_t>(raw_ptr,
                [](value_t* p)
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

template<typename value_t>
Tensor<value_t>& Tensor<value_t>::operator=(Tensor && other) noexcept
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

template<typename value_t>
Tensor<value_t> & Tensor<value_t>::operator=(const std::vector<value_t> & values)
{
    TEMPER_CHECK(m_dimensions.empty(),
        validation_error,
        R"(Tensor(values assignment):
            target tensor has no elements.)");

    uint64_t total_size = this->get_num_elements();

    uint64_t values_size = static_cast<uint64_t>(values.size());

    TEMPER_CHECK(values_size != total_size,
        validation_error,
        R"(Tensor(values assignment):
            size mismatch in 1D vector assignment.)");

    const int64_t rank = this->get_rank();
    bool dst_contig = true;
    if (rank > 0)
    {
        // Build canonical row-major strides for this shape
        std::vector<uint64_t> canon(rank);
        canon[rank - 1] = 1;
        for (int64_t i = rank - 2; i >= 0; --i)
        {
            canon[i] = canon[i + 1] * m_dimensions[i + 1];
        }
        dst_contig = (m_strides == canon);
    }

    if (dst_contig)
    {
        const size_t alloc_bytes =
            static_cast<size_t>(total_size) * sizeof(value_t);
        g_sycl_queue.memcpy(m_p_data.get(), values.data(), alloc_bytes).wait();
        return *this;
    }

    std::vector<uint64_t> divisors(rank);
    for (int64_t i = 0; i < rank; ++i)
    {
        uint64_t d = 1;
        for (int64_t j = i + 1; j < rank; ++j)
        {
            d *= m_dimensions[j];
        }
        divisors[i] = d;
    }

    sycl_utils::SyclArray<uint64_t> res_divs_ptr(g_sycl_queue, divisors,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> dst_strides_ptr(g_sycl_queue, m_strides,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<value_t> vals_shared_ptr(g_sycl_queue, values,
        MemoryLocation::HOST);

    const value_t* p_src_flat = vals_shared_ptr;
    value_t* p_dst = this->get_data();
    const uint64_t* p_res_divs = res_divs_ptr;
    const uint64_t* p_dst_strides = dst_strides_ptr;

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

    return *this;
}

template<typename value_t>
Tensor<value_t> & Tensor<value_t>::operator=(value_t val)
{
    if (m_dimensions.empty())
    {
        m_dimensions = {1};
        compute_strides();

        value_t* raw_ptr = nullptr;
        if (m_mem_loc == MemoryLocation::HOST)
        {
            raw_ptr = static_cast<value_t*>(
                sycl::malloc_shared(sizeof(value_t), g_sycl_queue));
        }
        else
        {
            raw_ptr = static_cast<value_t*>(
                sycl::malloc_device(sizeof(value_t), g_sycl_queue));
        }

        TEMPER_CHECK(!raw_ptr,
            device_error,
            R"(Tensor(operator= scalar):
                error allocating tensor memory on device.)");

        m_p_data = std::shared_ptr<value_t>(raw_ptr,
            [](value_t* p)
            {
                if (p) { sycl::free(p, g_sycl_queue); }
            }
        );

        m_own_data = true;
    }

    uint64_t total_size = this->get_num_elements();

    TEMPER_CHECK(total_size != 1,
        validation_error,
        R"(Tensor(single value assignment):
            scalar assignment only allowed for tensors with single element.)");

    if (m_mem_loc == MemoryLocation::HOST)
    {
        *m_p_data.get() = val;
    }
    else
    {
        g_sycl_queue.memcpy(m_p_data.get(), &val, sizeof(value_t)).wait();
    }
    return *this;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::operator[](uint64_t idx)
{
    const int64_t rank = this->get_rank();

    TEMPER_CHECK(rank == 0,
        validation_error,
        R"(Tensor(operator[]): tensor has no elements.)");

    TEMPER_CHECK(idx >= m_dimensions[0],
        bounds_error,
        R"(Tensor(operator[]): Index out of bounds.)");

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

template<typename value_t>
const Tensor<value_t> Tensor<value_t>::operator[](uint64_t idx) const
{
    const int64_t rank = this->get_rank();

    TEMPER_CHECK(rank == 0,
        validation_error,
        R"(Tensor(operator[]): tensor has no elements.)");

    TEMPER_CHECK(idx >= m_dimensions[0],
        bounds_error,
        R"(Tensor(operator[]): Index out of bounds.)");

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

template<typename value_t>
Tensor<value_t>::operator value_t() const
{
    TEMPER_CHECK(m_dimensions.empty(),
        validation_error,
        R"(Tensor(implicit type conversion): tensor has no elements.)");

    TEMPER_CHECK(this->get_num_elements() != 1,
        validation_error,
        R"(Tensor(implicit type conversion):
            scalar read only allowed for tensors with single element.)");

    value_t tmp;

    if (m_mem_loc == MemoryLocation::HOST)
    {
        tmp = *m_p_data.get();
    }
    else
    {
        g_sycl_queue.memcpy(&tmp, m_p_data.get(), sizeof(value_t)).wait();
    }

    return tmp;
}


template<typename value_t>
Tensor<value_t> Tensor<value_t>::operator+(const Tensor & other) const
{
    TEMPER_CHECK(m_dimensions.empty() || other.m_dimensions.empty(),
        validation_error,
        R"(Tensor(operator+): either tensor has no elements.)");

    utils::TensorDesc a_desc{m_dimensions, m_strides};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides};

    utils::BroadcastResult br = utils::compute_broadcast({a_desc, b_desc});

    std::vector<uint64_t> out_shape = std::move(br.shape);
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.strides[0]);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.strides[1]);
    std::vector<uint64_t> res_divs = std::move(br.divisors);

    const int64_t max_rank = static_cast<int64_t>(out_shape.size());

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

    sycl_utils::SyclArray<uint64_t> res_divs_ptr(g_sycl_queue, res_divs,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> a_strides_ptr(g_sycl_queue,
        a_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> b_strides_ptr(g_sycl_queue,
        b_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_ptr(g_sycl_queue, 1,
        MemoryLocation::HOST);

    const uint64_t* p_res_divs = res_divs_ptr;
    const uint64_t* p_a_strides = a_strides_ptr;
    const uint64_t* p_b_strides = b_strides_ptr;
    int32_t* p_error_flag = error_flag_ptr;

    *p_error_flag = 0;

    const value_t* p_a_data = get_data();
    const value_t* p_b_data = other.get_data();
    value_t* p_r_data = result.get_data();

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

            value_t a_val = p_a_data[offset_a];
            value_t b_val = p_b_data[offset_b];

            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(a_val), p_error_flag, 1);
            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(b_val), p_error_flag, 1);

            value_t res = a_val + b_val;

            TEMPER_DEVICE_CHECK(!sycl_utils::is_finite(res), p_error_flag, 2);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(Tensor(operator+): NaN detected in inputs.)");

    TEMPER_CHECK(err == 2,
        nonfinite_error,
        R"(Tensor(operator+): non-finite result (overflow or Inf).)");

    return result;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::operator-(const Tensor & other) const
{
    TEMPER_CHECK(m_dimensions.empty() || other.m_dimensions.empty(),
        validation_error,
        R"(Tensor(operator-): either tensor has no elements.)");

    utils::TensorDesc a_desc{m_dimensions, m_strides};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides};

    utils::BroadcastResult br = utils::compute_broadcast({a_desc, b_desc});

    std::vector<uint64_t> out_shape = std::move(br.shape);
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.strides[0]);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.strides[1]);
    std::vector<uint64_t> res_divs = std::move(br.divisors);

    const int64_t max_rank = static_cast<int64_t>(out_shape.size());

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

    sycl_utils::SyclArray<uint64_t> res_divs_ptr(g_sycl_queue, res_divs,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> a_strides_ptr(g_sycl_queue,
        a_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> b_strides_ptr(g_sycl_queue,
        b_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_ptr(g_sycl_queue, 1,
        MemoryLocation::HOST);

    const uint64_t* p_res_divs = res_divs_ptr;
    const uint64_t* p_a_strides = a_strides_ptr;
    const uint64_t* p_b_strides = b_strides_ptr;
    int32_t* p_error_flag = error_flag_ptr;

    *p_error_flag = 0;

    const value_t* p_a_data = get_data();
    const value_t* p_b_data = other.get_data();
    value_t* p_r_data = result.get_data();

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

            value_t a_val = p_a_data[offset_a];
            value_t b_val = p_b_data[offset_b];

            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(a_val), p_error_flag, 1);
            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(b_val), p_error_flag, 1);

            value_t res = a_val - b_val;

            TEMPER_DEVICE_CHECK(!sycl_utils::is_finite(res), p_error_flag, 2);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(Tensor(operator-): NaN detected in inputs.)");

    TEMPER_CHECK(err == 2,
        nonfinite_error,
        R"(Tensor(operator-): non-finite result (overflow or Inf).)");

    return result;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::operator*(const Tensor & other) const
{
    TEMPER_CHECK(m_dimensions.empty() || other.m_dimensions.empty(),
        validation_error,
        R"(Tensor(operator*): either tensor has no elements.)");

    utils::TensorDesc a_desc{m_dimensions, m_strides};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides};

    utils::BroadcastResult br = utils::compute_broadcast({a_desc, b_desc});

    std::vector<uint64_t> out_shape = std::move(br.shape);
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.strides[0]);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.strides[1]);
    std::vector<uint64_t> res_divs = std::move(br.divisors);

    const int64_t max_rank = static_cast<int64_t>(out_shape.size());

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

    sycl_utils::SyclArray<uint64_t> res_divs_ptr(g_sycl_queue, res_divs,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> a_strides_ptr(g_sycl_queue,
        a_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> b_strides_ptr(g_sycl_queue,
        b_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_ptr(g_sycl_queue, 1,
        MemoryLocation::HOST);

    const uint64_t* p_res_divs = res_divs_ptr;
    const uint64_t* p_a_strides = a_strides_ptr;
    const uint64_t* p_b_strides = b_strides_ptr;
    int32_t* p_error_flag = error_flag_ptr;

    *p_error_flag = 0;

    const value_t* p_a_data = get_data();
    const value_t* p_b_data = other.get_data();
    value_t* p_r_data = result.get_data();

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

            value_t a_val = p_a_data[offset_a];
            value_t b_val = p_b_data[offset_b];

            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(a_val), p_error_flag, 1);
            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(b_val), p_error_flag, 1);

            value_t res = a_val * b_val;

            TEMPER_DEVICE_CHECK(!sycl_utils::is_finite(res), p_error_flag, 2);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(Tensor(operator*): NaN detected in inputs.)");

    TEMPER_CHECK(err == 2,
        nonfinite_error,
        R"(Tensor(operator*): non-finite result (overflow or Inf).)");

    return result;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::operator/(const Tensor & other) const
{
    TEMPER_CHECK(m_dimensions.empty() || other.m_dimensions.empty(),
        validation_error,
        R"(Tensor(operator/): either tensor has no elements.)");

    utils::TensorDesc a_desc{m_dimensions, m_strides};
    utils::TensorDesc b_desc{other.m_dimensions, other.m_strides};

    utils::BroadcastResult br = utils::compute_broadcast({a_desc, b_desc});

    std::vector<uint64_t> out_shape = std::move(br.shape);
    std::vector<uint64_t> a_strides_broadcasted = std::move(br.strides[0]);
    std::vector<uint64_t> b_strides_broadcasted = std::move(br.strides[1]);
    std::vector<uint64_t> res_divs = std::move(br.divisors);

    const int64_t max_rank = static_cast<int64_t>(out_shape.size());

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

    sycl_utils::SyclArray<uint64_t> res_divs_ptr(g_sycl_queue, res_divs,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> a_strides_ptr(g_sycl_queue,
        a_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> b_strides_ptr(g_sycl_queue,
        b_strides_broadcasted, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_ptr(g_sycl_queue, 1,
        MemoryLocation::HOST);

    const uint64_t* p_res_divs = res_divs_ptr;
    const uint64_t* p_a_strides = a_strides_ptr;
    const uint64_t* p_b_strides = b_strides_ptr;
    int32_t* p_error_flag = error_flag_ptr;

    *p_error_flag = 0;

    const value_t* p_a_data = get_data();
    const value_t* p_b_data = other.get_data();
    value_t* p_r_data = result.get_data();

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

            value_t a_val = p_a_data[offset_a];
            value_t b_val = p_b_data[offset_b];

            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(a_val), p_error_flag, 1);
            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(b_val), p_error_flag, 1);

            TEMPER_DEVICE_CHECK(b_val == static_cast<value_t>(0),
                p_error_flag, 3);

            value_t res = a_val / b_val;

            TEMPER_DEVICE_CHECK(!sycl_utils::is_finite(res), p_error_flag, 2);

            p_r_data[flat_idx] = res;
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(Tensor(operator/): NaN detected in inputs.)");

    TEMPER_CHECK(err == 2,
        nonfinite_error,
        R"(Tensor(operator/): non-finite result (overflow or Inf).)");

    TEMPER_CHECK(err == 3,
        computation_error,
        R"(Tensor(operator/): division by zero detected.)");

    return result;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::operator-() const
{
    const int64_t rank = this->get_rank();
    TEMPER_CHECK(rank == 0,
        validation_error,
        R"(Tensor(operator-): tensor has no elements.)");

    uint64_t total_size = this->get_num_elements();

    MemoryLocation res_loc = m_mem_loc;
    Tensor result(m_dimensions, res_loc);

    std::vector<uint64_t> divisors = utils::compute_divisors(m_dimensions);

    sycl_utils::SyclArray<uint64_t> divs_ptr(g_sycl_queue, divisors,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> strides_ptr(g_sycl_queue,
        this->get_strides(), MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_ptr(g_sycl_queue, 1,
        MemoryLocation::HOST);

    const uint64_t* p_divs = divs_ptr;
    const uint64_t* p_strides = strides_ptr;
    int32_t* p_error_flag = error_flag_ptr;

    *p_error_flag = 0;

    const value_t* p_src = get_data();
    value_t* p_dst = result.get_data();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t off = sycl_utils::idx_of(flat_idx, p_divs, p_strides, rank);
            value_t val = p_src[off];

            TEMPER_DEVICE_CHECK(sycl_utils::is_nan(val), p_error_flag, 1);

            p_dst[flat_idx] = -val;
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(Tensor(operator-): NaN detected in inputs.)");

    return result;
}

template<typename value_t>
bool Tensor<value_t>::operator==(const Tensor & other) const
{
    const int64_t first_rank = this->get_rank();
    const int64_t second_rank = other.get_rank();

    // Check if both are empty.
    if (first_rank == 0 && second_rank == 0)
    {
        return true;
    }

    const std::vector<uint64_t> & first_dims = this->get_dimensions();
    // Check if the shapes are the same.
    if (first_dims != other.get_dimensions())
    {
        return false;
    }

    std::vector<uint64_t> divisors = utils::compute_divisors(first_dims);

    sycl_utils::SyclArray<uint64_t> divs_ptr(g_sycl_queue, divisors,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> first_strides_ptr(g_sycl_queue,
        this->get_strides(), MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> second_strides_ptr(g_sycl_queue,
        other.get_strides(), MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> flag_ptr(g_sycl_queue, 1,
        MemoryLocation::HOST);

    const uint64_t* p_divs = divs_ptr;
    const uint64_t* p_first_strides = first_strides_ptr;
    const uint64_t* p_second_strides = second_strides_ptr;
    int32_t * p_flag = flag_ptr;

    *p_flag = 0;

    const value_t* p_first = this->get_data();
    const value_t* p_second = other.get_data();

    const uint64_t total_elements = this->get_num_elements();

    g_sycl_queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(total_elements)),
        [=](sycl::id<1> idx)
    {
        uint64_t linear = static_cast<uint64_t>(idx[0]);
        uint64_t first_offset = sycl_utils::idx_of(
            linear, p_divs, p_first_strides, first_rank);
        uint64_t second_offset = sycl_utils::idx_of(
            linear, p_divs, p_second_strides, second_rank);

        // If mismatch set flag (atomic store).
        TEMPER_DEVICE_CHECK(!(p_first[first_offset] == p_second[second_offset]),
            p_flag, 1);

    }).wait();

    const bool equal = (*p_flag == static_cast<int32_t>(0));
    return equal;
}

template <typename value_t>
Tensor<value_t> Tensor<value_t>::clone() const
{
    const int64_t rank = this->get_rank();
    TEMPER_CHECK(rank == 0,
        validation_error,
        R"(Tensor(clone): tensor has no elements.)");

    Tensor<value_t> result(m_dimensions, m_mem_loc);

    const uint64_t total_elements = this->get_num_elements();

    std::vector<uint64_t> shape_divs = utils::compute_divisors(m_dimensions);

    sycl_utils::SyclArray<uint64_t> shape_divs_ptr(g_sycl_queue, shape_divs,
        MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> src_strides_ptr(g_sycl_queue,
        this->get_strides(), MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> dest_strides_ptr(g_sycl_queue,
        result.get_strides(), MemoryLocation::DEVICE);

    const uint64_t* p_shape_divs = shape_divs_ptr;
    const uint64_t* p_src_strides = src_strides_ptr;
    const uint64_t* p_dest_strides = dest_strides_ptr;

    const value_t* p_src_data = get_data();
    value_t* p_dest_data = result.get_data();

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

    return result;
}

template<typename value_t>
void Tensor<value_t>::copy_from(const Tensor & src)
{
    const int64_t rank_dst = this->get_rank();
    const int64_t rank_src = src.get_rank();

    TEMPER_CHECK(rank_dst == 0,
        validation_error,
        R"(Tensor(copy_from): target tensor has no elements.)");

    TEMPER_CHECK(rank_src == 0,
        validation_error,
        R"(Tensor(copy_from): source tensor has no elements.)");

    const uint64_t total_dst = this->get_num_elements();

    TEMPER_CHECK(rank_src > rank_dst,
        validation_error,
        R"(Tensor(copy_from): source rank > destination rank.)");

    const std::vector<uint64_t> & src_shape = src.get_dimensions();
    const std::vector<uint64_t> & dst_shape = this->get_dimensions();
    const std::vector<uint64_t> & src_strides = src.get_strides();
    const std::vector<uint64_t> & dst_strides = this->get_strides();

    bool same_shape = (dst_shape == src_shape);
    if (same_shape)
    {
        std::vector<uint64_t> canon(rank_dst);
        if (rank_dst > 0)
        {
            canon[rank_dst - 1] = 1;
            for (int64_t i = static_cast<int64_t>(rank_dst) - 2; i >= 0; --i)
            {
                canon[i] = canon[i + 1] * dst_shape[i + 1];
            }
        }

        bool dst_contig = (dst_strides == canon);
        bool src_contig = (src_strides == canon);

        if (dst_contig && src_contig)
        {
            const size_t bytes =
                static_cast<size_t>(total_dst) * sizeof(value_t);
            g_sycl_queue.memcpy(get_data(), src.get_data(), bytes).wait();
            return;
        }
    }

    utils::TensorDesc a_desc{src_shape, src_strides};
    utils::TensorDesc b_desc{dst_shape, dst_strides};

    utils::BroadcastResult br = utils::compute_broadcast({a_desc, b_desc});

    TEMPER_CHECK(br.shape != dst_shape,
        validation_error,
        R"(Tensor(copy_from):
            source cannot be broadcast to destination shape.)");

    sycl_utils::SyclArray<uint64_t> res_divs_ptr(g_sycl_queue,
        br.divisors, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> a_strides_ptr(g_sycl_queue,
        br.strides[0], MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> b_strides_ptr(g_sycl_queue,
        br.strides[1], MemoryLocation::DEVICE);

    const uint64_t* p_res_divs = res_divs_ptr;
    const uint64_t* p_a_strides = a_strides_ptr;
    const uint64_t* p_b_strides = b_strides_ptr;

    const value_t* p_src_data = src.get_data();
    value_t* p_dst_data = get_data();

    g_sycl_queue.submit([&](sycl::handler & cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_dst)),
            [=](sycl::id<1> idx)
        {
            uint64_t flat_idx = static_cast<uint64_t>(idx[0]);

            uint64_t off_src = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_a_strides, rank_dst);
            uint64_t off_dst = sycl_utils::idx_of
                (flat_idx, p_res_divs, p_b_strides, rank_dst);

            value_t val = p_src_data[off_src];

            p_dst_data[off_dst] = val;
        });
    }).wait();
}

template<typename value_t>
void Tensor<value_t>::to(MemoryLocation target_loc)
{
    TEMPER_CHECK(m_dimensions.empty(),
        validation_error,
        R"(Tensor(to): tensor has no elements.)");

    TEMPER_CHECK(!m_own_data,
        validation_error,
        R"(Tensor(to): cannot move memory of a Tensor view (non-owning).)");

    if (m_mem_loc == target_loc)
    {
        return;
    }

    const uint64_t total_size = this->get_num_elements();

    value_t* raw_ptr = nullptr;
    if (target_loc == MemoryLocation::HOST)
    {
        raw_ptr = static_cast<value_t*>(
            sycl::malloc_shared(total_size * sizeof(value_t), g_sycl_queue));
    }
    else
    {
        raw_ptr = static_cast<value_t*>(
            sycl::malloc_device(total_size * sizeof(value_t), g_sycl_queue));
    }

    TEMPER_CHECK(!raw_ptr,
        device_error,
        R"(Tensor(to):
            error allocating tensor memory on device.)");

    std::shared_ptr<value_t> new_ptr = std::shared_ptr<value_t>(raw_ptr,
        [](value_t* p)
        {
            if (p)
            {
                sycl::free(p, g_sycl_queue);
            }
        }
    );

    g_sycl_queue.memcpy
        (new_ptr.get(), m_p_data.get(), total_size * sizeof(value_t)).wait();

    m_p_data = std::move(new_ptr);
    m_mem_loc = target_loc;
}

template<typename value_t>
void Tensor<value_t>::reshape(const std::vector<uint64_t>& new_dimensions)
{
    TEMPER_CHECK(new_dimensions.empty(),
        validation_error,
        R"(Tensor(reshape): new_dimensions cannot be empty.)");

    TEMPER_CHECK(!m_own_data,
        validation_error,
        R"(Tensor(reshape): cannot reshape an alias/view tensor.)");

    constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

    uint64_t og_total_size = 1;
    for (uint64_t dim : m_dimensions)
    {
        og_total_size *= dim;
    }

    uint64_t new_total_size = 1;
    for (uint64_t dim : new_dimensions)
    {
        TEMPER_CHECK(dim == 0,
            validation_error,
            R"(Tensor(reshape): new_dimensions cannot contain zero.)");

        TEMPER_CHECK(new_total_size > U64_MAX / dim,
            bounds_error,
            R"(Tensor(reshape): dimension product overflow.)");

        new_total_size *= dim;
    }

    TEMPER_CHECK(new_total_size != og_total_size,
        validation_error,
        R"(Tensor(reshape): total number of elements must remain the same.)");

    m_dimensions = new_dimensions;
    compute_strides();
}

template<typename value_t>
void Tensor<value_t>::sort(std::optional<int64_t> axis_opt)
{
    const int64_t rank = this->get_rank();

    if (rank == 0)
    {
        return;
    }

    const bool flatten = ! axis_opt.has_value();
    int64_t axis;

    if (! flatten)
    {
        axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }

        TEMPER_CHECK(axis < 0 || axis >= rank,
            bounds_error,
            "Tensor(sort): axis out of bounds");
    }

    uint64_t total_size = get_num_elements();

    uint64_t effective_axis_size = 0;
    uint64_t slice_count = 1;
    uint64_t axis_stride = 1;
    if (flatten)
    {
        effective_axis_size = total_size;
        slice_count = 1;
        axis_stride = 1;
    }
    else
    {
        slice_count = 1;
        for (int64_t i = 0; i < rank; ++i)
        {
            if (i == axis)
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

    std::vector<uint64_t> divisors = utils:: compute_divisors(m_dimensions);

    std::vector<uint64_t> host_slice_base(static_cast<size_t>(slice_count), 0);
    if (!flatten)
    {
        std::vector<uint64_t> dims;
        std::vector<uint64_t> strides_for_dims;
        for (int64_t i = 0; i < rank; ++i)
        {
            if (i == axis) continue;
            dims.push_back(m_dimensions[i]);
            strides_for_dims.push_back(m_strides[i]);
        }
        std::vector<uint64_t> index_factors = utils:: compute_divisors(dims);
        const uint64_t D = static_cast<uint64_t>(dims. size());

        for (uint64_t s = 0; s < slice_count; ++s)
        {
            host_slice_base[s] = sycl_utils::
                idx_of(s, index_factors. data(), strides_for_dims.data(), D);
        }
    }

    sycl_utils::SyclArray<uint64_t> slice_base_arr(g_sycl_queue,
        host_slice_base, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> divisors_arr(g_sycl_queue,
        divisors, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> strides_arr(g_sycl_queue,
        m_strides, MemoryLocation::DEVICE);

    const uint64_t* p_slice_base = slice_base_arr;
    const uint64_t* p_divisors = divisors_arr;
    const uint64_t* p_strides = strides_arr;

    value_t* tensor_data = m_p_data. get();

    sycl_utils::SyclArray<value_t> merge_buffer_arr(g_sycl_queue,
        total_size, m_mem_loc);
    value_t* merge_buffer = merge_buffer_arr;

    value_t* merge_input = tensor_data;
    value_t* merge_output = merge_buffer;

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
                const uint64_t local_id = it. get_local_id(0);
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
                    if (flatten)
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
                    if (flatten)
                    {
                        return sycl_utils::merge_path_partition<value_t>(
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
                            value_t va = merge_input[a_off];
                            value_t vb = merge_input[b_off];
                            bool cmp;
                            if (! sycl_utils::is_nan(va) &&
                                sycl_utils::is_nan(vb))
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
                    value_t va = merge_input[a_idx];
                    value_t vb = merge_input[b_idx];
                    uint64_t out_idx = idx_of_local(left + out_k);
                    if ((! sycl_utils::is_nan(va) && sycl_utils::is_nan(vb))
                        || va < vb)
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

        value_t* tmp = merge_input;
        merge_input = merge_output;
        merge_output = tmp;
    }

    if (merge_input != tensor_data)
    {
        g_sycl_queue.memcpy(tensor_data, merge_input,
            static_cast<size_t>(total_size) * sizeof(value_t)).wait();
    }
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::sum(std::optional<int64_t> axis_opt) const
{
    const int64_t rank = this->get_rank();
    const MemoryLocation res_loc = this->get_memory_location();

    if (rank == 0)
    {
        return Tensor<value_t>({1}, res_loc);
    }

    const bool flatten = ! axis_opt.has_value();
    int64_t axis;

    if (!flatten)
    {
        axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }

        TEMPER_CHECK(axis < 0 || axis >= rank,
            bounds_error,
            "Tensor(sum): axis out of bounds");
    }

    uint64_t axis_size = 0;
    std::vector<uint64_t> new_dimensions(m_dimensions);

    if (!flatten)
    {
        axis_size = m_dimensions[axis];
        new_dimensions[axis] = 1;
    }

    uint64_t output_size = 1;
    for (uint64_t d : new_dimensions)
    {
        output_size *= d;
    }

    uint64_t effective_axis_size = 0;
    uint64_t effective_output_size = 0;

    const uint64_t total_size = this->get_num_elements();
    if (flatten)
    {
        effective_axis_size = total_size;
        effective_output_size = 1;
    }
    else
    {
        effective_axis_size = axis_size;
        effective_output_size = output_size;
    }

    Tensor<value_t> result;
    if (flatten)
    {
        result = Tensor<value_t>({1}, res_loc);
    }
    else
    {
        result = Tensor<value_t>(new_dimensions, res_loc);
    }

    value_t* p_out = result.get_data();
    const value_t* p_src = get_data();

    std::vector<uint64_t> fixed_dims;
    fixed_dims.reserve(rank - 1);
    for (int64_t i = 0; i < rank; ++i)
    {
        if (!flatten && i == axis) continue;
        fixed_dims.push_back(m_dimensions[i]);
    }
    const uint64_t fixed_count = fixed_dims.size();

    std::vector<uint64_t> fixed_divisors;
    if (fixed_count > 0)
    {
        fixed_divisors = temper:: utils::compute_divisors(fixed_dims);
    }

    auto [workgroup_size, num_groups_per_slice] =
        utils::compute_wg_and_groups(g_sycl_queue,
            static_cast<size_t>(effective_axis_size));

    size_t total_groups =
        static_cast<size_t>(effective_output_size) * num_groups_per_slice;
    size_t total_group_items = total_groups * workgroup_size;
    if (total_group_items == 0) total_group_items = workgroup_size;

    size_t partial_count =
        static_cast<size_t>(effective_output_size) * num_groups_per_slice;
    size_t alloc_partial_count = 0;
    if (partial_count == 0)
    {
        alloc_partial_count = 1;
    }
    else
    {
        alloc_partial_count = partial_count;
    }

    const std::vector<uint64_t> divisors = utils::compute_divisors(m_dimensions);

    sycl_utils::SyclArray<uint64_t> strides_arr(g_sycl_queue,
        this->get_strides(), MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> divisors_arr(g_sycl_queue,
        divisors, MemoryLocation:: DEVICE);
    sycl_utils::SyclArray<value_t> partials_arr(g_sycl_queue,
        alloc_partial_count, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_arr(g_sycl_queue,
        1, MemoryLocation::HOST);

    const uint64_t* p_strides_dev = strides_arr;
    const uint64_t* p_divisors_dev = divisors_arr;
    value_t* p_partials = partials_arr;
    int32_t* p_error_flag = error_flag_arr;

    std::optional<sycl_utils::SyclArray<uint64_t>> fixed_divs_arr_opt;

    if (fixed_count > 0)
    {
        fixed_divs_arr_opt.emplace(g_sycl_queue,
            fixed_divisors, MemoryLocation::DEVICE);
    }
    const uint64_t* p_fixed_divs_dev = (fixed_count > 0)
        ? fixed_divs_arr_opt->data() : nullptr;

    g_sycl_queue.memset
        (p_partials, 0, sizeof(value_t) * alloc_partial_count).wait();

    *p_error_flag = 0;

    // First kernel: compute partial sums.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh. parallel_for(
            sycl::nd_range<1>(sycl::range<1>(total_group_items),
                              sycl::range<1>(workgroup_size)),
            [=](sycl::nd_item<1> it)
        {
            const size_t global_group_id = it. get_group(0);
            const size_t slice = global_group_id / num_groups_per_slice;
            const size_t group_in_slice = global_group_id % num_groups_per_slice;
            const size_t local_id = it.get_local_id(0);

            value_t local_sum = value_t{};

            if (flatten)
            {
                size_t start = group_in_slice * workgroup_size + local_id;
                size_t stride = workgroup_size * num_groups_per_slice;
                const size_t N = static_cast<size_t>(effective_axis_size);
                for (size_t linear = start; linear < N; linear += stride)
                {
                    uint64_t offset = sycl_utils::idx_of(
                        static_cast<uint64_t>(linear),
                        p_divisors_dev, p_strides_dev, rank);

                    value_t v = p_src[offset];
                    TEMPER_DEVICE_CHECK(sycl_utils::is_nan(v), p_error_flag, 1);
                    local_sum += v;
                }
            }
            else
            {
                uint64_t remaining = slice;
                uint64_t base_offset = 0;
                uint64_t counter = 0;
                for (int64_t i = 0; i < rank; ++i)
                {
                    if (i == axis)
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
                        p_strides_dev[axis];
                    value_t v = p_src[offs];
                    TEMPER_DEVICE_CHECK(sycl_utils:: is_nan(v), p_error_flag, 1);
                    local_sum += v;
                }
            }

            auto group = it.get_group();
            value_t group_sum = sycl::reduce_over_group
                (group, local_sum, sycl:: plus<value_t>());

            TEMPER_DEVICE_CHECK(!sycl_utils:: is_finite(group_sum), p_error_flag, 2);

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
        if (num_groups_per_slice > candidate_min)
        {
            candidate_min = num_groups_per_slice;
        }
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
            const size_t local_range = it. get_local_range(0);

            value_t v = value_t{};
            for (size_t idx = lid; idx < num_groups_per_slice; idx += local_range)
            {
                value_t pv = p_partials[slice * num_groups_per_slice + idx];
                TEMPER_DEVICE_CHECK(sycl_utils::is_nan(pv), p_error_flag, 1);
                v += pv;
            }

            auto group = it.get_group();
            value_t total =
                sycl::reduce_over_group(group, v, sycl::plus<value_t>());
            TEMPER_DEVICE_CHECK(!sycl_utils::is_finite(total), p_error_flag, 2);

            if (lid == 0)
            {
                p_out[slice] = total;
            }
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(Tensor(sum): NaN detected in inputs.)");

    TEMPER_CHECK(err == 2,
        nonfinite_error,
        R"(Tensor(sum): non-finite result detected.)");

    return result;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::cumsum(std::optional<int64_t> axis_opt) const
{
    const int64_t rank = this->get_rank();
    const MemoryLocation res_loc = this->get_memory_location();

    if (rank == 0)
    {
        return Tensor<value_t>({1}, res_loc);
    }

    const bool flatten = ! axis_opt.has_value();
    int64_t axis;

    if (! flatten)
    {
        axis = axis_opt.value();
        if (axis < 0)
        {
            axis += rank;
        }

        TEMPER_CHECK(axis < 0 || axis >= rank,
            bounds_error,
            "Tensor(cumsum): axis out of bounds");
    }

    const uint64_t total_size = this->get_num_elements();
    uint64_t axis_size = 0;
    if (flatten)
    {
        axis_size = total_size;
    }
    else
    {
        axis_size = m_dimensions[axis];
    }

    std::vector<uint64_t> out_dims;
    if (flatten)
    {
        out_dims = std::vector<uint64_t>{ total_size };
    }
    else
    {
        out_dims = m_dimensions;
    }

    Tensor<value_t> result(out_dims, m_mem_loc);

    const value_t* p_src = get_data();
    value_t* p_out = result. get_data();
    std::vector<uint64_t> divisors = utils::compute_divisors(m_dimensions);

    std::vector<uint64_t> fixed_divisors;
    std::vector<uint64_t> fixed_strides;
    std::vector<uint64_t> fixed_out_strides;
    uint64_t fixed_count = 0;

    if (!  flatten)
    {
        std::vector<uint64_t> fixed_dims;
        fixed_dims.reserve(rank - 1);
        for (int64_t i = 0; i < rank; ++i)
        {
            if (i == axis) continue;
            fixed_dims. push_back(m_dimensions[i]);
        }
        fixed_count = static_cast<uint64_t>(fixed_dims.size());
        fixed_divisors = temper:: utils::compute_divisors(fixed_dims);

        fixed_strides.reserve(fixed_count);
        fixed_out_strides.reserve(fixed_count);
        for (int64_t d = 0; d < rank; ++d)
        {
            if (d == axis) continue;
            fixed_strides.push_back(m_strides[d]);
            fixed_out_strides.push_back(result.m_strides[d]);
        }
    }

    uint64_t effective_axis_size;
    if (flatten)
    {
        effective_axis_size = total_size;
    }
    else
    {
        effective_axis_size = axis_size;
    }

    uint64_t effective_output_size = 0;
    if (flatten)
    {
        effective_output_size = 1;
    }
    else
    {
        uint64_t out_sz = 1;
        for (int64_t i = 0; i < rank; ++i)
        {
            if (i == axis) continue;
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

    sycl_utils::SyclArray<uint64_t> divs_arr(g_sycl_queue,
        divisors, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<uint64_t> strides_arr(g_sycl_queue,
        m_strides, MemoryLocation::DEVICE);
    sycl_utils::SyclArray<int32_t> error_flag_arr(g_sycl_queue,
        1, MemoryLocation::HOST);
    sycl_utils::SyclArray<value_t> block_partials_arr(g_sycl_queue,
        total_groups, MemoryLocation::DEVICE);

    const uint64_t* p_divs_dev = divs_arr;
    const uint64_t* p_strides_dev = strides_arr;
    int32_t* p_error_flag = error_flag_arr;
    value_t* p_block_partials = block_partials_arr;

    std::optional<sycl_utils::SyclArray<uint64_t>> out_strides_arr_opt;
    if (!  flatten)
    {
        out_strides_arr_opt. emplace(g_sycl_queue,
            result.m_strides, MemoryLocation::DEVICE);
    }
    const uint64_t* p_out_strides_dev = (! flatten)
        ? out_strides_arr_opt->data() : nullptr;

    std::optional<sycl_utils::SyclArray<uint64_t>> fixed_divs_arr_opt;
    std::optional<sycl_utils::SyclArray<uint64_t>> fixed_strides_arr_opt;
    std::optional<sycl_utils::SyclArray<uint64_t>> fixed_out_strides_arr_opt;

    if (fixed_count > 0)
    {
        fixed_divs_arr_opt.emplace(g_sycl_queue,
            fixed_divisors, MemoryLocation::DEVICE);
        fixed_strides_arr_opt.emplace(g_sycl_queue,
            fixed_strides, MemoryLocation:: DEVICE);
        fixed_out_strides_arr_opt. emplace(g_sycl_queue,
            fixed_out_strides, MemoryLocation::DEVICE);
    }

    const uint64_t* p_fixed_divs_dev = (fixed_count > 0)
        ? fixed_divs_arr_opt->data() : nullptr;
    const uint64_t* p_fixed_strides_dev = (fixed_count > 0)
        ? fixed_strides_arr_opt->data() : nullptr;
    const uint64_t* p_fixed_out_strides_dev = (fixed_count > 0)
        ? fixed_out_strides_arr_opt->data() : nullptr;

    *p_error_flag = 0;
    if (total_groups > 0)
        g_sycl_queue. memset
            (p_block_partials, 0, sizeof(value_t) * total_groups).wait();

    // First kernel: per-group inclusive scan + write block partial.
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh. parallel_for(
            sycl::nd_range<1>(sycl::range<1>(total_group_items),
                              sycl::range<1>(workgroup_size)),
            [=](sycl::nd_item<1> it)
        {
            const size_t global_group_id = it. get_group(0);
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
                if (!  flatten)
                {
                    base_dst = sycl_utils::idx_of(slice,
                        p_fixed_divs_dev,
                        p_fixed_out_strides_dev,
                        static_cast<uint64_t>(fixed_count));
                }
            }

            uint64_t src_off = 0, dst_off = 0;
            if (flatten)
            {
                uint64_t linear = static_cast<uint64_t>(index_in_slice);
                src_off = sycl_utils::idx_of
                    (linear, p_divs_dev, p_strides_dev, rank);
                dst_off = static_cast<uint64_t>(index_in_slice);
            }
            else
            {
                src_off = base_src +
                static_cast<uint64_t>(index_in_slice) * p_strides_dev[axis];
                dst_off = base_dst + static_cast<uint64_t>(index_in_slice) *
                    p_out_strides_dev[axis];
            }

            value_t x = value_t{0};
            if (active)
            {
                x = p_src[src_off];
                TEMPER_DEVICE_CHECK(sycl_utils::is_nan(x), p_error_flag, 1);
            }

            value_t prefix = sycl:: inclusive_scan_over_group
                (it.get_group(), x, sycl::plus<value_t>());

            if (active)
            {
                p_out[dst_off] = prefix;
                TEMPER_DEVICE_CHECK(! sycl_utils::is_finite(prefix),
                    p_error_flag, 2);
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
            value_t running = value_t{0};
            size_t base = slice * num_groups_per_slice;
            for (size_t g = 0; g < num_groups_per_slice; ++g)
            {
                value_t v = p_block_partials[base + g];
                running += v;
                p_block_partials[base + g] = running;
            }
        });
    }).wait();

     // Third kernel: add previous-block prefix to each element (groups > 0).
    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh. parallel_for(
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
            if (flatten)
            {
                dst_off = static_cast<uint64_t>(index_in_slice);
            }
            else
            {
                dst_off = base_dst + static_cast<uint64_t>(index_in_slice) *
                    p_out_strides_dev[axis];
            }

            value_t add = p_block_partials
                [slice * num_groups_per_slice + (group_in_slice - 1)];
            p_out[dst_off] += add;
            TEMPER_DEVICE_CHECK(!sycl_utils::is_finite(p_out[dst_off]),
                p_error_flag, 2);
        });
    }).wait();

    int32_t err = *p_error_flag;

    TEMPER_CHECK(err == 1,
        nan_error,
        R"(Tensor(cumsum): NaN detected in inputs.)");

    TEMPER_CHECK(err == 2,
        nonfinite_error,
        R"(Tensor(cumsum): non-finite result detected.)");

    return result;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::transpose() const
{
    const int64_t rank = this->get_rank();

    TEMPER_CHECK(rank == 0,
        validation_error,
        R"(Tensor(transpose): cannot transpose an empty tensor.)");

    std::vector<int64_t> axes(rank);
    for (int64_t i = 0; i < rank; ++i)
    {
        axes[i] = rank - 1 - i;
    }

    return transpose(axes);
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::transpose(const std::vector<int64_t> & axes) const
{
    const int64_t rank = this->get_rank();

    TEMPER_CHECK(axes.size() != static_cast<uint64_t>(rank),
        validation_error,
        R"(Tensor(transpose):
            axes vector must have same length as tensor rank.)");

    std::vector<int64_t> new_axes(rank);

    std::vector<bool> seen(rank, false);
    for (uint64_t i = 0; i < axes.size(); ++i)
    {
        int64_t axis = axes[i];
        if (axis < 0)
        {
            axis += rank;
        }

        TEMPER_CHECK(axis < 0 || axis >= rank || seen[axis],
            bounds_error,
            R"(Tensor(transpose):
                axes must be a permutation of [-rank..rank-1].)");
        seen[axis] = true;
        new_axes[i] = axis;
    }

    std::vector<uint64_t> new_dims(rank);
    std::vector<uint64_t> new_strides(rank);
    for (int64_t i = 0; i < rank; ++i)
    {
        new_dims[i] = m_dimensions[new_axes[i]];
        new_strides[i] = m_strides[new_axes[i]];
    }

    std::vector<uint64_t> start_indices(rank, 0);

    return Tensor(*this, start_indices, new_dims, new_strides);
}

template<typename value_t>
void Tensor<value_t>::print(std::ostream& os) const
{
    std::function<void(uint64_t, uint64_t)> recurse
        = [&](uint64_t dim, uint64_t offset)
    {
        if (dim == m_dimensions.size() - 1)
        {
            os << "[";
            for (uint64_t i = 0; i < m_dimensions[dim]; ++i)
            {
                value_t val;

                uint64_t current_offset = offset + i * m_strides[dim];
                g_sycl_queue.memcpy(&val, m_p_data.get() + current_offset,
                    sizeof(value_t)).wait();

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

template<typename value_t>
void Tensor<value_t>::print_shape(std::ostream& os) const
{
    os << "[";
    for (size_t i = 0; i < m_dimensions.size(); ++i)
    {
        if (i > 0) os << ", ";
        os << m_dimensions[i];
    }
    os << "]\n";
}

template<typename value_t>
const value_t * Tensor<value_t>::get_data() const noexcept
{
    return m_p_data.get();
}

template<typename value_t>
value_t * Tensor<value_t>::get_data() noexcept
{
    return m_p_data.get();
}

template<typename value_t>
const std::vector<uint64_t> & Tensor<value_t>::get_dimensions() const noexcept
{
    return m_dimensions;
}

template<typename value_t>
const std::vector<uint64_t> & Tensor<value_t>::get_strides() const noexcept
{
    return m_strides;
}

template<typename value_t>
const std::vector<uint64_t> & Tensor<value_t>::get_shape() const noexcept
{
    return m_dimensions;
}

template<typename value_t>
int64_t Tensor<value_t>::get_rank() const noexcept
{
    return static_cast<int64_t>(m_dimensions.size());
}

template<typename value_t>
uint64_t Tensor<value_t>::get_num_elements() const noexcept
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

template<typename value_t>
MemoryLocation Tensor<value_t>::get_memory_location() const noexcept
{
    return m_mem_loc;
}

template<typename value_t>
bool Tensor<value_t>::get_owns_data() const noexcept
{
    return m_own_data;
}

template<typename value_t>
bool Tensor<value_t>::is_view() const noexcept
{
    return !m_own_data;
}

template<typename value_t>
uint64_t Tensor<value_t>::get_element_size_bytes() const noexcept
{
    return static_cast<uint64_t>(sizeof(value_t));
}

template<typename value_t>
uint64_t Tensor<value_t>::get_total_bytes() const noexcept
{
    const uint64_t elems = get_num_elements();
    const uint64_t elem_size = get_element_size_bytes();

    return elems * elem_size;
}

template<typename value_t>
std::vector<uint64_t> Tensor<value_t>::index_to_coords(uint64_t flat) const
{
    const int64_t rank = this->get_rank();
    if (rank == 0)
        return {};

    const uint64_t total = this->get_num_elements();

    TEMPER_CHECK(flat >= total,
        bounds_error,
        "Tensor::index_to_coords: flat out of range");

    std::vector<uint64_t> divs = utils::compute_divisors(m_dimensions);
    std::vector<uint64_t> coords(rank, 0);

    for (int64_t d = 0; d < rank; ++d)
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

template<typename value_t>
uint64_t Tensor<value_t>::coords_to_index
    (const std::vector<uint64_t>& coords) const
{
    const int64_t rank = this->get_rank();

    TEMPER_CHECK(coords.size() != static_cast<uint64_t>(rank),
        validation_error,
        R"(Tensor(coords_to_index): size mismatch)");

    if (rank == 0)
    {
        return 0;
    }

    std::vector<uint64_t> divs = temper::utils::compute_divisors(m_dimensions);
    uint64_t flat = 0;

    const std::vector<uint64_t> & dimensions = this->get_dimensions();

    for (int64_t d = 0; d < rank; ++d)
    {
        TEMPER_CHECK(coords[d] >= dimensions[d],
            bounds_error,
            R"(Tensor(coords_to_index): coord out of range)");

        flat += coords[d] * divs[d];
    }

    return flat;
}

template<typename value_t>
Tensor<value_t> Tensor<value_t>::at(uint64_t flat)
{
    const int64_t rank = this->get_rank();

    TEMPER_CHECK(rank == 0,
        validation_error,
        "Tensor(at): tensor has no elements.");

    const uint64_t total = this->get_num_elements();

    TEMPER_CHECK(flat >= total,
        bounds_error,
        "Tensor(at): flat index out of range.");

    std::vector<uint64_t> coords = this->index_to_coords(flat);
    return Tensor(*this, coords, std::vector<uint64_t>{1});
}

template<typename value_t>
const Tensor<value_t> Tensor<value_t>::at(uint64_t flat) const
{
    const int64_t rank = this->get_rank();

    TEMPER_CHECK(rank == 0,
        validation_error,
        "Tensor(at): tensor has no elements.");

    const uint64_t total = this->get_num_elements();

    TEMPER_CHECK(flat >= total,
        bounds_error,
        "Tensor(at): flat index out of range.");

    std::vector<uint64_t> coords = this->index_to_coords(flat);
    return Tensor(*this, coords, std::vector<uint64_t>{1});
}

template<typename value_t>
Tensor<value_t>::iterator Tensor<value_t>::begin() noexcept
{
    return iterator(this, 0);
}

template<typename value_t>
Tensor<value_t>::iterator Tensor<value_t>::end() noexcept
{
    return iterator(this, get_num_elements());
}

template<typename value_t>
Tensor<value_t>::const_iterator Tensor<value_t>::begin() const noexcept
{
    return const_iterator(this, 0);
}

template<typename value_t>
Tensor<value_t>::const_iterator Tensor<value_t>::end() const noexcept
{
    return const_iterator(this, get_num_elements());
}

template<typename value_t>
Tensor<value_t>::const_iterator Tensor<value_t>::cbegin() const noexcept
{
    return begin();
}

template<typename value_t>
Tensor<value_t>::const_iterator Tensor<value_t>::cend() const noexcept
{
    return end();
}

template class Tensor<float>;
template class Tensor<uint64_t>;

} // namespace temper