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

    if (!m_dimensions.empty())
    {
        m_strides.back() = 1;
        for (uint64_t i = m_dimensions.size() - 1; i > 0; --i)
        {
            m_strides[i - 1] = m_strides[i] * m_dimensions[i];
        }
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
    compute_strides();

    uint64_t total_size = 1;
    for (uint64_t d : dimensions)
    {
        total_size *= d;
    }

    float_t* raw_ptr = nullptr;
    if (m_mem_loc == MemoryLocation::HOST)
    {
        raw_ptr = static_cast<float_t*>(
            sycl::malloc_shared(total_size * sizeof(float_t), g_sycl_queue));
    }
    else
    {
        raw_ptr = static_cast<float_t*>(
            sycl::malloc_device(total_size * sizeof(float_t), g_sycl_queue));
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

    g_sycl_queue.memset(m_p_data.get(), 0, sizeof(float_t) * total_size).wait();
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
        uint64_t total_size = 1;
        for (uint64_t d : m_dimensions)
        {
            total_size *= d;
        }

        float_t* raw_ptr = nullptr;
        if (m_mem_loc == MemoryLocation::HOST)
        {
            raw_ptr = static_cast<float_t*>(
                sycl::malloc_shared(total_size * sizeof(float_t), g_sycl_queue));
        }
        else
        {
            raw_ptr = static_cast<float_t*>(
                sycl::malloc_device(total_size * sizeof(float_t), g_sycl_queue));
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
                            sizeof(float_t) * total_size).wait();
    }
    else
    {
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

    if (start_indices.size() != original_rank)
    {
        throw std::invalid_argument("Start_indices must match tensor rank.");
    }

    if (view_rank == 0 || view_rank > original_rank)
    {
        throw std::invalid_argument
            ("View shape rank must be between 1 and tensor rank.");
    }

    for (uint64_t i = 0; i < original_rank; ++i)
    {
        uint64_t start = start_indices[i];
        uint64_t dim = other.m_dimensions[i];
        if (start >= dim)
        {
            throw std::out_of_range("Start index out of bounds.");
        }
    }
    for (uint64_t j = 0; j < view_rank; ++j)
    {
        uint64_t i = original_rank - view_rank + j;
        uint64_t len = view_shape[j];
        uint64_t dim = other.m_dimensions[i];
        if (len == 0 || start_indices[i] + len > dim)
        {
            throw std::out_of_range("View shape out of bounds.");
        }
    }

    uint64_t offset = 0;
    for (uint64_t i = 0; i < original_rank; ++i)
    {
        offset += start_indices[i] * other.m_strides[i];
    }

    m_p_data = std::shared_ptr<float_t>
        (other.m_p_data, other.m_p_data.get() + offset);

    m_dimensions.resize(view_rank);
    m_strides.resize(view_rank);
    for (uint64_t j = 0; j < view_rank; ++j)
    {
        uint64_t i = original_rank - view_rank + j;
        m_dimensions[j] = view_shape[j];
        m_strides[j] = other.m_strides[i];
    }
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
            uint64_t total_size = 1;
            for (uint64_t d : m_dimensions)
            {
                total_size *= d;
            }

            float_t* raw_ptr = nullptr;
            if (m_mem_loc == MemoryLocation::HOST)
            {
                raw_ptr = static_cast<float_t*>(sycl::malloc_shared
                    (total_size * sizeof(float_t), g_sycl_queue));
            }
            else
            {
                raw_ptr = static_cast<float_t*>(sycl::malloc_device
                    (total_size * sizeof(float_t), g_sycl_queue));
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
                                sizeof(float_t) * total_size).wait();
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
        other.m_own_data = true;
    }
    return *this;
}


template<typename float_t>
Tensor<float_t> & Tensor<float_t>::operator=(const std::vector<float_t> & values)
{
    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        total_size *= d;
    }

    if (values.size() != total_size)
    {
        throw std::invalid_argument("Size mismatch in 1D vector assignment.");
    }

    g_sycl_queue.memcpy
        (m_p_data.get(), values.data(), sizeof(float_t) * values.size()).wait();

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
    for (uint64_t d : m_dimensions) total_size *= d;

    if (total_size != 1)
    {
        throw std::invalid_argument(
            "Scalar assignment only allowed for tensors with single element.");
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
        throw std::out_of_range("Tensor has no dimensions.");
    }
    if (idx >= m_dimensions[0])
    {
        throw std::out_of_range("Index out of bounds (operator[]).");
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
        throw std::out_of_range("Tensor has no dimensions.");
    }
    if (idx >= m_dimensions[0])
    {
        throw std::out_of_range("Index out of bounds (operator[] const).");
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
    uint64_t total_size = 1;
    for (uint64_t d : m_dimensions)
    {
        total_size *= d;
    }

    if (total_size != 1)
    {
        throw std::invalid_argument
            ("Scalar read only allowed for tensors with single element.");
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
        throw std::invalid_argument("Rank-0 tensors not supported.");
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
            throw std::invalid_argument("Incompatible shapes for broadcasting.");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        if (dim == 0)
        {
            throw std::invalid_argument("Zero-sized axis not supported.");
        }
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
            throw std::runtime_error("NaN detected in inputs.");
        }
        if (err == 2)
        {
            throw std::runtime_error("Non-finite result (overflow or Inf).");
        }
        throw std::runtime_error("Numeric error during element-wise addition.");
    }

    return result;
}

// ---------- operator- (broadcasting) ----------
template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator-(const Tensor & other) const
{
    if (m_dimensions.empty() || other.m_dimensions.empty())
    {
        throw std::invalid_argument("Rank-0 tensors not supported.");
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
            throw std::invalid_argument("Incompatible shapes for broadcasting.");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        if (dim == 0)
        {
            throw std::invalid_argument("Zero-sized axis not supported.");
        }
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
            throw std::runtime_error("NaN detected in inputs.");
        }
        if (err == 2)
        {
            throw std::runtime_error("Non-finite result (overflow or Inf).");
        }
        throw std::runtime_error("Numeric error during element-wise addition.");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator*(const Tensor & other) const
{
    if (m_dimensions.empty() || other.m_dimensions.empty())
    {
        throw std::invalid_argument("Rank-0 tensors not supported.");
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
            throw std::invalid_argument("Incompatible shapes for broadcasting.");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        if (dim == 0)
        {
            throw std::invalid_argument("Zero-sized axis not supported.");
        }
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
            throw std::runtime_error("NaN detected in inputs.");
        }
        if (err == 2)
        {
            throw std::runtime_error("Non-finite result (overflow or Inf).");
        }
        throw std::runtime_error("Numeric error during element-wise addition.");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator/(const Tensor & other) const
{
    if (m_dimensions.empty() || other.m_dimensions.empty())
    {
        throw std::invalid_argument("Rank-0 tensors not supported.");
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
            throw std::invalid_argument("Incompatible shapes for broadcasting.");
        }
    }

    uint64_t total_size = 1;
    for (uint64_t dim : out_shape)
    {
        if (dim == 0)
        {
            throw std::invalid_argument("Zero-sized axis not supported.");
        }
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

    /**
     * Shared error flag:
     * 0 = OK,
     * 1 = NaN in inputs,
     * 2 = division by zero,
     * 3 = non-finite result
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
            throw std::runtime_error("NaN detected in inputs.");
        }
        if (err == 2)
        {
            throw std::runtime_error("Division by zero detected.");
        }
        if (err == 3)
        {
            throw std::runtime_error("Non-finite result detected.");
        }
        throw std::runtime_error("Numeric error during element-wise division.");
    }

    return result;
}

template<typename float_t>
Tensor<float_t> Tensor<float_t>::operator-() const
{
    if (m_dimensions.empty())
    {
        throw std::invalid_argument("Rank-0 tensors not supported.");
    }

    const uint64_t rank = static_cast<uint64_t>(m_dimensions.size());

    std::vector<uint64_t> shape_aligned(rank);
    std::vector<uint64_t> strides_aligned(rank);

    for (uint64_t i = 0; i < rank; ++i)
    {
        if (m_dimensions[i] == 0)
        {
            throw std::invalid_argument("Zero-sized axis not supported.");
        }
        shape_aligned[i] = m_dimensions[i];
        strides_aligned[i] = m_strides[i];
    }

    uint64_t total_size = 1;
    for (auto dim : shape_aligned)
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
        (p_shape, shape_aligned.data(), sizeof(uint64_t) * rank).wait();
    g_sycl_queue.memcpy
        (p_strides, strides_aligned.data(), sizeof(uint64_t) * rank).wait();

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
        throw std::runtime_error("NaN detected in input.");
    }

    return result;
}

template<typename float_t>
void Tensor<float_t>::to(MemoryLocation target_loc)
{
    if (!m_own_data)
    {
        throw std::runtime_error
            ("Cannot move memory of a Tensor view (non-owning).");
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