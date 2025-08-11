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