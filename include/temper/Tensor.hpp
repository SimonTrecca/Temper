/**
 * @file Tensor.hpp
 * @brief Declaration of the Tensor data structure.
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstdint>
#include <stdexcept>
#include "SYCLQueue.hpp"

/**
 * @brief Class template for the Tensor data structure.
 * @tparam float_t Floating point adjacent numeric types (float, double, etc.).
 *
 * The class manages a linear buffer in row-major order.
 */
template <typename float_t>
class Tensor
{

private:

	float_t*               m_p_data;      ///< Member pointer to data.
    std::vector<uint64_t>  m_dimensions;  ///< Member dimensions for each axis.
    std::vector<uint64_t>  m_strides;     ///< Member strides for each axis.
    bool                   m_own_data;    ///< Member boolean for data ownership.

    /**
     * @brief Computes strides using dimensions.
     *
     * This method resizes the `m_strides` vector
     * and fills each element so that
     * m_strides[i] = product of all dimensions to the right of i.
     */
    void compute_strides();

public:

    Tensor() = default;

    /**
     * @brief Constructs a Tensor with the given dimensions.
     *
     * Initializes the Tensor's data buffer with zeros
     * based on the specified dimensions.
     *
     * @param dimensions The size of the tensor along each dimension.
     */
    Tensor(const std::vector<uint64_t>& dimensions);

    /**
     * @brief Copy constructor.
     *
     * Performs a deep copy of data and metadata.
     *
     * @param other The tensor to copy from.
     */
    Tensor(const Tensor& other);

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of data and metadata.
     *
     * @param other The tensor to move from.
     */
    Tensor(Tensor&& other) noexcept;

    /**
     * @brief Constructs a non-owning view into another Tensor.
     *
     * @param other Tensor to view into (must outlive the view).
     * @param start_indices Starting coordinate for the view (per axis).
     * @param new_dims Shape of the view.
     * @throws std::invalid_argument on rank mismatch or bounds error.
     */
    Tensor(Tensor& other,
           const std::vector<uint64_t>& start_indices,
           const std::vector<uint64_t>& view_shape);


    /**
     * @brief Copy assignment operator.
     *
     * Performs a deep copy of data and metadata.
     *
     * @param other The tensor to assign from.
     * @return Reference to this tensor.
     */
    Tensor& operator=(const Tensor& other);

    /**
     * @brief Move assignment operator.
     *
     * Transfers ownership of data and metadata.
     *
     * @param other The tensor to move from.
     * @return Reference to this tensor.
     */
    Tensor& operator=(Tensor&& other) noexcept;

    /**
     * @brief Assignment from flat std::vector.
     *
     * Size must match total size of the tensor.
     *
     * @param values The flat vector of values.
     * @return Reference to this tensor.
     */
    Tensor& operator=(const std::vector<float_t>& values);

	/**
     * @brief Tensor class destructor.
     *
     * Frees owned device memory.
     */
	~Tensor() noexcept;

};

/// Explicit instantiation for "float" data
extern template class Tensor<float>;

#endif // TENSOR_HPP