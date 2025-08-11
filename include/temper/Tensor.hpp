/**
 * @file Tensor.hpp
 * @brief Declaration of the Tensor data structure.
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include "SYCLQueue.hpp"

namespace temper {

/**
 * @brief Supported device types for Tensor storage.
 */
enum class MemoryLocation {
    HOST,   ///< Host memory
    DEVICE  ///< Device memory (via SYCL)
};

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

    /// Member pointer to data.
	std::shared_ptr<float_t> m_p_data;

    /// Member dimensions for each axis.
    std::vector<uint64_t>    m_dimensions;

    /// Member strides for each axis.
    std::vector<uint64_t>    m_strides;

    /// Member boolean for data ownership; true if it's owned by tensor
    bool                     m_own_data;

    /// Member enumeration to indicate if data is on host or device.
    MemoryLocation           m_mem_loc;

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
     * @param loc Where data memory should reside.
     */
    Tensor(const std::vector<uint64_t>& dimensions,
        MemoryLocation loc = MemoryLocation::DEVICE);

    /**
     * @brief Copy constructor.
     *
     * Performs a deep copy of data and metadata.
     *
     * @param other The tensor to copy from.
     */
    Tensor(const Tensor & other);

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of data and metadata.
     *
     * @param other The tensor to move from.
     */
    Tensor(Tensor && other) noexcept;

    /**
     * @brief Constructs a non-owning view into another Tensor.
     * View can live beyond owner's lifespan.
     *
     * @param other Tensor to view into (must outlive the view).
     * @param start_indices Starting coordinate for the view (per axis).
     * @param new_dims Shape of the view.
     * @throws std::invalid_argument on rank mismatch or bounds error.
     */
    Tensor(Tensor & other,
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
    Tensor& operator=(const Tensor & other);

    /**
     * @brief Move assignment operator.
     *
     * Transfers ownership of data and metadata.
     *
     * @param other The tensor to move from.
     * @return Reference to this tensor.
     */
    Tensor& operator=(Tensor && other) noexcept;

    /**
     * @brief Assignment from flat std::vector.
     *
     * Size must match total size of the tensor.
     *
     * @param values The flat vector of values.
     * @return Reference to this tensor.
     */
    Tensor& operator=(const std::vector<float_t> & values);

    /**
     * @brief Assigns a scalar value to this tensor.
     *
     * If the tensor was default-constructed (no dimensions), it is
     * automatically initialized as a scalar tensor with shape {1}
     * and memory allocated according to its memory location.
     *
     * If the tensor already has dimensions, it must contain exactly
     * one element or an exception is thrown.
     *
     * @param val Scalar value to assign.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If tensor size is not exactly 1.
     */
    Tensor& operator=(float_t val);

    /**
     * @brief Returns a non-owning view into a sub-tensor at the given index.
     *
     * Applies the index to the first dimension:
     * - Rank 1: view of a single element (shape {1})
     * - Rank > 1: view of the remaining dimensions (axis 0 dropped)
     *
     * @param idx Index along the first dimension.
     * @return Tensor view (non-owning) into the selected region.
     * @throws std::out_of_range If index is out of bounds or tensor
     * has no dimensions.
     */
    Tensor operator[](uint64_t idx);

    /**
     * @brief Const version of operator[] returning a non-owning view.
     *
     * Same behavior as the non-const version. Uses const_cast internally
     * but does not modify the original tensor.
     *
     * @param idx Index along the first dimension.
     * @return Tensor view (non-owning) into the selected region.
     * @throws std::out_of_range If index is out of bounds or tensor
     * has no dimensions.
     */
    Tensor operator[](uint64_t idx) const;

    /**
     * @brief Converts a scalar tensor to its underlying value.
     *
     * Copies the single element from host or device memory.
     * The tensor must contain exactly one element.
     *
     * @return Scalar value stored in this tensor.
     * @throws std::invalid_argument If tensor size is not exactly 1.
     */
    operator float_t() const;

    /**
     * @brief Moves tensor data between host (shared) and device memory.
     *
     * Transfers owned data to the specified memory location.
     *
     * @param target_loc Target memory location (HOST or DEVICE).
     * @throws std::runtime_error if called on a non-owning tensor (view).
     */
    void to(MemoryLocation target_loc);

    /**
     * @brief Prints the tensor elements to the provided output
     * stream in a nested format.
     *
     * The tensor elements are printed recursively as nested arrays
     * reflecting the tensor's shape.
     * This function handles tensors stored in host or
     * device memory transparently.
     *
     * Example output for a 2x2 tensor:
     * @code
     * [[1.0, 2.0],
     *  [3.0, 4.0]]
     * @endcode
     *
     * @param os The output stream to print to. Defaults to std::cout.
     */
    void print(std::ostream & os = std::cout) const;

	/**
     * @brief Tensor class destructor.
     *
     * Frees owned device memory.
     */
	~Tensor() noexcept = default;

};

/// Explicit instantiation for "float" data
extern template class Tensor<float>;

} // namespace temper

#endif // TENSOR_HPP