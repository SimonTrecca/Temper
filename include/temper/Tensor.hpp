/**
 * @file Tensor.hpp
 * @brief Declaration of the Tensor data structure.
 */

#ifndef TEMPER_TENSOR_HPP
#define TEMPER_TENSOR_HPP

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <limits>
#include "SYCLQueue.hpp"

namespace temper
{

/**
 * @brief Supported device types for Tensor storage.
 */
enum class MemoryLocation
{
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
	std::shared_ptr<float_t> m_p_data {};

    /// Member dimensions for each axis.
    std::vector<uint64_t>    m_dimensions {};

    /// Member strides for each axis.
    std::vector<uint64_t>    m_strides {};

    /// Member boolean for data ownership; true if it's owned by tensor
    bool                     m_own_data {true};

    /// Member enumeration to indicate if data is on host or device.
    MemoryLocation           m_mem_loc {MemoryLocation::DEVICE};

    /**
     * @brief Computes strides using dimensions.
     *
     * Resizes `m_strides` and fills each element so that
     * `m_strides[i]` equals the product of all dimensions to the right of `i`.
     *
     * @throws std::invalid_argument if any entry in `m_dimensions` is zero.
     * @throws std::overflow_error if stride multiplication
     * would overflow `uint64_t`.
     */
    void compute_strides();

public:

    /**
     * @brief Tensor default class constructor.
     *
     * Set as default.
     */
    Tensor() = default;

    /**
     * @brief Construct a tensor given the shape and
     * allocate the memory for its data.
     *
     * Sets the tensor's dimensions to @p dimensions, computes strides,
     * and allocates zero-initialized memory on the specified location.
     *
     * @param dimensions Shape of the tensor (each entry must be > 0).
     * @param loc Memory location for data (HOST or DEVICE).
     *
     * @throws std::invalid_argument if:
     * - @p dimensions is empty
     * - any entry in @p dimensions is zero
     * @throws std::overflow_error if:
     * - the product of @p dimensions would overflow uint64_t
     * - the total allocation size in bytes would not fit into size_t
     * @throws std::runtime_error if:
     * - the requested allocation exceeds device @c max_mem_alloc_size
     * - the requested allocation exceeds device @c global_mem_size
     * @throws std::bad_alloc if memory allocation fails.
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
     * @brief View constructor.
     * Constructs a non-owning view into another tensor.
     *
     * Creates a view that aliases the owner’s linear buffer (no reallocation).
     * The view logically does not own separate storage,
     * but it holds an aliasing `shared_ptr` to the owner's buffer so the buffer
     * remains alive while either control block is retained.
     *
     * @param owner Tensor to view into.
     * @param start_indices Starting coordinate of the view (one per owner axis).
     * @param view_shape Shape of the view.
     *
     * @throws std::invalid_argument if:
     * - @p start_indices.size() does not equal the owner's rank
     * - @p view_shape is empty or its rank is greater than the owner's rank
     * @throws std::out_of_range if:
     * - any entry in @p start_indices is >= the corresponding owner dimension
     * - any entry in @p view_shape is zero
     * - any start + length in @p view_shape extends beyond the owner dimension
     * @throws std::runtime_error if:
     * - @p other has no data (default-constructed or moved-from)
     */
    Tensor(const Tensor & owner,
            const std::vector<uint64_t> & start_indices,
            const std::vector<uint64_t> & view_shape);

    /**
     * @brief Alias view constructor.
     * Constructs a non-owning strided view into another tensor.
     *
     * Creates a view that aliases the owner’s linear buffer (no reallocation).
     * The view logically does not own separate storage,
     * but it holds an aliasing `shared_ptr` to the owner's buffer so the buffer
     * remains alive while either control block is retained.
     * Strides allow accessing elements in non-contiguous memory layouts.
     *
     * @param owner Tensor to view into.
     * @param start_indices Starting coordinate of the view (one per owner axis).
     * @param dims Dimensions of the view (size per axis).
     * Must have all entries > 0.
     * @param strides Strides of the view(step size in the underlying data
     * for each axis). Must have the same size as @p dims.
     *
     * @throws std::runtime_error if the owner tensor has no allocated data.
     * @throws std::invalid_argument if:
     * - @p start_indices.size() does not match the owner’s rank.
     * - @p dims.size() != @p strides.size().
     * - the view rank (number of dimensions) is zero.
     * - any entry in @p dims is zero.
     * @throws std::out_of_range if:
     * - any entry in @p start_indices is >= the corresponding
     * dimension of the owner.
     * - the computed maximum index in the owner buffer that the view
     * will access exceeds the owner’s total size.
     * @throws std::overflow_error if:
     * - stride * (dim-1) overflows for any axis.
     * - computing the maximum index for the view overflows uint64_t.
     */
    Tensor(const Tensor & owner,
            const std::vector<uint64_t> & start_indices,
            const std::vector<uint64_t> & dims,
            const std::vector<uint64_t> & strides);

    /**
     * @brief Copy assignment operator.
     *
     * Performs a deep copy of metadata and (if owning) the underlying buffer.
     *
     * @param other The tensor to assign from.
     * @return Reference to this tensor.
     *
     * @throws std::bad_alloc if allocation fails.
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
     * @brief Assign values from a flat std::vector into the tensor.
     *
     * Copies the contents of @p values into the tensor.
     * The length of @p values must equal the tensor's total element count.
     *
     * @param values Flat input vector.
     * @return Reference to this tensor.
     *
     * @throws std::invalid_argument if:
     * - the tensor has no dimensions
     * - the length of @p values differs from the tensor's total element count
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
     * @brief Element-wise addition with right-aligned broadcasting.
     *
     * Computes element-wise sum between this tensor and @p other using
     * right-aligned broadcasting (dimensions of size 1 are broadcastable).
     * Result memory location is DEVICE if either operand is on the device,
     * otherwise HOST. Executed synchronously on the library SYCL queue.
     *
     * @param other Tensor to add.
     * @return New tensor containing the broadcasted element-wise sum.
     *
     * @throws std::invalid_argument if either tensor is empty
     * or shapes are incompatible for broadcasting.
     * @throws std::runtime_error "NaN detected in inputs."
     * if a NaN was observed.
     * @throws std::runtime_error "Non-finite result (overflow or Inf)."
     * if a non-finite result was produced.
     * @throws std::runtime_error "Numeric error during element-wise addition."
     * for other numeric errors.
     * @throws std::bad_alloc if memory allocation fails.
     */
    Tensor operator+(const Tensor & other) const;

    /**
     * @brief Element-wise subtraction with right-aligned broadcasting.
     *
     * Computes element-wise difference (this - other) with right-aligned
     * broadcasting (dimensions of size 1 are broadcastable). Result memory
     * location follows the same device/host rule. Executed synchronously on
     * the library SYCL queue.
     *
     * @param other Tensor to subtract.
     * @return New tensor containing the broadcasted element-wise difference.
     *
     * @throws std::invalid_argument if either tensor is empty
     * or shapes are incompatible for broadcasting.
     * @throws std::runtime_error "NaN detected in inputs."
     * if a NaN was observed.
     * @throws std::runtime_error "Non-finite result (overflow or Inf)."
     * if a non-finite result was produced.
     * @throws std::runtime_error "Numeric error during element-wise addition."
     * for other numeric errors.
     * @throws std::bad_alloc if memory allocation fails.
     */
    Tensor operator-(const Tensor & other) const;

    /**
     * @brief Element-wise multiplication with right-aligned broadcasting.
     *
     * Computes element-wise product with right-aligned broadcasting
     * (dimensions of size 1 are broadcastable). Result memory location is
     * DEVICE if either operand is on device, otherwise HOST. Executed
     * synchronously on the library SYCL queue.
     *
     * @param other Tensor to multiply.
     * @return New tensor containing the broadcasted element-wise product.
     *
     * @throws std::invalid_argument if either tensor is empty
     * or shapes are incompatible for broadcasting.
     * @throws std::runtime_error "NaN detected in inputs."
     * if a NaN was observed.
     * @throws std::runtime_error "Non-finite result (overflow or Inf)."
     * if a non-finite result was produced.
     * @throws std::runtime_error "Numeric error during element-wise addition."
     * for other numeric errors.
     * @throws std::bad_alloc if memory allocation fails.
     */
    Tensor operator*(const Tensor & other) const;

    /**
     * @brief Element-wise division with right-aligned broadcasting and checks.
     *
     * Computes element-wise quotient (this / other) with right-aligned
     * broadcasting (dimensions of size 1 are broadcastable). Denominator
     * zero is detected and yields ±infinity in the output according to the
     * numerator sign and records a division-by-zero error. Result memory
     * location is DEVICE if either operand is on device, otherwise HOST.
     * Executed synchronously on the library SYCL queue.
     *
     * @param other Divisor tensor.
     * @return New tensor containing the broadcasted element-wise quotient.
     *
     * @throws std::invalid_argument if either tensor is empty
     * or shapes are incompatible for broadcasting.
     * @throws std::runtime_error "NaN detected in inputs."
     * if a NaN was observed.
     * @throws std::runtime_error "Division by zero detected."
     * if any divisor element equals zero.
     * @throws std::runtime_error "Non-finite result detected."
     * if a non-finite quotient was produced.
     * @throws std::runtime_error "Numeric error during element-wise division."
     * for other numeric errors.
     * @throws std::bad_alloc if memory allocation fails.
     */
    Tensor operator/(const Tensor & other) const;

    /**
     * @brief Unary element-wise negation.
     *
     * Returns a new tensor where each element is the negation of the
     * corresponding element in this tensor. Output has the same shape and
     * memory location as the input.
     * Executed synchronously on the library SYCL queue.
     *
     * @return New tensor containing element-wise negated values.
     *
     * @throws std::invalid_argument if this tensor empty.
     * @throws std::runtime_error "NaN detected in input." if a NaN was observed.
     * @throws std::bad_alloc if memory allocation fails.
     */
    Tensor operator-() const;

    /**
     * @brief Returns an owning, contiguous deep copy of this tensor.
     *
     * The clone has the same shape and memory location and contains an
     * independent copy of all elements. Operation is synchronous.
     *
     * @return Owning Tensor<float_t> with contiguous storage.
     * @throws std::invalid_argument if this tensor is empty.
     * @throws std::bad_alloc on allocation failure.
     */
    Tensor<float_t> clone() const;

    /**
     * @brief Moves tensor data between host (shared) and device memory.
     *
     * Transfers owned data to the specified memory location.
     * Only tensors that own their data can be moved.
     *
     * @param target_loc Target memory location (HOST or DEVICE).
     * @throws std::runtime_error if called on a non-owning tensor (view).
     * @throws std::invalid_argument if tensor has no elements.
     * @throws std::bad_alloc if memory allocation fails.
     */
    void to(MemoryLocation target_loc);

    /**
     * @brief Change tensor shape metadata without moving or reallocating data.
     *
     * Sets the tensor's dimensions to @p new_dimensions and recomputes strides.
     * The underlying linear buffer is preserved, but the logical shape changes.
     *
     * @param new_dimensions New shape for the tensor.
     *
     * @throws std::invalid_argument if:
     * - @p new_dimensions is empty
     * - tensor is non-owning
     * - any entry in @p new_dimensions is zero
     * - the product of @p new_dimensions differs from the current total
     * element count
     * @throws std::overflow_error if the product of dimensions in
     * @p new_dimensions would overflow uint64_t.
     */
    void reshape(const std::vector<uint64_t>& new_dimensions);

    /**
     * @brief Sort tensor elements (device merge-sort).
     *
     * Sorts the tensor either flattened (axis = -1) or independently along a
     * single axis. Uses device buffers and SYCL kernels; sorting is done
     * in place.
     *
     * @param axis Axis to sort along, -1 = flatten, otherwise 0..rank-1.
     * @throws std::invalid_argument if axis is out of range.
     * @throws std::bad_alloc if required device memory cannot be allocated.
     */
    void sort(int64_t axis = -1);

    /**
     * @brief Compute the sum of tensor elements (device reduction).
     *
     * Computes sums either flattened (axis = -1) or independently
     * along a single axis.
     * Uses device buffers and SYCL kernels; operation returns a new tensor.
     *
     * @param axis Axis to sum along, -1 = flatten (sum all elements),
     * otherwise 0..rank-1.
     * @return Tensor<float_t> New tensor containing the sums;
     * the returned tensor uses the same memory location as the input.
     * If the input tensor has no dimensions, a tensor
     * with shape {1} is returned.
     * @throws std::invalid_argument If axis is not -1 and is out of range.
     * @throws std::bad_alloc if required device memory cannot be allocated.
     * @throws std::runtime_error If NaN or non-finite values are encountered
     * in the inputs or the results.
     */
    Tensor<float_t> sum(int64_t axis = -1) const;

    /**
     * @brief Compute the cumulative sum of tensor elements (device scan).
     *
     * Computes cumulative sums either flattened (axis = -1) or
     * independently along a single axis. Uses device buffers and
     * SYCL kernels; operation returns a new tensor.
     *
     * @param axis Axis to scan along, -1 = flatten (treat as 1D and scan
     * all elements), otherwise 0..rank-1.
     * @return Tensor<float_t> New tensor containing the cumulative sums;
     * the returned tensor uses the same memory location as the input.
     * If the input tensor has no dimensions, a tensor
     * with shape {1} is returned.
     *
     * @throws std::invalid_argument If axis is not -1 and is out of range.
     * @throws std::bad_alloc If required device memory cannot be allocated.
     * @throws std::runtime_error If NaN or non-finite values are encountered
     * in the inputs or the results.
     */
    Tensor<float_t> cumsum(int64_t axis = -1) const;

    /**
     * @brief Returns a new tensor with axes reversed (full transpose).
     *
     * This function returns a view of the tensor with its axes reversed.
     * For example, a tensor of shape [2,3,4] will become [4,3,2].
     * The returned tensor shares the same underlying data as
     * the original tensor, so no data is copied.
     *
     * @throws std::runtime_error if the tensor is empty (rank 0).
     * @return Tensor<float_t> A new tensor view with reversed axes.
     */
    Tensor<float_t> transpose() const;

    /**
     * @brief Returns a new tensor with permuted axes according to
     * the given order.
     *
     * This function creates a view of the tensor with its axes
     * rearranged according to the `axes` vector.
     * The returned tensor shares the same underlying data as
     * the original tensor, so no data is copied.
     *
     * @param axes Vector specifying the new order of axes.
     * Must be a permutation of [0..rank-1], where `rank` is
     * the number of dimensions of the tensor.
     *
     * @throws std::invalid_argument if `axes.size()` != rank or
     * if `axes` is not a valid permutation.
     * @return Tensor<float_t> A new tensor view with permuted axes.
     */
    Tensor<float_t> transpose(const std::vector<uint64_t> & axes) const;

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
     * Set as default.
     */
	~Tensor() noexcept = default;

    /**
     * @brief Returns a raw pointer to the underlying tensor data.
     *
     * This pointer is read-only (`const`) and may point either to
     * owned memory (for tensors that allocate their own storage)
     * or to an alias of another tensor's memory (for views).
     *
     * @return const float_t* Pointer to the first element of tensor data.
     */
    const float_t * get_data() const noexcept;

    /**
     * @brief Returns a raw pointer to the underlying tensor data.
     *
     * The returned pointer allows modification of the underlying elements.
     * Modifying elements through this pointer mutates the tensor's storage
     * (and will affect other views that alias the same storage).
     *
     * @return const float_t* Pointer to the first element of tensor data.
     */
    float_t * get_data() noexcept;

    /**
     * @brief Returns the dimensions (shape) of the tensor.
     *
     * Dimensions are stored as a vector of extents for each axis.
     *
     * @return const std::vector<uint64_t>& Reference to dimensions vector.
     */
    const std::vector<uint64_t> & get_dimensions() const noexcept;

    /**
     * @brief Returns the strides of the tensor.
     *
     * Strides define the memory step size (in elements) for each axis.
     *
     * @return const std::vector<uint64_t>& Reference to strides vector.
     */
    const std::vector<uint64_t> & get_strides() const noexcept;

    /**
     * @brief Returns the shape of the tensor.
     *
     * Equivalent to @ref get_dimensions().
     *
     * @return const std::vector<uint64_t>& Reference to shape vector.
     */
    const std::vector<uint64_t> & get_shape() const noexcept;

    /**
     * @brief Returns the rank (number of dimensions) of the tensor.
     *
     * For example, a 2D matrix has rank 2, a vector has rank 1,
     * and if empty it has rank 0.
     *
     * @return uint64_t Tensor rank.
     */
    uint64_t get_rank() const noexcept;

    /**
     * @brief Returns the total number of elements in the tensor.
     *
     * Computed as the product of all dimensions.
     * If the tensor has no dimensions, returns 0.
     *
     * @return uint64_t Number of elements in the tensor.
     */
    uint64_t get_num_elements() const noexcept;

    /**
     * @brief Returns the memory location where the tensor data resides.
     *
     * Indicates whether the tensor memory is allocated on the
     * host or on a device.
     *
     * @return MemoryLocation Location of tensor data.
     */
    MemoryLocation get_memory_location() const noexcept;

    /**
     * @brief Indicates whether the tensor owns its memory.
     *
     * - Returns `true` for owner tensors (allocated storage).
     * - Returns `false` for views or alias tensors (non-owning).
     *
     * @return bool True if the tensor owns its memory, false otherwise.
     */
    bool get_owns_data() const noexcept;

    /**
     * @brief Checks if the tensor is a view (non-owning).
     *
     * A tensor is considered a view if it does not own its memory
     * and instead references memory from another tensor.
     * Complementary to get_owns_data().
     *
     * @return bool True if tensor is a view, false otherwise.
     */
    bool is_view() const noexcept;

    /**
     * @brief Returns the size in bytes of a single tensor element.
     *
     * Equivalent to `sizeof(float_t)`.
     *
     * @return uint64_t Element size in bytes.
     */
    uint64_t get_element_size_bytes() const noexcept;

    /**
     * @brief Returns the total size in bytes of the tensor data.
     *
     * Computed as @ref get_num_elements multiplied by
     * @ref get_element_size_bytes.
     *
     * @return uint64_t Total size in bytes.
     */
    uint64_t get_total_bytes() const noexcept;

    /**
     * @brief Convert a flat row-major index (0..N-1) into per-axis coordinates.
     *
     * - If the tensor has rank 0 (empty shape) returns an empty vector.
     * - Throws std::out_of_range if flat >= total elements.
     *
     * @param flat Flattened logical index.
     * @return Vector of coordinates (size == rank).
     */
    std::vector<uint64_t> index_to_coords(uint64_t flat) const;

    /**
     * @brief Convert per-axis coordinates into a flat row-major index.
     *
     * - coords.size() must equal tensor rank.
     * - each coords[d] must be < shape[d].
     *
     * @param coords Per-axis coordinates.
     * @return Flattened index corresponding to coords.
     * @throws std::invalid_argument if coords size mismatches rank.
     * @throws std::out_of_range if any coordinate is outside its extent.
     */
    uint64_t coords_to_index
        (const std::vector<uint64_t>& coords) const;

    /**
     * @brief Access element at the given flat (row-major) index.
     *
     * - Returns a reference to the underlying element.
     * - Flat index must be in [0, total_elements).
     * - Works for both host and device tensors.
     *
     * @param flat Flattened row-major index.
     * @return Reference to element at flat index.
     * @throws std::out_of_range if flat >= total elements.
     */
    float_t & at(uint64_t flat);

    /**
     * @brief Const version of at().
     */
    const float_t& at(uint64_t flat) const;

};

/// Explicit instantiation for float
extern template class Tensor<float>;

} // namespace temper

#endif // TEMPER_TENSOR_HPP