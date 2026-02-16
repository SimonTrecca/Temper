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
#include <optional>
#include "SYCLQueue.hpp"
#include "Autograd.hpp"

namespace temper
{

/**
 * @brief Class template for the Tensor data structure.
 * @tparam value_t Numeric types.
 *
 * The class manages a linear buffer in row-major order.
 */
template <typename value_t>
class Tensor
{

private:

    /// Member pointer to data.
	std::shared_ptr<value_t> m_p_data {};

    /// Member dimensions for each axis.
    std::vector<uint64_t>    m_dimensions {};

    /// Member strides for each axis.
    std::vector<uint64_t>    m_strides {};

    /// Member boolean for data ownership; true if it's owned by tensor
    bool                     m_own_data {true};

    /// Member enumeration to indicate if data is on host or device.
    MemoryLocation           m_mem_loc {MemoryLocation::DEVICE};

    /// Metadata for automatic differentiation and graph replay.
    AutogradMeta<value_t>    m_meta;

    /**
     * @brief Computes strides using dimensions.
     *
     * Resizes `m_strides` and fills each element so that
     * `m_strides[i]` equals the product of all dimensions to the right of `i`.
     */
    void compute_strides();

public:

    /**
     * @brief Random-access iterator for mutable traversal of a Tensor.
     *
     * The iterator stores a pointer to the owning Tensor and a flat (row-major)
     * index into the owner's storage. It models a RandomAccessIterator.
     *
     * Use these iterators to iterate over tensor elements in flat order.
     * Comparisons between iterators are meaningful only when they refer
     * to the same owner Tensor.
     */
    class iterator
    {

    private:

        /// Member pointer to owner Tensor.
        Tensor * m_p_owner= nullptr;

        /// Member flat index to owner Tensor memory.
        uint64_t m_flat_idx = 0;

    public:

        using iterator_category = std::random_access_iterator_tag;
        using value_type        = value_t;
        using difference_type   = int64_t;
        using pointer           = void;
        using reference         = Tensor<value_t>;

        /**
         * @brief Default-construct a Tensor (empty).
         *
         * Creates an empty, uninitialized tensor object.
         */
        iterator() = default;

        /**
         * @brief Construct an iterator referring to an element of a Tensor.
         *
         * @param owner Pointer to the owning Tensor (must not be nullptr).
         * @param flat_idx Flat (row-major) index within the owner.
         */
        iterator(Tensor * owner, uint64_t flat_idx) noexcept
            : m_p_owner(owner),
              m_flat_idx(flat_idx)
        {
            // No-op; everything already set before body.
        }

        /**
         * @brief Copy-construct an iterator (defaulted).
         *
         * Performs a memberwise copy of the iterator state.
         */
        iterator(const iterator &) = default;

        /**
         * @brief Move-construct an iterator (defaulted).
         *
         * Performs a memberwise move of the iterator state.
         */
        iterator(iterator &&) noexcept = default;

        /**
         * @brief Copy-assign an iterator (defaulted).
         *
         * Performs a memberwise copy assignment.
         */
        iterator & operator=(const iterator &) = default;

        /**
         * @brief Move-assign an iterator (defaulted).
         *
         * Performs a memberwise move assignment.
         */
        iterator & operator=(iterator &&) noexcept = default;

        /**
         * @brief Dereference the iterator.
         *
         * Returns a view to the tensor element at the iterator's
         * current flat index. Forwarding to Tensor::at().
         *
         * @return reference View Tensor to the element.
         */
        reference operator*() const
        {
            return m_p_owner->at(m_flat_idx);
        }

        /**
         * @brief Pre-increment: advance to the next element.
         *
         * @return iterator& Reference to the incremented iterator.
         */
        iterator& operator++()
        {
            ++m_flat_idx;
            return *this;
        }

        /**
         * @brief Post-increment: advance but return prior state.
         *
         * @return iterator Copy of iterator before increment.
         */
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        /**
         * @brief Pre-decrement: move to the previous element.
         *
         * @return iterator& Reference to the decremented iterator.
         */
        iterator& operator--()
        {
            --m_flat_idx;
            return *this;
        }

        /**
         * @brief Post-decrement: move back but return prior state.
         *
         * @return iterator Copy of iterator before decrement.
         */
        iterator operator--(int)
        {
            iterator tmp = *this;
            --(*this);
            return tmp;
        }

        /**
         * @brief Advance iterator by n positions (in-place).
         *
         * @param n Signed offset to add to the flat index.
         * @return iterator& Reference to this after modification.
         */
        iterator& operator+=(difference_type n)
        {
            m_flat_idx += static_cast<uint64_t>(n);
            return *this;
        }

        /**
         * @brief Move iterator back by n positions (in-place).
         *
         * @param n Signed offset to subtract from the flat index.
         * @return iterator& Reference to this after modification.
         */
        iterator& operator-=(difference_type n)
        {
            m_flat_idx -= static_cast<uint64_t>(n);
            return *this;
        }

        /**
         * @brief Return a new iterator advanced by n positions.
         *
         * Does not modify *this.
         *
         * @param n Signed offset to add.
         * @return iterator New iterator at the advanced position.
         */
        iterator operator+(difference_type n) const
        {
            return iterator( m_p_owner, m_flat_idx + static_cast<uint64_t>(n));
        }

        /**
         * @brief Return a new iterator moved back by n positions.
         *
         * Does not modify *this.
         *
         * @param n Signed offset to subtract.
         * @return iterator New iterator at the resulting position.
         */
        iterator operator-(difference_type n) const
        {
            return iterator(m_p_owner, m_flat_idx - static_cast<uint64_t>(n));
        }

        /**
         * @brief Compute signed distance between two iterators.
         * Both iterators refer to the same owner Tensor;
         * otherwise the result is meaningless.
         *
         * Result is (this->m_flat_idx - o.m_flat_idx).
         *
         * @param o Other iterator to compare.
         * @return difference_type Signed distance.
         */
        difference_type operator-(const iterator& o) const
        {
            return static_cast<difference_type>(
                static_cast<int64_t>(m_flat_idx) -
                static_cast<int64_t>(o.m_flat_idx)
            );
        }

        /**
         * @brief Equality comparison.
         *
         * True if both iterators refer to the same owner and same flat index.
         *
         * @param o Other iterator.
         * @return true if equal.
         */
        bool operator==(const iterator& o) const noexcept
        {
            return m_p_owner == o.m_p_owner && m_flat_idx == o.m_flat_idx;
        }

        /**
         * @brief Inequality comparison (negation of operator==).
         *
         * @param o Other iterator.
         * @return true if not equal.
         */
        bool operator!=(const iterator& o) const noexcept
        {
            return !(*this == o);
        }

        /**
         * @brief Strict weak ordering based on flat index (same-owner only).
         *
         * @param o Other iterator.
         * @return true if this index < o.index and owners are equal.
         */
        bool operator<(const iterator& o) const noexcept
        {
            return m_p_owner == o.m_p_owner && m_flat_idx < o.m_flat_idx;
        }

        /**
         * @brief Greater-than (implemented in terms of operator<).
         *
         * @param o Other iterator.
         * @return true if this > o.
         */
        bool operator>(const iterator& o) const noexcept
        {
            return o < *this;
        }

        /**
         * @brief Less-than-or-equal (implemented in terms of operator<).
         *
         * @param o Other iterator.
         * @return true if this <= o.
         */
        bool operator<=(const iterator& o) const noexcept
        {
            return !(o < *this);
        }

        /**
         * @brief Greater-than-or-equal (implemented in terms of operator<).
         *
         * @param o Other iterator.
         * @return true if this >= o.
         */
        bool operator>=(const iterator& o) const noexcept
        {
            return !(*this < o);
        }
    };

    /**
     * @brief Random-access iterator for read-only traversal of a Tensor.
     *
     * Similar to iterator but provides const access to the underlying elements.
     * Models RandomAccessIterator for const access.
     */
    class const_iterator
    {

    private:

        /// Member pointer to const owner Tensor.
        const Tensor * m_p_owner = nullptr;

        /// Member flat index to owner Tensor memory.
        uint64_t m_flat_idx = 0;

    public:

        using iterator_category = std::random_access_iterator_tag;
        using value_type        = value_t;
        using difference_type   = int64_t;
        using pointer           = void;
        using reference         = const Tensor<value_t>;

        /**
         * @brief Default-construct a const_iterator.
         *
         * Constructs a const_iterator in an unspecified (null) state.
         */
        const_iterator() = default;

        /**
         * @brief Construct a const_iterator referring to
         * an element of a Tensor.
         *
         * @param owner Pointer to the owning (const) Tensor.
         * @param flat_idx Flat (row-major) index within the owner.
         */
        const_iterator(const Tensor * owner, uint64_t flat_idx) noexcept
            : m_p_owner(owner),
              m_flat_idx(flat_idx)
        {
            // No-op; everything already set before body.
        }

        /**
         * @brief Construct a const_iterator by copying another
         * const_iterator (defaulted).
         *
         * Performs a memberwise copy of const_iterator state.
         */
        const_iterator(const const_iterator &) = default;

        /**
         * @brief Move-construct a const_iterator (defaulted).
         *
         * Performs a memberwise move of const_iterator state.
         */
        const_iterator(const_iterator &&) noexcept = default;

        /**
         * @brief Copy-assign a const_iterator (defaulted).
         *
         * Performs a memberwise copy assignment.
         */
        const_iterator & operator=(const const_iterator &) = default;

        /**
         * @brief Move-assign a const_iterator (defaulted).
         *
         * Performs a memberwise move assignment.
         */
        const_iterator & operator=(const_iterator &&) noexcept = default;

        /**
         * @brief Dereference the const_iterator.
         *
         * Returns a const view to the tensor element at the current index.
         *
         * @return reference Const view to the element.
         */
        reference operator*() const
        {
            return m_p_owner->at(m_flat_idx);
        }

        /**
         * @brief Pre-increment.
         *
         * @return const_iterator& Reference to the incremented iterator.
         */
        const_iterator& operator++()
        {
            ++m_flat_idx;
            return *this;
        }

        /**
         * @brief Post-increment.
         *
         * @return const_iterator Copy before increment.
         */
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        /**
         * @brief Pre-decrement.
         *
         * @return const_iterator& Reference to the decremented iterator.
         */
        const_iterator& operator--()
        {
            --m_flat_idx;
            return *this;
        }

        /**
         * @brief Post-decrement.
         *
         * @return const_iterator Copy before decrement.
         */
        const_iterator operator--(int)
        {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        /**
         * @brief Advance by n positions (in-place).
         *
         * @param n Signed offset.
         * @return const_iterator& Reference to this.
         */
        const_iterator& operator+=(difference_type n)
        {
            m_flat_idx += static_cast<uint64_t>(n);
            return *this;
        }

        /**
         * @brief Move back by n positions (in-place).
         *
         * @param n Signed offset.
         * @return const_iterator& Reference to this.
         */
        const_iterator& operator-=(difference_type n)
        {
            m_flat_idx -= static_cast<uint64_t>(n);
            return *this;
        }

         /**
         * @brief Return a new const_iterator advanced by n positions.
         *
         * @param n Signed offset.
         * @return const_iterator New const_iterator at advanced position.
         */
        const_iterator operator+(difference_type n) const
        {
            return const_iterator(m_p_owner,
                m_flat_idx + static_cast<uint64_t>(n));
        }

        /**
         * @brief Return a new const_iterator moved back by n positions.
         *
         * @param n Signed offset.
         * @return const_iterator New const_iterator at resulting position.
         */
        const_iterator operator-(difference_type n) const
        {
            return const_iterator(m_p_owner,
                m_flat_idx - static_cast<uint64_t>(n));
        }

        /**
         * @brief Compute signed distance between two const_iterators.
         *
         * @param o Other const_iterator.
         * @return difference_type Signed distance (this - o).
         */
        difference_type operator-(const const_iterator& o) const
        {
            return static_cast<difference_type>(
                static_cast<int64_t>(m_flat_idx) -
                static_cast<int64_t>(o.m_flat_idx)
            );
        }

        /**
         * @brief Equality comparison.
         *
         * @param o Other const_iterator.
         * @return true if equal.
         */
        bool operator==(const const_iterator& o) const noexcept
        {
            return m_p_owner == o.m_p_owner && m_flat_idx == o.m_flat_idx;
        }

        /**
         * @brief Inequality comparison.
         *
         * @param o Other const_iterator.
         * @return true if not equal.
         */
        bool operator!=(const const_iterator& o) const noexcept
        {
            return !(*this == o);
        }

        /**
         * @brief Less-than comparison (same-owner only).
         *
         * @param o Other const_iterator.
         * @return true if this < o.
         */
        bool operator<(const const_iterator& o) const noexcept
        {
            return m_p_owner == o.m_p_owner && m_flat_idx < o.m_flat_idx;
        }

        /**
         * @brief Greater-than comparison.
         *
         * @param o Other const_iterator.
         * @return true if this > o.
         */
        bool operator>(const const_iterator& o) const noexcept
        {
            return o < *this;
        }

        /**
         * @brief Less-than-or-equal comparison.
         *
         * @param o Other const_iterator.
         * @return true if this <= o.
         */
        bool operator<=(const const_iterator& o) const noexcept
        {
            return !(o < *this);
        }

        /**
         * @brief Greater-than-or-equal comparison.
         *
         * @param o Other const_iterator.
         * @return true if this >= o.
         */
        bool operator>=(const const_iterator& o) const noexcept
        {
            return !(*this < o);
        }
    };

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
     */
    explicit Tensor(const std::vector<uint64_t>& dimensions,
        MemoryLocation loc = MemoryLocation::DEVICE);

    /**
     * @brief Construct a tensor from an initializer list of dimensions.
     *
     * Equivalent to the vector-based constructor:
     * `Tensor(std::vector<uint64_t>(dimensions), loc)`.
     * Allocates zero-initialized memory on the specified location.
     *
     * @param dimensions Shape of the tensor (each entry must be > 0).
     * @param loc Memory location for data (HOST or DEVICE).
     */
    explicit Tensor(const std::initializer_list<uint64_t> & dimensions,
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
     * @brief Construct a scalar tensor from a single value.
     *
     * Creates a rank-1 tensor of shape `{1}` containing the given scalar @p val.
     * The tensor is allocated either on host or device memory depending on
     * @p loc.
     *
     * @param val Scalar value to initialize the tensor with.
     * @param loc Memory location where the tensor should be allocated
     *            (default: MemoryLocation::DEVICE).
     */
    explicit Tensor(value_t val, MemoryLocation loc = MemoryLocation::DEVICE);

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
     */
    Tensor& operator=(const std::vector<value_t> & values);

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
     */
    Tensor& operator=(value_t val);

    /**
     * @brief Returns a non-owning view into a sub-tensor at the given index.
     *
     * Applies the index to the first dimension:
     * - Rank 1: view of a single element (shape {1})
     * - Rank > 1: view of the remaining dimensions (axis 0 dropped)
     *
     * @param idx Index along the first dimension.
     * @return Tensor view (non-owning) into the selected region.
     */
    Tensor operator[](uint64_t idx);

    /**
     * @brief Returns a const non-owning view into a sub-tensor at
     * the given index, if the Tensor it's called on is also constant.
     *
     * Applies the index to the first dimension:
     * - Rank 1: view of a single element (shape {1})
     * - Rank > 1: view of the remaining dimensions (axis 0 dropped)
     *
     * @param idx Index along the first dimension.
     * @return Tensor view (non-owning) into the selected region.
     */
    const Tensor operator[](uint64_t idx) const;

    /**
     * @brief Converts a scalar tensor to its underlying value.
     *
     * Copies the single element from host or device memory.
     * The tensor must contain exactly one element.
     *
     * @return Scalar value stored in this tensor.
     */
    operator value_t() const;

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
     */
    Tensor operator-() const;

    /**
     * @brief Value equality: shape then element-wise compare.
     *
     * Returns true iff @p other has the same dimensions and every
     * corresponding element compares equal with `operator==`.
     *
     * @param other Tensor to compare.
     * @return true when shapes match and all elements are equal,
     * false otherwise.
     */
    bool operator==(const Tensor & other) const;

    /**
     * @brief Returns an owning, contiguous deep copy of this tensor.
     *
     * The clone has the same shape and memory location and contains an
     * independent copy of all elements. Operation is synchronous.
     *
     * @return Owning Tensor<value_t> with contiguous storage.
     */
    Tensor<value_t> clone() const;

    /**
     * @brief Copy elements from another tensor into this tensor
     * (supports broadcasting).
     *
     * Copies data from @p src into this tensor. If both tensors have identical
     * shapes and canonical (contiguous) strides a fast device memcpy is used.
     * Otherwise @p src is broadcast to the destination shape (prepended with
     * singleton dims as required) and copied element-wise on device using SYCL
     * kernels that respect strides.
     *
     * @param src Source tensor to copy from.
     */
    void copy_from(const Tensor & src);

    /**
     * @brief Moves tensor data between host (shared) and device memory.
     *
     * Transfers owned data to the specified memory location.
     * Only tensors that own their data can be moved.
     *
     * @param target_loc Target memory location (HOST or DEVICE).
     */
    void to(MemoryLocation target_loc);

    /**
     * @brief Change tensor shape metadata without moving or reallocating data.
     *
     * Sets the tensor's dimensions to @p new_dimensions and recomputes strides.
     * The underlying linear buffer is preserved, but the logical shape changes.
     *
     * @param new_dimensions New shape for the tensor.
     */
    void reshape(const std::vector<uint64_t>& new_dimensions);

    /**
     * @brief Sort tensor elements (device merge-sort).
     *
     * Sorts the tensor either flattened (axis = nullopt, default)
     * or independently along a single axis.
     * Uses device buffers and SYCL kernels; sorting is done in place.
     * Supports negative axis indexing to start from right to left.
     *
     * @param axis_opt Axis to sort along, nullopt = flatten,
     * otherwise -rank..rank-1.
     */
    void sort(std::optional<int64_t> axis_opt = std::nullopt);

    /**
     * @brief Compute the sum of tensor elements (device reduction).
     *
     * Computes sums either flattened (axis = nullopt) or independently
     * along a single axis.
     * Uses device buffers and SYCL kernels; operation returns a new tensor.
     *
     * @param axis_opt Axis to sum along, nullopt = flatten,
     * otherwise -rank..rank-1.
     * @return Tensor<value_t> New tensor containing the sums;
     * the returned tensor uses the same memory location as the input.
     * If the input tensor has no dimensions, a tensor
     * with shape {1} is returned.
     */
    Tensor<value_t> sum(std::optional<int64_t> axis_opt = std::nullopt) const;

    /**
     * @brief Compute the cumulative sum of tensor elements (device scan).
     *
     * Computes cumulative sums either flattened (axis = nullopt) or
     * independently along a single axis. Uses device buffers and
     * SYCL kernels; operation returns a new tensor.
     *
     * @param axis_opt Axis to cumsum along, nullopt = flatten,
     * otherwise -rank..rank-1.
     * @return Tensor<value_t> New tensor containing the cumulative sums;
     * the returned tensor uses the same memory location as the input.
     * If the input tensor has no dimensions, a tensor
     * with shape {1} is returned.
     */
    Tensor<value_t> cumsum(std::optional<int64_t> axis_opt = std::nullopt) const;

    /**
     * @brief Returns a new tensor with axes reversed (full transpose).
     *
     * This function returns a view of the tensor with its axes reversed.
     * For example, a tensor of shape [2,3,4] will become [4,3,2].
     * The returned tensor shares the same underlying data as
     * the original tensor, so no data is copied.
     *
     * @return Tensor<value_t> A new tensor view with reversed axes.
     */
    Tensor<value_t> transpose() const;

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
     * Must be a permutation of [-rank..rank-1], where `rank` is
     * the number of dimensions of the tensor.
     *
     * @return Tensor<value_t> A new tensor view with permuted axes.
     */
    Tensor<value_t> transpose(const std::vector<int64_t> & axes) const;

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
     * @brief Prints the tensor shape as a bracketed list of dimensions.
     *
     * Example:
     * @code
     * [2, 3, 4]
     * @endcode
     *
     * @param os Output stream to print to. Defaults to std::cout.
     */
    void print_shape(std::ostream& os = std::cout) const;

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
     * @return const value_t* Pointer to the first element of tensor data.
     */
    const value_t * get_data() const noexcept;

    /**
     * @brief Returns a raw pointer to the underlying tensor data.
     *
     * The returned pointer allows modification of the underlying elements.
     * Modifying elements through this pointer mutates the tensor's storage
     * (and will affect other views that alias the same storage).
     *
     * @return const value_t* Pointer to the first element of tensor data.
     */
    value_t * get_data() noexcept;

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
     * @return int64_t Tensor rank.
     */
    int64_t get_rank() const noexcept;

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
     * Equivalent to `sizeof(value_t)`.
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
     */
    uint64_t coords_to_index
        (const std::vector<uint64_t>& coords) const;

    /**
     * @brief Access element at the given flat index.
     *
     * - Returns a view of the underlying element.
     * - Flat index must be in [0, total_elements).
     * - Works for both host and device tensors.
     *
     * @param flat Flattened row-major index.
     * @return Reference to element at flat index.
     */
    Tensor<value_t> at(uint64_t flat);

    /**
     * @brief Access element at the given flat index (const version).
     *
     * - Returns a const view of the underlying element.
     * - Flat index must be in [0, total_elements).
     * - Works for both host and device tensors.
     *
     * @param flat Flattened row-major index.
     * @return Reference to element at flat index.
     */
    const Tensor<value_t> at(uint64_t flat) const;

    /**
     * @brief Implementation: iterator to first element (mutable).
     *
     * @return iterator Iterator constructed at flat index 0.
     */
    iterator begin() noexcept;

    /**
     * @brief Implementation: iterator one-past-the-end (mutable).
     *
     * @return iterator Iterator constructed at flat index get_num_elements().
     */
    iterator end() noexcept;

    /**
     * @brief Implementation: const_iterator to first element.
     *
     * @return const_iterator Const iterator constructed at flat index 0.
     */
    const_iterator begin() const noexcept;

    /**
     * @brief Implementation: const_iterator one-past-the-end.
     *
     * @return const_iterator Const iterator constructed at flat
     * index get_num_elements().
     */
    const_iterator end() const noexcept;

    /**
     * @brief Implementation: cbegin() forwards to begin() const.
     *
     * @return const_iterator Const iterator to first element.
     */
    const_iterator cbegin() const noexcept;

    /**
     * @brief Implementation: cend() forwards to end() const.
     *
     * @return const_iterator Const iterator one-past-the-end.
     */
    const_iterator cend() const noexcept;
};

/// Explicit instantiation for float
extern template class Tensor<float>;
/// Explicit instantiation for uint64_t
extern template class Tensor<uint64_t>;

} // namespace temper

#endif // TEMPER_TENSOR_HPP