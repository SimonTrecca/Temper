/**
 * @file Math.hpp
 * @brief Mathematical operations for tensors.
 *
 * Provides general-purpose mathematical functionality used across the library.
 */

#ifndef TEMPER_MATH_HPP
#define TEMPER_MATH_HPP

#include "Tensor.hpp"

namespace temper::math
{

/**
 * @brief Matrix multiplication between two tensors.
 *
 * Performs a batched or unbatched matrix multiplication between
 * @p first and @p second. Supports the following cases:
 * - Vector × Vector → scalar
 * - Vector × Matrix → vector
 * - Matrix × Vector → vector
 * - Matrix × Matrix → matrix
 * - Batched matrices with broadcasting on batch dimensions
 *
 * The tensors must satisfy the shape constraints:
 * - Last dimension of @p first matches second-to-last of @p second
 * - Batch dimensions are broadcastable
 *
 * The resulting tensor shape is determined according to standard
 * linear algebra rules, with batch dimensions broadcasted if needed.
 *
 * @param first Left-hand side tensor.
 * @param second Right-hand side tensor.
 * @return Tensor<float_t> Result of the matrix multiplication.
 *
 * @throws std::invalid_argument if:
 * - The last dimension of @p first does not match the second-to-last of @p second.
 * - Batch dimensions are not broadcastable.
 * @throws std::overflow_error if:
 * - The resulting tensor would exceed the maximum size representable by uint64_t.
 * @throws std::runtime_error if:
 * - A numeric error occurs during computation (e.g., non-finite values produced).
 */
template <typename float_t>
Tensor<float_t> matmul(const Tensor<float_t> & first,
                        const Tensor<float_t> & second);
/// Explicit instantiation for float
extern template Tensor<float> matmul<float>
	(const Tensor<float>&, const Tensor<float>&);


/* todo
    all tensor functions
    pad
    argmax
    linspace
    arange
    zeros
    integral
    factorial
    log?
    mean
    var
    std
    cov
    eigen
    */
} // namespace temper::math

#endif // TEMPER_MATH_HPP
