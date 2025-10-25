/**
 * @file ML.hpp
 * @brief Machine learning utilities.
 *
 * Provides functionality for classical
 * machine learning algorithms and workflows.
 */

#ifndef TEMPER_ML_HPP
#define TEMPER_ML_HPP

#include "Tensor.hpp"

namespace temper::ml
{

/**
 * @brief Replace the slice at `axis:axis_index` with a one-hot
 * expansion.
 *
 * For an input shape dims = {D0,...,Daxis,...,Dlast}, the returned
 * tensor has the same rank and the size of `axis` becomes
 * (Daxis - 1 + depth).
 *
 * Semantics:
 *  - If coord[axis] != axis_index: values are copied unchanged into
 *    the output (possibly shifted along that axis).
 *  - If coord[axis] == axis_index: the scalar is read as an integer
 *    label and replaced by `depth` values. Only the label position
 *    is set to `on_value`. All other positions take `off_value`.
 *
 *
 * @param tensor Input tensor.
 * @param axis Axis to target, -rank..rank-1.
 * @param axis_index Index along `axis` to read labels from
 *        (0..Daxis-1).
 * @param depth Number of classes to expand into (must be > 0).
 * @param on_value Value for the hot entry (default 1).
 * @param off_value Value for all other entries (default 0).
 *
 * @throws std::invalid_argument If depth == 0 or tensor rank == 0
 *         or axis invalid.
 * @throws std::out_of_range If axis_index is outside the axis
 *         extent or a label is out of range.
 * @throws std::runtime_error For non-integer or non-finite label
 *         values.
 * @throws std::bad_alloc On allocation failure.
 */
template <typename float_t>
Tensor<float_t> one_hot_expand_at(const Tensor<float_t>& tensor,
    int64_t axis,
    uint64_t axis_index,
    uint64_t depth,
    float_t on_value = static_cast<float_t>(1),
    float_t off_value = static_cast<float_t>(0));
/// Explicit instantiation of one_hot_expand_at for float
extern template Tensor<float> one_hot_expand_at<float>
    (const Tensor<float>&, int64_t, uint64_t, uint64_t, float, float);

/**
 * @brief Compute the softmax along a single axis.
 *
 * Produces a tensor where each slice along the given @p axis_opt is
 * exponentiated and normalized so that the values along that axis sum to 1.
 *
 * @param tensor   Input tensor. Must contain at least one element.
 * @param axis_opt Axis along which softmax is applied, nullopt = flatten,
 * otherwise -rank..rank-1.
 *
 * @return A new tensor with the same shape as @p tensor containing
 * the normalized values.
 *
 * @throws std::invalid_argument If the tensor is empty or @p axis_opt is
 * outside the valid range.
 * @throws std::bad_alloc If memory allocation fails.
 * @throws std::runtime_error If NaN or non-finite values are encountered
 * during computation.
 */
template<typename float_t>
Tensor<float_t> softmax(const Tensor<float_t> & tensor,
    std::optional<int64_t> axis_opt = std::nullopt);
/// Explicit instantiation of softmax for float
extern template Tensor<float> softmax<float>
    (const Tensor<float>&, std::optional<int64_t>);

/* todo
    cross entropy loss
    mse loss
    regularization penalties
    other loss functions?
    softmax
    hist
    linear regression
    logistic regression
    pca
    standardscaler
    powertransformer
    chisquare test
    minimize
    maximize
    anova
*/
} // namespace temper::ml

#endif // TEMPER_ML_HPP