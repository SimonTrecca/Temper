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

/**
 * @brief Compute categorical cross-entropy between predictions and targets.
 *
 * Computes the elementwise categorical cross-entropy loss
 * by taking -sum(labels * log(probs)) over the class axis.
 * If @p from_logits is true, @p logits are first converted to
 * probabilities with softmax along the requested axis.
 * Supports broadcasting.
 *
 * @param logits Input tensor containing either raw scores (logits)
 * or probabilities depending on @p from_logits.
 * Must contain at least one element.
 * @param labels Tensor of target values (one-hot or soft labels).
 * Must be shape-compatible with the probabilities
 * (or broadcastable to that shape).
 * @param axis_opt Axis of the class dimension for @p logits.
 * nullopt = flatten; otherwise -rank_logits..rank_logits-1.
 * @param from_logits If true (default) treat @p logits as raw scores and
 * apply softmax before taking the log. If false, treat
 * @p logits as probabilities already.
 * @param reduction_mean If true (default) return the mean scalar loss across
 * all remaining elements. If false return the loss reduced only along
 * the class axis (result keeps batch shape).
 *
 * @return Tensor<float_t> If @p reduction_mean is true a scalar tensor
 * containing the mean loss is returned. If false, the returned
 * tensor has the class axis removed (or is scalar when flattened).
 *
 * @throws std::invalid_argument If either input tensor is empty, or if
 * @p axis_opt is outside the valid range for @p logits.
 * @throws std::out_of_range If an internal view/index operation exceeds
 * tensor bounds (propagated from underlying tensor ops).
 * @throws std::runtime_error If non-finite values (NaN/Inf) are encountered
 * during computation (for example when taking the logarithm).
 * @throws std::bad_alloc On memory allocation failure.
 */
template<typename float_t>
Tensor<float_t> cross_entropy(const Tensor<float_t> & logits,
    const Tensor<float_t> & labels,
    std::optional<int64_t> axis_opt = std::nullopt,
    bool from_logits = true,
    bool reduction_mean = true);
/// Explicit instantiation of cross_entropy for float
extern template Tensor<float> cross_entropy<float>
(const Tensor<float>&, const Tensor<float>&, std::optional<int64_t>, bool, bool);

/* todo
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