/**
 * @file Autograd.hpp
 * @brief Base interfaces for the automatic differentiation system.
 *
 * Defines the GradientEdge interface, which represents the link between
 * tensors in the computational graph and handles backward propagation logic.
 */

#ifndef TEMPER_AUTOGRAD_HPP
#define TEMPER_AUTOGRAD_HPP

#include <vector>
#include <memory>

namespace temper
{

/**
 * @brief Forward declaration of Tensor.
 * Necessary to allow GradientEdge to reference Tensors without
 * creating a circular include dependency.
 */
template <typename value_t>
class Tensor;

/**
 * @brief Base interface for a functional edge in the computational graph.
 *
 * A GradientEdge represents the mathematical relationship between input
 * tensors and their resulting output. It stores the necessary context
 * from the forward pass to compute gradients during the backward pass.
 */
template <typename value_t>
class GradientEdge
{
public:
    /**
     * @brief Virtual destructor for safe inheritance.
     */
    virtual ~GradientEdge() = default;

    /**
     * @brief Get the input tensors connected by this edge.
     * Used by the autograd engine to perform topological sorting
     * and navigate the graph from outputs to inputs.
     *
     * @return A vector of shared pointers to the input tensors.
     */
    virtual std::vector<std::shared_ptr<Tensor<value_t>>> get_inputs() const = 0;

    /**
     * @brief Compute and propagate gradients through this edge.
     * This method implements the specific chain rule for the operation
     * represented by the edge. It calculates local derivatives and
     * calls accumulate_grad() on the input tensors.
     *
     * @param grad_output The gradient of the loss with respect to
     * the output tensor produced by this edge.
     */
    virtual void apply_backward(const Tensor<value_t>& grad_output) = 0;
};

} // namespace temper

#endif // TEMPER_AUTOGRAD_HPP