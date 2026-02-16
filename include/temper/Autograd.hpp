/**
 * @file Autograd.hpp
 * @brief Base interfaces for the functional graph and autograd system.
 *
 * Defines FunctionEdge, representing an operation node in the graph.
 * It handles forward re-execution and backward propagation logic.
 */

#ifndef TEMPER_AUTOGRAD_HPP
#define TEMPER_AUTOGRAD_HPP

#include <vector>
#include <memory>
#include <string>
#include <utility>

namespace temper
{

/**
 * @brief Forward declaration of Tensor.
 * Necessary to allow FunctionEdge to reference Tensors without
 * creating a circular include dependency.
 */
template <typename value_t>
class Tensor;

/**
 * @brief Base interface for a functional edge in the graph.
 *
 * A FunctionEdge represents the mathematical relationship between
 * input tensors and their output. It stores context to re-run the
 * calculation (forward) or compute gradients (backward).
 */
template <typename value_t>
class FunctionEdge
{
public:
    /**
     * @brief Construct an edge with a mandatory operation name.
     * @param op_name Unique identifier (e.g., "add", "matmul").
     */
    explicit FunctionEdge(std::string op_name)
        : m_op_name(std::move(op_name)) {}

    /**
     * @brief Virtual destructor for safe inheritance.
     */
    virtual ~FunctionEdge() = default;

    /**
     * @brief Get the unique name of the operation.
     * Used by serialization to identify the operation type.
     * @return A constant reference to the name string.
     */
    const std::string& name() const { return m_op_name; }

    /**
     * @brief Re-execute the forward pass.
     * Uses stored input tensors to re-calculate the output.
     */
    virtual void forward() = 0;

    /**
     * @brief Compute and propagate gradients through this edge.
     * Implements the chain rule, calculates local derivatives
     * and updates the gradients of the input tensors.
     *
     * @param grad_output Gradient of the loss w.r.t. the output.
     */
    virtual void backward(const Tensor<value_t>& grad_output) = 0;

    /**
     * @brief Get the input tensors connected by this edge.
     * @return A vector of shared pointers to the input tensors.
     */
    virtual std::vector<std::shared_ptr<Tensor<value_t>>>
    inputs() const = 0;

    /**
     * @brief Get the output tensor produced by this edge.
     * @return A shared pointer to the resulting tensor.
     */
    virtual std::shared_ptr<Tensor<value_t>> output() const = 0;

protected:
    std::string m_op_name; ///< Unique identifier for serialization.
};

} // namespace temper

#endif // TEMPER_AUTOGRAD_HPP