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
     *
     * Initializes an operation node with a unique name but no stored
     * input or output tensors.
     *
     * @param op_name Unique identifier (e.g., "add", "matmul").
     */
    explicit FunctionEdge(std::string op_name)
        : m_op_name(std::move(op_name)) {}

    /**
     * @brief Construct an edge with an operation name and connected tensors.
     *
     * Initializes an operation node with a unique name, the input tensors
     * consumed by the operation, and an optional weak reference to the
     * produced output tensor.
     *
     * @param op_name Unique identifier (e.g., "add", "matmul").
     * @param inputs Input tensors consumed by this operation.
     * @param output Output tensor produced by this operation.
     */
    FunctionEdge(std::string op_name,
        std::vector<std::shared_ptr<Tensor<value_t>>> inputs,
        std::weak_ptr<Tensor<value_t>> output = {})
        : m_op_name(std::move(op_name)),
          m_inputs(std::move(inputs)),
          m_output(std::move(output)) {}

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
    inputs() const
    {
        return m_inputs;
    }

    /**
     * @brief Get the output tensor produced by this edge.
     * @return A shared pointer to the resulting tensor.
     */
    virtual std::shared_ptr<Tensor<value_t>> output() const
    {
        return m_output.lock();
    }

protected:
    /**
     * @brief Replace the currently stored input tensors.
     *
     * Updates the stable references to the tensors consumed by this edge.
     * Intended for use by derived operation nodes during setup.
     *
     * @param inputs New input tensor list.
     */
    void set_inputs(std::vector<std::shared_ptr<Tensor<value_t>>> inputs)
    {
        m_inputs = std::move(inputs);
    }

    /**
     * @brief Replace the currently stored output tensor.
     *
     * Updates the weak reference to the tensor produced by this edge.
     * Intended for use by derived operation nodes during setup.
     *
     * @param output New weak reference to the output tensor.
     */
    void set_output(std::weak_ptr<Tensor<value_t>> output)
    {
        m_output = std::move(output);
    }

    /// Unique identifier for serialization.
    std::string m_op_name;

    /// Stable references to the tensors consumed by this operation.
    std::vector<std::shared_ptr<Tensor<value_t>>> m_inputs{};

    /// Weak reference to the tensor produced by this operation.
    std::weak_ptr<Tensor<value_t>> m_output{};
};

/**
 * @brief Struct to hold autograd metadata for a tensor.
 *
 * This struct encapsulates the information needed for automatic
 * differentiation, including the function edge that produced the
 * tensor, the gradient tensor, and whether gradients are required.
 */
template <typename value_t>
struct AutogradMeta
{
    /// Pointer to the function edge that produced this tensor.
    std::shared_ptr<FunctionEdge<value_t>> fn{nullptr};

    /// Pointer to the gradient tensor data.
    std::shared_ptr<value_t> grad{nullptr};

    /// Flag indicating whether this tensor requires gradient.
    bool requires_grad{false};
};

/**
 * @brief Autograd edge for element-wise tensor addition.
 *
 * Stores stable references to the two input tensors and an optional weak
 * reference to the produced output tensor via the @ref FunctionEdge base
 * class. Backward propagation logic will be implemented in a later step.
 *
 * @tparam value_t Tensor numeric type.
 */
template<typename value_t>
class AddEdge : public FunctionEdge<value_t>
{
public:
    /**
     * @brief Construct an addition edge.
     *
     * @param lhs Left-hand input tensor.
     * @param rhs Right-hand input tensor.
     * @param out Optional weak reference to the produced output tensor.
     */
    AddEdge(const std::shared_ptr<Tensor<value_t>> & lhs,
            const std::shared_ptr<Tensor<value_t>> & rhs,
            std::weak_ptr<Tensor<value_t>> out = {});

    /**
     * @brief Re-execute the forward pass.
     *
     * Forward replay is not implemented yet for addition edges.
     */
    void forward() override;

    /**
     * @brief Propagate gradients through the addition edge.
     *
     * Backward logic is intentionally deferred to a later step.
     *
     * @param grad_output Gradient of the loss w.r.t. the output.
     */
    void backward(const Tensor<value_t> & grad_output) override;
};

} // namespace temper

#endif // TEMPER_AUTOGRAD_HPP