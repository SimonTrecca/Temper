/**
 * @file Autograd.cpp
 * @brief Autograd edge implementations.
 */

#include "temper/Autograd.hpp"

namespace temper
{

template<typename value_t>
AddEdge<value_t>::AddEdge(const std::shared_ptr<Tensor<value_t>> & lhs,
                         const std::shared_ptr<Tensor<value_t>> & rhs,
                         std::weak_ptr<Tensor<value_t>> out)
    : FunctionEdge<value_t>("add", {lhs, rhs}, std::move(out))
{
    // No-op; all state is initialized by the base class.
}

template<typename value_t>
void AddEdge<value_t>::forward()
{
    // TODO: Recompute the output tensor from the stored inputs.
}

template<typename value_t>
void AddEdge<value_t>::backward(const Tensor<value_t> & grad_output)
{
    (void)grad_output;
    // TODO: Accumulate grad_output into both input tensors,
    //       reducing across broadcasted dimensions as needed.
}

template class AddEdge<float>;
template class AddEdge<uint64_t>;

} // namespace temper