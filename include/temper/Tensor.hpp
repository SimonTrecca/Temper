#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>

template <typename float_t>
class Tensor
{

private:

	float_t*				data;
    std::vector<uint64_t> 	dimensions;
    std::vector<uint64_t> 	strides;

public:

	~Tensor();

};

extern template class Tensor<float>;

#endif // TENSOR_HPP