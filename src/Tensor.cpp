#include "temper/Tensor.hpp"
#include <iostream>

template<typename float_t>
Tensor<float_t>::~Tensor()
{
	// sycl::free(data);
}

template class Tensor<float>;