/**
 * @file SYCLQueue.cpp
 * @brief Global SYCL queue definition.
 */

#include "temper/SYCLQueue.hpp"

namespace temper {

sycl::queue g_sycl_queue{ sycl::default_selector_v };

} // namespace temper