/**
 * @file SYCLQueue.cpp
 * @brief Global SYCL queue definition.
 */

#include "temper/SYCLQueue.hpp"

sycl::queue g_sycl_queue{ sycl::default_selector_v };