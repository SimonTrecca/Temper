/**
 * @file SYCLQueue.hpp
 * @brief Declaration of the global SYCL queue.
 *
 * This header declares a global `sycl::queue` variable that can be used
 * throughout the project, avoiding the need to pass it explicitly
 * between functions.
 */

#ifndef SYCLQUEUE_HPP
#define SYCLQUEUE_HPP

#include <sycl/sycl.hpp>

/**
 * @brief Global SYCL queue used for all asynchronous operations.
 */
extern sycl::queue g_sycl_queue;

#endif // SYCLQUEUE_HPP