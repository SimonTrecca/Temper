/**
 * @file Errors.hpp
 * @brief Centralized error handling utilities.
 *
 * Provides a macro for runtime error checking that can be
 * disabled at compile-time with TEMPER_DISABLE_ERROR_CHECKS flag.
 */
#ifndef TEMPER_ERRORS_HPP
#define TEMPER_ERRORS_HPP

#include <stdexcept>
#include <string>

namespace temper
{

/**
 * @brief nan error class for temper library.
 */
class nan_error : public std::invalid_argument
{
public:
    explicit nan_error(const std::string& message)
        : std::invalid_argument("NaN Error: " + message) {}
};

/**
 * @brief non finite error class for temper library.
 */
class nonfinite_error : public std::runtime_error
{
public:
    explicit nonfinite_error(const std::string& message)
        : std::runtime_error("Non-finite Error: " + message) {}
};

/**
 * @brief Validation error class for temper library.
 * Used to signal invalid inputs or arguments.
 */
class validation_error : public std::invalid_argument
{
public:
    explicit validation_error(const std::string& message)
        : std::invalid_argument("Validation Error: " + message) {}
};

/**
 * @brief Computation error class for temper library.
 * Used to signal errors during numerical or mathematical computations.
 */
class computation_error : public std::runtime_error
{
public:
    explicit computation_error(const std::string& message)
        : std::runtime_error("Computation Error: " + message) {}
};

/**
 * @brief Bounds error class for temper library.
 * Used to signal index out-of-range or invalid bounds during tensor operations.
 */
class bounds_error : public std::out_of_range
{
public:
    explicit bounds_error(const std::string& message)
        : std::out_of_range("Bounds Error: " + message) {}
};

/**
 * @brief Device-side error class for SYCL kernels in the temper library.
 * Used to signal issues specific to SYCL device execution.
 */
class device_error : public std::runtime_error
{
public:
    explicit device_error(const std::string& message)
        : std::runtime_error("Device Error: " + message) {}
};

} // namespace temper

/**
 * @brief Error checking macro.
 *
 * Evaluates a condition and throws the specified exception type
 * with the given message if the condition is true.
 * Can be disabled at compile-time with TEMPER_DISABLE_ERROR_CHECKS.
 *
 * @param condition The condition to check (throws if true)
 * @param exception_type The exception type to throw
 * @param message The error message
 *
 * Usage:
 *   TEMPER_CHECK(size <= 0, std::invalid_argument, "Size must be positive");
 *   TEMPER_CHECK(ptr == nullptr, std::runtime_error, "Null pointer");
 */
#ifndef TEMPER_DISABLE_ERROR_CHECKS
  #define TEMPER_CHECK(condition, exception_type, message) \
   do \
   { \
      if (condition) \
      { \
         throw exception_type(message); \
      } \
   } while(0)
#else
  #define TEMPER_CHECK(condition, exception_type, message) ((void)0)
#endif

/**
 * @brief Device-side error checking macro.
 *
 * Evaluates a condition inside a SYCL kernel and, if true:
 *   - atomically sets an error flag to the specified error code
 *   - immediately returns from the current work-item
 *
 * When TEMPER_DISABLE_ERROR_CHECKS is defined, the macro does nothing.
 *
 * @param condition The condition to evaluate
 * @param p_err Pointer to an int32_t error flag in global/shared memory
 * @param code Error code to atomically set when condition is true
 *
 * Usage inside a SYCL kernel:
 *   TEMPER_DEVICE_CHECK(is_nan(av), p_error, 1);
 *   TEMPER_DEVICE_CHECK(!is_finite(out), p_error, 2);
 */
#ifndef TEMPER_DISABLE_ERROR_CHECKS
  #define TEMPER_DEVICE_CHECK(condition, p_err, code) \
    do \
    { \
        if (condition) \
        { \
            auto atomic_err = sycl::atomic_ref<int32_t, \
                sycl::memory_order::relaxed, \
                sycl::memory_scope::device, \
                sycl::access::address_space::global_space>(*p_err); \
            int32_t expected = 0; \
            atomic_err.compare_exchange_strong(expected, code); \
            return; /* exit current kernel work-item */ \
        } \
    } while (0)
#else
  #define TEMPER_DEVICE_CHECK(condition, p_err, code) ((void)0)
#endif

#endif // TEMPER_ERRORS_HPP