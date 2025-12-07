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
        : std::invalid_argument(message) {}
};

/**
 * @brief non finite error class for temper library.
 */
class nonfinite_error : public std::overflow_error {
public:
    explicit nonfinite_error(const std::string& msg)
        : std::overflow_error(msg) {}
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

#endif // TEMPER_ERRORS_HPP