cmake_minimum_required (VERSION 3.19)
project (Temper)

#### setup

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Debug")
endif ()
if (NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND
    NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE value (${CMAKE_BUILD_TYPE})."
                        "Expected values: {Release, Debug}.")
endif ()

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror -fsycl")

message(STATUS "Checking for available GPU")

execute_process(
  COMMAND sh -c "rocm_agent_enumerator | awk 'NR==1'"
  OUTPUT_VARIABLE ROCMGPU
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT ROCMGPU STREQUAL "")
    message(STATUS "Detected GPU: ${ROCMGPU}")
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${ROCMGPU}")
else ()
    message(STATUS "Detected GPU is not AMD")
    message(FATAL_ERROR "todo")
endif ()

#### compile

add_subdirectory (include)

add_subdirectory (src)

#### test

enable_testing ()
add_subdirectory (tests)