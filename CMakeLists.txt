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
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror")

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl")

  message(STATUS "Checking for available GPU")

  execute_process(
    COMMAND sh -c "rocm_agent_enumerator | awk 'NR==1'"
    OUTPUT_VARIABLE ROCMGPU
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (NOT ROCMGPU STREQUAL "")
      message(STATUS "Detected AMD GPU: ${ROCMGPU}")
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${ROCMGPU}")
  elseif (EXISTS "/usr/bin/nvidia-smi")
      execute_process(
          COMMAND sh -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR==1 {split($1,a,\".\"); printf(\"sm_%d%d\", a[1], a[2])}'"
          OUTPUT_VARIABLE CUDA_ARCH
          OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      message(STATUS "Detected NVIDIA GPU: ${CUDA_ARCH}")
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=${CUDA_ARCH}")
  else()
      execute_process(
        COMMAND sh -c "sycl-ls | grep -i intel"
        OUTPUT_VARIABLE INTEL_PLATFORM
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if (NOT INTEL_PLATFORM STREQUAL "")
          message(STATUS "Detected Intel GPU")
          set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl-targets=spir64_gen")
      else()
          message(FATAL_ERROR "No supported GPU detected. Please install ROCm, NVIDIA, or Intel oneAPI drivers.")
      endif()
  endif()

endif ()


#### compile

add_subdirectory (include)

add_subdirectory (src)

#### test

enable_testing ()
add_subdirectory (tests)

#### doxygen

add_subdirectory (docs)