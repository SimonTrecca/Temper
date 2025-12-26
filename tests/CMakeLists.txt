if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  message(STATUS "IntelLLVM detected: sanitizing CMAKE_CXX_FLAGS…")

  string(REGEX REPLACE "-Wno-implicit-float-size-conversion" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-Winline"                             "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  foreach(cfg IN ITEMS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
    if (DEFINED ${cfg})
      string(REGEX REPLACE "-Wno-implicit-float-size-conversion" "" ${cfg} "${${cfg}}")
      string(REGEX REPLACE "-Winline"                             "" ${cfg} "${${cfg}}")
    endif()
  endforeach()
  add_compile_options(-w)
endif()

find_package(GTest QUIET)

if (NOT GTest_FOUND)
  message(STATUS "GoogleTest not found, downloading...")
  include(FetchContent)
  set(gtest_disable_werror ON CACHE BOOL "Disables Werror in GTest" FORCE)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_executable(temper_ut
  ut_Tensor.cpp
  ut_Math.cpp
  ut_ML.cpp
  ut_NN.cpp
  ut_Stats.cpp
  ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
  ${CMAKE_SOURCE_DIR}/src/SYCLQueue.cpp
  ${CMAKE_SOURCE_DIR}/src/Math.cpp
  ${CMAKE_SOURCE_DIR}/src/ML.cpp
  ${CMAKE_SOURCE_DIR}/src/NN.cpp
  ${CMAKE_SOURCE_DIR}/src/Stats.cpp
)

add_executable(tensor_ut
  ut_Tensor.cpp
  ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
  ${CMAKE_SOURCE_DIR}/src/SYCLQueue.cpp
  ${CMAKE_SOURCE_DIR}/src/Math.cpp
  ${CMAKE_SOURCE_DIR}/src/Stats.cpp
)

add_executable(math_ut
  ut_Math.cpp
  ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
  ${CMAKE_SOURCE_DIR}/src/SYCLQueue.cpp
  ${CMAKE_SOURCE_DIR}/src/Math.cpp
  ${CMAKE_SOURCE_DIR}/src/Stats.cpp
)

add_executable(ml_ut
  ut_ML.cpp
  ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
  ${CMAKE_SOURCE_DIR}/src/SYCLQueue.cpp
  ${CMAKE_SOURCE_DIR}/src/Math.cpp
  ${CMAKE_SOURCE_DIR}/src/ML.cpp
  ${CMAKE_SOURCE_DIR}/src/Stats.cpp
)

add_executable(stats_ut
  ut_Stats.cpp
  ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
  ${CMAKE_SOURCE_DIR}/src/SYCLQueue.cpp
  ${CMAKE_SOURCE_DIR}/src/Math.cpp
  ${CMAKE_SOURCE_DIR}/src/Stats.cpp
)

add_executable(nn_ut
  ut_NN.cpp
  ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
  ${CMAKE_SOURCE_DIR}/src/SYCLQueue.cpp
  ${CMAKE_SOURCE_DIR}/src/Math.cpp
  ${CMAKE_SOURCE_DIR}/src/NN.cpp
)

target_compile_definitions(temper_ut PRIVATE UNIT_TEST)
target_compile_definitions(tensor_ut PRIVATE UNIT_TEST)
target_compile_definitions(ml_ut PRIVATE UNIT_TEST)
target_compile_definitions(stats_ut PRIVATE UNIT_TEST)

target_link_libraries(
  temper_ut
  PRIVATE
    Common
    GTest::gtest
    GTest::gtest_main
)

target_link_libraries(
  tensor_ut
  PRIVATE
    Common
    GTest::gtest
    GTest::gtest_main
)

target_link_libraries(
  math_ut
  PRIVATE
    Common
    GTest::gtest
    GTest::gtest_main
)


target_link_libraries(
  ml_ut
  PRIVATE
    Common
    GTest::gtest
    GTest::gtest_main
)

target_link_libraries(
  stats_ut
  PRIVATE
    Common
    GTest::gtest
    GTest::gtest_main
)

target_link_libraries(
  nn_ut
  PRIVATE
    Common
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(temper_ut)
gtest_discover_tests(tensor_ut)
gtest_discover_tests(math_ut)
gtest_discover_tests(ml_ut)
gtest_discover_tests(stats_ut)
gtest_discover_tests(nn_ut)