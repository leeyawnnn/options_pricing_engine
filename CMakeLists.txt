cmake_minimum_required(VERSION 3.20)

project(
    options_pricing_engine
    VERSION 0.1.0
    DESCRIPTION "Multi-method options pricing library (Black-Scholes, CRR tree, Monte Carlo)"
    LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to a Release build with debug info if the caller did not choose one.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo
                                                 MinSizeRel)
endif()

# Export compile_commands.json for clang-tidy / editor tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Collect build outputs in predictable locations (build/bin, build/lib).
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ------------------------------------------------------------------------------
# Strict warnings (applied to our own targets only, never to dependencies)
# ------------------------------------------------------------------------------
add_library(opt_warnings INTERFACE)
if(MSVC)
    target_compile_options(opt_warnings INTERFACE /W4 /WX /permissive-)
else()
    target_compile_options(opt_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

# ------------------------------------------------------------------------------
# Dependencies via FetchContent
# ------------------------------------------------------------------------------
include(FetchContent)

# Eigen (header-only linear algebra).
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
# We only need the headers; skip Eigen's own tests/blas/etc.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

# Treat Eigen's headers as system headers so our -Werror policy never fires on
# them (Eigen is not -Wpedantic-clean and that is not our concern). Eigen3::Eigen
# is an ALIAS, so resolve the real target before setting properties on it.
get_target_property(_eigen_target Eigen3::Eigen ALIASED_TARGET)
if(NOT _eigen_target)
    set(_eigen_target Eigen3::Eigen)
endif()
get_target_property(_eigen_includes ${_eigen_target} INTERFACE_INCLUDE_DIRECTORIES)
if(_eigen_includes)
    set_target_properties(${_eigen_target} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_eigen_includes}")
endif()

# GoogleTest.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
# Honour our shared CRT on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------------------
# Core library: opt_lib
# ------------------------------------------------------------------------------
add_library(
    opt_lib STATIC
    src/black_scholes.cpp
    src/binomial_tree.cpp
    src/monte_carlo.cpp
    src/greeks.cpp
    src/pricing_grid.cpp)

target_include_directories(opt_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(opt_lib PUBLIC Eigen3::Eigen PRIVATE opt_warnings)
add_library(opt::lib ALIAS opt_lib)

# ------------------------------------------------------------------------------
# CLI demo: price_cli
# ------------------------------------------------------------------------------
add_executable(price_cli apps/price_cli.cpp)
target_link_libraries(price_cli PRIVATE opt_lib opt_warnings)

# ------------------------------------------------------------------------------
# Tests: opt_tests
# ------------------------------------------------------------------------------
enable_testing()
include(GoogleTest)

add_executable(
    opt_tests
    tests/test_option.cpp
    tests/test_black_scholes.cpp
    tests/test_binomial.cpp
    tests/test_monte_carlo.cpp
    tests/test_greeks.cpp
    tests/test_pricing_grid.cpp)

target_link_libraries(opt_tests PRIVATE opt_lib GTest::gtest_main opt_warnings)
gtest_discover_tests(opt_tests)

# ------------------------------------------------------------------------------
# Benchmarks: opt_bench (hand-rolled timer, no external dependency)
# ------------------------------------------------------------------------------
add_executable(opt_bench benchmarks/bench_pricing.cpp)
target_link_libraries(opt_bench PRIVATE opt_lib opt_warnings)
