cmake_minimum_required(VERSION 3.22)
project(fcw_host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FCW_BOARD "Build the KV260 backends (VART, XRT)" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d video videoio highgui)
find_package(Threads REQUIRED)

file(GLOB FCW_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp)
list(FILTER FCW_SOURCES EXCLUDE REGEX ".*/app_main\\.cpp$")
add_library(fcw STATIC ${FCW_SOURCES})
target_include_directories(fcw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fcw PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(fcw PRIVATE -Wall -Wextra)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/app_main.cpp)
  add_executable(fcw_app ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/app_main.cpp)
  target_link_libraries(fcw_app PRIVATE fcw)
endif()

# CPU-only baseline (no FPGA, no DPU) for the latency and energy comparison
add_executable(fcw_cpu_baseline ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpu_baseline.cpp)
target_link_libraries(fcw_cpu_baseline PRIVATE fcw)

enable_testing()
find_package(GTest REQUIRED)
file(GLOB FCW_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
add_executable(fcw_tests ${FCW_TESTS})
target_link_libraries(fcw_tests PRIVATE fcw GTest::gtest_main)
target_compile_definitions(fcw_tests PRIVATE FCW_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
                                           FCW_GOLDEN_LOCAL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden_local")
include(GoogleTest)
gtest_discover_tests(fcw_tests)
