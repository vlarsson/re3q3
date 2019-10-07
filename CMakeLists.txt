cmake_minimum_required (VERSION 3.1)
project (re3q3_test)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()


set (CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Werror)
include_directories(".")
find_package (Eigen3 REQUIRED NO_MODULE)
add_executable (test_re3q3 test_re3q3.cpp)
target_link_libraries (test_re3q3 LINK_PUBLIC Eigen3::Eigen)
