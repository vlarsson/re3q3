cmake_minimum_required (VERSION 3.1)
project(re3q3)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package (Eigen3 REQUIRED NO_MODULE)

add_library(re3q3 STATIC 
	re3q3/re3q3.cc
	re3q3/sturm8.cc
)
target_link_libraries(re3q3 Eigen3::Eigen)

add_executable(test_re3q3 test_re3q3.cc)
target_link_libraries(test_re3q3 re3q3 Eigen3::Eigen)

if(MSVC)	
	target_compile_options(re3q3 PRIVATE /bigobj /fp:fast)
	target_compile_options(test_re3q3 PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(re3q3 PRIVATE -march=native -Wall -Werror -Wno-sign-compare -Wno-unused-variable -ffast-math)
	target_compile_options(test_re3q3 PRIVATE -march=native -Wall -Werror -Wno-sign-compare -Wno-unused-variable -ffast-math)
endif()