cmake_minimum_required(VERSION 2.8)

project(polebench)

# Find the Shark libraries and includes
# set Shark_DIR to the proper location of Shark
find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

# Executable
add_executable(polebench main.cpp)
set_property(TARGET polebench PROPERTY CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
target_link_libraries(polebench ${SHARK_LIBRARIES})

