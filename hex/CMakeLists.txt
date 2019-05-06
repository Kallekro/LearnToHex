cmake_minimum_required(VERSION 2.8)

project(hex)

# Find the Shark libraries and includes
# set Shark_DIR to the proper location of Shark
find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

# Executable
add_executable(hex main.cpp Hex.hpp)
set_property(TARGET hex PROPERTY CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
target_link_libraries(hex ${SHARK_LIBRARIES})

# Executable for python application
add_executable(hex_python main.cpp Hex.hpp)
set_property(TARGET hex_python PROPERTY CXX_STANDARD 11)
target_compile_definitions(hex_python PRIVATE BUILD_FOR_PYTHON)
target_link_libraries(hex_python ${SHARK_LIBRARIES})