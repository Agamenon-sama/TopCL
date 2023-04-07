cmake_minimum_required(VERSION 3.24) # should need less
project("top88" VERSION 0.1)

# Settings
set(CMAKE_CXX_STANDARD 17) # use C++17

# Add src files
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Add header files
file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp
)

# Define executable
add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES})

# External dependencies
find_package(OpenCL REQUIRED)
message(STATUS "found OpenCL")
find_package(Threads REQUIRED)
message(STATUS "Found a threading library")

# Set libraries
set(LIBS OpenCL Threads::Threads)

# Linking libraries
target_link_libraries(${PROJECT_NAME} ${LIBS})
