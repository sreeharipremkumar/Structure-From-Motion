cmake_minimum_required(VERSION 3.10)
project(OpenCVSFM)

set(CMAKE_CXX_STANDARD 17)

# Use OpenCV with viz and PCL via vcpkg
find_package(OpenCV REQUIRED)
# find_package(PCL REQUIRED)
# find_package(Boost REQUIRED)

# Include dirs and definitions (optional, vcpkg handles this too)
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${PCL_INCLUDE_DIRS})
# add_definitions(${PCL_DEFINITIONS})

# Executable
add_executable(OpenCVSFM main.cpp)
set_target_properties(OpenCVSFM PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")

# Link libraries
target_link_libraries(OpenCVSFM ${OpenCV_LIBS})# ${PCL_LIBRARIES} ${Boost_LIBRARIES})\
