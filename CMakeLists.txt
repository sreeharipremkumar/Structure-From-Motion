cmake_minimum_required(VERSION 4.0)
project(OpenCVSFM)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_ICLUDE_DIRS})

add_executable(OpenCVSFM main.cpp)
set_target_properties(OpenCVSFM PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")

target_link_libraries(OpenCVSFM ${OpenCV_LIBS})