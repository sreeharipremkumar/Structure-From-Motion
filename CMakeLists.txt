cmake_minimum_required(VERSION 3.10)  # No need to go back to 4.0 — 3.10+ is modern & safe
project(OpenCVSFM)

set(CMAKE_CXX_STANDARD 17)

# Optional: If you installed PCL using the All-in-One Windows installer
set(PCL_DIR "C:/Program Files/PCL 1.15.0/cmake")

# Help CMake find Boost inside the PCL package
set(BOOST_ROOT "C:/Program Files/PCL 1.15.0/3rdParty/Boost")
set(Boost_INCLUDE_DIR "C:/Program Files/PCL 1.15.0/3rdParty/Boost/include")
set(Boost_LIBRARY_DIR "C:/Program Files/PCL 1.15.0/3rdParty/Boost/lib")
set(Boost_NO_SYSTEM_PATHS ON)

# Find OpenCV and Boost before PCL
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(PCL REQUIRED)

# Include dirs
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})

# Link dirs
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIR})

# Definitions
add_definitions(${PCL_DEFINITIONS})

# Executable
add_executable(OpenCVSFM main.cpp)
set_target_properties(OpenCVSFM PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")

# Link libraries
target_link_libraries(OpenCVSFM ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES})
