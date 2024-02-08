cmake_minimum_required(VERSION 3.27)
project(Detector)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )

add_executable(Detector main.cpp)
target_link_libraries(Detector ${OpenCV_LIBS})
