cmake_minimum_required(VERSION 3.26.0)
project(TorchDemo)

set(CMAKE_CXX_STANDARD 14)
set(Torch_DIR ~/libtorch/share/cmake/Torch)
set(OpenCV_DIR ~/opencv-4.7.0/build)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(TorchDemo main.cpp)
target_link_libraries(TorchDemo ${OpenCV_LIBS} ${TORCH_LIBRARIES} )
