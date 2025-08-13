cmake_minimum_required(VERSION 3.23)

# Set CUDA architecture (adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES 90)

project(HelloCuTe LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Set up your source files
set(SOURCES
    src/main.cu   # Replace with your source files
)

find_package(CUDAToolkit REQUIRED) 

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create the executable
add_executable(hello_cute ${SOURCES})
target_include_directories(hello_cute PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(hello_cute PRIVATE $ENV{HOME}/src/cutlass/include)
target_link_libraries(hello_cute PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(hello_cute PRIVATE CUDA::cublas)