cmake_minimum_required(VERSION 3.5.1)
project(learn_cuda LANGUAGES CXX)

# open cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} /usr/local/cuda/include cuda cpu)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/ub/workspace/Toolkits/TensorRT-7.0.0.11/include)
link_directories(/home/ub/workspace/Toolkits/TensorRT-7.0.0.11/lib)

# yololayer
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/tensorrt/yolov5/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

set(cuda_file cuda/cudamn.h
              cuda/cudamn.cu
              cuda/cuda_initialization.h
              cuda/cuda_initialization.cpp)

set(cpu_file cpu/cpu.h
             cpu/cpu.cpp
             cpu/cpu_initialization.h
             cpu/cpu_initialization.cpp)

set(file_opencv opencv_process/opencv_process.h
                opencv_process/opencv_process.cpp
                opencv_process/opencv_initialization.h
                opencv_process/opencv_initialization.cpp)

set(utils_file utils_cuda/config.h
               utils_cuda/file_process.h
               utils_cuda/file_process.cpp
               utils_cuda/memory_pool.h
               utils_cuda/memory_pool.cpp)

set(yolov5 tensorrt/yolov5/yolov5.h
           tensorrt/yolov5/yolov5.cpp
           tensorrt/yolov5/common.hpp
           tensorrt/yolov5/logging.h
           tensorrt/yolov5/utils_yolov5.h
           tensorrt/yolov5/yololayer.h
           tensorrt/yolov5/yololayer.cu)

#打印调试信息
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "PROJECT_SOURCE_DIR： ${PROJECT_SOURCE_DIR}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


#add_executable(learn_cuda main.cpp cuda/cuda.cpp cuda/cuda.h)
cuda_add_executable(learn_cuda main.cpp ${cuda_file} ${cpu_file} ${file_opencv} ${utils_file} ${yolov5}
                    OPTIONS -std=c++11;-gencode arch=compute_60,code=sm_60;-gencode arch=compute_61,code=sm_61;-gencode arch=compute_70,code=sm_70;-gencode arch=compute_75,code=sm_75)

target_link_libraries(learn_cuda ${OpenCV_LIBS})
target_link_libraries(learn_cuda nvinfer)
target_link_libraries(learn_cuda cudart)
target_link_libraries(learn_cuda myplugins)