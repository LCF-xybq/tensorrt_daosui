cmake_minimum_required(VERSION 3.18)
project(tensorrt_daosui_py LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA ──
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# ── TensorRT ──
set(TRT_ROOT "/home/lcf/Downloads/TensorRT-10.10.0.31")
include_directories(${TRT_ROOT}/include)
link_directories(${TRT_ROOT}/lib)

# ── Python & pybind11 ──
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Python root")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

set(pybind11_DIR "/home/lcf/Downloads/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# ── OpenCV ──
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# ── Sources ──
set(PYBIND_SOURCES
    pybind_module.cc
    tensorrt_inference.cc
    yolo_postprocess.cc
)

set(CUDA_SOURCES
    preprocess.cu
)

pybind11_add_module(_tensorrt_daosui ${PYBIND_SOURCES} ${CUDA_SOURCES})

target_include_directories(_tensorrt_daosui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(_tensorrt_daosui PRIVATE
    nvinfer
    cudart
    nppc
    nppig
    nppial
    ${OpenCV_LIBS}
)

set_target_properties(_tensorrt_daosui PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Set RPATH so the built module can find TensorRT libs at runtime
set_target_properties(_tensorrt_daosui PROPERTIES
    BUILD_RPATH "${TRT_ROOT}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    INSTALL_RPATH "${TRT_ROOT}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64"
)

install(TARGETS _tensorrt_daosui DESTINATION tensorrt_daosui)
