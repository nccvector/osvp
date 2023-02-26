cmake_minimum_required(VERSION 3.0.0)

set (CMAKE_BUILD_TYPE "Debug" CACHE
      STRING "Choose the type of build.")

project(abc)

# C++17 is required for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LOCAL_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/ext/")
message("LOCAL_3RDPARTY = " "${LOCAL_3RDPARTY}")

set(CMAKE_MODULE_PATH "${LOCAL_3RDPARTY}/CMake")
message("CMAKE_MODULE_PATH = " "${CMAKE_MODULE_PATH}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)
find_package(DevIL_1_8_0 REQUIRED)
find_package(CUDA REQUIRED)


set(OPTIX_INCLUDE_DIR "/home/vector/optix/include")

include_directories(
  "."
  "inc"
  "sutil"
  "ext"
  "ext/imgui"
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${IL_INCLUDE_DIR}
)

# OptiX SDK 7.5.0 and CUDA 11.7 added support for a new OptiX IR target, which is a binary intermediate format for the module input.
# The default module build target is PTX.
set(USE_OPTIX_IR FALSE)
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

set( IMGUI
  ext/imgui/imgui.cpp
  ext/imgui/imgui_widgets.cpp
  ext/imgui/imgui_tables.cpp
  ext/imgui/imgui_demo.cpp
  ext/imgui/imgui_draw.cpp
  ext/imgui/imgui_impl_glfw.cpp
  ext/imgui/imgui_impl_opengl3.cpp
)

set( SOURCES
  src/main.cpp
  sutil/Aabb.h
  sutil/Camera.cpp
  sutil/Camera.h
  sutil/CUDAOutputBuffer.h
  sutil/Exception.h
  sutil/GLDisplay.cpp
  sutil/GLDisplay.h
  sutil/Matrix.h
  sutil/PPMLoader.cpp
  sutil/PPMLoader.h
  sutil/Preprocessor.h
  sutil/Quaternion.h
  sutil/Record.h
  sutil/sutilapi.h
  sutil/sutil.cpp
  sutil/sutil.h
  sutil/Trackball.cpp
  sutil/Trackball.h
  sutil/vec_math.h
)

add_executable( ${PROJECT_NAME}
  ${IMGUI}
  ${SOURCES}
)

target_link_libraries( ${PROJECT_NAME} PUBLIC
  OpenGL::GL
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::nvrtc
  ${IL_LIBRARIES}
  ${ILU_LIBRARIES}
  ${ILUT_LIBRARIES}
)

