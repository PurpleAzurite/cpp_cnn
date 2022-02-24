# ------------------------------------------------------------------------------
# CMake
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------
project(starter_project LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Project Standards
# ------------------------------------------------------------------------------
add_library(projectStandards INTERFACE)
target_compile_features(projectStandards INTERFACE cxx_std_20)
target_compile_options(projectStandards INTERFACE
  -Wall
  -Wextra
  -Wpedantic
  -Wnon-virtual-dtor
  -Wcast-align
  -Wconversion
  -Wsign-conversion
  -Wnull-dereference
  -Wdouble-promotion
  -Wformat=2
)

# ------------------------------------------------------------------------------
# Vendor
# ------------------------------------------------------------------------------
add_subdirectory(vendor/spdlog)
add_subdirectory(vendor/glfw)
add_subdirectory(vendor/imgui)

# ------------------------------------------------------------------------------
# Includes
# ------------------------------------------------------------------------------
include_directories(vendor/spdlog/include)
include_directories(vendor/glfw/include)
include_directories(vendor/glad/include)
include_directories(vendor/imgui)

# ------------------------------------------------------------------------------
# Source Files
# ------------------------------------------------------------------------------
set(VENDOR_SOURCES
  vendor/glad/src/glad.c
  vendor/imgui/backends/imgui_impl_opengl3.cpp
  vendor/imgui/backends/imgui_impl_opengl3.h
  vendor/imgui/backends/imgui_impl_glfw.cpp
  vendor/imgui/backends/imgui_impl_glfw.h
)

set(SOURCE_FILES
  src/Main.cpp

  src/Logging.cpp
  src/Logging.h

  src/Application.cpp
  src/Application.h

  src/Events/Event.h
  src/Events/WindowEvents.h
  src/Events/MouseEvents.h
  src/Events/KeyEvents.h

  src/Window.cpp
  src/Window.h

  src/Renderer.h

  src/Layers/Layer.h
  src/Layers/LayerStack.cpp
  src/Layers/LayerStack.h
  src/Layers/ImGuiLayer.cpp
  src/Layers/ImGuiLayer.h
  src/Layers/MainLayer.cpp
  src/Layers/MainLayer.h
)

# ------------------------------------------------------------------------------
# Target
# ------------------------------------------------------------------------------
add_executable(project ${SOURCE_FILES} ${VENDOR_SOURCES})

# ------------------------------------------------------------------------------
# Definition
# ------------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(project PRIVATE DEBUG_BUILD)
endif()

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------
target_link_libraries(project PRIVATE projectStandards)
target_link_libraries(project PRIVATE spdlog)
target_link_libraries(project PRIVATE GL)
target_link_libraries(project PRIVATE glfw)
target_link_libraries(project PRIVATE imgui)
