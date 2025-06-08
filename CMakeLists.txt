cmake_minimum_required(VERSION 3.14)
project(RetroCityBuilder VERSION 0.1 LANGUAGES C CXX)

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prevent in-source builds
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Please build out-of-source: mkdir build && cd build && cmake ..")
endif()

# --- Include Directories (global) ---
# Ensure headers for Glad, GLFW, and ImGui are found
include_directories("${CMAKE_SOURCE_DIR}/external/glad/include")
include_directories("${CMAKE_SOURCE_DIR}/external/glfw/include")
include_directories("${CMAKE_SOURCE_DIR}/external/imgui")
include_directories("${CMAKE_SOURCE_DIR}/external/imgui/backends")
include_directories("${CMAKE_SOURCE_DIR}/external/glm")

# --- Third-Party Libraries ---
# GLFW (requires external/glfw/CMakeLists.txt)
add_subdirectory(external/glfw)

# GLAD (C loader)
# Generate the GLAD loader source and headers from https://glad.dav1d.de:
# 1. API: gl (OpenGL), Version: 4.5, Profile: Core, Language: C/C++
# 2. Generate and download ZIP, then extract to external/glad/ so you have:
#    external/glad/src/glad.c
#    external/glad/include/glad/glad.h
#    external/glad/include/KHR/khrplatform.h
add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# Dear ImGui (core + backends)
file(GLOB IMGUI_CORE
    external/imgui/*.cpp
)
file(GLOB IMGUI_BACKENDS
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_CORE} ${IMGUI_BACKENDS})
target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
    external/glfw/include  # for ImGui GLFW headers
)

# --- Application ---
# Explicitly list your .cpp sources so main() is included
set(APP_SOURCES
    src/main.cpp
    # add other .cpp files here
)
add_executable(RetroCityBuilder ${APP_SOURCES})

# Include paths for application
target_include_directories(RetroCityBuilder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/glfw/include
    ${CMAKE_SOURCE_DIR}/external/glad/include
    ${CMAKE_SOURCE_DIR}/external/imgui
    ${CMAKE_SOURCE_DIR}/external/imgui/backends
)

# Link libraries
target_link_libraries(RetroCityBuilder PRIVATE
    glfw
    glad
    imgui
)

# OpenGL
find_package(OpenGL REQUIRED)
if (OPENGL_FOUND)
  target_link_libraries(RetroCityBuilder PRIVATE OpenGL::GL)
else()
  message(FATAL_ERROR "OpenGL not found")
endif()

# Compiler warnings and options
if (MSVC)
  target_compile_options(RetroCityBuilder PRIVATE /W4 /permissive-)
else()
  target_compile_options(RetroCityBuilder PRIVATE -Wall -Wextra -Wpedantic)
endif()
