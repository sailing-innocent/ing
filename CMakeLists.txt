cmake_minimum_required(VERSION 3.18)

project(ing 
    VERSION 0.2
    DESCRIPTION "The ING Graphics Library"
    LANGUAGES C CUDA CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

################################## BUILD CONFIG ##################################

set(BUILD_EXAMPLES TRUE)
set(BUILD_TESTBED TRUE)
set(CUDA_ENABLED TRUE)

####################################################################################

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
else()
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No release type specified. Setting to 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()


###################### DEPS DEFINITION ###########################

set(MAIN_INCLUDES "D:/includes")
set(MAIN_LIBS "D:/libs")

set(IMGUI_DIR "D:/source/imgui/")
set(IMGUI_INCLUDE "${IMGUI_DIR}")
list(APPEND IMGUI_SOURCE
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
)

set(LIBSAIL_INCLUDE "D:/repos/sail/include")
set(LIBSAIL_DIR "D:/repos/sail/build/Debug")
set(LIBING_INCLUDE "D:/repos/ing/include")
set(LIBING_DIR "D:/repos/ing/build/Debug")

set(EIGEN_DIR "D:/libs/eigen/")
set(GLAD_SOURCE "D:/source/glad/glad.c")

#################################################################

list(APPEND ING_INCLUDE_DIR
    "${PROJECT_SOURCE_DIR}/include"
    "D:/includes"
    "D:/libs/vulkan/Include"
)

file(GLOB_RECURSE LIBING_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ing/*.cc)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${LIBING_SOURCE})

list(APPEND ING_SOURCE
    ${GLAD_SOURCE}
    ${LIBING_SOURCE}
)

list(APPEND ING_LIBS_DIR
    "D:/libs/glfw"
    "D:/libs/vulkan/Lib"
)

list(APPEND ING_LIBS
    "glfw3"
    "vulkan-1"
)

add_library(libing ${ING_SOURCE})
target_link_directories(libing PUBLIC ${ING_LIBS_DIR})
target_link_libraries(libing PUBLIC ${ING_LIBS})
target_include_directories(libing PUBLIC ${ING_INCLUDE_DIR})


if (${BUILD_TESTBED})
    add_subdirectory(testbed)
endif()

if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()
