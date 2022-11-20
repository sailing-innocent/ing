cmake_minimum_required(VERSION 3.18)

project(ing VERSION 0.1)

################################## BUILD CONFIG ##################################

set(BUILD_EXAMPLES TRUE)
set(BUILD_TESTBED FALSE)

####################################################################################

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
else()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No release type specified. Setting to 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

list(APPEND ING_INCLUDE_DIR
    "${PROJECT_SOURCE_DIR}/include"
    "E:/includes"
    "E:/sdk/Vulkan/Include"
)

file(GLOB LIBING_SOURCE CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ing/**/*.cc)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${LIBING_SOURCE})

list(APPEND ING_SOURCE
    "E:/source/glad/glad.c"
    "ing/core/timer.cc"
    "ing/utils/common_utils.cc"
    "ing/app/win_gl/gl_common.cc"
    "ing/utils/gl_utils.cc"
    "ing/utils/gl_shader.cc"
    "ing/utils/vk_utils.cc"
    "ing/app/win_vulkan/vk_common.cc"
    "ing/app/win_vulkan/vk_hello.cc"
    "ing/app/win_vulkan/vk_canvas.cc"
    "ing/app/win_vulkan/vk_scene.cc"
)

list(APPEND ING_LIBS_DIR
    "E:/libs/glfw"
    "E:/sdk/Vulkan/Lib"
)

list(APPEND ING_LIBS
    "glfw3"
    "vulkan-1"
)

add_library(libing ${ING_SOURCE})
target_link_directories(libing PUBLIC ${ING_LIBS_DIR})
target_link_libraries(libing PUBLIC ${ING_LIBS})
target_include_directories(libing PUBLIC ${ING_INCLUDE_DIR})

if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()

if (${BUILD_TESTBED})
    add_subdirectory(testbed)
endif()
