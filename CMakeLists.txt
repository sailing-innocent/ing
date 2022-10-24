cmake_minimum_required(VERSION 3.10)

project(ing VERSION 0.1)

set(TEST TRUE)
set(PURE_GL FALSE)
set(IMGUI_DRX FALSE)
set(BUILD_BASIC FALSE)
set(BUILD_CV FALSE)

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

if(${BUILD_BASIC})
add_definitions("-DUNICODE" "-D_UNICODE")
endif()


list(APPEND ING_INCLUDE_DIR
    "${PROJECT_SOURCE_DIR}/include"
    "E:/includes"
)

if(${TEST})
    list(APPEND ING_INCLUDE_DIR
        "E:/sdk/Vulkan/Include"
    )
    list(APPEND ING_LIBS_DIR
        "E:/libs/glfw"
        "E:/sdk/Vulkan/Lib"
    )
    list(APPEND ING_LIBS
        "glfw3"
        "vulkan-1"
    )
    list(APPEND ING_SOURCE
        "ing/utils/vk_utils.cc"
        "ing/app/win_vulkan/vk_common.cc"
        "ing/app/win_vulkan/vk_hello.cc"
        "examples/static/vk_hello_main.cc"
    )
    add_executable(ing ${ING_SOURCE})
    target_link_directories(ing PUBLIC ${ING_LIBS_DIR})
    target_link_libraries(ing PUBLIC ${ING_LIBS})
    target_include_directories(ing PUBLIC ${ING_INCLUDE_DIR})
elseif(${PURE_GL})
    list(APPEND ING_LIBS_DIR
        "E:/libs/glfw"
    )
    list(APPEND ING_LIBS
        "glfw3"
    )
    list(APPEND ING_SOURCE
        "E:/source/glad/glad.c"
        "examples/plain/pure_gl.cc"
    )
    add_executable(ing ${ING_SOURCE})
    target_link_directories(ing PUBLIC ${ING_LIBS_DIR})
    target_link_libraries(ing PUBLIC ${ING_LIBS})
    target_include_directories(ing PUBLIC ${ING_INCLUDE_DIR})
elseif(${IMGUI_DRX})
    set(IMGUI_DIR "E:/source/imgui")
    list(APPEND IMGUI_SOURCE
        "${IMGUI_DIR}/backends/imgui_impl_dx11.cpp"
        "${IMGUI_DIR}/backends/imgui_impl_win32.cpp"
        "${IMGUI_DIR}/imgui_draw.cpp"
        "${IMGUI_DIR}/imgui_tables.cpp"
        "${IMGUI_DIR}/imgui_widgets.cpp"
        "${IMGUI_DIR}/imgui_demo.cpp"
        "${IMGUI_DIR}/imgui.cpp"
    )
    list(APPEND ING_INCLUDE_DIR
        "${IMGUI_DIR}/"
        "${IMGUI_DIR}/backends"
    )
    list(APPEND ING_SOURCE
        ${IMGUI_SOURCE}
        "examples/static/ui.cpp"
    )
    list(APPEND ING_LIB
        d3d11
    )
    add_executable(ing ${ING_SOURCE})
    target_include_directories(ing PUBLIC ${ING_INCLUDE_DIR})
    target_link_libraries(ing PUBLIC ${ING_LIB})
elseif(${BUILD_CV})
    add_executable(ing "example/image/opencv_impl.cc")
elseif(${BUILD_BASIC})
    message("IS BUILDING BASIC")

    list(APPEND WIN_APP_SOURCE
        # "ing/app/win_app.cc"
        "ing/app/win_pure.cc"
        "examples/plain/pure_win.cc"
    )
    add_executable(ing WIN32 ${WIN_APP_SOURCE})
    target_include_directories(ing PUBLIC ${ING_INCLUDE_DIR})
else()
endif()