cmake_minimum_required(VERSION 3.10)

project(ing VERSION 0.1)

set(TEST TRUE)
set(BUILD_BASIC FALSE)
set(BUILD_CV FALSE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

if(${BUILD_BASIC})
add_definitions("-DUNICODE" "-D_UNICODE")
endif()

list(APPEND ING_INCLUDE_DIR
    "${PROJECT_SOURCE_DIR}/include"
)

if(${TEST})
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