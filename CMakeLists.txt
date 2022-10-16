cmake_minimum_required(VERSION 3.10)

project(ing VERSION 0.1)

set(TEST FALSE)
set(BUILD_BASIC TRUE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

if(${BUILD_BASIC})
add_definitions("-DUNICODE" "-D_UNICODE")
endif()

list(APPEND ING_INCLUDE_DIR
    "${PROJECT_SOURCE_DIR}/include"
)

if(${TEST})
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