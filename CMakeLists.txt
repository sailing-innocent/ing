cmake_minimum_required(VERSION 3.10)

project(ing VERSION 0.1)

set(TEST TRUE)
set(BUILD_BASIC FALSE)
set(PURE_WIN32 FALSE)

if(${BUILD_BASIC})
add_definitions("-DUNICODE" "-D_UNICODE")
endif()

if(${TEST})
    add_executable(ing "example/image/opencv_impl.cc")
else()
    add_executable(ing WIN32 "ing/app/win_app.cc")
endif()