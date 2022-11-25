cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME basic_testbed)

list(APPEND ING_EXAMPLE_LIBS
    testbed
)

add_executable(${TARGET_NAME} "pure_gl.cc")
target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
target_link_directories(${TARGET_NAME} PUBLIC ${ING_LIBS_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC ${ING_EXAMPLE_LIBS})
