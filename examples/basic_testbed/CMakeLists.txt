cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME basic_testbed)

list(APPEND ING_EXAMPLE_LIBS
    testbed
)

# add_executable(${TARGET_NAME} "basic/pure_gl.cc")
add_executable(${TARGET_NAME} "basic/cuda_gl.cu")
# add_executable(${TARGET_NAME} "basic/cuda_gl_extent.cu")
target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
target_link_directories(${TARGET_NAME} PUBLIC ${ING_LIBS_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC ${ING_EXAMPLE_LIBS})

set(TARGET_NAME raytracer)
# add_executable(${TARGET_NAME} "basic/pure_gl.cc")
add_executable(${TARGET_NAME} "raytracer/basic_tracer.cu")
target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
target_link_directories(${TARGET_NAME} PUBLIC ${ING_LIBS_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC ${ING_EXAMPLE_LIBS})
