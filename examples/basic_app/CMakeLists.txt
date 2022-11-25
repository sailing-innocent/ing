cmake_minimum_required(VERSION 3.18)
set(TARGET_NAME basic_app)

set(TEST FALSE)

set(GL TRUE)
set(VK FALSE)
set(DRX FALSE)

set(PURE TRUE)

set(VK_TEST FALSE)
set(PURE_GL TRUE)
set(IMGUI_DRX FALSE)
set(BUILD_BASIC FALSE)
set(BUILD_CV FALSE)

if(${TEST})
endif()

if(${GL})
    set(TARGET_NAME basic_app_gl)
    if(${PURE})
        list(APPEND ING_EXAMPLE_SOURCE
            # "dynamic/wave.cc"
            "static/gl_app.cc"
        ) 
    endif()
    add_executable(${TARGET_NAME} ${ING_EXAMPLE_SOURCE})
    target_link_directories(${TARGET_NAME} PUBLIC ${ING_LIBS_DIR})
    target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} ${ING_EXAMPLE_LIBS})
endif()

if(${VK})
    set(TARGET_NAME basic_app_vk)
    if(${PURE})
        list(APPEND ING_EXAMPLE_SOURCE
            "static/vk_canvas_main.cc"
        )
    endif()
    add_executable(${TARGET_NAME} ${ING_EXAMPLE_SOURCE})
    target_link_directories(${TARGET_NAME} PUBLIC ${ING_LIBS_DIR})
    target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} ${ING_EXAMPLE_LIBS})
endif()

if(${DRX})
    set(TARGET_NAME basic_app_drx)
    if(${PURE})
    endif()
    add_executable(${TARGET_NAME} ${ING_EXAMPLE_SOURCE})
    target_link_directories(${TARGET_NAME} PUBLIC ${ING_LIBS_DIR})
    target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} ${ING_EXAMPLE_LIBS})
endif()

if(${IMGUI_DRX})
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
    add_executable(${TARGET_NAME} ${ING_SOURCE})
    target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME} PUBLIC ${ING_LIB})
elseif(${BUILD_CV})
    add_executable(${TARGET_NAME} "example/image/opencv_impl.cc")
elseif(${BUILD_BASIC})
    message("IS BUILD${TARGET_NAME} BASIC")
    list(APPEND WIN_APP_SOURCE
        # "ing/app/win_app.cc"
        "ing/app/win_pure.cc"
        "examples/plain/pure_win.cc"
    )
    add_executable(${TARGET_NAME} WIN32 ${WIN_APP_SOURCE})
    target_include_directories(${TARGET_NAME} PUBLIC ${ING_INCLUDE_DIR})
else()
endif()
