cmake_minimum_required(VERSION 3.18)

if(${BUILD_BASIC})
    add_definitions("-DUNICODE" "-D_UNICODE")
endif()
####### BUILD CONFIG ############################

set(TEST FALSE)
set(VK_TEST FALSE)
set(PURE_GL TRUE)
set(IMGUI_DRX FALSE)
set(BUILD_BASIC FALSE)
set(BUILD_CV FALSE)

#################################################

if (${BUILD_EXAMPLES})
    list(APPEND ING_EXAMPLE_LIBS
        ${ING_LIBS}
        libing
    )
    if(${TEST})
        add_executable(ing "geometry/mygeotest.cc")
        target_include_directories(ing PUBLIC ${ING_INCLUDE_DIR})
    elseif(${VK_TEST})
        list(APPEND ING_EXAMPLE_SOURCE
            "static/vk_canvas_main.cc"
        )
    elseif(${PURE_GL})
        list(APPEND ING_EXAMPLE_SOURCE
            # "dynamic/wave.cc"
            "static/gl_app.cc"
        )    
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
    add_executable(ing ${ING_EXAMPLE_SOURCE})
    target_link_directories(ing PUBLIC ${ING_LIBS_DIR})
    target_include_directories(ing PUBLIC ${ING_INCLUDE_DIR})
    target_link_libraries(ing ${ING_EXAMPLE_LIBS})
endif()