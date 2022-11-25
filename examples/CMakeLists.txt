cmake_minimum_required(VERSION 3.18)

if(${BUILD_BASIC})
    add_definitions("-DUNICODE" "-D_UNICODE")
endif()
####### BUILD CONFIG ############################

set(BUILD_TESTBED TRUE)
set(BUILD_APP TRUE)

#################################################

list(APPEND ING_EXAMPLE_LIBS
    ${ING_LIBS}
    libing
)

if (${BUILD_APP})
   add_subdirectory("basic_app")
endif()

if (${BUILD_TESTBED})
    add_subdirectory("basic_testbed")
endif()