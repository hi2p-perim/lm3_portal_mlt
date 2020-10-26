cmake_minimum_required(VERSION 3.10)
project(portal_mlt)

option(LM_BUILD_TESTS "" ON)
option(LM_BUILD_EXAMPLES "" ON)
add_subdirectory(lightmetrica-v3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/lightmetrica-v3/cmake")
include(LmAddPlugin)

lm_add_plugin(
    NAME portal_mlt
    SOURCES
        "renderer_mlt.cpp"
        "mut.h"
        "mut_bidir.cpp"
        "mut_lens.cpp"
        "mut_portal.cpp"
        "portal.h"
        "debug.h"
)
