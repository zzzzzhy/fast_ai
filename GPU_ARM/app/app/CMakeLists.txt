# UnitTests
include(CheckIncludeFiles)

include_directories(SYSTEM "${THIRD_PARTY_INCLUDE_DIRS}")

if (BUILD_CAFFE_PARSER OR BUILD_TF_PARSER)
    set(app_sources
        app.cpp)

    add_executable_ex(app ${app_sources})
    target_include_directories(app PRIVATE ../src/armnnUtils)

    if (BUILD_CAFFE_PARSER)
        target_link_libraries(app armnnCaffeParser)
    endif()
    if (BUILD_TF_PARSER)
        target_link_libraries(app armnnTfParser)
    endif()
    target_link_libraries(app armnn)
    target_link_libraries(app ${CMAKE_THREAD_LIBS_INIT})
    if(OPENCL_LIBRARIES)
        target_link_libraries(app ${OPENCL_LIBRARIES})
    endif()
    target_link_libraries(app 
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY})
    addDllCopyCommands(app)
endif()
