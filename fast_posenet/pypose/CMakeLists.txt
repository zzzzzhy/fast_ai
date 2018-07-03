# UnitTests
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
include(CheckIncludeFiles)

include_directories(SYSTEM "${THIRD_PARTY_INCLUDE_DIRS}")

if (BUILD_CAFFE_PARSER OR BUILD_TF_PARSER)
    set(app_sources
        pypose.cpp)

    #add_executable_ex(app ${app_sources})
    pybind11_add_module(pose ${app_sources})
    target_include_directories(pose PRIVATE ../src/armnnUtils)

    if (BUILD_CAFFE_PARSER)
	    target_link_libraries(pose PRIVATE armnnCaffeParser)
    endif()
    if (BUILD_TF_PARSER)
	    target_link_libraries(pose PRIVATE armnnTfParser)
    endif()
    target_link_libraries(pose PRIVATE armnn)
    target_link_libraries(pose PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    if(OPENCL_LIBRARIES)
	    target_link_libraries(pose PRIVATE ${OPENCL_LIBRARIES})
    endif()
    target_link_libraries(pose PRIVATE
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY})
    #addDllCopyCommands(app)
endif()
