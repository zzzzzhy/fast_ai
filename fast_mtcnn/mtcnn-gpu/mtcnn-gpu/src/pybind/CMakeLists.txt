message(STATUS "============pybind module===========")

include_directories( ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/3rdparty/include
                   )

link_directories(${PROJECT_SOURCE_DIR}/lib ${PROJECT_SOURCE_DIR}/3rdparty/lib)

pybind11_add_module(face_detection main.cpp)
target_link_libraries(face_detection PRIVATE mtcnn ${OpenCV_LIBS})
