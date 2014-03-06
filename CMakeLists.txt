cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE ${CMAKE_CXX_FLAGS_DEBUG})
project( Gamar )

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

find_package( OpenCV REQUIRED )

add_executable( render_camera render_camera.cpp )
target_link_libraries( render_camera -lGL -lglut -lglfw -lGLEW -lGLU ${OpenCV_LIBS})
