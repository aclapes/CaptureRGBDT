#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(CaptureRGBDT)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Make project require C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

# Simple non robust way to find the librealsense library
if(WIN32)
    if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
        set(LIBRARY_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x64")          # TODO: Update this variable to correct path - folder where realsense2.lib is found
        set(DLL_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x64")              # TODO: Update this variable to correct path - folder where realsense2.dll is found
    else()
        set(LIBRARY_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x86")          # TODO: Update this variable to correct path - folder where realsense2.lib is found
        set(DLL_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x86")              # TODO: Update this variable to correct path - folder where realsense2.dll is found
    endif()
    set(PROJECT_BINARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/\$\(Configuration\)")                # TODO: Update this variable to correct path - folder to which your project will compile
    set(ADDITIONAL_INCLUDE_DIRS "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\include")   # TODO: Update this variable to correct path - folder where librealsense2 folder is found
else()
    set(ADDITIONAL_INCLUDE_DIRS "/usr/local/include")
    set(ADDITIONAL_LIBRARY_DIRS "/usr/local/lib")
endif()


find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()
# getting-started
find_library(LIBUVC_FOUND uvc HINTS ${LIBRARY_DIR} REQUIRED)
#find_library(OPENCV_FOUND opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_features2d opencv_calib3d opencv_objdetect opencv_contrib opencv_legacy opencv_flann opencv_imgcodecs HINTS ${LIBRARY_DIR} REQUIRED )
#find_library(BOOST_FOUND boost_system boost_filesystem HINTS ${LIBRARY_DIR} REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED)

add_executable(rs-getting-started rs-getting-started.cpp pt_pipeline.hpp safe_queue.hpp)
include_directories(rs-getting-started ${ADDITIONAL_INCLUDE_DIRS})
#link_directories(rs-getting-started ${ADDITIONAL_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})
target_link_libraries(rs-getting-started PRIVATE ${LIBUVC_FOUND} ${REALSENSE2_FOUND} ${Boost_LIBRARIES} ${OpenCV_LIBS})
set_target_properties (rs-getting-started PROPERTIES FOLDER Examples)

# Post Build script to copy realsense2.dll
if(WIN32)
message(STATUS "Adding Post build script to copy realsense2.dll to project's binary folder")
message(STATUS "Will try to copy from ${DLL_DIR} to ${PROJECT_BINARY_OUTPUT_PATH}")
add_custom_command(TARGET rs-getting-started POST_BUILD        # Adds a post-build event to rs-getting-started
    COMMAND ${CMAKE_COMMAND} -E copy_if_different              # which executes "cmake - E copy_if_different..."
        "${DLL_DIR}/realsense2.dll"                            # <--this is in-file
        ${PROJECT_BINARY_OUTPUT_PATH})                 # <--this is out-file path
endif()

install(
    TARGETS

    rs-getting-started

    RUNTIME DESTINATION
    ${CMAKE_INSTALL_PREFIX}/bin
)
