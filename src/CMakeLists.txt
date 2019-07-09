# Make project require C++11
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -pthread")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
# elseif(COMPILER_SUPPORTS_CXX0X)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -pthread")
# endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/")

add_executable(capture-rgbt capture-rgbt.cpp)
include_directories(capture-rgbt ${ADDITIONAL_INCLUDE_DIRS})
target_link_libraries(capture-rgbt PRIVATE ${LIBUVC_FOUND} ${REALSENSE2_FOUND} ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(viewer-rgbt viewer-rgbt.cpp)
include_directories(viewer-rgbt ${ADDITIONAL_INCLUDE_DIRS})
target_link_libraries(viewer-rgbt PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(calib-corners calib-corners.cpp)
include_directories(calib-corners ${ADDITIONAL_INCLUDE_DIRS})
target_link_libraries(calib-corners PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(calib-intrinsics calib-intrinsics.cpp)
include_directories(calib-intrinsics ${ADDITIONAL_INCLUDE_DIRS})
target_link_libraries(calib-intrinsics PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(calib-extrinsics calib-extrinsics.cpp)
include_directories(calib-extrinsics ${ADDITIONAL_INCLUDE_DIRS})
target_link_libraries(calib-extrinsics PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_subdirectory(tests)
