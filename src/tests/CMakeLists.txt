set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/")

# add_executable(resize resize.cpp)
# include_directories(resize ${ADDITIONAL_INCLUDE_DIRS})
# target_link_libraries(resize PRIVATE ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(struct_test struct_test.cpp)
include_directories(struct_test ${ADDITIONAL_INCLUDE_DIRS})
target_link_libraries(struct_test PRIVATE ${OpenCV_LIBS})

add_executable(project_info project_info.cpp)
include_directories(project_info ${ADDITIONAL_INCLUDE_DIRS})
target_link_libraries(project_info PRIVATE ${OpenCV_LIBS})