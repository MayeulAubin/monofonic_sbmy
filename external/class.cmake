# initialize the class submodule if necessary
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/class/Makefile")
  message(STATUS "class submodule is initialized.")
else()
  message(STATUS "class submodule is NOT initialized: executing git command")
  execute_process(COMMAND git submodule update --init -- external/class
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(CLASS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/class/include)

# list of object files generated by class
set(CLASS_OBJECT_FILES
  ${CMAKE_CURRENT_LIST_DIR}/class/build/arrays.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/background.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/common.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/dei_rkck.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/evolver_ndf15.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/evolver_rkck.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/growTable.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/helium.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/history.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/hydrogen.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/hyperspherical.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/hyrectools.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/input.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/lensing.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/nonlinear.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/output.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/parser.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/perturbations.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/primordial.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/quadrature.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/sparse.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/spectra.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/thermodynamics.o
  ${CMAKE_CURRENT_LIST_DIR}/class/build/transfer.o
)

# python3
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# command to build class using its own makefile
add_custom_command(OUTPUT ${CLASS_OBJECT_FILES}
  COMMAND PYTHON=${Python3_EXECUTABLE} CC=${CMAKE_C_COMPILER} make
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/class
)

# target for class objects
add_custom_target(class_objects DEPENDS ${CLASS_OBJECT_FILES})

# library for class cpp wrappers
add_library(class_cpp 
  ${CMAKE_CURRENT_LIST_DIR}/class/cpp/Engine.cc
  ${CMAKE_CURRENT_LIST_DIR}/class/cpp/ClassEngine.cc)
target_include_directories(class_cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/class/include)

# macro to setup include dir and link libraries for target using class
macro(target_setup_class target_name)
  target_include_directories(${target_name}
    PRIVATE ${CLASS_INCLUDE_DIR})
  target_link_libraries(${target_name} ${CLASS_OBJECT_FILES})
  target_link_libraries(${target_name} class_cpp)
  add_dependencies(${target_name} class_objects)
endmacro(target_setup_class)


# test executable
add_executable(testTk
  ${CMAKE_CURRENT_LIST_DIR}/class/cpp/testTk.cc)
target_setup_class(testTk)