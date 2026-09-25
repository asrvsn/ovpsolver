cmake_minimum_required(VERSION 3.15...3.26)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

project(ovpsolver LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using scikit-build-core.
  To build the Python extension, run:

    pip install --no-build-isolation -ve .
")
endif()

find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP)

nanobind_add_module(
  ovpsolver_ext
  STABLE_ABI
  NB_STATIC
  src/ovpsolver_ext.cpp
  src/burgers.cpp
)

target_compile_features(ovpsolver_ext PRIVATE cxx_std_17)

if(OpenMP_CXX_FOUND)
  target_link_libraries(ovpsolver_ext PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS ovpsolver_ext LIBRARY DESTINATION ovpsolver)
