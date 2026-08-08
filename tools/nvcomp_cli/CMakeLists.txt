# nvcomp_cli — see nvcomp_cli.cu and docs/adapters/nvcomp.md.
#
# nvCOMP is distributed as a prebuilt SDK (headers + libnvcomp.so), not a source
# tree, so this looks for an install root rather than add_subdirectory()-ing it:
#
#   cmake -S tools/nvcomp_cli -B tools/nvcomp_cli/build \
#         -DNVCOMP_ROOT=/path/to/nvcomp -DCMAKE_CUDA_ARCHITECTURES=90
#   cmake --build tools/nvcomp_cli/build -j
#
# NVCOMP_ROOT may also come from the environment (the env-*.sh scripts set it).
cmake_minimum_required(VERSION 3.20)
project(nvcomp_cli LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)   # H100; override for other GPUs
endif()

if(NOT NVCOMP_ROOT AND DEFINED ENV{NVCOMP_ROOT})
  set(NVCOMP_ROOT $ENV{NVCOMP_ROOT})
endif()
if(NOT NVCOMP_ROOT)
  message(FATAL_ERROR
    "NVCOMP_ROOT is not set. Point it at an nvCOMP SDK install containing "
    "include/nvcomp/zstd.hpp and lib/libnvcomp.so — e.g. "
    "-DNVCOMP_ROOT=$HOME/compressors/nvcomp")
endif()

find_path(NVCOMP_INCLUDE_DIR nvcomp/zstd.hpp
          HINTS "${NVCOMP_ROOT}/include" NO_DEFAULT_PATH)
find_library(NVCOMP_LIBRARY nvcomp
             HINTS "${NVCOMP_ROOT}/lib" "${NVCOMP_ROOT}/lib64" NO_DEFAULT_PATH)

if(NOT NVCOMP_INCLUDE_DIR OR NOT NVCOMP_LIBRARY)
  message(FATAL_ERROR
    "nvCOMP not found under NVCOMP_ROOT=${NVCOMP_ROOT} "
    "(need include/nvcomp/zstd.hpp and lib/libnvcomp.so). "
    "Note that Zstd requires nvCOMP >= 3.0; this tool is developed against 5.2.")
endif()

add_executable(nvcomp_cli nvcomp_cli.cu)
target_include_directories(nvcomp_cli PRIVATE "${NVCOMP_INCLUDE_DIR}")
target_link_libraries(nvcomp_cli PRIVATE "${NVCOMP_LIBRARY}")

# Bake the SDK's lib dir into the binary. nvCOMP is not on the system loader path
# and the harness invokes this as a bare subprocess with the caller's environment;
# without an RPATH every run would depend on LD_LIBRARY_PATH being set correctly
# in whatever shell (or SLURM job) launched benchkit.
set_target_properties(nvcomp_cli PROPERTIES
  BUILD_RPATH   "${NVCOMP_ROOT}/lib"
  INSTALL_RPATH "${NVCOMP_ROOT}/lib"
  CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "nvcomp_cli: nvCOMP headers ${NVCOMP_INCLUDE_DIR}")
message(STATUS "nvcomp_cli: nvCOMP library ${NVCOMP_LIBRARY}")
message(STATUS "nvcomp_cli: CUDA arch      ${CMAKE_CUDA_ARCHITECTURES}")
