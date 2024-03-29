cmake_minimum_required(VERSION 3.5)
project(TF_CUSTOM_OP)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "--std=c++11")

message("JULIA=${JULIA}")

IF(JULIA)
ELSE()
find_program(JULIA "julia")
ENDIF()

execute_process(COMMAND ${JULIA} -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)


list(GET JL_OUT 0 BINDIR)
list(GET JL_OUT 1 LIBDIR)
list(GET JL_OUT 2 TF_INC)
list(GET JL_OUT 3 TF_ABI)
list(GET JL_OUT 4 PREFIXDIR)
list(GET JL_OUT 5 CC)
list(GET JL_OUT 6 CXX)
list(GET JL_OUT 7 CMAKE)
list(GET JL_OUT 8 MAKE)
list(GET JL_OUT 9 GIT)
list(GET JL_OUT 10 PYTHON)
list(GET JL_OUT 11 TF_LIB_FILE)
list(GET JL_OUT 12 LIBCUDA)
list(GET JL_OUT 13 CUDA_INC)

message("Python path=${PYTHON}")
message("PREFIXDIR=${PREFIXDIR}")
message("TF_INC=${TF_INC}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")


if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_BUILD_TYPE Release)

if(MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
else()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()


include_directories(${TF_INC} ${PREFIXDIR} ${CUDA_INC})
link_directories(${LIBDIR})

add_library(ADTomo SHARED 
      Eikonal/Eikonal.cpp
      Eikonal3D/Eikonal3D.cpp
      Eikonal3D/EikonalThreeD.cpp
      )
add_definitions(-DNOGPU)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OPENMP Found")
    target_link_libraries(ADTomo OpenMP::OpenMP_CXX)
    target_compile_options(ADTomo PRIVATE -fopenmp)
endif()

set_property(TARGET ADTomo PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ADTomo ${TF_LIB_FILE})
