cmake_minimum_required (VERSION 3.8)


set(Torch_DIR C:/projects/common/libtorch/share/cmake/Torch)


find_package(Torch PATHS C:/projects/common/libtorch/share REQUIRED)
message(STATUS "show debug")
message(STATUS "${TORCH_LIBRARIES}")
message(STATUS "fin show debug")

include_directories(.)
#include_directories("C:/projects/common/libtorch/include/")

include_directories("C:/projects/common/opencv/build/include")
#add_executable(torchPredictor "torchBinaryClass.cpp" "torchBinaryClass.h" "torchPredictorRaw.cpp" "torchPredictorRaw.h" "torchPredictor.cpp" "torchPredictor.h")


add_library(torchPredictor SHARED "torchBinaryClass.cpp" "torchBinaryClass.h" "torchPredictorRaw.cpp" "torchPredictorRaw.h")


target_link_libraries(torchPredictor "${TORCH_LIBRARIES}")
target_link_libraries(torchPredictor "C:/projects/common/opencv/build/x64/vc15/lib/opencv_world400.lib")
set_property(TARGET torchPredictor PROPERTY CXX_STANDARD 11)



