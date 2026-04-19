cmake_minimum_required(VERSION 3.20)

project(Photogrammetry)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(libs)

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else ()
    message(FATAL_ERROR "OpenMP not found! You need OpenMP for speedup on multicore CPUs!")
endif ()

find_package(OpenCV 4.11.0 REQUIRED)
# в целом "как выполнить сборку OpenCV и проекта" можно узнать посмотрев на GitHub CI,
# который выполняет сборку и исполнение на linux/windows/macos
# описание шагов - в .github/workflows/cmake.yml
#
#      Linux: (рекомендуется Ubuntu 22.04/24.04, на крайний случай - в виртуалке)
# sudo .github/scripts/install_opencv_linux.sh
# укажите для CMake путь к OpenCV:
# в CLion это делается через File->Settings->Build, Execution, Deployment->CMake->CMake options: -DOpenCV_DIR=/opt/opencv-4.11.0/lib/cmake/opencv4
# там же сразу стоит добавить в дополнению к Debug сборке - сборку с включенными оптимизациями и отладочными символами - RelWithDebInfo
#
#     Windows: (но рекомендуется использовать Linux)
# выполните в powershell .github/scripts/windows/install_opencv.ps1
# альтернативно вы можете следовать инструкции - https://gist.github.com/PolarNick239/18036d942ae4b9aad9f1da756ce6c845#file-buildocv-sh-L10-L33
#

find_package(Eigen3 REQUIRED)
#      Linux:
# sudo apt install libeigen3-dev
#      Windows (чтобы удобно копировать в Git Bash консоль: выделите, нажмите Ctrl+C, в консоли нажмите Shift+Insert):
# 1) Скачать https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip
# 2) Распаковать и открыть Git Bash в папке распаковки (правый клик в проводнике -> Git Bash here)
# 3) myDirPath=$(pwd)
# 4) mkdir build_dir
# 5) cd build_dir
# 6) cmake -G "Visual Studio 16 2019" -DCMAKE_INSTALL_PREFIX="$myDirPath/install_dir" ..
# 7) cmake --build .  --target install
# 8) запомните папку куда была проведена установка Eigen3 - она имеет вид .../eigen-3.3.9/install_dir
# 9) Добавьте в File->Settings->CMake options: -DEigen3_DIR=<ПОПРАВЬТЕ ЭТОТ ПУТЬ>/eigen-3.3.9/install_dir/share/eigen3/cmake

find_package(Ceres 2.2.0 REQUIRED)
#      Linux: выполните скрипт .github/scripts/linux/install_ceres_solver.sh
#      Mac: выполните скрипт .github/scripts/macos/install_ceres_solver.sh
#      Windows: выполните скрипт .github/scripts/windows/install_ceres_solver.ps1 или следуйте инструкции http://ceres-solver.org/installation.html#windows

find_package(CGAL REQUIRED COMPONENTS Core)
# Mac: https://www.cgal.org/download/macosx.html - brew install cgal
# Linux: https://www.cgal.org/download/linux.html - sudo apt install -y libcgal-dev
#
# Windows: https://doc.cgal.org/latest/Manual/windows.html#sec-installing-with-vcpkg
# 1) Win+R -> cmd -> Enter
# 2) mkdir&cd C:\...\some\dir\for\vcpkg (потребуется порядка 5 гигабайт свободного места)
# 3) git clone https://github.com/microsoft/vcpkg
# 4) cd vcpkg
# 5) .\bootstrap-vcpkg.bat
# 6) vcpkg.exe install yasm-tool:x86-windows
# Следующий шаг займет много времени (минут 30), так что рекомендуется запустить его заранее:
# 7) vcpkg.exe install cgal:x64-windows
# 8) Указать для CMake пути (в CLion это делается через File->Settings->Build, Execution, Deployment->CMake->CMake options):
# -DGMP_LIBRARIES=C:\...\vcpkg\installed\x64-windows\lib\gmp.lib
# -DGMP_INCLUDE_DIR=C:\...\vcpkg\installed\x64-windows\include
# -DMPFR_LIBRARIES=C:\...\vcpkg\installed\x64-windows\lib\mpfr.lib
# -DMPFR_INCLUDE_DIR=C:\...\vcpkg\installed\x64-windows\include
# -DBoost_INCLUDE_DIR=C:\...\vcpkg\installed\x64-windows\include
# -DCGAL_DIR=C:\...\vcpkg\x64-windows\share\cgal
# 9) Скопируйте в папку cmake-build-relwithdebinfo и в папку cmake-build-debug эти два файла:
# C:\...\vcpkg\installed\x64-windows\bin\gmp.dll
# C:\...\vcpkg\installed\x64-windows\bin\mpfr-6.dll

# CGAL_USE_FILE doesn't work with RelWithDebInfo
set(CMAKE_BUILD_TYPE_BEFORE_CGAL ${CMAKE_BUILD_TYPE})
if ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "Release")
endif ()
# Newer CGAL packages may not define CGAL_USE_FILE at all; the imported
# targets from find_package(CGAL ...) are sufficient in that case.
if (DEFINED CGAL_USE_FILE AND CGAL_USE_FILE)
    include(${CGAL_USE_FILE})
endif ()
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BEFORE_CGAL})

set(CGAL_DEPENDENCIES_LIBRARIES ${MPFR_LIBRARIES} ${GMP_LIBRARIES})

set(SOURCES
        src/phg/core/calibration.cpp
        src/phg/core/calibration.h
        src/phg/core/camera.cpp
        src/phg/core/camera.h
        src/phg/sift/sift.cpp
        src/phg/sift/sift.h
        src/phg/matching/descriptor_matcher.cpp
        src/phg/matching/descriptor_matcher.h
        src/phg/matching/bruteforce_matcher.cpp
        src/phg/matching/bruteforce_matcher.h
        src/phg/matching/bruteforce_matcher_gpu.cpp
        src/phg/matching/bruteforce_matcher_gpu.h
        src/phg/matching/gms_matcher.cpp
        src/phg/matching/gms_matcher.h
        src/phg/matching/gms_matcher_impl.h
        src/phg/matching/flann_matcher.cpp
        src/phg/matching/flann_matcher.h
        src/phg/matching/flann_factory.h
        src/phg/mvs/depth_maps/pm_depth_maps.cpp
        src/phg/mvs/depth_maps/pm_depth_maps.h
        src/phg/mvs/depth_maps/pm_fast_random.cpp
        src/phg/mvs/depth_maps/pm_fast_random.h
        src/phg/mvs/depth_maps/pm_geometry.cpp
        src/phg/mvs/depth_maps/pm_geometry.h
        src/phg/mvs/model_min_cut/bk_solver/graph.cpp
        src/phg/mvs/model_min_cut/bk_solver/maxflow.cpp
        src/phg/mvs/model_min_cut/min_cut_cgal_structs.cpp
        src/phg/mvs/model_min_cut/min_cut_cgal_structs.h
        src/phg/mvs/model_min_cut/min_cut_model_builder.cpp
        src/phg/mvs/model_min_cut/min_cut_model_builder.h
        src/phg/sfm/defines.h
        src/phg/sfm/ematrix.cpp
        src/phg/sfm/ematrix.h
        src/phg/sfm/fmatrix.cpp
        src/phg/sfm/fmatrix.h
        src/phg/sfm/homography.cpp
        src/phg/sfm/homography.h
        src/phg/sfm/panorama_stitcher.cpp
        src/phg/sfm/panorama_stitcher.h
        src/phg/sfm/resection.cpp
        src/phg/sfm/resection.h
        src/phg/sfm/sfm_utils.cpp
        src/phg/sfm/sfm_utils.h
        src/phg/sfm/triangulation.cpp
        src/phg/sfm/triangulation.h
        src/phg/utils/cameras_bundler_export.cpp
        src/phg/utils/cameras_bundler_export.h
        src/phg/utils/cameras_bundler_import.cpp
        src/phg/utils/cameras_bundler_import.h
        src/phg/utils/mesh_export.cpp
        src/phg/utils/mesh_export.h
        src/phg/utils/point_cloud_export.cpp
        src/phg/utils/point_cloud_export.h
        )

# convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
# Она считывает все байты из файла bruteforce_matcher.cl (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле bruteforce_matcher_cl.h bruteforce_matcher_kernel
convertIntoHeader(src/phg/matching/cl/bruteforce_matcher.cl src/phg/matching/cl/bruteforce_matcher_cl.h bruteforce_matcher_kernel)
# Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл bruteforce_matcher_cl.h был перечислен среди исходников для компиляции при вызове add_executable
set(SOURCES ${SOURCES} src/phg/matching/cl/bruteforce_matcher_cl.h)

set(LIBRARIES
        libutils
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        Ceres::ceres
        ${CGAL_DEPENDENCIES_LIBRARIES}
        )

add_library(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

set(TEST_SOURCES
        tests/utils/test_utils.cpp
        tests/utils/test_utils.h
        )

set(TEST_LIBRARIES
        gtest
        gtest_main
        )

add_executable(test_sift            tests/test_sift.cpp             ${TEST_SOURCES})
add_executable(test_matching        tests/test_matching.cpp         ${TEST_SOURCES})
add_executable(test_sfm             tests/test_sfm.cpp              ${TEST_SOURCES})
add_executable(test_ceres_solver    tests/test_ceres_solver.cpp     ${TEST_SOURCES})
add_executable(test_sfm_ba          tests/test_sfm_ba.cpp           ${TEST_SOURCES})
add_executable(test_depth_maps_pm   tests/test_depth_maps_pm.cpp    ${TEST_SOURCES})
add_executable(test_mesh_min_cut    tests/test_mesh_min_cut.cpp     ${TEST_SOURCES})

target_link_libraries(test_sift          ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_matching      ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_sfm           ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_ceres_solver  ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_sfm_ba        ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_depth_maps_pm ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
target_link_libraries(test_mesh_min_cut  ${PROJECT_NAME} ${LIBRARIES} ${TEST_LIBRARIES})
