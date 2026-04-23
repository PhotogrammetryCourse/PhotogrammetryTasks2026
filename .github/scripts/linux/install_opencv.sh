#!/bin/bash
set -euo pipefail

wget https://github.com/opencv/opencv/archive/4.11.0.zip
unzip 4.11.0.zip
cd opencv-4.11.0

echo "Configuring OpenCV..."
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_LIST=features2d,highgui,flann,calib3d,imgcodecs \
  -DWITH_OPENEXR=ON -DBUILD_EXAMPLES=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/opt/opencv-4.11.0  ..

echo "Building OpenCV..."
make -j"$(nproc)"

echo "Installing OpenCV..."
sudo make install
