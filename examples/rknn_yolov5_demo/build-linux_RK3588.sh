set -e

TARGET_SOC="rk3588"
GCC_COMPILER=aarch64-linux-gnu

TOOL_CHAIN="/home/manu/softwares/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu"

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
# export CC=${GCC_COMPILER}-gcc
# export CXX=${GCC_COMPILER}-g++
export CC=${TOOL_CHAIN}/bin/aarch64-linux-gcc
export CXX=${TOOL_CHAIN}/bin/aarch64-linux-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC} -DCMAKE_BUILD_TYPE=Release
make -j4
make install
cd -

cp ./install /home/manu/nfs/tmp -rvf
# cp /media/manu/kingstop/workspace/rknn-toolkit2/examples/onnx/yolov5/yolov5s.rknn /home/manu/nfs/tmp/install/rknn_yolov5_demo_Linux/model/RK3588 -rvf
# cp /media/manu/kingstop/workspace/rknn-toolkit2/examples/onnx/yolov7/yolov7.rknn /home/manu/nfs/tmp/install/rknn_yolov5_demo_Linux/model/RK3588 -rvf
