#!/bin/bash
# 为海光 DCU 编译 bitsandbytes 的 HIP 后端。
# 官方 ROCm 预编译包不含 gfx928（K100_AI）等 DCU 架构，8bit 优化器会报 invalid device function。
#
# 用法: [BNB_ARCH=gfx928] build_bnb_dcu.sh <bitsandbytes 0.50.2 源码目录> <已安装 bitsandbytes 的 Python 路径>
#   例: build_bnb_dcu.sh ./bitsandbytes-0.50.2 /root/moss/ai-toolkit/venv/bin/python
#   BNB_ARCH 默认用 rocminfo 自动识别；需要 DTK（/opt/dtk 或 ROCM_PATH）、cmake、make
set -e
SRC=${1:?需要 bitsandbytes 源码目录}; PY=${2:?需要目标 Python 路径}
source ${ROCM_PATH:-/opt/dtk}/env.sh >/dev/null 2>&1
ARCH=${BNB_ARCH:-$(rocminfo 2>/dev/null | grep -o -m1 "gfx[0-9a-f]*")}; ARCH=${ARCH:-gfx928}
echo "[bnb] 目标架构: $ARCH"
cd $SRC
rm -rf build && mkdir build && cd build
HIPCLANG=$(ls -d $ROCM_PATH/llvm/bin/clang++ 2>/dev/null || ls -d $ROCM_PATH/lib/llvm/bin/clang++)
# --gpu-max-threads-per-block：DTK 严格检查 launch bounds，部分内核以 512 线程启动
cmake .. -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=$ARCH -DROCM_VERSION=63 \
  -DCMAKE_HIP_COMPILER=$HIPCLANG -DCMAKE_PREFIX_PATH=$ROCM_PATH -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="--gpu-max-threads-per-block=1024"
make -j${MAKE_JOBS:-$(nproc)}
cd ..
SITE=$(cd / && "$PY" -c "import bitsandbytes, os; print(os.path.dirname(bitsandbytes.__file__))")
cp -v bitsandbytes/libbitsandbytes_rocm63.so "$SITE/"
echo "[bnb] 完成：已替换 $SITE/libbitsandbytes_rocm63.so"
