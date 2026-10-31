#!/usr/bin/env bash
# On the disposable benchmark host, with PyTorch 2.7.1 / CUDA 12.8 devel.
set -euo pipefail
cd /workspace
export MAX_JOBS=4
# Same policy as the official demo; avoid adaptive tuning during timed repeats.
export FLEX_GEMM_AUTOTUNE_MODE=never
if [ ! -d MoGe/.git ]; then git clone https://github.com/microsoft/MoGe.git; fi
git -C MoGe checkout 74fbce054ebed49800de42d0ad0e83495065719a
python -m pip install 'setuptools==80.9.0' wheel
python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -e ./MoGe
git -C MoGe diff --exit-code
python -m pip freeze > original-environment.txt
python benchmark_original.py --inputs inputs --output original-cuda-default
python benchmark_original.py --inputs inputs --output original-cuda --strict-fp32
