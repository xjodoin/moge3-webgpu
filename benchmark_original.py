"""Run on NVIDIA with the pinned, unmodified original MoGe source installed.

python benchmark_original.py --inputs inputs --output original-cuda
Inputs: {fast,detail}/manifest.json and cases/{case}/input.f32.
Setup, downloads, warmup, diagnostics and serialization are excluded from timing.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import flex_gemm.config
from huggingface_hub import hf_hub_download
from moge.model.v3 import MoGeModel

SOURCE = '74fbce054ebed49800de42d0ad0e83495065719a'
CHECKPOINT = '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--strict-fp32', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    if args.strict_fp32:
        flex_gemm.config.SPCONV_ALLOW_TF32 = False
    checkpoint = hf_hub_download('Ruicheng/moge-3-vitl', 'model.pt', revision=CHECKPOINT)
    model = MoGeModel.from_pretrained(checkpoint).eval().cuda().float()
    report = dict(sourceCommit=SOURCE, checkpointRevision=CHECKPOINT,
                  checkpointSha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                  python=platform.python_version(), torch=torch.__version__,
                  packages={p: importlib.metadata.version(p) for p in ['flex-gemm', 'triton', 'numpy', 'scipy']},
                  precision='FP32; torch TF32 disabled; sparse TF32 ' + ('disabled' if args.strict_fp32 else 'allowed (original default)'),
                  sparseAllowTf32=flex_gemm.config.SPCONV_ALLOW_TF32,
                  autotuneMode=flex_gemm.config.AUTOTUNE_MODE,
                  nvidiaSmi=subprocess.check_output(['nvidia-smi'], text=True), runs=[])
    with torch.inference_mode():
        for profile in ('fast', 'detail'):
            base = args.inputs / profile
            meta = json.loads((base / 'manifest.json').read_text())
            for folder in sorted((base / 'cases').iterdir()):
                data = (folder / 'input.f32').read_bytes()
                image = torch.from_numpy(np.frombuffer(data, np.float32).copy().reshape(meta['inputShape'])).cuda()
                for steps in (1, 2, 3):
                    kwargs = dict(num_tokens=meta['numTokens'], refine_steps=steps, use_fp16=False)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    out = model.infer(image, **kwargs)
                    torch.cuda.synchronize()
                    warmup = time.perf_counter() - start
                    del out
                    times = []
                    torch.cuda.reset_peak_memory_stats()
                    for _ in range(args.repeats):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        out = model.infer(image, **kwargs)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)
                        del out
                    peak = torch.cuda.max_memory_allocated()
                    dest = args.output / profile / folder.name / str(steps)
                    dest.mkdir(parents=True, exist_ok=True)
                    # Independent full native inference, separate from timed runs.
                    out = model.infer(image, **kwargs)
                    for key in ('points', 'intrinsics', 'mask'):
                        array = out[key].cpu().numpy()
                        array.astype(np.uint8 if key == 'mask' else np.float32).tofile(dest / f'{key}.{"u8" if key == "mask" else "f32"}')
                    del out
                    row = dict(profile=profile, caseName=folder.name, refineSteps=steps,
                               inputSha256=hashlib.sha256(data).hexdigest(), warmupSeconds=warmup,
                               seconds=times, medianSeconds=float(np.median(times)), peakAllocatedBytes=peak)
                    report['runs'].append(row)
                    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(json.dumps(row), flush=True)
                # Trace a separate untimed run for locating cross-backend drift.
                debug = args.output / profile / folder.name / 'diagnostics'
                debug.mkdir(exist_ok=True)
                original = model._refine_logz
                counter = [0]

                def trace(raw, encoder):
                    step = counter[0]
                    if step == 0:
                        raw.cpu().numpy().tofile(debug / 'raw-0.f32')
                        encoder.permute(0, 2, 3, 1).contiguous().cpu().numpy().tofile(debug / 'encoder.f32')
                    z = original(raw, encoder)
                    coord = torch.cat([raw[..., :2], z.unsqueeze(-1)], dim=-1)
                    coord.cpu().numpy().tofile(debug / f'raw-{step + 1}.f32')
                    counter[0] += 1
                    return z

                model._refine_logz = trace
                try:
                    out = model.infer(image, num_tokens=meta['numTokens'], refine_steps=3, use_fp16=False)
                    del out
                finally:
                    model._refine_logz = original
                del image


if __name__ == '__main__':
    main()
