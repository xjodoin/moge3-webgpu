"""Export unchanged FP32 block parameters for the direct WGSL convolutions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def main():
    state = torch.load(ROOT / 'weights/model.pt', map_location='cpu', weights_only=True)['model']
    folder = ROOT / 'public/models/refiner-wgsl'
    folder.mkdir(exist_ok=True)
    blocks = {}
    names = ([f'down_stages.{i}.0' for i in range(5)] + ['bottleneck_stage.0']
             + [f'up_stages.{i}.0' for i in range(4)])
    for name in names:
        source = {k.removeprefix(f'refiner.{name}.'): v for k, v in state.items()
                  if k.startswith(f'refiner.{name}.')}
        channels = source['norm1.weight'].numel()
        arrays, fields, offset = [], {}, 0
        for field in ('norm1.weight', 'norm1.bias', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'):
            value = source[field]
            if field.endswith('weight') and field.startswith('conv'):
                # Original [Cout, 3, 3, 3, Cin] -> [27 * Cin, Cout].
                value = value.reshape(channels, -1).T.contiguous()
            array = value.numpy().astype('<f4', copy=False).reshape(-1)
            fields[field] = {'offset': offset, 'length': array.size}
            offset += array.size
            arrays.append(array)
        data = np.concatenate(arrays).tobytes()
        (folder / f'{name}.bin').write_bytes(data)
        blocks[name] = {'channels': channels, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'fields': fields}
    manifest = {'format': 1, 'dtype': 'float32-le', 'convolutionLayout': 'neighbor,inputChannel,outputChannel',
                'checkpointRevision': '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be', 'blocks': blocks}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(blocks)} unchanged FP32 blocks ({sum(b["bytes"] for b in blocks.values()):,} bytes)')


if __name__ == '__main__':
    main()
