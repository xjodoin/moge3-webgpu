# Contributing

This is an unofficial browser port of Microsoft MoGe-3. Keep learned weights and numerical semantics faithful to the pinned upstream implementation. Include all affected licenses when adapting code.

Run `npm ci`, `npm test`, and `npm run build`. Real GPU validation requires exported models and a WebGPU browser; CI unit tests are not evidence of GPU inference correctness. See the README for native MLX reference generation and browser tests.

For model changes, record the hardware/browser, input dimensions, token budget, refinement count, inference and load time, explicit GPU buffer peak, and numerical comparisons against native inference. Preserve mask disagreement counts and maximum errors, not only averages. Do not silently lower comparison thresholds to make a configuration pass.

Useful contributions include custom sparse WGSL kernels, GPU-side pooling, worker isolation, model download/cache management, and validation on other GPU/browser combinations. Start with an issue describing substantial architectural changes.

An upstream MoGe pull request should be a focused, independently reproducible contribution (for example, native-equivalent ONNX export helpers). This repository can ship the standalone viewer without requiring upstream maintainers to adopt a separate front-end application.
