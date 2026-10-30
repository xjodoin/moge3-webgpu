"""Preserve native DINO resize/position semantics in fixed-shape ONNX exports.

Upstream onnx_compatible_mode disables antialiasing and the historical positional
interpolation offset. We precompute positions and separable resize weights with
PyTorch's native operators, then export ordinary MatMul operations instead.
No learned weights are modified.
"""
import types
import torch
from torch.nn import functional as F


def prepare_native_encoder(model, height, width, token_rows, token_cols):
    encoder = model.encoder
    backbone = encoder.backbone
    backbone.onnx_compatible_mode = False
    with torch.no_grad():
        pos = backbone.interpolate_pos_encoding(
            torch.zeros(1, token_rows * token_cols + 1, backbone.embed_dim),
            token_rows * 14, token_cols * 14,
        ).detach()
        # Resizing an identity matrix yields the one-dimensional filter weights.
        wy = F.interpolate(torch.eye(height)[None, None], (token_rows * 14, height),
                           mode='bilinear', align_corners=False, antialias=True)[0, 0]
        wx = F.interpolate(torch.eye(width)[None, None], (token_cols * 14, width),
                           mode='bilinear', align_corners=False, antialias=True)[0, 0]
    backbone.register_buffer('browser_pos_embed', pos)
    backbone.interpolate_pos_encoding = types.MethodType(
        lambda self, x, h, w: self.browser_pos_embed, backbone)
    encoder.register_buffer('browser_resize_y', wy)
    encoder.register_buffer('browser_resize_x', wx.T.contiguous())

    def forward(self, image, token_rows, token_cols, return_class_token=False):
        image_14 = self.browser_resize_y @ image @ self.browser_resize_x
        image_14 = (image_14 - self.image_mean) / self.image_std
        features = self.backbone.get_intermediate_layers(
            image_14, n=self.intermediate_layers, return_class_token=True)
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
            for proj, (feat, _) in zip(self.output_projections, features)
        ], dim=1).sum(dim=1)
        return (x, features[-1][1]) if return_class_token else x

    encoder.forward = types.MethodType(forward, encoder)
