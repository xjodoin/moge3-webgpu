"""Custom 24-bit weight transport; computation remains IEEE FP32.

Round the magnitude's low eight bits half-up. The original low byte alone is
enough to undo that rounding, including its carry into the stored high bits.
Never change NaNs, infinities, or finite values that would round to infinity.
"""
import numpy as np


def split24(raw):
    if len(raw) % 4:
        raise ValueError('FP32 payload must contain whole words')
    words = np.frombuffer(raw, dtype='<u4')
    if np.any((words & 0x7f800000) == 0x7f800000):
        raise ValueError('Nonfinite weight')
    rounded = ((words.astype(np.uint64) + 128) & 0xffffff00).astype('<u4')
    if np.any((rounded & 0x7f800000) == 0x7f800000):
        raise ValueError('FP24 weight overflow')
    octets = rounded.view(np.uint8).reshape(-1, 4)
    # Separate exponent, middle mantissa byte, and sign/upper mantissa.
    packed = np.stack([((octets[:, 2] >> 7) | ((octets[:, 3] & 127) << 1)),
                       octets[:, 1], (octets[:, 3] & 128) | (octets[:, 2] & 127)]).tobytes()
    low = (words & 255).astype(np.uint8).tobytes()
    return packed, low, rounded.tobytes()


def expand24(packed):
    if len(packed) % 3:
        raise ValueError('Invalid FP24 plane length')
    planes = np.frombuffer(packed, np.uint8).reshape(3, -1)
    raw = np.zeros((planes.shape[1], 4), np.uint8)
    raw[:, 1] = planes[1]
    raw[:, 2] = (planes[2] & 127) | ((planes[0] & 1) << 7)
    raw[:, 3] = (planes[0] >> 1) | (planes[2] & 128)
    return raw.tobytes()


def restore32(rounded, low):
    if len(rounded) != len(low) * 4:
        raise ValueError('Invalid correction length')
    words = np.frombuffer(rounded, '<u4').copy()
    correction = np.frombuffer(low, np.uint8).astype(np.uint32)
    words -= (correction >> 7) << 8
    words |= correction
    return words.tobytes()
