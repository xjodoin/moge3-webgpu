"""Portable byte-exact FP32 transforms used by the adaptive model transport."""
import numpy as np
import zlib
import zstandard as zstd


def transform(raw, encoding):
    if encoding == 'identity':
        return raw
    if encoding == 'shuffle4':
        return np.frombuffer(raw, np.uint8).reshape(-1, 4).T.copy().tobytes()
    if encoding == 'exponent4':
        words = np.frombuffer(raw, '<u4')
        return np.stack([(words >> 23) & 255, words & 255,
                         (words >> 8) & 255, ((words >> 16) & 127) | ((words >> 24) & 128)]).astype(np.uint8).tobytes()
    if encoding == 'bitplane4':
        if len(raw) % 32:
            raise ValueError('Bit planes require groups of eight float32 values')
        values = np.frombuffer(raw, np.uint8).reshape(-1, 4)
        return b''.join(np.packbits((values[:, byte] >> bit) & 1).tobytes()
                        for byte in range(4) for bit in range(7, -1, -1))
    raise ValueError(encoding)


def inverse(raw, encoding):
    if encoding == 'identity':
        return raw
    planes = np.frombuffer(raw, np.uint8)
    if encoding == 'shuffle4':
        return planes.reshape(4, -1).T.copy().tobytes()
    if encoding == 'exponent4':
        p = planes.reshape(4, -1).astype(np.uint32)
        words = (p[0] << 23) | p[1] | (p[2] << 8) | ((p[3] & 127) << 16) | ((p[3] & 128) << 24)
        return words.astype('<u4').tobytes()
    if encoding == 'bitplane4':
        p = planes.reshape(32, -1)
        result = np.zeros((len(raw) // 4, 4), np.uint8)
        for byte in range(4):
            for bit in range(8):
                result[:, byte] |= np.unpackbits(p[byte*8+bit]) << (7-bit)
        return result.tobytes()
    raise ValueError(encoding)


def compress(raw, codec, level=12):
    if codec == 'deflate':
        return zlib.compress(raw, 6)
    if codec == 'zstd':
        return zstd.ZstdCompressor(level=level, write_checksum=True).compress(raw)
    raise ValueError(codec)


def decompress(raw, codec, size):
    return zlib.decompress(raw) if codec == 'deflate' else zstd.ZstdDecompressor().decompress(raw, max_output_size=size)
