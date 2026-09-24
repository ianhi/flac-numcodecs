"""Test data shared by the test modules."""
import numpy as np

# seeded so that a failure can be reproduced
rng = np.random.default_rng(0)


def make_noisy_sin_signals(shape=(30000,), sin_f=100, sin_amp=50, noise_amp=5,
                           sample_rate=30000, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) <= 3
    if len(shape) == 1:
        y = np.sin(2 * np.pi * sin_f * np.arange(shape[0]) / sample_rate) * sin_amp
        y = y + rng.standard_normal(shape[0]) * noise_amp
        y = y.astype(dtype)
    elif len(shape) == 2:
        nsamples, nchannels = shape
        y = np.zeros(shape, dtype=dtype)
        for ch in range(nchannels):
            y[:, ch] = make_noisy_sin_signals((nsamples,), sin_f, sin_amp, noise_amp,
                                              sample_rate, dtype)
    else:
        nsamples, nchannels1, nchannels2 = shape
        y = np.zeros(shape, dtype=dtype)
        for ch1 in range(nchannels1):
            for ch2 in range(nchannels2):
                y[:, ch1, ch2] = make_noisy_sin_signals((nsamples,), sin_f, sin_amp, noise_amp,
                                                        sample_rate, dtype)
    return y


def split_header(enc):
    """Split a complete FLAC stream into its header (magic + metadata blocks) and its frames."""
    assert enc[:4] == b"fLaC"
    i = 4
    while True:
        is_last = enc[i] & 0x80
        length = int.from_bytes(enc[i + 1:i + 4], "big")
        i += 4 + length
        if is_last:
            break
    return enc[:i], enc[i:]


def _crc(data, poly, bits):
    crc, top, mask = 0, 1 << (bits - 1), (1 << bits) - 1
    for byte in data:
        crc ^= byte << (bits - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & mask if crc & top else (crc << 1) & mask
    return crc


def bit_depth_from_streaminfo(frame):
    """Rewrite the header of a single FLAC frame so that it takes its bit depth from
    STREAMINFO (bit-depth code 000), recomputing the header's CRC-8 and the frame's CRC-16."""
    frame = bytearray(frame)
    # the frame header: sync code and codes (4 bytes), the UTF-8-coded frame number, the
    # block size and sample rate when their codes say they follow, then the CRC-8
    number_length = 1 if frame[4] < 0x80 else 8 - (~frame[4] & 0xFF).bit_length()
    blocksize_code, sample_rate_code = frame[2] >> 4, frame[2] & 0x0F
    crc8_at = (4 + number_length + {6: 1, 7: 2}.get(blocksize_code, 0)
               + {12: 1, 13: 2, 14: 2}.get(sample_rate_code, 0))
    assert frame[crc8_at] == _crc(frame[:crc8_at], 0x07, 8)
    frame[3] &= ~0b1110
    frame[crc8_at] = _crc(frame[:crc8_at], 0x07, 8)
    frame[-2:] = _crc(frame[:-2], 0x8005, 16).to_bytes(2, "big")
    return bytes(frame)
