"""
Zarr v3 codec for FLAC, registered under the name `flac` through the `zarr.codecs` entry point.

FLAC encodes samples, so the codec is an array-to-bytes codec (the array's serializer), and
takes the place of the `bytes` codec. It encodes and decodes with the same functions in
`flac_numcodecs.flac` as the Zarr v2 codec, `flac_numcodecs.Flac`. The module requires
`zarr>=3.1`.
"""
import asyncio
import re
from dataclasses import asdict, dataclass
from math import prod

import numpy as np
import zarr

if tuple(map(int, re.match(r"(\d+)\.(\d+)", zarr.__version__).groups())) < (3, 1):
    raise ImportError(f"The Zarr v3 FLAC codec requires zarr>=3.1, but zarr {zarr.__version__} "
                      "is installed")

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.array_spec import ArraySpec
from zarr.core.common import parse_named_configuration

from .flac import FlacNumpyEncoder, encode, _decode_frames, max_blocksize


def _check_int16(dtype):
    if not (dtype.kind == "i" and dtype.itemsize == 2):
        raise ValueError(f"Data type {dtype} not supported. Only int16 is supported")


@dataclass(frozen=True)
class Flac(ArrayBytesCodec):
    """Zarr v3 codec for FLAC (Free Lossless Audio Codec).

    The first dimension of a chunk is time, and the remaining dimensions, flattened, are
    channels. A chunk with up to 8 channels, the most a FLAC stream can hold, is encoded with
    one FLAC channel per array channel. A chunk with more is encoded as a single FLAC channel
    holding each channel in turn, so that consecutive samples stay consecutive in time,
    which is what FLAC's prediction relies on. The Zarr v2 codec, `flac_numcodecs.Flac`,
    flattens such chunks in C order instead. Only int16 arrays are supported.

    Decoding accepts complete FLAC streams as well as bare frames; see
    `flac_numcodecs.flac.decode`.

    Only the parameters that are set are recorded in the array metadata. `blocksize` and
    `sample_rate` describe the FLAC frames, so when they are set, decoding checks them
    against every frame and raises if a frame differs. Leave them unset for an array
    whose frames vary, or to record nothing about them.

    Parameters
    ----------
    level : int, optional
        The FLAC compression level (0-8) used for encoding, by default None (5)
    blocksize : int, optional
        The block size of the frames, by default None (see `flac_numcodecs.flac.encode`).
        At most 4608 up to 48 kHz and 16384 above, the limits of the FLAC streamable subset
    sample_rate : int, optional
        The sample rate of the frames, by default None (48000 is written to encoded streams)
    """
    level: int | None = None
    blocksize: int | None = None
    sample_rate: int | None = None

    is_fixed_size = False

    def __post_init__(self):
        if self.level is not None and not 0 <= self.level <= 8:
            raise ValueError(f"level must be between 0 and 8, got {self.level}")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.blocksize is not None:
            cap = max_blocksize(self.sample_rate or 48000)
            if not 16 <= self.blocksize <= cap:
                raise ValueError(f"blocksize must be between 16 and {cap} at this sample rate, "
                                 f"got {self.blocksize}")

    @property
    def _configuration(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data):
        _, configuration = parse_named_configuration(data, "flac", require_configuration=False)
        return cls(**(configuration or {}))

    def to_dict(self):
        if self._configuration:
            return dict(name="flac", configuration=self._configuration)
        return dict(name="flac")

    def validate(self, *, shape, dtype, chunk_grid):
        _check_int16(dtype.to_native_dtype())

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        samples, frames = _decode_frames(chunk_bytes.as_numpy_array())
        self._check_frames(frames)
        shape = chunk_spec.shape
        if samples.size != prod(shape):
            raise ValueError(f"FLAC data decoded to {samples.size} samples, but the chunk "
                             f"has shape {shape}")
        nchannels = prod(shape[1:])
        if nchannels > FlacNumpyEncoder.max_channels and samples.shape[1] == 1:
            samples = samples.reshape(nchannels, shape[0]).T
        elif samples.shape[1] != nchannels:
            raise ValueError(f"FLAC data has {samples.shape[1]} channels, but the chunk has "
                             f"{nchannels}")
        samples = samples.reshape(shape).astype(chunk_spec.dtype.to_native_dtype(), copy=False)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(samples)

    def _check_frames(self, frames):
        for i, (sample_rate, blocksize) in enumerate(frames):
            if self.sample_rate is not None and sample_rate != self.sample_rate:
                raise ValueError(f"FLAC frame {i} has sample rate {sample_rate}, but the codec "
                                 f"configuration says {self.sample_rate}")
            # the last frame of a stream may be shorter
            last = i == len(frames) - 1
            if self.blocksize is not None and (blocksize > self.blocksize if last
                                               else blocksize != self.blocksize):
                raise ValueError(f"FLAC frame {i} has block size {blocksize}, but the codec "
                                 f"configuration says {self.blocksize}")

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        samples = chunk_array.as_numpy_array()
        _check_int16(samples.dtype)
        samples = samples.astype(np.int16, copy=False).reshape(samples.shape[0], -1)
        if samples.shape[1] > FlacNumpyEncoder.max_channels:
            samples = samples.T.reshape(-1)
        # FLAC reads the samples in native byte order and C order
        enc = encode(np.ascontiguousarray(samples), **self._configuration)
        return chunk_spec.prototype.buffer.from_bytes(enc)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        return await asyncio.to_thread(self._encode_sync, chunk_array, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
