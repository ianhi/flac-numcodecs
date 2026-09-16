"""
Zarr v3 codec implementation for the FLAC codec.

Zarr v3 has its own codec protocol and its own registry: it discovers codecs through the
`zarr.codecs` entry point group, not through the `numcodecs.codecs` group used by the
numcodecs (Zarr v2) codec in `flac_numcodecs.flac`. This module provides the v3 codec, so
that `zarr.registry.get_codec_class("flac")` resolves after installing this package, with
no import or manual registration by the user.

The codec is only importable when `zarr` v3 is installed. Zarr imports the module lazily,
when a codec named `flac` is first looked up, so installations with Zarr v2 or without
Zarr are unaffected.
"""
from dataclasses import dataclass

import numpy as np

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer import Buffer
from zarr.core.array_spec import ArraySpec

from .flac import Flac as _Flac


@dataclass(frozen=True)
class Flac(BytesBytesCodec):
    """Zarr v3 codec for FLAC (Free Lossless Audio Codec).

    The chunk bytes are encoded as a FLAC stream. A chunk that is a bare run of FLAC
    frames, with no `fLaC` magic and no STREAMINFO metadata block, is also decoded: the
    four parameters below describe the stream the frames were taken from, which is what
    lets the missing header be synthesised. See `flac_numcodecs.Flac` for details.

    Parameters
    ----------
    blocksize : int
        The block size of the frames
    sample_rate : int
        The sample rate of the stream
    channels : int
        The number of channels of the stream
    bits_per_sample : int
        The bit depth of the stream
    """
    blocksize: int
    sample_rate: int
    channels: int
    bits_per_sample: int

    is_fixed_size = False

    @property
    def _codec(self):
        return _Flac(blocksize=self.blocksize, sample_rate=self.sample_rate,
                     channels=self.channels, bits_per_sample=self.bits_per_sample)

    @classmethod
    def from_dict(cls, data):
        return cls(**data["configuration"])

    def to_dict(self):
        return dict(
            name="flac",
            configuration=dict(
                blocksize=self.blocksize,
                sample_rate=self.sample_rate,
                channels=self.channels,
                bits_per_sample=self.bits_per_sample
            )
        )

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        dec = np.asarray(self._codec.decode(chunk_bytes.to_bytes()))
        return chunk_spec.prototype.buffer.from_bytes(dec.tobytes())

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        if dtype != np.int16:
            raise ValueError(f"Data type {dtype} not supported. Only int16 is supported")
        data = np.frombuffer(chunk_bytes.to_bytes(), dtype=np.int16)
        return chunk_spec.prototype.buffer.from_bytes(self._codec.encode(data))

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length
