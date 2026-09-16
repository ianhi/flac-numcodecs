import subprocess
import sys

import numpy as np
import pytest

from flac_numcodecs import Flac

zarr = pytest.importorskip("zarr", minversion="3")

from flac_numcodecs.zarr3 import Flac as FlacZarr3

from test_flac_codec import make_noisy_sin_signals, split_header


BLOCKSIZE = 1000
SAMPLE_RATE = 48000
CONFIGURATION = dict(blocksize=BLOCKSIZE, sample_rate=SAMPLE_RATE,
                     channels=1, bits_per_sample=16)


@pytest.mark.zarr3
def test_zarr3_codec_is_registered():
    # in a fresh interpreter, so that only the entry point can make the codec resolvable
    code = ("import zarr.registry;"
            "print(zarr.registry.get_codec_class('flac').__module__)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "flac_numcodecs.zarr3"


@pytest.mark.zarr3
def test_zarr3_codec_config():
    codec = FlacZarr3(**CONFIGURATION)
    assert codec.to_dict() == dict(name="flac", configuration=CONFIGURATION)
    assert FlacZarr3.from_dict(codec.to_dict()) == codec


@pytest.mark.zarr3
def test_zarr3_roundtrip(tmp_path):
    data = make_noisy_sin_signals(shape=(10 * BLOCKSIZE,), dtype="int16")

    z = zarr.create_array(store=str(tmp_path / "roundtrip.zarr"), shape=data.shape,
                          chunks=(5 * BLOCKSIZE,), dtype="int16",
                          compressors=[FlacZarr3(**CONFIGURATION)])
    z[:] = data

    z_read = zarr.open_array(str(tmp_path / "roundtrip.zarr"), mode="r")
    assert z_read.metadata.to_dict()["codecs"][-1]["configuration"] == CONFIGURATION
    assert np.all(z_read[:] == data)


@pytest.mark.zarr3
def test_zarr3_headerless_chunks(tmp_path):
    """Read an array whose chunks are bare runs of FLAC frames, as when the chunks are
    byte ranges pointing into existing FLAC files."""
    nblocks = 5
    data = make_noisy_sin_signals(shape=(2 * nblocks * BLOCKSIZE,), dtype="int16")

    store_path = tmp_path / "headerless.zarr"
    z = zarr.create_array(store=str(store_path), shape=data.shape,
                          chunks=(nblocks * BLOCKSIZE,), dtype="int16",
                          compressors=[FlacZarr3(**CONFIGURATION)])
    z[:] = data

    # replace the encoded chunks with headerless runs of frames
    for i in range(2):
        chunk = data[i * nblocks * BLOCKSIZE:(i + 1) * nblocks * BLOCKSIZE]
        _, frames = split_header(Flac(blocksize=BLOCKSIZE).encode(chunk))
        (store_path / "c" / str(i)).write_bytes(frames)

    z_read = zarr.open_array(str(store_path), mode="r")
    assert np.all(z_read[:] == data)
    assert np.all(z_read[:100] == data[:100])
