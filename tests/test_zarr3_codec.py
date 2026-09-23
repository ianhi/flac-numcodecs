import json
import subprocess
import sys

import numpy as np
import pytest

from flac_numcodecs import Flac

zarr = pytest.importorskip("zarr", minversion="3.1")

from flac_numcodecs.zarr3 import Flac as FlacZarr3

from helpers import make_noisy_sin_signals, split_header


BLOCKSIZE = 1000
CONFIGURATION = dict(level=5, blocksize=BLOCKSIZE, sample_rate=30000)


def create_array(store_path, data, chunks, codec=None):
    return zarr.create_array(store=str(store_path), shape=data.shape, chunks=chunks,
                             dtype=data.dtype, serializer=codec or FlacZarr3(**CONFIGURATION),
                             compressors=None)


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
    assert FlacZarr3.from_dict(dict(name="flac")) == FlacZarr3()
    with pytest.raises(ValueError):
        FlacZarr3.from_dict(dict(name="gzip"))

    # only the parameters that are set are recorded
    assert FlacZarr3().to_dict() == dict(name="flac")
    assert FlacZarr3(sample_rate=96000).to_dict() == dict(name="flac",
                                                          configuration=dict(sample_rate=96000))


@pytest.mark.zarr3
@pytest.mark.parametrize("kwargs", [dict(level=9), dict(sample_rate=0), dict(blocksize=8),
                                    dict(blocksize=8192), dict(blocksize=8192, sample_rate=44100)])
def test_zarr3_codec_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        FlacZarr3(**kwargs)
    # a block size that the streamable subset only allows above 48 kHz
    assert FlacZarr3(blocksize=8192, sample_rate=96000).blocksize == 8192


# a chunk with up to 8 channels is encoded with one FLAC channel per channel, a chunk with
# more as a single FLAC channel holding each channel in turn
@pytest.mark.zarr3
@pytest.mark.parametrize("shape, stream_shape", [
    ((10 * BLOCKSIZE,), (5 * BLOCKSIZE, 1)),
    ((10 * BLOCKSIZE, 2), (5 * BLOCKSIZE, 2)),
    ((10 * BLOCKSIZE, 8), (5 * BLOCKSIZE, 8)),
    ((10 * BLOCKSIZE, 2, 3), (5 * BLOCKSIZE, 6)),
    ((10 * BLOCKSIZE, 9), (45 * BLOCKSIZE, 1)),
])
def test_zarr3_chunk_layout(tmp_path, shape, stream_shape):
    data = make_noisy_sin_signals(shape=shape, dtype="int16")
    store_path = tmp_path / "roundtrip.zarr"

    z = create_array(store_path, data, chunks=(5 * BLOCKSIZE,) + shape[1:])
    z[:] = data

    codecs = json.loads((store_path / "zarr.json").read_text())["codecs"]
    assert codecs == [dict(name="flac", configuration=CONFIGURATION)]

    enc = (store_path / "c" / "/".join(["0"] * len(shape))).read_bytes()
    stream = Flac().decode(enc)
    assert stream.shape == stream_shape
    chunk = data[:5 * BLOCKSIZE].reshape(5 * BLOCKSIZE, -1)
    if stream_shape[1] == 1:
        assert np.all(stream.reshape(-1) == chunk.T.reshape(-1))
    else:
        assert np.all(stream == chunk)

    assert np.all(zarr.open_array(str(store_path), mode="r")[:] == data)


@pytest.mark.zarr3
def test_zarr3_big_endian_dtype(tmp_path):
    data = make_noisy_sin_signals(shape=(10 * BLOCKSIZE,), dtype="int16").astype(">i2")
    store_path = tmp_path / "big_endian.zarr"

    z = create_array(store_path, data, chunks=(5 * BLOCKSIZE,))
    z[:] = data

    enc = (store_path / "c" / "0").read_bytes()
    assert np.all(Flac().decode(enc).reshape(-1) == data[:5 * BLOCKSIZE])
    assert np.all(zarr.open_array(str(store_path), mode="r")[:] == data)


@pytest.mark.zarr3
def test_zarr3_rejects_other_dtypes(tmp_path):
    with pytest.raises(ValueError, match="Only int16 is supported"):
        zarr.create_array(store=str(tmp_path / "float.zarr"), shape=(100,), dtype="float32",
                          serializer=FlacZarr3(), compressors=None)


@pytest.mark.zarr3
def test_zarr3_sharded(tmp_path):
    data = make_noisy_sin_signals(shape=(20 * BLOCKSIZE,), dtype="int16")
    store_path = tmp_path / "sharded.zarr"

    z = zarr.create_array(store=str(store_path), shape=data.shape, dtype=data.dtype,
                          chunks=(5 * BLOCKSIZE,), shards=(10 * BLOCKSIZE,),
                          serializer=FlacZarr3(**CONFIGURATION), compressors=None)
    z[:] = data
    assert np.all(zarr.open_array(str(store_path), mode="r")[:] == data)


@pytest.mark.zarr3
def test_zarr3_bare_frame_chunks(tmp_path):
    """Read an array whose chunks are bare frames."""
    nblocks = 5
    data = make_noisy_sin_signals(shape=(2 * nblocks * BLOCKSIZE,), dtype="int16")
    store_path = tmp_path / "bare_frames.zarr"
    create_array(store_path, data, chunks=(nblocks * BLOCKSIZE,), codec=FlacZarr3())

    (store_path / "c").mkdir()
    for i in range(2):
        chunk = data[i * nblocks * BLOCKSIZE:(i + 1) * nblocks * BLOCKSIZE]
        _, frames = split_header(Flac(blocksize=BLOCKSIZE).encode(chunk))
        (store_path / "c" / str(i)).write_bytes(frames)

    z_read = zarr.open_array(str(store_path), mode="r")
    assert np.all(z_read[:] == data)
    assert np.all(z_read[:100] == data[:100])


@pytest.mark.zarr3
def test_zarr3_chunk_of_wrong_size_raises(tmp_path):
    data = make_noisy_sin_signals(shape=(2 * BLOCKSIZE,), dtype="int16")
    store_path = tmp_path / "wrong_size.zarr"
    create_array(store_path, data, chunks=(2 * BLOCKSIZE,), codec=FlacZarr3())

    (store_path / "c").mkdir()
    (store_path / "c" / "0").write_bytes(Flac().encode(data[:BLOCKSIZE]))

    with pytest.raises(ValueError, match="decoded to 1000 samples"):
        zarr.open_array(str(store_path), mode="r")[:]


@pytest.mark.zarr3
@pytest.mark.parametrize("kwargs, match", [
    (dict(sample_rate=44100), "sample rate 48000"),
    (dict(blocksize=500), "block size 1000"),
    (dict(blocksize=2000), "block size 1000"),
])
def test_zarr3_configuration_is_checked_against_frames(tmp_path, kwargs, match):
    data = make_noisy_sin_signals(shape=(5 * BLOCKSIZE,), dtype="int16")
    store_path = tmp_path / "mismatch.zarr"
    create_array(store_path, data, chunks=(5 * BLOCKSIZE,), codec=FlacZarr3(**kwargs))

    (store_path / "c").mkdir()
    _, frames = split_header(Flac(blocksize=BLOCKSIZE, sample_rate=48000).encode(data))
    (store_path / "c" / "0").write_bytes(frames)

    with pytest.raises(ValueError, match=match):
        zarr.open_array(str(store_path), mode="r")[:]


@pytest.mark.zarr3
def test_zarr3_configuration_matching_frames(tmp_path):
    # the last frame is shorter than the block size
    data = make_noisy_sin_signals(shape=(5 * BLOCKSIZE + 7,), dtype="int16")
    store_path = tmp_path / "match.zarr"
    create_array(store_path, data, chunks=data.shape,
                 codec=FlacZarr3(blocksize=BLOCKSIZE, sample_rate=48000))

    (store_path / "c").mkdir()
    _, frames = split_header(Flac(blocksize=BLOCKSIZE, sample_rate=48000).encode(data))
    (store_path / "c" / "0").write_bytes(frames)

    assert np.all(zarr.open_array(str(store_path), mode="r")[:] == data)


@pytest.mark.zarr3
def test_zarr3_chunk_with_wrong_channels_raises(tmp_path):
    # a stereo stream holds as many samples as a 4-channel chunk of half the length
    data = make_noisy_sin_signals(shape=(2 * BLOCKSIZE, 2), dtype="int16")
    store_path = tmp_path / "wrong_channels.zarr"
    zarr.create_array(store=str(store_path), shape=(BLOCKSIZE, 4), chunks=(BLOCKSIZE, 4),
                      dtype="int16", serializer=FlacZarr3(), compressors=None)

    (store_path / "c" / "0").mkdir(parents=True)
    (store_path / "c" / "0" / "0").write_bytes(Flac().encode(data))

    with pytest.raises(ValueError, match="2 channels, but the chunk has 4"):
        zarr.open_array(str(store_path), mode="r")[:]
