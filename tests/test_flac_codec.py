from flac_numcodecs import Flac
import numpy as np
import numcodecs
import zarr
import pytest
from pyflac.decoder import DecoderProcessException

from flac_numcodecs.flac import FlacNumpyEncoder
from flac_numcodecs.zarr3 import Flac as FlacZarr3

from helpers import make_noisy_sin_signals, split_header

DEBUG = False

# dtypes = ["int8", "int16", "int32", "float32"]
dtypes = ["int16"]

def run_all_options(data):
    dtype = data.dtype
    for level in range(1, 9):
        for bs in [None, 100, 1000]:
            print(f"Dtype {dtype} - level {level} - blocksize {bs}")
            cod = Flac(level=level)
            enc = cod.encode(data)
            dec = cod.decode(enc)

            assert len(enc) < len(dec)
            print("CR", len(dec) / len(enc))
            data_dec = np.frombuffer(dec, dtype=dtype).reshape(data.shape)
            assert np.all(data_dec == data)
        

def generate_test_signals(dtype):
    test1d = make_noisy_sin_signals(shape=(3000,), dtype=dtype)
    test1d_long = make_noisy_sin_signals(shape=(200000,), dtype=dtype)
    test2d = make_noisy_sin_signals(shape=(3000, 10), dtype=dtype)
    test2d_long = make_noisy_sin_signals(shape=(200000, 20), dtype=dtype)
    test2d_extra = make_noisy_sin_signals(shape=(3000, 300), dtype=dtype)
    test3d = make_noisy_sin_signals(shape=(1000, 5, 5), dtype=dtype)

    return [test1d, test1d_long, test2d, test2d_long, test2d_extra, test3d]

@pytest.mark.numcodecs
def test_flac_numcodecs():
    for dtype in dtypes:
        print(f"\n\nNUMCODECS: testing dtype {dtype}\n\n")

        test_signals = generate_test_signals(dtype)

        for test_sig in test_signals:
            print(f"signal shape: {test_sig.shape}")
            run_all_options(test_sig)

ZARR_CASES = [
    ((3000,), (3000,)),
    ((200000,), (1000,)),
    ((3000, 10), (3000, 10)),
    ((200000, 20), (1000, 20)),
    ((200000, 20), (200000, 10)),
    ((3000, 300), (3000, 300)),
    ((1000, 5, 5), (1000, 5, 5)),
    ((1000, 5, 5), (1000, 2, 5)),
    ((1000, 5, 5), (1000, 2, 3)),
]


# the Zarr v2 codec is a compressor, the Zarr v3 codec a serializer
@pytest.mark.zarr
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("shape, chunks", ZARR_CASES)
def test_flac_zarr(zarr_format, shape, chunks):
    data = make_noisy_sin_signals(shape=shape, dtype="int16")
    if zarr_format == 2:
        codecs = dict(compressors=Flac())
    else:
        codecs = dict(serializer=FlacZarr3(), compressors=None)

    z = zarr.create_array({}, shape=shape, chunks=chunks, dtype="int16",
                          zarr_format=zarr_format, **codecs)
    z[:] = data

    assert z.nbytes > z.nbytes_stored()
    assert np.all(z[:] == data)
    assert np.all(z[:100] == data[:100])


@pytest.mark.bare_frames
def test_flac_partial_bare_frames():
    blocksize = 1000
    nblocks = 10
    data = make_noisy_sin_signals(shape=(nblocks * blocksize,), dtype="int16")

    # frames are encoded independently, so the frames of an encoded prefix of the signal
    # are a byte-prefix of the frames of the whole signal: that gives us frame boundaries
    _, frames = split_header(Flac(blocksize=blocksize).encode(data))
    offsets = {nb: len(split_header(Flac(blocksize=blocksize).encode(data[:nb * blocksize]))[1])
               for nb in (3, 7, 8)}

    cod = Flac()

    # a run of frames from the middle of the stream
    dec = cod.decode(frames[offsets[3]:offsets[7]])
    assert np.all(dec.reshape(-1) == data[3 * blocksize:7 * blocksize])

    # a run that includes the (possibly shorter) final frame
    dec = cod.decode(frames[offsets[8]:])
    assert np.all(dec.reshape(-1) == data[8 * blocksize:])


@pytest.mark.bare_frames
@pytest.mark.parametrize("blocksize", [16, 1000, 4608])
@pytest.mark.parametrize("sample_rate", [44100, 12345])
@pytest.mark.parametrize("nchannels", [1, 2])
def test_flac_bare_frames_roundtrip(blocksize, sample_rate, nchannels):
    shape = (10 * blocksize + 7, nchannels) if nchannels > 1 else (10 * blocksize + 7,)
    data = make_noisy_sin_signals(shape=shape, dtype="int16")
    _, frames = split_header(Flac(blocksize=blocksize, sample_rate=sample_rate).encode(data))

    dec = Flac().decode(frames)
    assert dec.shape == (shape[0], nchannels)
    assert np.all(dec.reshape(data.shape) == data)


# rates a frame header cannot state, so that each frame refers to STREAMINFO for its rate
@pytest.mark.bare_frames
@pytest.mark.parametrize("sample_rate", [96001, 768000, 1048575])
@pytest.mark.parametrize("nchannels", [1, 2])
def test_flac_bare_frames_rate_from_streaminfo(sample_rate, nchannels):
    blocksize = 1000
    shape = (10 * blocksize + 7, nchannels) if nchannels > 1 else (10 * blocksize + 7,)
    data = make_noisy_sin_signals(shape=shape, dtype="int16")
    enc = Flac(blocksize=blocksize, sample_rate=sample_rate).encode(data)
    assert np.all(Flac().decode(enc).reshape(data.shape) == data)

    _, frames = split_header(enc)
    # the low 4 bits of the frame header's third byte: 0 means "see STREAMINFO"
    assert frames[2] & 0x0F == 0
    assert np.all(Flac().decode(frames).reshape(data.shape) == data)


@pytest.mark.bare_frames
def test_flac_bare_frames_bit_depth_from_streaminfo_raises(tmp_path):
    # a frame header states only 8, 12, 16, 20, 24 or 32 bits, and takes any other bit depth
    # from STREAMINFO, so bare 15-bit frames cannot be decoded
    data = np.full((1000, 1), 3, dtype=np.int16)
    flac_file = tmp_path / "15bit.flac"
    encoder = FlacNumpyEncoder(data, flac_file, blocksize=1000, streamable_subset=False)
    encoder._channels = 1
    encoder._bits_per_sample = 15
    encoder._init()
    encoder.process()

    _, frames = split_header(flac_file.read_bytes())
    # bits 1-3 of the frame header's fourth byte: 0 means "see STREAMINFO"
    assert (frames[3] >> 1) & 0b111 == 0
    with pytest.raises(ValueError, match="take their bit depth from the STREAMINFO"):
        Flac().decode(frames)


@pytest.mark.numcodecs
def test_flac_decode_stream_with_id3_tag():
    data = make_noisy_sin_signals(shape=(5000,), dtype="int16")
    # an ID3v2.4 header for a tag of 10 bytes, followed by the tag
    id3 = b"ID3\x04\x00\x00\x00\x00\x00\x0a" + bytes(10)
    dec = Flac().decode(id3 + Flac().encode(data))
    assert np.all(dec.reshape(-1) == data)


@pytest.mark.numcodecs
def test_flac_decode_errors_raise():
    blocksize = 1000
    data = make_noisy_sin_signals(shape=(10 * blocksize,), dtype="int16")
    enc = Flac(blocksize=blocksize).encode(data)
    _, frames = split_header(enc)

    corrupted = bytearray(enc)
    corrupted[len(enc) // 2] ^= 0xFF
    for buf in [enc[:-50], frames[:-50], frames[10:], bytes(corrupted)]:
        with pytest.raises(DecoderProcessException):
            Flac().decode(buf)


@pytest.mark.numcodecs
@pytest.mark.parametrize("nchannels, stream_channels", [(8, 8), (9, 1)])
def test_flac_channels(nchannels, stream_channels):
    # up to 8 channels, the most a FLAC stream holds, are encoded as separate FLAC channels
    data = make_noisy_sin_signals(shape=(1000, nchannels), dtype="int16")
    dec = Flac().decode(Flac().encode(data))
    assert dec.shape == (1000 * nchannels // stream_channels, stream_channels)
    assert np.all(dec.reshape(data.shape) == data)


@pytest.mark.numcodecs
def test_flac_config():
    cod = Flac(level=8, blocksize=1000, sample_rate=16000)
    assert numcodecs.get_codec(cod.get_config()).get_config() == cod.get_config()

    default_config = Flac().get_config()
    assert numcodecs.get_codec(default_config).get_config() == default_config


if __name__ == '__main__':
    pytest.main([__file__])
