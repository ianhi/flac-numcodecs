from flac_numcodecs import Flac
import numpy as np
import numcodecs
import zarr
import pytest

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
        

def make_noisy_sin_signals(shape=(30000,), sin_f=100, sin_amp=50, noise_amp=5,
                           sample_rate=30000, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) <= 3
    if len(shape) == 1:
        y = np.sin(2 * np.pi * sin_f * np.arange(shape[0]) / sample_rate) * sin_amp
        y = y + np.random.randn(shape[0]) * noise_amp
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

@pytest.mark.zarr
def test_flac_zarr():
    for dtype in dtypes:
        print(f"\n\nZARR: testing dtype {dtype}\n\n")
        test_signals = generate_test_signals(dtype)

        compressor = Flac()

        for test_sig in test_signals:
            print(f"signal shape: {test_sig.shape}")
            if test_sig.ndim == 1:
                z = zarr.array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100].shape == test_sig[:100].shape
                assert z.nbytes > z.nbytes_stored

                z = zarr.array(test_sig, chunks=(1000), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100].shape == test_sig[:100].shape

            elif test_sig.ndim == 2:
                z = zarr.array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape
                assert z.nbytes > z.nbytes_stored

                z = zarr.array(test_sig, chunks=(1000, None), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape

                z = zarr.array(test_sig, chunks=(None, 10), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape

            else: # 3d
                z = zarr.array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape
                assert z.nbytes > z.nbytes_stored

                z = zarr.array(test_sig, chunks=(1000, 2, None), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape

                z = zarr.array(test_sig, chunks=(None, 2, 3), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape



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


@pytest.mark.headerless
def test_flac_headerless_roundtrip():
    blocksize = 1000
    data = make_noisy_sin_signals(shape=(10 * blocksize,), dtype="int16")

    _, frames = split_header(Flac(blocksize=blocksize).encode(data))

    cod = Flac(blocksize=blocksize, sample_rate=48000, channels=1, bits_per_sample=16)
    dec = cod.decode(frames)
    data_dec = np.frombuffer(dec, dtype=data.dtype)
    assert np.all(data_dec == data)


@pytest.mark.headerless
def test_flac_headerless_partial_frames():
    blocksize = 1000
    nblocks = 10
    data = make_noisy_sin_signals(shape=(nblocks * blocksize,), dtype="int16")

    # frames are encoded independently, so the frames of an encoded prefix of the signal
    # are a byte-prefix of the frames of the whole signal: that gives us frame boundaries
    _, frames = split_header(Flac(blocksize=blocksize).encode(data))
    offsets = {}
    for nb in range(1, nblocks):
        _, prefix_frames = split_header(Flac(blocksize=blocksize).encode(data[:nb * blocksize]))
        assert frames.startswith(prefix_frames)
        offsets[nb] = len(prefix_frames)

    cod = Flac(blocksize=blocksize, sample_rate=48000, channels=1, bits_per_sample=16)

    # a run of frames from the middle of the stream
    dec = cod.decode(frames[offsets[3]:offsets[7]])
    data_dec = np.frombuffer(dec, dtype=data.dtype)
    assert np.all(data_dec == data[3 * blocksize:7 * blocksize])

    # a run that includes the (possibly shorter) final frame
    dec = cod.decode(frames[offsets[8]:])
    data_dec = np.frombuffer(dec, dtype=data.dtype)
    assert np.all(data_dec == data[8 * blocksize:])


@pytest.mark.headerless
def test_flac_headerless_codec_decodes_complete_stream():
    blocksize = 1000
    data = make_noisy_sin_signals(shape=(10 * blocksize,), dtype="int16")
    enc = Flac(blocksize=blocksize).encode(data)

    cod = Flac(blocksize=blocksize, sample_rate=48000, channels=1, bits_per_sample=16)
    data_dec = np.frombuffer(cod.decode(enc), dtype=data.dtype)
    assert np.all(data_dec == data)

    # the default codec is unaffected by the headerless options
    default_dec = np.frombuffer(Flac().decode(enc), dtype=data.dtype)
    assert np.all(default_dec == data)


@pytest.mark.headerless
def test_flac_headerless_config():
    cod = Flac(level=8, blocksize=1000, sample_rate=16000, channels=2, bits_per_sample=16)
    assert numcodecs.get_codec(cod.get_config()).get_config() == cod.get_config()

    default_config = Flac().get_config()
    assert default_config["channels"] is None
    assert numcodecs.get_codec(default_config).get_config() == default_config

    with pytest.raises(ValueError):
        Flac(channels=1, bits_per_sample=16)


if __name__ == '__main__':
    test_flac_numcodecs()
    test_flac_zarr()
    test_flac_headerless_roundtrip()
    test_flac_headerless_partial_frames()
    test_flac_headerless_codec_decodes_complete_stream()
    test_flac_headerless_config()
