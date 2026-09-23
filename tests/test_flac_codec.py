from flac_numcodecs import Flac
import numpy as np
import zarr
import pytest
from pyflac.decoder import DecoderProcessException

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

def array(data, chunks, compressor):
    # zarr 3 only takes a numcodecs compressor for zarr format 2 arrays, and does not read
    # None in a chunk shape as the full extent of that dimension
    if isinstance(chunks, tuple):
        chunks = tuple(n if c is None else c for c, n in zip(chunks, data.shape))
    return zarr.array(data, chunks=chunks, compressor=compressor, zarr_format=2)


@pytest.mark.zarr
def test_flac_zarr():
    for dtype in dtypes:
        print(f"\n\nZARR: testing dtype {dtype}\n\n")
        test_signals = generate_test_signals(dtype)

        compressor = Flac()

        for test_sig in test_signals:
            print(f"signal shape: {test_sig.shape}")
            if test_sig.ndim == 1:
                z = array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100].shape == test_sig[:100].shape
                assert z.nbytes > z.nbytes_stored()

                z = array(test_sig, chunks=(1000), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100].shape == test_sig[:100].shape

            elif test_sig.ndim == 2:
                z = array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape
                assert z.nbytes > z.nbytes_stored()

                z = array(test_sig, chunks=(1000, None), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape

                z = array(test_sig, chunks=(None, 10), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape

            else: # 3d
                z = array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape
                assert z.nbytes > z.nbytes_stored()

                z = array(test_sig, chunks=(1000, 2, None), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape

                z = array(test_sig, chunks=(None, 2, 3), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape


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


if __name__ == '__main__':
    test_flac_numcodecs()
    test_flac_zarr()
