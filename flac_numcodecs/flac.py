"""
Numcodecs Codec implementation for FLAC codec
    
The approach is:
- for compression: to convert to the audio file and read it as the encoded bytes
- for decompression: dump the encoded data to a tmp file and decode it using the codec

Multi-channel data exceeding the number of channels that can be encoded by the codec are reshaped to fit the 
compression procedure.
"""
from pathlib import Path
import threading
import numpy as np

import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy, ensure_contiguous_ndarray

from tempfile import TemporaryDirectory

from pyflac.encoder import _Encoder, EncoderInitException
from pyflac.decoder import _Decoder, DecoderInitException, DecoderProcessException, DecoderState

from pyflac._encoder import ffi as e_ffi
from pyflac._encoder import lib as e_lib

from pyflac._decoder import ffi as d_ffi
from pyflac._decoder import lib as d_lib


#### Helper classes for compression/decompression ####
class FlacNumpyEncoder(_Encoder):
    """
    The pyFLAC data encoder converts data from np.array to a FLAC file.

    Args:
        data (numpy.ndarray): the data to encode (n_samples x n_channels, up to 8 channels)
        output_file (pathlib.Path): Path to the output FLAC file, a temporary
            file will be created if unspecified.
        sample_rate (int): the sample rate
        compression_level (int): The compression level parameter that
            varies from 0 (fastest) to 8 (slowest). The default setting
            is 5, see https://en.wikipedia.org/wiki/FLAC for more details.
        blocksize (int): The size of the block to be returned in the
            callback. The default is 0 which allows libFLAC to determine
            the best block size.
        streamable_subset (bool): Whether to use the streamable subset for encoding.
            If true the encoder will check settings for compatibility. If false,
            the settings may take advantage of the full range that the format allows.
        verify (bool): If `True`, the encoder will verify it's own
            encoded output by feeding it through an internal decoder and
            comparing the original signal against the decoded signal.
            If a mismatch occurs, the `process` method will raise a
            `EncoderProcessException`.  Note that this will slow the
            encoding process by the extra time required for decoding and comparison.

    Raises:
        ValueError: If any invalid values are passed in to the constructor.
    """
    # the most channels a FLAC stream can hold
    max_channels = 8

    def __init__(self,
                 data,
                 output_file,
                 sample_rate=48000,
                 compression_level: int = 5,
                 blocksize: int = 0,
                 streamable_subset: bool = True,
                 verify: bool = False):
        assert data.shape[1] <= self.max_channels
        super().__init__()

        self.__raw_audio = data
        self.__output_file = output_file

        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._compression_level = compression_level
        self._streamable_subset = streamable_subset
        self._verify = verify

    def _init(self):
        """
        Initialise the encoder to write to a file.

        Raises:
            EncoderInitException: if initialisation fails.
        """
        c_output_filename = e_ffi.new('char[]', str(self.__output_file).encode('utf-8'))
        rc = e_lib.FLAC__stream_encoder_init_file(
            self._encoder,
            c_output_filename,
            e_lib._progress_callback,
            self._encoder_handle,
        )
        e_ffi.release(c_output_filename)
        if rc != e_lib.FLAC__STREAM_ENCODER_INIT_STATUS_OK:
            raise EncoderInitException(rc)

        self._initialised = True

    def process(self) -> bytes:
        """
        Process the audio data from the WAV file.

        Returns:
            (bytes): The FLAC encoded bytes.

        Raises:
            EncoderProcessException: if an error occurs when processing the samples
        """
        super().process(self.__raw_audio)
        self.finish()


class FlacNumpyDecoder(_Decoder):
    """
    The pyFLAC file decoder reads the encoded audio data directly from a FLAC
    file and returns a numpy array.

    Args:
        input_file (pathlib.Path): Path to the input FLAC file
        output_file (pathlib.Path): Path to the output WAV file, a temporary
            file will be created if unspecified.

    Raises:
        DecoderInitException: If initialisation of the decoder fails
    """

    def __init__(self, input_file):
        super().__init__()

        self.write_callback = self._write_callback
        self.total_samples = 0
        self.decoded_data_list = []
        self.decoded_data = None
        # pyflac's error callback sets `_event` after recording the error in `_error`, but
        # only its StreamDecoder and OneShotDecoder define it; without it, every libFLAC
        # error also prints an AttributeError traceback from the callback
        self._event = threading.Event()

        c_input_filename = d_ffi.new('char[]', str(input_file).encode('utf-8'))
        rc = d_lib.FLAC__stream_decoder_init_file(
            self._decoder,
            c_input_filename,
            d_lib._write_callback,
            d_ffi.NULL,
            d_lib._error_callback,
            self._decoder_handle,
        )
        d_ffi.release(c_input_filename)
        if rc != d_lib.FLAC__STREAM_DECODER_INIT_STATUS_OK:
            raise DecoderInitException(rc)

    def process(self):
        """
        Process the audio data from the FLAC file.

        Returns:
            (tuple): A tuple of the decoded numpy audio array, and the sample rate of the audio data.

        Raises:
            DecoderProcessException: if any fatal read, write, or memory allocation
                error occurred (meaning decoding must stop)
        """
        result = d_lib.FLAC__stream_decoder_process_until_end_of_stream(self._decoder)
        if self.state != DecoderState.END_OF_STREAM and not result:
            raise DecoderProcessException(str(self.state))

        self.finish()
        # libFLAC reports lost sync and CRC mismatches through the error callback and then
        # keeps decoding, so the output is incomplete or wrong unless the error is checked
        if self._error is not None:
            raise DecoderProcessException(self._error)
        self.decoded_data = np.vstack(self.decoded_data_list)

    def _write_callback(self, data: np.ndarray, sample_rate: int, num_channels: int, num_samples: int):
        """
        Internal callback to write the decoded data to a Numpy array file.
        """
        self.decoded_data_list.append(data)
        self.total_samples += num_samples


def _prepare_data(data):
    assert data.dtype == np.int16, "Data type not supported. Only int16 is supported"
    if data.ndim == 2 and data.shape[1] <= FlacNumpyEncoder.max_channels:
        return data
    return data.reshape(-1)[:, None]


def encode(data, level=5, blocksize=None, sample_rate=48000, tmpdir=None):
    """Encode an int16 array as a complete FLAC stream.

    A 1D array is encoded as a single channel, and a 2D array with up to 8 columns, the most
    a FLAC stream can hold, as one channel per column. Any other array is flattened in C
    order and encoded as a single channel.

    Parameters
    ----------
    data : numpy.ndarray
        The int16 data to encode
    level : int, optional
        The FLAC compression level (0-8), by default 5
    blocksize : int, optional
        The block size of the frames, by default None (the number of samples, or 0 to let
        libFLAC choose when there are 16 samples or fewer). Either way it is capped at the
        largest block size the FLAC streamable subset allows for `sample_rate`: 4608 up to
        48 kHz, 16384 above
    sample_rate : int, optional
        The sample rate recorded in the stream, by default 48000
    tmpdir : str or Path, optional
        The folder where to save tmp flac files, by default None (default temporary folder)

    Returns
    -------
    bytes
        The FLAC stream
    """
    data = _prepare_data(data)
    nsamples = data.shape[0]
    max_blocksize = Flac.max_blocksizes[0] if sample_rate <= 48000 else Flac.max_blocksizes[1]

    if blocksize is None:
        blocksize = min(nsamples, max_blocksize) if nsamples > 16 else 0
    else:
        blocksize = min(blocksize, max_blocksize)

    with TemporaryDirectory(dir=tmpdir) as tmp:
        tmp_file = Path(tmp) / "tmp.flac"
        encoder = FlacNumpyEncoder(data, tmp_file, compression_level=level,
                                   blocksize=blocksize, sample_rate=sample_rate)
        encoder.process()
        return tmp_file.read_bytes()


def decode(buf, tmpdir=None):
    """Decode a complete FLAC stream, or bare frames.

    Bare frames are complete FLAC frames without the file-level metadata that precedes them
    in a FLAC file: the `fLaC` signature and the metadata blocks, including STREAMINFO
    (RFC 9639, section 8). A byte range cut from a FLAC file along frame boundaries holds
    bare frames. libFLAC decodes them as they are: it synchronises on the first frame, and
    reads the block size, sample rate, channels and bit depth from each frame header. This
    holds for frames written at a standard sample rate and bit depth; a frame whose header
    refers to the missing STREAMINFO block instead cannot be decoded.

    Only 16-bit audio is supported.

    Parameters
    ----------
    buf : buffer-like
        The encoded bytes
    tmpdir : str or Path, optional
        The folder where to save tmp flac files, by default None (default temporary folder)

    Returns
    -------
    numpy.ndarray
        The decoded samples, with shape (n_samples, n_channels)

    Raises
    ------
    DecoderProcessException
        If libFLAC reports an error: a frame truncated by the end of the buffer, a buffer
        that does not start on a frame boundary, or corrupted data
    """
    with TemporaryDirectory(dir=tmpdir) as tmp:
        tmp_file = Path(tmp) / "tmp.flac"
        tmp_file.write_bytes(ensure_contiguous_ndarray(buf))
        decoder = FlacNumpyDecoder(tmp_file)
        decoder.process()
    return decoder.decoded_data


### NUMCODECS Codec ###
class Flac(Codec):
    """Codec for FLAC (Free Lossless Audio Codec).

    The implementation uses [pyFlac](https://github.com/sonos/pyFLAC).
    If the block has more than 8 channels, the data is flattened before compression.

    Decoding accepts both complete FLAC streams and bare frames; see
    `flac_numcodecs.flac.decode`.

    Parameters
    ----------
    level : int, optional
        The FLAC compression level (0-8), by default 5
    blocksize :  int, optional
        The block size used to chunk data, by default None
    sample_rate :  int, optional
        The internal sample rate used by FLAC, by default 48000
    tmpdir : str or Path, optional
        The folder where to save tmp flac files, by default None (default temporary folder)
    """
    codec_id = "flac"
    max_channels = FlacNumpyEncoder.max_channels
    max_blocksizes = [4608, 16384]

    def __init__(self, level=5, blocksize=None, sample_rate=48000, tmpdir=None):
        self.tmpdir = tmpdir
        self.compression_level = level
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        if sample_rate <= 48000:
            self.max_blocksize = self.max_blocksizes[0]
        else:
            self.max_blocksize = self.max_blocksizes[1]

    def encode(self, buf):
        return encode(buf, level=self.compression_level, blocksize=self.blocksize,
                      sample_rate=self.sample_rate, tmpdir=self.tmpdir)

    def decode(self, buf, out=None):
        return ndarray_copy(decode(buf, tmpdir=self.tmpdir), out)

    def get_config(self):
        # override to handle encoding dtypes
        return dict(
            id=self.codec_id,
            tmpdir=str(self.tmpdir) if self.tmpdir is not None else None,
            level=self.compression_level,
            blocksize=self.blocksize,
            sample_rate=self.sample_rate
        )


numcodecs.register_codec(Flac)
