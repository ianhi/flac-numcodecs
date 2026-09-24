[![PyPI version](https://badge.fury.io/py/flac-numcodecs.svg)](https://badge.fury.io/py/flac-numcodecs) ![tests](https://github.com/AllenNeuralDynamics/flac-numcodecs/actions/workflows/python-package.yml/badge.svg)

# FLAC - numcodecs implementation

[Numcodecs](https://numcodecs.readthedocs.io/en/latest/index.html) wrapper to the 
[FLAC](https://xiph.org/flac/index.html) audio codec using [pyFLAC](https://github.com/sonos/pyFLAC).

This implementation lets [Zarr](https://zarr.readthedocs.io/en/stable/index.html) write and read
arrays stored as FLAC, as a compressor in Zarr v2 and as a [codec](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#codecs) in Zarr v3.

## Installation

Install via `pip`:

```
pip install flac-numcodecs
```

Or from sources:

```
git clone https://github.com/AllenNeuralDynamics/flac-numcodecs.git
cd flac-numcodecs
pip install .
```

## Usage

### Zarr v3

Installing this package registers a Zarr v3 [codec](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#codecs) named `flac`. It's
an array-to-bytes codec, so pass it as the `serializer` (in place of the default `bytes`
codec), not in `compressors`. It needs `zarr>=3.1` installed alongside this package.

To write an array:

```
import numpy as np
import zarr
from flac_numcodecs.zarr3 import Flac

# one second of 44.1 kHz stereo audio
t = np.arange(44100) / 44100
data = (np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)], axis=1)
        * 10000).astype("int16")

z = zarr.create_array("audio.zarr", shape=data.shape, chunks=(4096, 2), dtype="int16",
                      serializer=Flac(sample_rate=44100), compressors=None)
z[:] = data
```

Each chunk is stored as a FLAC stream, here with one FLAC channel per column. The array
metadata records `"codecs": [{"name": "flac", "configuration": {"sample_rate": 44100}}]`.
You don't need to import this package to read the data back. zarr-python finds the codec
automatically:

```
import zarr

z = zarr.open_array("audio.zarr", mode="r")
data = z[:]
```

FLAC supports at most 8 channels. For a chunk with more, the Zarr v3 codec writes the
channels one after another into a single FLAC channel, which compresses much better than
interleaving them. The Zarr v2 codec can't do this, because Zarr v2 doesn't pass the chunk
shape to the codec.

`flac_numcodecs.zarr3.Flac` takes four optional parameters:

- `level`: the FLAC compression level, from 0 to 8 (default 5)
- `blocksize`: the number of samples in each FLAC frame, from 16 to 65535 (default: as many
  as the chunk holds, up to 4608 at 48 kHz or below and 16384 above)
- `sample_rate`: the sample rate written into each FLAC stream, from 1 to 1048575 Hz
  (default 48000)
- `bits_per_sample`: the bit depth of the FLAC frames; only 16 is supported

Only the parameters you set are recorded in the array metadata. When `blocksize`,
`sample_rate` or `bits_per_sample` is set, reading checks it against every FLAC frame and
raises if they disagree. A chunk may hold either a complete FLAC stream or bare frames
(see below).

### Zarr v2

This is a simple example on how to use the `Flac` codec with `zarr`:

```
from flac_numcodecs import Flac

data = ... # any numpy array

# instantiate Flac compressor
flac_compressor = Flac(level=5)

z = zarr.array(data, compressor=flac_compressor)

data_read = z[:]
```
Available `**kwargs` can be browsed with: `Flac?`

**NOTE:** 
In order to reload in zarr an array saved with the `Flac`, you just need to have the `flac_numcodecs` package
installed.

In Zarr v2 arrays, a chunk with more than 8 channels is flattened into a single FLAC
channel, with the channels interleaved sample by sample. The Zarr v3 codec above stores such
chunks one channel at a time.

With zarr-python 3, pass `zarr_format=2` to `zarr.array` to create a Zarr v2 array. For Zarr
v3 arrays, use the codec described above.

### Decoding bare frames

Bare frames are FLAC frames without the file-level metadata that precedes them in a file:
the `fLaC` signature and the metadata blocks, including `STREAMINFO`
([RFC 9639, section 8](https://www.rfc-editor.org/rfc/rfc9639.html#section-8)). A byte range cut from a FLAC file
along frame boundaries contains bare frames, and the codec decodes them directly:

```
from flac_numcodecs import Flac

frames = ... # a byte range covering whole frames of a FLAC file

data = Flac().decode(frames)  # shape (n_samples, n_channels)
```

Most frame headers record the frame's sample rate and bit depth. A few leave them to
`STREAMINFO`:

- sample rates above 65535 Hz that aren't a multiple of 10, or above 655350 Hz
- bit depths other than 8, 12, 16, 20, 24 and 32 bits

For bare frames like these, the Zarr v3 codec uses its `sample_rate` and `bits_per_sample`
settings instead, and can't check them against the frames. Without `sample_rate`, the sample
rate is unknown, which doesn't affect the decoded samples. Without `bits_per_sample`, reading
raises `ValueError`.

Only 16-bit audio is supported. A buffer that does not hold whole, intact frames raises
`pyflac.decoder.DecoderProcessException`.
