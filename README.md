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

Installing this package registers a Zarr v3 [codec](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#codecs) named `flac`. It
needs `zarr>=3.1` installed alongside this package.

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
Reading it back needs no import from this package: once it is installed, Zarr finds the
codec by its name:

```
import zarr

z = zarr.open_array("audio.zarr", mode="r")
data = z[:]
```

A FLAC stream holds at most 8 channels. The Zarr v3 codec stores a chunk with more as a
single FLAC channel that holds each channel in turn, which compresses much better than
interleaving them sample by sample. Zarr v2 arrays cannot use this layout, because a Zarr v2
codec does not receive the chunk shape it would need to undo it.

`flac_numcodecs.zarr3.Flac` takes three optional parameters, all used when writing:

- `level`: the FLAC compression level, from 0 to 8 (default 5)
- `blocksize`: the number of samples in each FLAC frame (default: as many as the chunk
  holds, up to 4608 at 48 kHz or below and 16384 above)
- `sample_rate`: the sample rate written into each FLAC stream (default 48000)

Only the parameters you set are recorded in the array metadata. When `blocksize` or
`sample_rate` is set, reading checks it against every FLAC frame and raises if they
disagree, so the metadata cannot silently misdescribe the audio. Reading needs no
parameters, and a chunk may hold either a complete FLAC stream or bare frames (see below).

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

In Zarr v2 arrays, a chunk with more than 8 channels, the most a FLAC stream holds, is
flattened in C order into a single FLAC channel, which interleaves the channels sample by
sample. The Zarr v3 codec above stores such chunks one channel at a time.

With version 3 of the `zarr` package installed, this example needs `zarr_format=2` in the
call to `zarr.array`, because it creates a Zarr v2 array. For Zarr v3 arrays, use the
codec described above.

### Decoding bare frames

A FLAC file starts with file-level metadata, the `fLaC` signature and metadata blocks such
as `STREAMINFO` ([RFC 9639, section 8](https://www.rfc-editor.org/rfc/rfc9639.html#section-8)), followed by
the frames. Each frame has its own header, so the codec also decodes bare frames: frames
without the file-level metadata, such as a byte range cut from a FLAC file along frame
boundaries:

```
from flac_numcodecs import Flac

frames = ... # a byte range covering whole frames of a FLAC file

data = Flac().decode(frames)  # shape (n_samples, n_channels)
```

It works for 16-bit audio at a standard sample rate, which covers the output of common
encoders such as libFLAC and ffmpeg. A buffer that does not hold whole, intact frames,
because it starts or ends mid-frame or is corrupted, raises
`pyflac.decoder.DecoderProcessException`.
