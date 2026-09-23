[![PyPI version](https://badge.fury.io/py/flac-numcodecs.svg)](https://badge.fury.io/py/flac-numcodecs) ![tests](https://github.com/AllenNeuralDynamics/flac-numcodecs/actions/workflows/python-package.yml/badge.svg)

# FLAC - numcodecs implementation

[Numcodecs](https://numcodecs.readthedocs.io/en/latest/index.html) wrapper to the 
[FLAC](https://xiph.org/flac/index.html) audio codec using [pyFLAC](https://github.com/sonos/pyFLAC).

This implementation enables one to use FLAC as a compressor in 
[Zarr](https://zarr.readthedocs.io/en/stable/index.html).

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
