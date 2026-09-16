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

### Decoding headerless frames

A buffer sliced out of the middle of a FLAC file is a run of complete frames with no
`fLaC` magic and no `STREAMINFO` metadata block, which libFLAC refuses to decode. Give the
codec the four stream parameters and it synthesises the missing header when it sees a
buffer that does not start with the `fLaC` magic:

```
from flac_numcodecs import Flac

# parameters of the stream the frames were taken from
flac_codec = Flac(blocksize=4096, sample_rate=16000, channels=1, bits_per_sample=16)

frames = ... # bytes covering whole frames, e.g. a byte range of a FLAC file

data = flac_codec.decode(frames)
```

Complete streams are still decoded as before, so a codec configured this way handles both.

### Zarr v3

Zarr v3 has its own codec protocol and registry. Installing this package registers a v3
codec under the name `flac`, so an array whose metadata names it is read with no import
or registration by the user:

```
import zarr

z = zarr.open_array("array.zarr", mode="r")  # "codecs": [{"name": "flac", "configuration": {...}}]
data = z[:]
```

The configuration holds the same four parameters, and chunks may be either complete FLAC
streams or bare runs of frames. To create such an array:

```
import zarr
from flac_numcodecs.zarr3 import Flac

z = zarr.create_array("array.zarr", shape=data.shape, chunks=(4096,), dtype="int16",
                      compressors=[Flac(blocksize=4096, sample_rate=16000,
                                        channels=1, bits_per_sample=16)])
z[:] = data
```

The v3 codec requires `zarr>=3` (`pip install flac-numcodecs[zarr]`); the numcodecs codec
above is unaffected by which version of Zarr, if any, is installed.

**NOTE:** 
In order to reload in zarr an array saved with the `Flac`, you just need to have the `flac_numcodecs` package
installed.