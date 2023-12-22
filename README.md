# napari-czifile2

<a href="https://pypi.org/project/napari-czifile2/">
    <img src="https://img.shields.io/pypi/v/napari-czifile2" alt="PyPI" />
</a>
<a href="https://github.com/BodenmillerGroup/napari-czifile2/blob/main/LICENSE.md">
    <img src="https://img.shields.io/pypi/l/napari-czifile2" alt="License" />
</a>
<a href="https://www.python.org/">
    <img src="https://img.shields.io/pypi/pyversions/napari-czifile2" alt="Python" />
</a>
<a href="https://github.com/BodenmillerGroup/napari-czifile2/issues">
    <img src="https://img.shields.io/github/issues/BodenmillerGroup/napari-czifile2" alt="Issues" />
</a>
<a href="https://github.com/BodenmillerGroup/napari-czifile2/pulls">
    <img src="https://img.shields.io/github/issues-pr/BodenmillerGroup/napari-czifile2" alt="Pull requests" />
</a>

# This fork
The goal of this fork is to load multi scene images in different viewers.

## Installation
pip install git+https://github.com/pnewstein/napari-czifile2

## Usage 
Dragging on a czi file with multiple scenes will load the first scene
on that viewer, and the other scenes on newly spawned viewers.

### from python

```python
from napari_czifile2 import reader_function_with_args
path = "path/to/image.czi"
# load scene 0
for data, metadata, _ in reader_function_with_args(path, scene_index=0, next_scene_inds=[]):
    viewer.add_image(data=data, **metadata)

# load scenes 0 - 5
for data, metadata, _ in reader_function_with_args(path, scene_index=0, next_scene_inds=list(range(1, 6))):
    viewer.add_image(data=data, **metadata)
```


# BodenmillerGroup/napari-czifile2

Carl Zeiss Image (.czi) file type support for napari

Open .czi files and interactively view scenes co-registered in the machine's coordinate system using napari

## Installation

You can install napari-czifile2 via [pip](https://pypi.org/project/pip/):

    pip install napari-czifile2

Alternatively, you can install napari-czifile2 via [conda](https://conda.io/):

    conda install -c conda-forge napari-czifile2

## Authors

Created and maintained by [Jonas Windhager](mailto:jonas@windhager.io) until February 2023.

Maintained by [Milad Adibi](mailto:milad.adibi@uzh.ch) from February 2023.

## Contributing

[Contributing](https://github.com/BodenmillerGroup/napari-czifile2/blob/main/CONTRIBUTING.md)

## Changelog

[Changelog](https://github.com/BodenmillerGroup/napari-czifile2/blob/main/CHANGELOG.md)

## License

[MIT](https://github.com/BodenmillerGroup/napari-czifile2/blob/main/LICENSE.md)
