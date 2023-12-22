from typing import Union, Optional, List, Tuple
from multiprocessing import cpu_count
from pathlib import Path
from functools import partial

import numpy as np

from .io import CZISceneFile
import napari


def napari_get_reader(path):
    if isinstance(path, list):
        if any(Path(p).suffix.lower() != ".czi" for p in path):
            return None
    else:
        if Path(path).suffix.lower() != ".czi":
            return None
    return partial(reader_function_with_args, scene_index=0, next_scene_inds=None)


def reader_function_with_args(
    paths: Union[List[Union[str, Path]], [str, Path]],
    scene_index: int,
    next_scene_inds: Optional[List[int]],
) -> List[Tuple[np.ndarray, dict, str]]:
    """
    Can be used with a partial to create a napari reader Function
    other arguments are:
    scene_index (int): the index of the scene to load
    next_scene_inds (optional[list[int]]) an ordered list of scene indecies to
    load after the current one
    """
    layer_data = []
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        num_scenes = CZISceneFile.get_num_scenes(path)
        if num_scenes != 1 and next_scene_inds is None:
            # TODO ask the user which ones to load
            next_scene_inds = list(range(num_scenes))  # for now asssume load all
        if num_scenes != 1 and next_scene_inds is not None:
            # if there are more to do after this, rerun this with the next arg
            if next_scene_inds:
                next_scene_index = next_scene_inds.pop()
                viewer = napari.viewer.Viewer()
                for data, metadata, _ in reader_function_with_args(
                    paths, next_scene_index, next_scene_inds
                ):
                    viewer.add_image(data=data, **metadata)
        # load this scene_index
        with CZISceneFile(path, scene_index) as f:
            data = f.as_tzcyx0_array(max_workers=cpu_count())
            # https://github.com/BodenmillerGroup/napari-czifile2/issues/5
            contrast_limits = None
            if data.dtype == np.uint16:
                contrast_limits = (0, 65535)
            # https://github.com/napari/napari/issues/2348
            if not f.is_rgb:
                data = data[:, :, :, :, :, 0]
            metadata = {
                "rgb": f.is_rgb,
                "channel_axis": 2,
                "translate": (f.pos_t_seconds, f.pos_z_um, f.pos_y_um, f.pos_x_um),
                "scale": (
                    f.scale_t_seconds,
                    f.scale_z_um,
                    f.scale_y_um,
                    f.scale_x_um,
                ),
                "contrast_limits": contrast_limits,
            }
            if f.channel_names is not None:
                if num_scenes == 1:
                    metadata["name"] = f.channel_names
                elif num_scenes > 1:
                    metadata["name"] = [
                        f"S{scene_index:02d} {channel_name}"
                        for channel_name in f.channel_names
                    ]
        layer_data.append((data, metadata, "image"))
    return layer_data
