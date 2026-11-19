"""Write RGB camera frames as H.264 video."""

from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import av
import numpy as np


def write_video(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    frames = iter(frames)
    first_frame = next(frames)
    height, width, _ = first_frame.shape

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.thread_type = "FRAME"
        stream.thread_count = 2
        stream.options = {"crf": "21", "profile": "high"}

        for frame in chain((first_frame,), frames):
            if frame.shape != first_frame.shape:
                raise ValueError(f"Camera frame shape changed in {path}: {first_frame.shape} to {frame.shape}")
            video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
