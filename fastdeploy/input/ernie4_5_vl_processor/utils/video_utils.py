"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import io
import os
from tempfile import NamedTemporaryFile as ntf

import numpy as np

try:
    # moviepy 1.0
    import moviepy.editor as mp
except:
    # moviepy 2.0
    import moviepy as mp


def is_gif(data: bytes) -> bool:
    """
    check if a bytes is a gif based on the magic head
    """
    return data[:6] in (b"GIF87a", b"GIF89a")


class _NumpyFrame:
    """Wrapper so that frame[idx].asnumpy() keeps working with paddlecodec."""

    def __init__(self, array):
        self._array = array

    def asnumpy(self):
        return self._array


class VideoReaderWrapper:
    """paddlecodec VideoDecoder wrapper with GIF support."""

    def __init__(self, video_path, *args, **kwargs):
        import sys

        import paddle

        with ntf(delete=True, suffix=".gif") as gif_file:
            gif_input = None
            self.original_file = None
            if isinstance(video_path, str):
                self.original_file = video_path
                if video_path.lower().endswith(".gif"):
                    gif_input = video_path
            elif isinstance(video_path, bytes):
                if is_gif(video_path):
                    gif_file.write(video_path)
                    gif_input = gif_file.name
            elif isinstance(video_path, io.BytesIO):
                video_path.seek(0)
                tmp_bytes = video_path.read()
                video_path.seek(0)
                if is_gif(tmp_bytes):
                    gif_file.write(tmp_bytes)
                    gif_input = gif_file.name

            if gif_input is not None:
                clip = mp.VideoFileClip(gif_input)
                mp4_file = ntf(delete=False, suffix=".mp4")
                clip.write_videofile(mp4_file.name, verbose=False, logger=None)
                clip.close()
                video_path = mp4_file.name
                self.original_file = video_path

            if "torchcodec" in sys.modules:
                del sys.modules["torchcodec"]
            paddle.enable_compat(scope={"torchcodec"})
            from torchcodec.decoders import VideoDecoder

            num_threads = kwargs.get("num_threads", 0)
            self._decoder = VideoDecoder(
                video_path,
                seek_mode="exact",
                num_ffmpeg_threads=num_threads,
                device="cpu",
            )
            paddle.disable_compat()

    def __len__(self):
        return self._decoder.metadata.num_frames

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            frame = self._decoder.get_frames_at(indices=[int(key)]).data[0]
            return _NumpyFrame(frame.numpy())
        indices = list(key) if not isinstance(key, list) else key
        frames = self._decoder.get_frames_at(indices=indices).data
        return _NumpyFrame(frames.numpy())

    def get_avg_fps(self):
        return self._decoder.metadata.average_fps

    def __del__(self):
        if self.original_file and os.path.exists(self.original_file):
            os.remove(self.original_file)
