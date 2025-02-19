# ffmpeg_ort

Decode input video using `ffmpeg`, then run inference on the decoded frames using `ort`, and then encode the processed frames to a new video.

This is basically a gstreamer pipeline, but using `ffmpeg` instead.