# ffmpeg_ort

Decode input video using `ffmpeg`, then run inference on the decoded frames using `ort`, and then encode the processed frames to a new video.

This is effectively the same as our gstreamer pipeline, but using `ffmpeg` instead.

Compiling this requires you have the relevant `ffmpeg` packages in your system.
Look at ffmpeg crate docs for [guidance](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building).
