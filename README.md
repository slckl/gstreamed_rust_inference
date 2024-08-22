# Gstreamer x ML in Rust

This repo contains a few experimental examples for running a computer vision pipeline on video or image inputs using rust under Apache 2 or MIT license.

[gstreamer](https://gitlab.freedesktop.org/gstreamer/gstreamer-rs) is used for video decoding/encoding and display, while inference is run either via [candle](https://github.com/huggingface/candle) or [ort](https://github.com/pykeio/ort).

There are 2 main examples currently:
- `gstreamed_candle` - runs yolov8 on image or video input using `candle` library.
- `gstreamed_ort` - runs yolov8 on image or video input using onnxruntime via `ort` library.

Only object detection has been implemented here, there is no support for segmentation or pose estimation yet.

## gstreamed_candle

This is a largely adapted yolov8 example from candle examples, using the same model, just adapted to run inside a gstreamer pipeline. Models are downloaded from huggingface hub, from candle example models.

Run from workspace directory as follows:
```shell
cargo run -r -p gstreamed_candle -- <INPUT> 
```

Additional CLI options:
- `--cuda` - launches candle pipeline with cuda

## gstreamed_ort

This is a modified version of `gstreamed_candle` to use `onnxruntime` via `ort` instead of `candle`. This version boasts better performance and includes a few more whistles because of it.

Run from workspace directory as follows:
```shell
cargo run -r -p gstreamed_ort -- <INPUT>
```

Additional CLI options:
- `--cuda` - launches ort pipeline with cuda, may fail silently, watch your logs.
- `--model <MODEL>` - allows specifying path to your own yolov8 .onnx file. Code assumes it's using COCO classes.
- `--live` - whether to display "live" the processed video using gst's `autodisplaysink`. Currently very slow on nvidia (idk why).

### Models

This version uses yolov8 onnx models from [ultralytics](https://github.com/ultralytics/ultralytics).

Models can be downloaded and exported by
1. Installing ultralytics cli: `pip install ultralytics`
2. Using cli to export the download & export the desired model: `yolo export model=yolov8m.pt format=onnx simplify dynamic`

### Onnx runtime

`ort` supports downloading pre-built onnxruntime binaries, if your arch supports it, but it can be tricky.

TODO description of download vs system onnxruntime features