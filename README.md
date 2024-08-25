# Gstreamer x ML in Rust

This repo contains a few experimental examples for running a computer vision pipeline on video or image inputs using rust under Apache 2 or MIT license.

[gstreamer](https://gitlab.freedesktop.org/gstreamer/gstreamer-rs) is used for video decoding/encoding and display, while inference is run either via [candle](https://github.com/huggingface/candle) or [ort](https://github.com/pykeio/ort).

We also implement basic object tracking via SORT tracker, via [similari](https://github.com/insight-platform/Similari).

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

This will process the `<INPUT>` video file and save the processed output in `<INPUT>.out.mkv` video file.

Additional CLI options:
- `--cuda` - launches candle pipeline with cuda

## gstreamed_ort

This is a modified version of `gstreamed_candle` to use `onnxruntime` via `ort` instead of `candle`. This version boasts better performance and includes a few more whistles because of it.

Run from workspace directory as follows:
```shell
cargo run -r -p gstreamed_ort -- <INPUT>
```
In case of video files, this will process the `<INPUT>` video file and save the processed output in `<INPUT>.out.mkv` video file.

In case of image files, this will output `<INPUT>.out.jpg`.

Additional CLI options:
- `--cuda` - launches ort pipeline with cuda, may fail silently, watch your logs.
- `--model <MODEL>` - allows specifying path to your own yolov8 .onnx file. Code assumes it's using COCO classes.
- `--live` - whether to display "live" the processed video using gst's `autodisplaysink`. Currently very slow on nvidia (idk why).

### Models

This version uses yolov8 onnx models from [ultralytics](https://github.com/ultralytics/ultralytics).

Models can be downloaded and exported by
1. Installing ultralytics cli: `pip install ultralytics`
2. Using cli to export the download & export the desired model: `yolo export model=yolov8m.pt format=onnx simplify dynamic`

## Performance

Currently, with yolov8 `ort` seems to be considerably faster than `candle`.

These benchmarks are very much not scientific, but do show practical difference.

Running inference on the same 1280x720 30 fps file, using yolov8s model, average times.

Data used for the comparison can be found in the `_perf_data` directory.

### Machine A: AMD Ryzen 5900x, RTX 3070.

| Library | Executor | Buffer to tensor | Forward pass | Postprocess (tensor 2 data) |
| ------- | -------- | ---------------- | ------------ | --------------------------- |
| Candle  | CPU      | 1.14 ms          | 298.64 ms    | 2.63 ms                     |
| ORT     | CPU      | 0.75 ms          | 80.91 ms     | 0.87 ms                     |
| Candle  | CUDA     | 0.09 ms          | 21.76 ms     | 3.39 ms                     |
| ORT     | CUDA     | 0.78 ms          | 5.53 ms      | 0.68 ms                     |

### Machine B: Intel 12700H, RTX A2000

| Library | Executor | Buffer to tensor | Forward pass | Postprocess (tensor 2 data) |
| ------- | -------- | ---------------- | ------------ | --------------------------- |
| Candle  | CPU      | 3.13 ms          | 589.98 ms    | 7.55 ms                     |
| ORT     | CPU      | 1.85 ms          | 86.67 ms     | 1.33 ms                     |
| Candle  | CUDA     | 0.16 ms          | 38.92 ms     | 6.22 ms                     |
| ORT     | CUDA     | 1.37 ms          | 10.06 ms     | 1.20 ms                     |