# Owl Vision

Owl Vision is a Flutter mobile app for real-time object detection using the device camera and a bundled TensorFlow Lite EfficientDet-Lite2 model. Inference runs locally on the device; no image upload or account is required.

## What is included

- Live camera preview with bounding boxes, class labels and confidence scores
- Adjustable confidence threshold and pause/resume controls
- Front/rear camera switching where the device supports it
- On-device inference using `models/object.tflite`
- Clear loading and camera/model error states

## Requirements

- Flutter SDK compatible with Dart `^3.5.3`
- Android Studio/Xcode for the target platform
- A physical device with a camera is recommended; simulators may not provide a usable camera or delegate support

## Run

```bash
flutter pub get
flutter run
```

To run tests:

```bash
flutter test
```

To build an Android APK:

```bash
flutter build apk --release
```

## How it works

The app captures a camera still at a throttled interval, resizes it to the model's 448×448 RGB input, runs inference with TensorFlow Lite, and draws returned normalized bounding boxes over the preview. The separate `models/model.tflite` image-classification model is retained in the repository but is not used by the live detector.

## Important model note

Object labels are mapped using the common COCO label set. Model class indexing can vary by model export; if labels look consistently offset on your device, confirm the label-map embedded in the exact `object.tflite` export and update `_labels` in `lib/services/tensor.dart`. Detection quality depends on lighting, camera position, and the model's supported categories.

## Permissions and privacy

The app requests camera access only. Camera frames are processed locally and temporary captures are deleted after inference when possible. No audio, location, storage, or notification permission is needed for the current feature set.
