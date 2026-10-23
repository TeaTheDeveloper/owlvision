import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class DetectedObject {
  const DetectedObject({
    required this.label,
    required this.score,
    required this.box,
  });

  final String label;
  final double score;

  /// Normalized coordinates (0..1): left, top, width, height.
  final List<double> box;
}

class ObjectDetectionService {
  Interpreter? _interpreter;
  List<int> _inputShape = const [];

  bool get isLoaded => _interpreter != null;

  // Standard 80-class COCO labels, in YOLOv8 class-index order.
  static const List<String> _labels = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
  ];

  Future<void> loadModel() async {
    _interpreter?.close();
    _interpreter = null;

    final options = InterpreterOptions()..threads = 4;

    final interpreter = await Interpreter.fromAsset(
      'models/yolov8n.tflite',
      options: options,
    );

    final inputs = interpreter.getInputTensors();
    final outputs = interpreter.getOutputTensors();

    if (inputs.length != 1 ||
        inputs[0].shape.length != 4 ||
        inputs[0].shape[0] != 1 ||
        inputs[0].shape[1] != 3) {
      interpreter.close();
      throw StateError(
        'Expected YOLO input [1, 3, H, W]. '
        'Received ${inputs.map((t) => t.shape).toList()}',
      );
    }

    if (inputs[0].type != TensorType.float32) {
      final type = inputs[0].type;
      interpreter.close();
      throw StateError(
        'This preprocessing expects float32 input, but the model '
        'expects $type. Input quantization must be handled separately.',
      );
    }

    if (outputs.length != 1 ||
        outputs[0].shape.length != 3 ||
        outputs[0].shape[0] != 1 ||
        outputs[0].shape[1] != 84) {
      final shapes = outputs.map((t) => t.shape).toList();
      interpreter.close();
      throw StateError(
        'Expected YOLO output [1, 84, candidates]. '
        'Received $shapes',
      );
    }

    _interpreter = interpreter;
    _inputShape = inputs[0].shape;

    print('OwlVision YOLO input: ${inputs[0].shape}');
    print('OwlVision YOLO output: ${outputs[0].shape}');
  }

  Future<List<DetectedObject>> detect(
    ui.Image frame, {
    double threshold = 0.40,
    double iouThreshold = 0.45,
    int maxDetections = 100,
  }) async {
    final interpreter = _interpreter;

    if (interpreter == null) {
      throw StateError('The detection model is not loaded.');
    }

    if (threshold < 0 || threshold > 1) {
      throw ArgumentError.value(threshold, 'threshold', 'Must be 0..1.');
    }

    final byteData = await frame.toByteData(format: ui.ImageByteFormat.png);

    if (byteData == null) {
      throw StateError('Could not read the camera frame.');
    }

    final decoded = img.decodeImage(byteData.buffer.asUint8List());

    if (decoded == null) {
      throw StateError('Could not decode the camera frame.');
    }

    final height = _inputShape[2];
    final width = _inputShape[3];

    // Resize to the model's expected dimensions.
    final resized = img.copyResize(decoded, width: width, height: height);

    // YOLO NCHW input: [1, 3, H, W], RGB float32 normalized to 0..1.
    final planeSize = width * height;
    final input = Float32List(3 * planeSize);

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final pixel = resized.getPixel(x, y);
        final index = y * width + x;

        input[index] = pixel.r.toDouble() / 255.0;
        input[planeSize + index] = pixel.g.toDouble() / 255.0;
        input[2 * planeSize + index] = pixel.b.toDouble() / 255.0;
      }
    }

    final shapedInput = input.reshape(_inputShape);
    final outputTensor = interpreter.getOutputTensor(0);
    final outputShape = outputTensor.shape;

    final outputBuffer = _zeroForShape(outputShape);
    interpreter.run(shapedInput, outputBuffer);

    final output = _flatten(outputBuffer);
    const attributes = 84;
    final candidates = outputShape[2];

    final detections = <DetectedObject>[];

    for (var n = 0; n < candidates; n++) {
      final cx = output[n].toDouble();
      final cy = output[candidates + n].toDouble();
      final boxWidth = output[2 * candidates + n].toDouble();
      final boxHeight = output[3 * candidates + n].toDouble();

      if (!cx.isFinite ||
          !cy.isFinite ||
          !boxWidth.isFinite ||
          !boxHeight.isFinite ||
          boxWidth <= 0 ||
          boxHeight <= 0) {
        continue;
      }

      var bestClass = -1;
      var bestScore = 0.0;

      for (var c = 4; c < attributes; c++) {
        final score = output[c * candidates + n].toDouble();

        if (score.isFinite && score > bestScore) {
          bestScore = score;
          bestClass = c - 4;
        }
      }

      if (bestScore < threshold || bestClass < 0) {
        continue;
      }

      // Convert model pixel coordinates to normalized frame coordinates.
      final left = ((cx - boxWidth / 2) / width).clamp(0.0, 1.0).toDouble();
      final top = ((cy - boxHeight / 2) / height).clamp(0.0, 1.0).toDouble();
      final right = ((cx + boxWidth / 2) / width).clamp(0.0, 1.0).toDouble();
      final bottom = ((cy + boxHeight / 2) / height).clamp(0.0, 1.0).toDouble();

      if (right <= left || bottom <= top) {
        continue;
      }

      detections.add(
        DetectedObject(
          label: _labels[bestClass],
          score: bestScore,
          box: [left, top, right - left, bottom - top],
        ),
      );
    }

    // Suppress duplicate boxes for the same class.
    return _nonMaximumSuppression(
      detections,
      iouThreshold: iouThreshold,
      maxDetections: maxDetections,
    );
  }

  List<DetectedObject> _nonMaximumSuppression(
    List<DetectedObject> detections, {
    required double iouThreshold,
    required int maxDetections,
  }) {
    detections.sort((a, b) => b.score.compareTo(a.score));

    final kept = <DetectedObject>[];

    for (final candidate in detections) {
      var suppress = false;

      for (final accepted in kept) {
        if (candidate.label != accepted.label) {
          continue;
        }

        if (_intersectionOverUnion(candidate.box, accepted.box) >
            iouThreshold) {
          suppress = true;
          break;
        }
      }

      if (!suppress) {
        kept.add(candidate);

        if (kept.length >= maxDetections) {
          break;
        }
      }
    }

    return kept;
  }

  double _intersectionOverUnion(List<double> a, List<double> b) {
    final left = a[0] > b[0] ? a[0] : b[0];
    final top = a[1] > b[1] ? a[1] : b[1];

    final rightA = a[0] + a[2];
    final rightB = b[0] + b[2];
    final bottomA = a[1] + a[3];
    final bottomB = b[1] + b[3];

    final right = rightA < rightB ? rightA : rightB;
    final bottom = bottomA < bottomB ? bottomA : bottomB;

    final intersectionWidth = (right - left).clamp(0.0, 1.0);
    final intersectionHeight = (bottom - top).clamp(0.0, 1.0);
    final intersection = intersectionWidth * intersectionHeight;

    final areaA = a[2] * a[3];
    final areaB = b[2] * b[3];
    final union = areaA + areaB - intersection;

    return union <= 0 ? 0.0 : intersection / union;
  }

  Object _zeroForShape(List<int> shape) {
    if (shape.isEmpty) return 0.0;

    return List.generate(
      shape.first,
      (_) => _zeroForShape(shape.skip(1).toList()),
    );
  }

  List<num> _flatten(Object value) {
    final result = <num>[];

    void visit(dynamic item) {
      if (item is List) {
        for (final child in item) {
          visit(child);
        }
      } else if (item is num) {
        result.add(item);
      }
    }

    visit(value);
    return result;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _inputShape = const [];
  }
}