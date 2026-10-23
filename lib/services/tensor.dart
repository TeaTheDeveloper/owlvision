import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class DetectedObject {
  const DetectedObject({required this.label, required this.score, required this.box});
  final String label;
  final double score;
  /// Normalized coordinates (0..1), in left, top, width, height order.
  final List<double> box;
}

class ObjectDetectionService {
  Interpreter? _interpreter;
  List<int> _inputShape = const [];
  bool get isLoaded => _interpreter != null;

  static const _labels = <String>[
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','street sign','stop sign','parking meter','bench','bird','cat','dog','horse','sheep',
    'cow','elephant','bear','zebra','giraffe','hat','backpack','umbrella','shoe','eye glasses','handbag',
    'tie','suitcase',' frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','plate','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'mirror','dining table','window','desk','toilet','door','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','blender','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
  ];

  Future<void> loadModel() async {
    final options = InterpreterOptions()..threads = 4;

    _interpreter = await Interpreter.fromAsset(
      'models/yolov8n.tflite',
      options: options,
    );

    final inputs = _interpreter!.getInputTensors();
    final outputs = _interpreter!.getOutputTensors();

    print('=== OWLVISION YOLO MODEL ===');

    for (var i = 0; i < inputs.length; i++) {
      print(
        'INPUT $i: shape=${inputs[i].shape}, '
        'type=${inputs[i].type}',
      );
    }

    for (var i = 0; i < outputs.length; i++) {
      print(
        'OUTPUT $i: shape=${outputs[i].shape}, '
        'type=${outputs[i].type}',
      );
    }

    _inputShape = inputs[0].shape;
  }

  Future<List<DetectedObject>> detect(ui.Image frame, {double threshold = .40}) async {
    final interpreter = _interpreter;
    if (interpreter == null) throw StateError('The detection model is not loaded.');
    final byteData = await frame.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) throw StateError('Could not read the camera frame.');
    final decoded = img.decodeImage(byteData.buffer.asUint8List());
    if (decoded == null) throw StateError('Could not decode the camera frame.');

    final shape = _inputShape;

    // YOLOv8 NCHW: [1, 3, height, width]
    if (shape.length != 4 || shape[0] != 1 || shape[1] != 3) {
      throw StateError('Expected YOLO NCHW input [1, 3, H, W], got $shape');
    }

    final height = shape[2];
    final width = shape[3];

    final resized = img.copyResize(decoded, width: width, height: height);

    // Planar RGB data: all red values, then green, then blue.
    final input = Float32List(3 * height * width);
    final planeSize = height * width;

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final p = resized.getPixel(x, y);
        final pixelIndex = y * width + x;

        input[pixelIndex] = p.r / 255.0;
        input[planeSize + pixelIndex] = p.g / 255.0;
        input[2 * planeSize + pixelIndex] = p.b / 255.0;
      }
    }

    final shapedInput = input.reshape(shape);
    final tensors = List.generate(interpreter.getOutputTensors().length, (index) => interpreter.getOutputTensor(index));
    final outputs = <int, Object>{};
    for (var index = 0; index < tensors.length; index++) {
      outputs[index] = _zeroForShape(tensors[index].shape);
    }
    interpreter.runForMultipleInputs([shapedInput], outputs);

    // YOLOv8 output: [1, 84, 8400]
    final outputShape = tensors[0].shape;

    if (outputShape.length != 3 ||
        outputShape[0] != 1 ||
        outputShape[1] != 84) {
      throw StateError('Unsupported YOLO output shape: $outputShape');
    }

    final output = _flatten(outputs[0]!);
    const attributes = 84;
    final candidates = outputShape[2];

    final result = <DetectedObject>[];

    for (var n = 0; n < candidates; n++) {
      // YOLOv8 layout: [1, attributes, candidates].
      final cx = output[0 * candidates + n].toDouble();
      final cy = output[1 * candidates + n].toDouble();
      final w = output[2 * candidates + n].toDouble();
      final h = output[3 * candidates + n].toDouble();

      var bestClass = -1;
      var bestScore = 0.0;

      for (var c = 4; c < attributes; c++) {
        final score = output[c * candidates + n].toDouble();

        if (score > bestScore) {
          bestScore = score;
          bestClass = c - 4;
        }
      }

      if (!bestScore.isFinite || bestScore < threshold || bestClass < 0) {
        continue;
      }

      // Standard YOLOv8 outputs use center-x, center-y, width, height
      // in input-image pixel coordinates.
      final left = ((cx - w / 2) / _inputShape[3]).clamp(0.0, 1.0);
      final top = ((cy - h / 2) / _inputShape[2]).clamp(0.0, 1.0);
      final right = ((cx + w / 2) / _inputShape[3]).clamp(0.0, 1.0);
      final bottom = ((cy + h / 2) / _inputShape[2]).clamp(0.0, 1.0);

      if (right <= left || bottom <= top) continue;

      final label = bestClass < _labels.length
          ? _labels[bestClass].trim()
          : 'Object ${bestClass + 1}';

      result.add(
        DetectedObject(
          label: label,
          score: bestScore,
          box: [left, top, right - left, bottom - top],
        ),
      );
    }

    result.sort((a, b) => b.score.compareTo(a.score));
    return result;
  }

  Object _zeroForShape(List<int> shape) {
    if (shape.isEmpty) return 0.0;
    return List.generate(shape.first, (_) => _zeroForShape(shape.skip(1).toList()));
  }

  List<num> _flatten(Object value) {
    final result = <num>[];
    void visit(dynamic item) {
      if (item is List) { for (final child in item) { visit(child); } }
      else if (item is num) {
        result.add(item);
      }
    }
    visit(value);
    return result;
  }

  void dispose() { _interpreter?.close(); _interpreter = null; }
}
