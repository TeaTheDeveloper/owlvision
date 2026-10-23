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
    _interpreter = await Interpreter.fromAsset('models/yolov8n.tflite', options: options);
    _inputShape = _interpreter!.getInputTensor(0).shape;
    if (_inputShape.length != 4 || _inputShape.last != 3) {
      throw StateError('The bundled object detection model has an unsupported input shape: $_inputShape');
    }
  }

  Future<List<DetectedObject>> detect(ui.Image frame, {double threshold = .40}) async {
    final interpreter = _interpreter;
    if (interpreter == null) throw StateError('The detection model is not loaded.');
    final byteData = await frame.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) throw StateError('Could not read the camera frame.');
    final decoded = img.decodeImage(byteData.buffer.asUint8List());
    if (decoded == null) throw StateError('Could not decode the camera frame.');

    final height = _inputShape[1], width = _inputShape[2];
    final resized = img.copyResize(decoded, width: width, height: height);
    final input = Uint8List(width * height * 3);
    var i = 0;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final p = resized.getPixel(x, y);
        input[i++] = p.r.toInt(); input[i++] = p.g.toInt(); input[i++] = p.b.toInt();
      }
    }
    final shapedInput = input.reshape(_inputShape);
    final tensors = List.generate(interpreter.getOutputTensors().length, (index) => interpreter.getOutputTensor(index));
    final outputs = <int, Object>{};
    for (var index = 0; index < tensors.length; index++) {
      outputs[index] = _zeroForShape(tensors[index].shape);
    }
    interpreter.runForMultipleInputs([shapedInput], outputs);

    // TFLite Model Maker EfficientDet convention: locations, classes, scores, count.
    int locationIndex = -1, classIndex = -1, scoreIndex = -1, countIndex = -1;
    for (var index = 0; index < tensors.length; index++) {
      final shape = tensors[index].shape;
      if (shape.length >= 2 && shape.last == 4) {
        locationIndex = index;
      } else if (shape.length <= 1 || (shape.length == 2 && shape.last == 1)) {
        countIndex = index;
      }
    }
    final candidates = <int>[];
    for (var index = 0; index < tensors.length; index++) {
      if (index == locationIndex || index == countIndex) continue;
      candidates.add(index);
    }
    if (candidates.length >= 2) { classIndex = candidates[0]; scoreIndex = candidates[1]; }
    if (locationIndex < 0 || classIndex < 0 || scoreIndex < 0) {
      throw StateError('Unsupported model output tensors: ${tensors.map((t) => t.shape).toList()}');
    }
    final locations = _flatten(outputs[locationIndex]!);
    final classes = _flatten(outputs[classIndex]!);
    final scores = _flatten(outputs[scoreIndex]!);
    final result = <DetectedObject>[];
    final total = [locations.length ~/ 4, classes.length, scores.length].reduce((a,b) => a < b ? a : b);
    for (var n = 0; n < total; n++) {
      final score = scores[n].toDouble();
      if (!score.isFinite || score < threshold) continue;
      final classId = classes[n].toDouble().round();
      // EfficientDet location values are [top, left, bottom, right], normalized.
      final top = locations[n * 4].toDouble().clamp(0.0, 1.0);
      final left = locations[n * 4 + 1].toDouble().clamp(0.0, 1.0);
      final bottom = locations[n * 4 + 2].toDouble().clamp(0.0, 1.0);
      final right = locations[n * 4 + 3].toDouble().clamp(0.0, 1.0);
      if (right <= left || bottom <= top) continue;
      final label = classId >= 0 && classId < _labels.length ? _labels[classId] : 'Object ${classId + 1}';
      result.add(DetectedObject(label: label.trim(), score: score,
        box: [left, top, right - left, bottom - top]));
    }
    result.sort((a,b) => b.score.compareTo(a.score));
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
