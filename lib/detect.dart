import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'services/tensor.dart';

class ObjectDetectionScreen extends StatefulWidget {
  const ObjectDetectionScreen({super.key});
  @override
  State<ObjectDetectionScreen> createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> with WidgetsBindingObserver {
  final _detector = ObjectDetectionService();
  CameraController? _camera;
  List<CameraDescription> _cameras = [];
  List<DetectedObject> _objects = [];
  bool _busy = false, _running = true, _modelReady = false, _switching = false;
  String? _error;
  double _threshold = .40;
  DateTime? _lastScan;
  Timer? _timer;

  @override
  void initState() { super.initState(); WidgetsBinding.instance.addObserver(this); _initialize(); }

  Future<void> _initialize() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) throw StateError('No camera was found on this device.');
      await _detector.loadModel();
      _modelReady = true;
      await _startCamera(_preferredCamera());
      _timer = Timer.periodic(const Duration(milliseconds: 900), (_) => _scan());
      if (mounted) setState(() {});
    } catch (e) { if (mounted) setState(() => _error = _friendlyError(e)); }
  }

  CameraDescription _preferredCamera() => _cameras.firstWhere(
    (c) => c.lensDirection == CameraLensDirection.back, orElse: () => _cameras.first);

  Future<void> _startCamera(CameraDescription description) async {
    final old = _camera; _camera = null;
    if (old != null) await old.dispose();
    final controller = CameraController(description, ResolutionPreset.medium, enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.jpeg : ImageFormatGroup.bgra8888);
    _camera = controller;
    await controller.initialize();
    if (!mounted) { await controller.dispose(); return; }
    setState(() { _error = null; });
  }

  Future<void> _scan() async {
    final camera = _camera;
    if (!_running || _busy || !_modelReady || camera == null || !camera.value.isInitialized || camera.value.isTakingPicture) return;
    _busy = true;
    try {
      final file = await camera.takePicture();
      final bytes = await File(file.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return;
      final completer = Completer<ui.Image>();
      ui.decodeImageFromList(bytes, (image) => completer.complete(image));
      final frame = await completer.future;
      final found = await _detector.detect(frame, threshold: _threshold);
      frame.dispose();
      if (mounted) setState(() { _objects = found; _lastScan = DateTime.now(); });
      try { await File(file.path).delete(); } catch (_) {}
    } catch (e) {
      if (mounted && _error == null) setState(() => _error = 'Detection paused: ${_friendlyError(e)}');
    } finally { _busy = false; }
  }

  String _friendlyError(Object error) {
    final message = error.toString().replaceFirst('Exception: ', '').replaceFirst('Bad state: ', '');
    return message.length > 180 ? '${message.substring(0, 177)}…' : message;
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2 || _switching) return;
    _switching = true;
    try {
      final current = _camera?.description;
      final next = _cameras.firstWhere((c) => c.lensDirection != current?.lensDirection, orElse: () => _cameras.first);
      await _startCamera(next);
    } catch (e) { if (mounted) setState(() => _error = _friendlyError(e)); }
    finally { _switching = false; }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _timer?.cancel(); _timer = null; _camera?.dispose(); _camera = null;
    } else if (state == AppLifecycleState.resumed && _cameras.isNotEmpty && _camera == null) {
      _startCamera(_preferredCamera()).then((_) { if (mounted) _timer = Timer.periodic(const Duration(milliseconds: 900), (_) => _scan()); })
        .catchError((Object e) { if (mounted) setState(() => _error = _friendlyError(e)); });
    }
  }

  @override
  void dispose() { WidgetsBinding.instance.removeObserver(this); _timer?.cancel(); _camera?.dispose(); _detector.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    return Scaffold(
      appBar: AppBar(title: const Row(children: [Icon(Icons.center_focus_strong_rounded, color: Color(0xFF8BCEB5)), SizedBox(width: 10), Text('Owl Vision', style: TextStyle(fontWeight: FontWeight.w700))]),
        actions: [IconButton(tooltip: 'Switch camera', onPressed: _cameras.length > 1 ? _switchCamera : null, icon: const Icon(Icons.cameraswitch_rounded))]),
      body: SafeArea(child: Column(children: [
        Padding(padding: const EdgeInsets.fromLTRB(20, 4, 20, 16), child: Row(children: [
          Container(width: 8, height: 8, decoration: BoxDecoration(color: _error == null ? const Color(0xFF8BCEB5) : Colors.orange, shape: BoxShape.circle)),
          const SizedBox(width: 8), Expanded(child: Text(_error ?? (_modelReady ? 'LIVE OBJECT DETECTION' : 'PREPARING DETECTOR'), style: const TextStyle(fontSize: 11, letterSpacing: 1.5, fontWeight: FontWeight.w700, color: Colors.white70))),
          Text('${_objects.length} found', style: const TextStyle(color: Color(0xFF8BCEB5), fontWeight: FontWeight.w600)),
        ])),
        Expanded(child: Padding(padding: const EdgeInsets.symmetric(horizontal: 14), child: ClipRRect(borderRadius: BorderRadius.circular(24), child: Stack(fit: StackFit.expand, children: [
          if (camera != null && camera.value.isInitialized) CameraPreview(camera) else Container(color: const Color(0xFF1A2420), child: const Center(child: CircularProgressIndicator())),
          if (camera != null && camera.value.isInitialized) CustomPaint(painter: DetectionPainter(_objects)),
          if (_error != null) Align(alignment: Alignment.topCenter, child: Container(margin: const EdgeInsets.all(12), padding: const EdgeInsets.all(12), decoration: BoxDecoration(color: Colors.black.withOpacity(.78), borderRadius: BorderRadius.circular(14)), child: Text(_error!, style: const TextStyle(color: Colors.white)))),
          Positioned(left: 14, top: 14, child: _glassTag(Icons.radar_rounded, _running ? 'SCANNING' : 'PAUSED')),
          Positioned(right: 14, bottom: 14, child: _glassTag(Icons.memory_rounded, 'ON-DEVICE AI')),
        ])))),
        Padding(padding: const EdgeInsets.fromLTRB(20, 18, 20, 18), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [const Expanded(child: Text('Detection confidence', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600))), Text('${(_threshold * 100).round()}%', style: const TextStyle(color: Color(0xFF8BCEB5), fontWeight: FontWeight.w700))]),
          Slider(value: _threshold, min: .2, max: .8, divisions: 12, activeColor: const Color(0xFF8BCEB5), onChanged: (v) => setState(() => _threshold = v)),
          const SizedBox(height: 6),
          Row(children: [Expanded(child: OutlinedButton.icon(onPressed: () => setState(() { _running = !_running; if (_running) _error = null; }), icon: Icon(_running ? Icons.pause_rounded : Icons.play_arrow_rounded), label: Text(_running ? 'Pause scan' : 'Resume scan'), style: OutlinedButton.styleFrom(minimumSize: const Size(0, 50), foregroundColor: Colors.white, side: const BorderSide(color: Colors.white24)))), const SizedBox(width: 10),
            Expanded(child: FilledButton.icon(onPressed: _objects.isEmpty ? null : () => _showResults(context), icon: const Icon(Icons.list_alt_rounded), label: const Text('View results'), style: FilledButton.styleFrom(minimumSize: const Size(0, 50), backgroundColor: const Color(0xFF8BCEB5), foregroundColor: const Color(0xFF102019))))]),
          const SizedBox(height: 12), const Text('Objects are processed on your device. Results depend on lighting, camera angle and the model’s training labels.', style: TextStyle(color: Colors.white54, fontSize: 12, height: 1.4)),
        ])),
      ])),
    );
  }

  Widget _glassTag(IconData icon, String text) => Container(padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 8), decoration: BoxDecoration(color: Colors.black.withOpacity(.62), borderRadius: BorderRadius.circular(30)), child: Row(mainAxisSize: MainAxisSize.min, children: [Icon(icon, size: 15, color: const Color(0xFF8BCEB5)), const SizedBox(width: 6), Text(text, style: const TextStyle(fontSize: 10, letterSpacing: 1.2, fontWeight: FontWeight.w700, color: Colors.white))]));

  void _showResults(BuildContext context) => showModalBottomSheet<void>(context: context, backgroundColor: const Color(0xFF1A2420), isScrollControlled: true, builder: (context) => SafeArea(child: Padding(padding: const EdgeInsets.all(20), child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Detected objects', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700)), const SizedBox(height: 14), ..._objects.take(12).map((o) => ListTile(contentPadding: EdgeInsets.zero, leading: const CircleAvatar(backgroundColor: Color(0xFF293D34), child: Icon(Icons.visibility_rounded, color: Color(0xFF8BCEB5))), title: Text(o.label, style: const TextStyle(fontWeight: FontWeight.w600)), trailing: Text('${(o.score * 100).round()}%', style: const TextStyle(color: Color(0xFF8BCEB5), fontWeight: FontWeight.w700)))), if (_objects.isEmpty) const Text('No objects above the confidence threshold.')]))));
}

class DetectionPainter extends CustomPainter {
  DetectionPainter(this.objects);
  final List<DetectedObject> objects;
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = const Color(0xFF8BCEB5)..style = PaintingStyle.stroke..strokeWidth = 2.2;
    final fill = Paint()..color = const Color(0xFF8BCEB5).withOpacity(.18)..style = PaintingStyle.fill;
    for (final object in objects) {
      final b = object.box;
      final rect = Rect.fromLTWH(b[0] * size.width, b[1] * size.height, b[2] * size.width, b[3] * size.height);
      canvas.drawRect(rect, fill); canvas.drawRect(rect, paint);
      final label = '${object.label} ${(object.score * 100).round()}%';
      final text = TextPainter(text: TextSpan(text: label, style: const TextStyle(color: Color(0xFF102019), fontSize: 12, fontWeight: FontWeight.w700)), textDirection: TextDirection.ltr)..layout(maxWidth: size.width * .7);
      final tag = Rect.fromLTWH(rect.left.clamp(0, size.width - text.width - 12), (rect.top - text.height - 10).clamp(0, size.height - text.height - 10), text.width + 12, text.height + 8);
      canvas.drawRRect(RRect.fromRectAndRadius(tag, const Radius.circular(5)), paint..style = PaintingStyle.fill);
      text.paint(canvas, Offset(tag.left + 6, tag.top + 4)); paint.style = PaintingStyle.stroke;
    }
  }
  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) => oldDelegate.objects != objects;
}
