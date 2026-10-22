import 'package:flutter/material.dart';
import 'detect.dart';

void main() => runApp(const ObjectOwlApp());

class ObjectOwlApp extends StatelessWidget {
  const ObjectOwlApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF8BCEB5);
    return MaterialApp(
      title: 'Object Owl',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.dark,
          surface: const Color(0xFF101714)),
        scaffoldBackgroundColor: const Color(0xFF101714),
        appBarTheme: const AppBarTheme(backgroundColor: Color(0xFF101714), foregroundColor: Colors.white),
        cardTheme: CardThemeData(color: const Color(0xFF1A2420), elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20))),
      ),
      home: const ObjectDetectionScreen(),
    );
  }
}
