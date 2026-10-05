import 'dart:math' as math;
import 'package:flutter/material.dart';

/// A fluid, animated 3-dot typing indicator with vertical bouncing and opacity wave.
class TypingDotsIndicator extends StatefulWidget {
  final Color color;
  final double dotSize;
  final double spacing;
  final double bounceHeight;

  const TypingDotsIndicator({
    super.key,
    this.color = const Color(0xFF0F1C3F),
    this.dotSize = 7.0,
    this.spacing = 5.0,
    this.bounceHeight = 5.0,
  });

  @override
  State<TypingDotsIndicator> createState() => _TypingDotsIndicatorState();
}

class _TypingDotsIndicatorState extends State<TypingDotsIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(3, (index) {
            // Stagger each dot by phase
            final delay = index * 0.22;
            double progress = (_controller.value - delay);
            if (progress < 0) progress += 1.0;

            // Sine wave calculation for smooth vertical bounce
            final wave = math.sin(progress * math.pi * 2);
            // Only lift upwards (negative Y offset)
            final offsetY = wave > 0 ? -wave * widget.bounceHeight : 0.0;
            // Opacity pulses along with the height
            final opacity = 0.35 + (0.65 * (wave > 0 ? wave : 0.0));

            return Padding(
              padding: EdgeInsets.symmetric(horizontal: widget.spacing / 2),
              child: Transform.translate(
                offset: Offset(0, offsetY),
                child: Container(
                  width: widget.dotSize,
                  height: widget.dotSize,
                  decoration: BoxDecoration(
                    color: widget.color.withOpacity(opacity.clamp(0.2, 1.0)),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            );
          }),
        );
      },
    );
  }
}
