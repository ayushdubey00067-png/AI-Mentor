// lib/utils/app_theme.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const Color primary      = Color(0xFF1A2B5F);
  static const Color primaryLight = Color(0xFF2D4A9E);
  static const Color accent       = Color(0xFFD4A843);
  static const Color accentLight  = Color(0xFFF0C96B);
  static const Color surface      = Color(0xFFF8F7F2);
  static const Color surfaceDark  = Color(0xFFEEECE4);
  static const Color textPrimary  = Color(0xFF1A1A2E);
  static const Color textSecondary= Color(0xFF5A6070);
  static const Color userBubble   = Color(0xFF1A2B5F);
  static const Color aiBubble     = Color(0xFFFFFFFF);
  static const Color mentorBubble = Color(0xFF2D6A4F);
  static const Color success      = Color(0xFF2D6A4F);
  static const Color warning      = Color(0xFFE76F51);
  static const Color error        = Color(0xFFC1121F);

  static ThemeData get theme => ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.light(
      primary: primary, secondary: accent,
      surface: surface, error: error,
    ),
    textTheme: GoogleFonts.merriweatherTextTheme().copyWith(
      displayLarge: GoogleFonts.playfairDisplay(fontSize: 32, fontWeight: FontWeight.w700, color: primary),
      displayMedium: GoogleFonts.playfairDisplay(fontSize: 24, fontWeight: FontWeight.w600, color: primary),
      headlineMedium: GoogleFonts.playfairDisplay(fontSize: 20, fontWeight: FontWeight.w600, color: primary),
      titleLarge: GoogleFonts.playfairDisplay(fontSize: 18, fontWeight: FontWeight.w600, color: primary),
      bodyLarge: GoogleFonts.lato(fontSize: 15, color: textPrimary),
      bodyMedium: GoogleFonts.lato(fontSize: 14, color: textPrimary),
      bodySmall: GoogleFonts.lato(fontSize: 12, color: textSecondary),
      labelLarge: GoogleFonts.lato(fontSize: 14, fontWeight: FontWeight.w600, color: primary),
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: primary, foregroundColor: Colors.white, elevation: 0,
      titleTextStyle: GoogleFonts.playfairDisplay(fontSize: 20, fontWeight: FontWeight.w600, color: Colors.white),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: primary, foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        textStyle: GoogleFonts.lato(fontSize: 15, fontWeight: FontWeight.w600),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true, fillColor: Colors.white,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Color(0xFFDDD9CE))),
      enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: Color(0xFFDDD9CE))),
      focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: primary, width: 2)),
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
    ),
    cardTheme: CardThemeData(
      color: Colors.white, elevation: 2,
      shadowColor: primary.withAlpha(25),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    scaffoldBackgroundColor: surface,
  );
}