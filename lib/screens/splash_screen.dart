// lib/screens/splash_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../services/auth_provider.dart';
import '../utils/app_theme.dart';
import 'auth_screen.dart';
import 'student/student_home_screen.dart';
import 'mentor/mentor_dashboard_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _fade, _scale;
  bool _navigated = false;

  @override
  void initState() {
    super.initState();
    _ctrl  = AnimationController(duration: const Duration(milliseconds: 1000), vsync: this);
    _fade  = Tween<double>(begin: 0, end: 1).animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeIn));
    _scale = Tween<double>(begin: 0.75, end: 1).animate(CurvedAnimation(parent: _ctrl, curve: Curves.elasticOut));
    _ctrl.forward();
  }

  void _navigate(AuthProvider auth) {
    if (_navigated) return;
    _navigated = true;
    Future.delayed(const Duration(milliseconds: 1500), () {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => auth.isLoggedIn
            ? (auth.isMentor ? const MentorDashboardScreen() : const StudentHomeScreen())
            : const AuthScreen(),
      ));
    });
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [Color(0xFF0D1B3E), AppTheme.primary, Color(0xFF2D4A9E)],
          ),
        ),
        child: Consumer<AuthProvider>(
          builder: (_, auth, __) {
            if (auth.sessionChecked) _navigate(auth);
            return Center(
              child: FadeTransition(opacity: _fade,
                child: ScaleTransition(scale: _scale,
                  child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                    Container(
                      width: 110, height: 110,
                      decoration: BoxDecoration(
                        color: AppTheme.accent, shape: BoxShape.circle,
                        boxShadow: [BoxShadow(color: AppTheme.accent.withOpacity(0.5),
                            blurRadius: 30, offset: const Offset(0, 12))],
                      ),
                      child: const Icon(Icons.smart_toy_rounded, color: Colors.white, size: 58),
                    ),
                    const SizedBox(height: 28),
                    Text('AI ChatBot',
                        style: GoogleFonts.playfairDisplay(
                            fontSize: 36, fontWeight: FontWeight.w700, color: Colors.white)),
                    Text('Your Academic Guide, Always Here',
                        style: GoogleFonts.lato(
                            fontSize: 15, color: AppTheme.accentLight, letterSpacing: 1)),
                    const SizedBox(height: 60),
                    const SizedBox(width: 32, height: 32,
                      child: CircularProgressIndicator(color: AppTheme.accent, strokeWidth: 2.5)),
                  ]),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}