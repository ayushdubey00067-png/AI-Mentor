// lib/screens/auth_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../services/auth_provider.dart';
import '../utils/app_theme.dart';
import '../utils/constants.dart';
import 'student/student_home_screen.dart';
import 'mentor/mentor_dashboard_screen.dart';
import 'admin/admin_dashboard_screen.dart';

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});
  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  final _loginEmailCtrl = TextEditingController();
  final _loginPassCtrl = TextEditingController();
  final _loginKey = GlobalKey<FormState>();
  bool _hidePassword = true;
  String? _errorMsg;

  @override
  void dispose() {
    _loginEmailCtrl.dispose();
    _loginPassCtrl.dispose();
    super.dispose();
  }

  void _goHome(AuthProvider auth) {
    if (auth.isAdmin) {
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => const AdminDashboardScreen(),
      ));
    } else if (auth.isMentor) {
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => const MentorDashboardScreen(),
      ));
    } else {
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => const StudentHomeScreen(),
      ));
    }
  }

  Future<void> _login() async {
    if (!(_loginKey.currentState?.validate() ?? false)) return;
    setState(() => _errorMsg = null);
    final auth = context.read<AuthProvider>();
    final error = await auth.login(_loginEmailCtrl.text.trim(), _loginPassCtrl.text.trim());
    if (!mounted) return;
    if (error == null) {
      _goHome(auth);
      return;
    }
    if (error == 'invalid_credentials') {
      setState(() => _errorMsg = 'Invalid email/phone or password. Please verify your institutional login credentials.');
    } else {
      setState(() => _errorMsg = error.contains('RLS')
          ? '⚠️ Database permission error. Please verify Supabase access.'
          : error.replaceAll('Exception: ', ''));
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final size = MediaQuery.of(context).size;

    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0A1128),
              Color(0xFF0F1C3F),
              Color(0xFF1E293B),
            ],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 460),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // ── Brand Header ────────────────────────────
                    _buildBrandHeader(),
                    const SizedBox(height: 28),

                    // ── Login Card ──────────────────────────────
                    _buildLoginCard(auth),
                    const SizedBox(height: 24),

                    // ── Institutional Footer ────────────────────
                    _buildInstitutionalFooter(),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBrandHeader() {
    return Column(
      children: [
        Container(
          width: 72,
          height: 72,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: AppTheme.accentGold,
            boxShadow: [
              BoxShadow(
                color: AppTheme.accentGold.withOpacity(0.35),
                blurRadius: 24,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: const Icon(
            Icons.school_rounded,
            color: AppTheme.primaryNavy,
            size: 38,
          ),
        ),
        const SizedBox(height: 16),
        Text(
          kAppName,
          style: GoogleFonts.outfit(
            fontSize: 32,
            fontWeight: FontWeight.bold,
            color: Colors.white,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          'Institutional Academic Concierge & EMS',
          style: GoogleFonts.inter(
            fontSize: 13,
            color: Colors.white70,
            letterSpacing: 0.2,
          ),
        ),
      ],
    );
  }

  Widget _buildLoginCard(AuthProvider auth) {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 40,
            offset: const Offset(0, 16),
          ),
        ],
      ),
      child: Form(
        key: _loginKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: AppTheme.primaryNavy.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.lock_person_rounded,
                    color: AppTheme.primaryNavy,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Portal Sign In',
                      style: GoogleFonts.outfit(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.primaryNavy,
                      ),
                    ),
                    Text(
                      'Admin • Faculty Mentor • Student',
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const Divider(height: 28),

            if (_errorMsg != null) ...[
              Container(
                padding: const EdgeInsets.all(12),
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red, size: 20),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        _errorMsg!,
                        style: GoogleFonts.inter(color: Colors.red.shade900, fontSize: 12, height: 1.4),
                      ),
                    ),
                    GestureDetector(
                      onTap: () => setState(() => _errorMsg = null),
                      child: const Icon(Icons.close, color: Colors.red, size: 16),
                    ),
                  ],
                ),
              ),
            ],

            // Email or Phone Field
            Text(
              'Official Email / User ID',
              style: GoogleFonts.inter(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: Colors.black87,
              ),
            ),
            const SizedBox(height: 6),
            TextFormField(
              controller: _loginEmailCtrl,
              keyboardType: TextInputType.emailAddress,
              textInputAction: TextInputAction.next,
              validator: (v) => (v == null || v.trim().isEmpty) ? 'Please enter your registered email' : null,
              decoration: InputDecoration(
                hintText: 'e.g. gunjan@mru.edu.in or student@mru.ac.in',
                prefixIcon: const Icon(Icons.email_outlined, size: 20),
                isDense: true,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
              ),
            ),
            const SizedBox(height: 18),

            // Password Field
            Text(
              'Password / Mobile Number',
              style: GoogleFonts.inter(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: Colors.black87,
              ),
            ),
            const SizedBox(height: 6),
            TextFormField(
              controller: _loginPassCtrl,
              obscureText: _hidePassword,
              textInputAction: TextInputAction.done,
              onFieldSubmitted: (_) => _login(),
              validator: (v) => (v == null || v.trim().isEmpty) ? 'Please enter your password' : null,
              decoration: InputDecoration(
                hintText: '••••••••',
                prefixIcon: const Icon(Icons.lock_outline, size: 20),
                suffixIcon: IconButton(
                  icon: Icon(
                    _hidePassword ? Icons.visibility_off_outlined : Icons.visibility_outlined,
                    size: 20,
                  ),
                  onPressed: () => setState(() => _hidePassword = !_hidePassword),
                ),
                isDense: true,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
              ),
            ),
            const SizedBox(height: 26),

            // Submit Button
            ElevatedButton(
              onPressed: auth.isLoading ? null : _login,
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.primaryNavy,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                elevation: 2,
              ),
              child: auth.isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : Text(
                      'Sign In to Dashboard',
                      style: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.bold),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInstitutionalFooter() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.06),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.12)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.shield_outlined, color: AppTheme.accentGold, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'Institutional Access Only: Accounts and classes are provisioned directly by College Administration. For account assistance or credentials, contact your department office.',
              style: GoogleFonts.inter(fontSize: 11, color: Colors.white70, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }
}
