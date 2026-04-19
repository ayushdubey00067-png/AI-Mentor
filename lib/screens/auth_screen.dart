// lib/screens/auth_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../services/auth_provider.dart';
import '../utils/app_theme.dart';
import 'student/student_home_screen.dart';
import 'mentor/mentor_dashboard_screen.dart';

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});
  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tab;
  bool   _isLogin  = true;
  String _role     = 'student';
  String? _errorMsg;

  // Login
  final _loginEmailCtrl = TextEditingController();
  final _loginPassCtrl  = TextEditingController();
  final _loginKey       = GlobalKey<FormState>();
  bool  _hideLogin      = true;

  // Register
  final _rNameCtrl       = TextEditingController();
  final _rEmailCtrl      = TextEditingController();
  final _rPassCtrl       = TextEditingController();
  final _rConfCtrl       = TextEditingController();
  final _rProgCtrl       = TextEditingController();
  final _rBranchCtrl     = TextEditingController();
  final _rSemCtrl        = TextEditingController();
  final _rRollCtrl       = TextEditingController(); // ← student roll no
  final _rMentorCtrl     = TextEditingController(); // ← mentor email
  final _regKey          = GlobalKey<FormState>();
  bool  _hideReg         = true;
  bool  _hideConf        = true;

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
    _tab.addListener(() {
      if (_tab.indexIsChanging) return;
      setState(() { _isLogin = _tab.index == 0; _errorMsg = null; });
    });
  }

  @override
  void dispose() {
    _tab.dispose();
    for (final c in [_loginEmailCtrl, _loginPassCtrl, _rNameCtrl, _rEmailCtrl,
      _rPassCtrl, _rConfCtrl, _rProgCtrl, _rBranchCtrl, _rSemCtrl, _rRollCtrl, _rMentorCtrl])
      c.dispose();
    super.dispose();
  }

  void _goHome(AuthProvider auth) {
    Navigator.of(context).pushReplacement(MaterialPageRoute(
      builder: (_) => auth.isMentor
          ? const MentorDashboardScreen()
          : const StudentHomeScreen(),
    ));
  }

  Future<void> _login() async {
    if (!(_loginKey.currentState?.validate() ?? false)) return;
    setState(() => _errorMsg = null);
    final auth  = context.read<AuthProvider>();
    final error = await auth.login(_loginEmailCtrl.text, _loginPassCtrl.text);
    if (!mounted) return;
    if (error == null) { _goHome(auth); return; }
    if (error == 'invalid_credentials') {
      _rEmailCtrl.text = _loginEmailCtrl.text.trim();
      _showNotFoundSheet();
    } else {
      setState(() => _errorMsg = error.contains('RLS')
          ? '⚠️ DB permission error. Run supabase_schema.sql in Supabase SQL Editor.'
          : error.replaceAll('Exception: ', ''));
    }
  }

  Future<void> _register() async {
    if (!(_regKey.currentState?.validate() ?? false)) return;
    if (_rPassCtrl.text.trim() != _rConfCtrl.text.trim()) {
      setState(() => _errorMsg = 'Passwords do not match'); return;
    }
    setState(() => _errorMsg = null);
    final auth  = context.read<AuthProvider>();
    final error = await auth.register(
      email: _rEmailCtrl.text, password: _rPassCtrl.text,
      name: _rNameCtrl.text, role: _role,
      program: _rProgCtrl.text, branch: _rBranchCtrl.text, semester: _rSemCtrl.text,
      rollNumber: _rRollCtrl.text,
      mentorEmail: _role == 'student' ? _rMentorCtrl.text : null,
    );
    if (!mounted) return;
    if (error == null) { _goHome(auth); return; }
    if (error.contains('duplicate_email') || error.contains('already')) {
      _loginEmailCtrl.text = _rEmailCtrl.text.trim();
      setState(() => _errorMsg = 'Email already registered. Please Sign In.');
      _tab.animateTo(0);
    } else if (error.contains('invalid_mentor')) {
      setState(() => _errorMsg = error
          .replaceAll('Exception: ', '')
          .replaceAll('invalid_mentor: ', ''));
    } else {
      setState(() => _errorMsg = error
          .replaceAll('Exception: ', '')
          .replaceAll('duplicate_email: ', ''));
    }
  }

  void _showNotFoundSheet() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      builder: (ctx) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Container(width: 40, height: 4,
              decoration: BoxDecoration(color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(4))),
          const SizedBox(height: 20),
          const Icon(Icons.person_search, size: 42, color: AppTheme.primary),
          const SizedBox(height: 12),
          Text('No Account Found',
              style: GoogleFonts.playfairDisplay(fontSize: 20,
                  fontWeight: FontWeight.w700, color: AppTheme.primary)),
          const SizedBox(height: 8),
          Text('No account for\n${_loginEmailCtrl.text.trim()}\n\nWould you like to register?',
              textAlign: TextAlign.center,
              style: GoogleFonts.lato(fontSize: 14,
                  color: AppTheme.textSecondary, height: 1.6)),
          const SizedBox(height: 24),
          SizedBox(width: double.infinity,
            child: ElevatedButton.icon(
              icon: const Icon(Icons.app_registration),
              label: const Text('Register Now'),
              onPressed: () { Navigator.pop(ctx); _tab.animateTo(1); },
            ),
          ),
          TextButton(onPressed: () => Navigator.pop(ctx),
              child: Text('Try again',
                  style: GoogleFonts.lato(color: AppTheme.textSecondary))),
        ]),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [Color(0xFF0D1B3E), AppTheme.primary, Color(0xFF1E3A7B)]),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 28),
              child: Column(children: [
                _header(),
                const SizedBox(height: 32),
                _card(auth),
              ]),
            ),
          ),
        ),
      ),
    );
  }

  Widget _header() => Column(children: [
    Container(width: 86, height: 86,
      decoration: BoxDecoration(color: AppTheme.accent, shape: BoxShape.circle,
        boxShadow: [BoxShadow(color: AppTheme.accent.withOpacity(0.45),
            blurRadius: 24, offset: const Offset(0, 10))]),
      child: const Icon(Icons.smart_toy_rounded, color: Colors.white, size: 44),
    ),
    const SizedBox(height: 16),
    Text('AI ChatBot', style: GoogleFonts.playfairDisplay(fontSize: 30,
        fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 1)),
    const SizedBox(height: 6),
    Text('Your Academic Guide, Always Here',
        style: GoogleFonts.lato(fontSize: 13, color: Colors.white70)),
  ]);

  Widget _card(AuthProvider auth) => Container(
    constraints: const BoxConstraints(maxWidth: 440),
    decoration: BoxDecoration(color: Colors.white,
      borderRadius: BorderRadius.circular(28),
      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.25),
          blurRadius: 48, offset: const Offset(0, 20))]),
    child: Column(children: [
      Container(margin: const EdgeInsets.all(14),
        decoration: BoxDecoration(color: AppTheme.surfaceDark,
            borderRadius: BorderRadius.circular(14)),
        child: TabBar(
          controller: _tab,
          indicator: BoxDecoration(color: AppTheme.primary,
              borderRadius: BorderRadius.circular(12),
              boxShadow: [BoxShadow(color: AppTheme.primary.withOpacity(0.3),
                  blurRadius: 8, offset: const Offset(0, 4))]),
          indicatorSize: TabBarIndicatorSize.tab,
          labelColor: Colors.white, unselectedLabelColor: AppTheme.textSecondary,
          dividerColor: Colors.transparent,
          labelStyle: GoogleFonts.lato(fontSize: 15, fontWeight: FontWeight.w700),
          unselectedLabelStyle: GoogleFonts.lato(fontSize: 15),
          tabs: const [Tab(text: 'Sign In'), Tab(text: 'Register')],
        ),
      ),
      if (_errorMsg != null)
        Container(
          margin: const EdgeInsets.fromLTRB(16, 0, 16, 8),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(color: AppTheme.error.withOpacity(0.08),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppTheme.error.withOpacity(0.3))),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Icon(Icons.error_outline, color: AppTheme.error, size: 18),
            const SizedBox(width: 8),
            Expanded(child: Text(_errorMsg!, style: GoogleFonts.lato(
                color: AppTheme.error, fontSize: 13, height: 1.5))),
            GestureDetector(onTap: () => setState(() => _errorMsg = null),
                child: const Icon(Icons.close, color: AppTheme.error, size: 16)),
          ]),
        ),
      Padding(
        padding: const EdgeInsets.fromLTRB(22, 0, 22, 26),
        child: _isLogin ? _loginForm(auth) : _regForm(auth),
      ),
    ]),
  );

  Widget _loginForm(AuthProvider auth) => Form(key: _loginKey,
    child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _lbl('Email Address'),
      TextFormField(controller: _loginEmailCtrl,
        keyboardType: TextInputType.emailAddress,
        textInputAction: TextInputAction.next,
        validator: (v) => (v == null || !v.contains('@')) ? 'Enter a valid email' : null,
        decoration: _deco('your@email.com', Icons.email_outlined)),
      const SizedBox(height: 14),
      _lbl('Password'),
      TextFormField(controller: _loginPassCtrl, obscureText: _hideLogin,
        textInputAction: TextInputAction.done,
        onFieldSubmitted: (_) => _login(),
        validator: (v) => (v == null || v.trim().length < 4) ? 'Enter your password' : null,
        decoration: _deco('••••••••', Icons.lock_outline,
          suffix: _eye(_hideLogin, () => setState(() => _hideLogin = !_hideLogin)))),
      const SizedBox(height: 26),
      _btn('Sign In', auth.isLoading, _login),
      const SizedBox(height: 18),
      _link("Don't have an account? ", 'Register here', () => _tab.animateTo(1)),
    ]),
  );

  Widget _regForm(AuthProvider auth) => Form(key: _regKey,
    child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _lbl('I am a'),
      Row(children: [
        _roleChip('student', 'Student', Icons.person_rounded),
        const SizedBox(width: 10),
        _roleChip('mentor', 'Mentor / Admin', Icons.admin_panel_settings_rounded),
      ]),
      const SizedBox(height: 14),
      _lbl('Full Name *'),
      TextFormField(controller: _rNameCtrl,
        textCapitalization: TextCapitalization.words,
        textInputAction: TextInputAction.next,
        validator: (v) => (v == null || v.trim().isEmpty) ? 'Name is required' : null,
        decoration: _deco('eg your name', Icons.badge_outlined)),
      const SizedBox(height: 12),
      _lbl('Email Address *'),
      TextFormField(controller: _rEmailCtrl,
        keyboardType: TextInputType.emailAddress,
        textInputAction: TextInputAction.next,
        validator: (v) => (v == null || !v.contains('@')) ? 'Enter a valid email' : null,
        decoration: _deco('your@email.com', Icons.email_outlined)),
      const SizedBox(height: 12),
      _lbl('Password *'),
      TextFormField(controller: _rPassCtrl, obscureText: _hideReg,
        textInputAction: TextInputAction.next,
        validator: (v) => (v == null || v.trim().length < 6) ? 'Min 6 characters' : null,
        decoration: _deco('••••••••', Icons.lock_outline,
          suffix: _eye(_hideReg, () => setState(() => _hideReg = !_hideReg)))),
      const SizedBox(height: 12),
      _lbl('Confirm Password *'),
      TextFormField(controller: _rConfCtrl, obscureText: _hideConf,
        textInputAction: TextInputAction.next,
        validator: (v) {
          if (v == null || v.isEmpty) return 'Confirm password';
          if (v.trim() != _rPassCtrl.text.trim()) return 'Passwords do not match';
          return null;
        },
        decoration: _deco('••••••••', Icons.lock_outline,
          suffix: _eye(_hideConf, () => setState(() => _hideConf = !_hideConf)))),

      // Student-only fields
      if (_role == 'student') ...[
        const SizedBox(height: 14),

        // ── Mentor Email box ─────────────────────────────────────────────────
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: AppTheme.accent.withOpacity(0.06),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppTheme.accent.withOpacity(0.4))),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              const Icon(Icons.link, color: AppTheme.accent, size: 16),
              const SizedBox(width: 6),
              Text('Link to Your Mentor',
                  style: GoogleFonts.lato(fontSize: 13,
                      fontWeight: FontWeight.w700, color: AppTheme.accent)),
            ]),
            const SizedBox(height: 4),
            Text('Enter your mentor\'s registered email to link your account.',
                style: GoogleFonts.lato(fontSize: 12, color: AppTheme.textSecondary)),
            const SizedBox(height: 10),
            TextFormField(
              controller: _rMentorCtrl,
              keyboardType: TextInputType.emailAddress,
              textInputAction: TextInputAction.next,
              validator: (v) {
                if (v != null && v.trim().isNotEmpty && !v.contains('@'))
                  return 'Enter a valid mentor email';
                return null;
              },
              decoration: _deco('mentor@university.com', Icons.supervisor_account_outlined),
            ),
          ]),
        ),

        const SizedBox(height: 12),

        // ── Academic Details ─────────────────────────────────────────────────
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: AppTheme.primary.withOpacity(0.04),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppTheme.primary.withOpacity(0.15))),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Academic Details (optional)',
                style: GoogleFonts.lato(fontSize: 12,
                    fontWeight: FontWeight.w700, color: AppTheme.primary)),
            const SizedBox(height: 12),
            TextFormField(controller: _rProgCtrl, textInputAction: TextInputAction.next,
                decoration: _deco('Program (e.g. B.Tech)', Icons.school_outlined)),
            const SizedBox(height: 10),
            TextFormField(controller: _rBranchCtrl, textInputAction: TextInputAction.next,
                decoration: _deco('Branch (e.g. Computer Science)',
                    Icons.account_tree_outlined)),
            const SizedBox(height: 10),
            TextFormField(controller: _rSemCtrl,
                keyboardType: TextInputType.number,
                textInputAction: TextInputAction.next,
                decoration: _deco('Semester (e.g. 5)',
                    Icons.calendar_today_outlined)),
            const SizedBox(height: 10),
            TextFormField(controller: _rRollCtrl,
                textInputAction: TextInputAction.done,
                onFieldSubmitted: (_) => _register(),
                decoration: _deco('Roll Number (e.g. 21BCE102)',
                    Icons.numbers_outlined)),
          ]),
        ),
      ],

      const SizedBox(height: 24),
      _btn('Create Account', auth.isLoading, _register),
      const SizedBox(height: 16),
      _link('Already have an account? ', 'Sign In', () => _tab.animateTo(0)),
    ]),
  );

  Widget _lbl(String t) => Padding(padding: const EdgeInsets.only(bottom: 6),
    child: Text(t, style: GoogleFonts.lato(fontSize: 13,
        fontWeight: FontWeight.w600, color: AppTheme.textSecondary)));

  InputDecoration _deco(String hint, IconData icon, {Widget? suffix}) => InputDecoration(
    hintText: hint,
    hintStyle: GoogleFonts.lato(color: AppTheme.textSecondary.withOpacity(0.55), fontSize: 14),
    prefixIcon: Icon(icon, color: AppTheme.textSecondary, size: 20),
    suffixIcon: suffix, filled: true, fillColor: AppTheme.surface,
    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFFDDD9CE))),
    enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFFDDD9CE))),
    focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppTheme.primary, width: 2)),
    errorBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppTheme.error)),
    focusedErrorBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppTheme.error, width: 2)),
  );

  Widget _eye(bool hide, VoidCallback onTap) => IconButton(
    icon: Icon(hide ? Icons.visibility_off_outlined : Icons.visibility_outlined,
        color: AppTheme.textSecondary, size: 20), onPressed: onTap);

  Widget _roleChip(String role, String label, IconData icon) {
    final sel = _role == role;
    return Expanded(child: GestureDetector(
      onTap: () => setState(() { _role = role; _errorMsg = null; }),
      child: AnimatedContainer(duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 11),
        decoration: BoxDecoration(
          color: sel ? AppTheme.primary : AppTheme.surfaceDark,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: sel ? AppTheme.primary : const Color(0xFFDDD9CE)),
          boxShadow: sel ? [BoxShadow(color: AppTheme.primary.withOpacity(0.2),
              blurRadius: 8, offset: const Offset(0, 4))] : [],
        ),
        child: Column(children: [
          Icon(icon, color: sel ? Colors.white : AppTheme.textSecondary, size: 22),
          const SizedBox(height: 4),
          Text(label, textAlign: TextAlign.center, style: GoogleFonts.lato(
              color: sel ? Colors.white : AppTheme.textSecondary,
              fontWeight: FontWeight.w600, fontSize: 12)),
        ]),
      ),
    ));
  }

  Widget _btn(String label, bool loading, VoidCallback onTap) => Container(
    height: 52,
    decoration: BoxDecoration(
      gradient: const LinearGradient(colors: [AppTheme.primary, AppTheme.primaryLight]),
      borderRadius: BorderRadius.circular(14),
      boxShadow: [BoxShadow(color: AppTheme.primary.withOpacity(0.35),
          blurRadius: 12, offset: const Offset(0, 6))],
    ),
    child: Material(color: Colors.transparent,
      child: InkWell(borderRadius: BorderRadius.circular(14),
        onTap: loading ? null : onTap,
        child: Center(child: loading
            ? const SizedBox(width: 22, height: 22,
                child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5))
            : Text(label, style: GoogleFonts.lato(fontSize: 16,
                fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 0.4))),
      ),
    ),
  );

  Widget _link(String prefix, String linkText, VoidCallback onTap) =>
      Center(child: GestureDetector(onTap: onTap,
        child: RichText(text: TextSpan(
          text: prefix,
          style: GoogleFonts.lato(fontSize: 13, color: AppTheme.textSecondary),
          children: [TextSpan(text: linkText, style: GoogleFonts.lato(fontSize: 13,
              color: AppTheme.primary, fontWeight: FontWeight.w700,
              decoration: TextDecoration.underline))],
        )),
      ));
}