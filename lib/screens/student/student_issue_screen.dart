// lib/screens/student/student_issue_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/auth_provider.dart';
import '../../services/supabase_service.dart';
import '../../utils/app_theme.dart';

class StudentIssueScreen extends StatefulWidget {
  const StudentIssueScreen({super.key});
  @override
  State<StudentIssueScreen> createState() => _StudentIssueScreenState();
}

class _StudentIssueScreenState extends State<StudentIssueScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tab;
  List<IssueReport> _myIssues = [];
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadIssues());
  }

  @override
  void dispose() { _tab.dispose(); super.dispose(); }

  Future<void> _loadIssues() async {
    final auth = context.read<AuthProvider>();
    setState(() => _loading = true);
    final issues = await SupabaseService.getStudentIssues(auth.currentUser!.id);
    if (mounted) setState(() { _myIssues = issues; _loading = false; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.surface,
      appBar: AppBar(
        backgroundColor: AppTheme.primary,
        title: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Issue Reports', style: GoogleFonts.playfairDisplay(
              fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white)),
          Text('Report college problems to your mentor',
              style: GoogleFonts.lato(fontSize: 11, color: Colors.white70)),
        ]),
      ),
      body: Column(children: [
        Container(
          color: Colors.white,
          child: TabBar(
            controller: _tab,
            labelColor: AppTheme.primary,
            unselectedLabelColor: AppTheme.textSecondary,
            indicatorColor: AppTheme.primary,
            indicatorWeight: 3,
            dividerColor: const Color(0xFFDDD9CE),
            labelStyle: GoogleFonts.lato(fontSize: 14, fontWeight: FontWeight.w700),
            tabs: [
              const Tab(icon: Icon(Icons.add_circle_outline, size: 18), text: 'New Issue'),
              Tab(icon: const Icon(Icons.list_alt, size: 18),
                  text: 'My Issues (${_myIssues.length})'),
            ],
          ),
        ),
        Expanded(child: TabBarView(controller: _tab, children: [
          _newIssueForm(),
          _myIssuesList(),
        ])),
      ]),
    );
  }

  // ── New Issue Form ─────────────────────────────────────────
  Widget _newIssueForm() {
    return _IssueForm(onSubmitted: () {
      _loadIssues();
      _tab.animateTo(1);
    });
  }

  // ── My Issues List ─────────────────────────────────────────
  Widget _myIssuesList() {
    if (_loading) return const Center(child: CircularProgressIndicator());
    if (_myIssues.isEmpty) return Center(
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(Icons.inbox_outlined, size: 56,
            color: AppTheme.textSecondary.withOpacity(0.4)),
        const SizedBox(height: 16),
        Text('No issues submitted yet',
            style: GoogleFonts.playfairDisplay(fontSize: 18,
                fontWeight: FontWeight.w600, color: AppTheme.primary)),
        const SizedBox(height: 8),
        Text('Tap "New Issue" to report a problem',
            style: GoogleFonts.lato(fontSize: 13, color: AppTheme.textSecondary)),
      ]),
    );

    return RefreshIndicator(
      onRefresh: _loadIssues,
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _myIssues.length,
        itemBuilder: (_, i) => _issueCard(_myIssues[i]),
      ),
    );
  }

  Widget _issueCard(IssueReport issue) {
    final statusInfo   = IssueReport.statusInfo(issue.status);
    final priorityInfo = IssueReport.priorityInfo(issue.priority);
    final statusColor  = issue.isResolved ? AppTheme.success
        : issue.isInProgress ? AppTheme.warning : AppTheme.primary;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          // Header
          Row(children: [
            Expanded(child: Text(issue.title,
                style: GoogleFonts.lato(fontSize: 15,
                    fontWeight: FontWeight.w700, color: AppTheme.textPrimary))),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(color: statusColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12)),
              child: Text('${statusInfo['emoji']} ${statusInfo['label']}',
                  style: GoogleFonts.lato(fontSize: 11,
                      fontWeight: FontWeight.w700, color: statusColor)),
            ),
          ]),
          const SizedBox(height: 8),

          // Category + Priority
          Wrap(spacing: 8, children: [
            _chip(IssueReport.categoryLabel(issue.category), AppTheme.primary),
            _chip('${priorityInfo['emoji']} ${priorityInfo['label']}',
                issue.isUrgent ? AppTheme.error : AppTheme.textSecondary),
          ]),
          const SizedBox(height: 8),

          // Description preview
          Text(issue.description,
              maxLines: 2, overflow: TextOverflow.ellipsis,
              style: GoogleFonts.lato(fontSize: 13,
                  color: AppTheme.textSecondary, height: 1.5)),

          // Mentor response
          if (issue.hasResponse) ...[
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppTheme.mentorBubble.withOpacity(0.08),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppTheme.mentorBubble.withOpacity(0.3)),
              ),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(children: [
                  const Icon(Icons.admin_panel_settings_rounded,
                      size: 14, color: AppTheme.mentorBubble),
                  const SizedBox(width: 6),
                  Text('Mentor Response',
                      style: GoogleFonts.lato(fontSize: 12,
                          fontWeight: FontWeight.w700, color: AppTheme.mentorBubble)),
                ]),
                const SizedBox(height: 6),
                Text(issue.mentorResponse!,
                    style: GoogleFonts.lato(fontSize: 13,
                        color: AppTheme.textPrimary, height: 1.5)),
              ]),
            ),
          ],

          const SizedBox(height: 8),
          Text(DateFormat('MMM d, y • h:mm a').format(issue.createdAt.toLocal()),
              style: GoogleFonts.lato(fontSize: 11, color: AppTheme.textSecondary)),
        ]),
      ),
    );
  }

  Widget _chip(String label, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
    decoration: BoxDecoration(color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20)),
    child: Text(label, style: GoogleFonts.lato(
        fontSize: 11, fontWeight: FontWeight.w600, color: color)),
  );
}

// ── Issue Submission Form ──────────────────────────────────────────────────
class _IssueForm extends StatefulWidget {
  final VoidCallback onSubmitted;
  const _IssueForm({required this.onSubmitted});
  @override
  State<_IssueForm> createState() => _IssueFormState();
}

class _IssueFormState extends State<_IssueForm> {
  final _formKey    = GlobalKey<FormState>();
  final _titleCtrl  = TextEditingController();
  final _descCtrl   = TextEditingController();
  String _category  = 'academic';
  String _priority  = 'medium';
  bool   _submitting = false;
  String? _error;

  static const List<Map<String, String>> _categories = [
    {'value': 'academic',       'label': '📚 Academic'},
    {'value': 'attendance',     'label': '📅 Attendance'},
    {'value': 'examination',    'label': '📝 Examination'},
    {'value': 'hostel',         'label': '🏠 Hostel'},
    {'value': 'financial',      'label': '💰 Financial'},
    {'value': 'placement',      'label': '💼 Placement'},
    {'value': 'personal',       'label': '👤 Personal'},
    {'value': 'faculty',        'label': '👨‍🏫 Faculty'},
    {'value': 'infrastructure', 'label': '🏗️ Infrastructure'},
    {'value': 'other',          'label': '📌 Other'},
  ];

  static const List<Map<String, String>> _priorities = [
    {'value': 'low',    'label': '🟢 Low'},
    {'value': 'medium', 'label': '🟡 Medium'},
    {'value': 'high',   'label': '🟠 High'},
    {'value': 'urgent', 'label': '🔴 Urgent'},
  ];

  @override
  void dispose() {
    _titleCtrl.dispose(); _descCtrl.dispose(); super.dispose();
  }

  Future<void> _submit() async {
    if (!(_formKey.currentState?.validate() ?? false)) return;
    setState(() { _submitting = true; _error = null; });

    try {
      final auth = context.read<AuthProvider>();
      final user = auth.currentUser!;

      if (user.mentorEmail == null || user.mentorEmail!.isEmpty) {
        setState(() {
          _error = 'You have not linked a mentor. Please register again with your mentor\'s email.';
          _submitting = false;
        });
        return;
      }

      await SupabaseService.submitIssue(
        studentId:       user.id,
        studentName:     user.name,
        studentEmail:    user.email,
        studentProgram:  user.program,
        studentBranch:   user.branch,
        studentSemester: user.semester,
        studentRollNo:   user.rollNumber,
        mentorEmail:     user.mentorEmail,
        category:        _category,
        title:           _titleCtrl.text.trim(),
        description:     _descCtrl.text.trim(),
        priority:        _priority,
      );

      _titleCtrl.clear(); _descCtrl.clear();
      setState(() { _category = 'academic'; _priority = 'medium'; });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('✅ Issue submitted to your mentor!',
              style: GoogleFonts.lato(color: Colors.white)),
          backgroundColor: AppTheme.success,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ));
        widget.onSubmitted();
      }
    } catch (e) {
      setState(() => _error = e.toString().replaceAll('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Form(
        key: _formKey,
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          // Info banner
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.06),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.primary.withOpacity(0.2)),
            ),
            child: Row(children: [
              const Icon(Icons.info_outline, color: AppTheme.primary, size: 20),
              const SizedBox(width: 10),
              Expanded(child: Text(
                'Your issue will be sent to your mentor. '
                'Your private AI chat will NOT be shared.',
                style: GoogleFonts.lato(fontSize: 13,
                    color: AppTheme.primary, height: 1.4),
              )),
            ]),
          ),
          const SizedBox(height: 16),

          // Error
          if (_error != null)
            Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppTheme.error.withOpacity(0.08),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppTheme.error.withOpacity(0.3)),
              ),
              child: Text(_error!, style: GoogleFonts.lato(
                  color: AppTheme.error, fontSize: 13)),
            ),

          // Category
          _lbl('Issue Category *'),
          DropdownButtonFormField<String>(
            value: _category,
            decoration: _deco('Select category', Icons.category_outlined),
            items: _categories.map((c) => DropdownMenuItem(
              value: c['value'], child: Text(c['label']!,
                  style: GoogleFonts.lato(fontSize: 14)))).toList(),
            onChanged: (v) => setState(() => _category = v!),
          ),
          const SizedBox(height: 14),

          // Priority
          _lbl('Priority *'),
          DropdownButtonFormField<String>(
            value: _priority,
            decoration: _deco('Select priority', Icons.flag_outlined),
            items: _priorities.map((p) => DropdownMenuItem(
              value: p['value'], child: Text(p['label']!,
                  style: GoogleFonts.lato(fontSize: 14)))).toList(),
            onChanged: (v) => setState(() => _priority = v!),
          ),
          const SizedBox(height: 14),

          // Title
          _lbl('Issue Title *'),
          TextFormField(
            controller: _titleCtrl,
            textCapitalization: TextCapitalization.sentences,
            validator: (v) => (v == null || v.trim().isEmpty)
                ? 'Title is required' : null,
            decoration: _deco('Brief summary of your issue', Icons.title),
          ),
          const SizedBox(height: 14),

          // Description
          _lbl('Detailed Description *'),
          TextFormField(
            controller: _descCtrl,
            maxLines: 6, minLines: 4,
            textCapitalization: TextCapitalization.sentences,
            validator: (v) => (v == null || v.trim().length < 20)
                ? 'Please describe your issue in detail (min 20 chars)' : null,
            decoration: _deco(
                'Describe your problem in detail. Include relevant dates, '
                'subject names, or any other helpful information.',
                Icons.description_outlined),
          ),
          const SizedBox(height: 24),

          // Submit button
          Container(
            height: 54,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                  colors: [AppTheme.primary, AppTheme.primaryLight]),
              borderRadius: BorderRadius.circular(14),
              boxShadow: [BoxShadow(color: AppTheme.primary.withOpacity(0.35),
                  blurRadius: 12, offset: const Offset(0, 6))],
            ),
            child: Material(color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(14),
                onTap: _submitting ? null : _submit,
                child: Center(child: _submitting
                    ? const SizedBox(width: 22, height: 22,
                        child: CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 2.5))
                    : Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        const Icon(Icons.send, color: Colors.white, size: 20),
                        const SizedBox(width: 10),
                        Text('Submit Issue to Mentor',
                            style: GoogleFonts.lato(fontSize: 16,
                                fontWeight: FontWeight.w700, color: Colors.white)),
                      ])),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Center(child: Text('🔒 Your AI chat history stays private',
              style: GoogleFonts.lato(fontSize: 12, color: AppTheme.textSecondary))),
        ]),
      ),
    );
  }

  Widget _lbl(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 6),
    child: Text(t, style: GoogleFonts.lato(fontSize: 13,
        fontWeight: FontWeight.w600, color: AppTheme.textSecondary)));

  InputDecoration _deco(String hint, IconData icon) => InputDecoration(
    hintText: hint,
    hintStyle: GoogleFonts.lato(
        color: AppTheme.textSecondary.withOpacity(0.6), fontSize: 13),
    prefixIcon: Icon(icon, color: AppTheme.textSecondary, size: 20),
    filled: true, fillColor: Colors.white,
    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFFDDD9CE))),
    enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFFDDD9CE))),
    focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppTheme.primary, width: 2)),
    errorBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppTheme.error)),
  );
}