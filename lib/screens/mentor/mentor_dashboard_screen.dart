// lib/screens/mentor/mentor_dashboard_screen.dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/pdf_extraction_service.dart';
import '../../services/auth_provider.dart';
import '../../services/chat_provider.dart';
import '../../services/document_queue_service.dart';
import '../../services/supabase_service.dart';
import '../../utils/app_theme.dart';
import '../../utils/file_opener.dart';
import '../../widgets/document_viewer_dialog.dart';
import '../../widgets/timetable_grid_view.dart';
import '../../widgets/timetable_sync_dialog.dart';
import '../../widgets/teacher_selector_dialog.dart';
import '../../utils/mru_timetable_data.dart';
import '../auth_screen.dart';
import 'mentor_ai_chat_screen.dart';

class MentorDashboardScreen extends StatefulWidget {
  const MentorDashboardScreen({super.key});
  @override
  State<MentorDashboardScreen> createState() => _MentorDashboardScreenState();
}

class _MentorDashboardScreenState extends State<MentorDashboardScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tab;
  String _issueFilter = 'all';
  List<IssueReport> _issues = [];
  bool _issuesLoading = false;

  // Documents State
  List<StudentDocument> _mentorDocs = [];
  bool _docsLoading = false;
  bool _docUploading = false;
  String _docUploadStatus = '';

  // Academic Session & Term Filter State
  String _selectedAcademicYear = '2026-2027';
  String _selectedTerm = 'odd'; // 'odd' (Jun-Dec) or 'even' (Jan-May)
  String _selectedMentorSection = 'CSE 5A';
  static const List<String> _academicYears = ['2026-2027', '2025-2026', '2024-2025'];

  static const List<Map<String, dynamic>> _docTypes = [
    {'value': 'academic_calendar', 'label': 'Academic Calendar',  'emoji': '🗓️', 'color': Color(0xFF8B5CF6)},
    {'value': 'syllabus',          'label': 'Syllabus',           'emoji': '📚', 'color': Color(0xFF10B981)},
    {'value': 'marksheet',         'label': 'Marksheet / Result', 'emoji': '📊', 'color': Color(0xFFF59E0B)},
    {'value': 'attendance',        'label': 'Attendance Register','emoji': '✅', 'color': Color(0xFF06B6D4)},
    {'value': 'assignment',        'label': 'Assignment Notice',  'emoji': '📝', 'color': Color(0xFFEF4444)},
    {'value': 'other',             'label': 'Circular / Other',   'emoji': '🏛️', 'color': Color(0xFF6B7280)},
  ];

  @override
  void initState() {
    super.initState();
    // Auto-detect term based on current date
    final now = DateTime.now();
    if (now.month >= 6) {
      _selectedAcademicYear = '${now.year}-${now.year + 1}';
      _selectedTerm = 'odd';
    } else {
      _selectedAcademicYear = '${now.year - 1}-${now.year}';
      _selectedTerm = 'even';
    }

    _tab = TabController(length: 5, vsync: this);
    _tab.addListener(() => setState(() {}));
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadAll());
  }

  Future<void> _openDocumentRaw(StudentDocument doc) async {
    try {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (_) => const Center(child: CircularProgressIndicator()),
      );
      final bytes = await SupabaseService.getDocumentBytes(doc);
      if (mounted) Navigator.pop(context);

      if (bytes != null && bytes.isNotEmpty) {
        openRawBytes(
          bytes: bytes,
          fileName: doc.fileName,
          mimeType: doc.mimeType,
        );
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Opening document "${doc.fileName}"...'),
              duration: const Duration(seconds: 2),
              behavior: SnackBarBehavior.floating,
            ),
          );
        }
      } else if (doc.contentBase64 != null && doc.contentBase64!.isNotEmpty) {
        openRawDocument(
          base64Content: doc.contentBase64!,
          fileName: doc.fileName,
          mimeType: doc.mimeType,
        );
      } else {
        if (mounted) {
          DocumentViewerDialog.show(context, doc);
        }
      }
    } catch (e) {
      if (mounted && Navigator.canPop(context)) Navigator.pop(context);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not open document: $e')),
        );
      }
    }
  }

  @override
  void dispose() { _tab.dispose(); super.dispose(); }

  Future<void> _loadAll() async {
    final auth  = context.read<AuthProvider>();
    final chat  = context.read<ChatProvider>();
    final email = auth.currentUser?.email ?? '';
    final userId = auth.currentUser?.id ?? '';
    chat.setCurrentUser(auth.currentUser);
    await chat.loadMentorDashboard(email);
    await chat.loadProgressReports(email);
    await _loadIssues(email);
    if (userId.isNotEmpty) {
      await _loadMentorDocuments(userId);
    }
  }

  Future<void> _loadMentorDocuments(String mentorId) async {
    setState(() => _docsLoading = true);
    final docs = await SupabaseService.getMentorDocuments(mentorId);
    if (mounted) setState(() { _mentorDocs = docs; _docsLoading = false; });
  }

  Future<void> _loadIssues(String mentorEmail) async {
    setState(() => _issuesLoading = true);
    final issues = await SupabaseService.getMentorIssues(mentorEmail);
    if (mounted) setState(() { _issues = issues; _issuesLoading = false; });
  }

  List<IssueReport> get _filteredIssues => _issueFilter == 'all'
      ? _issues : _issues.where((i) => i.status == _issueFilter).toList();

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final chat = context.watch<ChatProvider>();
    final openIssues = _issues.where((i) => i.isOpen).length;

    return Scaffold(
      backgroundColor: const Color(0xFFF0F4FF),
      body: Column(children: [
        _buildHeader(auth, chat, openIssues),
        _buildTabBar(chat, openIssues),
        Expanded(child: TabBarView(controller: _tab, children: [
          _classTab(chat),
          _documentsTab(auth),
          _issuesTab(auth),
          _progressTab(chat),
          _aiChatTab(),
        ])),
      ]),
    );
  }

  // ══════════════════════════════════════════════════════════
  // HEADER — gradient with stats
  // ══════════════════════════════════════════════════════════
  Widget _buildHeader(AuthProvider auth, ChatProvider chat, int openIssues) {
    final name     = auth.currentUser?.name ?? 'Mentor';
    final initials = name.trim().split(' ')
        .map((w) => w.isNotEmpty ? w[0] : '').take(2).join().toUpperCase();
    final students  = chat.myStudents.length;
    final active    = chat.conversations.where((c) => c.status == 'active').length;
    final resolved  = chat.conversations.where((c) => c.status == 'resolved').length;

    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF1A2B5F), Color(0xFF2D4A9E), Color(0xFF1A3A8F)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
          child: Column(children: [
            // Top row: avatar + name + actions
            Row(children: [
              Container(
                width: 52, height: 52,
                decoration: BoxDecoration(
                  color: AppTheme.accent,
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white.withOpacity(0.3), width: 2),
                  boxShadow: [BoxShadow(color: AppTheme.accent.withOpacity(0.4),
                      blurRadius: 12, offset: const Offset(0, 4))],
                ),
                child: Center(child: Text(initials, style: GoogleFonts.playfairDisplay(
                    fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white))),
              ),
              const SizedBox(width: 14),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Mentor Dashboard', style: GoogleFonts.lato(
                      fontSize: 12, color: Colors.white60, letterSpacing: 1)),
                  Text(name, style: GoogleFonts.playfairDisplay(
                      fontSize: 20, fontWeight: FontWeight.w700, color: Colors.white)),
                ])),
              // Refresh
              _headerBtn(Icons.refresh_rounded, onTap: _loadAll),
              const SizedBox(width: 8),
              // Logout
              _headerBtn(Icons.logout_rounded, onTap: () async {
                await context.read<AuthProvider>().logout();
                if (!mounted) return;
                Navigator.of(context).pushReplacement(
                    MaterialPageRoute(builder: (_) => const AuthScreen()));
              }),
            ]),
            const SizedBox(height: 16),
            // Stats cards row
            Row(children: [
              _statCard('Students', '$students', Icons.people_alt_rounded,
                  const Color(0xFF60A5FA), const Color(0xFF1E40AF)),
              const SizedBox(width: 8),
              _statCard('Issues', '$openIssues', Icons.warning_amber_rounded,
                  openIssues > 0 ? const Color(0xFFFBBF24) : const Color(0xFF34D399),
                  openIssues > 0 ? const Color(0xFF92400E) : const Color(0xFF065F46)),
              const SizedBox(width: 8),
              _statCard('Active', '$active', Icons.chat_bubble_rounded,
                  const Color(0xFF34D399), const Color(0xFF065F46)),
              const SizedBox(width: 8),
              _statCard('Resolved', '$resolved', Icons.check_circle_rounded,
                  const Color(0xFFA78BFA), const Color(0xFF4C1D95)),
            ]),
          ]),
        ),
      ),
    );
  }

  Widget _headerBtn(IconData icon, {required VoidCallback onTap}) =>
      GestureDetector(
        onTap: onTap,
        child: Container(
          width: 38, height: 38,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.12),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: Colors.white.withOpacity(0.2)),
          ),
          child: Icon(icon, color: Colors.white, size: 18),
        ),
      );

  Widget _statCard(String label, String value, IconData icon,
      Color iconColor, Color bgColor) {
    return Expanded(child: Container(
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.15)),
      ),
      child: Column(children: [
        Container(
          width: 30, height: 30,
          decoration: BoxDecoration(
            color: bgColor.withOpacity(0.3),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: iconColor, size: 16),
        ),
        const SizedBox(height: 6),
        Text(value, style: GoogleFonts.playfairDisplay(
            fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white)),
        Text(label, style: GoogleFonts.lato(
            fontSize: 9, color: Colors.white60, letterSpacing: 0.2),
            textAlign: TextAlign.center, maxLines: 2),
      ]),
    ));
  }

  // ══════════════════════════════════════════════════════════
  // TAB BAR
  // ══════════════════════════════════════════════════════════
  Widget _buildTabBar(ChatProvider chat, int openIssues) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        boxShadow: [BoxShadow(color: Color(0x0F000000), blurRadius: 8,
            offset: Offset(0, 2))],
      ),
      child: TabBar(
        controller: _tab,
        labelColor: AppTheme.primary,
        unselectedLabelColor: const Color(0xFF9CA3AF),
        indicatorColor: AppTheme.primary,
        indicatorWeight: 3,
        indicatorSize: TabBarIndicatorSize.tab,
        dividerColor: Colors.transparent,
        labelStyle: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700),
        unselectedLabelStyle: GoogleFonts.lato(fontSize: 12),
        tabs: [
          _buildTab(Icons.groups_rounded, 'My Class',
              badge: chat.myStudents.length),
          _buildTab(Icons.folder_shared_rounded, 'Documents',
              badge: _mentorDocs.length),
          _buildTab(Icons.report_problem_rounded, 'Issues',
              badge: openIssues, badgeColor: openIssues > 0 ? Colors.red : null),
          _buildTab(Icons.insights_rounded, 'Progress'),
          _buildTab(Icons.smart_toy_rounded, 'AI Chat',
              highlight: true),
        ],
      ),
    );
  }

  Widget _buildTab(IconData icon, String label,
      {int? badge, Color? badgeColor, bool highlight = false}) {
    return Tab(
      height: 54,
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Stack(alignment: Alignment.topRight, children: [
          Icon(icon, size: 20),
          if (badge != null && badge > 0)
            Positioned(
              top: -2, right: -4,
              child: Container(
                padding: const EdgeInsets.all(3),
                decoration: BoxDecoration(
                  color: badgeColor ?? AppTheme.primary,
                  shape: BoxShape.circle,
                ),
                child: Text('$badge', style: GoogleFonts.lato(
                    fontSize: 8, color: Colors.white, fontWeight: FontWeight.w700)),
              ),
            ),
          if (highlight)
            Positioned(
              top: -2, right: -4,
              child: Container(
                width: 8, height: 8,
                decoration: const BoxDecoration(
                    color: AppTheme.mentorBubble, shape: BoxShape.circle),
              ),
            ),
        ]),
        const SizedBox(height: 3),
        Text(label),
      ]),
    );
  }

  // ══════════════════════════════════════════════════════════
  // TAB 1 — MY CLASS
  // ══════════════════════════════════════════════════════════
  Widget _classTab(ChatProvider chat) {
    if (chat.isLoading) return _loadingView();
    if (chat.myStudents.isEmpty) {
      return _emptyView(
      Icons.school_outlined,
      'No students yet',
      'Students appear here when they\nregister using your email address',
    );
    }

    return RefreshIndicator(
      onRefresh: _loadAll,
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: chat.myStudents.length,
        itemBuilder: (_, i) => _studentCard(chat.myStudents[i], chat, i),
      ),
    );
  }

  Widget _studentCard(UserModel s, ChatProvider chat, int index) {
    final convs    = chat.conversations.where((c) => c.studentId == s.id).toList();
    final myIssues = _issues.where((i) => i.studentId == s.id).toList();
    final openI    = myIssues.where((i) => i.isOpen).length;
    final initials = s.name.trim().split(' ')
        .map((w) => w.isNotEmpty ? w[0] : '').take(2).join().toUpperCase();

    final colors = [
      [const Color(0xFFEFF6FF), const Color(0xFF3B82F6)],
      [const Color(0xFFF0FDF4), const Color(0xFF22C55E)],
      [const Color(0xFFFFF7ED), const Color(0xFFF97316)],
      [const Color(0xFFFDF4FF), const Color(0xFFA855F7)],
    ];
    final c = colors[index % colors.length];

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05),
            blurRadius: 12, offset: const Offset(0, 4))],
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            // Avatar
            Container(
              width: 52, height: 52,
              decoration: BoxDecoration(
                color: c[0], borderRadius: BorderRadius.circular(14),
              ),
              child: Center(child: Text(initials, style: GoogleFonts.playfairDisplay(
                  fontSize: 20, fontWeight: FontWeight.w700, color: c[1]))),
            ),
            const SizedBox(width: 14),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(s.name, style: GoogleFonts.lato(fontSize: 16,
                    fontWeight: FontWeight.w700, color: const Color(0xFF111827))),
                const SizedBox(height: 2),
                Text(s.email, style: GoogleFonts.lato(fontSize: 12,
                    color: const Color(0xFF6B7280)), overflow: TextOverflow.ellipsis),
              ])),
            if (openI > 0)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: const Color(0xFFFEF2F2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFFFCA5A5)),
                ),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  const Icon(Icons.warning_amber_rounded, size: 12,
                      color: Color(0xFFEF4444)),
                  const SizedBox(width: 4),
                  Text('$openI issue${openI > 1 ? 's' : ''}',
                      style: GoogleFonts.lato(fontSize: 11,
                          fontWeight: FontWeight.w700, color: const Color(0xFFEF4444))),
                ]),
              ),
          ]),

          // Academic chips
          if (s.program != null || s.branch != null) ...[
            const SizedBox(height: 12),
            Wrap(spacing: 8, runSpacing: 6, children: [
              if (s.program?.isNotEmpty == true)
                _infoChip(Icons.school_rounded, s.program!, c[1]),
              if (s.rollNumber?.isNotEmpty == true)
                _infoChip(Icons.numbers_rounded, s.rollNumber!, c[1]),
              if (s.branch?.isNotEmpty == true)
                _infoChip(Icons.account_tree_rounded, s.branch!, c[1]),
              if (s.semester?.isNotEmpty == true)
                _infoChip(Icons.calendar_today_rounded, 'Sem ${s.semester!}', c[1]),
            ]),
          ],

          const SizedBox(height: 12),
          // Stats row
          Container(
            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
            decoration: BoxDecoration(
              color: const Color(0xFFF9FAFB),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _miniStat2('Chats', '${convs.length}',
                    Icons.chat_bubble_outline_rounded, c[1]),
                _divider2(),
                _miniStat2('Issues', '${myIssues.length}',
                    Icons.report_outlined, c[1]),
                _divider2(),
                _miniStat2('Resolved', '${convs.where((c) => c.isResolved).length}',
                    Icons.check_circle_outline_rounded, c[1]),
              ]),
          ),
        ]),
      ),
    );
  }

  Widget _infoChip(IconData icon, String label, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    decoration: BoxDecoration(
      color: color.withOpacity(0.08),
      borderRadius: BorderRadius.circular(20),
    ),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 12, color: color),
      const SizedBox(width: 5),
      Text(label, style: GoogleFonts.lato(fontSize: 12,
          color: color, fontWeight: FontWeight.w600)),
    ]),
  );

  Widget _miniStat2(String label, String value, IconData icon, Color color) =>
      Column(children: [
        Icon(icon, size: 18, color: color),
        const SizedBox(height: 4),
        Text(value, style: GoogleFonts.lato(fontSize: 16,
            fontWeight: FontWeight.w700, color: const Color(0xFF111827))),
        Text(label, style: GoogleFonts.lato(fontSize: 11,
            color: const Color(0xFF6B7280))),
      ]);

  Widget _divider2() => Container(
      height: 32, width: 1, color: const Color(0xFFE5E7EB));

  // ══════════════════════════════════════════════════════════
  // TAB 2 — ISSUES
  // ══════════════════════════════════════════════════════════
  Widget _issuesTab(AuthProvider auth) {
    if (_issuesLoading) return _loadingView();

    return Column(children: [
      // Privacy notice
      Container(
        margin: const EdgeInsets.fromLTRB(16, 14, 16, 0),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: const Color(0xFFF0FDF4),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFF86EFAC)),
        ),
        child: Row(children: [
          const Icon(Icons.lock_outline_rounded, color: Color(0xFF16A34A), size: 16),
          const SizedBox(width: 8),
          Expanded(child: Text(
            'Student AI conversations are private. Only issue reports are shown here.',
            style: GoogleFonts.lato(fontSize: 12,
                color: const Color(0xFF15803D), height: 1.4))),
        ]),
      ),
      _issueFilterChips(),
      Expanded(
        child: _filteredIssues.isEmpty
            ? _emptyView(Icons.inbox_outlined, 'No issues',
                _issueFilter == 'all' ? 'No issues submitted yet'
                    : 'No ${_issueFilter.replaceAll('_', ' ')} issues')
            : RefreshIndicator(
                onRefresh: () => _loadIssues(auth.currentUser?.email ?? ''),
                child: ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _filteredIssues.length,
                  itemBuilder: (_, i) => _issueCard(_filteredIssues[i], auth),
                ),
              ),
      ),
    ]);
  }

  Widget _issueFilterChips() {
    final filters = [
      ('all', 'All', const Color(0xFF6366F1)),
      ('open', 'Open', const Color(0xFFEF4444)),
      ('in_progress', 'In Progress', const Color(0xFFF59E0B)),
      ('resolved', 'Resolved', const Color(0xFF10B981)),
    ];
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      child: Row(children: filters.map((f) {
        final sel = _issueFilter == f.$1;
        return Padding(padding: const EdgeInsets.only(right: 8),
          child: GestureDetector(
            onTap: () => setState(() => _issueFilter = f.$1),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: sel ? f.$3 : Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: sel ? f.$3 : const Color(0xFFE5E7EB)),
                boxShadow: sel ? [BoxShadow(color: f.$3.withOpacity(0.3),
                    blurRadius: 8, offset: const Offset(0, 3))] : [],
              ),
              child: Text(f.$2, style: GoogleFonts.lato(
                  fontSize: 13, fontWeight: FontWeight.w600,
                  color: sel ? Colors.white : const Color(0xFF6B7280))),
            ),
          ));
      }).toList()),
    );
  }

  Widget _issueCard(IssueReport issue, AuthProvider auth) {
    final statusInfo   = IssueReport.statusInfo(issue.status);
    final priorityInfo = IssueReport.priorityInfo(issue.priority);

    final priorityColors = {
      'urgent': [const Color(0xFFFEF2F2), const Color(0xFFEF4444)],
      'high':   [const Color(0xFFFFF7ED), const Color(0xFFF97316)],
      'medium': [const Color(0xFFFFFBEB), const Color(0xFFF59E0B)],
      'low':    [const Color(0xFFF0FDF4), const Color(0xFF10B981)],
    };
    final pc = priorityColors[issue.priority] ??
        [const Color(0xFFF9FAFB), const Color(0xFF6B7280)];

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05),
            blurRadius: 12, offset: const Offset(0, 4))],
      ),
      child: Column(children: [
        // Top colored strip
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          decoration: BoxDecoration(
            color: pc[0],
            borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Row(children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: pc[1].withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text('${priorityInfo['emoji']} ${priorityInfo['label']}',
                  style: GoogleFonts.lato(fontSize: 11,
                      fontWeight: FontWeight.w700, color: pc[1])),
            ),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.6),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(IssueReport.categoryLabel(issue.category),
                  style: GoogleFonts.lato(fontSize: 11,
                      fontWeight: FontWeight.w600, color: AppTheme.primary)),
            ),
            const Spacer(),
            Text('${statusInfo['emoji']} ${statusInfo['label']}',
                style: GoogleFonts.lato(fontSize: 11,
                    fontWeight: FontWeight.w700, color: AppTheme.textSecondary)),
          ]),
        ),

        // Content
        Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            // Student info
            Row(children: [
              Container(
                width: 38, height: 38,
                decoration: BoxDecoration(
                  color: AppTheme.primary.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Center(child: Text(
                  issue.studentName?.isNotEmpty == true
                      ? issue.studentName![0].toUpperCase() : 'S',
                  style: GoogleFonts.playfairDisplay(fontSize: 16,
                      fontWeight: FontWeight.w700, color: AppTheme.primary))),
              ),
              const SizedBox(width: 10),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(issue.studentName ?? 'Student', style: GoogleFonts.lato(
                      fontSize: 14, fontWeight: FontWeight.w700,
                      color: const Color(0xFF111827))),
                  if (issue.studentProgram != null)
                    Text('${issue.studentProgram} • ${issue.studentBranch ?? ''}'
                        '${issue.studentSemester != null ? " • Sem ${issue.studentSemester}" : ""}',
                        style: GoogleFonts.lato(fontSize: 11,
                            color: const Color(0xFF6B7280))),
                ])),
            ]),
            const SizedBox(height: 12),

            // Title
            Text(issue.title, style: GoogleFonts.lato(fontSize: 15,
                fontWeight: FontWeight.w700, color: const Color(0xFF111827))),
            const SizedBox(height: 6),
            Text(issue.description, maxLines: 2, overflow: TextOverflow.ellipsis,
                style: GoogleFonts.lato(fontSize: 13,
                    color: const Color(0xFF6B7280), height: 1.5)),

            // Mentor response
            if (issue.hasResponse) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFF0FDF4),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: const Color(0xFF86EFAC)),
                ),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      const Icon(Icons.reply_rounded,
                          size: 14, color: Color(0xFF16A34A)),
                      const SizedBox(width: 6),
                      Text('Your Response', style: GoogleFonts.lato(
                          fontSize: 12, fontWeight: FontWeight.w700,
                          color: const Color(0xFF16A34A))),
                    ]),
                    const SizedBox(height: 6),
                    Text(issue.mentorResponse!, style: GoogleFonts.lato(
                        fontSize: 13, color: const Color(0xFF15803D), height: 1.4)),
                  ]),
              ),
            ],

            const SizedBox(height: 12),
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Row(children: [
                const Icon(Icons.access_time_rounded,
                    size: 12, color: Color(0xFF9CA3AF)),
                const SizedBox(width: 4),
                Text(DateFormat('MMM d • h:mm a')
                    .format(issue.createdAt.toLocal()),
                    style: GoogleFonts.lato(fontSize: 11,
                        color: const Color(0xFF9CA3AF))),
              ]),
              if (!issue.isResolved)
                GestureDetector(
                  onTap: () => _showRespondDialog(issue, auth),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 7),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                          colors: [Color(0xFF1A2B5F), Color(0xFF2D4A9E)]),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [BoxShadow(
                          color: AppTheme.primary.withOpacity(0.3),
                          blurRadius: 8, offset: const Offset(0, 3))],
                    ),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      const Icon(Icons.reply_rounded,
                          color: Colors.white, size: 14),
                      const SizedBox(width: 6),
                      Text('Respond', style: GoogleFonts.lato(
                          fontSize: 12, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                    ]),
                  ),
                ),
            ]),
          ]),
        ),
      ]),
    );
  }

  void _showRespondDialog(IssueReport issue, AuthProvider auth) {
    final ctrl = TextEditingController();
    String selectedStatus = 'in_progress';

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        title: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Respond to Issue', style: GoogleFonts.playfairDisplay(
              fontSize: 18, fontWeight: FontWeight.w700, color: AppTheme.primary)),
          const SizedBox(height: 4),
          Text(issue.title, style: GoogleFonts.lato(fontSize: 13,
              color: const Color(0xFF6B7280))),
        ]),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(
            controller: ctrl, maxLines: 4, minLines: 3,
            textCapitalization: TextCapitalization.sentences,
            decoration: InputDecoration(
              hintText: 'Write your response...',
              hintStyle: GoogleFonts.lato(color: const Color(0xFF9CA3AF), fontSize: 14),
              filled: true, fillColor: const Color(0xFFF9FAFB),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(14),
                  borderSide: const BorderSide(color: Color(0xFFE5E7EB))),
              enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(14),
                  borderSide: const BorderSide(color: Color(0xFFE5E7EB))),
              focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(14),
                  borderSide: const BorderSide(color: AppTheme.primary, width: 2)),
              contentPadding: const EdgeInsets.all(14),
            ),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            initialValue: selectedStatus,
            decoration: InputDecoration(
              labelText: 'Update Status',
              labelStyle: GoogleFonts.lato(color: const Color(0xFF6B7280), fontSize: 13),
              filled: true, fillColor: const Color(0xFFF9FAFB),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(14),
                  borderSide: const BorderSide(color: Color(0xFFE5E7EB))),
              contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            ),
            items: ['in_progress', 'resolved', 'closed'].map((s) =>
                DropdownMenuItem(value: s,
                    child: Text('${IssueReport.statusInfo(s)['emoji']} '
                        '${IssueReport.statusInfo(s)['label']}',
                        style: GoogleFonts.lato(fontSize: 14)))).toList(),
            onChanged: (v) => selectedStatus = v!,
          ),
        ]),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text('Cancel', style: GoogleFonts.lato(
                color: const Color(0xFF6B7280))),
          ),
          Container(
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                  colors: [Color(0xFF1A2B5F), Color(0xFF2D4A9E)]),
              borderRadius: BorderRadius.circular(12),
            ),
            child: TextButton(
              onPressed: () async {
                if (ctrl.text.trim().isEmpty) return;
                await SupabaseService.respondToIssue(
                  issueId: issue.id,
                  response: ctrl.text.trim(),
                  newStatus: selectedStatus,
                );
                Navigator.pop(ctx);
                await _loadIssues(auth.currentUser?.email ?? '');
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                  content: Row(children: [
                    const Icon(Icons.check_circle_rounded,
                        color: Colors.white, size: 16),
                    const SizedBox(width: 8),
                    Text('Response sent!', style: GoogleFonts.lato(color: Colors.white)),
                  ]),
                  backgroundColor: const Color(0xFF10B981),
                  behavior: SnackBarBehavior.floating,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ));
                }
              },
              child: Text('Send Response', style: GoogleFonts.lato(
                  color: Colors.white, fontWeight: FontWeight.w700)),
            ),
          ),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════════════════════
  // TAB 3 — PROGRESS
  // ══════════════════════════════════════════════════════════
  Widget _progressTab(ChatProvider chat) {
    if (chat.loadingProgress) return _loadingView();
    if (chat.progressReports.isEmpty) {
      return _emptyView(
        Icons.insights_outlined, 'No data yet',
        'Reports appear once students start chatting');
    }

    final sorted = [...chat.progressReports]
      ..sort((a, b) => b.engagementScore.compareTo(a.engagementScore));
    final avg = sorted.isEmpty ? 0.0
        : sorted.fold(0.0, (s, r) => s + r.engagementScore) / sorted.length;

    return RefreshIndicator(
      onRefresh: _loadAll,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        _classOverviewCard(sorted, avg),
        const SizedBox(height: 20),
        Text('Individual Progress', style: GoogleFonts.playfairDisplay(
            fontSize: 16, fontWeight: FontWeight.w700,
            color: const Color(0xFF111827))),
        const SizedBox(height: 12),
        ...sorted.map((r) => _progressCard(r)),
      ]),
    );
  }

  Widget _classOverviewCard(List<StudentProgressReport> reports, double avg) {
    final activeCount = reports.where((r) => r.totalMessages > 0).length;
    final totalIssues = reports.fold(0, (s, r) => s + r.openIssues);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
            colors: [Color(0xFF1A2B5F), Color(0xFF2D4A9E), Color(0xFF1A3A8F)],
            begin: Alignment.topLeft, end: Alignment.bottomRight),
        borderRadius: BorderRadius.circular(24),
        boxShadow: [BoxShadow(color: AppTheme.primary.withOpacity(0.3),
            blurRadius: 20, offset: const Offset(0, 8))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.bar_chart_rounded, color: AppTheme.accentLight, size: 20),
          const SizedBox(width: 8),
          Text('Class Overview', style: GoogleFonts.playfairDisplay(
              fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white)),
        ]),
        const SizedBox(height: 20),
        Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
          _overviewStat('Total', '${reports.length}'),
          _overviewStat('Active', '$activeCount'),
          _overviewStat('Open Issues', '$totalIssues'),
          _overviewStat('Avg Score', '${avg.toStringAsFixed(0)}%'),
        ]),
        const SizedBox(height: 20),
        Text('Class Engagement', style: GoogleFonts.lato(
            fontSize: 12, color: Colors.white60, letterSpacing: 0.5)),
        const SizedBox(height: 8),
        Stack(children: [
          Container(height: 10,
              decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(5))),
          FractionallySizedBox(
            widthFactor: avg / 100,
            child: Container(height: 10,
              decoration: BoxDecoration(
                color: avg >= 60 ? const Color(0xFF34D399) : const Color(0xFFFBBF24),
                borderRadius: BorderRadius.circular(5),
              )),
          ),
        ]),
        const SizedBox(height: 6),
        Text('${avg.toStringAsFixed(1)}% average engagement',
            style: GoogleFonts.lato(fontSize: 12, color: Colors.white60)),
      ]),
    );
  }

  Widget _overviewStat(String label, String value) => Column(children: [
    Text(value, style: GoogleFonts.playfairDisplay(
        fontSize: 22, fontWeight: FontWeight.w700, color: Colors.white)),
    Text(label, style: GoogleFonts.lato(fontSize: 11, color: Colors.white60)),
  ]);

  Widget _progressCard(StudentProgressReport r) {
    final score = r.engagementScore;
    final color = score >= 80 ? const Color(0xFF10B981)
        : score >= 60 ? const Color(0xFF3B82F6)
        : score >= 40 ? const Color(0xFFF59E0B)
        : const Color(0xFFEF4444);
    final bgColor = score >= 80 ? const Color(0xFFF0FDF4)
        : score >= 60 ? const Color(0xFFEFF6FF)
        : score >= 40 ? const Color(0xFFFFFBEB)
        : const Color(0xFFFEF2F2);
    final initials = r.student.name.trim().split(' ')
        .map((w) => w.isNotEmpty ? w[0] : '').take(2).join().toUpperCase();

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05),
            blurRadius: 12, offset: const Offset(0, 4))],
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Container(
              width: 48, height: 48,
              decoration: BoxDecoration(
                  color: bgColor, borderRadius: BorderRadius.circular(14)),
              child: Center(child: Text(initials,
                  style: GoogleFonts.playfairDisplay(fontSize: 18,
                      fontWeight: FontWeight.w700, color: color)))),
            const SizedBox(width: 12),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(r.student.name, style: GoogleFonts.lato(fontSize: 15,
                    fontWeight: FontWeight.w700, color: const Color(0xFF111827))),
                if (r.student.program != null)
                  Text('${r.student.program} • Sem ${r.student.semester ?? '-'}',
                      style: GoogleFonts.lato(fontSize: 12,
                          color: const Color(0xFF6B7280))),
              ])),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: bgColor,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: color.withOpacity(0.3)),
              ),
              child: Text(r.engagementLabel, style: GoogleFonts.lato(
                  fontSize: 12, fontWeight: FontWeight.w700, color: color)),
            ),
          ]),
          const SizedBox(height: 14),
          Row(children: [
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Engagement Score', style: GoogleFonts.lato(
                    fontSize: 11, color: const Color(0xFF9CA3AF))),
                const SizedBox(height: 6),
                Stack(children: [
                  Container(height: 8,
                    decoration: BoxDecoration(
                        color: const Color(0xFFF3F4F6),
                        borderRadius: BorderRadius.circular(4))),
                  FractionallySizedBox(
                    widthFactor: score / 100,
                    child: Container(height: 8,
                      decoration: BoxDecoration(
                          color: color,
                          borderRadius: BorderRadius.circular(4))),
                  ),
                ]),
              ])),
            const SizedBox(width: 12),
            Text('${score.toStringAsFixed(0)}%', style: GoogleFonts.lato(
                fontSize: 18, fontWeight: FontWeight.w700, color: color)),
          ]),
          const SizedBox(height: 12),
          Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
            _miniStat('Chats', '${r.conversations.length}',
                Icons.forum_outlined, color),
            _miniStat('Messages', '${r.totalMessages}',
                Icons.chat_bubble_outline, color),
            _miniStat('Issues', '${r.issues.length}',
                Icons.report_outlined, color),
            _miniStat('Resolved', '${r.resolvedConversations}',
                Icons.check_circle_outline, color),
          ]),
        ]),
      ),
    );
  }

  Widget _miniStat(String label, String value, IconData icon, Color color) =>
      Column(children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(height: 3),
        Text(value, style: GoogleFonts.lato(fontSize: 14,
            fontWeight: FontWeight.w700, color: const Color(0xFF111827))),
        Text(label, style: GoogleFonts.lato(fontSize: 10,
            color: const Color(0xFF6B7280))),
      ]);

  // ══════════════════════════════════════════════════════════
  // TAB 4 — AI CHAT
  // ══════════════════════════════════════════════════════════
  Widget _aiChatTab() => SingleChildScrollView(
    padding: const EdgeInsets.fromLTRB(24, 32, 24, 40),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Center(child: Container(
          width: 90, height: 90,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
                colors: [Color(0xFF065F46), Color(0xFF059669)]),
            shape: BoxShape.circle,
            boxShadow: [BoxShadow(color: AppTheme.mentorBubble.withOpacity(0.3),
                blurRadius: 20, offset: const Offset(0, 8))],
          ),
          child: const Icon(Icons.smart_toy_rounded,
              size: 44, color: Colors.white),
        )),
        const SizedBox(height: 20),
        Text('AI Assistant', textAlign: TextAlign.center,
            style: GoogleFonts.playfairDisplay(
            fontSize: 24, fontWeight: FontWeight.w700,
            color: const Color(0xFF111827))),
        const SizedBox(height: 8),
        Text(
          'Your personal AI assistant for student management, '
          'communications, and academic insights.',
          textAlign: TextAlign.center,
          style: GoogleFonts.lato(fontSize: 14,
              color: const Color(0xFF6B7280), height: 1.6)),
        const SizedBox(height: 24),
        Wrap(spacing: 8, runSpacing: 8, alignment: WrapAlignment.center,
          children: [
            '📊 Student insights', '✉️ Draft emails',
            '🎯 Interventions', '📅 Calendar',
            '🧠 Support advice',
          ].map((f) => Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              color: const Color(0xFFF0FDF4),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: const Color(0xFF86EFAC)),
            ),
            child: Text(f, style: GoogleFonts.lato(fontSize: 13,
                color: const Color(0xFF15803D), fontWeight: FontWeight.w500)),
          )).toList()),
        const SizedBox(height: 28),
        GestureDetector(
          onTap: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const MentorAiChatScreen())),
          child: Container(
            width: double.infinity, height: 56,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                  colors: [Color(0xFF065F46), Color(0xFF059669)]),
              borderRadius: BorderRadius.circular(16),
              boxShadow: [BoxShadow(
                  color: AppTheme.mentorBubble.withOpacity(0.4),
                  blurRadius: 16, offset: const Offset(0, 6))],
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              const Icon(Icons.smart_toy_rounded, color: Colors.white, size: 22),
              const SizedBox(width: 10),
              Text('Open AI Chat', style: GoogleFonts.lato(
                  fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white)),
            ]),
          ),
        ),
      ]),
  );

  // ══════════════════════════════════════════════════════════
  // TAB — DOCUMENTS (Institutional Ingestion)
  // ══════════════════════════════════════════════════════════
  Widget _buildTermFilterBar() {
    final isOdd = _selectedTerm == 'odd';
    final now = DateTime.now();
    final currentYear = now.month >= 6 ? '${now.year}-${now.year + 1}' : '${now.year - 1}-${now.year}';
    final currentTerm = now.month >= 6 ? 'odd' : 'even';
    final isCurrentSession = _selectedAcademicYear == currentYear && _selectedTerm == currentTerm;

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.filter_list_rounded, size: 18, color: AppTheme.primary),
              const SizedBox(width: 8),
              Text(
                'Academic Session & Term',
                style: GoogleFonts.lato(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF111827),
                ),
              ),
              if (isCurrentSession) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFFECFDF5),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: const Color(0xFFA7F3D0)),
                  ),
                  child: Text(
                    'Active Term',
                    style: GoogleFonts.lato(
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF047857),
                    ),
                  ),
                ),
              ],
              const Spacer(),
              // Year Selector Dropdown
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                decoration: BoxDecoration(
                  color: const Color(0xFFF9FAFB),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: const Color(0xFFE5E7EB)),
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: _selectedAcademicYear,
                    isDense: true,
                    style: GoogleFonts.lato(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF1F2937),
                    ),
                    items: _academicYears.map((yr) {
                      return DropdownMenuItem<String>(
                        value: yr,
                        child: Text(yr),
                      );
                    }).toList(),
                    onChanged: (val) {
                      if (val != null) setState(() => _selectedAcademicYear = val);
                    },
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Term Switcher (Odd vs Even)
          Row(
            children: [
              Expanded(
                child: GestureDetector(
                  onTap: () => setState(() => _selectedTerm = 'odd'),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 9),
                    decoration: BoxDecoration(
                      color: isOdd ? AppTheme.primary : const Color(0xFFF9FAFB),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: isOdd ? AppTheme.primary : const Color(0xFFE5E7EB),
                      ),
                    ),
                    child: Center(
                      child: Text(
                        'Odd Semester (Jun - Dec)',
                        style: GoogleFonts.lato(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: isOdd ? Colors.white : const Color(0xFF4B5563),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: GestureDetector(
                  onTap: () => setState(() => _selectedTerm = 'even'),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 9),
                    decoration: BoxDecoration(
                      color: !isOdd ? AppTheme.primary : const Color(0xFFF9FAFB),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: !isOdd ? AppTheme.primary : const Color(0xFFE5E7EB),
                      ),
                    ),
                    child: Center(
                      child: Text(
                        'Even Semester (Jan - May)',
                        style: GoogleFonts.lato(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: !isOdd ? Colors.white : const Color(0xFF4B5563),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _documentsTab(AuthProvider auth) {
    final mentorId = auth.currentUser?.id ?? '';

    if (_docUploading) {
      return Center(
        child: Container(
          width: 480,
          margin: const EdgeInsets.all(24),
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 32),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.06),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
            ],
            border: Border.all(color: const Color(0xFFE2E8F0)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 64,
                height: 64,
                decoration: BoxDecoration(
                  color: const Color(0xFFEFF6FF),
                  shape: BoxShape.circle,
                  border: Border.all(color: const Color(0xFFBFDBFE)),
                ),
                child: const Center(
                  child: SizedBox(
                    width: 32,
                    height: 32,
                    child: CircularProgressIndicator(
                      strokeWidth: 3.0,
                      valueColor: AlwaysStoppedAnimation(Color(0xFF0284C7)),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'Publishing Academic Document',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF0F172A),
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _docUploadStatus.isNotEmpty ? _docUploadStatus : 'Processing document & building AI semantic index...',
                textAlign: TextAlign.center,
                style: GoogleFonts.lato(
                  fontSize: 13.5,
                  fontWeight: FontWeight.w600,
                  color: const Color(0xFF0284C7),
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Large files (up to 50+ pages) are extracted page-by-page and vectorized in the background without UI lag.',
                textAlign: TextAlign.center,
                style: GoogleFonts.lato(
                  fontSize: 11.5,
                  color: const Color(0xFF64748B),
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      );
    }

    // Filter documents matching the selected academic year and term
    final sessionDocs = _mentorDocs.where((doc) {
      final matchesYear = (doc.academicYear ?? '2026-2027') == _selectedAcademicYear;
      final matchesTerm = doc.term == _selectedTerm;
      return matchesYear && matchesTerm;
    }).toList();

    // Deduplicate by category: keep only the latest active document per category & scope
    final Map<String, StudentDocument> categoryMap = {};
    for (final doc in sessionDocs) {
      final key = '${doc.docType}_${doc.targetScope}_${doc.targetRollNo ?? ""}';
      if (!categoryMap.containsKey(key)) {
        categoryMap[key] = doc;
      }
    }
    final displayDocs = categoryMap.values.toList();

    return Scaffold(
      backgroundColor: Colors.transparent,
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _showMentorUploadDialog(auth),
        backgroundColor: AppTheme.primary,
        icon: const Icon(Icons.upload_file_rounded, color: Colors.white),
        label: Text('Publish Document',
            style: GoogleFonts.lato(color: Colors.white, fontWeight: FontWeight.w700)),
      ),
      body: _docsLoading
          ? _loadingView()
          : RefreshIndicator(
              onRefresh: () => _loadMentorDocuments(mentorId),
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 90),
                children: [
                  _buildTermFilterBar(),
                  _buildMentorTimetableSectionCard(auth),
                  if (displayDocs.isEmpty)
                    Container(
                      padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: const Color(0xFFE5E7EB)),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.folder_open_rounded, size: 48, color: Color(0xFF9CA3AF)),
                          const SizedBox(height: 12),
                          Text(
                            'No Documents for $_selectedAcademicYear (${_selectedTerm.toUpperCase()})',
                            style: GoogleFonts.lato(
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                              color: const Color(0xFF374151),
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            'Tap "Publish Document" to upload the official Academic Calendar, Timetable, or Syllabus for this term.',
                            textAlign: TextAlign.center,
                            style: GoogleFonts.lato(fontSize: 12, color: const Color(0xFF6B7280)),
                          ),
                        ],
                      ),
                    )
                  else ...[
                    ...displayDocs.map((doc) => _mentorDocCard(doc)).toList(),
                  ],
                ],
              ),
            ),
    );
  }

  Widget _buildMentorTimetableSectionCard(AuthProvider auth) {
    final availableClasses = MRUTimetableRepository.availableClasses;
    final activeSection = availableClasses.contains(_selectedMentorSection) ? _selectedMentorSection : 'CSE 5A';
    final mentorName = auth.currentUser?.name.trim().isNotEmpty == true
        ? auth.currentUser!.name.trim()
        : 'PRINIMA GUPTA';
    final resolvedTeacher = MRUTimetableRepository.resolveTeacher(mentorName);

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE2E8F0), width: 1.2),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF0F172A).withValues(alpha: 0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFFEFF6FF),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: const Color(0xFFDBEAFE)),
                ),
                child: const Icon(Icons.calendar_month_rounded, color: Color(0xFF0284C7), size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          'Official Class & Faculty Timetable',
                          style: GoogleFonts.lato(
                            fontSize: 16,
                            fontWeight: FontWeight.w800,
                            color: const Color(0xFF0F172A),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2.5),
                          decoration: BoxDecoration(
                            color: const Color(0xFFEFF6FF),
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(color: const Color(0xFFBFDBFE)),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.lock_outline_rounded, size: 11, color: Color(0xFF0284C7)),
                              const SizedBox(width: 3),
                              Text(
                                activeSection,
                                style: GoogleFonts.lato(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w800,
                                  color: const Color(0xFF0369A1),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Sector 43, Faridabad | aSc Timetables Online Verified',
                      style: GoogleFonts.lato(
                        fontSize: 11.5,
                        color: const Color(0xFF64748B),
                      ),
                    ),
                  ],
                ),
              ),
              GestureDetector(
                onTap: () {
                  TimetableSyncDialog.show(
                    context,
                    targetSection: activeSection,
                    userRole: 'mentor',
                    onSynced: () {
                      if (mounted) setState(() {});
                    },
                  );
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF0FDF4),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0xFFBBF7D0)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.sync_rounded, size: 13, color: Color(0xFF16A34A)),
                      const SizedBox(width: 5),
                      Text(
                        'Sync MRU',
                        style: GoogleFonts.lato(
                          fontSize: 11,
                          fontWeight: FontWeight.w800,
                          color: const Color(0xFF15803D),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Official lecture hours, 100-minute double-width lab allocations, room numbers, and faculty assignments synchronized directly from mru.edupage.org • Last Synced: ${MRUTimetableRepository.getSectionLastSyncedFormatted(activeSection)}',
            style: GoogleFonts.lato(
              fontSize: 12,
              color: const Color(0xFF64748B),
              height: 1.4,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0F172A),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    elevation: 0,
                  ),
                  onPressed: () => TimetableViewerDialog.show(
                    context,
                    className: activeSection,
                    teacherName: resolvedTeacher,
                    initialMode: 'student',
                    userRole: 'mentor',
                  ),
                  icon: const Icon(Icons.table_chart_rounded, size: 16),
                  label: Text(
                    'Class Timetable ($activeSection)',
                    style: GoogleFonts.lato(
                      fontSize: 12.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0284C7),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    elevation: 0,
                  ),
                  onPressed: () => TeacherSelectorDialog.showAndOpenViewer(
                    context,
                    className: activeSection,
                    currentTeacher: resolvedTeacher,
                    userRole: 'mentor',
                  ),
                  icon: const Icon(Icons.person_search_rounded, size: 16),
                  label: Text(
                    'Teacher Timetable 🔍',
                    style: GoogleFonts.lato(
                      fontSize: 12.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _mentorDocCard(StudentDocument doc) {
    final ti = _docTypes.firstWhere((t) => t['value'] == doc.docType,
        orElse: () => _docTypes.last);
    final color = ti['color'] as Color;
    final isClassScope = doc.targetScope == 'class';

    final queue = context.watch<DocumentQueueService>();
    final job = queue.getJob(doc.id);
    final isRunning = queue.isProcessing(doc.id);
    final isPaused = queue.isPaused(doc.id) || doc.isOcrPaused;
    final isCompleted = queue.isCompleted(doc.id) || doc.isOcrCompleted || (doc.extractedText != null && doc.extractedText!.isNotEmpty);

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: isRunning
            ? Border.all(color: const Color(0xFF38BDF8), width: 1.5)
            : Border.all(color: const Color(0xFFE2E8F0)),
        boxShadow: [
          BoxShadow(
              color: isRunning
                  ? const Color(0xFF0284C7).withOpacity(0.08)
                  : Colors.black.withOpacity(0.04),
              blurRadius: 10,
              offset: const Offset(0, 4)),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () => DocumentViewerDialog.show(
            context,
            doc,
            onUpdated: () {
              final auth = context.read<AuthProvider>();
              if (auth.currentUser != null) {
                _loadMentorDocuments(auth.currentUser!.id);
              }
            },
          ),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                        color: color.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12)),
                    child: Center(
                        child: Text(ti['emoji'] as String,
                            style: const TextStyle(fontSize: 22))),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Row(children: [
                        Expanded(
                          child: Text(doc.title,
                              style: GoogleFonts.lato(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w700,
                                  color: const Color(0xFF0F172A))),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: isClassScope
                                ? const Color(0xFFEFF6FF)
                                : const Color(0xFFFEF3C7),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                                color: isClassScope
                                    ? const Color(0xFFBFDBFE)
                                    : const Color(0xFFFDE68A)),
                          ),
                          child: Text(
                            isClassScope ? '🌐 Entire Class' : '👤 ${doc.targetRollNo ?? "Individual"}',
                            style: GoogleFonts.lato(
                                fontSize: 10.5,
                                fontWeight: FontWeight.w700,
                                color: isClassScope
                                    ? const Color(0xFF1E40AF)
                                    : const Color(0xFFB45309)),
                          ),
                        ),
                      ]),
                      const SizedBox(height: 3),
                      Text(
                        '${doc.fileName} • ${DateFormat('MMM d, yyyy').format(doc.createdAt.toLocal())}',
                        style: GoogleFonts.lato(
                            fontSize: 12, color: const Color(0xFF64748B)),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ]),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    icon: const Icon(Icons.delete_outline_rounded, color: Color(0xFFEF4444), size: 20),
                    tooltip: 'Delete Document',
                    onPressed: () => _deleteMentorDoc(doc),
                  ),
                ]),

                const SizedBox(height: 12),

                // ── Interactive OCR Queue Status & Control Bar ──
                if (isRunning) ...[
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF0F9FF),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFFBAE6FD)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation(Color(0xFF0284C7)),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Page ${job?.currentPage ?? 0} of ${job?.totalPages ?? 1} (${((job?.progress ?? 0) * 100).toInt()}%) • ${job?.statusMessage ?? "Processing..."}',
                                style: GoogleFonts.lato(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                  color: const Color(0xFF0369A1),
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            const SizedBox(width: 8),
                            InkWell(
                              onTap: () => queue.stopOcr(doc.id),
                              borderRadius: BorderRadius.circular(6),
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                decoration: BoxDecoration(
                                  color: const Color(0xFFFEF2F2),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(color: const Color(0xFFFCA5A5)),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    const Icon(Icons.stop_rounded, size: 13, color: Color(0xFFEF4444)),
                                    const SizedBox(width: 3),
                                    Text('Stop',
                                        style: GoogleFonts.lato(
                                            fontSize: 11,
                                            fontWeight: FontWeight.w700,
                                            color: const Color(0xFFEF4444))),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: LinearProgressIndicator(
                            value: job?.progress ?? 0.0,
                            minHeight: 6,
                            backgroundColor: const Color(0xFFE2E8F0),
                            valueColor: const AlwaysStoppedAnimation(Color(0xFF0284C7)),
                          ),
                        ),
                      ],
                    ),
                  ),
                ] else if (isPaused) ...[
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFFFBEB),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: const Color(0xFFFDE68A)),
                        ),
                        child: Text(
                          '⚠️ Paused at Page ${job?.currentPage ?? doc.ocrProgress?['current'] ?? 0}/${job?.totalPages ?? doc.ocrProgress?['total'] ?? "?"}',
                          style: GoogleFonts.lato(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: const Color(0xFFB45309)),
                        ),
                      ),
                      const SizedBox(width: 10),
                      ElevatedButton.icon(
                        onPressed: () => queue.resumeOcr(doc),
                        icon: const Icon(Icons.play_arrow_rounded, size: 14, color: Colors.white),
                        label: Text('Resume OCR',
                            style: GoogleFonts.lato(fontSize: 11, fontWeight: FontWeight.w700, color: Colors.white)),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF0284C7),
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
                          elevation: 0,
                        ),
                      ),
                    ],
                  ),
                ] else if (isCompleted) ...[
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFFF0FDF4),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: const Color(0xFFBBF7D0)),
                        ),
                        child: Text(
                          doc.extractedJson != null && doc.extractedJson!.isNotEmpty
                              ? '✅ Indexed (Native JSON - ${doc.extractedJson?.keys.where((k) => k.startsWith('page_')).length ?? doc.ocrProgress?['total'] ?? "All"} Pages)'
                              : '✅ Indexed',
                          style: GoogleFonts.lato(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: const Color(0xFF16A34A)),
                        ),
                      ),
                      if (doc.extractedJson != null && doc.extractedJson!.isNotEmpty) ...[
                        const SizedBox(width: 8),
                        InkWell(
                          onTap: () => _showExtractedJsonViewer(doc),
                          borderRadius: BorderRadius.circular(6),
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF3E8FF),
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(color: const Color(0xFFDDD6FE)),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(Icons.data_object_rounded, size: 13, color: Color(0xFF7C3AED)),
                                const SizedBox(width: 4),
                                Text('View JSON',
                                    style: GoogleFonts.lato(
                                        fontSize: 11,
                                        fontWeight: FontWeight.w700,
                                        color: const Color(0xFF7C3AED))),
                              ],
                            ),
                          ),
                        ),
                      ],
                      const Spacer(),
                      TextButton.icon(
                        onPressed: () => queue.startOcr(doc, forceReindex: true),
                        icon: const Icon(Icons.refresh_rounded, size: 12, color: Color(0xFF64748B)),
                        label: Text('Re-index', style: GoogleFonts.lato(fontSize: 11, color: const Color(0xFF64748B))),
                      ),
                    ],
                  ),
                ] else ...[
                  Row(
                    children: [
                      ElevatedButton.icon(
                        onPressed: () => queue.startOcr(doc),
                        icon: const Icon(Icons.bolt_rounded, size: 16, color: Colors.white),
                        label: Text('⚡ Start OCR',
                            style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w800, color: Colors.white)),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFF59E0B),
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          elevation: 1,
                        ),
                      ),
                      const SizedBox(width: 10),
                      Text('Background native JSON extraction',
                          style: GoogleFonts.lato(fontSize: 11.5, color: const Color(0xFF94A3B8))),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _showExtractedJsonViewer(StudentDocument doc) async {
    StudentDocument currentDoc = doc;
    try {
      final fresh = await SupabaseService.getDocumentWithContent(doc.id);
      if (fresh != null && fresh.extractedJson != null && fresh.extractedJson!.isNotEmpty) {
        currentDoc = fresh;
      }
    } catch (_) {}

    if (!mounted) return;

    final extracted = currentDoc.extractedJson ?? {};
    final pages = extracted.keys.where((k) => k.startsWith('page_')).toList();
    pages.sort((a, b) {
      final na = int.tryParse(a.replaceFirst('page_', '')) ?? 0;
      final nb = int.tryParse(b.replaceFirst('page_', '')) ?? 0;
      return na.compareTo(nb);
    });

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return DraggableScrollableSheet(
          initialChildSize: 0.85,
          minChildSize: 0.5,
          maxChildSize: 0.95,
          expand: false,
          builder: (context, scrollController) {
            return DefaultTabController(
              length: 2,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Container(
                        width: 40,
                        height: 4,
                        decoration: BoxDecoration(
                          color: const Color(0xFFCBD5E1),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: const Color(0xFFF3E8FF),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.data_object_rounded, color: Color(0xFF7C3AED), size: 22),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                doc.title,
                                style: GoogleFonts.playfairDisplay(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w700,
                                  color: const Color(0xFF0F172A),
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                              Text(
                                '${pages.length} Pages Indexed (Native JSON) • ${doc.docType.toUpperCase()}',
                                style: GoogleFonts.lato(fontSize: 12, color: const Color(0xFF64748B)),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.close_rounded),
                          onPressed: () => Navigator.pop(ctx),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    TabBar(
                      labelColor: AppTheme.primary,
                      unselectedLabelColor: const Color(0xFF64748B),
                      indicatorColor: AppTheme.primary,
                      tabs: const [
                        Tab(text: 'Structured Records', icon: Icon(Icons.table_chart_rounded, size: 18)),
                        Tab(text: 'Raw JSON', icon: Icon(Icons.code_rounded, size: 18)),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Expanded(
                      child: TabBarView(
                        children: [
                          _buildStructuredRecordsTab(pages, extracted, scrollController),
                          _buildRawJsonTab(extracted, scrollController),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  void _showStudentDetailDialog(Map<String, dynamic> student, Map<String, dynamic> pageData) {
    final name = (student['name'] ?? 'Unknown Student').toString();
    final rollNo = (student['roll_no'] ?? student['roll_number'] ?? 'N/A').toString();
    final fatherName = (student['father_name'] ?? 'N/A').toString();
    final sgpaStr = (student['sgpa']?.toString() ?? 'N/A');
    final resultStr = (student['result'] ?? ((student['sgpa'] is num && (student['sgpa'] as num) >= 4.0) ? 'PASS' : 'N/A')).toString().toUpperCase();

    final university = (pageData['university_name'] ?? pageData['institution'] ?? 'Manav Rachna University').toString();
    final school = (pageData['school_name'] ?? 'School of Engineering').toString();
    final programme = (pageData['programme_name'] ?? pageData['programme'] ?? 'B.Tech Computer Science & Engineering').toString();
    final semester = (pageData['semester'] ?? 'Semester 2').toString();
    final session = (pageData['result_session'] ?? pageData['examination']?['session'] ?? 'MAY-2024').toString();

    final rawCourses = (pageData['courses'] as List?)?.map((e) => Map<String, dynamic>.from(e as Map)).toList() ?? [];
    final Map<String, Map<String, dynamic>> courseMap = {};
    for (final c in rawCourses) {
      final code = (c['code'] ?? '').toString().trim();
      if (code.isNotEmpty) {
        courseMap[code] = c;
      }
    }

    final grades = (student['grades'] as Map<String, dynamic>?) ?? {};

    Color getGradeColor(String grade) {
      final g = grade.trim().toUpperCase();
      if (['O', 'A+', 'A', 'PASS'].contains(g)) return const Color(0xFF059669); // Emerald
      if (['B+', 'B'].contains(g)) return const Color(0xFF2563EB); // Blue
      if (['C', 'P'].contains(g)) return const Color(0xFFD97706); // Amber
      if (['F', 'DB', 'AB', 'FAIL', 'R'].contains(g)) return const Color(0xFFDC2626); // Red
      return const Color(0xFF64748B);
    }

    showDialog(
      context: context,
      builder: (ctx) {
        return Dialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          backgroundColor: Colors.white,
          insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          child: Container(
            width: 620,
            constraints: const BoxConstraints(maxHeight: 700),
            padding: const EdgeInsets.all(22),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                // Top Header: Institution & Student Info
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: const Color(0xFFEFF6FF),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(Icons.school_rounded, color: Color(0xFF1D4ED8), size: 28),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            name,
                            style: GoogleFonts.playfairDisplay(
                              fontSize: 20,
                              fontWeight: FontWeight.w800,
                              color: const Color(0xFF0F172A),
                            ),
                          ),
                          const SizedBox(height: 3),
                          Wrap(
                            spacing: 8,
                            runSpacing: 4,
                            crossAxisAlignment: WrapCrossAlignment.center,
                            children: [
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                decoration: BoxDecoration(
                                  color: const Color(0xFFF1F5F9),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(color: const Color(0xFFCBD5E1)),
                                ),
                                child: Text(
                                  'Roll No: $rollNo',
                                  style: GoogleFonts.lato(fontSize: 11.5, fontWeight: FontWeight.w700, color: const Color(0xFF334155)),
                                ),
                              ),
                              if (fatherName != 'N/A' && fatherName.isNotEmpty)
                                Text(
                                  'Father: $fatherName',
                                  style: GoogleFonts.lato(fontSize: 11.5, color: const Color(0xFF64748B)),
                                ),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '$university • $school\n$programme • $semester ($session)',
                            style: GoogleFonts.lato(fontSize: 11, color: const Color(0xFF94A3B8), height: 1.3),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close_rounded, color: Color(0xFF64748B)),
                      onPressed: () => Navigator.pop(ctx),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                const Divider(height: 1, color: Color(0xFFE2E8F0)),
                const SizedBox(height: 14),

                // Table Header
                Text(
                  'COURSE PERFORMANCE BREAKDOWN',
                  style: GoogleFonts.lato(fontSize: 11, fontWeight: FontWeight.w800, color: const Color(0xFF475569), letterSpacing: 0.8),
                ),
                const SizedBox(height: 8),

                // Scrollable Course List
                Flexible(
                  child: grades.isEmpty
                      ? Center(
                          child: Padding(
                            padding: const EdgeInsets.all(20),
                            child: Text('No individual course marks available.', style: GoogleFonts.lato(color: Colors.grey)),
                          ),
                        )
                      : ListView.separated(
                          shrinkWrap: true,
                          itemCount: grades.length,
                          separatorBuilder: (_, __) => const SizedBox(height: 6),
                          itemBuilder: (context, cIdx) {
                            final code = grades.keys.elementAt(cIdx);
                            final gradeVal = grades[code]?.toString() ?? 'N/A';
                            final courseMeta = courseMap[code];
                            final title = (courseMeta?['title'] ?? code).toString();
                            final credits = courseMeta?['credits']?.toString() ?? '4';
                            final gradeColor = getGradeColor(gradeVal);

                            return Container(
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                              decoration: BoxDecoration(
                                color: const Color(0xFFF8FAFC),
                                borderRadius: BorderRadius.circular(10),
                                border: Border.all(color: const Color(0xFFE2E8F0)),
                              ),
                              child: Row(
                                children: [
                                  Container(
                                    width: 32,
                                    height: 32,
                                    alignment: Alignment.center,
                                    decoration: BoxDecoration(
                                      color: const Color(0xFFE2E8F0),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Text(
                                      '${cIdx + 1}',
                                      style: GoogleFonts.lato(fontSize: 11, fontWeight: FontWeight.w700, color: const Color(0xFF475569)),
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          title,
                                          style: GoogleFonts.lato(fontSize: 12.5, fontWeight: FontWeight.w700, color: const Color(0xFF1E293B)),
                                          maxLines: 2,
                                          overflow: TextOverflow.ellipsis,
                                        ),
                                        Text(
                                          'Code: $code • $credits Credits',
                                          style: GoogleFonts.lato(fontSize: 11, color: const Color(0xFF64748B)),
                                        ),
                                      ],
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                    decoration: BoxDecoration(
                                      color: gradeColor.withOpacity(0.12),
                                      borderRadius: BorderRadius.circular(8),
                                      border: Border.all(color: gradeColor.withOpacity(0.35)),
                                    ),
                                    child: Text(
                                      gradeVal,
                                      style: GoogleFonts.lato(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w800,
                                        color: gradeColor,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),
                ),

                const SizedBox(height: 16),
                const Divider(height: 1, color: Color(0xFFE2E8F0)),
                const SizedBox(height: 14),

                // Bottom Footer: Result status & SGPA
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: (resultStr == 'PASS') ? const Color(0xFFDCFCE7) : const Color(0xFFFEE2E2),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: (resultStr == 'PASS') ? const Color(0xFF86EFAC) : const Color(0xFFFCA5A5)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            (resultStr == 'PASS') ? Icons.check_circle_rounded : Icons.warning_rounded,
                            size: 15,
                            color: (resultStr == 'PASS') ? const Color(0xFF16A34A) : const Color(0xFFDC2626),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            'RESULT: $resultStr',
                            style: GoogleFonts.lato(
                              fontSize: 11.5,
                              fontWeight: FontWeight.w800,
                              color: (resultStr == 'PASS') ? const Color(0xFF15803D) : const Color(0xFFB91C1C),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF0F172A), Color(0xFF1E293B)],
                        ),
                        borderRadius: BorderRadius.circular(10),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.08),
                            blurRadius: 8,
                            offset: const Offset(0, 3),
                          ),
                        ],
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            'SGPA: ',
                            style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w600, color: const Color(0xFF94A3B8)),
                          ),
                          Text(
                            sgpaStr,
                            style: GoogleFonts.lato(
                              fontSize: 15,
                              fontWeight: FontWeight.w900,
                              color: const Color(0xFFF59E0B), // Gold
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildStructuredRecordsTab(List<String> pages, Map<String, dynamic> extracted, ScrollController scrollController) {
    if (pages.isEmpty) {
      return Center(
        child: Text('No structured JSON records found.', style: GoogleFonts.lato(color: Colors.grey)),
      );
    }

    return ListView.builder(
      controller: scrollController,
      itemCount: pages.length,
      itemBuilder: (ctx, idx) {
        final pageKey = pages[idx];
        final pageData = extracted[pageKey] as Map<String, dynamic>? ?? {};
        final students = (pageData['students'] as List?)?.map((e) => Map<String, dynamic>.from(e as Map)).toList() ?? [];
        final courses = (pageData['courses'] as List?)?.map((e) => Map<String, dynamic>.from(e as Map)).toList() ?? [];

        final progName = (pageData['programme_name'] ?? pageData['programme'] ?? pageData['examination_info']?['programme'] ?? '').toString();
        final semName = (pageData['semester'] ?? pageData['examination_info']?['semester'] ?? '').toString();
        final batchName = (pageData['batch'] ?? pageData['examination_info']?['batch'] ?? '').toString();

        final subtitleParts = <String>[];
        if (progName.isNotEmpty) subtitleParts.add(progName);
        if (semName.isNotEmpty) subtitleParts.add('Sem: $semName');
        if (batchName.isNotEmpty) subtitleParts.add('Batch: $batchName');

        return Card(
          margin: const EdgeInsets.only(bottom: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14), side: const BorderSide(color: Color(0xFFE2E8F0))),
          elevation: 0,
          color: const Color(0xFFF8FAFC),
          child: ExpansionTile(
            initiallyExpanded: idx == 0,
            title: Text(
              '${pageKey.toUpperCase().replaceAll('_', ' ')} (${students.isNotEmpty ? "${students.length} Students" : "${courses.length} Courses"})',
              style: GoogleFonts.lato(fontSize: 14, fontWeight: FontWeight.w700, color: const Color(0xFF1E293B)),
            ),
            subtitle: subtitleParts.isNotEmpty
                ? Text(subtitleParts.join(' • '),
                    style: GoogleFonts.lato(fontSize: 11.5, color: const Color(0xFF64748B)),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis)
                : null,
            children: [
              if (courses.isNotEmpty) ...[
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Courses Evaluated (${courses.length}):',
                          style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700, color: const Color(0xFF334155))),
                      const SizedBox(height: 6),
                      Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: courses.map((c) => Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: const Color(0xFFEFF6FF),
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(color: const Color(0xFFBFDBFE)),
                          ),
                          child: Text(
                            '${c["code"]}: ${c["title"]}',
                            style: GoogleFonts.lato(fontSize: 11, fontWeight: FontWeight.w600, color: const Color(0xFF1E40AF)),
                          ),
                        )).toList(),
                      ),
                    ],
                  ),
                ),
              ],
              if (students.isNotEmpty) ...[
                Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    children: students.map((s) {
                      final grades = s['grades'] as Map<String, dynamic>? ?? {};
                      final studentResult = (s['result'] ?? (s['sgpa'] != null && (s['sgpa'] as num) >= 4.0 ? 'PASS' : 'N/A')).toString().toUpperCase();
                      final isPass = studentResult == 'PASS';

                      return InkWell(
                        onTap: () => _showStudentDetailDialog(s, pageData),
                        borderRadius: BorderRadius.circular(10),
                        child: Container(
                          margin: const EdgeInsets.only(bottom: 8),
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(color: const Color(0xFFE2E8F0)),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.02),
                                blurRadius: 4,
                                offset: const Offset(0, 1),
                              ),
                            ],
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Expanded(
                                    child: Text(
                                      s['name'] ?? 'Unknown',
                                      style: GoogleFonts.lato(fontSize: 13.5, fontWeight: FontWeight.w800, color: const Color(0xFF0F172A)),
                                    ),
                                  ),
                                  if (studentResult != 'N/A')
                                    Container(
                                      margin: const EdgeInsets.only(right: 8),
                                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: isPass ? const Color(0xFFDCFCE7) : const Color(0xFFFEE2E2),
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: Text(
                                        studentResult,
                                        style: GoogleFonts.lato(
                                          fontSize: 10,
                                          fontWeight: FontWeight.w800,
                                          color: isPass ? const Color(0xFF16A34A) : const Color(0xFFDC2626),
                                        ),
                                      ),
                                    ),
                                  Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                    decoration: BoxDecoration(
                                      color: const Color(0xFFFEF3C7),
                                      borderRadius: BorderRadius.circular(6),
                                    ),
                                    child: Text(
                                      'SGPA: ${s['sgpa'] ?? "N/A"}',
                                      style: GoogleFonts.lato(fontSize: 11.5, fontWeight: FontWeight.w800, color: const Color(0xFFB45309)),
                                    ),
                                  ),
                                  const SizedBox(width: 4),
                                  const Icon(Icons.chevron_right_rounded, size: 18, color: Color(0xFF94A3B8)),
                                ],
                              ),
                              const SizedBox(height: 3),
                              Row(
                                children: [
                                  Text(
                                    'Roll No: ${s['roll_no'] ?? s['roll_number'] ?? "N/A"}',
                                    style: GoogleFonts.lato(fontSize: 11.5, fontWeight: FontWeight.w600, color: const Color(0xFF475569)),
                                  ),
                                  if (s['father_name'] != null && s['father_name'].toString().isNotEmpty) ...[
                                    const Text(' • ', style: TextStyle(color: Color(0xFFCBD5E1))),
                                    Expanded(
                                      child: Text(
                                        'Father: ${s['father_name']}',
                                        style: GoogleFonts.lato(fontSize: 11, color: const Color(0xFF64748B)),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  ],
                                ],
                              ),
                              if (grades.isNotEmpty) ...[
                                const SizedBox(height: 6),
                                Text(
                                  'Grades: ' + grades.entries.map((e) => '${e.key}: ${e.value}').join(', '),
                                  style: GoogleFonts.lato(fontSize: 11, color: const Color(0xFF64748B)),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ],
                              const SizedBox(height: 4),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.end,
                                children: [
                                  Text(
                                    'Tap to view scorecard',
                                    style: GoogleFonts.lato(fontSize: 10.5, fontWeight: FontWeight.w600, color: const Color(0xFF2563EB)),
                                  ),
                                  const Icon(Icons.arrow_forward_rounded, size: 12, color: Color(0xFF2563EB)),
                                ],
                              ),
                            ],
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ] else ...[
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                    const JsonEncoder.withIndent('  ').convert(pageData),
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
                  ),
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  Widget _buildRawJsonTab(Map<String, dynamic> extracted, ScrollController scrollController) {
    final prettyJson = const JsonEncoder.withIndent('  ').convert(extracted);
    return Column(
      children: [
        Align(
          alignment: Alignment.centerRight,
          child: TextButton.icon(
            onPressed: () {
              Clipboard.setData(ClipboardData(text: prettyJson));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Raw JSON copied to clipboard!'),
                  duration: Duration(seconds: 2),
                  backgroundColor: Color(0xFF0F172A),
                ),
              );
            },
            icon: const Icon(Icons.copy_rounded, size: 14),
            label: const Text('Copy JSON'),
          ),
        ),
        Expanded(
          child: SingleChildScrollView(
            controller: scrollController,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: const Color(0xFF0F172A),
                borderRadius: BorderRadius.circular(12),
              ),
              child: SelectableText(
                prettyJson,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 11, color: Color(0xFF38BDF8)),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Future<void> _deleteMentorDoc(StudentDocument doc) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Delete Document',
            style: GoogleFonts.playfairDisplay(fontSize: 18, fontWeight: FontWeight.w700)),
        content: Text(
            'Are you sure you want to delete "${doc.title}"?\nThis removes the document and its AI vector index.',
            style: GoogleFonts.lato(fontSize: 13)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirm == true) {
      final auth = context.read<AuthProvider>();
      await SupabaseService.deleteDocument(doc.id);
      if (auth.currentUser != null) {
        await _loadMentorDocuments(auth.currentUser!.id);
      }
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('Document deleted'),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ));
    }
  }

  Future<void> _showMentorUploadDialog(AuthProvider auth) async {
    String selectedType = 'timetable';
    String scope = 'class'; // 'class' or 'individual'
    String modalYear = _selectedAcademicYear;
    String modalTerm = _selectedTerm;
    final titleCtrl = TextEditingController();
    final rollCtrl = TextEditingController();

    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setS) => Container(
          padding: EdgeInsets.only(bottom: MediaQuery.of(ctx).viewInsets.bottom),
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
            child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(2)),
                ),
              ),
              const SizedBox(height: 16),
              Text('Publish Academic Document',
                  style: GoogleFonts.playfairDisplay(fontSize: 18, fontWeight: FontWeight.w700, color: const Color(0xFF111827))),
              Text('Upload official materials to the institutional repository',
                  style: GoogleFonts.lato(fontSize: 12, color: const Color(0xFF6B7280))),
              const SizedBox(height: 16),

              Text('Academic Session & Term',
                  style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700, color: const Color(0xFF374151))),
              const SizedBox(height: 8),
              Row(children: [
                Expanded(
                  flex: 4,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF9FAFB),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: const Color(0xFFD1D5DB)),
                    ),
                    child: DropdownButtonHideUnderline(
                      child: DropdownButton<String>(
                        value: modalYear,
                        isExpanded: true,
                        items: _academicYears.map((y) => DropdownMenuItem(value: y, child: Text(y, style: GoogleFonts.lato(fontSize: 12.5)))).toList(),
                        onChanged: (v) { if (v != null) setS(() => modalYear = v); },
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  flex: 5,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF9FAFB),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: const Color(0xFFD1D5DB)),
                    ),
                    child: DropdownButtonHideUnderline(
                      child: DropdownButton<String>(
                        value: modalTerm,
                        isExpanded: true,
                        items: const [
                          DropdownMenuItem(value: 'odd', child: Text('Odd Sem (Jun-Dec)', style: TextStyle(fontSize: 12))),
                          DropdownMenuItem(value: 'even', child: Text('Even Sem (Jan-May)', style: TextStyle(fontSize: 12))),
                        ],
                        onChanged: (v) { if (v != null) setS(() => modalTerm = v); },
                      ),
                    ),
                  ),
                ),
              ]),
              const SizedBox(height: 16),

              Text('Scope', style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700, color: const Color(0xFF374151))),
              const SizedBox(height: 8),
              Row(children: [
                Expanded(
                  child: ChoiceChip(
                    label: const Center(child: Text('🌐 Entire Class')),
                    selected: scope == 'class',
                    onSelected: (val) { if (val) setS(() => scope = 'class'); },
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ChoiceChip(
                    label: const Center(child: Text('👤 Individual Student')),
                    selected: scope == 'individual',
                    onSelected: (val) { if (val) setS(() => scope = 'individual'); },
                  ),
                ),
              ]),

              if (scope == 'individual') ...[
                const SizedBox(height: 12),
                TextField(
                  controller: rollCtrl,
                  decoration: InputDecoration(
                    labelText: 'Student Roll Number',
                    hintText: 'e.g. 2K24CSUN01015',
                    filled: true,
                    fillColor: const Color(0xFFF9FAFB),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ],

              const SizedBox(height: 16),
              Text('Document Category',
                  style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700, color: const Color(0xFF374151))),
              const SizedBox(height: 8),
              SizedBox(
                height: 80,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: _docTypes.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 8),
                  itemBuilder: (_, i) {
                    final t = _docTypes[i];
                    final sel = selectedType == t['value'];
                    return GestureDetector(
                      onTap: () => setS(() => selectedType = t['value'] as String),
                      child: Container(
                        width: 85,
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: sel ? (t['color'] as Color).withOpacity(0.15) : const Color(0xFFF9FAFB),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                              color: sel ? t['color'] as Color : const Color(0xFFE5E7EB),
                              width: sel ? 2 : 1),
                        ),
                        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                          Text(t['emoji'] as String, style: const TextStyle(fontSize: 20)),
                          const SizedBox(height: 4),
                          Text(t['label'] as String,
                              textAlign: TextAlign.center,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.lato(
                                  fontSize: 9,
                                  fontWeight: FontWeight.w700,
                                  color: sel ? t['color'] as Color : const Color(0xFF4B5563))),
                        ]),
                      ),
                    );
                  },
                ),
              ),

              const SizedBox(height: 14),
              if (selectedType == 'academic_calendar') ...[
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF3E8FF),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: const Color(0xFFDDD6FE)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline_rounded, color: Color(0xFF7C3AED), size: 18),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Academic Calendars apply to the whole college/term. No subject name needed.',
                          style: GoogleFonts.lato(fontSize: 12, color: const Color(0xFF5B21B6), fontWeight: FontWeight.w500),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: titleCtrl,
                  decoration: InputDecoration(
                    labelText: 'Calendar Title (Optional)',
                    hintText: 'e.g. Academic Calendar $modalYear (${modalTerm.toUpperCase()} Sem)',
                    filled: true,
                    fillColor: const Color(0xFFF9FAFB),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ] else if (selectedType == 'syllabus') ...[
                TextField(
                  controller: titleCtrl,
                  decoration: InputDecoration(
                    labelText: 'Subject / Course Name (Required)',
                    hintText: 'e.g. Data Structures & Algorithms (CS201)',
                    filled: true,
                    fillColor: const Color(0xFFF9FAFB),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ] else if (selectedType == 'timetable') ...[
                TextField(
                  controller: titleCtrl,
                  decoration: InputDecoration(
                    labelText: 'Timetable Description (Optional)',
                    hintText: 'e.g. CSE Weekly Timetable',
                    filled: true,
                    fillColor: const Color(0xFFF9FAFB),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ] else ...[
                TextField(
                  controller: titleCtrl,
                  decoration: InputDecoration(
                    labelText: 'Document Title',
                    hintText: 'e.g. Mid-Term Marksheet / Circular',
                    filled: true,
                    fillColor: const Color(0xFFF9FAFB),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ],

              const SizedBox(height: 18),
              Row(children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.pop(ctx);
                      _pickAndPublish(
                        auth: auth,
                        source: ImageSource.gallery,
                        docType: selectedType,
                        scope: scope,
                        title: titleCtrl.text.trim(),
                        targetRollNo: rollCtrl.text.trim(),
                        academicYear: modalYear,
                        semester: modalTerm,
                      );
                    },
                    icon: const Icon(Icons.photo_library_rounded),
                    label: const Text('Image'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      Navigator.pop(ctx);
                      _pickAndPublishFile(
                        auth: auth,
                        docType: selectedType,
                        scope: scope,
                        title: titleCtrl.text.trim(),
                        targetRollNo: rollCtrl.text.trim(),
                        academicYear: modalYear,
                        semester: modalTerm,
                      );
                    },
                    icon: const Icon(Icons.upload_file_rounded, color: Colors.white),
                    label: const Text('File / PDF', style: TextStyle(color: Colors.white)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.primary,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                  ),
                ),
              ]),
            ]),
          ),
        ),
      ),
    );
  }

  Future<void> _pickAndPublish({
    required AuthProvider auth,
    required ImageSource source,
    required String docType,
    required String scope,
    required String title,
    required String targetRollNo,
    String academicYear = '2026-2027',
    String semester = 'odd',
  }) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: source, imageQuality: 85, maxWidth: 1600);
    if (picked == null) return;

    final bytes = await picked.readAsBytes();
    final String defaultTitle = docType == 'academic_calendar'
        ? 'Academic Calendar'
        : docType == 'timetable'
            ? 'Class Timetable'
            : picked.name;

    await _executeIngestionPipeline(
      auth: auth,
      docType: docType,
      scope: scope,
      title: title.isNotEmpty ? title : defaultTitle,
      fileName: picked.name,
      mimeType: 'image/jpeg',
      fileSize: bytes.length,
      contentBase64: base64Encode(bytes),
      rawBytes: bytes,
      targetRollNo: targetRollNo.isNotEmpty ? targetRollNo : null,
      academicYear: academicYear,
      semester: semester,
    );
  }

  Future<void> _pickAndPublishFile({
    required AuthProvider auth,
    required String docType,
    required String scope,
    required String title,
    required String targetRollNo,
    String academicYear = '2026-2027',
    String semester = 'odd',
  }) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'jpg', 'jpeg', 'png', 'txt'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;
    final bytes = file.bytes;
    if (bytes == null) return;

    final ext = file.extension?.toLowerCase() ?? 'bin';
    final mimeType = ext == 'pdf' ? 'application/pdf' : ext == 'txt' ? 'text/plain' : 'image/jpeg';

    final String defaultTitle = docType == 'academic_calendar'
        ? 'Academic Calendar'
        : docType == 'timetable'
            ? 'Class Timetable'
            : file.name;

    // Only compute base64 if small file (< 1.5 MB) or non-PDF to save memory and avoid UI freeze
    String? contentBase64;
    if (file.size <= 1572864 && ext != 'pdf') {
      contentBase64 = base64Encode(bytes);
    }

    await _executeIngestionPipeline(
      auth: auth,
      docType: docType,
      scope: scope,
      title: title.isNotEmpty ? title : defaultTitle,
      fileName: file.name,
      mimeType: mimeType,
      fileSize: file.size,
      contentBase64: contentBase64,
      rawBytes: bytes,
      targetRollNo: targetRollNo.isNotEmpty ? targetRollNo : null,
      academicYear: academicYear,
      semester: semester,
    );
  }

  Future<void> _executeIngestionPipeline({
    required AuthProvider auth,
    required String docType,
    required String scope,
    required String title,
    required String fileName,
    required String mimeType,
    required int fileSize,
    String? contentBase64,
    Uint8List? rawBytes,
    String? targetRollNo,
    String academicYear = '2026-2027',
    String semester = 'odd',
  }) async {
    final mentorId = auth.currentUser?.id ?? '';
    setState(() {
      _docUploading = true;
      _docUploadStatus = 'Uploading document to cloud storage...';
    });

    try {
      String extractedText = '';

      // Fast check: Extract digital text if available in PDF (runs in milliseconds)
      if (mimeType == 'application/pdf' && rawBytes != null && rawBytes.isNotEmpty) {
        try {
          final isDigital = await PDFExtractionService.isDigitalPdf(rawBytes);
          if (isDigital) {
            extractedText = await PDFExtractionService.extractTextFromPdfBytes(rawBytes);
          }
        } catch (_) {}
      }

      // Upload Document to Supabase
      final doc = await SupabaseService.uploadDocument(
        uploadedBy: mentorId,
        docType: docType,
        title: title,
        fileName: fileName,
        mimeType: mimeType,
        fileSize: fileSize,
        contentBase64: contentBase64,
        rawBytes: rawBytes,
        extractedText: extractedText.isNotEmpty ? extractedText : null,
        targetScope: scope,
        targetRollNo: targetRollNo,
        academicYear: academicYear,
        semester: semester,
      );

      await _loadMentorDocuments(mentorId);
      setState(() {
        _selectedAcademicYear = academicYear;
        _selectedTerm = semester;
        _docUploading = false;
        _docUploadStatus = '';
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('✅ "$title" published! Tap "⚡ Start OCR" to run AI indexing in the background.'),
            action: SnackBarAction(
              label: 'Start OCR',
              textColor: const Color(0xFFF59E0B),
              onPressed: () {
                context.read<DocumentQueueService>().startOcr(doc);
              },
            ),
            behavior: SnackBarBehavior.floating,
            duration: const Duration(seconds: 6),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _docUploading = false;
        _docUploadStatus = '';
      });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Publish error: $e'),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ));
    }
  }

  // ── Helpers ───────────────────────────────────────────────
  Widget _loadingView() => const Center(
    child: CircularProgressIndicator(
        valueColor: AlwaysStoppedAnimation(AppTheme.primary)));

  Widget _emptyView(IconData icon, String title, String subtitle) =>
      Center(child: Padding(padding: const EdgeInsets.all(40),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Container(
            width: 80, height: 80,
            decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.08),
                shape: BoxShape.circle),
            child: Icon(icon, size: 38,
                color: AppTheme.primary.withOpacity(0.4))),
          const SizedBox(height: 16),
          Text(title, style: GoogleFonts.playfairDisplay(
              fontSize: 20, fontWeight: FontWeight.w700,
              color: const Color(0xFF111827))),
          const SizedBox(height: 8),
          Text(subtitle, textAlign: TextAlign.center,
              style: GoogleFonts.lato(fontSize: 13,
                  color: const Color(0xFF6B7280), height: 1.6)),
        ]),
      ));
}
