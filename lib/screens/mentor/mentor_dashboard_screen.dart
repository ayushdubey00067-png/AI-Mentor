// lib/screens/mentor/mentor_dashboard_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/auth_provider.dart';
import '../../services/chat_provider.dart';
import '../../services/supabase_service.dart';
import '../../utils/app_theme.dart';
import '../auth_screen.dart';
import 'mentor_chat_view_screen.dart';
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

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 4, vsync: this);
    _tab.addListener(() => setState(() {}));
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadAll());
  }

  @override
  void dispose() { _tab.dispose(); super.dispose(); }

  Future<void> _loadAll() async {
    final auth  = context.read<AuthProvider>();
    final chat  = context.read<ChatProvider>();
    final email = auth.currentUser?.email ?? '';
    chat.setCurrentUser(auth.currentUser);
    await chat.loadMentorDashboard(email);
    await chat.loadProgressReports(email);
    await _loadIssues(email);
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
    if (chat.myStudents.isEmpty) return _emptyView(
      Icons.school_outlined,
      'No students yet',
      'Students appear here when they\nregister using your email address',
    );

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
            value: selectedStatus,
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
                if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(
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
    if (chat.progressReports.isEmpty) return _emptyView(
        Icons.insights_outlined, 'No data yet',
        'Reports appear once students start chatting');

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