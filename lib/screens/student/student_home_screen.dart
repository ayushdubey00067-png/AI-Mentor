// lib/screens/student/student_home_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/auth_provider.dart';
import '../../services/chat_provider.dart';
import '../../utils/app_theme.dart';
import '../auth_screen.dart';
import 'student_chat_screen.dart';
import 'student_documents_screen.dart';
import 'student_issue_screen.dart';

class StudentHomeScreen extends StatefulWidget {
  const StudentHomeScreen({super.key});
  @override
  State<StudentHomeScreen> createState() => _StudentHomeScreenState();
}

class _StudentHomeScreenState extends State<StudentHomeScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tab;
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 3, vsync: this);
    _tab.addListener(() => setState(() {}));
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  @override
  void dispose() { _tab.dispose(); super.dispose(); }

  Future<void> _load() async {
    final auth = context.read<AuthProvider>();
    if (auth.currentUser == null) return;
    final chat = context.read<ChatProvider>();
    chat.setCurrentUser(auth.currentUser);
    await chat.loadStudentConversations(auth.currentUser!.id);
  }

  Future<void> _startNewChat() async {
    final auth = context.read<AuthProvider>();
    if (auth.currentUser == null) return;
    final chat = context.read<ChatProvider>();
    await chat.startNewConversation(
      auth.currentUser!.id,
      mentorEmail: auth.currentUser!.mentorEmail,
    );
    if (!mounted) return;
    Navigator.of(context)
        .push(MaterialPageRoute(builder: (_) => const StudentChatScreen()));
  }

  Future<void> _openConversation(ConversationModel conv) async {
    final auth = context.read<AuthProvider>();
    await context.read<ChatProvider>()
        .loadConversation(conv.id, auth.currentUser!.id);
    if (!mounted) return;
    // Close drawer first
    Navigator.of(context).pop();
    Navigator.of(context)
        .push(MaterialPageRoute(builder: (_) => const StudentChatScreen()));
  }

  Future<void> _deleteConv(ConversationModel conv) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text('Delete Chat?', style: GoogleFonts.playfairDisplay(fontWeight: FontWeight.w700)),
        content: Text('This will permanently delete this conversation and all its messages.',
            style: GoogleFonts.lato(fontSize: 14)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text('Cancel', style: GoogleFonts.lato(color: Colors.grey)),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.redAccent,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            child: Text('Delete', style: GoogleFonts.lato(color: Colors.white, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );

    if (confirmed == true && mounted) {
      await context.read<ChatProvider>().deleteConversation(conv.id);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Conversation deleted'), duration: Duration(seconds: 2)),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final chat = context.watch<ChatProvider>();

    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: const Color(0xFFF3F5FB),
      // ── LEFT DRAWER — Chat History ─────────────────────────
      drawer: _buildDrawer(auth, chat),
      body: Column(children: [
        _buildHeader(auth, chat),
        _buildTabBar(chat),
        Expanded(child: TabBarView(controller: _tab, children: [
          _aiChatTab(chat),
          const StudentDocumentsScreen(),
          const StudentIssueScreen(),
        ])),
      ]),
      floatingActionButton: _tab.index == 0
          ? _buildFAB()
          : null,
    );
  }

  // ══════════════════════════════════════════════════════════
  // DRAWER — Chat History
  // ══════════════════════════════════════════════════════════
  Widget _buildDrawer(AuthProvider auth, ChatProvider chat) {
    final user = auth.currentUser;
    final initials = _getInitials(user?.name ?? 'S');

    return Drawer(
      width: MediaQuery.of(context).size.width * 0.82,
      child: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF0F1C3F), Color(0xFF1A2B5F)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: Column(children: [
            // ── Drawer Header ─────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 0),
              child: Column(children: [
                Row(children: [
                  Container(
                    width: 56, height: 56,
                    decoration: BoxDecoration(
                      color: AppTheme.accent,
                      shape: BoxShape.circle,
                      border: Border.all(
                          color: Colors.white.withOpacity(0.3), width: 2),
                      boxShadow: [BoxShadow(
                          color: AppTheme.accent.withOpacity(0.5),
                          blurRadius: 12, offset: const Offset(0, 4))],
                    ),
                    child: Center(child: Text(initials,
                        style: GoogleFonts.playfairDisplay(
                            fontSize: 20, fontWeight: FontWeight.w700,
                            color: Colors.white))),
                  ),
                  const SizedBox(width: 14),
                  Expanded(child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(user?.name ?? 'Student',
                          style: GoogleFonts.playfairDisplay(
                              fontSize: 16, fontWeight: FontWeight.w700,
                              color: Colors.white),
                          overflow: TextOverflow.ellipsis),
                      Text(user?.email ?? '',
                          style: GoogleFonts.lato(
                              fontSize: 12, color: Colors.white60),
                          overflow: TextOverflow.ellipsis),
                    ],
                  )),
                  // Close button
                  GestureDetector(
                    onTap: () => Navigator.of(context).pop(),
                    child: Container(
                      width: 34, height: 34,
                      decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(10)),
                      child: const Icon(Icons.close_rounded,
                          color: Colors.white, size: 18)),
                  ),
                ]),

                const SizedBox(height: 16),

                // Academic info chips
                if (user?.program != null)
                  Wrap(spacing: 8, runSpacing: 6, children: [
                    if (user!.program?.isNotEmpty == true)
                      _drawerChip(user.program!),
                    if (user.branch?.isNotEmpty == true)
                      _drawerChip(user.branch!),
                    if (user.semester?.isNotEmpty == true)
                      _drawerChip('Sem ${user.semester!}'),
                  ]),

                const SizedBox(height: 20),

                // New Chat button
                GestureDetector(
                  onTap: () {
                    Navigator.of(context).pop(); // close drawer
                    _startNewChat();
                  },
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                          colors: [AppTheme.accent, Color(0xFFE8B84B)]),
                      borderRadius: BorderRadius.circular(14),
                      boxShadow: [BoxShadow(
                          color: AppTheme.accent.withOpacity(0.4),
                          blurRadius: 10, offset: const Offset(0, 4))],
                    ),
                    child: Row(mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.add_rounded,
                            color: Color(0xFF1A1A1A), size: 18),
                        const SizedBox(width: 8),
                        Text('New Chat', style: GoogleFonts.lato(
                            fontSize: 14, fontWeight: FontWeight.w700,
                            color: const Color(0xFF1A1A1A))),
                      ]),
                  ),
                ),
                const SizedBox(height: 20),
              ]),
            ),

            // ── Divider ───────────────────────────────────────
            Container(height: 1,
                color: Colors.white.withOpacity(0.1),
                margin: const EdgeInsets.symmetric(horizontal: 20)),

            // ── Chat History Header ───────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
              child: Row(children: [
                const Icon(Icons.history_rounded,
                    color: Colors.white60, size: 16),
                const SizedBox(width: 8),
                Text('Chat History',
                    style: GoogleFonts.lato(fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: Colors.white60, letterSpacing: 0.5)),
                const Spacer(),
                Text('${chat.conversations.length} chats',
                    style: GoogleFonts.lato(fontSize: 11,
                        color: Colors.white38)),
              ]),
            ),

            // ── Conversation List ─────────────────────────────
            Expanded(
              child: chat.conversations.isEmpty
                  ? _drawerEmptyHistory()
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      itemCount: chat.conversations.length,
                      itemBuilder: (_, i) =>
                          _drawerConvTile(chat.conversations[i]),
                    ),
            ),

            // ── Bottom: Logout ────────────────────────────────
            Container(
              margin: const EdgeInsets.all(16),
              child: GestureDetector(
                onTap: () async {
                  Navigator.of(context).pop();
                  await auth.logout();
                  if (!mounted) return;
                  Navigator.of(context).pushReplacement(
                      MaterialPageRoute(builder: (_) => const AuthScreen()));
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                          color: Colors.white.withOpacity(0.15))),
                  child: Row(mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.logout_rounded,
                          color: Colors.white60, size: 16),
                      const SizedBox(width: 8),
                      Text('Logout', style: GoogleFonts.lato(
                          fontSize: 13, color: Colors.white60,
                          fontWeight: FontWeight.w600)),
                    ]),
                ),
              ),
            ),
          ]),
        ),
      ),
    );
  }

  Widget _drawerChip(String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
    decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.2))),
    child: Text(label, style: GoogleFonts.lato(
        fontSize: 11, color: Colors.white70, fontWeight: FontWeight.w500)),
  );

  Widget _drawerEmptyHistory() => Padding(
    padding: const EdgeInsets.all(24),
    child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      Icon(Icons.chat_bubble_outline_rounded,
          color: Colors.white.withOpacity(0.2), size: 48),
      const SizedBox(height: 12),
      Text('No conversations yet',
          style: GoogleFonts.lato(fontSize: 14,
              color: Colors.white38), textAlign: TextAlign.center),
      const SizedBox(height: 8),
      Text('Start a new chat to begin',
          style: GoogleFonts.lato(fontSize: 12, color: Colors.white24),
          textAlign: TextAlign.center),
    ]),
  );

  Widget _drawerConvTile(ConversationModel conv) {
    final isActive   = conv.status == 'active';
    final isResolved = conv.status == 'resolved';
    final dotColor   = isResolved ? const Color(0xFF4ADE80)
        : isActive ? const Color(0xFF60A5FA) : const Color(0xFFFBBF24);

    return GestureDetector(
      onTap: () => _openConversation(conv),
      child: Container(
        margin: const EdgeInsets.only(bottom: 6),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.06),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.white.withOpacity(0.08)),
        ),
        child: Row(children: [
          // Status dot
          Container(width: 8, height: 8,
              decoration: BoxDecoration(
                  color: dotColor, shape: BoxShape.circle,
                  boxShadow: [BoxShadow(color: dotColor.withOpacity(0.6),
                      blurRadius: 4)])),
          const SizedBox(width: 12),
          Expanded(child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(conv.title,
                  style: GoogleFonts.lato(fontSize: 13,
                      fontWeight: FontWeight.w600, color: Colors.white),
                  overflow: TextOverflow.ellipsis, maxLines: 1),
              const SizedBox(height: 2),
              Text(
                DateFormat('MMM d, h:mm a')
                    .format(conv.updatedAt.toLocal()),
                style: GoogleFonts.lato(
                    fontSize: 11, color: Colors.white38),
              ),
            ],
          )),
          const Icon(Icons.chevron_right_rounded,
              color: Colors.white30, size: 16),
        ]),
      ),
    );
  }

  // ══════════════════════════════════════════════════════════
  // HEADER
  // ══════════════════════════════════════════════════════════
  Widget _buildHeader(AuthProvider auth, ChatProvider chat) {
    final user      = auth.currentUser;
    final initials  = _getInitials(user?.name ?? 'S');
    final greeting  = _getGreeting();

    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF0F1C3F), Color(0xFF1A2B5F), Color(0xFF243680)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Column(children: [
          // Top bar
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Row(children: [
              // Drawer burger
              GestureDetector(
                onTap: () => _scaffoldKey.currentState?.openDrawer(),
                child: Container(
                  width: 40, height: 40,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                        color: Colors.white.withOpacity(0.2)),
                  ),
                  child: Stack(alignment: Alignment.center, children: [
                    const Icon(Icons.menu_rounded,
                        color: Colors.white, size: 20),
                    // Badge if there are conversations
                    if (chat.conversations.isNotEmpty)
                      Positioned(top: 8, right: 8,
                        child: Container(width: 7, height: 7,
                          decoration: const BoxDecoration(
                              color: AppTheme.accent,
                              shape: BoxShape.circle))),
                  ]),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(greeting, style: GoogleFonts.lato(
                      fontSize: 12, color: Colors.white54,
                      letterSpacing: 0.3)),
                  Text(user?.name ?? 'Student',
                      style: GoogleFonts.playfairDisplay(
                          fontSize: 18, fontWeight: FontWeight.w700,
                          color: Colors.white),
                      overflow: TextOverflow.ellipsis),
                ],
              )),
              // Avatar
              Container(
                width: 42, height: 42,
                decoration: BoxDecoration(
                  color: AppTheme.accent,
                  shape: BoxShape.circle,
                  border: Border.all(
                      color: Colors.white.withOpacity(0.3), width: 2),
                  boxShadow: [BoxShadow(
                      color: AppTheme.accent.withOpacity(0.4),
                      blurRadius: 8, offset: const Offset(0, 3))],
                ),
                child: Center(child: Text(initials,
                    style: GoogleFonts.playfairDisplay(
                        fontSize: 16, fontWeight: FontWeight.w700,
                        color: Colors.white))),
              ),
            ]),
          ),

          // Academic info strip
          if (user?.program != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 0),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(children: [
                  if (user!.program?.isNotEmpty == true)
                    _headerChip(Icons.school_rounded, user.program!),
                  if (user.branch?.isNotEmpty == true) ...[
                    const SizedBox(width: 8),
                    _headerChip(Icons.account_tree_rounded, user.branch!),
                  ],
                  if (user.semester?.isNotEmpty == true) ...[
                    const SizedBox(width: 8),
                    _headerChip(Icons.calendar_today_rounded,
                        'Sem ${user.semester!}'),
                  ],
                  if (user.mentorEmail?.isNotEmpty == true) ...[
                    const SizedBox(width: 8),
                    _headerChip(Icons.supervisor_account_rounded,
                        'Mentor linked'),
                  ],
                ]),
              ),
            ),

          // Stats row
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
            child: Row(children: [
              _statPill('${chat.conversations.length}',
                  'Chats', Icons.chat_bubble_rounded,
                  const Color(0xFF60A5FA)),
              const SizedBox(width: 10),
              _statPill(
                  '${chat.conversations.where((c) => c.status == 'resolved').length}',
                  'Resolved', Icons.check_circle_rounded,
                  const Color(0xFF4ADE80)),
              const SizedBox(width: 10),
              _statPill(
                  '${chat.conversations.where((c) => c.status == 'active').length}',
                  'Active', Icons.radio_button_checked_rounded,
                  AppTheme.accentLight),
            ]),
          ),
        ]),
      ),
    );
  }

  Widget _headerChip(IconData icon, String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.2))),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 12, color: Colors.white70),
      const SizedBox(width: 5),
      Text(label, style: GoogleFonts.lato(fontSize: 11,
          color: Colors.white70, fontWeight: FontWeight.w500)),
    ]),
  );

  Widget _statPill(String value, String label, IconData icon, Color color) =>
      Expanded(child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withOpacity(0.25)),
        ),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 6),
          Text(value, style: GoogleFonts.lato(
              fontSize: 15, fontWeight: FontWeight.w700, color: Colors.white)),
          const SizedBox(width: 4),
          Text(label, style: GoogleFonts.lato(
              fontSize: 10, color: Colors.white54)),
        ]),
      ));

  // ══════════════════════════════════════════════════════════
  // TAB BAR
  // ══════════════════════════════════════════════════════════
  Widget _buildTabBar(ChatProvider chat) => Container(
    decoration: const BoxDecoration(
      color: Colors.white,
      boxShadow: [BoxShadow(
          color: Color(0x10000000), blurRadius: 8,
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
      labelStyle: GoogleFonts.lato(
          fontSize: 12, fontWeight: FontWeight.w700),
      unselectedLabelStyle: GoogleFonts.lato(fontSize: 12),
      tabs: [
        Tab(height: 50, child: _tabItem(
            Icons.smart_toy_rounded, 'AI Chat',
            badge: chat.conversations.isNotEmpty
                ? '${chat.conversations.length}' : null)),
        const Tab(height: 50, child: _TabItem(
            Icons.folder_rounded, 'Documents')),
        const Tab(height: 50, child: _TabItem(
            Icons.report_problem_outlined, 'Issues')),
      ],
    ),
  );

  Widget _tabItem(IconData icon, String label, {String? badge}) =>
      Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Stack(alignment: Alignment.topRight, children: [
          Icon(icon, size: 20),
          if (badge != null)
            Positioned(top: -3, right: -6,
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 5, vertical: 1),
                decoration: BoxDecoration(
                    color: AppTheme.primary,
                    borderRadius: BorderRadius.circular(10)),
                child: Text(badge, style: GoogleFonts.lato(
                    fontSize: 8, color: Colors.white,
                    fontWeight: FontWeight.w700)))),
        ]),
        const SizedBox(height: 3),
        Text(label),
      ]);

  // ══════════════════════════════════════════════════════════
  // AI CHAT TAB
  // ══════════════════════════════════════════════════════════
  Widget _aiChatTab(ChatProvider chat) {
    if (chat.isLoading) {
      return const Center(child: CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation(AppTheme.primary)));
    }
    if (chat.conversations.isEmpty) return _emptyChat();

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.builder(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
        itemCount: chat.conversations.length,
        itemBuilder: (_, i) => _convCard(chat.conversations[i], i),
      ),
    );
  }

  Widget _emptyChat() => SingleChildScrollView(
    child: Padding(
      padding: const EdgeInsets.fromLTRB(24, 40, 24, 40),
      child: Column(children: [
        // Hero illustration
        Container(
          width: 120, height: 120,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                AppTheme.primary.withOpacity(0.1),
                AppTheme.primary.withOpacity(0.05),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            shape: BoxShape.circle,
            border: Border.all(
                color: AppTheme.primary.withOpacity(0.15), width: 2),
          ),
          child: const Icon(Icons.smart_toy_rounded,
              size: 52, color: AppTheme.primary),
        ),
        const SizedBox(height: 24),
        Text('Your AI Academic Assistant',
            style: GoogleFonts.playfairDisplay(
                fontSize: 22, fontWeight: FontWeight.w700,
                color: const Color(0xFF111827)),
            textAlign: TextAlign.center),
        const SizedBox(height: 10),
        Text(
          'Ask anything about your academics,\n'
          'career, timetable, or college life.',
          style: GoogleFonts.lato(fontSize: 14,
              color: const Color(0xFF6B7280), height: 1.6),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 32),

        // Feature cards
        _featureCard('📅', 'Timetable',
            'Ask "What are my classes today?" after uploading timetable'),
        const SizedBox(height: 10),
        _featureCard('📊', 'Marks & Attendance',
            'Upload marksheet to get subject-wise analysis'),
        const SizedBox(height: 10),
        _featureCard('🚀', 'Career Guidance',
            'Get personalized career paths for your branch'),
        const SizedBox(height: 10),
        _featureCard('💬', 'Personal Support',
            'Stress management, study plans, and motivation'),

        const SizedBox(height: 36),
        GestureDetector(
          onTap: _startNewChat,
          child: Container(
            width: double.infinity, height: 56,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                  colors: [Color(0xFF0F1C3F), Color(0xFF1A2B5F)]),
              borderRadius: BorderRadius.circular(16),
              boxShadow: [BoxShadow(
                  color: AppTheme.primary.withOpacity(0.4),
                  blurRadius: 16, offset: const Offset(0, 6))],
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.chat_rounded,
                    color: Colors.white, size: 20),
                const SizedBox(width: 10),
                Text('Start AI Chat', style: GoogleFonts.lato(
                    fontSize: 16, fontWeight: FontWeight.w700,
                    color: Colors.white)),
              ]),
          ),
        ),
      ]),
    ),
  );

  Widget _featureCard(String emoji, String title, String desc) =>
      Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 8, offset: const Offset(0, 2))],
        ),
        child: Row(children: [
          Container(width: 44, height: 44,
            decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.07),
                borderRadius: BorderRadius.circular(12)),
            child: Center(child: Text(emoji,
                style: const TextStyle(fontSize: 20)))),
          const SizedBox(width: 14),
          Expanded(child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: GoogleFonts.lato(
                  fontSize: 14, fontWeight: FontWeight.w700,
                  color: const Color(0xFF111827))),
              Text(desc, style: GoogleFonts.lato(
                  fontSize: 12, color: const Color(0xFF6B7280),
                  height: 1.4)),
            ],
          )),
        ]),
      );

  Widget _convCard(ConversationModel conv, int index) {
    final colors = [
      [const Color(0xFFEFF6FF), const Color(0xFF3B82F6)],
      [const Color(0xFFF0FDF4), const Color(0xFF22C55E)],
      [const Color(0xFFFFF7ED), const Color(0xFFF97316)],
      [const Color(0xFFFDF4FF), const Color(0xFFA855F7)],
      [const Color(0xFFFFF1F2), const Color(0xFFF43F5E)],
    ];
    final c = colors[index % colors.length];

    final isResolved = conv.status == 'resolved';
    final isFlagged  = conv.status == 'flagged';
    final statusColor = isResolved ? const Color(0xFF22C55E)
        : isFlagged ? const Color(0xFFF59E0B) : c[1];

    return GestureDetector(
      onTap: () async {
        final auth = context.read<AuthProvider>();
        await context.read<ChatProvider>()
            .loadConversation(conv.id, auth.currentUser!.id);
        if (!mounted) return;
        Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => const StudentChatScreen()));
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 12, offset: const Offset(0, 4))],
        ),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(children: [
            // Icon box
            Container(
              width: 50, height: 50,
              decoration: BoxDecoration(
                color: c[0],
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(Icons.smart_toy_rounded, color: c[1], size: 24),
            ),
            const SizedBox(width: 14),
            Expanded(child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(conv.title, style: GoogleFonts.lato(
                    fontSize: 14, fontWeight: FontWeight.w700,
                    color: const Color(0xFF111827)),
                    overflow: TextOverflow.ellipsis, maxLines: 1),
                const SizedBox(height: 4),
                Row(children: [
                  const Icon(Icons.access_time_rounded,
                      size: 12, color: Color(0xFF9CA3AF)),
                  const SizedBox(width: 4),
                  Text(DateFormat('MMM d • h:mm a')
                      .format(conv.updatedAt.toLocal()),
                      style: GoogleFonts.lato(
                          fontSize: 11, color: const Color(0xFF9CA3AF))),
                ]),
              ],
            )),
            // Status badge
            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                  color: statusColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: statusColor.withOpacity(0.3))),
              child: Text(conv.status.toUpperCase(),
                  style: GoogleFonts.lato(
                      fontSize: 9, fontWeight: FontWeight.w700,
                      color: statusColor, letterSpacing: 0.5)),
            ),
            const SizedBox(width: 8),
            // Delete button
            IconButton(
              onPressed: () => _deleteConv(conv),
              icon: Icon(Icons.delete_outline_rounded, color: Colors.red.withOpacity(0.5), size: 18),
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(),
              splashRadius: 20,
              tooltip: 'Delete Chat',
            ),
          ]),
        ),
      ),
    );
  }

  // ══════════════════════════════════════════════════════════
  // FAB
  // ══════════════════════════════════════════════════════════
  Widget _buildFAB() => Container(
    decoration: BoxDecoration(
      gradient: const LinearGradient(
          colors: [AppTheme.accent, Color(0xFFE8B84B)]),
      borderRadius: BorderRadius.circular(16),
      boxShadow: [BoxShadow(
          color: AppTheme.accent.withOpacity(0.5),
          blurRadius: 16, offset: const Offset(0, 6))],
    ),
    child: Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: _startNewChat,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.add_rounded,
                color: Color(0xFF1A1A1A), size: 20),
            const SizedBox(width: 8),
            Text('New Chat', style: GoogleFonts.lato(
                fontSize: 14, fontWeight: FontWeight.w700,
                color: const Color(0xFF1A1A1A))),
          ]),
        ),
      ),
    ),
  );

  // ── Helpers ───────────────────────────────────────────────
  String _getInitials(String name) => name.trim().split(' ')
      .map((w) => w.isNotEmpty ? w[0] : '')
      .take(2).join().toUpperCase();

  String _getGreeting() {
    final h = DateTime.now().hour;
    if (h < 12) return 'Good morning! 🌤️';
    if (h < 17) return 'Good afternoon! ☀️';
    return 'Good evening! 🌙';
  }
}

// ── Stateless tab item widget ─────────────────────────────────
class _TabItem extends StatelessWidget {
  final IconData icon;
  final String label;
  const _TabItem(this.icon, this.label);

  @override
  Widget build(BuildContext context) => Column(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      Icon(icon, size: 20),
      const SizedBox(height: 3),
      Text(label),
    ],
  );
}