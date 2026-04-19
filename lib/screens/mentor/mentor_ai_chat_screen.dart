// lib/screens/mentor/mentor_ai_chat_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../services/ai_service.dart';
import '../../services/auth_provider.dart';
import '../../services/chat_provider.dart';
import '../../utils/app_theme.dart';

class _Msg {
  final String role, content;
  final DateTime time;
  _Msg({required this.role, required this.content}) : time = DateTime.now();
  bool get isUser => role == 'user';
}

class MentorAiChatScreen extends StatefulWidget {
  const MentorAiChatScreen({super.key});
  @override
  State<MentorAiChatScreen> createState() => _MentorAiChatScreenState();
}

class _MentorAiChatScreenState extends State<MentorAiChatScreen> {
  final TextEditingController _ctrl   = TextEditingController();
  final ScrollController       _scroll = ScrollController();
  final List<_Msg>             _msgs   = [];
  bool _isTyping    = false;
  bool _initialized = false;

  static const List<Map<String, dynamic>> _quickActions = [
    {'emoji': '📊', 'label': 'Progress Report',   'prompt': 'Give me a summary of how my students are performing overall.'},
    {'emoji': '⚠️', 'label': 'At-Risk Students',  'prompt': 'Which students might be at risk based on low engagement or issues?'},
    {'emoji': '✉️', 'label': 'Warning Email',      'prompt': 'Draft a professional attendance warning email for a struggling student.'},
    {'emoji': '📅', 'label': 'Academic Calendar',  'prompt': 'What are the key academic events and deadlines I should track this semester?'},
    {'emoji': '💡', 'label': 'Intervention Tips',  'prompt': 'What are the best intervention strategies for a student with declining marks?'},
    {'emoji': '🎯', 'label': 'Career Guidance',    'prompt': 'How can I help my B.Tech students with career planning and placements?'},
    {'emoji': '🧠', 'label': 'Student Stress',     'prompt': 'How do I support a student who is showing signs of stress or anxiety?'},
    {'emoji': '📋', 'label': 'Class Report',       'prompt': 'Create a structured template for a monthly class progress report.'},
    {'emoji': '🏛️', 'label': 'College Policy',    'prompt': 'What are the standard procedures for academic disciplinary actions?'},
    {'emoji': '👨‍👩‍👧', 'label': 'Parent Meeting', 'prompt': 'How should I prepare for a parent-teacher meeting about a student\'s performance?'},
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _greet());
  }

  @override
  void dispose() { _ctrl.dispose(); _scroll.dispose(); super.dispose(); }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients)
        _scroll.animateTo(_scroll.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
    });
  }

  Future<void> _greet() async {
    if (_initialized) return;
    _initialized = true;
    final auth = context.read<AuthProvider>();
    final chat = context.read<ChatProvider>();
    final name = auth.currentUser?.name.split(' ').first ?? 'Mentor';
    final count = chat.myStudents.length;

    setState(() => _msgs.add(_Msg(
      role: 'assistant',
      content:
          "Hello **$name**! 👋 I'm your **AI Academic Management Assistant** (Gemini).\n\n"
          "I can help you with:\n"
          "- 📊 **Student progress** — reports, trends, at-risk identification\n"
          "- ⚠️ **Issue management** — interventions, escalation strategies\n"
          "- ✉️ **Draft communications** — emails, warning letters, reports\n"
          "- 🏛️ **College policies** — academic rules, procedures, deadlines\n"
          "- 🎯 **Career guidance** — help students with placement prep\n"
          "- 🧠 **Student support** — mental health, stress, counseling referral\n\n"
          "You have **$count student${count != 1 ? 's' : ''}** in your class.\n\n"
          "How can I assist you today?",
    )));
  }

  Future<void> _send([String? quick]) async {
    final text = (quick ?? _ctrl.text).trim();
    if (text.isEmpty || _isTyping) return;
    _ctrl.clear();

    final auth = context.read<AuthProvider>();
    final chat = context.read<ChatProvider>();

    setState(() { _msgs.add(_Msg(role: 'user', content: text)); _isTyping = true; });
    _scrollToBottom();

    try {
      // Build history (skip first assistant greeting for cleaner context)
      final history = _msgs.skip(1).where((m) => !m.isUser || m != _msgs.last)
          .map((m) => {'role': m.role, 'content': m.content})
          .toList();

      final reply = await chat.sendMentorAiMessage(
        history:       history.cast<Map<String, dynamic>>(),
        newMessage:    text,
        mentorName:    auth.currentUser?.name ?? 'Mentor',
      );
      setState(() => _msgs.add(_Msg(role: 'assistant', content: reply)));
    } catch (e) {
      setState(() => _msgs.add(_Msg(
          role: 'assistant', content: AIService.friendlyError(e.toString()))));
    } finally {
      setState(() => _isTyping = false);
      _scrollToBottom();
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final name = auth.currentUser?.name.split(' ').first ?? 'Mentor';

    return Scaffold(
      backgroundColor: const Color(0xFFF0F4FF),
      appBar: AppBar(
        backgroundColor: AppTheme.mentorBubble,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
          onPressed: () => Navigator.of(context).pop()),
        title: Row(children: [
          Container(width: 36, height: 36,
            decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2), shape: BoxShape.circle),
            child: const Icon(Icons.smart_toy_rounded, color: Colors.white, size: 20)),
          const SizedBox(width: 10),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('AI Assistant — Mentor Mode',
                style: GoogleFonts.playfairDisplay(
                    fontSize: 15, fontWeight: FontWeight.w600, color: Colors.white)),
            Text(_isTyping ? 'Thinking...' : 'Powered by Gemini (Free)',
                style: GoogleFonts.lato(fontSize: 11, color: Colors.white70)),
          ]),
        ]),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded, color: Colors.white),
            tooltip: 'New conversation',
            onPressed: () {
              setState(() { _msgs.clear(); _initialized = false; });
              _greet();
            }),
        ],
      ),
      body: Column(children: [
        Expanded(
          child: ListView.builder(
            controller: _scroll,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            itemCount: _msgs.length + (_isTyping ? 1 : 0),
            itemBuilder: (_, i) {
              if (i == _msgs.length) return _typingIndicator();
              final msg = _msgs[i];
              final showTime = i == 0 ||
                  _msgs[i].time.difference(_msgs[i - 1].time).inMinutes > 5;
              return Column(children: [
                if (showTime)
                  Padding(padding: const EdgeInsets.symmetric(vertical: 6),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                      decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.06),
                          borderRadius: BorderRadius.circular(20)),
                      child: Text(DateFormat('h:mm a').format(msg.time.toLocal()),
                          style: GoogleFonts.lato(fontSize: 11,
                              color: const Color(0xFF6B7280))))),
                _bubble(msg),
              ]);
            },
          ),
        ),

        // Quick actions
        if (_msgs.length <= 2 && !_isTyping)
          Container(
            color: Colors.white,
            padding: const EdgeInsets.fromLTRB(16, 10, 16, 6),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Quick Actions', style: GoogleFonts.lato(fontSize: 12,
                  fontWeight: FontWeight.w600, color: const Color(0xFF9CA3AF))),
              const SizedBox(height: 8),
              SizedBox(height: 38,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: _quickActions.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 8),
                  itemBuilder: (_, i) {
                    final q = _quickActions[i];
                    return GestureDetector(
                      onTap: () => _send(q['prompt'] as String),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: AppTheme.mentorBubble.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                              color: AppTheme.mentorBubble.withOpacity(0.25))),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          Text(q['emoji'] as String,
                              style: const TextStyle(fontSize: 13)),
                          const SizedBox(width: 6),
                          Text(q['label'] as String, style: GoogleFonts.lato(
                              fontSize: 12, color: AppTheme.mentorBubble,
                              fontWeight: FontWeight.w600)),
                        ]),
                      ),
                    );
                  },
                ),
              ),
            ]),
          ),

        _inputArea(),
      ]),
    );
  }

  Widget _bubble(_Msg msg) {
    final isUser = msg.isUser;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!isUser) ...[
            Container(width: 32, height: 32,
              decoration: BoxDecoration(color: AppTheme.mentorBubble,
                  borderRadius: BorderRadius.circular(10)),
              child: const Icon(Icons.smart_toy_rounded,
                  color: Colors.white, size: 18)),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              constraints: BoxConstraints(
                  maxWidth: MediaQuery.of(context).size.width * 0.78),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: isUser ? AppTheme.mentorBubble : Colors.white,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(18),
                  topRight: const Radius.circular(18),
                  bottomLeft: Radius.circular(isUser ? 18 : 4),
                  bottomRight: Radius.circular(isUser ? 4 : 18)),
                boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.06),
                    blurRadius: 6, offset: const Offset(0, 2))]),
              child: isUser
                  ? Text(msg.content, style: GoogleFonts.lato(
                      color: Colors.white, fontSize: 14, height: 1.5))
                  : MarkdownBody(
                      data: msg.content,
                      styleSheet: MarkdownStyleSheet(
                        p: GoogleFonts.lato(fontSize: 14, height: 1.6,
                            color: const Color(0xFF111827)),
                        strong: GoogleFonts.lato(fontWeight: FontWeight.w700,
                            color: AppTheme.mentorBubble),
                        h3: GoogleFonts.lato(fontSize: 14,
                            fontWeight: FontWeight.w700, color: AppTheme.primary),
                        listBullet: GoogleFonts.lato(
                            color: const Color(0xFF111827)),
                      )),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8),
            Container(width: 32, height: 32,
              decoration: BoxDecoration(color: AppTheme.accent,
                  borderRadius: BorderRadius.circular(10)),
              child: const Icon(Icons.admin_panel_settings_rounded,
                  color: Colors.white, size: 16)),
          ],
        ],
      ),
    );
  }

  Widget _typingIndicator() => Padding(
    padding: const EdgeInsets.only(bottom: 12),
    child: Row(children: [
      Container(width: 32, height: 32,
        decoration: BoxDecoration(color: AppTheme.mentorBubble,
            borderRadius: BorderRadius.circular(10)),
        child: const Icon(Icons.smart_toy_rounded, color: Colors.white, size: 18)),
      const SizedBox(width: 8),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(color: Colors.white,
            borderRadius: BorderRadius.circular(18),
            boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.06),
                blurRadius: 6)]),
        child: Row(children: [
          _dot(), const SizedBox(width: 4),
          _dot(), const SizedBox(width: 4),
          _dot(),
        ])),
    ]),
  );

  Widget _dot() => TweenAnimationBuilder<double>(
    tween: Tween(begin: 0.0, end: 1.0),
    duration: const Duration(milliseconds: 600),
    curve: Curves.easeInOut,
    builder: (_, v, __) => Container(width: 8, height: 8,
      decoration: BoxDecoration(
          color: AppTheme.mentorBubble.withOpacity(0.4 + v * 0.6),
          shape: BoxShape.circle)));

  Widget _inputArea() => Container(
    padding: const EdgeInsets.fromLTRB(12, 8, 12, 16),
    decoration: const BoxDecoration(color: Colors.white,
      boxShadow: [BoxShadow(color: Color(0x10000000),
          blurRadius: 12, offset: Offset(0, -4))]),
    child: SafeArea(top: false,
      child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
        Expanded(
          child: Container(
            constraints: const BoxConstraints(maxHeight: 120),
            decoration: BoxDecoration(
              color: const Color(0xFFF9FAFB),
              borderRadius: BorderRadius.circular(22),
              border: Border.all(color: AppTheme.mentorBubble.withOpacity(0.25))),
            child: TextField(
              controller: _ctrl, maxLines: 5, minLines: 1,
              textCapitalization: TextCapitalization.sentences,
              onSubmitted: (_) => _send(),
              decoration: InputDecoration(
                hintText: 'Ask about students, policies, drafts...',
                hintStyle: GoogleFonts.lato(
                    color: const Color(0xFF9CA3AF), fontSize: 13),
                border: InputBorder.none,
                contentPadding: const EdgeInsets.symmetric(
                    horizontal: 18, vertical: 11)),
            ),
          ),
        ),
        const SizedBox(width: 8),
        GestureDetector(
          onTap: _isTyping ? null : () => _send(),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: 42, height: 42,
            decoration: BoxDecoration(
              color: _isTyping ? const Color(0xFFD1D5DB) : AppTheme.mentorBubble,
              borderRadius: BorderRadius.circular(12),
              boxShadow: _isTyping ? [] : [BoxShadow(
                  color: AppTheme.mentorBubble.withOpacity(0.35),
                  blurRadius: 8, offset: const Offset(0, 3))]),
            child: Icon(
                _isTyping ? Icons.hourglass_bottom_rounded : Icons.send_rounded,
                color: Colors.white, size: 20))),
      ]),
    ),
  );
}