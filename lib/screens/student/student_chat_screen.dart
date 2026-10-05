// lib/screens/student/student_chat_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/auth_provider.dart';
import '../../services/chat_provider.dart';
import '../../services/supabase_service.dart';
import '../../utils/app_theme.dart';
import '../../widgets/chat_interactive_options.dart';
import '../../widgets/typing_dots_indicator.dart';

class StudentChatScreen extends StatefulWidget {
  const StudentChatScreen({super.key});
  @override
  State<StudentChatScreen> createState() => _StudentChatScreenState();
}

class _StudentChatScreenState extends State<StudentChatScreen>
    with TickerProviderStateMixin {
  final TextEditingController _ctrl = TextEditingController();
  final ScrollController _scroll = ScrollController();
  final FocusNode _focus = FocusNode();

  List<StudentDocument> _availableDocs = [];
  bool _showSuggestions = true;

  // Quick suggestion chips
  static const List<Map<String, dynamic>> _suggestions = [
    {'icon': '📅', 'text': "What's my schedule today?"},
    {'icon': '📊', 'text': 'Show my marks summary'},
    {'icon': '✅', 'text': 'Check my attendance status'},
    {'icon': '🗓️', 'text': 'When are my upcoming exams?'},
    {'icon': '🚀', 'text': 'Career guidance for my branch'},
    {'icon': '😰', 'text': "I'm feeling stressed, help me"},
    {'icon': '📚', 'text': 'What topics are in my syllabus?'},
    {'icon': '💡', 'text': 'Give me study tips'},
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadDocs());
  }

  @override
  void dispose() {
    _ctrl.dispose();
    _scroll.dispose();
    _focus.dispose();
    super.dispose();
  }

  Future<void> _loadDocs() async {
    final auth = context.read<AuthProvider>();
    if (auth.currentUser == null) return;
    final docs = await SupabaseService.getStudentAccessibleDocuments(
      studentId: auth.currentUser!.id,
      rollNo: auth.currentUser!.rollNumber,
      program: auth.currentUser!.program,
      branch: auth.currentUser!.branch,
    );
    if (mounted) setState(() => _availableDocs = docs);
  }

  void _scrollToBottom({bool animated = true}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      if (animated) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 350),
          curve: Curves.easeOutCubic,
        );
      } else {
        _scroll.jumpTo(_scroll.position.maxScrollExtent);
      }
    });
  }

  Future<void> _send([String? quickText]) async {
    final text = (quickText ?? _ctrl.text).trim();
    if (text.isEmpty) return;
    _ctrl.clear();
    setState(() {
      _showSuggestions = false;
    });
    _focus.unfocus();

    final auth = context.read<AuthProvider>();

    await context.read<ChatProvider>().sendStudentMessage(
          text,
          auth.currentUser!.id,
        );
    _scrollToBottom();
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatProvider>();
    if (chat.messages.isNotEmpty) _scrollToBottom();
    if (chat.messages.length > 1) _showSuggestions = false;

    return Scaffold(
      backgroundColor: const Color(0xFFF3F5FB),
      appBar: _buildAppBar(chat),
      body: Column(children: [
        Expanded(child: _messageList(chat)),
        if (chat.isTyping) _typingBubble(),
        if (_showSuggestions && chat.messages.length <= 1) _suggestionsBar(),
        _inputBar(chat),
      ]),
    );
  }

  // ══════════════════════════════════════════════════════════
  // APP BAR
  // ══════════════════════════════════════════════════════════
  PreferredSizeWidget _buildAppBar(ChatProvider chat) {
    return AppBar(
      backgroundColor: Colors.transparent,
      elevation: 0,
      flexibleSpace: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF0F1C3F), Color(0xFF1A2B5F)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
      ),
      leading: IconButton(
        icon: Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(10)),
          child: const Icon(Icons.arrow_back_rounded,
              color: Colors.white, size: 18),
        ),
        onPressed: () {
          chat.clearCurrentConversation();
          Navigator.of(context).pop();
        },
      ),
      title: Row(children: [
        // AI Avatar
        Container(
          width: 38,
          height: 38,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
                colors: [AppTheme.accent, Color(0xFFE8B84B)]),
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                  color: AppTheme.accent.withOpacity(0.4),
                  blurRadius: 8,
                  offset: const Offset(0, 2))
            ],
          ),
          child: const Icon(Icons.smart_toy_rounded,
              color: Color(0xFF1A1A1A), size: 20),
        ),
        const SizedBox(width: 10),
        Expanded(
            child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('AI Academic Assistant',
                style: GoogleFonts.playfairDisplay(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Colors.white)),
            Row(children: [
              Container(
                width: 7,
                height: 7,
                decoration: const BoxDecoration(
                    color: Color(0xFF4ADE80), shape: BoxShape.circle),
              ),
              const SizedBox(width: 5),
              Text(
                  chat.isTyping
                      ? 'Thinking...'
                      : _availableDocs.isNotEmpty
                          ? 'RAG Active • ${_availableDocs.length} docs'
                          : 'Gemini Free',
                  style: GoogleFonts.lato(fontSize: 11, color: Colors.white60)),
            ]),
          ],
        )),
      ]),
      actions: [
        // Verified Docs count badge
        if (_availableDocs.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: Colors.white.withOpacity(0.25)),
                ),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  const Icon(Icons.verified_rounded,
                      color: AppTheme.accent, size: 14),
                  const SizedBox(width: 4),
                  Text('${_availableDocs.length} Verified',
                      style: GoogleFonts.lato(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: Colors.white)),
                ]),
              ),
            ),
          ),
      ],
    );
  }

  // ══════════════════════════════════════════════════════════
  // MESSAGE LIST
  // ══════════════════════════════════════════════════════════
  Widget _messageList(ChatProvider chat) {
    if (chat.isLoading) {
      return const Center(
          child: CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation(AppTheme.primary)));
    }

    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      itemCount: chat.messages.length,
      itemBuilder: (_, i) {
        final msg = chat.messages[i];
        final prev = i > 0 ? chat.messages[i - 1] : null;
        final showTime = prev == null ||
            msg.createdAt.difference(prev.createdAt).inMinutes > 5;
        final showAvatar = !msg.isUser &&
            (i == chat.messages.length - 1 || chat.messages[i + 1].isUser);

        return Column(children: [
          if (showTime) _timeStamp(msg.createdAt),
          _messageBubble(msg, showAvatar: showAvatar),
        ]);
      },
    );
  }

  Widget _timeStamp(DateTime dt) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 12),
        child: Center(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 5),
            decoration: BoxDecoration(
                color: const Color(0xFFE5E7EB),
                borderRadius: BorderRadius.circular(20)),
            child: Text(DateFormat('MMM d, h:mm a').format(dt.toLocal()),
                style: GoogleFonts.lato(
                    fontSize: 11, color: const Color(0xFF6B7280))),
          ),
        ),
      );

  Widget _messageBubble(MessageModel msg, {bool showAvatar = false}) {
    final isUser = msg.isUser;
    final isMentor = msg.isMentor;

    return Padding(
      padding: EdgeInsets.only(
          bottom: 4, left: isUser ? 48 : 0, right: isUser ? 0 : 48),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          // AI avatar (only for last AI message in group)
          if (!isUser)
            Padding(
              padding: const EdgeInsets.only(right: 8, bottom: 2),
              child: showAvatar
                  ? Container(
                      width: 30,
                      height: 30,
                      decoration: const BoxDecoration(
                        gradient: LinearGradient(
                            colors: [AppTheme.accent, Color(0xFFE8B84B)]),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.smart_toy_rounded,
                          color: Color(0xFF1A1A1A), size: 16))
                  : const SizedBox(width: 30),
            ),

          Flexible(
            child: GestureDetector(
              onLongPress: () => _copyMessage(msg.content),
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: BoxDecoration(
                  color: isUser
                      ? AppTheme.primary
                      : isMentor
                          ? AppTheme.mentorBubble
                          : Colors.white,
                  borderRadius: BorderRadius.only(
                    topLeft: const Radius.circular(20),
                    topRight: const Radius.circular(20),
                    bottomLeft: Radius.circular(isUser ? 20 : 4),
                    bottomRight: Radius.circular(isUser ? 4 : 20),
                  ),
                  boxShadow: [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.07),
                        blurRadius: 8,
                        offset: const Offset(0, 2))
                  ],
                ),
                child: isUser
                    ? Text(msg.content,
                        style: GoogleFonts.lato(
                            color: Colors.white, fontSize: 14, height: 1.5))
                    : Builder(builder: (context) {
                        final parsed = InteractiveOptionParser.parse(msg.content);
                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            MarkdownBody(
                              data: parsed.cleanText,
                              styleSheet: MarkdownStyleSheet(
                                p: GoogleFonts.lato(
                                    fontSize: 14,
                                    height: 1.45,
                                    color: isMentor
                                        ? Colors.white
                                        : const Color(0xFF111827)),
                                strong: GoogleFonts.lato(
                                    fontWeight: FontWeight.w700,
                                    color:
                                        isMentor ? Colors.white : AppTheme.primary),
                                listBullet: GoogleFonts.lato(
                                    color: isMentor
                                        ? Colors.white
                                        : const Color(0xFF111827)),
                                h2: GoogleFonts.playfairDisplay(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                    color: AppTheme.primary),
                                h3: GoogleFonts.lato(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w700,
                                    color: AppTheme.primary),
                                code: GoogleFonts.sourceCodePro(
                                    fontSize: 13,
                                    backgroundColor: const Color(0xFFF3F4F6)),
                                blockquoteDecoration: BoxDecoration(
                                  color: AppTheme.primary.withOpacity(0.05),
                                  borderRadius: BorderRadius.circular(4),
                                  border: const Border(
                                      left: BorderSide(
                                          color: AppTheme.primary, width: 3)),
                                ),
                              ),
                            ),
                            if (parsed.options.isNotEmpty) ...[
                              const SizedBox(height: 6),
                              ChatInteractiveOptionsView(
                                options: parsed.options,
                                primaryColor: isMentor
                                    ? Colors.white
                                    : AppTheme.primary,
                                onOptionSelected: (selected) => _send(selected),
                              ),
                            ],
                          ],
                        );
                      }),
              ),
            ),
          ),

          // User avatar
          if (isUser)
            Padding(
              padding: const EdgeInsets.only(left: 8, bottom: 2),
              child: Container(
                  width: 30,
                  height: 30,
                  decoration: const BoxDecoration(
                      color: AppTheme.accent, shape: BoxShape.circle),
                  child: const Icon(Icons.person_rounded,
                      color: Color(0xFF1A1A1A), size: 16)),
            ),
        ],
      ),
    );
  }

  void _copyMessage(String text) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text('Copied to clipboard',
          style: GoogleFonts.lato(color: Colors.white)),
      backgroundColor: AppTheme.primary,
      duration: const Duration(seconds: 2),
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
    ));
  }

  // ══════════════════════════════════════════════════════════
  // TYPING INDICATOR
  // ══════════════════════════════════════════════════════════
  Widget _typingBubble() => Padding(
        padding: const EdgeInsets.fromLTRB(16, 0, 48, 4),
        child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Container(
              width: 30,
              height: 30,
              decoration: const BoxDecoration(
                  gradient: LinearGradient(
                      colors: [AppTheme.accent, Color(0xFFE8B84B)]),
                  shape: BoxShape.circle),
              child: const Icon(Icons.smart_toy_rounded,
                  color: Color(0xFF1A1A1A), size: 16)),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(20),
                topRight: Radius.circular(20),
                bottomRight: Radius.circular(20),
                bottomLeft: Radius.circular(4),
              ),
              boxShadow: [
                BoxShadow(
                    color: Colors.black.withOpacity(0.07),
                    blurRadius: 8,
                    offset: const Offset(0, 2))
              ],
            ),
            child: const TypingDotsIndicator(
              color: AppTheme.primary,
              dotSize: 7.5,
              spacing: 5.0,
            ),
          ),
        ]),
      );

  // ══════════════════════════════════════════════════════════
  // SUGGESTION CHIPS
  // ══════════════════════════════════════════════════════════
  Widget _suggestionsBar() => Container(
        color: Colors.white,
        padding: const EdgeInsets.fromLTRB(16, 10, 16, 6),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Suggested questions',
              style: GoogleFonts.lato(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: const Color(0xFF9CA3AF),
                  letterSpacing: 0.3)),
          const SizedBox(height: 8),
          SizedBox(
            height: 40,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: _suggestions.length,
              separatorBuilder: (_, __) => const SizedBox(width: 8),
              itemBuilder: (_, i) {
                final s = _suggestions[i];
                return GestureDetector(
                  onTap: () => _send(s['text'] as String),
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    decoration: BoxDecoration(
                        color: const Color(0xFFF0F4FF),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                            color: AppTheme.primary.withOpacity(0.2))),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Text(s['icon'] as String,
                          style: const TextStyle(fontSize: 13)),
                      const SizedBox(width: 6),
                      Text(s['text'] as String,
                          style: GoogleFonts.lato(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: AppTheme.primary)),
                    ]),
                  ),
                );
              },
            ),
          ),
        ]),
      );

  // ══════════════════════════════════════════════════════════
  // INPUT BAR
  // ══════════════════════════════════════════════════════════
  Widget _inputBar(ChatProvider chat) {
    final isTyping = chat.isTyping;

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.08),
              blurRadius: 16,
              offset: const Offset(0, -4))
        ],
      ),
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 10),
      child: SafeArea(
        top: false,
        child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
          // Text field
          Expanded(
            child: Container(
              constraints: const BoxConstraints(maxHeight: 130),
              decoration: BoxDecoration(
                color: const Color(0xFFF3F5FB),
                borderRadius: BorderRadius.circular(22),
                border: Border.all(color: const Color(0xFFE5E7EB)),
              ),
              child: Focus(
                onKeyEvent: (node, event) {
                  if (event is KeyDownEvent &&
                      event.logicalKey == LogicalKeyboardKey.enter &&
                      !HardwareKeyboard.instance.isShiftPressed) {
                    if (!isTyping) {
                      _send();
                    }
                    return KeyEventResult.handled;
                  }
                  return KeyEventResult.ignored;
                },
                child: TextField(
                  controller: _ctrl,
                  focusNode: _focus,
                  maxLines: 6,
                  minLines: 1,
                  textCapitalization: TextCapitalization.sentences,
                  textInputAction: TextInputAction.send,
                  onSubmitted: isTyping ? null : (_) => _send(),
                  decoration: InputDecoration(
                    hintText:
                        'Ask anything about attendance, marks, schedule...',
                    hintStyle: GoogleFonts.lato(
                        color: const Color(0xFF9CA3AF), fontSize: 14),
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.symmetric(
                        horizontal: 18, vertical: 11),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),

          // Send button
          GestureDetector(
            onTap: isTyping ? null : () => _send(),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                gradient: isTyping
                    ? null
                    : const LinearGradient(
                        colors: [Color(0xFF1A2B5F), Color(0xFF243680)]),
                color: isTyping ? const Color(0xFFE5E7EB) : null,
                borderRadius: BorderRadius.circular(12),
                boxShadow: isTyping
                    ? null
                    : [
                        BoxShadow(
                            color: AppTheme.primary.withOpacity(0.4),
                            blurRadius: 8,
                            offset: const Offset(0, 3))
                      ],
              ),
              child: Icon(
                  isTyping
                      ? Icons.hourglass_bottom_rounded
                      : Icons.send_rounded,
                  color: isTyping ? const Color(0xFF9CA3AF) : Colors.white,
                  size: 19),
            ),
          ),
        ]),
      ),
    );
  }
}
