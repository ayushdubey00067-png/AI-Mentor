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

class StudentChatScreen extends StatefulWidget {
  const StudentChatScreen({super.key});
  @override
  State<StudentChatScreen> createState() => _StudentChatScreenState();
}

class _StudentChatScreenState extends State<StudentChatScreen>
    with TickerProviderStateMixin {
  final TextEditingController _ctrl   = TextEditingController();
  final ScrollController       _scroll = ScrollController();
  final FocusNode              _focus  = FocusNode();

  List<StudentDocument> _availableDocs = [];
  List<StudentDocument> _attachedDocs  = [];
  bool _showSuggestions = true;
  bool _showDocPanel    = false;

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
    _ctrl.dispose(); _scroll.dispose(); _focus.dispose();
    super.dispose();
  }

  Future<void> _loadDocs() async {
    final auth = context.read<AuthProvider>();
    if (auth.currentUser == null) return;
    final docs =
        await SupabaseService.getStudentDocuments(auth.currentUser!.id);
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
    setState(() { _showSuggestions = false; _showDocPanel = false; });
    _focus.unfocus();

    final auth = context.read<AuthProvider>();

    // Fetch full content for attached docs
    List<StudentDocument> fullDocs = [];
    for (final doc in _attachedDocs) {
      final full = await SupabaseService.getDocumentWithContent(doc.id);
      if (full != null) fullDocs.add(full);
    }
    setState(() => _attachedDocs = []);

    await context.read<ChatProvider>().sendStudentMessage(
      text, auth.currentUser!.id,
      attachedDocs: fullDocs.isNotEmpty ? fullDocs : null,
    );
    _scrollToBottom();
  }

  void _toggleDocPanel() {
    setState(() => _showDocPanel = !_showDocPanel);
    if (_showDocPanel) _focus.unfocus();
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
        if (_attachedDocs.isNotEmpty) _attachedBar(),
        if (_showDocPanel) _docPanel(),
        if (_showSuggestions && chat.messages.length <= 1)
          _suggestionsBar(),
        _inputBar(chat),
      ]),
    );
  }

  // ══════════════════════════════════════════════════════════
  // APP BAR
  // ══════════════════════════════════════════════════════════
  PreferredSizeWidget _buildAppBar(ChatProvider chat) {
    final conv = chat.currentConversation;

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
          width: 36, height: 36,
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
          width: 38, height: 38,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
                colors: [AppTheme.accent, Color(0xFFE8B84B)]),
            shape: BoxShape.circle,
            boxShadow: [BoxShadow(
                color: AppTheme.accent.withOpacity(0.4),
                blurRadius: 8, offset: const Offset(0, 2))],
          ),
          child: const Icon(Icons.smart_toy_rounded,
              color: Color(0xFF1A1A1A), size: 20),
        ),
        const SizedBox(width: 10),
        Expanded(child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('AI Academic Assistant',
                style: GoogleFonts.playfairDisplay(
                    fontSize: 14, fontWeight: FontWeight.w600,
                    color: Colors.white)),
            Row(children: [
              Container(width: 7, height: 7,
                decoration: const BoxDecoration(
                    color: Color(0xFF4ADE80), shape: BoxShape.circle),
              ),
              const SizedBox(width: 5),
              Text(
                chat.isTyping ? 'Thinking...'
                    : _availableDocs.isNotEmpty
                        ? 'RAG Active • ${_availableDocs.length} docs'
                        : 'Gemini Free',
                style: GoogleFonts.lato(
                    fontSize: 11, color: Colors.white60)),
            ]),
          ],
        )),
      ]),
      actions: [
        // Docs count badge
        if (_availableDocs.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: GestureDetector(
              onTap: _toggleDocPanel,
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: _showDocPanel
                      ? AppTheme.accent
                      : Colors.white.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: _showDocPanel
                          ? AppTheme.accent
                          : Colors.white.withOpacity(0.25)),
                ),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.folder_rounded,
                      color: _showDocPanel
                          ? const Color(0xFF1A1A1A) : Colors.white,
                      size: 13),
                  const SizedBox(width: 4),
                  Text('${_availableDocs.length}',
                      style: GoogleFonts.lato(
                          fontSize: 12, fontWeight: FontWeight.w700,
                          color: _showDocPanel
                              ? const Color(0xFF1A1A1A) : Colors.white)),
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
      return const Center(child: CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation(AppTheme.primary)));
    }

    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      itemCount: chat.messages.length,
      itemBuilder: (_, i) {
        final msg      = chat.messages[i];
        final prev     = i > 0 ? chat.messages[i - 1] : null;
        final showTime = prev == null ||
            msg.createdAt.difference(prev.createdAt).inMinutes > 5;
        final showAvatar = !msg.isUser &&
            (i == chat.messages.length - 1 ||
                chat.messages[i + 1].isUser);

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
    final isUser   = msg.isUser;
    final isMentor = msg.isMentor;

    return Padding(
      padding: EdgeInsets.only(
          bottom: 4,
          left: isUser ? 48 : 0,
          right: isUser ? 0 : 48),
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
                      width: 30, height: 30,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
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
                padding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: isUser
                      ? AppTheme.primary
                      : isMentor
                          ? AppTheme.mentorBubble
                          : Colors.white,
                  borderRadius: BorderRadius.only(
                    topLeft: const Radius.circular(20),
                    topRight: const Radius.circular(20),
                    bottomLeft:
                        Radius.circular(isUser ? 20 : 4),
                    bottomRight:
                        Radius.circular(isUser ? 4 : 20),
                  ),
                  boxShadow: [BoxShadow(
                      color: Colors.black.withOpacity(0.07),
                      blurRadius: 8, offset: const Offset(0, 2))],
                ),
                child: isUser
                    ? Text(msg.content,
                        style: GoogleFonts.lato(
                            color: Colors.white, fontSize: 14,
                            height: 1.5))
                    : MarkdownBody(
                        data: msg.content,
                        styleSheet: MarkdownStyleSheet(
                          p: GoogleFonts.lato(
                              fontSize: 14, height: 1.65,
                              color: isMentor ? Colors.white
                                  : const Color(0xFF111827)),
                          strong: GoogleFonts.lato(
                              fontWeight: FontWeight.w700,
                              color: isMentor ? Colors.white
                                  : AppTheme.primary),
                          listBullet: GoogleFonts.lato(
                              color: isMentor ? Colors.white
                                  : const Color(0xFF111827)),
                          h2: GoogleFonts.playfairDisplay(
                              fontSize: 16, fontWeight: FontWeight.w700,
                              color: AppTheme.primary),
                          h3: GoogleFonts.lato(fontSize: 14,
                              fontWeight: FontWeight.w700,
                              color: AppTheme.primary),
                          code: GoogleFonts.sourceCodePro(
                              fontSize: 13,
                              backgroundColor: const Color(0xFFF3F4F6)),
                          blockquoteDecoration: BoxDecoration(
                            color: AppTheme.primary.withOpacity(0.05),
                            borderRadius: BorderRadius.circular(4),
                            border: const Border(left: BorderSide(
                                color: AppTheme.primary, width: 3)),
                          ),
                        ),
                      ),
              ),
            ),
          ),

          // User avatar
          if (isUser)
            Padding(
              padding: const EdgeInsets.only(left: 8, bottom: 2),
              child: Container(
                width: 30, height: 30,
                decoration: BoxDecoration(
                    color: AppTheme.accent,
                    shape: BoxShape.circle),
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
        width: 30, height: 30,
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
          boxShadow: [BoxShadow(
              color: Colors.black.withOpacity(0.07),
              blurRadius: 8, offset: const Offset(0, 2))],
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          _animatedDot(0),
          const SizedBox(width: 5),
          _animatedDot(150),
          const SizedBox(width: 5),
          _animatedDot(300),
        ]),
      ),
    ]),
  );

  Widget _animatedDot(int delayMs) => TweenAnimationBuilder<double>(
    tween: Tween(begin: 0.0, end: 1.0),
    duration: const Duration(milliseconds: 700),
    curve: Curves.easeInOut,
    builder: (_, v, __) {
      final opacity = (v * 2 * 3.14159).abs() < 3.14159
          ? v : 1.0 - v;
      return Container(
        width: 9, height: 9,
        decoration: BoxDecoration(
            color: AppTheme.primary.withOpacity(0.3 + opacity * 0.7),
            shape: BoxShape.circle),
      );
    },
  );

  // ══════════════════════════════════════════════════════════
  // ATTACHED DOCS BAR
  // ══════════════════════════════════════════════════════════
  Widget _attachedBar() => Container(
    color: AppTheme.primary.withOpacity(0.04),
    padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
    child: Row(children: [
      const Icon(Icons.attach_file_rounded,
          color: AppTheme.primary, size: 14),
      const SizedBox(width: 6),
      Text('Attached:', style: GoogleFonts.lato(
          fontSize: 12, fontWeight: FontWeight.w600,
          color: AppTheme.primary)),
      const SizedBox(width: 8),
      Expanded(child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(children: _attachedDocs.map((doc) {
          final ti = _getTypeInfo(doc.docType);
          return Container(
            margin: const EdgeInsets.only(right: 8),
            padding: const EdgeInsets.symmetric(
                horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.08),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                  color: AppTheme.primary.withOpacity(0.25)),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Text(ti['emoji'] as String,
                  style: const TextStyle(fontSize: 12)),
              const SizedBox(width: 5),
              Text(doc.title, style: GoogleFonts.lato(
                  fontSize: 11, fontWeight: FontWeight.w600,
                  color: AppTheme.primary),
                  overflow: TextOverflow.ellipsis),
              const SizedBox(width: 6),
              GestureDetector(
                onTap: () => setState(
                    () => _attachedDocs.removeWhere((d) => d.id == doc.id)),
                child: const Icon(Icons.close_rounded,
                    size: 13, color: AppTheme.primary)),
            ]),
          );
        }).toList()),
      )),
    ]),
  );

  // ══════════════════════════════════════════════════════════
  // DOCUMENT PANEL
  // ══════════════════════════════════════════════════════════
  Widget _docPanel() => Container(
    constraints: const BoxConstraints(maxHeight: 260),
    decoration: BoxDecoration(
      color: Colors.white,
      boxShadow: [BoxShadow(
          color: Colors.black.withOpacity(0.08),
          blurRadius: 16, offset: const Offset(0, -4))],
    ),
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      // Header
      Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
        child: Row(children: [
          const Icon(Icons.folder_open_rounded,
              color: AppTheme.primary, size: 16),
          const SizedBox(width: 8),
          Text('Attach Documents', style: GoogleFonts.lato(
              fontSize: 14, fontWeight: FontWeight.w700,
              color: const Color(0xFF111827))),
          const Spacer(),
          if (_attachedDocs.isNotEmpty)
            TextButton(
              onPressed: () => setState(() => _showDocPanel = false),
              style: TextButton.styleFrom(
                  padding: EdgeInsets.zero,
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap),
              child: Text('Done (${_attachedDocs.length})',
                  style: GoogleFonts.lato(fontWeight: FontWeight.w700,
                      color: AppTheme.primary)),
            ),
        ]),
      ),

      if (_availableDocs.isEmpty)
        Padding(
          padding: const EdgeInsets.all(24),
          child: Column(children: [
            const Icon(Icons.folder_off_outlined,
                color: Color(0xFF9CA3AF), size: 36),
            const SizedBox(height: 8),
            Text('No documents uploaded yet',
                style: GoogleFonts.lato(
                    color: const Color(0xFF6B7280))),
            const SizedBox(height: 4),
            Text('Go to Documents tab to upload',
                style: GoogleFonts.lato(
                    fontSize: 12, color: const Color(0xFF9CA3AF))),
          ]),
        )
      else
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
            itemCount: _availableDocs.length,
            itemBuilder: (_, i) {
              final doc    = _availableDocs[i];
              final isSelected = _attachedDocs.any((d) => d.id == doc.id);
              final ti     = _getTypeInfo(doc.docType);
              final color  = ti['color'] as Color;

              return GestureDetector(
                onTap: () => setState(() {
                  if (isSelected) {
                    _attachedDocs.removeWhere((d) => d.id == doc.id);
                  } else {
                    _attachedDocs.add(doc);
                  }
                }),
                child: Container(
                  margin: const EdgeInsets.only(bottom: 8),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? AppTheme.primary.withOpacity(0.05)
                        : const Color(0xFFF9FAFB),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                        color: isSelected
                            ? AppTheme.primary : const Color(0xFFE5E7EB),
                        width: isSelected ? 2 : 1)),
                  child: Row(children: [
                    Container(width: 38, height: 38,
                      decoration: BoxDecoration(
                          color: color.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(10)),
                      child: Center(child: Text(ti['emoji'] as String,
                          style: const TextStyle(fontSize: 18)))),
                    const SizedBox(width: 12),
                    Expanded(child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(doc.title, style: GoogleFonts.lato(
                            fontSize: 13, fontWeight: FontWeight.w600,
                            color: const Color(0xFF111827))),
                        Row(children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                                color: doc.extractedText?.isNotEmpty == true
                                    ? const Color(0xFFF0FDF4)
                                    : const Color(0xFFFFF7ED),
                                borderRadius: BorderRadius.circular(6)),
                            child: Text(
                              doc.extractedText?.isNotEmpty == true
                                  ? '✅ Indexed' : '⚠️ No index',
                              style: GoogleFonts.lato(fontSize: 9,
                                  fontWeight: FontWeight.w600,
                                  color: doc.extractedText?.isNotEmpty == true
                                      ? const Color(0xFF16A34A)
                                      : const Color(0xFFD97706)))),
                        ]),
                      ],
                    )),
                    Icon(
                        isSelected
                            ? Icons.check_circle_rounded
                            : Icons.radio_button_unchecked_rounded,
                        color: isSelected
                            ? AppTheme.primary : const Color(0xFFD1D5DB),
                        size: 22),
                  ]),
                ),
              );
            },
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
      Text('Suggested questions', style: GoogleFonts.lato(
          fontSize: 11, fontWeight: FontWeight.w600,
          color: const Color(0xFF9CA3AF), letterSpacing: 0.3)),
      const SizedBox(height: 8),
      SizedBox(height: 40,
        child: ListView.separated(
          scrollDirection: Axis.horizontal,
          itemCount: _suggestions.length,
          separatorBuilder: (_, __) => const SizedBox(width: 8),
          itemBuilder: (_, i) {
            final s = _suggestions[i];
            return GestureDetector(
              onTap: () => _send(s['text'] as String),
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: const Color(0xFFF0F4FF),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: AppTheme.primary.withOpacity(0.2))),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Text(s['icon'] as String,
                      style: const TextStyle(fontSize: 13)),
                  const SizedBox(width: 6),
                  Text(s['text'] as String, style: GoogleFonts.lato(
                      fontSize: 12, fontWeight: FontWeight.w600,
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
        boxShadow: [BoxShadow(
            color: Colors.black.withOpacity(0.08),
            blurRadius: 16, offset: const Offset(0, -4))],
      ),
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
      child: SafeArea(top: false,
        child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
          // Attach button
          GestureDetector(
            onTap: _toggleDocPanel,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 42, height: 42,
              decoration: BoxDecoration(
                color: _showDocPanel || _attachedDocs.isNotEmpty
                    ? AppTheme.primary
                    : const Color(0xFFF3F5FB),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                    color: _showDocPanel || _attachedDocs.isNotEmpty
                        ? AppTheme.primary : const Color(0xFFE5E7EB)),
              ),
              child: Stack(alignment: Alignment.center, children: [
                Icon(Icons.attach_file_rounded,
                    color: _showDocPanel || _attachedDocs.isNotEmpty
                        ? Colors.white : const Color(0xFF9CA3AF),
                    size: 19),
                if (_attachedDocs.isNotEmpty)
                  Positioned(top: 5, right: 5,
                    child: Container(
                      width: 13, height: 13,
                      decoration: BoxDecoration(
                          color: AppTheme.accent,
                          shape: BoxShape.circle,
                          border: Border.all(
                              color: Colors.white, width: 1.5)),
                      child: Center(child: Text(
                          '${_attachedDocs.length}',
                          style: GoogleFonts.lato(fontSize: 7,
                              color: const Color(0xFF1A1A1A),
                              fontWeight: FontWeight.w700))))),
              ]),
            ),
          ),
          const SizedBox(width: 8),

          // Text field
          Expanded(
            child: Container(
              constraints: const BoxConstraints(maxHeight: 130),
              decoration: BoxDecoration(
                color: const Color(0xFFF3F5FB),
                borderRadius: BorderRadius.circular(22),
                border: Border.all(color: const Color(0xFFE5E7EB)),
              ),
              child: TextField(
                controller: _ctrl,
                focusNode: _focus,
                maxLines: 6, minLines: 1,
                textCapitalization: TextCapitalization.sentences,
                onSubmitted: isTyping ? null : (_) => _send(),
                decoration: InputDecoration(
                  hintText: _attachedDocs.isNotEmpty
                      ? 'Ask about attached documents...'
                      : 'Ask anything about your academics...',
                  hintStyle: GoogleFonts.lato(
                      color: const Color(0xFF9CA3AF), fontSize: 14),
                  border: InputBorder.none,
                  contentPadding: const EdgeInsets.symmetric(
                      horizontal: 18, vertical: 11),
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
              width: 42, height: 42,
              decoration: BoxDecoration(
                gradient: isTyping
                    ? null
                    : const LinearGradient(
                        colors: [Color(0xFF1A2B5F), Color(0xFF243680)]),
                color: isTyping ? const Color(0xFFE5E7EB) : null,
                borderRadius: BorderRadius.circular(12),
                boxShadow: isTyping ? null : [
                  BoxShadow(color: AppTheme.primary.withOpacity(0.4),
                      blurRadius: 8, offset: const Offset(0, 3))
                ],
              ),
              child: Icon(
                isTyping ? Icons.hourglass_bottom_rounded
                    : Icons.send_rounded,
                color: isTyping ? const Color(0xFF9CA3AF) : Colors.white,
                size: 19),
            ),
          ),
        ]),
      ),
    );
  }

  // ── Helpers ───────────────────────────────────────────────
  Map<String, dynamic> _getTypeInfo(String type) {
    const types = [
      {'value': 'timetable',         'emoji': '📅', 'color': Color(0xFF3B82F6)},
      {'value': 'academic_calendar', 'emoji': '🗓️', 'color': Color(0xFF8B5CF6)},
      {'value': 'syllabus',          'emoji': '📚', 'color': Color(0xFF10B981)},
      {'value': 'marksheet',         'emoji': '📊', 'color': Color(0xFFF59E0B)},
      {'value': 'attendance',        'emoji': '✅', 'color': Color(0xFF06B6D4)},
      {'value': 'assignment',        'emoji': '📝', 'color': Color(0xFFEF4444)},
      {'value': 'other',             'emoji': '🏛️', 'color': Color(0xFF6B7280)},
    ];
    return types.firstWhere((t) => t['value'] == type,
        orElse: () => types.last);
  }
}