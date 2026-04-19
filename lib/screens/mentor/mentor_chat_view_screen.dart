// lib/screens/mentor/mentor_chat_view_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/auth_provider.dart';
import '../../services/chat_provider.dart';
import '../../utils/app_theme.dart';

class MentorChatViewScreen extends StatefulWidget {
  final ConversationModel conversation;
  const MentorChatViewScreen({super.key, required this.conversation});

  @override
  State<MentorChatViewScreen> createState() => _MentorChatViewScreenState();
}

class _MentorChatViewScreenState extends State<MentorChatViewScreen> {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  bool _isMentorMode = false; // When true, mentor is responding directly

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMentorMessage() async {
    final text = _messageController.text.trim();
    if (text.isEmpty) return;

    _messageController.clear();
    final auth = context.read<AuthProvider>();
    await context.read<ChatProvider>().sendMentorMessage(
          text,
          auth.currentUser!.id,
          widget.conversation.id,
        );
    _scrollToBottom();
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatProvider>();
    final auth = context.watch<AuthProvider>();
    final conv = widget.conversation;

    if (chat.messages.isNotEmpty) _scrollToBottom();

    return Scaffold(
      backgroundColor: AppTheme.surface,
      appBar: _buildAppBar(conv, chat, auth),
      body: Column(
        children: [
          _buildStudentInfoBar(conv),
          Expanded(
            child: chat.isLoading
                ? const Center(child: CircularProgressIndicator())
                : _buildMessageList(chat),
          ),
          if (_isMentorMode) _buildMentorInputArea(),
        ],
      ),
      bottomNavigationBar: !_isMentorMode
          ? _buildMentorControlBar(conv, auth, chat)
          : null,
    );
  }

  PreferredSizeWidget _buildAppBar(
      ConversationModel conv, ChatProvider chat, AuthProvider auth) {
    return AppBar(
      backgroundColor: AppTheme.primary,
      leading: IconButton(
        icon: const Icon(Icons.arrow_back, color: Colors.white),
        onPressed: () {
          chat.clearCurrentConversation();
          Navigator.of(context).pop();
        },
      ),
      title: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            conv.studentName ?? 'Student Conversation',
            style: GoogleFonts.playfairDisplay(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: Colors.white),
          ),
          Text(
            'Mentor View',
            style: GoogleFonts.lato(fontSize: 11, color: Colors.white70),
          ),
        ],
      ),
      actions: [
        // Status chip
        Container(
          margin: const EdgeInsets.only(right: 8),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: conv.isResolved
                ? AppTheme.success.withOpacity(0.2)
                : conv.isFlagged
                    ? AppTheme.warning.withOpacity(0.2)
                    : Colors.white.withOpacity(0.15),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            conv.status.toUpperCase(),
            style: GoogleFonts.lato(
              fontSize: 10,
              fontWeight: FontWeight.w700,
              color: conv.isResolved
                  ? const Color(0xFF81C995)
                  : conv.isFlagged
                      ? AppTheme.warning
                      : Colors.white70,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStudentInfoBar(ConversationModel conv) {
    if (conv.studentName == null) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      color: AppTheme.primary.withOpacity(0.05),
      child: Row(
        children: [
          const Icon(Icons.person_outline, size: 16, color: AppTheme.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              '${conv.studentName} • ${conv.studentProgram ?? ''} ${conv.studentBranch ?? ''} • Sem ${conv.studentSemester ?? ''}',
              style: GoogleFonts.lato(
                fontSize: 13,
                color: AppTheme.primary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageList(ChatProvider chat) {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      itemCount: chat.messages.length,
      itemBuilder: (context, index) {
        final msg = chat.messages[index];
        final showTime = index == 0 ||
            chat.messages[index].createdAt
                    .difference(chat.messages[index - 1].createdAt)
                    .inMinutes >
                5;

        return Column(
          children: [
            if (showTime)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Text(
                  DateFormat('h:mm a').format(msg.createdAt.toLocal()),
                  style: GoogleFonts.lato(
                      fontSize: 11, color: AppTheme.textSecondary),
                ),
              ),
            _buildMessageBubble(msg),
          ],
        );
      },
    );
  }

  Widget _buildMessageBubble(MessageModel msg) {
    final isStudent = msg.isUser;
    final isMentor = msg.isMentor;
    final isAI = msg.isAssistant;

    Color bubbleColor = isStudent
        ? AppTheme.userBubble
        : isMentor
            ? AppTheme.mentorBubble
            : Colors.white;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment:
            isStudent ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!isStudent) ...[
            CircleAvatar(
              radius: 16,
              backgroundColor: isMentor ? AppTheme.mentorBubble : AppTheme.primary,
              child: Icon(
                isMentor ? Icons.admin_panel_settings : Icons.psychology,
                color: Colors.white,
                size: 16,
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Column(
              crossAxisAlignment: isStudent
                  ? CrossAxisAlignment.end
                  : CrossAxisAlignment.start,
              children: [
                if (isMentor)
                  Padding(
                    padding: const EdgeInsets.only(left: 4, bottom: 2),
                    child: Text(
                      '👤 Mentor (direct)',
                      style: GoogleFonts.lato(
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.mentorBubble,
                      ),
                    ),
                  ),
                if (isAI)
                  Padding(
                    padding: const EdgeInsets.only(left: 4, bottom: 2),
                    child: Text(
                      '🤖 AI Response',
                      style: GoogleFonts.lato(
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.primary,
                      ),
                    ),
                  ),
                Container(
                  constraints: BoxConstraints(
                    maxWidth: MediaQuery.of(context).size.width * 0.75,
                  ),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: bubbleColor,
                    borderRadius: BorderRadius.only(
                      topLeft: const Radius.circular(18),
                      topRight: const Radius.circular(18),
                      bottomLeft: Radius.circular(isStudent ? 18 : 4),
                      bottomRight: Radius.circular(isStudent ? 4 : 18),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.06),
                        blurRadius: 6,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: isStudent
                      ? Text(
                          msg.content,
                          style: GoogleFonts.lato(
                              color: Colors.white, fontSize: 14, height: 1.5),
                        )
                      : MarkdownBody(
                          data: msg.content,
                          styleSheet: MarkdownStyleSheet(
                            p: GoogleFonts.lato(
                              fontSize: 14,
                              height: 1.6,
                              color: isMentor
                                  ? Colors.white
                                  : AppTheme.textPrimary,
                            ),
                            strong: GoogleFonts.lato(
                              fontWeight: FontWeight.w700,
                              color: isMentor
                                  ? Colors.white
                                  : AppTheme.primary,
                            ),
                          ),
                        ),
                ),
              ],
            ),
          ),
          if (isStudent) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              radius: 16,
              backgroundColor: AppTheme.accent,
              child: const Icon(Icons.person, color: Colors.white, size: 16),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildMentorControlBar(
      ConversationModel conv, AuthProvider auth, ChatProvider chat) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 12,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: !conv.isResolved
                    ? () async {
                        await chat.resolveConversation(
                            conv.id, auth.currentUser!.id);
                        if (mounted) Navigator.of(context).pop();
                      }
                    : null,
                icon: const Icon(Icons.check_circle_outline, size: 16),
                label: const Text('Resolve'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.success,
                  side: const BorderSide(color: AppTheme.success),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: !conv.isFlagged && !conv.isResolved
                    ? () async {
                        await chat.flagConversation(
                            conv.id, auth.currentUser!.id);
                      }
                    : null,
                icon: const Icon(Icons.flag_outlined, size: 16),
                label: const Text('Flag'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.warning,
                  side: const BorderSide(color: AppTheme.warning),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
            const SizedBox(width: 10),
            ElevatedButton.icon(
              onPressed: () => setState(() => _isMentorMode = true),
              icon: const Icon(Icons.reply, size: 16),
              label: const Text('Reply'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMentorInputArea() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(
          top: BorderSide(color: AppTheme.mentorBubble.withOpacity(0.3), width: 2),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 12,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Column(
          children: [
            Row(
              children: [
                const Icon(Icons.admin_panel_settings,
                    color: AppTheme.mentorBubble, size: 16),
                const SizedBox(width: 6),
                Text(
                  'Mentor Direct Reply',
                  style: GoogleFonts.lato(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.mentorBubble,
                  ),
                ),
                const Spacer(),
                GestureDetector(
                  onTap: () => setState(() {
                    _isMentorMode = false;
                    _messageController.clear();
                  }),
                  child: Icon(Icons.close,
                      size: 18, color: AppTheme.textSecondary),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: AppTheme.mentorBubble.withOpacity(0.05),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                          color: AppTheme.mentorBubble.withOpacity(0.3)),
                    ),
                    child: TextField(
                      controller: _messageController,
                      maxLines: 4,
                      minLines: 1,
                      textCapitalization: TextCapitalization.sentences,
                      decoration: InputDecoration(
                        hintText: 'Type your direct message to student...',
                        hintStyle: GoogleFonts.lato(
                            color: AppTheme.textSecondary, fontSize: 14),
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(
                            horizontal: 20, vertical: 12),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Container(
                  width: 50,
                  height: 50,
                  decoration: const BoxDecoration(
                    color: AppTheme.mentorBubble,
                    shape: BoxShape.circle,
                  ),
                  child: IconButton(
                    onPressed: _sendMentorMessage,
                    icon: const Icon(Icons.send, color: Colors.white, size: 20),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}