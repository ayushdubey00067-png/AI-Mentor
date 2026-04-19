// lib/services/chat_provider.dart
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:intl/intl.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/models.dart';
import 'supabase_service.dart';
import 'ai_service.dart';

class ChatProvider extends ChangeNotifier {
  ConversationModel?      _currentConversation;
  List<MessageModel>      _messages       = [];
  List<ConversationModel> _conversations  = [];
  List<UserModel>         _myStudents     = [];
  List<StudentProgressReport> _progressReports = [];
  UserModel?              _currentUser;

  bool _isTyping    = false;
  bool _isLoading   = false;
  bool _loadingProgress = false;

  RealtimeChannel? _messageChannel;

  ConversationModel?          get currentConversation => _currentConversation;
  List<MessageModel>          get messages            => _messages;
  List<ConversationModel>     get conversations       => _conversations;
  List<UserModel>             get myStudents          => _myStudents;
  List<StudentProgressReport> get progressReports     => _progressReports;
  bool get isTyping         => _isTyping;
  bool get isLoading        => _isLoading;
  bool get loadingProgress  => _loadingProgress;
  UserModel? get currentUser => _currentUser;

  void setCurrentUser(UserModel? user) {
    _currentUser = user;
    notifyListeners();
  }

  // ── Student: Start new conversation ───────────────────────
  Future<void> startNewConversation(String studentId, {String? mentorEmail}) async {
    _isLoading = true; notifyListeners();
    try {
      final conv = await SupabaseService.createConversation(
          studentId, mentorEmail: mentorEmail);
      _currentConversation = conv;
      _messages = [];

      final greeting = AIService.generateGreeting();
      final msg = await SupabaseService.sendMessage(
        conversationId: conv.id, content: greeting,
        senderRole: 'assistant', isAiGenerated: true,
      );
      _messages.add(msg);
      await SupabaseService.markFirstMessageDone(conv.id);
      _currentConversation = _rebuild(conv, isFirstDone: true);
      _subscribeMessages(conv.id);
    } catch (e) {
      debugPrint('❌ startNewConversation: $e');
    } finally {
      _isLoading = false; notifyListeners();
    }
  }

  Future<void> loadConversation(String conversationId, String studentId) async {
    _isLoading = true; notifyListeners();
    try {
      final convs = await SupabaseService.getStudentConversations(studentId);
      _currentConversation = convs.firstWhere((c) => c.id == conversationId);
      _messages = await SupabaseService.getMessages(conversationId);
      _subscribeMessages(conversationId);
    } finally {
      _isLoading = false; notifyListeners();
    }
  }

  Future<void> loadStudentConversations(String studentId) async {
    _conversations = await SupabaseService.getStudentConversations(studentId);
    notifyListeners();
  }

  // ── Student: Send message with FULL RAG PIPELINE ──────────
  Future<void> sendStudentMessage(
      String content, String studentId,
      {List<StudentDocument>? attachedDocs}) async {
    if (_currentConversation == null || content.trim().isEmpty) return;

    // 1. Save user message immediately
    final userMsg = await SupabaseService.sendMessage(
      conversationId: _currentConversation!.id,
      content: content.trim(), senderRole: 'user', senderId: studentId,
    );
    _messages.add(userMsg);
    _isTyping = true; notifyListeners();

    try {
      // 2. Extract student details if first time
      if (!_currentConversation!.studentDetailsCollected) {
        final det = await AIService.extractStudentDetails(content);
        if (det != null && det['name']!.isNotEmpty) {
          await SupabaseService.saveStudentDetails(
            conversationId: _currentConversation!.id,
            name: det['name']!, program: det['program']!,
            branch: det['branch']!, semester: det['semester']!,
          );
          _currentConversation = _rebuild(_currentConversation!,
            studentName: det['name'], studentProgram: det['program'],
            studentBranch: det['branch'], studentSemester: det['semester'],
            detailsCollected: true,
            title: '${det['name']} - ${det['program']}',
          );
        }
      }

      // 3. ═══ RAG PIPELINE ═══
      String ragContext = '';

      // 3a. If specific docs are attached, use them directly
      if (attachedDocs != null && attachedDocs.isNotEmpty) {
        final contextParts = <String>[];
        for (final doc in attachedDocs) {
          final full = await SupabaseService.getDocumentWithContent(doc.id);
          if (full?.extractedText?.isNotEmpty == true) {
            contextParts.add(
                '=== ${StudentDocument.typeLabel(doc.docType)}: ${doc.title} ===\n'
                '${full!.extractedText}');
          }
        }
        ragContext = contextParts.join('\n\n');
        debugPrint('📎 Using ${attachedDocs.length} attached docs for context');
      } else {
        // 3b. Vector search — find relevant chunks from all student docs
        try {
          final queryEmbedding = await AIService.createEmbedding(content);
          final chunks = await SupabaseService.searchSimilarChunks(
            studentId:      studentId,
            queryEmbedding: queryEmbedding,
            limit:          5,
            minSimilarity:  0.45,
          );
          if (chunks.isNotEmpty) {
            ragContext = chunks.join('\n\n---\n\n');
            debugPrint('🔍 RAG: Injecting ${chunks.length} chunks as context');
          }
        } catch (e) {
          debugPrint('⚠️ RAG embedding/search failed (continuing without): $e');
          // Continue without RAG — Gemini will still answer from general knowledge
        }
      }
      
      // 3c. ═══ STRUCTURED ACADEMIC LOOKUP [NEW] ═══
      final lowerMsg = content.toLowerCase();
      if (lowerMsg.contains('timetable') || lowerMsg.contains('schedule')) {
        final day = DateFormat('EEEE').format(DateTime.now());
        final todaySched = await SupabaseService.getTimetable(studentId, day);
        if (todaySched.isNotEmpty) {
          ragContext += '\n\n[TODAY\'S TIMETABLE ($day)]\n${jsonEncode(todaySched)}';
        }
      }
      if (lowerMsg.contains('marks') || lowerMsg.contains('result') || lowerMsg.contains('attendance')) {
        final records = await SupabaseService.getAcademicRecord(studentId, rollNo: _currentConversation?.studentRollNo);
        if (records.isNotEmpty) {
          final attendance = records.where((r) => r['record_type'] == 'attendance').toList();
          final results = records.where((r) => r['record_type'] == 'result').toList();

          if (attendance.isNotEmpty) {
            ragContext += '\n\n[OFFICIAL ATTENDANCE RECORDS]:\n';
            for (var r in attendance) {
              final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown Subject';
              final att = r['attendance_percentage'] ?? 'N/A';
              ragContext += '- $sub: $att% attendance (Status: ${r['status'] ?? 'N/A'})\n';
              if (r['total_classes'] != null) {
                ragContext += '  [Details: ${r['attended_classes']}/${r['total_classes']} classes]\n';
              }
            }
          }

          if (results.isNotEmpty) {
            ragContext += '\n\n[OFFICIAL ACADEMIC RESULTS/MARKS]:\n';
            for (var r in results) {
              final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown Subject';
              final marks = r['marks_obtained'] ?? r['marks'] ?? 'N/A';
              final total = r['total_marks'] ?? 'N/A';
              final grade = r['grade'] ?? 'N/A';
              final exam  = r['exam_type'] ?? 'Examination';
              ragContext += '- $sub ($exam): Marks $marks/$total, Grade: $grade\n';
            }
          }
        }
      }

      // 4. Send to Gemini with RAG context
      final aiText = await AIService.sendStudentMessage(
        history:      _messages,
        newMessage:   content.trim(),
        studentName:  _currentConversation!.studentName,
        rollNo:       _currentConversation!.studentRollNo,
        dept:         _currentConversation!.studentDept,
        program:      _currentConversation!.studentProgram,
        branch:       _currentConversation!.studentBranch,
        semester:     _currentConversation!.studentSemester,
        skills:       _currentConversation!.studentSkills,
        interests:    _currentConversation!.studentInterests,
        ragContext:   ragContext.isNotEmpty ? ragContext : null,
      );

      // 5. Save AI response
      final aiMsg = await SupabaseService.sendMessage(
        conversationId: _currentConversation!.id,
        content: aiText, senderRole: 'assistant', isAiGenerated: true,
      );
      _messages.add(aiMsg);

    } catch (e) {
      debugPrint('❌ sendStudentMessage error: $e');
      final errMsg = await SupabaseService.sendMessage(
        conversationId: _currentConversation!.id,
        content: AIService.friendlyError(e.toString()),
        senderRole: 'assistant', isAiGenerated: true,
      );
      _messages.add(errMsg);
    } finally {
      _isTyping = false; notifyListeners();
    }
  }

  // ── Mentor ─────────────────────────────────────────────────
  Future<void> loadMentorDashboard(String mentorEmail) async {
    _isLoading = true; notifyListeners();
    try {
      _myStudents    = await SupabaseService.getMyStudents(mentorEmail);
      _conversations = await SupabaseService.getMentorConversations(mentorEmail);
    } finally {
      _isLoading = false; notifyListeners();
    }
  }

  Future<void> loadAllConversations() async {
    _conversations = await SupabaseService.getAllConversations();
    notifyListeners();
  }

  Future<void> loadProgressReports(String mentorEmail) async {
    _loadingProgress = true; notifyListeners();
    try {
      final students = await SupabaseService.getMyStudents(mentorEmail);
      final reports  = <StudentProgressReport>[];
      for (final s in students) {
        reports.add(await SupabaseService.getStudentProgress(s));
      }
      _progressReports = reports;
    } finally {
      _loadingProgress = false; notifyListeners();
    }
  }

  Future<void> loadConversationForMentor(String conversationId) async {
    _isLoading = true; notifyListeners();
    try {
      final convs = await SupabaseService.getAllConversations();
      _currentConversation = convs.firstWhere((c) => c.id == conversationId);
      _messages = await SupabaseService.getMessages(conversationId);
      _subscribeMessages(conversationId);
    } finally {
      _isLoading = false; notifyListeners();
    }
  }

  Future<void> sendMentorMessage(String content, String mentorId, String convId) async {
    final msg = await SupabaseService.sendMessage(
      conversationId: convId, content: content.trim(),
      senderRole: 'mentor', senderId: mentorId,
    );
    _messages.add(msg); notifyListeners();
    await SupabaseService.logMentorIntervention(
        conversationId: convId, mentorId: mentorId, type: 'takeover');
  }

  // ── Mentor AI Assistant (Smart Academic Analytics) ────────
  Future<String> sendMentorAiMessage({
    required List<Map<String, dynamic>> history,
    required String newMessage,
    required String mentorName,
  }) async {
    String ragContext = '';

    // 1. Detect if mentor is asking about a specific student
    // Split by spaces, commas, or colons
    final words = newMessage.split(RegExp(r'[\s,:]+')).where((w) => w.length > 2).toList();
    
    // Sort words by length descending (longer words are more likely to be unique IDs/names)
    words.sort((a, b) => b.length.compareTo(a.length));

    for (final word in words) {
      final student = await SupabaseService.findStudentByQuery(word, 
          mentorEmail: _myStudents.isNotEmpty ? _myStudents.first.mentorEmail : null);
          
      if (student != null) {
        final records = await SupabaseService.getAcademicRecord(student.id, rollNo: student.rollNumber);
        
        ragContext += '\n[!!! CRITICAL: OFFICIAL_COLLEGE_DATABASE_CONTENT !!!]\n';
        ragContext += '[DB_SOURCE]: Supabase Verified\n';
        ragContext += '[STUDENT_PROFILE]:\n';
        ragContext += '- REAL_NAME: ${student.name}\n';
        ragContext += '- ROLL_NUMBER: ${student.rollNumber ?? 'N/A'}\n';
        ragContext += '- PROGRAM: ${student.program ?? 'N/A'}\n';
        ragContext += '- SEMESTER: ${student.semester ?? 'N/A'}\n';
        
        if (records.isNotEmpty) {
          final attendance = records.where((r) => r['record_type'] == 'attendance').toList();
          final results = records.where((r) => r['record_type'] == 'result').toList();

          if (attendance.isNotEmpty) {
            ragContext += '\n[ACADEMIC_ATTENDANCE_TRANSCRIPT]:\n';
            for (var r in attendance) {
              final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown';
              final att = r['attendance_percentage'] ?? 'N/A';
              ragContext += '- SUBJECT: ${sub.toUpperCase()} | ATTENDANCE: $att% | CLASSES: ${r['attended_classes']}/${r['total_classes']}\n';
            }
          }

          if (results.isNotEmpty) {
            ragContext += '\n[ACADEMIC_RESULTS_MARKS_TRANSCRIPT]:\n';
            for (var r in results) {
              final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown';
              final marks = r['marks_obtained'] ?? r['marks'] ?? 'N/A';
              final total = r['total_marks'] ?? 'N/A';
              final grade = r['grade'] ?? 'N/A';
              final exam  = r['exam_type'] ?? 'Examination';
              ragContext += '- SUBJECT: ${sub.toUpperCase()} | EXAM: $exam | MARKS: $marks/$total | GRADE: $grade\n';
            }
          }

          ragContext += '\n[STRICT_INSTRUCTION]: Use ONLY the subjects and data listed above. If a subject (like OS) is not in the list above, do NOT mention it.\n';
        } else {
          ragContext += '\n[ALERT]: NO SUBJECT-WISE RECORDS FOUND IN DATABASE FOR THIS STUDENT ACCOUNT.\n';
        }
        ragContext += '[!!! END_OF_DATABASE_CONTENT !!!]\n';
        break; 
      }
    }

    return await AIService.sendMentorMessage(
      history:       history,
      newMessage:    newMessage,
      mentorName:    mentorName,
      designation:   _currentUser?.designation,
      dept:          _currentUser?.department,
      expertise:     _currentUser?.expertise,
      totalStudents: _myStudents.length,
      activeChats:   _conversations.where((c) => c.status == 'active').length,
      ragContext:    ragContext.isNotEmpty ? ragContext : null,
    );
  }

  Future<void> flagConversation(String convId, String mentorId) async {
    await SupabaseService.updateConversationStatus(convId, 'flagged');
    await SupabaseService.logMentorIntervention(
        conversationId: convId, mentorId: mentorId, type: 'flag');
    notifyListeners();
  }

  Future<void> resolveConversation(String convId, String mentorId) async {
    await SupabaseService.updateConversationStatus(convId, 'resolved');
    await SupabaseService.logMentorIntervention(
        conversationId: convId, mentorId: mentorId, type: 'resolve');
    notifyListeners();
  }

  Future<void> deleteConversation(String convId) async {
    try {
      await SupabaseService.deleteConversation(convId);
      _conversations.removeWhere((c) => c.id == convId);
      if (_currentConversation?.id == convId) {
        _currentConversation = null;
        _messages = [];
      }
      notifyListeners();
    } catch (e) {
      debugPrint('❌ deleteConversation: $e');
    }
  }

  // ── Realtime ───────────────────────────────────────────────
  void _subscribeMessages(String conversationId) {
    _messageChannel?.unsubscribe();
    _messageChannel = SupabaseService.subscribeToMessages(conversationId, (msg) {
      if (!_messages.any((m) => m.id == msg.id)) {
        _messages.add(msg); notifyListeners();
      }
    });
  }

  void clearCurrentConversation() {
    _messageChannel?.unsubscribe();
    _currentConversation = null; _messages = [];
    notifyListeners();
  }

  ConversationModel _rebuild(ConversationModel b, {
    bool? isFirstDone, bool? detailsCollected,
    String? studentName, String? studentProgram,
    String? studentBranch, String? studentSemester, String? title,
  }) => ConversationModel.fromMap({
    'id': b.id, 'student_id': b.studentId,
    'title': title ?? b.title,
    'is_first_message_done': isFirstDone ?? b.isFirstMessageDone,
    'student_details_collected': detailsCollected ?? b.studentDetailsCollected,
    'student_name': studentName ?? b.studentName,
    'student_program': studentProgram ?? b.studentProgram,
    'student_branch': studentBranch ?? b.studentBranch,
    'student_semester': studentSemester ?? b.studentSemester,
    'mentor_email': b.mentorEmail, 'status': b.status,
    'created_at': b.createdAt.toIso8601String(),
    'updated_at': DateTime.now().toIso8601String(),
  });

  @override
  void dispose() { _messageChannel?.unsubscribe(); super.dispose(); }
}