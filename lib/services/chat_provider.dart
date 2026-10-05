// lib/services/chat_provider.dart
import 'package:flutter/foundation.dart';
import 'package:intl/intl.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/models.dart';
import 'supabase_service.dart';
import 'ai_service.dart';
import '../utils/mru_timetable_data.dart';

class ChatProvider extends ChangeNotifier {
  ConversationModel? _currentConversation;
  List<MessageModel> _messages = [];
  List<ConversationModel> _conversations = [];
  List<UserModel> _myStudents = [];
  List<StudentProgressReport> _progressReports = [];
  UserModel? _currentUser;

  bool _isTyping = false;
  bool _isLoading = false;
  bool _loadingProgress = false;

  RealtimeChannel? _messageChannel;

  ConversationModel? get currentConversation => _currentConversation;
  List<MessageModel> get messages => _messages;
  List<ConversationModel> get conversations => _conversations;
  List<UserModel> get myStudents => _myStudents;
  List<StudentProgressReport> get progressReports => _progressReports;
  bool get isTyping => _isTyping;
  bool get isLoading => _isLoading;
  bool get loadingProgress => _loadingProgress;
  UserModel? get currentUser => _currentUser;

  void setCurrentUser(UserModel? user) {
    _currentUser = user;
    notifyListeners();
  }

  // ── Student: Start new conversation ───────────────────────
  Future<void> startNewConversation(String studentId,
      {String? mentorEmail}) async {
    _isLoading = true;
    notifyListeners();
    try {
      final conv = await SupabaseService.createConversation(
        studentId,
        mentorEmail: mentorEmail,
        studentName: _currentUser?.name,
        studentRollNo: _currentUser?.rollNumber,
        studentProgram: _currentUser?.program,
        studentBranch: _currentUser?.branch,
        studentSemester: _currentUser?.semester,
        studentDetailsCollected:
            _currentUser?.name != null && _currentUser!.name.isNotEmpty,
      );
      _currentConversation = conv;
      _messages = [];

      final greeting = AIService.generateGreeting(
        _currentUser?.name,
        branch: _currentUser?.branch,
        semester: _currentUser?.semester,
      );
      final msg = await SupabaseService.sendMessage(
        conversationId: conv.id,
        content: greeting,
        senderRole: 'assistant',
        isAiGenerated: true,
      );
      _messages.add(msg);
      await SupabaseService.markFirstMessageDone(conv.id);
      _currentConversation = _rebuild(conv, isFirstDone: true);
      _subscribeMessages(conv.id);
    } catch (e) {
      debugPrint('❌ startNewConversation: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> loadConversation(String conversationId, String studentId) async {
    _isLoading = true;
    notifyListeners();
    try {
      final convs = await SupabaseService.getStudentConversations(studentId);
      _currentConversation = convs.firstWhere((c) => c.id == conversationId);
      _messages = await SupabaseService.getMessages(conversationId);
      _subscribeMessages(conversationId);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> loadStudentConversations(String studentId) async {
    _conversations = await SupabaseService.getStudentConversations(studentId);
    notifyListeners();
  }

  // ── Student: Send message with FULL RAG PIPELINE ──────────
  Future<void> sendStudentMessage(String content, String studentId,
      {List<StudentDocument>? attachedDocs}) async {
    if (_currentConversation == null || content.trim().isEmpty) return;

    // 1. Save user message immediately
    final userMsg = await SupabaseService.sendMessage(
      conversationId: _currentConversation!.id,
      content: content.trim(),
      senderRole: 'user',
      senderId: studentId,
    );
    _messages.add(userMsg);

    // Use the first user message as a short, useful conversation title.
    if (_currentConversation!.title == 'New Conversation') {
      final title = _conversationTitle(content);
      await SupabaseService.updateConversation(
        _currentConversation!.id,
        {'title': title},
      );
      _currentConversation = _rebuild(_currentConversation!, title: title);
    } else {
      await SupabaseService.updateConversation(
        _currentConversation!.id,
        <String, dynamic>{},
      );
      _currentConversation = _rebuild(_currentConversation!);
    }
    _isTyping = true;
    notifyListeners();

    try {
      // 2. Extract student details only if completely missing and not collected
      if (!_currentConversation!.studentDetailsCollected &&
          (_currentConversation!.studentName == null ||
              _currentConversation!.studentName!.isEmpty)) {
        final det = await AIService.extractStudentDetails(content);
        if (det != null && det['name']!.isNotEmpty) {
          await SupabaseService.saveStudentDetails(
            conversationId: _currentConversation!.id,
            name: det['name']!,
            program: det['program']!,
            branch: det['branch']!,
            semester: det['semester']!,
          );
          _currentConversation = _rebuild(
            _currentConversation!,
            studentName: det['name'],
            studentProgram: det['program'],
            studentBranch: det['branch'],
            studentSemester: det['semester'],
            detailsCollected: true,
          );
        }
      }

      // 3. ═══ AUTONOMOUS MULTI-TOOL RETRIEVAL PIPELINE ═══
      String ragContext = '';
      final lowerMsg = content.toLowerCase();

      // 3a. Autonomous Timetable & Daily Class Schedule Retrieval (PRIMARY FOR SCHEDULES)
      final isTimetableTopic = lowerMsg.contains('timetable') ||
          lowerMsg.contains('time table') ||
          lowerMsg.contains('schedule') ||
          lowerMsg.contains('class') ||
          lowerMsg.contains('lecture') ||
          lowerMsg.contains('lab') ||
          lowerMsg.contains('tomorrow') ||
          lowerMsg.contains('today') ||
          lowerMsg.contains('monday') ||
          lowerMsg.contains('tuesday') ||
          lowerMsg.contains('wednesday') ||
          lowerMsg.contains('thursday') ||
          lowerMsg.contains('friday') ||
          lowerMsg.contains('period') ||
          lowerMsg.contains('slot') ||
          lowerMsg.contains('faculty') ||
          lowerMsg.contains('room');

      if (isTimetableTopic) {
        try {
          final now = DateTime.now();
          final todayName = DateFormat('EEEE').format(now);
          final tomorrowName = DateFormat('EEEE').format(now.add(const Duration(days: 1)));
          final todayDateStr = DateFormat('MMMM d, yyyy').format(now);
          final currentTimeStr = DateFormat('h:mm a').format(now);
          final currentHour = now.hour;
          final currentMinute = now.minute;
          final currentTotalMinutes = currentHour * 60 + currentMinute;
          final studentSection = MRUTimetableRepository.resolveSection(
            program: _currentUser?.program,
            branch: _currentUser?.branch,
            semester: _currentUser?.semester,
            section: _currentUser?.section,
          );

          final scheduleMap = MRUTimetableRepository.getClassSchedule(studentSection);

          int parseMinutes(String timeStr) {
            final parts = timeStr.trim().split(':');
            if (parts.length == 2) {
              return (int.tryParse(parts[0]) ?? 0) * 60 + (int.tryParse(parts[1]) ?? 0);
            }
            return 0;
          }

          // Build full weekly markdown timetable
          final sb = StringBuffer();
          sb.writeln('# Academic Timetable: $studentSection');
          sb.writeln('**Institution**: Manav Rachna University, Sector 43, Faridabad');
          for (final day in MRUTimetableRepository.days) {
            final dayFull = MRUTimetableRepository.dayNames[day] ?? day;
            sb.writeln('\n## $dayFull');
            sb.writeln('| Period | Time | Subject | Faculty | Room | Group | Type |');
            sb.writeln('| :--- | :--- | :--- | :--- | :--- | :--- | :--- |');
            final daySlots = scheduleMap[day] ?? {};
            for (final slot in MRUTimetableRepository.periodSlots) {
              if (slot.label == 'Lunch') {
                sb.writeln('| LUNCH | 11:30 - 12:20 | **LUNCH BREAK** | - | - | Entire class | Break |');
                continue;
              }
              final entries = daySlots[slot.period] ?? [];
              if (entries.isEmpty) {
                sb.writeln('| Period ${slot.label} | ${slot.startTime} - ${slot.endTime} | *[Free / Self-Study]* | - | - | Entire class | Free |');
              } else {
                for (final e in entries) {
                  final isLab = e.durationPeriods >= 2 || e.subject.toLowerCase().contains('lab');
                  final typeStr = isLab ? '100-Min Lab' : '50-Min Lecture';
                  sb.writeln('| Period ${slot.label} | ${slot.startTime} - ${slot.endTime} | **${e.subject}** | ${e.teachers.join(", ")} | ${e.rooms.join(", ")} | ${e.group.isNotEmpty ? e.group : "Entire class"} | $typeStr |');
                }
              }
            }
          }
          final fullTimetableText = sb.toString();

          // Calculate real-time breakdown for today
          final todayCode = MRUTimetableRepository.days.firstWhere(
            (d) => (MRUTimetableRepository.dayNames[d] ?? '').toLowerCase() == todayName.toLowerCase(),
            orElse: () => 'Mo',
          );
          final isWeekend = todayName == 'Saturday' || todayName == 'Sunday';
          final todaySlots = scheduleMap[todayCode] ?? {};

          final completedSlotsList = <String>[];
          final ongoingSlotsList = <String>[];
          final remainingSlotsList = <String>[];

          if (!isWeekend) {
            for (final slot in MRUTimetableRepository.periodSlots) {
              final startMin = parseMinutes(slot.startTime);
              final endMin = parseMinutes(slot.endTime);
              final isLunch = slot.label == 'Lunch';
              final entries = todaySlots[slot.period] ?? [];

              String desc;
              if (isLunch) {
                desc = '• 11:30 - 12:20 | Lunch Break';
              } else if (entries.isEmpty) {
                desc = '• ${slot.startTime} - ${slot.endTime} | Period ${slot.label}: Free Slot / Self-Study';
              } else {
                final tracks = entries.map((e) {
                  final grp = e.group.isNotEmpty ? ' (${e.group})' : '';
                  final rm = e.rooms.isNotEmpty ? ' in ${e.rooms.join(", ")}' : '';
                  final tchr = e.teachers.isNotEmpty ? ' with ${e.teachers.join(", ")}' : '';
                  return '**${e.subject}**$grp$rm$tchr';
                }).join(' OR ');
                desc = '• ${slot.startTime} - ${slot.endTime} | Period ${slot.label}: $tracks';
              }

              if (currentTotalMinutes >= endMin) {
                completedSlotsList.add('$desc [COMPLETED]');
              } else if (currentTotalMinutes >= startMin && currentTotalMinutes < endMin) {
                ongoingSlotsList.add('$desc [CURRENTLY ONGOING NOW]');
                remainingSlotsList.add('$desc [CURRENTLY ONGOING NOW]');
              } else {
                remainingSlotsList.add('$desc [UPCOMING]');
              }
            }
          }

          ragContext += '\n[OFFICIAL UNIVERSITY TIMETABLE & REAL-TIME ATTENDANCE CONTEXT]\n'
              '• Student Section: $studentSection\n'
              '• Current Live Clock: $todayName, $todayDateStr at $currentTimeStr (Tomorrow is $tomorrowName)\n'
              '• Live Status: ${isWeekend ? "Weekend (No classes scheduled today)" : (ongoingSlotsList.isNotEmpty ? "Currently in progress: ${ongoingSlotsList.first}" : (currentTotalMinutes < 490 ? "College hours have not started yet." : (currentTotalMinutes > 990 ? "College hours are over for today." : "Passing / transition period")))}\n\n'
              '═══ TODAY\'S REMAINING CLASSES TO ATTEND (FROM $currentTimeStr ONWARDS) ═══\n'
              '${isWeekend ? "Today is $todayName (Weekend). No more classes today." : (remainingSlotsList.isNotEmpty ? remainingSlotsList.join("\n") : "All scheduled classes for today have concluded.")}\n\n'
              '═══ COMPLETED CLASSES EARLIER TODAY (PRIOR TO $currentTimeStr) ═══\n'
              '${completedSlotsList.isNotEmpty ? completedSlotsList.join("\n") : "None"}\n\n'
              '═══ FULL WEEKLY TIMETABLE FOR $studentSection ═══\n'
              '$fullTimetableText\n'
              '═══════════════════════════════════════════════════════════\n'
              'CRITICAL REAL-TIME BEHAVIOR RULES:\n'
              '1. REAL-TIME REMAINING CLASSES: If the student asks "what more classes do I have to attend", "what classes are left today", "what is my next class", "classes left", or "from now onwards", ONLY list the upcoming and ongoing classes from $currentTimeStr onwards! Do not dump earlier completed classes unless the student explicitly asks for the "full day schedule" or "entire timetable".\n'
              '2. DIRECT HELPFUL ANSWER: NEVER say "I already provided your timetable in my previous response" or "please refer to that". Always answer immediately, concisely, and helpfully with their exact upcoming schedule.\n'
              '3. LAB AWARENESS: Highlight 100-minute lab sessions (and which group/room they are in, e.g. Group 1 or Group 2).\n'
              '[END OFFICIAL UNIVERSITY TIMETABLE]\n';
          debugPrint('📅 Timetable RAG: Injected live timetable for $studentSection at $currentTimeStr (Today: $todayName)');
        } catch (e) {
          debugPrint('⚠️ Timetable retrieval error: $e');
        }
      }

      // 3b. Autonomous Calendar & Holiday Inspection (HOLIDAYS / CALENDAR DATES ONLY)
      final isCalendarTopic = !isTimetableTopic && (lowerMsg.contains('calendar') ||
          lowerMsg.contains('holiday') ||
          lowerMsg.contains('vacation') ||
          lowerMsg.contains('break') ||
          lowerMsg.contains('exam date') ||
          lowerMsg.contains('datesheet') ||
          lowerMsg.contains('semester start') ||
          lowerMsg.contains('semester end'));

      if (isCalendarTopic) {
        try {
          final docs = await SupabaseService.getStudentAccessibleDocuments(
            studentId: studentId,
            rollNo: _currentConversation?.studentRollNo,
          );
          final calendarDocs = docs.where((d) => d.docType == 'academic_calendar').toList();
          if (calendarDocs.isNotEmpty) {
            final fullCal = await SupabaseService.getDocumentWithContent(calendarDocs.first.id);
            if (fullCal?.extractedText?.isNotEmpty == true) {
              ragContext += '\n[OFFICIAL ACADEMIC CALENDAR & HOLIDAY SCHEDULE (YEAR ${calendarDocs.first.academicYear ?? '2026'})]:\n'
                  '${fullCal!.extractedText}\n[END ACADEMIC CALENDAR]\n';
            }
          }
        } catch (e) {
          debugPrint('⚠️ Autonomous calendar retrieval: $e');
        }
      }

      // 3c. Autonomous Category-Partitioned Vector Search
      try {
        final contextMeta = _detectCategoryAndTemporalContext(content);
        final queryEmbedding = await AIService.createEmbedding(content);
        final chunks = await SupabaseService.searchSimilarChunks(
          studentId: studentId,
          rollNo: _currentConversation?.studentRollNo,
          docType: contextMeta['category'],
          academicYear: contextMeta['year'],
          term: contextMeta['term'],
          queryEmbedding: queryEmbedding,
          limit: 6,
          minSimilarity: 0.25,
        );
        if (chunks.isNotEmpty) {
          ragContext += '\n\n[OFFICIAL DEPARTMENT MATERIAL CHUNKS (${contextMeta['category']?.toUpperCase() ?? 'ACADEMIC'})]:\n' + chunks.join('\n\n---\n\n');
          debugPrint('🔍 Student Category RAG: Injected ${chunks.length} chunks for ${contextMeta['category'] ?? "general"}');
        }
      } catch (e) {
        debugPrint('⚠️ RAG embedding/search failed: $e');
      }
      if (lowerMsg.contains('marks') ||
          lowerMsg.contains('result') ||
          lowerMsg.contains('score') ||
          lowerMsg.contains('grade') ||
          lowerMsg.contains('sgpa') ||
          lowerMsg.contains('attendance')) {
        final rollNo = _currentConversation?.studentRollNo ?? _currentUser?.rollNumber;

        // 1. Structured Native JSON lookup from institutional marksheets & registers
        if (rollNo != null && rollNo.isNotEmpty) {
          try {
            final docs = await SupabaseService.getStudentAccessibleDocuments(
              studentId: studentId,
              rollNo: rollNo,
            );
            for (final doc in docs) {
              final fullDoc = await SupabaseService.getDocumentWithContent(doc.id);
              final extJson = fullDoc?.extractedJson ?? doc.extractedJson;
              if (extJson != null && extJson.isNotEmpty) {
                for (final pageKey in extJson.keys.where((k) => k.startsWith('page_'))) {
                  final pageData = extJson[pageKey] as Map<String, dynamic>? ?? {};
                  final students = (pageData['students'] as List?)?.map((e) => Map<String, dynamic>.from(e as Map)).toList() ?? [];
                  for (final s in students) {
                    final sRoll = (s['roll_no'] as String? ?? '').trim().toLowerCase();
                    final sName = (s['name'] as String? ?? '').trim().toLowerCase();
                    final currentName = (_currentUser?.name ?? '').trim().toLowerCase();

                    if (sRoll.contains(rollNo.trim().toLowerCase()) ||
                        rollNo.trim().toLowerCase().contains(sRoll) ||
                        (currentName.isNotEmpty && sName.contains(currentName))) {
                      final grades = (s['grades'] as Map?)?.entries.map((e) => '${e.key}: ${e.value}').join(', ') ?? '';
                      ragContext += '\n[OFFICIAL INSTITUTIONAL EXAMINATION RECORD FROM NATIVE JSON]:\n'
                          '- Roll Number: ${s['roll_no']}\n'
                          '- Student Name: ${s['name']}\n'
                          '- SGPA: ${s['sgpa']}\n'
                          '- Subject Grades: $grades\n'
                          '[END OF INSTITUTIONAL RECORD]\n';
                    }
                  }
                }
              }
            }
          } catch (_) {}
        }

        // 2. Relational database lookup (attendance & academic_results)
        final records = await SupabaseService.getAcademicRecord(studentId,
            rollNo: _currentConversation?.studentRollNo);
        if (records.isNotEmpty) {
          final attendance =
              records.where((r) => r['record_type'] == 'attendance').toList();
          final results =
              records.where((r) => r['record_type'] == 'result').toList();

          if (attendance.isNotEmpty) {
            ragContext += '\n\n[OFFICIAL ATTENDANCE RECORDS]:\n';
            for (var r in attendance) {
              final sub =
                  r['subject_name'] ?? r['subject_code'] ?? 'Unknown Subject';
              final att = r['attendance_percentage'] ?? 'N/A';
              ragContext +=
                  '- $sub: $att% attendance (Status: ${r['status'] ?? 'N/A'})\n';
              if (r['total_classes'] != null) {
                ragContext +=
                    '  [Details: ${r['attended_classes']}/${r['total_classes']} classes]\n';
              }
            }
          }

          if (results.isNotEmpty) {
            ragContext += '\n\n[OFFICIAL ACADEMIC RESULTS/MARKS]:\n';
            for (var r in results) {
              final sub =
                  r['subject_name'] ?? r['subject_code'] ?? 'Unknown Subject';
              final marks = r['marks_obtained'] ?? r['marks'] ?? 'N/A';
              final total = r['max_marks'] ?? r['total_marks'] ?? 'N/A';
              final grade = r['grade'] ?? 'N/A';
              final exam = r['exam_type'] ?? 'Examination';
              ragContext +=
                  '- $sub ($exam): Marks $marks/$total, Grade: $grade\n';
            }
          }
        }
      }

      // 4. Send to Gemini with RAG context and stream token chunks
      MessageModel? streamingMsg;
      final aiText = await AIService.sendStudentMessage(
        history: _messages.where((m) => m.id != streamingMsg?.id).toList(),
        newMessage: content.trim(),
        studentName: _currentConversation!.studentName ?? _currentUser?.name,
        rollNo: _currentConversation!.studentRollNo,
        dept: _currentConversation!.studentDept,
        program: _currentConversation!.studentProgram,
        branch: _currentConversation!.studentBranch,
        semester: _currentConversation!.studentSemester,
        section: _currentUser?.section ?? 'CSE 5A',
        skills: _currentConversation!.studentSkills,
        interests: _currentConversation!.studentInterests,
        ragContext: ragContext.isNotEmpty ? ragContext : null,
        onStreamChunk: (partial) {
          if (_isTyping) {
            _isTyping = false;
          }
          if (streamingMsg == null) {
            streamingMsg = MessageModel(
              id: 'temp_stream_${DateTime.now().millisecondsSinceEpoch}',
              conversationId: _currentConversation!.id,
              senderRole: 'assistant',
              content: partial,
              createdAt: DateTime.now(),
              isAiGenerated: true,
            );
            _messages.add(streamingMsg!);
          } else {
            final idx = _messages.indexWhere((m) => m.id == streamingMsg!.id);
            if (idx != -1) {
              _messages[idx] = MessageModel(
                id: streamingMsg!.id,
                conversationId: streamingMsg!.conversationId,
                senderRole: streamingMsg!.senderRole,
                content: partial,
                createdAt: streamingMsg!.createdAt,
                isAiGenerated: true,
              );
            }
          }
          notifyListeners();
        },
      );

      // 5. Save AI response to Supabase
      final aiMsg = await SupabaseService.sendMessage(
        conversationId: _currentConversation!.id,
        content: aiText,
        senderRole: 'assistant',
        isAiGenerated: true,
      );

      if (streamingMsg != null) {
        final idx = _messages.indexWhere((m) => m.id == streamingMsg!.id);
        if (idx != -1) {
          _messages[idx] = aiMsg;
        } else {
          _messages.add(aiMsg);
        }
      } else {
        _messages.add(aiMsg);
      }
    } catch (e) {
      debugPrint('❌ sendStudentMessage error: $e');
      final errMsg = await SupabaseService.sendMessage(
        conversationId: _currentConversation!.id,
        content: AIService.friendlyError(e.toString()),
        senderRole: 'assistant',
        isAiGenerated: true,
      );
      _messages.add(errMsg);
    } finally {
      _isTyping = false;
      notifyListeners();
    }
  }

  // ── Mentor ─────────────────────────────────────────────────
  Future<void> loadMentorDashboard(String mentorEmail) async {
    _isLoading = true;
    notifyListeners();
    try {
      _myStudents = await SupabaseService.getMyStudents(mentorEmail);
      _conversations =
          await SupabaseService.getMentorConversations(mentorEmail);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> loadAllConversations() async {
    _conversations = await SupabaseService.getAllConversations();
    notifyListeners();
  }

  Future<void> loadProgressReports(String mentorEmail) async {
    _loadingProgress = true;
    notifyListeners();
    try {
      final students = await SupabaseService.getMyStudents(mentorEmail);
      final reports = <StudentProgressReport>[];
      for (final s in students) {
        reports.add(await SupabaseService.getStudentProgress(s));
      }
      _progressReports = reports;
    } finally {
      _loadingProgress = false;
      notifyListeners();
    }
  }

  Future<void> loadConversationForMentor(String conversationId) async {
    _isLoading = true;
    notifyListeners();
    try {
      final convs = await SupabaseService.getAllConversations();
      _currentConversation = convs.firstWhere((c) => c.id == conversationId);
      _messages = await SupabaseService.getMessages(conversationId);
      _subscribeMessages(conversationId);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> sendMentorMessage(
      String content, String mentorId, String convId) async {
    final msg = await SupabaseService.sendMessage(
      conversationId: convId,
      content: content.trim(),
      senderRole: 'mentor',
      senderId: mentorId,
    );
    _messages.add(msg);
    notifyListeners();
    await SupabaseService.logMentorIntervention(
        conversationId: convId, mentorId: mentorId, type: 'takeover');
  }

  // ── Mentor AI Assistant (Smart Academic Analytics) ────────
  Future<String> sendMentorAiMessage({
    required List<Map<String, dynamic>> history,
    required String newMessage,
    required String mentorName,
    void Function(String partialText)? onStreamChunk,
  }) async {
    String ragContext = '';

    final normalizedMessage = newMessage.toLowerCase();
    final isStudentListRequest = RegExp(r'\b(list|roster|students?|class)\b')
            .hasMatch(normalizedMessage) &&
        RegExp(r'\b(student|students|class|assigned|roll|name)\b')
            .hasMatch(normalizedMessage);

    if (isStudentListRequest) {
      ragContext = '\n[ASSIGNED_STUDENT_LIST_FROM_MY_CLASS_TAB]\n'
          'This is the complete list of students assigned to this mentor.\n';
      if (_myStudents.isEmpty) {
        ragContext += '- No students are currently assigned.\n';
      } else {
        for (final student in _myStudents) {
          ragContext +=
              '- NAME: ${student.name} | ROLL_NUMBER: ${student.rollNumber ?? 'N/A'}\n';
        }
      }
      ragContext += '[END_ASSIGNED_STUDENT_LIST]\n';
    }

    // 1. Detect if mentor is asking about a specific student
    // Split by spaces, commas, or colons
    final words = isStudentListRequest
        ? <String>[]
        : newMessage
            .split(RegExp(r'[\s,:]+'))
            .where((w) => w.length > 2)
            .toList();

    // Sort words by length descending (longer words are more likely to be unique IDs/names)
    words.sort((a, b) => b.length.compareTo(a.length));

    for (final word in words) {
      final student = await SupabaseService.findStudentByQuery(word,
          mentorEmail: _currentUser?.email ??
              (_myStudents.isNotEmpty ? _myStudents.first.mentorEmail : null));

      if (student != null) {
        final records = await SupabaseService.getAcademicRecord(student.id,
            rollNo: student.rollNumber);

        ragContext +=
            '\n[!!! CRITICAL: OFFICIAL_COLLEGE_DATABASE_CONTENT !!!]\n';
        ragContext += '[DB_SOURCE]: Supabase Verified\n';
        ragContext += '[STUDENT_PROFILE]:\n';
        ragContext += '- REAL_NAME: ${student.name}\n';
        ragContext += '- ROLL_NUMBER: ${student.rollNumber ?? 'N/A'}\n';
        ragContext += '- PROGRAM: ${student.program ?? 'N/A'}\n';
        ragContext += '- SEMESTER: ${student.semester ?? 'N/A'}\n';

        if (records.isNotEmpty) {
          final attendance =
              records.where((r) => r['record_type'] == 'attendance').toList();
          final results =
              records.where((r) => r['record_type'] == 'result').toList();

          if (attendance.isNotEmpty) {
            ragContext += '\n[ACADEMIC_ATTENDANCE_TRANSCRIPT]:\n';
            for (var r in attendance) {
              final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown';
              final att = r['attendance_percentage'] ?? 'N/A';
              ragContext +=
                  '- SUBJECT: ${sub.toUpperCase()} | ATTENDANCE: $att% | CLASSES: ${r['attended_classes']}/${r['total_classes']}\n';
            }
          }

          if (results.isNotEmpty) {
            ragContext += '\n[ACADEMIC_RESULTS_MARKS_TRANSCRIPT]:\n';
            for (var r in results) {
              final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown';
              final marks = r['marks_obtained'] ?? r['marks'] ?? 'N/A';
              final total = r['max_marks'] ?? r['total_marks'] ?? 'N/A';
              final grade = r['grade'] ?? 'N/A';
              final exam = r['exam_type'] ?? 'Examination';
              ragContext +=
                  '- SUBJECT: ${sub.toUpperCase()} | EXAM: $exam | MARKS: $marks/$total | GRADE: $grade\n';
            }
          }

          ragContext +=
              '\n[STRICT_INSTRUCTION]: Use ONLY the subjects and data listed above. If a subject (like OS) is not in the list above, do NOT mention it.\n';
        } else {
          ragContext +=
              '\n[ALERT]: NO SUBJECT-WISE RECORDS FOUND IN DATABASE FOR THIS STUDENT ACCOUNT.\n';
        }
        ragContext += '[!!! END_OF_DATABASE_CONTENT !!!]\n';
        break;
      }
    }

    final lowerMsg = newMessage.toLowerCase();
    final mentorId = _currentUser?.id ?? '';

    // If asking about students, class, overall performance, summary, marks, or all students
    final isClassOrAllStudentsQuery = lowerMsg.contains('student') ||
        lowerMsg.contains('class') ||
        lowerMsg.contains('overall') ||
        lowerMsg.contains('performance') ||
        lowerMsg.contains('summary') ||
        lowerMsg.contains('attendance') ||
        lowerMsg.contains('result') ||
        lowerMsg.contains('mark') ||
        lowerMsg.contains('grade') ||
        lowerMsg.contains('score') ||
        lowerMsg.contains('all') ||
        lowerMsg.contains('who') ||
        lowerMsg.contains('list') ||
        lowerMsg.contains('everyone');

    if (isClassOrAllStudentsQuery) {
      if (_myStudents.isEmpty && _currentUser?.email != null) {
        try {
          _myStudents = await SupabaseService.getMyStudents(_currentUser!.email);
        } catch (e) {
          debugPrint('⚠️ Error loading students for mentor AI: $e');
        }
      }

      if (_myStudents.isNotEmpty) {
        final studentsWithResults = <UserModel>[];
        final studentsWithoutResults = <UserModel>[];
        final studentsWithAttendance = <UserModel>[];
        final studentRecordsMap = <String, List<Map<String, dynamic>>>{};

        for (final s in _myStudents) {
          final records = await SupabaseService.getAcademicRecord(s.id, rollNo: s.rollNumber);
          studentRecordsMap[s.id] = records;

          final hasResults = records.any((r) => r['record_type'] == 'result');
          final hasAttendance = records.any((r) => r['record_type'] == 'attendance');

          if (hasResults) {
            studentsWithResults.add(s);
          } else {
            studentsWithoutResults.add(s);
          }

          if (hasAttendance) {
            studentsWithAttendance.add(s);
          }
        }

        ragContext += '\n[!!! CRITICAL: OFFICIAL_CLASS_STUDENTS_ROSTER_AND_PERFORMANCE !!!]\n';
        ragContext += 'Mentor: ${_currentUser?.name ?? mentorName} (${_currentUser?.email ?? 'N/A'})\n';
        ragContext += '=== CLASS SUMMARY METRICS ===\n';
        ragContext += '- Total Assigned Students in Class: ${_myStudents.length}\n';
        ragContext += '- Students with Examination Results in Database: ${studentsWithResults.length} out of ${_myStudents.length} (${studentsWithResults.map((s) => s.name).join(', ')})\n';
        ragContext += '- Students without Results Uploaded Yet: ${studentsWithoutResults.length} out of ${_myStudents.length} (${studentsWithoutResults.map((s) => s.name).join(', ')})\n';
        ragContext += '- Students with Attendance Records in Database: ${studentsWithAttendance.length} out of ${_myStudents.length} (${studentsWithAttendance.map((s) => s.name).join(', ')})\n\n';

        for (int i = 0; i < _myStudents.length; i++) {
          final s = _myStudents[i];
          final records = studentRecordsMap[s.id] ?? [];
          ragContext += '=== STUDENT ${i + 1}: ${s.name.toUpperCase()} ===\n';
          ragContext += '- Roll Number: ${s.rollNumber ?? 'Not Assigned'}\n';
          ragContext += '- Program: ${s.program ?? 'N/A'} | Branch: ${s.branch ?? 'N/A'} | Semester: ${s.semester ?? 'N/A'}\n';
          ragContext += '- Email: ${s.email}\n';

          if (records.isNotEmpty) {
            final attendance = records.where((r) => r['record_type'] == 'attendance').toList();
            final results = records.where((r) => r['record_type'] == 'result').toList();

            if (attendance.isNotEmpty) {
              ragContext += '- Attendance Transcript:\n';
              for (var r in attendance) {
                final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown';
                final att = r['attendance_percentage'] ?? 'N/A';
                final attended = r['attended_classes'] ?? '?';
                final total = r['total_classes'] ?? '?';
                ragContext += '  * ${sub.toUpperCase()}: $att% ($attended/$total classes attended)\n';
              }
            } else {
              ragContext += '- Attendance Transcript: No subject attendance records uploaded yet.\n';
            }

            if (results.isNotEmpty) {
              ragContext += '- Academic Results & Marks:\n';
              for (var r in results) {
                final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown';
                final marks = r['marks_obtained'] ?? r['marks'] ?? 'N/A';
                final total = r['max_marks'] ?? r['total_marks'] ?? 'N/A';
                final grade = r['grade'] ?? 'N/A';
                final exam = r['exam_type'] ?? 'Examination';
                ragContext += '  * ${sub.toUpperCase()} ($exam): $marks/$total (Grade: $grade)\n';
              }
            } else {
              ragContext += '- Academic Results: No examination marks uploaded yet.\n';
            }
          } else {
            ragContext += '- Status: Enrolled student. No subject attendance or exam marks have been uploaded to the database yet.\n';
          }
          ragContext += '\n';
        }
        ragContext += '[STRICT_ANALYTICS_INSTRUCTION]: When asked about student counts, how many students are present in results, or class summaries, directly quote the exact numbers from the CLASS SUMMARY METRICS above (e.g. "There are exactly 2 students with examination results recorded in the database: Aayush Dubey and Aditya Vats out of 7 total assigned students"). Do not confuse "present in result" with class attendance presence.\n';
        ragContext += '[!!! END_OF_CLASS_STUDENTS_ROSTER !!!]\n';
      }
    }

    // 2. Deep Document Text Scanning for Student Names, Roll Numbers, or Document Queries
    try {
      List<StudentDocument> mentorDocs = [];
      if (mentorId.isNotEmpty) {
        mentorDocs = await SupabaseService.getMentorDocuments(mentorId);
      }
      if (mentorDocs.isEmpty) {
        mentorDocs = await SupabaseService.getStudentAccessibleDocuments(studentId: mentorId);
      }

      final isDocOrResultQuery = lowerMsg.contains('document') ||
          lowerMsg.contains('result') ||
          lowerMsg.contains('marksheet') ||
          lowerMsg.contains('score card') ||
          lowerMsg.contains('scorecard') ||
          lowerMsg.contains('pdf') ||
          lowerMsg.contains('sheet') ||
          lowerMsg.contains('file') ||
          lowerMsg.contains('name of student') ||
          lowerMsg.contains('names of student') ||
          lowerMsg.contains('all student') ||
          lowerMsg.contains('all students') ||
          lowerMsg.contains('list of student') ||
          lowerMsg.contains('present in') ||
          lowerMsg.contains('who got') ||
          lowerMsg.contains('highest') ||
          lowerMsg.contains('lowest') ||
          lowerMsg.contains('sgpa') ||
          lowerMsg.contains('cgpa') ||
          lowerMsg.contains('pass') ||
          lowerMsg.contains('fail');

      for (final d in mentorDocs) {
        final fullDoc = await SupabaseService.getDocumentWithContent(d.id);
        final extJson = fullDoc?.extractedJson ?? d.extractedJson;
        final extText = fullDoc?.extractedText ?? d.extractedText ?? '';
        if (extText.trim().isEmpty && (extJson == null || extJson.isEmpty)) continue;

        // Check if message specifically mentions this document title/filename or is a general document query for marksheets/results
        final titleMatched = lowerMsg.contains(d.title.toLowerCase()) ||
            lowerMsg.contains(d.fileName.toLowerCase().replaceAll('.pdf', '').replaceAll('_', ' ')) ||
            (isDocOrResultQuery && (d.docType == 'marksheet' || d.docType == 'result' || d.docType == 'other'));

        // 1. High-Precision Native JSON Extraction
        if (extJson != null && extJson.isNotEmpty) {
          final StringBuffer jsonSummary = StringBuffer();
          final pages = extJson.keys.where((k) => k.startsWith('page_')).toList();
          pages.sort();

          for (final pageKey in pages) {
            final pageData = extJson[pageKey] as Map<String, dynamic>? ?? {};
            final students = (pageData['students'] as List?)?.map((e) => Map<String, dynamic>.from(e as Map)).toList() ?? [];
            final courses = (pageData['courses'] as List?)?.map((e) => Map<String, dynamic>.from(e as Map)).toList() ?? [];

            if (courses.isNotEmpty) {
              jsonSummary.writeln('COURSES EVALUATED (${pageKey.toUpperCase()}):');
              for (final c in courses) {
                jsonSummary.writeln('- ${c['code']}: ${c['title']} (${c['credits'] ?? ''} credits)');
              }
            }

            if (students.isNotEmpty) {
              jsonSummary.writeln('\nSTUDENT SCORECARDS (${pageKey.toUpperCase()}):');
              for (final s in students) {
                final grades = (s['grades'] as Map?)?.entries.map((e) => '${e.key}:${e.value}').join(', ') ?? '';
                jsonSummary.writeln('- ROLL: ${s['roll_no'] ?? "N/A"} | NAME: ${s['name'] ?? "N/A"} | SGPA: ${s['sgpa'] ?? "N/A"} | GRADES: [$grades]');
              }
            }
          }

          final structuredStr = jsonSummary.toString().trim();
          if (structuredStr.isNotEmpty) {
            bool anyMatch = titleMatched;
            for (final word in words) {
              if (word.length >= 3 && structuredStr.toLowerCase().contains(word.toLowerCase())) {
                anyMatch = true;
                break;
              }
            }
            if (anyMatch) {
              ragContext += '\n[VERIFIED NATIVE JSON DATABASE TRANSCRIPT: "${d.title}" | Type: ${d.docType.toUpperCase()}]:\n';
              ragContext += '$structuredStr\n';
              ragContext += '[END OF VERIFIED NATIVE JSON TRANSCRIPT]\n';
              ragContext += '[STRICT PAGINATION RULE]: If the user asks for all students or names from this document, DO NOT dump all names at once. State the total count, aggregate stats (Highest/Lowest SGPA, Passed/Failed count), display the first range (Top 5-8 students in a compact markdown table), and provide interactive range chips: [OPTIONS: Show Next 10 (Students 9-18) | View Top Performers | View Failed / Backlogs | Search by Roll No].\n';
            }
          }
        }

        // 2. Fallback / supplementary flat-text scan
        if (extText.trim().isNotEmpty) {
          bool studentMatchedInDoc = false;
          final matchedLines = <String>[];

          for (final word in words) {
            if (word.length >= 4 && extText.toLowerCase().contains(word.toLowerCase())) {
              studentMatchedInDoc = true;
              final lines = extText.split('\n');
              for (int li = 0; li < lines.length; li++) {
                if (lines[li].toLowerCase().contains(word.toLowerCase())) {
                  final startIdx = (li - 2).clamp(0, lines.length - 1);
                  final endIdx = (li + 2).clamp(0, lines.length - 1);
                  matchedLines.add(lines.sublist(startIdx, endIdx + 1).join('\n'));
                }
              }
            }
          }

          if (titleMatched || studentMatchedInDoc) {
            ragContext += '\n[VERIFIED PUBLISHED INSTITUTIONAL DOCUMENT TEXT: "${d.title}" (${d.fileName}) | Type: ${d.docType.toUpperCase()}]:\n';
            if (studentMatchedInDoc && matchedLines.isNotEmpty && !titleMatched) {
              ragContext += 'MATCHED STUDENT RECORD IN DOCUMENT:\n${matchedLines.toSet().join('\n---\n')}\n';
            } else {
              final contentText = extText.length > 50000 ? '${extText.substring(0, 50000)}\n... [truncated for token safety]' : extText;
              ragContext += '$contentText\n';
            }
            ragContext += '[END OF VERIFIED DOCUMENT TEXT]\n';
          }
        }
      }

      if (isDocOrResultQuery) {
        ragContext += '\n[STRICT_DOCUMENT_ANALYSIS_INSTRUCTION]: The mentor is asking directly about the uploaded document content above. When asked to list all student names or count students in the document, extract and list EVERY student record present in the document text above (with Roll Number, Full Name, SGPA, and Status). Do not restrict your answer to only the 7 registered app users when answering questions about the uploaded document.\n';
      }

      // Autonomous Calendar / Holiday injection
      final isCalendarQuery = lowerMsg.contains('calendar') ||
          lowerMsg.contains('holiday') ||
          lowerMsg.contains('event') ||
          lowerMsg.contains('exam') ||
          lowerMsg.contains('break') ||
          lowerMsg.contains('vacation') ||
          lowerMsg.contains('semester');

      if (isCalendarQuery) {
        final calDocs = mentorDocs.where((d) => d.docType == 'academic_calendar').toList();
        if (calDocs.isNotEmpty) {
          final fullCal = await SupabaseService.getDocumentWithContent(calDocs.first.id);
          if (fullCal?.extractedText?.isNotEmpty == true) {
            ragContext += '\n[COMPLETE OFFICIAL ACADEMIC CALENDAR & SCHEDULE TEXT]:\n${fullCal!.extractedText}\n[END OF ACADEMIC CALENDAR]\n';
          }
        }
      }
    } catch (e) {
      debugPrint('⚠️ Mentor document lookup error: $e');
    }

    // 3. Autonomous Category-Partitioned Vector Semantic Search
    try {
      final contextMeta = _detectCategoryAndTemporalContext(newMessage);
      final queryEmbedding = await AIService.createEmbedding(newMessage);
      final chunks = await SupabaseService.searchSimilarChunks(
        studentId: null, // Search across all published class & institutional category chunks
        docType: contextMeta['category'],
        academicYear: contextMeta['year'],
        term: contextMeta['term'],
        queryEmbedding: queryEmbedding,
        limit: 8,
        minSimilarity: 0.20,
      );
      if (chunks.isNotEmpty) {
        ragContext += '\n\n[RELEVANT PUBLISHED MATERIAL EXCERPTS (${contextMeta['category']?.toUpperCase() ?? 'INSTITUTIONAL'})]:\n' + chunks.join('\n\n---\n\n');
        debugPrint('🔍 Mentor Category RAG: Injected ${chunks.length} chunks for ${contextMeta['category'] ?? "all"}');
      }
    } catch (e) {
      debugPrint('⚠️ Mentor vector search error: $e');
    }

    return await AIService.sendMentorMessage(
      history: history,
      newMessage: newMessage,
      mentorName: mentorName,
      designation: _currentUser?.designation,
      dept: _currentUser?.department,
      expertise: _currentUser?.expertise,
      totalStudents: _myStudents.length,
      activeChats: _conversations.where((c) => c.status == 'active').length,
      ragContext: ragContext.isNotEmpty ? ragContext : null,
      onStreamChunk: onStreamChunk,
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
    _messageChannel =
        SupabaseService.subscribeToMessages(conversationId, (msg) {
      if (!_messages.any((m) => m.id == msg.id)) {
        _messages.add(msg);
        notifyListeners();
      }
    });
  }

  void clearCurrentConversation() {
    _messageChannel?.unsubscribe();
    _currentConversation = null;
    _messages = [];
    notifyListeners();
  }

  ConversationModel _rebuild(
    ConversationModel b, {
    bool? isFirstDone,
    bool? detailsCollected,
    String? studentName,
    String? studentProgram,
    String? studentBranch,
    String? studentSemester,
    String? title,
  }) =>
      ConversationModel.fromMap({
        'id': b.id,
        'student_id': b.studentId,
        'title': title ?? b.title,
        'is_first_message_done': isFirstDone ?? b.isFirstMessageDone,
        'student_details_collected':
            detailsCollected ?? b.studentDetailsCollected,
        'student_name': studentName ?? b.studentName,
        'student_program': studentProgram ?? b.studentProgram,
        'student_branch': studentBranch ?? b.studentBranch,
        'student_semester': studentSemester ?? b.studentSemester,
        'mentor_email': b.mentorEmail,
        'status': b.status,
        'created_at': b.createdAt.toIso8601String(),
        'updated_at': DateTime.now().toIso8601String(),
      });

  String _conversationTitle(String message) {
    final cleaned = message.replaceAll(RegExp(r'\s+'), ' ').trim();
    final text = cleaned.toLowerCase();
    final topics = <String>[];

    if (text.contains('stress') ||
        text.contains('anxious') ||
        text.contains('tension') ||
        text.contains('worried')) {
      topics.add('Stress & personal support');
    }
    if (text.contains('assignment') ||
        text.contains('exam') ||
        text.contains('study') ||
        text.contains('homework')) {
      topics.add('Assignments & exam preparation');
    }
    if (text.contains('timetable') ||
        text.contains('schedule') ||
        text.contains('class')) {
      topics.add('Timetable & classes');
    }
    if (text.contains('attendance') ||
        text.contains('marks') ||
        text.contains('result')) {
      topics.add('Marks & attendance');
    }
    if (text.contains('career') ||
        text.contains('job') ||
        text.contains('internship')) {
      topics.add('Career guidance');
    }

    if (topics.isNotEmpty) return topics.take(2).join(' • ');
    if (cleaned.length <= 34) return cleaned;
    return '${cleaned.substring(0, 34).trimRight()}...';
  }

  /// Automatically parses user messages for academic domain category, academic session year, and term
  Map<String, String?> _detectCategoryAndTemporalContext(String query) {
    final lower = query.toLowerCase();
    String? category;
    String? year;
    String? term;

    // 1. Detect Category Domain
    if (lower.contains('result') ||
        lower.contains('marksheet') ||
        lower.contains('grade') ||
        lower.contains('sgpa') ||
        lower.contains('cgpa') ||
        lower.contains('scorecard') ||
        lower.contains('score card') ||
        lower.contains('topper') ||
        lower.contains('failed') ||
        lower.contains('reappear') ||
        lower.contains('re-appear')) {
      category = 'marksheet';
    } else if (lower.contains('attendance') ||
        lower.contains('absent') ||
        lower.contains('shortage') ||
        lower.contains('debar') ||
        lower.contains('defaulter')) {
      category = 'attendance';
    } else if (lower.contains('syllabus') ||
        lower.contains('curriculum') ||
        lower.contains('unit 1') ||
        lower.contains('unit 2') ||
        lower.contains('unit 3') ||
        lower.contains('unit 4') ||
        lower.contains('module') ||
        lower.contains('topics')) {
      category = 'syllabus';
    } else if (lower.contains('calendar') ||
        lower.contains('holiday') ||
        lower.contains('vacation') ||
        lower.contains('datesheet') ||
        lower.contains('date sheet') ||
        lower.contains('break') ||
        lower.contains('exam date')) {
      category = 'academic_calendar';
    } else if (lower.contains('assignment') ||
        lower.contains('deadline') ||
        lower.contains('submission') ||
        lower.contains('homework')) {
      category = 'assignment';
    } else if (lower.contains('notice') ||
        lower.contains('circular') ||
        lower.contains('hostel') ||
        lower.contains('policy') ||
        lower.contains('regulation')) {
      category = 'circular';
    }

    // 2. Detect Academic Year
    if (lower.contains('2025-26') || lower.contains('2025-2026') || lower.contains('2025')) {
      year = '2025-2026';
    } else if (lower.contains('2026-27') || lower.contains('2026-2027') || lower.contains('2026')) {
      year = '2026-2027';
    } else if (lower.contains('2024-25') || lower.contains('2024-2025') || lower.contains('2024')) {
      year = '2024-2025';
    }

    // 3. Detect Term (Odd vs Even / Semester)
    if (lower.contains('may') ||
        lower.contains('june') ||
        lower.contains('even') ||
        lower.contains('second') ||
        lower.contains('sem 2') ||
        lower.contains('4th') ||
        lower.contains('6th') ||
        lower.contains('8th')) {
      term = 'even';
    } else if (lower.contains('dec') ||
        lower.contains('nov') ||
        lower.contains('odd') ||
        lower.contains('first') ||
        lower.contains('sem 1') ||
        lower.contains('3rd') ||
        lower.contains('5th') ||
        lower.contains('7th')) {
      term = 'odd';
    }

    return {
      'category': category,
      'year': year,
      'term': term,
    };
  }

  @override
  void dispose() {
    _messageChannel?.unsubscribe();
    super.dispose();
  }
}
