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
      // 3d. Always Retrieve Official Student Attendance, Examination Marks & Native Records
      final studentRoll = (_currentUser?.rollNumber != null && _currentUser!.rollNumber!.isNotEmpty)
          ? _currentUser!.rollNumber!
          : (_currentConversation?.studentRollNo ?? '');
      final studentSem = (_currentUser?.semester != null && _currentUser!.semester!.isNotEmpty)
          ? _currentUser!.semester!
          : (_currentConversation?.studentSemester ?? '5');
      final studentName = _currentUser?.name ?? _currentConversation?.studentName ?? 'Student';
      final studentProgram = _currentUser?.program ?? _currentConversation?.studentProgram ?? 'B.Tech';
      final studentBranch = _currentUser?.branch ?? _currentConversation?.studentBranch ?? 'Computer Science';
      final studentSection = MRUTimetableRepository.resolveSection(
        program: studentProgram,
        branch: studentBranch,
        semester: studentSem,
        section: _currentUser?.section,
      );

      if (studentRoll.isNotEmpty) {
        // 1. Structured Native JSON lookup from institutional marksheets & registers
        try {
          final docs = await SupabaseService.getStudentAccessibleDocuments(
            studentId: studentId,
            rollNo: studentRoll,
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
                  final currentName = studentName.trim().toLowerCase();

                  if (sRoll.contains(studentRoll.trim().toLowerCase()) ||
                      studentRoll.trim().toLowerCase().contains(sRoll) ||
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

        // 2. Relational database lookup (attendance & student_attendance_summary)
        try {
          var attRecords = await SupabaseService.getStudentAttendanceRecords(
            studentRoll,
            semester: studentSem,
          );
          // Fallback if semester filter yields empty
          if (attRecords.isEmpty) {
            attRecords = await SupabaseService.getStudentAttendanceRecords(studentRoll);
          }

          var attSummary = await SupabaseService.getStudentAttendanceSummary(
            studentRoll,
            semester: studentSem,
          );
          // Fallback if semester filter yields null
          attSummary ??= await SupabaseService.getStudentAttendanceSummary(studentRoll);

          if (attRecords.isNotEmpty || attSummary != null) {
            ragContext += '\n[OFFICIAL INSTITUTIONAL ATTENDANCE REPORT (STUDENT: ${studentName.toUpperCase()}, ROLL NO: $studentRoll)]:\n';
            if (attSummary != null) {
              ragContext += '• Overall Attendance: ${attSummary.overallPercentage.toStringAsFixed(2)}% (Criteria: >=75.0% Mandatory)\n';
              ragContext += '• Defaulter Subjects Count: ${attSummary.defaulterSubjectCount}\n';
              ragContext += '• Eligibility Status: ${attSummary.isCritical ? "⚠️ CRITICAL SHORTAGE (<75% overall / multi-subject shortage)" : "✅ Eligible / Good Standing (>=75%)"}\n';
              ragContext += '• Monitoring Cycle: ${attSummary.monitoringCycle}\n';
            }
            if (attRecords.isNotEmpty) {
              ragContext += '\n| Subject Code | Subject Name | Type | Faculty | Attendance % | Status |\n';
              ragContext += '| :--- | :--- | :--- | :--- | :---: | :--- |\n';
              for (final r in attRecords) {
                final statusStr = r.isDefaulter ? "⚠️ DEFAULTER (<75%)" : "Eligible";
                ragContext += '| ${r.subjectCode} | ${r.subjectName} | ${r.courseType} | ${r.facultyName ?? "Faculty"} | ${r.attendancePercentage.toStringAsFixed(2)}% | $statusStr |\n';
              }
            }
            ragContext += '[END OFFICIAL ATTENDANCE REPORT]\n';
          }
        } catch (e) {
          debugPrint('⚠️ Student attendance lookup error: $e');
        }

        // 3. Official marks & results
        try {
          final records = await SupabaseService.getAcademicRecord(studentId, rollNo: studentRoll);
          if (records.isNotEmpty) {
            final results = records.where((r) => r['record_type'] == 'result').toList();
            if (results.isNotEmpty) {
              ragContext += '\n\n[OFFICIAL ACADEMIC RESULTS/MARKS]:\n';
              for (var r in results) {
                final sub = r['subject_name'] ?? r['subject_code'] ?? 'Unknown Subject';
                final marks = r['marks_obtained'] ?? r['marks'] ?? 'N/A';
                final total = r['max_marks'] ?? r['total_marks'] ?? 'N/A';
                final grade = r['grade'] ?? 'N/A';
                final exam = r['exam_type'] ?? 'Examination';
                ragContext += '- $sub ($exam): Marks $marks/$total, Grade: $grade\n';
              }
            }
          }
        } catch (_) {}
      }

      // 4. Send to Gemini with RAG context and stream token chunks
      MessageModel? streamingMsg;
      final aiText = await AIService.sendStudentMessage(
        history: _messages.where((m) => m.id != streamingMsg?.id).toList(),
        newMessage: content.trim(),
        studentName: studentName,
        rollNo: studentRoll.isNotEmpty ? studentRoll : null,
        dept: _currentUser?.department ?? _currentConversation?.studentDept,
        program: studentProgram,
        branch: studentBranch,
        semester: studentSem,
        section: studentSection,
        skills: _currentUser?.skills ?? _currentConversation?.studentSkills,
        interests: _currentUser?.careerInterests ?? _currentConversation?.studentInterests,
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
    String? mentorEmail,
    String? mentorId,
    String? assignedClass,
    void Function(String partialText)? onStreamChunk,
  }) async {
    final StringBuffer sb = StringBuffer();
    final effectiveEmail = (mentorEmail != null && mentorEmail.isNotEmpty)
        ? mentorEmail
        : (_currentUser?.email ?? '');
    final effectiveId = (mentorId != null && mentorId.isNotEmpty)
        ? mentorId
        : (_currentUser?.id ?? '');
    final effectiveClass = (assignedClass != null && assignedClass.isNotEmpty)
        ? assignedClass
        : (_currentUser?.assignedClass ?? 'CSE 5A');
    final lowerMsg = newMessage.toLowerCase();

    // 1. Ensure mentor's assigned class roster is loaded
    if ((_myStudents.isEmpty || _myStudents.length < 5) && effectiveEmail.isNotEmpty) {
      try {
        _myStudents = await SupabaseService.getMyStudents(effectiveEmail);
      } catch (e) {
        debugPrint('⚠️ Error loading students for mentor AI: $e');
      }
    }

    final rollNumbers = _myStudents
        .map((s) => s.rollNumber?.trim() ?? '')
        .where((r) => r.isNotEmpty)
        .toList();

    // 2. High-Performance Parallel Bulk Queries for Class Data (1 Round-Trip)
    List<Map<String, dynamic>> summaries = [];
    List<Map<String, dynamic>> subjectRecords = [];
    List<Map<String, dynamic>> examResults = [];

    if (rollNumbers.isNotEmpty) {
      try {
        final summariesFuture = SupabaseService.getClassAttendanceSummaries(rollNumbers);
        final subjectsFuture = SupabaseService.getClassSubjectAttendance(rollNumbers);
        final resultsFuture = SupabaseService.getClassAcademicResults(rollNumbers);

        final results = await Future.wait([summariesFuture, subjectsFuture, resultsFuture]);
        summaries = results[0];
        subjectRecords = results[1];
        examResults = results[2];
      } catch (e) {
        debugPrint('⚠️ Bulk class data fetch error: $e');
      }
    }

    // Build Student Lookup Map
    final studentMap = <String, UserModel>{};
    for (final s in _myStudents) {
      if (s.rollNumber != null && s.rollNumber!.isNotEmpty) {
        studentMap[s.rollNumber!.trim().toUpperCase()] = s;
      }
    }

    // 3. Inject Verified Institutional Class Context
    sb.writeln('\n[!!! CRITICAL: VERIFIED_OFFICIAL_CLASS_DATABASE_RECORDS !!!]');
    sb.writeln('MENTOR: ${_currentUser?.name ?? mentorName} ($effectiveEmail)');
    sb.writeln('ASSIGNED CLASS: $effectiveClass (Total Enrolled: ${_myStudents.length} Students)');

    if (summaries.isNotEmpty) {
      final totalWithAttendance = summaries.length;
      final criticalDefaulters = summaries
          .where((s) => ((s['overall_percentage'] as num?) ?? 0) < 75.0 || s['is_critical'] == true)
          .toList();

      sb.writeln('\n=== OFFICIAL CLASS ATTENDANCE SUMMARY (CYCLE: ${summaries.first['monitoring_cycle'] ?? "CURRENT WAVE"}) ===');
      sb.writeln('- Total Students with Uploaded Attendance: $totalWithAttendance out of ${_myStudents.length}');
      sb.writeln('- Total Critical Defaulters (< 75.0% Overall): ${criticalDefaulters.length}');

      // Sort Ascending to immediately provide Top Lowest Attendance rankings
      final sortedSummaries = List<Map<String, dynamic>>.from(summaries)
        ..sort((a, b) => ((a['overall_percentage'] as num?) ?? 0).compareTo((b['overall_percentage'] as num?) ?? 0));

      sb.writeln('\n=== TOP 10 LOWEST ATTENDANCE STUDENTS (RANKED 1 TO 10 ASCENDING) ===');
      sb.writeln('| Rank | Roll Number | Student Name | Overall Attendance % | Defaulter Courses | Status |');
      sb.writeln('| :---: | :--- | :--- | :---: | :---: | :--- |');
      for (int i = 0; i < sortedSummaries.length && i < 10; i++) {
        final sum = sortedSummaries[i];
        final roll = (sum['student_roll_no'] ?? '').toString().trim();
        final student = studentMap[roll.toUpperCase()];
        final name = student?.name ?? 'Student $roll';
        final pct = ((sum['overall_percentage'] as num?) ?? 0).toStringAsFixed(2);
        final defCount = sum['defaulter_subject_cnt'] ?? 0;
        final isCrit = (sum['is_critical'] == true) || (((sum['overall_percentage'] as num?) ?? 0) < 75.0);
        final status = isCrit ? '⚠️ CRITICAL DEFAULTER (<75%)' : 'Eligible (>=75%)';
        sb.writeln('| ${i + 1} | $roll | $name | $pct% | $defCount | $status |');
      }

      if (criticalDefaulters.isNotEmpty) {
        sb.writeln('\n=== COMPLETE LIST OF ALL CRITICAL DEFAULTERS (< 75.0% ATTENDANCE) ===');
        for (final d in criticalDefaulters) {
          final roll = (d['student_roll_no'] ?? '').toString().trim();
          final student = studentMap[roll.toUpperCase()];
          final name = student?.name ?? 'Student $roll';
          final pct = ((d['overall_percentage'] as num?) ?? 0).toStringAsFixed(2);
          final defCount = d['defaulter_subject_cnt'] ?? 0;
          sb.writeln('- $name ($roll): $pct% overall ($defCount subjects below 75%)');
        }
      }
    } else {
      sb.writeln('\n[ATTENDANCE STATUS]: No attendance monitoring report has been uploaded yet for this class.');
    }

    if (subjectRecords.isNotEmpty) {
      final subjectGroups = <String, List<num>>{};
      for (final r in subjectRecords) {
        final sub = (r['subject_name'] ?? r['subject_code'] ?? 'Unknown').toString();
        final pct = (r['attendance_percentage'] as num?) ?? 0;
        subjectGroups.putIfAbsent(sub, () => []).add(pct);
      }
      sb.writeln('\n=== SUBJECT-WISE CLASS METRICS ===');
      subjectGroups.forEach((sub, pcts) {
        final avg = pcts.reduce((a, b) => a + b) / pcts.length;
        final defs = pcts.where((p) => p < 75).length;
        sb.writeln('- ${sub.toUpperCase()}: Class Avg ${avg.toStringAsFixed(1)}% ($defs students below 75%)');
      });
    }

    if (examResults.isNotEmpty) {
      sb.writeln('\n=== ACADEMIC EXAMINATION RESULTS & MARKS ===');
      for (final res in examResults) {
        final roll = (res['student_roll_no'] ?? '').toString().trim();
        final student = studentMap[roll.toUpperCase()];
        final name = student?.name ?? 'Student $roll';
        final sub = res['subject_name'] ?? res['subject_code'] ?? 'Subject';
        final marks = res['marks_obtained'] ?? res['marks'] ?? 'N/A';
        final total = res['max_marks'] ?? '100';
        final grade = res['grade'] ?? 'N/A';
        sb.writeln('- $name ($roll) - ${sub.toString().toUpperCase()}: $marks/$total (Grade: $grade)');
      }
    }

    // 4. Targeted student specific query lookup
    final words = newMessage
        .split(RegExp(r'[\s,:]+'))
        .where((w) => w.length >= 3)
        .toList();

    for (final word in words) {
      final student = await SupabaseService.findStudentByQuery(word,
          mentorEmail: effectiveEmail.isNotEmpty ? effectiveEmail : null);
      if (student != null) {
        final roll = (student.rollNumber ?? '').trim().toUpperCase();
        final stSubs = subjectRecords.where((r) => (r['student_roll_no'] ?? '').toString().trim().toUpperCase() == roll).toList();
        final stSum = summaries.firstWhere(
          (s) => (s['student_roll_no'] ?? '').toString().trim().toUpperCase() == roll,
          orElse: () => {},
        );

        sb.writeln('\n=== DETAILED PROFILE & RECORD FOR STUDENT: ${student.name.toUpperCase()} ($roll) ===');
        sb.writeln('- Full Name: ${student.name} | Roll Number: $roll');
        sb.writeln('- Program: ${student.program ?? 'B.Tech'} | Branch: ${student.branch ?? 'CSE'} | Sem: ${student.semester ?? '5'}');
        sb.writeln('- Official Email: ${student.officialEmail ?? student.email} | Mobile: ${student.phone ?? student.mobileNo ?? 'N/A'}');
        if (stSum.isNotEmpty) {
          final pct = ((stSum['overall_percentage'] as num?) ?? 0).toStringAsFixed(2);
          sb.writeln('- Overall Attendance: $pct% (Shortage in ${stSum['defaulter_subject_cnt'] ?? 0} courses)');
        }
        if (stSubs.isNotEmpty) {
          final stSubsDef = stSubs.where((sub) => ((sub['attendance_percentage'] as num?) ?? 0) < 75).length;
          sb.writeln('- Subject-Wise Attendance Breakdown:');
          for (final sub in stSubs) {
            final sName = sub['subject_name'] ?? sub['subject_code'] ?? 'Subject';
            final sPct = ((sub['attendance_percentage'] as num?) ?? 0).toStringAsFixed(1);
            final fName = sub['faculty_name'] ?? 'Faculty';
            final defStr = ((sub['attendance_percentage'] as num?) ?? 0) < 75 ? '⚠️ DEFAULTER' : 'Eligible';
            sb.writeln('  * ${sName.toString().toUpperCase()}: $sPct% ($defStr) | Faculty: $fName');
          }
        }
        break;
      }
    }

    sb.writeln('\n[STRICT ANALYTICS DIRECTIVE]:');
    sb.writeln('1. When asked for top 10 lowest attendance students, quote the EXACT rankings and percentages from the TOP 10 LOWEST ATTENDANCE STUDENTS table above.');
    sb.writeln('2. When asked follow-ups (e.g. "recheck it", "rank them", "who is the lowest?"), use the verified data above directly.');
    sb.writeln('3. Maintain strict zero hallucination — never fabricate percentages or student names.');
    sb.writeln('[END OF VERIFIED DATABASE CONTENT]\n');

    String ragContext = sb.toString();

    // 2. Deep Document Text Scanning for Student Names, Roll Numbers, or Document Queries
    try {
      List<StudentDocument> mentorDocs = [];
      if (effectiveId.isNotEmpty) {
        mentorDocs = await SupabaseService.getMentorDocuments(effectiveId);
      }
      if (mentorDocs.isEmpty) {
        mentorDocs = await SupabaseService.getStudentAccessibleDocuments(studentId: effectiveId);
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
