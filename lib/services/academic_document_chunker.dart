// lib/services/academic_document_chunker.dart

/// Specialized Institutional Document Chunking & Ingestion Service for Acadly
/// Transforms raw OCR markdown and tables from university documents (Academic Calendars,
/// Syllabi, Timetables, Regulations) into semantically labeled, date-aware vector chunks.
class AcademicDocumentChunker {
  /// Chunks an institutional document based on its document type.
  /// Applies semantic boundary detection so monthly schedules, unit syllabi,
  /// and weekly timetables are never arbitrarily cut mid-table.
  static List<String> createSemanticChunks({
    required String docType,
    required String docTitle,
    required String rawText,
    String? academicYear,
  }) {
    final cleanText = rawText.trim();
    if (cleanText.isEmpty) return [];

    final yearTag = academicYear != null && academicYear.isNotEmpty
        ? ' (Academic Year: $academicYear)'
        : '';

    switch (docType) {
      case 'academic_calendar':
        return _chunkAcademicCalendar(docTitle, cleanText, yearTag);
      case 'syllabus':
        return _chunkSyllabus(docTitle, cleanText, yearTag);
      case 'timetable':
        return _chunkTimetable(docTitle, cleanText, yearTag);
      case 'marksheet':
      case 'result':
      case 'attendance':
        return _chunkMarksheet(docTitle, cleanText, yearTag);
      default:
        return _chunkGenericDocument(docTitle, cleanText, yearTag);
    }
  }

  /// Specialized chunker for Academic Calendars & Holiday Lists.
  /// Preserves full monthly schedules, holiday tables, and examination deadlines intact.
  static List<String> _chunkAcademicCalendar(
      String title, String text, String yearTag) {
    final chunks = <String>[];
    final lines = text.split('\n');

    final months = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];

    List<String> currentChunkLines = [];
    int wordCount = 0;

    for (final line in lines) {
      final trimmedLine = line.trim();
      if (trimmedLine.isEmpty) continue;

      // Detect Month Headers or Major Activity Sections
      final isMonthHeader = months.any((m) =>
          trimmedLine.toLowerCase().contains(m.toLowerCase()) &&
          (trimmedLine.length < 50 || trimmedLine.contains('|')));
      
      final isHolidaySection = trimmedLine.toLowerCase().contains('holiday') ||
          trimmedLine.toLowerCase().contains('vacation') ||
          trimmedLine.toLowerCase().contains('break');

      final lineWords = trimmedLine.split(RegExp(r'\s+')).length;

      if ((isMonthHeader || isHolidaySection) && wordCount > 300) {
        if (currentChunkLines.isNotEmpty) {
          chunks.add(
            '[INSTITUTIONAL ACADEMIC CALENDAR: $title$yearTag]\n'
            '${currentChunkLines.join('\n')}\n'
            '[END OF CALENDAR SECTION]',
          );
          currentChunkLines = [];
          wordCount = 0;
        }
      }

      currentChunkLines.add(trimmedLine);
      wordCount += lineWords;

      // Soft ceiling of 550 words per chunk
      if (wordCount >= 550) {
        chunks.add(
          '[INSTITUTIONAL ACADEMIC CALENDAR: $title$yearTag]\n'
          '${currentChunkLines.join('\n')}\n'
          '[END OF CALENDAR SECTION]',
        );
        currentChunkLines = [];
        wordCount = 0;
      }
    }

    if (currentChunkLines.isNotEmpty) {
      chunks.add(
        '[INSTITUTIONAL ACADEMIC CALENDAR: $title$yearTag]\n'
        '${currentChunkLines.join('\n')}\n'
        '[END OF CALENDAR SECTION]',
      );
    }

    return chunks.isNotEmpty ? chunks : [text];
  }

  /// Specialized chunker for Course Syllabi (chunks by Unit / Module / Topic)
  static List<String> _chunkSyllabus(
      String title, String text, String yearTag) {
    final chunks = <String>[];
    final lines = text.split('\n');

    List<String> currentChunkLines = [];
    int wordCount = 0;

    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;

      final isUnitHeader = RegExp(r'^(unit|module|chapter|part)\s+\w+', caseSensitive: false)
          .hasMatch(trimmed);

      final lineWords = trimmed.split(RegExp(r'\s+')).length;

      if (isUnitHeader && wordCount > 250) {
        if (currentChunkLines.isNotEmpty) {
          chunks.add(
            '[COURSE SYLLABUS: $title$yearTag]\n'
            '${currentChunkLines.join('\n')}\n'
            '[END OF SYLLABUS MODULE]',
          );
          currentChunkLines = [];
          wordCount = 0;
        }
      }

      currentChunkLines.add(trimmed);
      wordCount += lineWords;

      if (wordCount >= 500) {
        chunks.add(
          '[COURSE SYLLABUS: $title$yearTag]\n'
          '${currentChunkLines.join('\n')}\n'
          '[END OF SYLLABUS MODULE]',
        );
        currentChunkLines = [];
        wordCount = 0;
      }
    }

    if (currentChunkLines.isNotEmpty) {
      chunks.add(
        '[COURSE SYLLABUS: $title$yearTag]\n'
        '${currentChunkLines.join('\n')}\n'
        '[END OF SYLLABUS MODULE]',
      );
    }

    return chunks.isNotEmpty ? chunks : [text];
  }

  /// Specialized chunker for Class Timetables
  static List<String> _chunkTimetable(
      String title, String text, String yearTag) {
    final wordCount = text.split(RegExp(r'\s+')).length;
    if (wordCount <= 1200) {
      return [
        '[OFFICIAL CLASS TIMETABLE: $title$yearTag]\n'
        '$text\n'
        '[END OF TIMETABLE]',
      ];
    }
    return _chunkGenericDocument(title, text, yearTag);
  }

  /// Specialized chunker for Marksheets, Results, and Examination Scorecards.
  /// Extracts header metadata (University, Degree, Semester, Courses list) and preserves
  /// full student rows without splitting records across chunks.
  static List<String> _chunkMarksheet(
      String title, String text, String yearTag) {
    final chunks = <String>[];
    final lines = text.split('\n');

    final headerLines = <String>[];
    final studentLines = <String>[];
    bool inStudentsSection = false;

    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;

      final upper = trimmed.toUpperCase();
      if (upper.startsWith('STUDENTS:') ||
          upper.startsWith('STUDENT DATA:') ||
          upper.startsWith('| STUDENT ROLL NO') ||
          upper.startsWith('STUDENT:')) {
        inStudentsSection = true;
      }

      if (!inStudentsSection) {
        headerLines.add(trimmed);
      } else {
        if (!upper.startsWith('STUDENTS:') && !upper.startsWith('STUDENT DATA:')) {
          studentLines.add(trimmed);
        }
      }
    }

    final headerText = headerLines.isNotEmpty
        ? headerLines.join('\n')
        : 'DOCUMENT: $title$yearTag';

    if (studentLines.isEmpty) {
      final allLines = lines.where((l) => l.trim().isNotEmpty).toList();
      const int linesPerChunk = 15;
      for (int i = 0; i < allLines.length; i += linesPerChunk) {
        final end = (i + linesPerChunk < allLines.length) ? i + linesPerChunk : allLines.length;
        final slice = allLines.sublist(i, end).join('\n');
        chunks.add(
          '[OFFICIAL MARKSHEET / RESULT RECORD: $title$yearTag]\n'
          '$slice\n'
          '[END OF RECORD SECTION]',
        );
      }
      return chunks.isNotEmpty ? chunks : [text];
    }

    const int studentsPerChunk = 4;
    for (int i = 0; i < studentLines.length; i += studentsPerChunk) {
      final end = (i + studentsPerChunk < studentLines.length)
          ? i + studentsPerChunk
          : studentLines.length;
      final studentBlock = studentLines.sublist(i, end).join('\n');

      chunks.add(
        '[OFFICIAL MARKSHEET / RESULT RECORD: $title$yearTag]\n'
        '$headerText\n\n'
        'STUDENT RECORDS:\n'
        '$studentBlock\n'
        '[END OF RECORD SECTION]',
      );
    }

    return chunks.isNotEmpty ? chunks : [text];
  }

  /// Default sliding window for general circulars and announcements
  static List<String> _chunkGenericDocument(
      String title, String text, String yearTag) {
    final words = text.split(RegExp(r'\s+'));
    final chunks = <String>[];
    const chunkSize = 400;
    const overlap = 60;

    for (int i = 0; i < words.length; i += (chunkSize - overlap)) {
      final end = (i + chunkSize).clamp(0, words.length);
      final slice = words.sublist(i, end).join(' ').trim();
      if (slice.length > 10) {
        chunks.add(
          '[OFFICIAL DOCUMENT: $title$yearTag]\n'
          '$slice\n'
          '[END OF SECTION]',
        );
      }
      if (end >= words.length) break;
    }
    return chunks;
  }

  // ══════════════════════════════════════════════════════════
  // DUAL-FIDELITY: NATIVE JSON -> SEMANTIC NATURAL LANGUAGE CARDS
  // ══════════════════════════════════════════════════════════

  /// Converts validated Native JSON documents into high-density semantic text cards for vector embedding
  static List<String> generateSemanticCardsFromJson({
    required String docType,
    required String docTitle,
    required Map<String, dynamic> jsonMap,
    String? academicYear,
    String? semester,
  }) {
    switch (docType.toLowerCase()) {
      case 'marksheet':
      case 'result':
      case 'academic_results':
        return _cardsFromMarksheetJson(docTitle, jsonMap, academicYear, semester);
      case 'attendance':
      case 'attendance_register':
        return _cardsFromAttendanceJson(docTitle, jsonMap, academicYear, semester);
      case 'syllabus':
        return _cardsFromSyllabusJson(docTitle, jsonMap, academicYear, semester);
      case 'academic_calendar':
      case 'calendar':
        return _cardsFromCalendarJson(docTitle, jsonMap, academicYear, semester);
      default:
        return _cardsFromGenericJson(docTitle, jsonMap, academicYear, semester);
    }
  }

  static List<String> _cardsFromMarksheetJson(
    String title,
    Map<String, dynamic> jsonMap,
    String? academicYear,
    String? semester,
  ) {
    final chunks = <String>[];
    final inst = jsonMap['university_name'] ?? jsonMap['institution'] ?? 'Manav Rachna University';
    final school = jsonMap['school_name'] != null ? '\nSchool: ${jsonMap['school_name']}' : '';
    final prog = jsonMap['programme_name'] ?? jsonMap['programme'] ?? 'Academic Programme';
    final exam = jsonMap['examination'] as Map? ?? {};
    final session = jsonMap['result_session'] ?? exam['session'] ?? 'Current Examination Session';
    final sem = jsonMap['semester'] ?? exam['semester'] ?? semester ?? 'Semester';
    final batch = jsonMap['batch'] ?? exam['batch'] ?? academicYear ?? '';

    // Build Course Dictionary
    final courseDict = <String, String>{};
    final rawCourses = jsonMap['courses'];
    if (rawCourses is List) {
      for (final c in rawCourses) {
        if (c is Map) {
          final code = c['code']?.toString().trim() ?? '';
          final title = c['title']?.toString().trim() ?? code;
          final cr = c['credits'] != null ? ' (${c['credits']} Cr)' : '';
          if (code.isNotEmpty) {
            courseDict[code] = '$title$cr';
          }
        }
      }
    }

    final headerBlock = '[OFFICIAL MARKSHEET RECORD: $inst]$school\n'
        'Programme: $prog\n'
        'Examination: Session: $session | Semester: $sem${batch.isNotEmpty ? ' | Batch: $batch' : ''}';

    final rawStudents = jsonMap['students'];
    if (rawStudents is! List || rawStudents.isEmpty) {
      return [
        '$headerBlock\n\n'
        'Raw Data Summary: ${jsonMap.toString()}\n'
        '[END OF RECORD]'
      ];
    }

    final studentCards = <String>[];
    for (final s in rawStudents) {
      if (s is! Map) continue;
      final sNo = s['s_no'] != null ? '#${s['s_no']} ' : '';
      final spec = s['specialization'] != null ? ' (${s['specialization']})' : '';
      final rollNo = s['roll_no']?.toString().trim() ?? 'Unknown Roll';
      final name = s['name']?.toString().trim() ?? 'Unknown Student';
      final father = s['father_name']?.toString().trim();
      final sgpa = s['sgpa']?.toString() ?? 'N/A';
      final status = s['status']?.toString().toUpperCase() ?? 'RECORDED';

      final sb = StringBuffer();
      sb.writeln('STUDENT SCORECARD:');
      sb.writeln('• Student: $sNo$name$spec');
      sb.writeln('• Roll Number: $rollNo');
      if (father != null && father.isNotEmpty) sb.writeln('• Father\'s Name: $father');
      sb.writeln('• Semester SGPA: $sgpa');
      sb.writeln('• Academic Status: $status');

      final grades = s['grades'];
      if (grades is Map && grades.isNotEmpty) {
        sb.writeln('• Evaluated Subject Grades:');
        for (final entry in grades.entries) {
          final code = entry.key.toString().trim();
          final grade = entry.value.toString().trim();
          final courseName = courseDict[code] ?? code;
          sb.writeln('  - $code ($courseName): Grade $grade');
        }
      }
      studentCards.add(sb.toString().trim());
    }

    // Group 3 student cards per chunk with full header preserved
    const int perChunk = 3;
    for (int i = 0; i < studentCards.length; i += perChunk) {
      final end = (i + perChunk < studentCards.length) ? i + perChunk : studentCards.length;
      final group = studentCards.sublist(i, end).join('\n\n---\n\n');
      chunks.add(
        '$headerBlock\n\n'
        '$group\n'
        '[END OF RECORD SECTION]',
      );
    }

    return chunks.isNotEmpty ? chunks : [headerBlock];
  }

  static List<String> _cardsFromAttendanceJson(
    String title,
    Map<String, dynamic> jsonMap,
    String? academicYear,
    String? semester,
  ) {
    final chunks = <String>[];
    final inst = jsonMap['institution'] ?? 'University Record';
    final sub = jsonMap['subject'] as Map? ?? {};
    final subCode = sub['code'] ?? '';
    final subTitle = sub['title'] ?? 'Attendance Register';
    final period = jsonMap['attendance_period'] ?? 'Academic Session';

    final header = '[OFFICIAL ATTENDANCE REGISTER: $inst]\n'
        'Subject: $subCode ($subTitle) | Period: $period\n'
        'Academic Year: ${academicYear ?? '2026-2027'} | Term: ${semester ?? 'odd'}';

    final rawStudents = jsonMap['students'];
    if (rawStudents is List) {
      final rows = <String>[];
      for (final s in rawStudents) {
        if (s is Map) {
          final roll = s['roll_no'] ?? '';
          final name = s['name'] ?? '';
          final pct = s['percentage'] ?? '';
          final attended = s['attended_classes'] ?? '';
          final total = s['total_classes'] ?? '';
          final isDef = s['is_defaulter'] == true ? ' [AT RISK <75%]' : '';
          rows.add('• $name ($roll): $pct% ($attended/$total classes attended)$isDef');
        }
      }

      const int perChunk = 8;
      for (int i = 0; i < rows.length; i += perChunk) {
        final end = (i + perChunk < rows.length) ? i + perChunk : rows.length;
        chunks.add(
          '$header\n\n'
          'STUDENT ATTENDANCE:\n'
          '${rows.sublist(i, end).join('\n')}\n'
          '[END OF RECORD SECTION]',
        );
      }
    }

    return chunks.isNotEmpty ? chunks : [header];
  }

  static List<String> _cardsFromSyllabusJson(
    String title,
    Map<String, dynamic> jsonMap,
    String? academicYear,
    String? semester,
  ) {
    final chunks = <String>[];
    final code = jsonMap['course_code'] ?? '';
    final subTitle = jsonMap['course_title'] ?? title;
    final modules = jsonMap['modules'];

    final header = '[COURSE SYLLABUS: $code - $subTitle]\n'
        'Year: ${academicYear ?? '2026-2027'} | Semester: ${semester ?? 'odd'}';

    if (modules is List) {
      for (final m in modules) {
        if (m is Map) {
          final mNum = m['module_number'] ?? '';
          final mTitle = m['title'] ?? '';
          final topics = (m['topics'] as List?)?.join(', ') ?? '';
          final readings = (m['recommended_readings'] as List?)?.join(', ') ?? '';

          chunks.add(
            '$header\n\n'
            'UNIT / MODULE $mNum: $mTitle\n'
            'Topics Covered: $topics\n'
            '${readings.isNotEmpty ? 'References: $readings\n' : ''}'
            '[END OF SYLLABUS MODULE]',
          );
        }
      }
    }

    return chunks.isNotEmpty ? chunks : [header];
  }

  static List<String> _cardsFromCalendarJson(
    String title,
    Map<String, dynamic> jsonMap,
    String? academicYear,
    String? semester,
  ) {
    final chunks = <String>[];
    final inst = jsonMap['institution'] ?? 'University Academic Calendar';
    final events = jsonMap['events'];

    final header = '[OFFICIAL ACADEMIC CALENDAR: $inst]\n'
        'Academic Year: ${academicYear ?? '2026-2027'} | Term: ${semester ?? 'odd'}';

    if (events is List) {
      final eventLines = <String>[];
      for (final e in events) {
        if (e is Map) {
          final date = e['date'] ?? '';
          final day = e['day_of_week'] ?? '';
          final act = e['activity'] ?? '';
          final hol = e['is_holiday'] == true ? ' [HOLIDAY]' : '';
          eventLines.add('• $date ($day): $act$hol');
        }
      }

      const int perChunk = 10;
      for (int i = 0; i < eventLines.length; i += perChunk) {
        final end = (i + perChunk < eventLines.length) ? i + perChunk : eventLines.length;
        chunks.add(
          '$header\n\n'
          'SCHEDULE OF DATES & DEADLINES:\n'
          '${eventLines.sublist(i, end).join('\n')}\n'
          '[END OF CALENDAR SECTION]',
        );
      }
    }

    return chunks.isNotEmpty ? chunks : [header];
  }

  static List<String> _cardsFromGenericJson(
    String title,
    Map<String, dynamic> jsonMap,
    String? academicYear,
    String? semester,
  ) {
    final header = '[OFFICIAL INSTITUTIONAL RECORD: $title]\n'
        'Year: ${academicYear ?? '2026-2027'} | Term: ${semester ?? 'odd'}\n\n';
    return [
      '$header'
      'Content: ${jsonMap.toString()}\n'
      '[END OF RECORD]'
    ];
  }
}
