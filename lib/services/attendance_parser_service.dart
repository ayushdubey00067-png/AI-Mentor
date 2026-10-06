// lib/services/attendance_parser_service.dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:excel/excel.dart';
import 'package:flutter/foundation.dart';
import '../models/models.dart';

class AttendanceParserService {
  /// Parses attendance report from raw file bytes (.csv, .xlsx, or .xls)
  static ParsedAttendanceReport parseBytes(Uint8List bytes, String fileName) {
    final ext = fileName.toLowerCase();
    if (ext.endsWith('.csv')) {
      final content = utf8.decode(bytes, allowMalformed: true);
      return parseCsvContent(content);
    } else if (ext.endsWith('.xlsx') || ext.endsWith('.xls')) {
      return parseExcelBytes(bytes);
    } else {
      // Try CSV as fallback
      try {
        final content = utf8.decode(bytes, allowMalformed: true);
        return parseCsvContent(content);
      } catch (e) {
        return ParsedAttendanceReport(
          branch: 'NA',
          department: 'NA',
          className: 'NA',
          section: 'A',
          semester: '5',
          cycleName: 'ATTENDANCE MONITORING REPORT',
          dateRange: 'NA',
          minCriteria: 75.0,
          programCoordinator: 'NA',
          subjects: [],
          studentRows: [],
          totalStudents: 0,
          criticalCount: 0,
          isFormatValid: false,
          validationError: 'Unsupported file format. Please upload a .csv, .xlsx, or .xls file.',
        );
      }
    }
  }

  /// Parses raw CSV string using robust zero-dependency RFC 4180 grid parser
  static ParsedAttendanceReport parseCsvContent(String csvText) {
    try {
      final rows = _parseCsvGrid(csvText);
      return _processGenericRows(rows);
    } catch (e) {
      debugPrint('❌ CSV Attendance Parse Error: $e');
      return ParsedAttendanceReport(
        branch: 'NA',
        department: 'NA',
        className: 'NA',
        section: 'A',
        semester: '5',
        cycleName: 'ATTENDANCE MONITORING REPORT',
        dateRange: 'NA',
        minCriteria: 75.0,
        programCoordinator: 'NA',
        subjects: [],
        studentRows: [],
        totalStudents: 0,
        criticalCount: 0,
        isFormatValid: false,
        validationError: 'Failed to parse CSV: $e',
      );
    }
  }

  /// Zero-dependency RFC 4180 streaming CSV grid parser
  static List<List<dynamic>> _parseCsvGrid(String csvText) {
    final List<List<dynamic>> rows = [];
    final List<dynamic> currentRow = [];
    final StringBuffer currentCell = StringBuffer();
    bool inQuotes = false;

    for (int i = 0; i < csvText.length; i++) {
      final char = csvText[i];
      final nextChar = (i + 1 < csvText.length) ? csvText[i + 1] : null;

      if (char == '"') {
        if (inQuotes && nextChar == '"') {
          currentCell.write('"');
          i++; // Skip escaped quote
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char == ',' && !inQuotes) {
        currentRow.add(currentCell.toString().trim());
        currentCell.clear();
      } else if ((char == '\n' || char == '\r') && !inQuotes) {
        if (char == '\r' && nextChar == '\n') {
          i++; // Skip \n
        }
        currentRow.add(currentCell.toString().trim());
        currentCell.clear();
        if (currentRow.any((c) => c.toString().isNotEmpty)) {
          rows.add(List.from(currentRow));
        }
        currentRow.clear();
      } else {
        currentCell.write(char);
      }
    }

    if (currentCell.isNotEmpty || currentRow.isNotEmpty) {
      currentRow.add(currentCell.toString().trim());
      if (currentRow.any((c) => c.toString().isNotEmpty)) {
        rows.add(List.from(currentRow));
      }
    }

    return rows;
  }

  /// Parses Excel bytes (.xlsx, .xls)
  static ParsedAttendanceReport parseExcelBytes(Uint8List bytes) {
    try {
      final excel = Excel.decodeBytes(bytes);
      if (excel.tables.isEmpty) {
        throw Exception('Excel workbook contains no sheets.');
      }

      // Pick the first sheet or sheet with 'attendance' in name
      final sheetName = excel.tables.keys.firstWhere(
        (k) => k.toLowerCase().contains('att') || k.toLowerCase().contains('sheet'),
        orElse: () => excel.tables.keys.first,
      );

      final sheet = excel.tables[sheetName]!;
      final List<List<dynamic>> rows = [];

      for (final row in sheet.rows) {
        final List<dynamic> rowValues = [];
        for (final cell in row) {
          if (cell == null || cell.value == null) {
            rowValues.add('');
          } else {
            rowValues.add(cell.value.toString().trim());
          }
        }
        rows.add(rowValues);
      }

      return _processGenericRows(rows);
    } catch (e) {
      debugPrint('❌ Excel Attendance Parse Error: $e');
      return ParsedAttendanceReport(
        branch: 'NA',
        department: 'NA',
        className: 'NA',
        section: 'A',
        semester: '5',
        cycleName: 'ATTENDANCE MONITORING REPORT',
        dateRange: 'NA',
        minCriteria: 75.0,
        programCoordinator: 'NA',
        subjects: [],
        studentRows: [],
        totalStudents: 0,
        criticalCount: 0,
        isFormatValid: false,
        validationError: 'Failed to parse Excel file: $e',
      );
    }
  }

  /// Core logic to process 2D grid matrix rows into ParsedAttendanceReport
  static ParsedAttendanceReport _processGenericRows(List<List<dynamic>> rows) {
    String branch = 'MRU-School of Engineering';
    String department = 'Computer Science and Engineering';
    String className = 'BTech CSE Sem 5';
    String section = 'A';
    String semester = '5';
    String cycleName = 'SECOND ATTENDANCE MONITORING REPORT';
    String dateRange = '';
    double minCriteria = 75.0;
    String programCoordinator = '';

    int headerRowIndex = -1;

    // 1. Scan metadata block (usually top 15 rows)
    for (int i = 0; i < rows.length && i < 20; i++) {
      final row = rows[i];
      final fullRowStr = row.map((c) => c.toString()).join(' ');

      if (fullRowStr.toUpperCase().contains('ATTENDANCE MONITORING REPORT')) {
        cycleName = _extractCycleName(fullRowStr);
      }
      if (fullRowStr.toLowerCase().contains('branch:')) {
        branch = _extractAfterColon(fullRowStr, 'branch:');
      }
      if (fullRowStr.toLowerCase().contains('department:')) {
        department = _extractAfterColon(fullRowStr, 'department:');
      }
      if (fullRowStr.toLowerCase().contains('class name:')) {
        className = _extractAfterColon(fullRowStr, 'class name:');
        semester = _extractSemester(className);
      }
      if (fullRowStr.toLowerCase().contains('division:') || fullRowStr.toLowerCase().contains('section:')) {
        section = _extractDivision(fullRowStr);
      }
      if (fullRowStr.toLowerCase().contains('date:')) {
        dateRange = _extractAfterColon(fullRowStr, 'date:');
      }
      if (fullRowStr.toLowerCase().contains('coordinator:')) {
        programCoordinator = _extractAfterColon(fullRowStr, 'coordinator:');
      }

      // Check if this row is the primary Table Header (contains Roll No and Student Name)
      final rowJoined = row.map((c) => c.toString().toLowerCase().trim()).toList();
      if ((rowJoined.contains('roll no') || rowJoined.contains('roll number') || rowJoined.contains('roll no.')) &&
          (rowJoined.contains('student name') || rowJoined.contains('name'))) {
        headerRowIndex = i;
        break;
      }
    }

    if (headerRowIndex == -1) {
      return ParsedAttendanceReport(
        branch: branch,
        department: department,
        className: className,
        section: section,
        semester: semester,
        cycleName: cycleName,
        dateRange: dateRange,
        minCriteria: minCriteria,
        programCoordinator: programCoordinator,
        subjects: [],
        studentRows: [],
        totalStudents: 0,
        criticalCount: 0,
        isFormatValid: false,
        validationError: 'Could not find attendance table headers with "Roll No" and "Student Name".',
      );
    }

    // 2. Parse 4-tier matrix headers
    // Row headerRowIndex: Subject Names
    // Row headerRowIndex + 1: Subject Codes (or criteria row)
    // Row headerRowIndex + 2: Course Category & Overall header
    // Row headerRowIndex + 3: Faculty Names
    final rowSubjectNames = rows[headerRowIndex];
    final rowCodes = (headerRowIndex + 1 < rows.length) ? rows[headerRowIndex + 1] : <dynamic>[];
    final rowCategories = (headerRowIndex + 2 < rows.length) ? rows[headerRowIndex + 2] : <dynamic>[];
    final rowFaculty = (headerRowIndex + 3 < rows.length) ? rows[headerRowIndex + 3] : <dynamic>[];

    // Find summary columns indices
    int rollColIndex = 1;
    int nameColIndex = 2;
    int overallColIndex = -1;
    int defaulterColIndex = -1;
    int criticalColIndex = -1;

    for (int c = 0; c < rowSubjectNames.length; c++) {
      final nameText = rowSubjectNames[c].toString().toLowerCase().trim();
      if (nameText.contains('roll no') || nameText.contains('roll number')) {
        rollColIndex = c;
      } else if (nameText.contains('student name') || nameText == 'name') {
        nameColIndex = c;
      }
    }

    // Look in rowCategories for summary headers
    for (int c = 0; c < rowCategories.length; c++) {
      final catText = rowCategories[c].toString().toLowerCase().trim();
      if (catText.contains('overall') || catText.contains('overall %') || catText.contains('total %')) {
        overallColIndex = c;
      } else if (catText.contains('below minimum') || catText.contains('count of courses')) {
        defaulterColIndex = c;
      } else if (catText.contains('whether critical') || catText.contains('critical')) {
        criticalColIndex = c;
      }
    }

    // If overall column not found in rowCategories, search rowSubjectNames
    if (overallColIndex == -1) {
      for (int c = 0; c < rowSubjectNames.length; c++) {
        final sText = rowSubjectNames[c].toString().toLowerCase().trim();
        if (sText.contains('overall') || sText.contains('total %')) {
          overallColIndex = c;
          break;
        }
      }
    }

    // Identify Subject Columns (from column after Student Name up to Overall Column)
    final int startSubjectCol = nameColIndex + 1;
    final int endSubjectCol = (overallColIndex != -1) ? overallColIndex : rowSubjectNames.length;

    final List<SubjectColumnInfo> subjects = [];
    for (int c = startSubjectCol; c < endSubjectCol; c++) {
      final subName = (c < rowSubjectNames.length) ? rowSubjectNames[c].toString().trim() : '';
      if (subName.isEmpty || subName.toLowerCase().startsWith('please update') || subName.toLowerCase().contains('sr no')) {
        continue;
      }

      final subCode = (c < rowCodes.length) ? rowCodes[c].toString().trim() : 'SUB$c';
      final category = (c < rowCategories.length) ? rowCategories[c].toString().trim() : 'CORE';
      final faculty = (c < rowFaculty.length) ? rowFaculty[c].toString().trim() : '';

      subjects.add(SubjectColumnInfo(
        subjectName: subName,
        subjectCode: subCode.isNotEmpty ? subCode : 'SUB$c',
        courseType: category.isNotEmpty ? category : 'CORE',
        facultyName: faculty,
        colIndex: c,
      ));
    }

    // 3. Scan Student Rows
    final List<StudentAttendanceRow> studentRows = [];
    int criticalCount = 0;

    int studentStartRow = headerRowIndex + 4;
    // Check if headerRowIndex + 4 is actually a data row or faculty row
    if (studentStartRow < rows.length && rows[studentStartRow].length > nameColIndex) {
      final checkStr = rows[studentStartRow][0].toString().toLowerCase();
      if (checkStr.contains('faculty') || checkStr.contains('code') || checkStr.contains('type')) {
        studentStartRow++;
      }
    }

    for (int r = studentStartRow; r < rows.length; r++) {
      final row = rows[r];
      if (row.isEmpty) continue;

      final firstCell = (row.isNotEmpty) ? row[0].toString().trim() : '';
      final rollCell = (rollColIndex < row.length) ? row[rollColIndex].toString().trim() : '';
      final nameCell = (nameColIndex < row.length) ? row[nameColIndex].toString().trim() : '';

      // Check for footer summary rows
      if (firstCell.toLowerCase().contains('number of students') ||
          firstCell.toLowerCase().contains('attendance monitoring') ||
          rollCell.toLowerCase().contains('number of students') ||
          nameCell.toLowerCase().contains('number of students')) {
        break; // Reached summary statistics footer
      }

      // Valid student row must have a roll number or name
      if (rollCell.isEmpty && nameCell.isEmpty) continue;

      // Extract subject percentages
      final Map<String, double?> subjectPcts = {};
      int rowDefaulters = 0;

      for (final sub in subjects) {
        if (sub.colIndex < row.length) {
          final valStr = row[sub.colIndex].toString().trim();
          if (valStr.isEmpty || valStr.toUpperCase() == 'NA' || valStr == '-') {
            subjectPcts[sub.subjectCode] = null; // Student not enrolled in this elective
          } else {
            final parsedPct = double.tryParse(valStr);
            subjectPcts[sub.subjectCode] = parsedPct;
            if (parsedPct != null && parsedPct < minCriteria) {
              rowDefaulters++;
            }
          }
        } else {
          subjectPcts[sub.subjectCode] = null;
        }
      }

      // Extract overall percentage
      double overallPct = 0.0;
      if (overallColIndex != -1 && overallColIndex < row.length) {
        final ovStr = row[overallColIndex].toString().trim();
        overallPct = double.tryParse(ovStr) ?? _computeAverage(subjectPcts);
      } else {
        overallPct = _computeAverage(subjectPcts);
      }

      // Extract count of defaulter courses
      int defaulterCount = rowDefaulters;
      if (defaulterColIndex != -1 && defaulterColIndex < row.length) {
        final defStr = row[defaulterColIndex].toString().trim();
        defaulterCount = int.tryParse(defStr) ?? rowDefaulters;
      }

      // Extract Critical status
      bool isCritical = false;
      if (criticalColIndex != -1 && criticalColIndex < row.length) {
        final critStr = row[criticalColIndex].toString().toUpperCase().trim();
        isCritical = critStr.contains('CRITICAL') || critStr == 'YES' || critStr == 'TRUE';
      } else {
        isCritical = overallPct < 65.0 || defaulterCount >= 3;
      }

      if (isCritical) criticalCount++;

      int srNo = studentRows.length + 1;
      if (firstCell.isNotEmpty) {
        srNo = int.tryParse(firstCell) ?? (studentRows.length + 1);
      }

      studentRows.add(StudentAttendanceRow(
        srNo: srNo,
        rollNo: rollCell.isNotEmpty ? rollCell : 'ROLL_${srNo}',
        studentName: nameCell.isNotEmpty ? nameCell : 'Student $srNo',
        subjectPercentages: subjectPcts,
        overallPercentage: overallPct,
        defaulterCount: defaulterCount,
        isCritical: isCritical,
      ));
    }

    return ParsedAttendanceReport(
      branch: branch,
      department: department,
      className: className,
      section: section,
      semester: semester,
      cycleName: cycleName,
      dateRange: dateRange,
      minCriteria: minCriteria,
      programCoordinator: programCoordinator,
      subjects: subjects,
      studentRows: studentRows,
      totalStudents: studentRows.length,
      criticalCount: criticalCount,
      isFormatValid: true,
    );
  }

  /// Verifies whether the logged-in mentor is authorized to upload this report
  static Map<String, dynamic> validateMentorAccess({
    required UserModel mentor,
    required ParsedAttendanceReport report,
  }) {
    if (mentor.isAdmin) {
      return {'isAllowed': true, 'reason': 'Admin authorized for all classes.'};
    }

    if (!mentor.isMentor) {
      return {
        'isAllowed': false,
        'reason': 'Only faculty mentors or administrators can upload attendance reports.'
      };
    }

    final assignedClass = (mentor.assignedClass ?? '').toUpperCase().trim();
    if (assignedClass.isEmpty) {
      return {
        'isAllowed': false,
        'reason': 'You do not have an assigned class in your mentor profile. Please contact the administrator.'
      };
    }

    // Build comparable string patterns
    // Report gives: className e.g. "BTech CSE Sem 5", section e.g. "A", semester e.g. "5"
    // Assigned class could be: "CSE 5A", "CSE 4A", "5A", "BTech CSE Sem 5 Sec A"
    final repClassNorm = report.className.toUpperCase().replaceAll(' ', '');
    final repSecNorm = report.section.toUpperCase().replaceAll(' ', '');
    final repSem = report.semester;
    final assignedNorm = assignedClass.replaceAll(' ', '');

    // Common matches
    final isDirectMatch = assignedNorm.contains('CSE${repSem}$repSecNorm') ||
        assignedNorm.contains('${repSem}$repSecNorm') ||
        (assignedNorm.contains(repClassNorm) && assignedNorm.contains(repSecNorm)) ||
        (assignedNorm.contains('SEM$repSem') && assignedNorm.contains(repSecNorm));

    if (isDirectMatch) {
      return {'isAllowed': true, 'reason': 'Class matches assigned mentor allocation.'};
    }

    return {
      'isAllowed': false,
      'reason': 'Access Denied: You are assigned to "$assignedClass", but this attendance report is for "${report.className} (Sec ${report.section})". You may only upload attendance for your own class.',
    };
  }

  // ── Helper parsing methods ──────────────────────────────────────────

  static String _extractAfterColon(String line, String key) {
    final idx = line.toLowerCase().indexOf(key);
    if (idx == -1) return '';
    final rest = line.substring(idx + key.length).trim();
    // Split by comma if any
    final parts = rest.split(',');
    return parts.first.trim();
  }

  static String _extractDivision(String line) {
    if (line.toLowerCase().contains('division:')) {
      return _extractAfterColon(line, 'division:');
    }
    if (line.toLowerCase().contains('section:')) {
      return _extractAfterColon(line, 'section:');
    }
    return 'A';
  }

  static String _extractSemester(String className) {
    final match = RegExp(r'sem(?:ester)?\s*(\d+)', caseSensitive: false).firstMatch(className);
    if (match != null) {
      return match.group(1)!;
    }
    final digitMatch = RegExp(r'\b([1-8])\b').firstMatch(className);
    if (digitMatch != null) {
      return digitMatch.group(1)!;
    }
    return '5';
  }

  static String _extractCycleName(String fullText) {
    final upper = fullText.toUpperCase();
    if (upper.contains('FIRST ATTENDANCE') || upper.contains('1ST ATTENDANCE')) {
      return 'FIRST ATTENDANCE MONITORING REPORT';
    }
    if (upper.contains('SECOND ATTENDANCE') || upper.contains('2ND ATTENDANCE')) {
      return 'SECOND ATTENDANCE MONITORING REPORT';
    }
    if (upper.contains('THIRD ATTENDANCE') || upper.contains('3RD ATTENDANCE')) {
      return 'THIRD ATTENDANCE MONITORING REPORT';
    }
    if (upper.contains('FINAL ATTENDANCE')) {
      return 'FINAL ATTENDANCE REPORT';
    }
    return 'SECOND ATTENDANCE MONITORING REPORT';
  }

  static double _computeAverage(Map<String, double?> subjectPcts) {
    final validValues = subjectPcts.values.where((v) => v != null).map((v) => v!).toList();
    if (validValues.isEmpty) return 0.0;
    final sum = validValues.reduce((a, b) => a + b);
    return double.parse((sum / validValues.length).toStringAsFixed(2));
  }
}
