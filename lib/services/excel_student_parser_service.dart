// lib/services/excel_student_parser_service.dart
import 'dart:typed_data';
import 'package:excel/excel.dart';

class BatchStudentParseResult {
  final List<Map<String, dynamic>> students;
  final String detectedProgram;
  final String detectedBranch;
  final String detectedSection;
  final String detectedSemester;
  final String detectedDepartment;
  final int totalRows;
  final int validCount;
  final List<String> warnings;

  BatchStudentParseResult({
    required this.students,
    required this.detectedProgram,
    required this.detectedBranch,
    required this.detectedSection,
    required this.detectedSemester,
    required this.detectedDepartment,
    required this.totalRows,
    required this.validCount,
    required this.warnings,
  });
}

class ExcelStudentParserService {
  /// Parses bytes of an Excel file (.xlsx) into a structured student directory
  static Future<BatchStudentParseResult> parseStudentExcel(Uint8List fileBytes) async {
    final excel = Excel.decodeBytes(fileBytes);
    final warnings = <String>[];

    // 1. Locate the Student Directory sheet
    String? targetSheetName;
    for (final name in excel.tables.keys) {
      if (name.toLowerCase().contains('student') || name.toLowerCase().contains('directory')) {
        targetSheetName = name;
        break;
      }
    }
    targetSheetName ??= excel.tables.keys.firstOrNull;

    if (targetSheetName == null || excel.tables[targetSheetName] == null) {
      throw Exception('No valid sheets found in the Excel workbook.');
    }

    final sheet = excel.tables[targetSheetName]!;
    if (sheet.maxRows < 3) {
      throw Exception('Sheet "$targetSheetName" does not contain sufficient header or student rows.');
    }

    // 2. Extract Metadata from Top Header Rows (Rows 1 & 2 if present)
    String detectedProgram = 'B.Tech';
    String detectedBranch = 'Computer Science & Engineering';
    String detectedSection = 'A';
    String detectedSemester = '4';
    String detectedDepartment = 'Dept. of Computer Science & Technology';

    for (int r = 0; r < 2 && r < sheet.rows.length; r++) {
      final line = sheet.rows[r]
          .map((c) => _cellToString(c?.value))
          .where((s) => s.isNotEmpty)
          .join(' | ');

      if (line.contains('BTech') || line.contains('B.Tech')) detectedProgram = 'B.Tech';
      if (line.contains('CSE') || line.contains('Computer Science')) detectedBranch = 'Computer Science & Engineering';
      
      final semMatch = RegExp(r'Sem(?:ester)?\s*([0-9])', caseSensitive: false).firstMatch(line);
      if (semMatch != null) detectedSemester = semMatch.group(1)!;

      final secMatch = RegExp(r'Section\s*([A-Z0-9]+)|CSE\s*[0-9]+([A-Z])', caseSensitive: false).firstMatch(line);
      if (secMatch != null) {
        detectedSection = secMatch.group(1) ?? secMatch.group(2) ?? 'A';
      }

      if (line.contains('Dept.') || line.contains('Department')) {
        final parts = line.split('|');
        for (final p in parts) {
          if (p.toLowerCase().contains('dept') || p.toLowerCase().contains('department')) {
            detectedDepartment = p.trim();
          }
        }
      }
    }

    // 3. Locate the column headers row (usually Row index 2 -> 3rd row)
    int headerRowIndex = 2;
    Map<String, int> colMap = {};

    for (int r = 0; r < sheet.rows.length && r < 6; r++) {
      final row = sheet.rows[r];
      final rowTexts = row.map((c) => _cellToString(c?.value).toLowerCase().trim()).toList();
      if (rowTexts.any((t) => t.contains('roll') || t.contains('official email') || t.contains('name'))) {
        headerRowIndex = r;
        for (int c = 0; c < rowTexts.length; c++) {
          final h = rowTexts[c];
          if (h.contains('roll')) colMap['roll_number'] = c;
          else if (h.contains('full name') || h == 'name' || h.contains('student name')) colMap['full_name'] = c;
          else if (h.contains('gender') || h == 'sex') colMap['gender'] = c;
          else if (h.contains('official email') || h.contains('college email')) colMap['official_email'] = c;
          else if (h.contains('personal email') || h.contains('alt email')) colMap['personal_email'] = c;
          else if (h.contains('mobile') && !h.contains('father') && !h.contains('mother')) colMap['mobile_no'] = c;
          else if (h.contains('father') && h.contains('name')) colMap['father_name'] = c;
          else if (h.contains('father') && h.contains('mobile')) colMap['father_mobile'] = c;
          else if (h.contains('mother') && h.contains('name')) colMap['mother_name'] = c;
          else if (h.contains('mother') && h.contains('mobile')) colMap['mother_mobile'] = c;
          else if (h == 'section' || h.contains('sec')) colMap['section'] = c;
          else if (h.contains('class')) colMap['student_class'] = c;
          else if (h.contains('state') || h.contains('domicile')) colMap['domicile_state'] = c;
          else if (h.contains('pincode') || h.contains('pin')) colMap['pincode'] = c;
          else if (h.contains('application')) colMap['application_no'] = c;
          else if (h.contains('admission') || h.contains('year')) colMap['admission_date'] = c;
          else if (h.contains('status')) colMap['status'] = c;
        }
        break;
      }
    }

    if (!colMap.containsKey('roll_number') && !colMap.containsKey('official_email')) {
      throw Exception('Could not locate essential headers (Roll No, Official Email) in sheet.');
    }

    // 4. Parse Student Data Rows
    final students = <Map<String, dynamic>>[];
    for (int r = headerRowIndex + 1; r < sheet.rows.length; r++) {
      final row = sheet.rows[r];
      if (row.isEmpty) continue;

      String getVal(String key) {
        final idx = colMap[key];
        if (idx == null || idx >= row.length) return 'NA';
        final val = _cellToString(row[idx]?.value);
        return val.isEmpty ? 'NA' : val;
      }

      final rollNo = getVal('roll_number');
      final fullName = getVal('full_name');
      final officialEmail = getVal('official_email');

      // Skip empty or placeholder rows
      if (rollNo == 'NA' && officialEmail == 'NA') continue;
      if (rollNo.toLowerCase().contains('total') || fullName.toLowerCase().contains('total')) continue;

      final mobileNo = _cleanPhone(getVal('mobile_no'));
      final personalEmail = getVal('personal_email');
      final sec = getVal('section') != 'NA' ? getVal('section') : detectedSection;
      final stClass = getVal('student_class') != 'NA' 
          ? getVal('student_class') 
          : '$detectedProgram $detectedBranch Sem $detectedSemester (Sec $sec)';

      students.add({
        'roll_number': rollNo,
        'full_name': fullName != 'NA' ? fullName : 'Student $rollNo',
        'gender': getVal('gender'),
        'official_email': officialEmail != 'NA' ? officialEmail.toLowerCase() : '${rollNo.toLowerCase()}@mru.ac.in',
        'personal_email': personalEmail != 'NA' ? personalEmail.toLowerCase() : null,
        'mobile_no': mobileNo.isNotEmpty ? mobileNo : '9999999999',
        'password_hash': mobileNo.isNotEmpty ? mobileNo : '9999999999',
        'father_name': getVal('father_name'),
        'father_mobile': _cleanPhone(getVal('father_mobile')),
        'mother_name': getVal('mother_name'),
        'mother_mobile': _cleanPhone(getVal('mother_mobile')),
        'section': sec,
        'student_class': stClass,
        'program': detectedProgram,
        'branch': detectedBranch,
        'department': detectedDepartment,
        'semester': detectedSemester,
        'base_semester': detectedSemester,
        'domicile_state': getVal('domicile_state'),
        'pincode': getVal('pincode'),
        'application_no': getVal('application_no'),
        'admission_date': getVal('admission_date'),
        'status': getVal('status') != 'NA' ? getVal('status') : 'active',
      });
    }

    return BatchStudentParseResult(
      students: students,
      detectedProgram: detectedProgram,
      detectedBranch: detectedBranch,
      detectedSection: detectedSection,
      detectedSemester: detectedSemester,
      detectedDepartment: detectedDepartment,
      totalRows: sheet.rows.length,
      validCount: students.length,
      warnings: warnings,
    );
  }

  static String _cellToString(CellValue? cell) {
    if (cell == null) return '';
    if (cell is TextCellValue) return cell.value.toString().trim();
    if (cell is IntCellValue) return cell.value.toString().trim();
    if (cell is DoubleCellValue) return cell.value.toInt().toString().trim();
    if (cell is DateCellValue) return '${cell.year}-${cell.month}-${cell.day}';
    return cell.toString().trim();
  }

  static String _cleanPhone(String raw) {
    if (raw == 'NA' || raw.isEmpty) return 'NA';
    final digits = raw.replaceAll(RegExp(r'[^0-9]'), '');
    if (digits.length >= 10) {
      return digits.substring(digits.length - 10);
    }
    return digits.isNotEmpty ? digits : 'NA';
  }
}
