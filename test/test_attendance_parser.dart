import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ai_chatbot/services/attendance_parser_service.dart';

void main() {
  test('AttendanceParserService parses 5CSA CSV file accurately', () {
    final file = File(r'C:\Users\hp\Downloads\5CSA_second Attendance monitoring.csv');
    expect(file.existsSync(), isTrue);

    final bytes = file.readAsBytesSync();
    final report = AttendanceParserService.parseBytes(bytes, '5CSA_second Attendance monitoring.csv');

    expect(report.isFormatValid, isTrue);
    expect(report.semester, equals('5'));
    expect(report.section, equals('A'));
    expect(report.cycleName, equals('SECOND ATTENDANCE MONITORING REPORT'));
    expect(report.subjects.length, equals(18));
    expect(report.totalStudents, equals(54));

    debugPrint('✅ Verified: ${report.subjects.length} subjects parsed.');
    debugPrint('✅ Verified: ${report.totalStudents} students parsed.');
    debugPrint('✅ Verified: ${report.criticalCount} critical students detected.');

    final firstStudent = report.studentRows.first;
    expect(firstStudent.rollNo, equals('2K24CSUN01002'));
    expect(firstStudent.studentName, equals('Aayush Dubey'));
    expect(firstStudent.overallPercentage, equals(86.96));
    debugPrint('✅ Verified First Student: ${firstStudent.studentName} (${firstStudent.rollNo}) -> ${firstStudent.overallPercentage}%');
  });
}
