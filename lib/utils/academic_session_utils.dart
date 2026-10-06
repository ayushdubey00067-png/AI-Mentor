// lib/utils/academic_session_utils.dart

class AcademicSessionUtils {
  /// Returns 'odd' (July-December) or 'even' (January-June)
  static String getCurrentTerm([DateTime? date]) {
    final d = date ?? DateTime.now();
    return (d.month >= 7 && d.month <= 12) ? 'odd' : 'even';
  }

  /// Returns academic year string like '2025-2026' or '2026-2027'
  static String getCurrentAcademicYear([DateTime? date]) {
    final d = date ?? DateTime.now();
    if (d.month >= 7) {
      return '${d.year}-${d.year + 1}';
    } else {
      return '${d.year - 1}-${d.year}';
    }
  }

  /// Dynamically computes active semester from base semester & session date
  /// If student was enrolled in July-Dec (Odd, e.g. Sem 3), in Jan-June (Even) it becomes Sem 4.
  /// If manualOverrideSemester is non-null and not empty, it takes precedence.
  static String getEffectiveSemester({
    required String baseSemester,
    String? manualOverrideSemester,
    String? baseYear,
    DateTime? currentDate,
  }) {
    if (manualOverrideSemester != null && manualOverrideSemester.trim().isNotEmpty) {
      return manualOverrideSemester.trim();
    }

    final parsedBase = int.tryParse(baseSemester.replaceAll(RegExp(r'[^0-9]'), ''));
    if (parsedBase == null) return baseSemester;

    final now = currentDate ?? DateTime.now();
    final currentYearStr = getCurrentAcademicYear(now);
    final currentTerm = getCurrentTerm(now);

    // If no base year provided, infer from current session
    final baseYearStr = baseYear ?? currentYearStr;

    // Calculate year difference
    final baseStartYear = int.tryParse(baseYearStr.split('-').first) ?? now.year;
    final currentStartYear = int.tryParse(currentYearStr.split('-').first) ?? now.year;
    final yearDiff = currentStartYear - baseStartYear;

    int computedSem = parsedBase + (yearDiff * 2);
    // If we transitioned from odd to even in the same academic year
    if (currentTerm == 'even' && parsedBase % 2 != 0) {
      computedSem += 1;
    }

    // Upper bound cap at 8 for standard 4-year engineering
    if (computedSem > 8) computedSem = 8;
    if (computedSem < 1) computedSem = 1;

    return computedSem.toString();
  }

  /// Formats class string e.g. "BTech CSE Sem 5 (Sec A)"
  static String formatStudentClass({
    required String program,
    required String branch,
    required String semester,
    required String section,
  }) {
    final shortBranch = branch.contains('Computer') ? 'CSE' : branch;
    return '$program $shortBranch Sem $semester (Sec $section)';
  }
}
