import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/models.dart';
import '../utils/file_opener.dart';
import '../utils/mru_timetable_data.dart';
import 'timetable_sync_dialog.dart';
import 'teacher_selector_dialog.dart';

class TimetableViewerDialog extends StatefulWidget {
  final String className;
  final String? teacherName;
  final String initialMode; // 'student' or 'teacher'
  final StudentDocument? document;
  final String userRole;

  const TimetableViewerDialog({
    super.key,
    required this.className,
    this.teacherName,
    this.initialMode = 'student',
    this.document,
    this.userRole = 'student',
  });

  static Future<void> show(
    BuildContext context, {
    required String className,
    String? teacherName,
    String initialMode = 'student',
    StudentDocument? document,
    String userRole = 'student',
  }) {
    return showDialog(
      context: context,
      barrierDismissible: true,
      builder: (_) => TimetableViewerDialog(
        className: className,
        teacherName: teacherName,
        initialMode: initialMode,
        document: document,
        userRole: userRole,
      ),
    );
  }

  @override
  State<TimetableViewerDialog> createState() => _TimetableViewerDialogState();
}

class _TimetableViewerDialogState extends State<TimetableViewerDialog> {
  late String _activeMode; // 'student' or 'teacher'
  late String _lockedSection;
  late String _assignedTeacher;
  late Map<String, Map<int, List<TimetableSlotEntry>>> _studentSchedule;
  late Map<String, Map<int, List<TimetableSlotEntry>>> _teacherSchedule;
  String _selectedGroupFilter = 'All'; // 'All', 'Group 1', 'Group 2'

  @override
  void initState() {
    super.initState();
    _activeMode = widget.initialMode;
    _lockedSection = widget.className.trim().isEmpty ? 'CSE 5A' : widget.className.trim();
    _assignedTeacher = MRUTimetableRepository.resolveTeacher(widget.teacherName);
    _reloadSchedules();
  }

  void _reloadSchedules() {
    _studentSchedule = MRUTimetableRepository.getClassSchedule(_lockedSection);
    _teacherSchedule = MRUTimetableRepository.getTeacherSchedule(_assignedTeacher);
  }

  Map<String, Map<int, List<TimetableSlotEntry>>> get _currentSchedule =>
      _activeMode == 'student' ? _studentSchedule : _teacherSchedule;

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final isMobile = screenWidth < 700;

    return Dialog(
      backgroundColor: Colors.white,
      insetPadding: EdgeInsets.symmetric(
        horizontal: isMobile ? 8 : 20,
        vertical: isMobile ? 12 : 20,
      ),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Container(
        width: 1320,
        height: 780,
        constraints: BoxConstraints(
          maxWidth: screenWidth * 0.98,
          maxHeight: MediaQuery.of(context).size.height * 0.96,
        ),
        child: Column(
          children: [
            // ── Dialog Header (Clean University Navy/White) ──
            _buildHeader(context),
            const Divider(height: 1, color: Color(0xFFCBD5E1)),

            // ── Mode Switcher & Group Filter Bar ──
            _buildControlsBar(),
            const Divider(height: 1, color: Color(0xFFCBD5E1)),

            // ── Timetable Grid Table (Unified 2D Scrollable) ──
            Expanded(
              child: SingleChildScrollView(
                scrollDirection: Axis.vertical,
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Padding(
                    padding: const EdgeInsets.all(14.0),
                    child: _buildTimetableGrid(),
                  ),
                ),
              ),
            ),

            // ── Bottom Action Bar ──
            const Divider(height: 1, color: Color(0xFFCBD5E1)),
            _buildFooter(context),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    final titleText = _activeMode == 'student'
        ? 'MANAV RACHNA UNIVERSITY • CLASS TIMETABLE'
        : 'MANAV RACHNA UNIVERSITY • FACULTY TIMETABLE';
    final badgeText = _activeMode == 'student' ? _lockedSection : _assignedTeacher;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: const BoxDecoration(
        color: Color(0xFF0F172A),
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: const Color(0xFF0284C7).withValues(alpha: 0.25),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(
              _activeMode == 'student' ? Icons.school_rounded : Icons.person_rounded,
              color: const Color(0xFF38BDF8),
              size: 22,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      titleText,
                      style: GoogleFonts.lato(
                        fontSize: 13.5,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1.0,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 10),
                    InkWell(
                      borderRadius: BorderRadius.circular(6),
                      onTap: _activeMode == 'teacher'
                          ? () async {
                              final chosen = await TeacherSelectorDialog.show(
                                context,
                                currentTeacher: _assignedTeacher,
                              );
                              if (chosen != null && mounted) {
                                await MRUTimetableRepository.saveSelectedTeacher(chosen);
                                setState(() {
                                  _assignedTeacher = chosen;
                                  _reloadSchedules();
                                });
                              }
                            }
                          : null,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                        decoration: BoxDecoration(
                          color: const Color(0xFF0284C7),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _activeMode == 'teacher' ? Icons.search_rounded : Icons.lock_outline_rounded,
                              size: 12,
                              color: Colors.white70,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              badgeText.toUpperCase(),
                              style: GoogleFonts.lato(
                                fontSize: 12,
                                fontWeight: FontWeight.w800,
                                color: Colors.white,
                              ),
                            ),
                            if (_activeMode == 'teacher') ...[
                              const SizedBox(width: 4),
                              const Icon(Icons.arrow_drop_down_rounded, size: 14, color: Colors.white70),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 3),
                Text(
                  'Sector 43, Faridabad • Last Synced: ${MRUTimetableRepository.getSectionLastSyncedFormatted(_lockedSection)} • aSc Timetables Online Verified',
                  style: GoogleFonts.lato(
                    fontSize: 11,
                    color: const Color(0xFF94A3B8),
                  ),
                ),
              ],
            ),
          ),
          ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF0284C7),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
              elevation: 0,
            ),
            onPressed: () {
              TimetableSyncDialog.show(
                context,
                targetSection: _lockedSection,
                userRole: widget.userRole,
                onSynced: () {
                  if (mounted) {
                    setState(() {
                      _reloadSchedules();
                    });
                  }
                },
              );
            },
            icon: const Icon(Icons.sync_rounded, size: 16),
            label: Text(
              'Sync MRU',
              style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w800),
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            onPressed: () => Navigator.of(context).pop(),
            icon: const Icon(Icons.close_rounded, color: Colors.white70),
            tooltip: 'Close',
          ),
        ],
      ),
    );
  }

  Widget _buildControlsBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: const Color(0xFFF8FAFC),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        physics: const BouncingScrollPhysics(),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Mode Switcher Tabs
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildModeTab('student', 'Student Timetable ($_lockedSection)', Icons.school_rounded),
                const SizedBox(width: 8),
                _buildModeTab('teacher', 'Teacher Timetable ($_assignedTeacher)', Icons.person_rounded),
              ],
            ),
            const SizedBox(width: 14),
            // Group Filter for Student mode or Search Switcher for Teacher mode
            if (_activeMode == 'student')
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Group:',
                    style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700, color: const Color(0xFF475569)),
                  ),
                  const SizedBox(width: 6),
                  ...['All', 'Group 1', 'Group 2'].map((grp) {
                    final isSelected = _selectedGroupFilter == grp;
                    return Padding(
                      padding: const EdgeInsets.only(right: 6),
                      child: InkWell(
                        borderRadius: BorderRadius.circular(6),
                        onTap: () => setState(() => _selectedGroupFilter = grp),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                          decoration: BoxDecoration(
                            color: isSelected ? const Color(0xFF0F172A) : Colors.white,
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(
                              color: isSelected ? const Color(0xFF0F172A) : const Color(0xFFCBD5E1),
                              width: 1.0,
                            ),
                          ),
                          child: Text(
                            grp,
                            style: GoogleFonts.lato(
                              fontSize: 11,
                              fontWeight: isSelected ? FontWeight.w800 : FontWeight.w600,
                              color: isSelected ? Colors.white : const Color(0xFF334155),
                            ),
                          ),
                        ),
                      ),
                    );
                  }),
                ],
              )
            else
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFF0284C7),
                      side: const BorderSide(color: Color(0xFF0284C7), width: 1.2),
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                    onPressed: () async {
                      final chosen = await TeacherSelectorDialog.show(
                        context,
                        currentTeacher: _assignedTeacher,
                      );
                      if (chosen != null && mounted) {
                        await MRUTimetableRepository.saveSelectedTeacher(chosen);
                        setState(() {
                          _assignedTeacher = chosen;
                          _reloadSchedules();
                        });
                      }
                    },
                    icon: const Icon(Icons.search_rounded, size: 14),
                    label: Text(
                      'Switch Teacher / Search',
                      style: GoogleFonts.lato(fontSize: 11.5, fontWeight: FontWeight.w800),
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildModeTab(String mode, String label, IconData icon) {
    final isSelected = _activeMode == mode;

    return InkWell(
      borderRadius: BorderRadius.circular(8),
      onTap: () => setState(() => _activeMode = mode),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF0284C7) : Colors.white,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected ? const Color(0xFF0284C7) : const Color(0xFFCBD5E1),
            width: 1.2,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 14,
              color: isSelected ? Colors.white : const Color(0xFF0284C7),
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: GoogleFonts.lato(
                fontSize: 12,
                fontWeight: isSelected ? FontWeight.w800 : FontWeight.w600,
                color: isSelected ? Colors.white : const Color(0xFF334155),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTimetableGrid() {
    const dayColWidth = 58.0;
    const periodWidth = 140.0;
    const lunchWidth = 64.0;
    const rowHeight = 120.0;

    const periods = MRUTimetableRepository.periodSlots;
    const days = MRUTimetableRepository.days;
    final schedule = _currentSchedule;

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: const Color(0xFFCBD5E1), width: 1.5),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header Row (Periods & Timings) ──
          Container(
            decoration: const BoxDecoration(
              color: Color(0xFFF1F5F9),
              border: Border(bottom: BorderSide(color: Color(0xFFCBD5E1), width: 1.5)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Day Corner Cell
                Container(
                  width: dayColWidth,
                  height: 48,
                  alignment: Alignment.center,
                  decoration: const BoxDecoration(
                    border: Border(right: BorderSide(color: Color(0xFFCBD5E1), width: 1.0)),
                  ),
                  child: Text(
                    'Day',
                    style: GoogleFonts.lato(
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                      color: const Color(0xFF334155),
                    ),
                  ),
                ),
                // Period Header Cells (P1 to P4, Lunch, P6 to P10)
                ...periods.map((slot) {
                  final isLunch = slot.label == 'Lunch';
                  final width = isLunch ? lunchWidth : periodWidth;
                  return Container(
                    width: width,
                    height: 48,
                    padding: const EdgeInsets.symmetric(horizontal: 2),
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: isLunch ? const Color(0xFFFEF3C7) : null,
                      border: const Border(right: BorderSide(color: Color(0xFFCBD5E1), width: 1.0)),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          isLunch ? 'Lunch' : '${slot.label}.',
                          style: GoogleFonts.lato(
                            fontSize: isLunch ? 12 : 13,
                            fontWeight: FontWeight.w900,
                            color: isLunch ? const Color(0xFF92400E) : const Color(0xFF0F172A),
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '${slot.startTime} - ${slot.endTime}',
                          style: GoogleFonts.lato(
                            fontSize: 9.5,
                            fontWeight: FontWeight.w700,
                            color: isLunch ? const Color(0xFFB45309) : const Color(0xFF64748B),
                          ),
                        ),
                      ],
                    ),
                  );
                }),
              ],
            ),
          ),

          // ── Day Rows (Mo, Tu, We, Th, Fr) ──
          ...days.asMap().entries.map((dayEntry) {
            final isLastDay = dayEntry.key == days.length - 1;
            final dayCode = dayEntry.value;
            final daySchedule = schedule[dayCode] ?? {};

            return Container(
              decoration: BoxDecoration(
                border: isLastDay
                    ? null
                    : const Border(bottom: BorderSide(color: Color(0xFFCBD5E1), width: 1.0)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Day Label Cell (Mo, Tu, We, Th, Fr)
                  Container(
                    width: dayColWidth,
                    height: rowHeight,
                    alignment: Alignment.center,
                    decoration: const BoxDecoration(
                      color: Color(0xFFF8FAFC),
                      border: Border(right: BorderSide(color: Color(0xFFCBD5E1), width: 1.0)),
                    ),
                    child: Text(
                      dayCode,
                      style: GoogleFonts.lato(
                        fontSize: 16,
                        fontWeight: FontWeight.w900,
                        color: const Color(0xFF1E293B),
                      ),
                    ),
                  ),

                  // Morning Slots: Periods 1 to 4 with 2-Period Lab Spanning
                  ..._buildPeriodRangeSlots(
                    startPeriod: 1,
                    endPeriod: 4,
                    daySchedule: daySchedule,
                    periodWidth: periodWidth,
                    rowHeight: rowHeight,
                  ),

                  // Lunch Break Column
                  Container(
                    width: lunchWidth,
                    height: rowHeight,
                    alignment: Alignment.center,
                    decoration: const BoxDecoration(
                      color: Color(0xFFFFFBEB),
                      border: Border(right: BorderSide(color: Color(0xFFCBD5E1), width: 1.0)),
                    ),
                    child: RotatedBox(
                      quarterTurns: 3,
                      child: Text(
                        'LUNCH',
                        style: GoogleFonts.lato(
                          fontSize: 10,
                          fontWeight: FontWeight.w900,
                          letterSpacing: 2.0,
                          color: const Color(0xFFD97706),
                        ),
                      ),
                    ),
                  ),

                  // Afternoon Slots: Periods 6 to 10 with 2-Period Lab Spanning
                  ..._buildPeriodRangeSlots(
                    startPeriod: 6,
                    endPeriod: 10,
                    daySchedule: daySchedule,
                    periodWidth: periodWidth,
                    rowHeight: rowHeight,
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  List<TimetableSlotEntry> _filterEntries(List<TimetableSlotEntry> rawEntries) {
    if (_activeMode == 'teacher' || _selectedGroupFilter == 'All') return rawEntries;
    final filtered = rawEntries.where((e) {
      final grp = e.group.toLowerCase();
      if (grp.isEmpty || grp.contains('entire') || grp.contains('all')) return true;
      if (_selectedGroupFilter == 'Group 1' && (grp.contains('g1') || grp.contains('group 1') || grp.contains('group-1'))) return true;
      if (_selectedGroupFilter == 'Group 2' && (grp.contains('g2') || grp.contains('group 2') || grp.contains('group-2'))) return true;
      return false;
    }).toList();

    return filtered.isNotEmpty ? filtered : rawEntries;
  }

  List<Widget> _buildPeriodRangeSlots({
    required int startPeriod,
    required int endPeriod,
    required Map<int, List<TimetableSlotEntry>> daySchedule,
    required double periodWidth,
    required double rowHeight,
  }) {
    final widgets = <Widget>[];
    int p = startPeriod;

    while (p <= endPeriod) {
      final rawEntries = daySchedule[p] ?? [];
      final entries = _filterEntries(rawEntries);
      final hasSpan = p < endPeriod && rawEntries.isNotEmpty && rawEntries.any((e) => e.durationPeriods >= 2);

      if (hasSpan) {
        final cellWidth = periodWidth * 2;
        widgets.add(
          Container(
            width: cellWidth,
            height: rowHeight,
            decoration: const BoxDecoration(
              border: Border(right: BorderSide(color: Color(0xFFCBD5E1), width: 1.0)),
            ),
            child: _buildSlotCell(
              entries: entries,
              width: cellWidth,
              height: rowHeight,
              isSpan: true,
              periodLabel: 'P$p-P${p + 1}',
            ),
          ),
        );
        p += 2;
      } else {
        widgets.add(
          Container(
            width: periodWidth,
            height: rowHeight,
            decoration: const BoxDecoration(
              border: Border(right: BorderSide(color: Color(0xFFCBD5E1), width: 1.0)),
            ),
            child: _buildSlotCell(
              entries: entries,
              width: periodWidth,
              height: rowHeight,
              isSpan: false,
              periodLabel: 'P$p',
            ),
          ),
        );
        p += 1;
      }
    }

    return widgets;
  }

  Widget _buildSlotCell({
    required List<TimetableSlotEntry> entries,
    required double width,
    required double height,
    required bool isSpan,
    required String periodLabel,
  }) {
    if (entries.isEmpty) {
      return Container(
        width: width,
        height: height,
        color: Colors.white,
      );
    }

    if (entries.length == 1) {
      final e = entries.first;

      return Container(
        width: width,
        height: height,
        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 4),
        color: Colors.white,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                if (e.rooms.isNotEmpty && e.rooms.first.isNotEmpty)
                  Flexible(
                    flex: 3,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1.5),
                      decoration: BoxDecoration(
                        color: const Color(0xFF0F172A),
                        borderRadius: BorderRadius.circular(3),
                      ),
                      child: Text(
                        e.rooms.join(','),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.lato(
                          fontSize: 8.5,
                          fontWeight: FontWeight.w800,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                if (e.group.isNotEmpty) ...[
                  const SizedBox(width: 4),
                  Flexible(
                    flex: 2,
                    child: Align(
                      alignment: Alignment.centerRight,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                        decoration: BoxDecoration(
                          color: const Color(0xFFEFF6FF),
                          border: Border.all(color: const Color(0xFFBFDBFE), width: 0.8),
                          borderRadius: BorderRadius.circular(3),
                        ),
                        child: Text(
                          e.group,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.lato(
                            fontSize: 7.5,
                            fontWeight: FontWeight.w800,
                            color: const Color(0xFF1D4ED8),
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
            Expanded(
              child: Center(
                child: Text(
                  e.subject,
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.left,
                  style: GoogleFonts.lato(
                    fontSize: isSpan ? 10.5 : 9.5,
                    fontWeight: FontWeight.w800,
                    color: const Color(0xFF0F172A),
                    height: 1.18,
                  ),
                ),
              ),
            ),
            Text(
              e.teachers.join(', '),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.lato(
                fontSize: 8.5,
                fontWeight: FontWeight.w700,
                color: const Color(0xFF475569),
              ),
            ),
          ],
        ),
      );
    }

    if (isSpan && entries.length == 2) {
      return Container(
        width: width,
        height: height,
        color: Colors.white,
        child: Column(
          children: [
            Expanded(
              child: _buildHorizontalLabHalf(entries[0], isTop: true),
            ),
            const Divider(height: 1, thickness: 1, color: Color(0xFFCBD5E1)),
            Expanded(
              child: _buildHorizontalLabHalf(entries[1], isTop: false),
            ),
          ],
        ),
      );
    }

    return Container(
      width: width,
      height: height,
      color: Colors.white,
      padding: const EdgeInsets.all(3),
      child: SingleChildScrollView(
        physics: const ClampingScrollPhysics(),
        child: Column(
          children: entries.map((e) {
            return Container(
              margin: const EdgeInsets.symmetric(vertical: 1.5),
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
              decoration: BoxDecoration(
                color: const Color(0xFFF8FAFC),
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: const Color(0xFFE2E8F0), width: 0.8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      if (e.rooms.isNotEmpty)
                        Flexible(
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 1),
                            decoration: BoxDecoration(
                              color: const Color(0xFF0F172A),
                              borderRadius: BorderRadius.circular(2),
                            ),
                            child: Text(
                              e.rooms.join(','),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.lato(
                                fontSize: 7.5,
                                fontWeight: FontWeight.w800,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ),
                      if (e.group.isNotEmpty) ...[
                        const SizedBox(width: 3),
                        Flexible(
                          child: Text(
                            e.group,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            textAlign: TextAlign.right,
                            style: GoogleFonts.lato(
                              fontSize: 7.0,
                              fontWeight: FontWeight.w800,
                              color: const Color(0xFF2563EB),
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 1.5),
                  Text(
                    e.subject,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.lato(
                      fontSize: 8.5,
                      fontWeight: FontWeight.w800,
                      color: const Color(0xFF0F172A),
                    ),
                  ),
                  if (e.teachers.isNotEmpty)
                    Text(
                      e.teachers.join(', '),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.lato(fontSize: 7.5, color: const Color(0xFF64748B)),
                    ),
                ],
              ),
            );
          }).toList(),
        ),
      ),
    );
  }

  Widget _buildHorizontalLabHalf(TimetableSlotEntry e, {required bool isTop}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      color: Colors.white,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              if (e.rooms.isNotEmpty && e.rooms.first.isNotEmpty)
                Flexible(
                  flex: 3,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0F172A),
                      borderRadius: BorderRadius.circular(3),
                    ),
                    child: Text(
                      e.rooms.join(','),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.lato(
                        fontSize: 8.5,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
              if (e.group.isNotEmpty) ...[
                const SizedBox(width: 4),
                Flexible(
                  flex: 2,
                  child: Text(
                    e.group,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    textAlign: TextAlign.right,
                    style: GoogleFonts.lato(
                      fontSize: 8.0,
                      fontWeight: FontWeight.w800,
                      color: const Color(0xFF1D4ED8),
                    ),
                  ),
                ),
              ],
            ],
          ),
          Center(
            child: Text(
              e.subject,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.lato(
                fontSize: 9.0,
                fontWeight: FontWeight.w800,
                color: const Color(0xFF0F172A),
              ),
            ),
          ),
          Text(
            e.teachers.join(', '),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.lato(
              fontSize: 8.0,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF475569),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFooter(BuildContext context) {
    final statusText = _activeMode == 'student'
        ? 'Class $_lockedSection verified from mru.edupage.org'
        : 'Faculty $_assignedTeacher verified from mru.edupage.org';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      decoration: const BoxDecoration(
        color: Color(0xFFF8FAFC),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(20)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              const Icon(Icons.verified_rounded, size: 14, color: Color(0xFF0284C7)),
              const SizedBox(width: 6),
              Text(
                '$statusText • Real-time synchronized.',
                style: GoogleFonts.lato(fontSize: 11, fontWeight: FontWeight.w600, color: const Color(0xFF475569)),
              ),
            ],
          ),
          Row(
            children: [
              OutlinedButton.icon(
                style: OutlinedButton.styleFrom(
                  foregroundColor: const Color(0xFF0284C7),
                  side: const BorderSide(color: Color(0xFF0284C7)),
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                onPressed: () {
                  TimetableSyncDialog.show(
                    context,
                    targetSection: _lockedSection,
                    userRole: widget.userRole,
                    onSynced: () {
                      if (mounted) {
                        setState(() {
                          _reloadSchedules();
                        });
                      }
                    },
                  );
                },
                icon: const Icon(Icons.sync_rounded, size: 16),
                label: Text('Sync from MRU Portal', style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700)),
              ),
              const SizedBox(width: 8),
              if (widget.document != null)
                OutlinedButton.icon(
                  style: OutlinedButton.styleFrom(
                    foregroundColor: const Color(0xFF0F172A),
                    side: const BorderSide(color: Color(0xFFCBD5E1)),
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  onPressed: () {
                    if (widget.document?.contentBase64 != null) {
                      openRawDocument(
                        base64Content: widget.document!.contentBase64!,
                        fileName: widget.document!.fileName,
                        mimeType: 'application/pdf',
                      );
                    }
                  },
                  icon: const Icon(Icons.picture_as_pdf_outlined, size: 16),
                  label: Text('Official Vector PDF', style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700)),
                ),
              const SizedBox(width: 10),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF0F172A),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  elevation: 0,
                ),
                onPressed: () => Navigator.of(context).pop(),
                child: Text('Close', style: GoogleFonts.lato(fontSize: 12.5, fontWeight: FontWeight.w700)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
