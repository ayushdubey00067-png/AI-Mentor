// lib/screens/student/student_documents_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/auth_provider.dart';
import '../../services/supabase_service.dart';
import '../../utils/app_theme.dart';
import '../../utils/file_opener.dart';
import '../../widgets/document_viewer_dialog.dart';
import '../../widgets/timetable_grid_view.dart';
import '../../widgets/timetable_sync_dialog.dart';
import '../../widgets/teacher_selector_dialog.dart';
import '../../utils/mru_timetable_data.dart';

class StudentDocumentsScreen extends StatefulWidget {
  const StudentDocumentsScreen({super.key});
  @override
  State<StudentDocumentsScreen> createState() => _StudentDocumentsScreenState();
}

class _StudentDocumentsScreenState extends State<StudentDocumentsScreen> {
  List<StudentDocument> _docs = [];
  bool _loading = false;

  // Academic Session & Term Filter State
  String _selectedAcademicYear = '2026-2027';
  String _selectedTerm = 'odd'; // 'odd' (Jun-Dec) or 'even' (Jan-May)
  static const List<String> _academicYears = [
    '2026-2027',
    '2025-2026',
    '2024-2025'
  ];

  static const List<Map<String, dynamic>> _docTypes = [
    {
      'value': 'academic_calendar',
      'label': 'Academic Calendar',
      'emoji': '🗓️',
      'color': Color(0xFF8B5CF6)
    },
    {
      'value': 'syllabus',
      'label': 'Syllabus',
      'emoji': '📚',
      'color': Color(0xFF10B981)
    },
    {
      'value': 'marksheet',
      'label': 'Marksheet / Marks',
      'emoji': '📊',
      'color': Color(0xFFF59E0B)
    },
    {
      'value': 'attendance',
      'label': 'Attendance Sheet',
      'emoji': '✅',
      'color': Color(0xFF06B6D4)
    },
    {
      'value': 'assignment',
      'label': 'Assignment',
      'emoji': '📝',
      'color': Color(0xFFEF4444)
    },
    {
      'value': 'other',
      'label': 'Policy / Other',
      'emoji': '🏛️',
      'color': Color(0xFF6B7280)
    },
  ];

  @override
  void initState() {
    super.initState();
    // Auto-detect term based on current date
    final now = DateTime.now();
    if (now.month >= 6) {
      _selectedAcademicYear = '${now.year}-${now.year + 1}';
      _selectedTerm = 'odd';
    } else {
      _selectedAcademicYear = '${now.year - 1}-${now.year}';
      _selectedTerm = 'even';
    }
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadDocs());
  }

  Future<void> _openDocumentRaw(StudentDocument doc) async {
    try {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (_) => const Center(child: CircularProgressIndicator()),
      );
      final bytes = await SupabaseService.getDocumentBytes(doc);
      if (mounted) Navigator.pop(context);

      if (bytes != null && bytes.isNotEmpty) {
        openRawBytes(
          bytes: bytes,
          fileName: doc.fileName,
          mimeType: doc.mimeType,
        );
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Opening authentic document "${doc.fileName}"...'),
              duration: const Duration(seconds: 2),
              behavior: SnackBarBehavior.floating,
            ),
          );
        }
      } else if (doc.contentBase64 != null && doc.contentBase64!.isNotEmpty) {
        openRawDocument(
          base64Content: doc.contentBase64!,
          fileName: doc.fileName,
          mimeType: doc.mimeType,
        );
      } else {
        if (mounted) {
          DocumentViewerDialog.show(context, doc);
        }
      }
    } catch (e) {
      if (mounted && Navigator.canPop(context)) Navigator.pop(context);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not open document: $e')),
        );
      }
    }
  }

  Widget _buildTermFilterBar() {
    final isOdd = _selectedTerm == 'odd';
    final now = DateTime.now();
    final currentYear = now.month >= 6
        ? '${now.year}-${now.year + 1}'
        : '${now.year - 1}-${now.year}';
    final currentTerm = now.month >= 6 ? 'odd' : 'even';
    final isCurrentSession =
        _selectedAcademicYear == currentYear && _selectedTerm == currentTerm;

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          LayoutBuilder(
            builder: (context, constraints) {
              final isCompact = constraints.maxWidth < 380;
              final activeTermBadge = isCurrentSession
                  ? Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 7, vertical: 2),
                      decoration: BoxDecoration(
                        color: const Color(0xFFECFDF5),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: const Color(0xFFA7F3D0)),
                      ),
                      child: Text(
                        'Active Term',
                        style: GoogleFonts.lato(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          color: const Color(0xFF047857),
                        ),
                      ),
                    )
                  : const SizedBox.shrink();
              final yearSelector = Container(
                padding: EdgeInsets.symmetric(
                  horizontal: isCompact ? 6 : 10,
                  vertical: 2,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFFF9FAFB),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: const Color(0xFFE5E7EB)),
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: _selectedAcademicYear,
                    isDense: true,
                    iconSize: isCompact ? 18 : 24,
                    style: GoogleFonts.lato(
                      fontSize: isCompact ? 10 : 12,
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF1F2937),
                    ),
                    items: _academicYears.map((yr) {
                      return DropdownMenuItem<String>(
                        value: yr,
                        child: Text(yr),
                      );
                    }).toList(),
                    onChanged: (val) {
                      if (val != null)
                        setState(() => _selectedAcademicYear = val);
                    },
                  ),
                ),
              );
              final heading = Row(
                children: [
                  if (!isCompact) ...[
                    const Icon(Icons.filter_list_rounded,
                        size: 18, color: AppTheme.primary),
                    const SizedBox(width: 8),
                  ],
                  Expanded(
                    child: Wrap(
                      alignment: WrapAlignment.start,
                      spacing: 8,
                      runSpacing: 4,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        Text(
                          'Academic Session & Term',
                          style: GoogleFonts.lato(
                            fontSize: isCompact ? 12 : 13,
                            fontWeight: FontWeight.w700,
                            color: const Color(0xFF111827),
                          ),
                        ),
                        if (isCurrentSession) activeTermBadge,
                        if (isCompact) yearSelector,
                      ],
                    ),
                  ),
                  if (!isCompact) ...[
                    const SizedBox(width: 8),
                    yearSelector,
                  ],
                ],
              );

              return heading;
            },
          ),
          const SizedBox(height: 12),
          // Term Switcher (Odd vs Even)
          Row(
            children: [
              Expanded(
                child: GestureDetector(
                  onTap: () => setState(() => _selectedTerm = 'odd'),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 9),
                    decoration: BoxDecoration(
                      color: isOdd ? AppTheme.primary : const Color(0xFFF9FAFB),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color:
                            isOdd ? AppTheme.primary : const Color(0xFFE5E7EB),
                      ),
                    ),
                    child: Center(
                      child: Text(
                        'Odd Semester (Jun - Dec)',
                        style: GoogleFonts.lato(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: isOdd ? Colors.white : const Color(0xFF4B5563),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: GestureDetector(
                  onTap: () => setState(() => _selectedTerm = 'even'),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 9),
                    decoration: BoxDecoration(
                      color:
                          !isOdd ? AppTheme.primary : const Color(0xFFF9FAFB),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color:
                            !isOdd ? AppTheme.primary : const Color(0xFFE5E7EB),
                      ),
                    ),
                    child: Center(
                      child: Text(
                        'Even Semester (Jan - May)',
                        style: GoogleFonts.lato(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color:
                              !isOdd ? Colors.white : const Color(0xFF4B5563),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Future<void> _loadDocs() async {
    final auth = context.read<AuthProvider>();
    if (auth.currentUser == null) return;
    setState(() => _loading = true);
    final docs = await SupabaseService.getStudentAccessibleDocuments(
      studentId: auth.currentUser!.id,
      rollNo: auth.currentUser!.rollNumber,
      program: auth.currentUser!.program,
      branch: auth.currentUser!.branch,
    );
    if (mounted)
      setState(() {
        _docs = docs;
        _loading = false;
      });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF0F4FF),
      appBar: AppBar(
        backgroundColor: AppTheme.primary,
        elevation: 0,
        title: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Academic Documents',
              style: GoogleFonts.playfairDisplay(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: Colors.white)),
          Text('Mentor-published schedules, syllabus & records',
              style: GoogleFonts.lato(fontSize: 11, color: Colors.white70)),
        ]),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded,
                color: Colors.white, size: 20),
            onPressed: _loadDocs,
          ),
        ],
      ),
      body: _loading
          ? const Center(
              child: CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation(AppTheme.primary)))
          : _docsList(),
    );
  }

  Widget _docsList() {
    final auth = context.watch<AuthProvider>();
    // Filter documents matching the selected academic year and term
    final sessionDocs = _docs.where((doc) {
      final matchesYear =
          (doc.academicYear ?? '2026-2027') == _selectedAcademicYear;
      final matchesTerm = doc.term == _selectedTerm;
      return matchesYear && matchesTerm;
    }).toList();

    // Deduplicate by category: keep only the latest active document per category & scope (excluding timetable which has its own card)
    final Map<String, StudentDocument> categoryMap = {};
    for (final doc in sessionDocs) {
      if (doc.docType == 'timetable') continue;
      final key = '${doc.docType}_${doc.targetScope}_${doc.targetRollNo ?? ""}';
      if (!categoryMap.containsKey(key)) {
        categoryMap[key] = doc;
      }
    }
    final displayDocs = categoryMap.values.toList();

    return RefreshIndicator(
      onRefresh: _loadDocs,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
        children: [
          Container(
            padding: const EdgeInsets.all(14),
            margin: const EdgeInsets.only(bottom: 14),
            decoration: BoxDecoration(
                color: const Color(0xFFEFF6FF),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: const Color(0xFFBFDBFE))),
            child: Row(children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: const BoxDecoration(
                  color: Color(0xFFDBEAFE),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.verified_user_rounded,
                    color: Color(0xFF2563EB), size: 20),
              ),
              const SizedBox(width: 12),
              Expanded(
                  child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Institutional Repository',
                      style: GoogleFonts.lato(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: const Color(0xFF1E40AF))),
                  const SizedBox(height: 2),
                  Text(
                      'These materials are officially uploaded and verified by your faculty mentor. Tap any document to open the authentic raw PDF.',
                      style: GoogleFonts.lato(
                          fontSize: 11,
                          color: const Color(0xFF3B82F6),
                          height: 1.4)),
                ],
              )),
            ]),
          ),
          _buildTermFilterBar(),
          _buildTimetableSectionCard(auth, sessionDocs),
          if (displayDocs.isEmpty)
            Container(
              padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFE5E7EB)),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.folder_open_rounded,
                      size: 48, color: Color(0xFF9CA3AF)),
                  const SizedBox(height: 12),
                  Text(
                    'No Other Documents for $_selectedAcademicYear (${_selectedTerm.toUpperCase()})',
                    style: GoogleFonts.lato(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF374151),
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Your mentor has not published additional verified materials (such as syllabus or marksheets) for this session yet.',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.lato(
                        fontSize: 12, color: const Color(0xFF6B7280)),
                  ),
                ],
              ),
            )
          else ...[
            ..._buildGroupedList(displayDocs),
          ],
        ],
      ),
    );
  }

  Widget _buildTimetableSectionCard(
      AuthProvider auth, List<StudentDocument> sessionDocs) {
    final studentSection = MRUTimetableRepository.resolveSection(
      program: auth.currentUser?.program,
      branch: auth.currentUser?.branch,
      semester: auth.currentUser?.semester,
      section: auth.currentUser?.section,
    );
    final mentorName =
        auth.currentUser?.mentorEmail?.split('@').first ?? 'PRINIMA GUPTA';
    final resolvedTeacher = MRUTimetableRepository.resolveTeacher(mentorName);

    final timetableDoc = sessionDocs.firstWhere(
      (d) => d.docType == 'timetable',
      orElse: () => _docs.firstWhere(
        (d) => d.docType == 'timetable',
        orElse: () => StudentDocument(
          id: 'official_timetable',
          title: 'Class Timetable ($studentSection)',
          fileName: 'OFFICIAL TIMETABLE $studentSection ODD 2026-27.pdf',
          mimeType: 'application/pdf',
          docType: 'timetable',
          targetScope: 'class',
          createdAt: DateTime.now(),
        ),
      ),
    );

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE2E8F0), width: 1.2),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF0F172A).withValues(alpha: 0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFFEFF6FF),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: const Color(0xFFDBEAFE)),
                ),
                child: const Icon(Icons.calendar_month_rounded,
                    color: Color(0xFF2563EB), size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        Text(
                          'Class & Faculty Timetable',
                          style: GoogleFonts.lato(
                            fontSize: 16,
                            fontWeight: FontWeight.w800,
                            color: const Color(0xFF0F172A),
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 2.5),
                          decoration: BoxDecoration(
                            color: const Color(0xFFEFF6FF),
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(color: const Color(0xFFBFDBFE)),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.lock_outline_rounded,
                                  size: 11, color: Color(0xFF2563EB)),
                              const SizedBox(width: 3),
                              Text(
                                studentSection,
                                style: GoogleFonts.lato(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w800,
                                  color: const Color(0xFF1D4ED8),
                                ),
                              ),
                            ],
                          ),
                        ),
                        GestureDetector(
                          onTap: () {
                            TimetableSyncDialog.show(
                              context,
                              targetSection: studentSection,
                              userRole: 'student',
                              onSynced: () {
                                if (mounted) setState(() {});
                              },
                            );
                          },
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 7, vertical: 4),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF0FDF4),
                              borderRadius: BorderRadius.circular(7),
                              border:
                                  Border.all(color: const Color(0xFFBBF7D0)),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(Icons.sync_rounded,
                                    size: 11, color: Color(0xFF16A34A)),
                                const SizedBox(width: 4),
                                Text(
                                  'Sync MRU',
                                  style: GoogleFonts.lato(
                                    fontSize: 9,
                                    fontWeight: FontWeight.w800,
                                    color: const Color(0xFF15803D),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Manav Rachna University | Odd Term 2026-2027',
                      style: GoogleFonts.lato(
                        fontSize: 11.5,
                        color: const Color(0xFF64748B),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Official lecture hours, 100-min lab allocations, faculty details, and classroom assignments verified from mru.edupage.org • Last Synced: ${MRUTimetableRepository.getSectionLastSyncedFormatted(studentSection)}',
            style: GoogleFonts.lato(
              fontSize: 12,
              color: const Color(0xFF64748B),
              height: 1.4,
            ),
          ),
          const SizedBox(height: 16),

          // Two Distinct Clickable Buttons (Student vs Teacher Timetable)
          LayoutBuilder(
            builder: (context, constraints) {
              final isCompact = constraints.maxWidth < 380;
              final studentButton = Tooltip(
                message: 'Open student timetable',
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0F172A),
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(
                      horizontal: isCompact ? 4 : 12,
                      vertical: isCompact ? 9 : 13,
                    ),
                    minimumSize: Size(0, isCompact ? 40 : 48),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8)),
                    elevation: 0,
                  ),
                  onPressed: () => TimetableViewerDialog.show(
                    context,
                    className: studentSection,
                    teacherName: resolvedTeacher,
                    initialMode: 'student',
                    document: timetableDoc,
                    userRole: 'student',
                  ),
                  child: isCompact
                      ? Text(
                          'Student Timetable',
                          maxLines: 1,
                          softWrap: false,
                          style: GoogleFonts.lato(
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        )
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.school_rounded, size: 16),
                            const SizedBox(width: 8),
                            Flexible(
                              child: Text(
                                'Student Timetable ($studentSection)',
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.lato(
                                  fontSize: 12.5,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                          ],
                        ),
                ),
              );
              final teacherButton = Tooltip(
                message: 'Open teacher timetable',
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0284C7),
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(
                      horizontal: isCompact ? 4 : 12,
                      vertical: isCompact ? 9 : 13,
                    ),
                    minimumSize: Size(0, isCompact ? 40 : 48),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8)),
                    elevation: 0,
                  ),
                  onPressed: () => TeacherSelectorDialog.showAndOpenViewer(
                    context,
                    className: studentSection,
                    currentTeacher: resolvedTeacher,
                    document: timetableDoc,
                    userRole: 'student',
                  ),
                  child: isCompact
                      ? Text(
                          'Teacher Timetable',
                          maxLines: 1,
                          softWrap: false,
                          style: GoogleFonts.lato(
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        )
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.person_search_rounded, size: 16),
                            const SizedBox(width: 8),
                            const Text(
                              'Teacher Timetable',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                fontSize: 12.5,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                ),
              );
              return Row(
                children: [
                  Expanded(child: studentButton),
                  SizedBox(width: isCompact ? 6 : 10),
                  Expanded(child: teacherButton),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  List<Widget> _buildGroupedList(List<StudentDocument> docs) {
    final grouped = <String, List<StudentDocument>>{};
    for (final d in docs) {
      grouped.putIfAbsent(d.docType, () => []).add(d);
    }

    final widgets = <Widget>[];
    for (final entry in grouped.entries) {
      final ti = _docTypes.firstWhere((t) => t['value'] == entry.key,
          orElse: () => _docTypes.last);
      widgets.add(Padding(
        padding: const EdgeInsets.only(top: 8, bottom: 8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
              color: (ti['color'] as Color).withOpacity(0.1),
              borderRadius: BorderRadius.circular(20)),
          child: Text('${ti['emoji']} ${ti['label']} (${entry.value.length})',
              style: GoogleFonts.lato(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: ti['color'] as Color)),
        ),
      ));
      for (final doc in entry.value) {
        widgets.add(_docCard(doc, ti));
      }
      widgets.add(const SizedBox(height: 6));
    }
    return widgets;
  }

  Widget _docCard(StudentDocument doc, Map<String, dynamic> ti) {
    final color = ti['color'] as Color;
    final indexed = doc.extractedText?.isNotEmpty == true;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 8,
              offset: const Offset(0, 3)),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () {
            if (doc.docType == 'timetable') {
              final auth = context.read<AuthProvider>();
              TimetableViewerDialog.show(
                context,
                className: auth.currentUser?.section ?? 'CSE 5A',
                document: doc,
              );
            } else {
              _openDocumentRaw(doc);
            }
          },
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(children: [
              Container(
                width: 50,
                height: 50,
                decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12)),
                child: Center(
                    child: Text(ti['emoji'] as String,
                        style: const TextStyle(fontSize: 22))),
              ),
              const SizedBox(width: 12),
              Expanded(
                  child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    Expanded(
                      child: Text(doc.title,
                          style: GoogleFonts.lato(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                              color: const Color(0xFF111827))),
                    ),
                    if (doc.targetScope == 'class')
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: const Color(0xFFF3F4F6),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text('Class',
                            style: GoogleFonts.lato(
                                fontSize: 10,
                                fontWeight: FontWeight.w600,
                                color: const Color(0xFF4B5563))),
                      )
                    else if (doc.targetRollNo != null)
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFEF3C7),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text('Personal',
                            style: GoogleFonts.lato(
                                fontSize: 10,
                                fontWeight: FontWeight.w700,
                                color: const Color(0xFFB45309))),
                      ),
                  ]),
                  const SizedBox(height: 2),
                  Text(doc.fileName,
                      style: GoogleFonts.lato(
                          fontSize: 12, color: const Color(0xFF6B7280)),
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 6),
                  Row(children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                          color: indexed
                              ? const Color(0xFFF0FDF4)
                              : const Color(0xFFFFF7ED),
                          borderRadius: BorderRadius.circular(8)),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        Icon(
                          indexed
                              ? Icons.check_circle_rounded
                              : Icons.info_outline_rounded,
                          size: 11,
                          color: indexed
                              ? const Color(0xFF16A34A)
                              : const Color(0xFFD97706),
                        ),
                        const SizedBox(width: 4),
                        Text(indexed ? 'AI Active' : 'Indexed',
                            style: GoogleFonts.lato(
                                fontSize: 10,
                                fontWeight: FontWeight.w600,
                                color: indexed
                                    ? const Color(0xFF16A34A)
                                    : const Color(0xFFD97706))),
                      ]),
                    ),
                    const SizedBox(width: 8),
                    Text(
                        DateFormat('MMM d, yyyy')
                            .format(doc.createdAt.toLocal()),
                        style: GoogleFonts.lato(
                            fontSize: 11, color: const Color(0xFF9CA3AF))),
                  ]),
                ],
              )),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: const Icon(Icons.description_outlined,
                        size: 20, color: Color(0xFF6B7280)),
                    tooltip: 'View Document Details',
                    onPressed: () {
                      if (doc.docType == 'timetable') {
                        final auth = context.read<AuthProvider>();
                        TimetableViewerDialog.show(
                          context,
                          className: auth.currentUser?.section ?? 'CSE 5A',
                          document: doc,
                        );
                      } else {
                        DocumentViewerDialog.show(context, doc);
                      }
                    },
                  ),
                  const Icon(Icons.open_in_new_rounded,
                      color: AppTheme.primary, size: 20),
                ],
              ),
            ]),
          ),
        ),
      ),
    );
  }
}
