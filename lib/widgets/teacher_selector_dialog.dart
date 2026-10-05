import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/models.dart';
import '../utils/mru_timetable_data.dart';
import 'timetable_grid_view.dart';

class TeacherSelectorDialog extends StatefulWidget {
  final String? currentTeacher;
  final String? className;
  final String userRole;
  final StudentDocument? document;
  final bool autoOpenViewer;

  const TeacherSelectorDialog({
    super.key,
    this.currentTeacher,
    this.className,
    this.userRole = 'student',
    this.document,
    this.autoOpenViewer = false,
  });

  /// Displays the search & select dialog and returns the chosen teacher name.
  static Future<String?> show(
    BuildContext context, {
    String? currentTeacher,
  }) {
    return showDialog<String>(
      context: context,
      barrierDismissible: true,
      builder: (_) => TeacherSelectorDialog(
        currentTeacher: currentTeacher,
        autoOpenViewer: false,
      ),
    );
  }

  /// Displays the search & select dialog, and upon selection immediately opens the TimetableViewerDialog.
  static Future<void> showAndOpenViewer(
    BuildContext context, {
    required String className,
    String? currentTeacher,
    String userRole = 'student',
    StudentDocument? document,
  }) async {
    final selectedTeacher = await showDialog<String>(
      context: context,
      barrierDismissible: true,
      builder: (_) => TeacherSelectorDialog(
        currentTeacher: currentTeacher,
        className: className,
        userRole: userRole,
        document: document,
        autoOpenViewer: false,
      ),
    );

    if (selectedTeacher != null && context.mounted) {
      // Save user preference
      await MRUTimetableRepository.saveSelectedTeacher(selectedTeacher);
      if (context.mounted) {
        TimetableViewerDialog.show(
          context,
          className: className,
          teacherName: selectedTeacher,
          initialMode: 'teacher',
          document: document,
          userRole: userRole,
        );
      }
    }
  }

  @override
  State<TeacherSelectorDialog> createState() => _TeacherSelectorDialogState();
}

class _TeacherSelectorDialogState extends State<TeacherSelectorDialog> {
  final TextEditingController _searchController = TextEditingController();
  List<String> _allTeachers = [];
  List<String> _filteredTeachers = [];

  // Popular / Quick access teachers
  final List<String> _quickSuggestions = [
    'PRINIMA GUPTA',
    'BABITA YADAV',
    'SIMPLE SHARMA',
    'DR GUNJAN',
    'POOJA AHUJA',
    'DEEPAK SHARMA',
    'ANUPAMA PRASHAR',
  ];

  @override
  void initState() {
    super.initState();
    _allTeachers = MRUTimetableRepository.availableTeachers;
    _filteredTeachers = List.from(_allTeachers);
    _searchController.addListener(_onSearchChanged);
  }

  @override
  void dispose() {
    _searchController.removeListener(_onSearchChanged);
    _searchController.dispose();
    super.dispose();
  }

  void _onSearchChanged() {
    final query = _searchController.text.trim().toLowerCase();
    setState(() {
      if (query.isEmpty) {
        _filteredTeachers = List.from(_allTeachers);
      } else {
        _filteredTeachers = _allTeachers.where((teacher) {
          final tLower = teacher.toLowerCase();
          return tLower.contains(query);
        }).toList();
      }
    });
  }

  Color _getAvatarColor(String name) {
    final colors = [
      const Color(0xFF0284C7),
      const Color(0xFF0D9488),
      const Color(0xFF7C3AED),
      const Color(0xFFD97706),
      const Color(0xFF2563EB),
      const Color(0xFF059669),
      const Color(0xFFE11D48),
      const Color(0xFF4F46E5),
    ];
    final hash = name.codeUnits.fold(0, (prev, curr) => prev + curr);
    return colors[hash % colors.length];
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final isMobile = screenWidth < 600;

    return Dialog(
      backgroundColor: Colors.white,
      insetPadding: EdgeInsets.symmetric(
        horizontal: isMobile ? 12 : 24,
        vertical: isMobile ? 16 : 30,
      ),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Container(
        width: 620,
        height: 680,
        constraints: BoxConstraints(
          maxWidth: screenWidth * 0.95,
          maxHeight: MediaQuery.of(context).size.height * 0.90,
        ),
        child: Column(
          children: [
            // ── Header ──
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
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
                    child: const Icon(Icons.person_search_rounded, color: Color(0xFF38BDF8), size: 22),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Select Faculty Timetable',
                          style: GoogleFonts.lato(
                            fontSize: 16,
                            fontWeight: FontWeight.w900,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '${_allTeachers.length} Manav Rachna University Faculty Members',
                          style: GoogleFonts.lato(
                            fontSize: 11.5,
                            color: const Color(0xFF94A3B8),
                          ),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.close_rounded, color: Colors.white70),
                    tooltip: 'Close',
                  ),
                ],
              ),
            ),

            // ── Live Search Input Bar ──
            Container(
              padding: const EdgeInsets.fromLTRB(18, 16, 18, 8),
              color: const Color(0xFFF8FAFC),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFFCBD5E1), width: 1.2),
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFF0F172A).withValues(alpha: 0.04),
                          blurRadius: 8,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: TextField(
                      controller: _searchController,
                      autofocus: true,
                      style: GoogleFonts.lato(fontSize: 14, fontWeight: FontWeight.w600, color: const Color(0xFF0F172A)),
                      decoration: InputDecoration(
                        hintText: 'Search faculty by name (e.g. Rajik, Prinima, Babita)...',
                        hintStyle: GoogleFonts.lato(fontSize: 13, color: const Color(0xFF94A3B8)),
                        prefixIcon: const Icon(Icons.search_rounded, color: Color(0xFF0284C7), size: 20),
                        suffixIcon: _searchController.text.isNotEmpty
                            ? IconButton(
                                icon: const Icon(Icons.clear_rounded, size: 18, color: Color(0xFF64748B)),
                                onPressed: () {
                                  _searchController.clear();
                                },
                              )
                            : null,
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),

                  // Quick Suggestion Chips
                  SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      children: [
                        Text(
                          'Quick:',
                          style: GoogleFonts.lato(fontSize: 11, fontWeight: FontWeight.w700, color: const Color(0xFF64748B)),
                        ),
                        const SizedBox(width: 6),
                        ..._quickSuggestions.map((teacher) {
                          final isMatch = _allTeachers.contains(teacher);
                          if (!isMatch) return const SizedBox.shrink();
                          return Padding(
                            padding: const EdgeInsets.only(right: 6.0),
                            child: ActionChip(
                              label: Text(
                                teacher,
                                style: GoogleFonts.lato(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700,
                                  color: const Color(0xFF0369A1),
                                ),
                              ),
                              backgroundColor: const Color(0xFFEFF6FF),
                              side: const BorderSide(color: Color(0xFFBFDBFE)),
                              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 0),
                              visualDensity: VisualDensity.compact,
                              onPressed: () {
                                _searchController.text = teacher;
                              },
                            ),
                          );
                        }),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const Divider(height: 1, color: Color(0xFFE2E8F0)),

            // ── Search Results Count Header ──
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
              color: Colors.white,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'FACULTY DIRECTORY (${_filteredTeachers.length} FOUND)',
                    style: GoogleFonts.lato(
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0.5,
                      color: const Color(0xFF64748B),
                    ),
                  ),
                  if (_searchController.text.isNotEmpty)
                    Text(
                      'Filtering by "${_searchController.text.trim()}"',
                      style: GoogleFonts.lato(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: const Color(0xFF0284C7),
                      ),
                    ),
                ],
              ),
            ),
            const Divider(height: 1, color: Color(0xFFF1F5F9)),

            // ── Scrollable Teacher List ──
            Expanded(
              child: _filteredTeachers.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.person_off_rounded, size: 48, color: Color(0xFF94A3B8)),
                          const SizedBox(height: 12),
                          Text(
                            'No faculty found matching "${_searchController.text}"',
                            style: GoogleFonts.lato(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                              color: const Color(0xFF475569),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Try searching with partial spelling or first name',
                            style: GoogleFonts.lato(fontSize: 12, color: const Color(0xFF94A3B8)),
                          ),
                        ],
                      ),
                    )
                  : ListView.separated(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                      itemCount: _filteredTeachers.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 6),
                      itemBuilder: (context, index) {
                        final teacherName = _filteredTeachers[index];
                        final slotCount = MRUTimetableRepository.getTeacherSlotCount(teacherName);
                        final avatarColor = _getAvatarColor(teacherName);
                        final isCurrentlySelected = widget.currentTeacher != null &&
                            MRUTimetableRepository.resolveTeacher(widget.currentTeacher) == teacherName;

                        return InkWell(
                          onTap: () {
                            Navigator.of(context).pop(teacherName);
                          },
                          borderRadius: BorderRadius.circular(12),
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
                            decoration: BoxDecoration(
                              color: isCurrentlySelected ? const Color(0xFFF0F9FF) : Colors.white,
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(
                                color: isCurrentlySelected ? const Color(0xFF38BDF8) : const Color(0xFFE2E8F0),
                                width: isCurrentlySelected ? 1.5 : 1.0,
                              ),
                            ),
                            child: Row(
                              children: [
                                // Avatar Circle with Initial
                                CircleAvatar(
                                  radius: 18,
                                  backgroundColor: avatarColor.withValues(alpha: 0.15),
                                  child: Text(
                                    teacherName.isNotEmpty ? teacherName[0].toUpperCase() : 'T',
                                    style: GoogleFonts.lato(
                                      fontSize: 14,
                                      fontWeight: FontWeight.w900,
                                      color: avatarColor,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 12),

                                // Teacher Name & Metadata
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        teacherName,
                                        style: GoogleFonts.lato(
                                          fontSize: 13.5,
                                          fontWeight: FontWeight.w800,
                                          color: const Color(0xFF0F172A),
                                        ),
                                      ),
                                      const SizedBox(height: 2),
                                      Row(
                                        children: [
                                          const Icon(Icons.menu_book_rounded, size: 12, color: Color(0xFF64748B)),
                                          const SizedBox(width: 4),
                                          Text(
                                            '$slotCount teaching periods / week',
                                            style: GoogleFonts.lato(
                                              fontSize: 11.5,
                                              fontWeight: FontWeight.w500,
                                              color: const Color(0xFF64748B),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ],
                                  ),
                                ),

                                // Selection Pill / Action Button
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF0284C7).withValues(alpha: 0.1),
                                    borderRadius: BorderRadius.circular(8),
                                    border: Border.all(color: const Color(0xFF0284C7).withValues(alpha: 0.3)),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Text(
                                        'View Timetable',
                                        style: GoogleFonts.lato(
                                          fontSize: 11,
                                          fontWeight: FontWeight.w700,
                                          color: const Color(0xFF0284C7),
                                        ),
                                      ),
                                      const SizedBox(width: 3),
                                      const Icon(Icons.arrow_forward_ios_rounded, size: 10, color: Color(0xFF0284C7)),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
            ),

            // ── Footer ──
            const Divider(height: 1, color: Color(0xFFE2E8F0)),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Real-time data from mru.edupage.org',
                    style: GoogleFonts.lato(fontSize: 11, color: const Color(0xFF94A3B8)),
                  ),
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: Text(
                      'Cancel',
                      style: GoogleFonts.lato(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: const Color(0xFF64748B),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
