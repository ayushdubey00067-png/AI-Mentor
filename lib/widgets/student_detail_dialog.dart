// lib/widgets/student_detail_dialog.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/models.dart';
import '../services/supabase_service.dart';
import '../utils/app_theme.dart';
import '../utils/academic_session_utils.dart';
import 'edit_section_semester_dialog.dart';

class StudentDetailDialog extends StatefulWidget {
  final UserModel student;
  final VoidCallback onDataChanged;

  const StudentDetailDialog({
    super.key,
    required this.student,
    required this.onDataChanged,
  });

  static Future<void> show(
    BuildContext context, {
    required UserModel student,
    required VoidCallback onDataChanged,
  }) async {
    await showDialog(
      context: context,
      builder: (_) => StudentDetailDialog(
        student: student,
        onDataChanged: onDataChanged,
      ),
    );
  }

  @override
  State<StudentDetailDialog> createState() => _StudentDetailDialogState();
}

class _StudentDetailDialogState extends State<StudentDetailDialog> {
  bool _isLoadingAttendance = true;
  StudentAttendanceSummary? _attendanceSummary;
  List<AttendanceRecord> _attendanceRecords = [];

  @override
  void initState() {
    super.initState();
    _loadAttendance();
  }

  Future<void> _loadAttendance() async {
    final roll = widget.student.rollNumber;
    if (roll == null || roll.isEmpty) {
      if (mounted) setState(() => _isLoadingAttendance = false);
      return;
    }

    try {
      final summary = await SupabaseService.getStudentAttendanceSummary(
        roll,
        semester: widget.student.semester,
      );
      final records = await SupabaseService.getStudentAttendanceRecords(
        roll,
        semester: widget.student.semester,
      );

      if (mounted) {
        setState(() {
          _attendanceSummary = summary;
          _attendanceRecords = records;
          _isLoadingAttendance = false;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _isLoadingAttendance = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final size = MediaQuery.of(context).size;
    final student = widget.student;

    final effectiveSem = AcademicSessionUtils.getEffectiveSemester(
      baseSemester: student.baseSemester ?? student.semester ?? '4',
      manualOverrideSemester: student.semester,
      baseYear: student.baseYear,
    );

    final initials = student.name.trim().split(' ')
        .map((w) => w.isNotEmpty ? w[0] : '').take(2).join().toUpperCase();

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      backgroundColor: isDark ? const Color(0xFF111827) : Colors.white,
      child: Container(
        width: size.width > 750 ? 700 : size.width * 0.95,
        constraints: BoxConstraints(maxHeight: size.height * 0.9),
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header ──────────────────────────────────────────
            Row(
              children: [
                CircleAvatar(
                  radius: 28,
                  backgroundColor: AppTheme.primaryNavy,
                  child: Text(
                    initials,
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Flexible(
                            child: Text(
                              student.name,
                              style: GoogleFonts.outfit(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: isDark ? Colors.white : AppTheme.primaryNavy,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFF10B981).withOpacity(0.15),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              student.status ?? 'active',
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                                color: const Color(0xFF10B981),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'Roll Number: ${student.rollNumber ?? 'N/A'} • Semester $effectiveSem (${student.section ?? 'Sec A'})',
                        style: GoogleFonts.inter(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: AppTheme.accentGold,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
            const Divider(height: 24),

            // ── Scrollable Body ─────────────────────────────────
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Section 0: Live Official Attendance Status (If records available)
                    _buildSectionHeader(Icons.fact_check_outlined, 'Official Academic Attendance (Semester $effectiveSem)'),
                    _buildAttendanceCard(isDark),
                    const SizedBox(height: 18),

                    // Section 1: Academic Identity
                    _buildSectionHeader(Icons.school_outlined, 'Academic Identity & Department'),
                    _buildInfoCard(isDark, [
                      _buildRow('Program / Degree', student.program ?? 'B.Tech'),
                      _buildRow('Branch / Specialization', student.branch ?? 'Computer Science & Engineering'),
                      _buildRow('Department', student.department ?? 'Dept. of Computer Science & Technology'),
                      _buildRow('Class Label', student.studentClass ?? 'BTech CSE Sem $effectiveSem'),
                      _buildRow('Current Semester', 'Semester $effectiveSem'),
                      _buildRow('Section', student.section ?? 'A'),
                      _buildRow('Assigned Mentor', student.mentorEmail ?? 'None Assigned'),
                    ]),
                    const SizedBox(height: 18),

                    // Section 2: Login & Authentication
                    _buildSectionHeader(Icons.vpn_key_outlined, 'Login & Contact Credentials'),
                    _buildInfoCard(isDark, [
                      _buildRow('Official Email (Primary Login)', student.officialEmail ?? student.email),
                      _buildRow('Personal Email (Alt Login)', student.personalEmail ?? 'NA'),
                      _buildRow('Mobile Number (Login Password)', student.mobileNo ?? student.phone ?? 'NA'),
                      _buildRow('Gender', student.gender ?? 'NA'),
                    ]),
                    const SizedBox(height: 18),

                    // Section 3: Guardian & Family Information
                    _buildSectionHeader(Icons.family_restroom_outlined, 'Guardian & Family Contacts'),
                    _buildInfoCard(isDark, [
                      _buildRow("Father's Name", student.fatherName ?? 'NA'),
                      _buildRow("Father's Mobile", student.fatherMobile ?? 'NA'),
                      _buildRow("Mother's Name", student.motherName ?? 'NA'),
                      _buildRow("Mother's Mobile", student.motherMobile ?? 'NA'),
                    ]),
                    const SizedBox(height: 18),

                    // Section 4: Institutional & Admission Records
                    _buildSectionHeader(Icons.badge_outlined, 'Institutional & Demographic Records'),
                    _buildInfoCard(isDark, [
                      _buildRow('Application Number', student.applicationNo ?? 'NA'),
                      _buildRow('Admission Date', student.admissionDate ?? 'NA'),
                      _buildRow('Domicile State', student.domicileState ?? 'NA'),
                      _buildRow('Pincode', student.pincode ?? 'NA'),
                      _buildRow('Profile Created', student.createdAt.toLocal().toString().split('.')[0]),
                      _buildRow('Last Active', student.lastActive.toLocal().toString().split('.')[0]),
                    ]),
                  ],
                ),
              ),
            ),
            const Divider(height: 24),

            // ── Footer Actions ──────────────────────────────────
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                OutlinedButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                    EditSectionSemesterDialog.show(
                      context,
                      student: student,
                      onSaved: widget.onDataChanged,
                    );
                  },
                  icon: const Icon(Icons.edit_note_rounded, size: 18),
                  label: const Text('Edit Section & Semester'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppTheme.primaryNavy,
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: () => Navigator.pop(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryNavy,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                  child: const Text('Close'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAttendanceCard(bool isDark) {
    if (_isLoadingAttendance) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1F2937) : const Color(0xFFF9FAFB),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(child: CircularProgressIndicator(strokeWidth: 2)),
      );
    }

    if (_attendanceSummary == null && _attendanceRecords.isEmpty) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1F2937) : const Color(0xFFF9FAFB),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.grey.withOpacity(0.15)),
        ),
        child: Text(
          'No attendance records uploaded for this semester yet.',
          style: GoogleFonts.inter(fontSize: 12, color: Colors.grey.shade600),
        ),
      );
    }

    final overall = _attendanceSummary?.overallPercentage ??
        (_attendanceRecords.isNotEmpty
            ? _attendanceRecords.map((r) => r.attendancePercentage).reduce((a, b) => a + b) / _attendanceRecords.length
            : 0.0);
    final isCritical = _attendanceSummary?.isCritical ?? (overall < 75.0);
    final defaulters = _attendanceSummary?.defaulterSubjectCount ??
        _attendanceRecords.where((r) => r.isDefaulter).length;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1F2937) : const Color(0xFFF9FAFB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isCritical ? Colors.red.withOpacity(0.4) : Colors.grey.withOpacity(0.15),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Summary Banner
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: overall >= 75.0 ? const Color(0xFF10B981).withOpacity(0.15) : const Color(0xFFEF4444).withOpacity(0.15),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Overall: ${overall.toStringAsFixed(1)}%',
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                    color: overall >= 75.0 ? const Color(0xFF10B981) : const Color(0xFFEF4444),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              if (defaulters > 0)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.orange.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    '$defaulters Subject(s) <75%',
                    style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.orange),
                  ),
                ),
              const Spacer(),
              if (isCritical)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(color: Colors.red, borderRadius: BorderRadius.circular(6)),
                  child: const Text('CRITICAL ALERT', style: TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
                ),
            ],
          ),
          if (_attendanceRecords.isNotEmpty) ...[
            const SizedBox(height: 12),
            const Divider(height: 1),
            const SizedBox(height: 8),
            Text(
              'Enrolled Subjects & Faculty Breakdown:',
              style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.grey.shade700),
            ),
            const SizedBox(height: 6),
            ..._attendanceRecords.map((rec) {
              final isSubDefaulter = rec.attendancePercentage < 75.0;
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  children: [
                    Expanded(
                      flex: 4,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            rec.subjectName,
                            style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600),
                          ),
                          Text(
                            'Code: ${rec.subjectCode} • ${rec.facultyName ?? "Faculty"}',
                            style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      flex: 3,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: (rec.attendancePercentage / 100.0).clamp(0.0, 1.0),
                          backgroundColor: Colors.grey.shade200,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            isSubDefaulter ? Colors.red : const Color(0xFF10B981),
                          ),
                          minHeight: 6,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      '${rec.attendancePercentage.toStringAsFixed(1)}%',
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: isSubDefaulter ? Colors.red : const Color(0xFF10B981),
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
        ],
      ),
    );
  }

  Widget _buildSectionHeader(IconData icon, String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Icon(icon, size: 16, color: AppTheme.accentGold),
          const SizedBox(width: 8),
          Text(
            title,
            style: GoogleFonts.outfit(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: AppTheme.primaryNavy,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard(bool isDark, List<Widget> children) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1F2937) : const Color(0xFFF9FAFB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.withOpacity(0.15)),
      ),
      child: Column(children: children),
    );
  }

  Widget _buildRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 180,
            child: Text(
              label,
              style: GoogleFonts.inter(
                fontSize: 12,
                color: Colors.grey.shade600,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              value,
              style: GoogleFonts.inter(
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
