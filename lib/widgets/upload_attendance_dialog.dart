// lib/widgets/upload_attendance_dialog.dart
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/models.dart';
import '../services/attendance_parser_service.dart';
import '../services/supabase_service.dart';
import '../utils/app_theme.dart';

class UploadAttendanceDialog extends StatefulWidget {
  final UserModel currentUser;
  final VoidCallback onUploadSuccess;

  const UploadAttendanceDialog({
    super.key,
    required this.currentUser,
    required this.onUploadSuccess,
  });

  static Future<void> show(
    BuildContext context, {
    required UserModel currentUser,
    required VoidCallback onUploadSuccess,
  }) async {
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => UploadAttendanceDialog(
        currentUser: currentUser,
        onUploadSuccess: onUploadSuccess,
      ),
    );
  }

  @override
  State<UploadAttendanceDialog> createState() => _UploadAttendanceDialogState();
}

class _UploadAttendanceDialogState extends State<UploadAttendanceDialog>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  bool _isParsing = false;
  bool _isUploading = false;
  String? _selectedFileName;
  ParsedAttendanceReport? _parsedReport;
  Map<String, dynamic>? _authCheck;
  String _searchQuery = '';
  bool _filterCriticalOnly = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _pickAndParseFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['csv', 'xlsx', 'xls'],
        withData: true,
      );

      if (result == null || result.files.isEmpty) return;

      final file = result.files.first;
      final bytes = file.bytes;
      if (bytes == null) {
        _showSnackbar('Could not read file bytes.', isError: true);
        return;
      }

      setState(() {
        _isParsing = true;
        _selectedFileName = file.name;
        _parsedReport = null;
        _authCheck = null;
      });

      // Parse report using unified parser
      final report = AttendanceParserService.parseBytes(bytes, file.name);

      if (!report.isFormatValid) {
        setState(() {
          _isParsing = false;
          _parsedReport = report;
        });
        _showSnackbar(report.validationError ?? 'Invalid attendance format.', isError: true);
        return;
      }

      // Validate security access
      final auth = AttendanceParserService.validateMentorAccess(
        mentor: widget.currentUser,
        report: report,
      );

      setState(() {
        _isParsing = false;
        _parsedReport = report;
        _authCheck = auth;
      });
    } catch (e) {
      setState(() => _isParsing = false);
      _showSnackbar('File processing error: $e', isError: true);
    }
  }

  Future<void> _commitUpload() async {
    if (_parsedReport == null || _authCheck?['isAllowed'] != true) return;

    setState(() => _isUploading = true);

    try {
      final res = await SupabaseService.uploadAttendanceReport(
        report: _parsedReport!,
        uploaderEmail: widget.currentUser.email,
      );

      setState(() => _isUploading = false);

      if (res['success'] == true) {
        if (mounted) {
          Navigator.pop(context);
          _showSnackbar(
            '✅ Attendance report ingested! (${res['total_subjects_updated']} subjects, ${res['total_students_updated']} students)',
          );
          widget.onUploadSuccess();
        }
      } else {
        _showSnackbar('Upload failed: ${res['error']}', isError: true);
      }
    } catch (e) {
      setState(() => _isUploading = false);
      _showSnackbar('Upload error: $e', isError: true);
    }
  }

  void _showSnackbar(String msg, {bool isError = false}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        backgroundColor: isError ? Colors.red.shade700 : const Color(0xFF10B981),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final size = MediaQuery.of(context).size;

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      backgroundColor: isDark ? const Color(0xFF111827) : Colors.white,
      child: Container(
        width: size.width > 900 ? 850 : size.width * 0.95,
        height: size.height * 0.88,
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header ──────────────────────────────────────────
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppTheme.primaryNavy.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.fact_check_outlined, color: AppTheme.primaryNavy, size: 28),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Upload Attendance Monitoring Report',
                        style: GoogleFonts.outfit(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: isDark ? Colors.white : AppTheme.primaryNavy,
                        ),
                      ),
                      Text(
                        'Supports official .csv, .xlsx, .xls matrix attendance formats with multi-semester subject mapping',
                        style: GoogleFonts.inter(fontSize: 12, color: Colors.grey.shade600),
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

            // ── Body ────────────────────────────────────────────
            Expanded(
              child: _parsedReport == null
                  ? _buildUploadDropzone(isDark)
                  : _buildPreviewContent(isDark),
            ),

            const Divider(height: 24),

            // ── Footer Actions ──────────────────────────────────
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                if (_parsedReport != null)
                  TextButton.icon(
                    onPressed: _isUploading ? null : _pickAndParseFile,
                    icon: const Icon(Icons.refresh, size: 18),
                    label: Text(_selectedFileName != null ? 'File: $_selectedFileName (Change)' : 'Pick Different File'),
                  )
                else
                  const SizedBox.shrink(),
                Row(
                  children: [
                    OutlinedButton(
                      onPressed: _isUploading ? null : () => Navigator.pop(context),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ),
                      child: const Text('Cancel'),
                    ),
                    if (_parsedReport != null && _authCheck?['isAllowed'] == true) ...[
                      const SizedBox(width: 12),
                      ElevatedButton.icon(
                        onPressed: _isUploading ? null : _commitUpload,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF10B981),
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        ),
                        icon: _isUploading
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                              )
                            : const Icon(Icons.cloud_upload_outlined, size: 20),
                        label: Text(_isUploading ? 'Ingesting...' : 'Confirm & Ingest Attendance'),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // ── Dropzone File Selection View ────────────────────────────────────
  Widget _buildUploadDropzone(bool isDark) {
    return Center(
      child: Container(
        padding: const EdgeInsets.all(32),
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1F2937) : const Color(0xFFF9FAFB),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.grey.withOpacity(0.2), width: 2),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_isParsing) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text('Parsing attendance matrix...', style: GoogleFonts.inter(fontWeight: FontWeight.w600)),
            ] else ...[
              Icon(Icons.table_chart_outlined, size: 64, color: AppTheme.accentGold),
              const SizedBox(height: 16),
              Text(
                'Select Official Attendance Monitoring Report',
                style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                'Upload your class monitoring sheet (.csv, .xlsx, .xls)\nMentor access constraint: Only attendance for your assigned class will be accepted.',
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(fontSize: 13, color: Colors.grey.shade600),
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: _pickAndParseFile,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.primaryNavy,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                icon: const Icon(Icons.file_open_outlined),
                label: const Text('Browse Files (.csv, .xlsx, .xls)'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  // ── Multi-Tab Preview View ──────────────────────────────────────────
  Widget _buildPreviewContent(bool isDark) {
    final report = _parsedReport!;
    final isAllowed = _authCheck?['isAllowed'] == true;
    final authReason = _authCheck?['reason'] ?? '';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Security Gate Banner
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: isAllowed ? const Color(0xFF10B981).withOpacity(0.12) : const Color(0xFFEF4444).withOpacity(0.12),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: isAllowed ? const Color(0xFF10B981).withOpacity(0.4) : const Color(0xFFEF4444).withOpacity(0.4),
            ),
          ),
          child: Row(
            children: [
              Icon(
                isAllowed ? Icons.verified_user_outlined : Icons.gpp_bad_outlined,
                color: isAllowed ? const Color(0xFF10B981) : const Color(0xFFEF4444),
                size: 24,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      isAllowed ? 'Class Security Verification: Authorized' : 'Security Isolation: Access Denied',
                      style: GoogleFonts.inter(
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                        color: isAllowed ? const Color(0xFF047857) : const Color(0xFFB91C1C),
                      ),
                    ),
                    Text(
                      authReason,
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        color: isAllowed ? const Color(0xFF065F46) : const Color(0xFF991B1B),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),

        // Report Metadata Card
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: isDark ? const Color(0xFF1F2937) : const Color(0xFFF3F4F6),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildMetaPill('Class & Sec', '${report.className} (Sec ${report.section})', Icons.school),
              _buildMetaPill('Semester', 'Semester ${report.semester}', Icons.calendar_month),
              _buildMetaPill('Cycle', report.cycleName, Icons.repeat),
              _buildMetaPill('Students', '${report.totalStudents}', Icons.people),
              _buildMetaPill('Critical (<75%)', '${report.criticalCount}', Icons.warning_amber, isAlert: report.criticalCount > 0),
            ],
          ),
        ),
        const SizedBox(height: 12),

        // Tabs
        TabBar(
          controller: _tabController,
          labelColor: AppTheme.primaryNavy,
          unselectedLabelColor: Colors.grey,
          indicatorColor: AppTheme.primaryNavy,
          tabs: [
            Tab(text: 'Students Matrix (${report.totalStudents})'),
            Tab(text: 'Subjects & Faculty (${report.subjects.length})'),
            Tab(text: 'Critical Defaulters (${report.criticalCount})'),
          ],
        ),
        const SizedBox(height: 8),

        // Tab Views
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildStudentsMatrixTab(isDark, report),
              _buildSubjectsTab(isDark, report),
              _buildCriticalDefaultersTab(isDark, report),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildMetaPill(String label, String value, IconData icon, {bool isAlert = false}) {
    return Column(
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: isAlert ? Colors.red : AppTheme.primaryNavy),
            const SizedBox(width: 4),
            Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
          ],
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: GoogleFonts.inter(
            fontSize: 12,
            fontWeight: FontWeight.bold,
            color: isAlert ? Colors.red.shade700 : null,
          ),
        ),
      ],
    );
  }

  Widget _buildStudentsMatrixTab(bool isDark, ParsedAttendanceReport report) {
    final filtered = report.studentRows.where((s) {
      if (_filterCriticalOnly && !s.isCritical) return false;
      if (_searchQuery.isEmpty) return true;
      final q = _searchQuery.toLowerCase();
      return s.rollNo.toLowerCase().contains(q) || s.studentName.toLowerCase().contains(q);
    }).toList();

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    hintText: 'Search by student name or roll number...',
                    prefixIcon: const Icon(Icons.search, size: 18),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                  onChanged: (val) => setState(() => _searchQuery = val),
                ),
              ),
              const SizedBox(width: 12),
              FilterChip(
                label: const Text('Critical Only (<75%)'),
                selected: _filterCriticalOnly,
                onSelected: (val) => setState(() => _filterCriticalOnly = val),
                selectedColor: Colors.red.shade100,
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: filtered.length,
            itemBuilder: (ctx, i) {
              final st = filtered[i];
              return Card(
                margin: const EdgeInsets.only(bottom: 6),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  child: Row(
                    children: [
                      Container(
                        width: 32,
                        height: 32,
                        decoration: BoxDecoration(
                          color: st.isCritical ? Colors.red.shade100 : Colors.green.shade100,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Center(
                          child: Text(
                            '${st.srNo}',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                              color: st.isCritical ? Colors.red.shade800 : Colors.green.shade800,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              st.studentName,
                              style: GoogleFonts.inter(fontWeight: FontWeight.bold, fontSize: 13),
                            ),
                            Text(
                              'Roll: ${st.rollNo} • Defaulter Subjects: ${st.defaulterCount}',
                              style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                            ),
                          ],
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: st.overallPercentage >= 75.0
                              ? const Color(0xFF10B981).withOpacity(0.15)
                              : const Color(0xFFEF4444).withOpacity(0.15),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          '${st.overallPercentage.toStringAsFixed(1)}%',
                          style: GoogleFonts.inter(
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                            color: st.overallPercentage >= 75.0
                                ? const Color(0xFF10B981)
                                : const Color(0xFFEF4444),
                          ),
                        ),
                      ),
                      if (st.isCritical) ...[
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.red,
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: const Text(
                            'CRITICAL',
                            style: TextStyle(color: Colors.white, fontSize: 9, fontWeight: FontWeight.bold),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildSubjectsTab(bool isDark, ParsedAttendanceReport report) {
    return ListView.builder(
      itemCount: report.subjects.length,
      itemBuilder: (ctx, i) {
        final sub = report.subjects[i];
        return Card(
          margin: const EdgeInsets.only(bottom: 8),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          child: ListTile(
            leading: CircleAvatar(
              backgroundColor: AppTheme.primaryNavy.withOpacity(0.1),
              child: Text(
                '${i + 1}',
                style: const TextStyle(color: AppTheme.primaryNavy, fontWeight: FontWeight.bold),
              ),
            ),
            title: Text(
              sub.subjectName,
              style: GoogleFonts.inter(fontWeight: FontWeight.bold, fontSize: 13),
            ),
            subtitle: Text(
              'Code: ${sub.subjectCode} • Category: ${sub.courseType}\nInstructor: ${sub.facultyName.isNotEmpty ? sub.facultyName : "Unassigned"}',
              style: const TextStyle(fontSize: 11),
            ),
          ),
        );
      },
    );
  }

  Widget _buildCriticalDefaultersTab(bool isDark, ParsedAttendanceReport report) {
    final criticalStudents = report.studentRows.where((s) => s.isCritical).toList();

    if (criticalStudents.isEmpty) {
      return const Center(
        child: Text('🎉 No students in critical attendance status (<75%).'),
      );
    }

    return ListView.builder(
      itemCount: criticalStudents.length,
      itemBuilder: (ctx, i) {
        final st = criticalStudents[i];
        return Card(
          margin: const EdgeInsets.only(bottom: 8),
          color: const Color(0xFFFEF2F2),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
            side: const BorderSide(color: Color(0xFFFCA5A5)),
          ),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                const Icon(Icons.warning_amber_rounded, color: Color(0xFFDC2626), size: 24),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        st.studentName,
                        style: GoogleFonts.inter(
                          fontWeight: FontWeight.bold,
                          color: const Color(0xFF991B1B),
                        ),
                      ),
                      Text(
                        'Roll No: ${st.rollNo} • Short Attendance in ${st.defaulterCount} Subjects',
                        style: const TextStyle(fontSize: 11, color: Color(0xFFB91C1C)),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFFDC2626),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    '${st.overallPercentage.toStringAsFixed(1)}%',
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
