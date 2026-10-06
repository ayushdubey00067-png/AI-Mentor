import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/models.dart';
import '../services/excel_student_parser_service.dart';
import '../services/supabase_service.dart';
import '../utils/app_theme.dart';

class UploadBatchStudentsDialog extends StatefulWidget {
  final UserModel currentMentor;
  final VoidCallback onImportSuccess;

  const UploadBatchStudentsDialog({
    super.key,
    required this.currentMentor,
    required this.onImportSuccess,
  });

  static Future<void> show(BuildContext context, {
    required UserModel currentMentor,
    required VoidCallback onImportSuccess,
  }) async {
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => UploadBatchStudentsDialog(
        currentMentor: currentMentor,
        onImportSuccess: onImportSuccess,
      ),
    );
  }

  @override
  State<UploadBatchStudentsDialog> createState() => _UploadBatchStudentsDialogState();
}

class _UploadBatchStudentsDialogState extends State<UploadBatchStudentsDialog> {
  bool _isParsing = false;
  bool _isImporting = false;
  String? _fileName;
  BatchStudentParseResult? _parseResult;
  String? _errorMessage;

  List<UserModel> _mentorsList = [];
  String? _selectedMentorEmail;

  @override
  void initState() {
    super.initState();
    _selectedMentorEmail = widget.currentMentor.email;
    _loadMentors();
  }

  Future<void> _loadMentors() async {
    try {
      final mentors = await SupabaseService.getAllMentors();
      if (mounted) {
        setState(() {
          _mentorsList = mentors;
          if (_mentorsList.any((m) => m.email.toLowerCase() == widget.currentMentor.email.toLowerCase())) {
            _selectedMentorEmail = widget.currentMentor.email;
          } else if (_mentorsList.isNotEmpty) {
            _selectedMentorEmail = _mentorsList.first.email;
          }
        });
      }
    } catch (_) {}
  }

  Future<void> _pickAndParseExcel() async {
    setState(() {
      _isParsing = true;
      _errorMessage = null;
    });

    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['xlsx', 'xls'],
        withData: true,
      );

      if (result == null || result.files.isEmpty) {
        setState(() => _isParsing = false);
        return;
      }

      final file = result.files.first;
      final bytes = file.bytes;

      if (bytes == null) {
        throw Exception('Could not read file data. Please try again.');
      }

      _fileName = file.name;

      final parsed = await ExcelStudentParserService.parseStudentExcel(bytes);
      setState(() {
        _parseResult = parsed;
        _isParsing = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to parse spreadsheet: $e';
        _isParsing = false;
      });
    }
  }

  Future<void> _confirmImport() async {
    if (_parseResult == null || _parseResult!.students.isEmpty) return;

    setState(() => _isImporting = true);

    try {
      final targetMentor = _selectedMentorEmail ?? widget.currentMentor.email;
      await SupabaseService.batchRegisterStudents(
        studentsList: _parseResult!.students,
        mentorEmail: targetMentor,
      );

      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: const Color(0xFF10B981),
            behavior: SnackBarBehavior.floating,
            content: Row(
              children: [
                const Icon(Icons.check_circle_rounded, color: Colors.white),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Successfully imported ${_parseResult!.students.length} students into class directory!',
                    style: GoogleFonts.inter(fontWeight: FontWeight.w600, color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        );
        widget.onImportSuccess();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Import failed: $e';
          _isImporting = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final size = MediaQuery.of(context).size;
    final isDark = theme.brightness == Brightness.dark;

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      backgroundColor: isDark ? const Color(0xFF111827) : Colors.white,
      child: Container(
        width: size.width > 900 ? 860 : size.width * 0.95,
        height: size.height > 800 ? 720 : size.height * 0.9,
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Dialog Header ─────────────────────────────────
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppTheme.accentGold.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.file_upload_outlined, color: AppTheme.accentGold, size: 24),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Upload Batch Students (Excel)',
                        style: GoogleFonts.outfit(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: isDark ? Colors.white : AppTheme.primaryNavy,
                        ),
                      ),
                      Text(
                        'Import class directory from .xlsx spreadsheet (Sheet 1: Student Directory)',
                        style: GoogleFonts.inter(fontSize: 12, color: Colors.grey.shade600),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: _isImporting ? null : () => Navigator.pop(context),
                ),
              ],
            ),
            const Divider(height: 28),

            // ── Main Body ─────────────────────────────────────
            Expanded(
              child: _parseResult == null
                  ? _buildUploadArea(isDark)
                  : _buildPreviewArea(isDark),
            ),

            if (_errorMessage != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.red.shade300),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _errorMessage!,
                        style: GoogleFonts.inter(fontSize: 12, color: Colors.red.shade700),
                      ),
                    ),
                  ],
                ),
              ),
            ],

            const SizedBox(height: 16),
            // ── Dialog Footer Actions ─────────────────────────
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                if (_parseResult != null)
                  TextButton.icon(
                    onPressed: _isImporting ? null : () => setState(() => _parseResult = null),
                    icon: const Icon(Icons.refresh),
                    label: const Text('Pick Another File'),
                  ),
                const Spacer(),
                OutlinedButton(
                  onPressed: _isImporting ? null : () => Navigator.pop(context),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                  child: const Text('Cancel'),
                ),
                const SizedBox(width: 12),
                if (_parseResult != null)
                  ElevatedButton.icon(
                    onPressed: _isImporting ? null : _confirmImport,
                    icon: _isImporting
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.cloud_upload_rounded),
                    label: Text(
                      _isImporting
                          ? 'Importing...'
                          : 'Register ${_parseResult!.students.length} Students',
                      style: GoogleFonts.inter(fontWeight: FontWeight.bold),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.primaryNavy,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 13),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUploadArea(bool isDark) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(28),
            decoration: BoxDecoration(
              color: isDark ? const Color(0xFF1F2937) : Colors.grey.shade100,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.upload_file_rounded, size: 64, color: AppTheme.accentGold),
          ),
          const SizedBox(height: 20),
          Text(
            'Select Class Student Directory (.xlsx)',
            style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            'Supports official MRU format (Roll No, Names, Emails, Mobile Passwords, Guardian Contacts)',
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(fontSize: 13, color: Colors.grey.shade600),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: _isParsing ? null : _pickAndParseExcel,
            icon: _isParsing
                ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Icon(Icons.folder_open_rounded),
            label: Text(_isParsing ? 'Reading Excel...' : 'Browse Computer (.xlsx)'),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryNavy,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPreviewArea(bool isDark) {
    final res = _parseResult!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── Class Meta Cards ─────────────────────────────────
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: isDark ? const Color(0xFF1F2937) : const Color(0xFFF3F4F6),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.grey.withOpacity(0.2)),
          ),
          child: Wrap(
            spacing: 12,
            runSpacing: 8,
            children: [
              _buildMetaBadge(Icons.school, '${res.detectedProgram} (${res.detectedBranch})'),
              _buildMetaBadge(Icons.class_, 'Semester ${res.detectedSemester}'),
              _buildMetaBadge(Icons.meeting_room, 'Section ${res.detectedSection}'),
              _buildMetaBadge(Icons.people, '${res.students.length} Students Parsed', color: Colors.green),
              _buildMetaBadge(Icons.insert_drive_file, _fileName ?? 'spreadsheet.xlsx'),
            ],
          ),
        ),
        const SizedBox(height: 14),

        // ── Assign Mentor Dropdown ────────────────────────────
        Row(
          children: [
            Text(
              'Assign Mentor:',
              style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w600),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.withOpacity(0.3)),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    isExpanded: true,
                    value: _selectedMentorEmail,
                    items: _mentorsList.map((m) {
                      return DropdownMenuItem<String>(
                        value: m.email,
                        child: Text(
                          '${m.name} (${m.email})',
                          style: GoogleFonts.inter(fontSize: 13),
                          overflow: TextOverflow.ellipsis,
                        ),
                      );
                    }).toList(),
                    onChanged: (val) {
                      if (val != null) setState(() => _selectedMentorEmail = val);
                    },
                  ),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),

        // ── Table Header ──────────────────────────────────────
        Text(
          'Student Directory Preview (${res.students.length} Records)',
          style: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),

        // ── Data Table Preview ────────────────────────────────
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey.withOpacity(0.2)),
              borderRadius: BorderRadius.circular(10),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: SingleChildScrollView(
                scrollDirection: Axis.vertical,
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: DataTable(
                    headingRowColor: WidgetStateProperty.all(
                      isDark ? const Color(0xFF374151) : Colors.grey.shade200,
                    ),
                    columnSpacing: 18,
                    horizontalMargin: 12,
                    dataRowMinHeight: 40,
                    dataRowMaxHeight: 45,
                    columns: const [
                      DataColumn(label: Text('#')),
                      DataColumn(label: Text('Roll No')),
                      DataColumn(label: Text('Full Name')),
                      DataColumn(label: Text('Official Email (Login)')),
                      DataColumn(label: Text('Mobile (Password)')),
                      DataColumn(label: Text('Gender')),
                      DataColumn(label: Text('Father Name')),
                      DataColumn(label: Text('Domicile State')),
                      DataColumn(label: Text('Pincode')),
                      DataColumn(label: Text('App No')),
                    ],
                    rows: res.students.asMap().entries.map((entry) {
                      final i = entry.key + 1;
                      final s = entry.value;
                      return DataRow(
                        cells: [
                          DataCell(Text('$i')),
                          DataCell(Text(s['roll_number'] ?? '', style: const TextStyle(fontWeight: FontWeight.bold))),
                          DataCell(Text(s['full_name'] ?? '')),
                          DataCell(Text(s['official_email'] ?? '')),
                          DataCell(Text(s['mobile_no'] ?? '', style: const TextStyle(color: Colors.blueGrey))),
                          DataCell(Text(s['gender'] ?? 'NA')),
                          DataCell(Text(s['father_name'] ?? 'NA')),
                          DataCell(Text(s['domicile_state'] ?? 'NA')),
                          DataCell(Text(s['pincode'] ?? 'NA')),
                          DataCell(Text(s['application_no'] ?? 'NA')),
                        ],
                      );
                    }).toList(),
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMetaBadge(IconData icon, String text, {Color? color}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: (color ?? AppTheme.primaryNavy).withOpacity(0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: (color ?? AppTheme.primaryNavy).withOpacity(0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color ?? AppTheme.primaryNavy),
          const SizedBox(width: 6),
          Text(
            text,
            style: GoogleFonts.inter(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: color ?? AppTheme.primaryNavy,
            ),
          ),
        ],
      ),
    );
  }
}
