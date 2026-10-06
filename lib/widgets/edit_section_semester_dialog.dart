// lib/widgets/edit_section_semester_dialog.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/models.dart';
import '../services/supabase_service.dart';
import '../utils/app_theme.dart';

class EditSectionSemesterDialog extends StatefulWidget {
  final UserModel student;
  final VoidCallback onSaved;

  const EditSectionSemesterDialog({
    super.key,
    required this.student,
    required this.onSaved,
  });

  static Future<void> show(BuildContext context, {
    required UserModel student,
    required VoidCallback onSaved,
  }) async {
    await showDialog(
      context: context,
      builder: (_) => EditSectionSemesterDialog(
        student: student,
        onSaved: onSaved,
      ),
    );
  }

  @override
  State<EditSectionSemesterDialog> createState() => _EditSectionSemesterDialogState();
}

class _EditSectionSemesterDialogState extends State<EditSectionSemesterDialog> {
  late TextEditingController _sectionCtrl;
  late String _selectedSemester;
  bool _isSaving = false;
  String? _errorMessage;

  static const List<String> _semesters = ['1', '2', '3', '4', '5', '6', '7', '8'];

  @override
  void initState() {
    super.initState();
    _sectionCtrl = TextEditingController(text: widget.student.section ?? 'A');
    final curSem = widget.student.semester ?? '4';
    _selectedSemester = _semesters.contains(curSem) ? curSem : '4';
  }

  @override
  void dispose() {
    _sectionCtrl.dispose();
    super.dispose();
  }

  Future<void> _saveChanges() async {
    final roll = widget.student.rollNumber ?? '';
    if (roll.isEmpty) return;

    setState(() {
      _isSaving = true;
      _errorMessage = null;
    });

    try {
      await SupabaseService.updateStudentSectionAndSemester(
        rollNumber: roll,
        newSection: _sectionCtrl.text.trim(),
        newSemester: _selectedSemester,
      );

      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: const Color(0xFF10B981),
            behavior: SnackBarBehavior.floating,
            content: Text(
              'Updated ${widget.student.name} to Semester $_selectedSemester (Section ${_sectionCtrl.text.trim()})',
              style: GoogleFonts.inter(color: Colors.white, fontWeight: FontWeight.w600),
            ),
          ),
        );
        widget.onSaved();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to update: $e';
          _isSaving = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      backgroundColor: isDark ? const Color(0xFF111827) : Colors.white,
      child: Container(
        width: 440,
        padding: const EdgeInsets.all(22),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: AppTheme.accentGold.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(Icons.edit_calendar_rounded, color: AppTheme.accentGold, size: 22),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Edit Section & Semester',
                        style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      Text(
                        widget.student.name,
                        style: GoogleFonts.inter(fontSize: 12, color: Colors.grey.shade600),
                        overflow: TextOverflow.ellipsis,
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

            // Roll Number Info
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: isDark ? const Color(0xFF1F2937) : Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  const Icon(Icons.badge_outlined, size: 16, color: Colors.blueGrey),
                  const SizedBox(width: 8),
                  Text(
                    'Roll No: ${widget.student.rollNumber ?? 'N/A'}',
                    style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Semester Selector
            Text('Academic Semester:', style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey.withOpacity(0.3)),
                borderRadius: BorderRadius.circular(10),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  isExpanded: true,
                  value: _selectedSemester,
                  items: _semesters.map((s) {
                    return DropdownMenuItem<String>(
                      value: s,
                      child: Text('Semester $s', style: GoogleFonts.inter(fontSize: 14)),
                    );
                  }).toList(),
                  onChanged: (val) {
                    if (val != null) setState(() => _selectedSemester = val);
                  },
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Section Text Field
            Text('Section:', style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            TextField(
              controller: _sectionCtrl,
              decoration: InputDecoration(
                hintText: 'e.g. A, B, CSE 4A, CSE 5A',
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              ),
            ),

            if (_errorMessage != null) ...[
              const SizedBox(height: 12),
              Text(_errorMessage!, style: const TextStyle(color: Colors.red, fontSize: 12)),
            ],

            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: _isSaving ? null : () => Navigator.pop(context),
                  child: const Text('Cancel'),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: _isSaving ? null : _saveChanges,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryNavy,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 11),
                  ),
                  child: _isSaving
                      ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                      : const Text('Save Changes'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
