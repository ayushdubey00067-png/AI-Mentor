// lib/widgets/edit_mentor_dialog.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/models.dart';
import '../services/supabase_service.dart';
import '../utils/app_theme.dart';

class EditMentorDialog extends StatefulWidget {
  final UserModel mentor;
  final int assignedStudentCount;
  final VoidCallback onSaved;

  const EditMentorDialog({
    super.key,
    required this.mentor,
    required this.assignedStudentCount,
    required this.onSaved,
  });

  static Future<void> show(
    BuildContext context, {
    required UserModel mentor,
    required int assignedStudentCount,
    required VoidCallback onSaved,
  }) async {
    await showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => EditMentorDialog(
        mentor: mentor,
        assignedStudentCount: assignedStudentCount,
        onSaved: onSaved,
      ),
    );
  }

  @override
  State<EditMentorDialog> createState() => _EditMentorDialogState();
}

class _EditMentorDialogState extends State<EditMentorDialog> {
  late TextEditingController _nameCtrl;
  late TextEditingController _emailCtrl;
  late TextEditingController _passCtrl;
  late TextEditingController _classCtrl;
  late TextEditingController _deptCtrl;
  late TextEditingController _desigCtrl;
  late TextEditingController _phoneCtrl;
  late TextEditingController _officeCtrl;
  late TextEditingController _hoursCtrl;

  bool _obscurePass = true;
  bool _isSaving = false;
  bool _isDeleting = false;
  String? _errorMessage;

  static const List<String> _quickClasses = [
    'CSE 4A', 'CSE 4B', 'CSE 5A', 'CSE 5B', 'CSE 6A', 'CSE 6B', 'CSE 7A', 'CSE 8A',
  ];

  static const List<String> _quickDesignations = [
    'Assistant Professor', 'Associate Professor', 'Professor', 'Head of Department (HOD)', 'Dean',
  ];

  @override
  void initState() {
    super.initState();
    _nameCtrl = TextEditingController(text: widget.mentor.name);
    _emailCtrl = TextEditingController(text: widget.mentor.email);
    _passCtrl = TextEditingController(text: widget.mentor.passwordHash ?? widget.mentor.phone ?? 'mentor123');
    _classCtrl = TextEditingController(text: widget.mentor.assignedClass ?? 'CSE 5A');
    _deptCtrl = TextEditingController(text: widget.mentor.department ?? 'Dept. of Computer Science & Technology');
    _desigCtrl = TextEditingController(text: widget.mentor.designation ?? 'Assistant Professor');
    _phoneCtrl = TextEditingController(text: widget.mentor.phone ?? '');
    _officeCtrl = TextEditingController(text: widget.mentor.officeLocation ?? 'Room 304, Block B');
    _hoursCtrl = TextEditingController(text: widget.mentor.officeHours ?? 'Mon - Fri: 2:00 PM - 4:00 PM');
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _classCtrl.dispose();
    _deptCtrl.dispose();
    _desigCtrl.dispose();
    _phoneCtrl.dispose();
    _officeCtrl.dispose();
    _hoursCtrl.dispose();
    super.dispose();
  }

  Future<void> _saveChanges() async {
    final name = _nameCtrl.text.trim();
    final email = _emailCtrl.text.trim().toLowerCase();
    final pass = _passCtrl.text.trim();
    final assignedClass = _classCtrl.text.trim();

    if (name.isEmpty || email.isEmpty || pass.isEmpty) {
      setState(() => _errorMessage = 'Full Name, Email, and Password are required.');
      return;
    }

    setState(() {
      _isSaving = true;
      _errorMessage = null;
    });

    try {
      final updateData = <String, dynamic>{
        'name': name,
        'email': email,
        'password_hash': pass,
        'assigned_class': assignedClass,
        'department': _deptCtrl.text.trim(),
        'designation': _desigCtrl.text.trim(),
        'phone': _phoneCtrl.text.trim(),
        'office_location': _officeCtrl.text.trim(),
        'office_hours': _hoursCtrl.text.trim(),
      };

      await SupabaseService.updateMentorProfile(
        id: widget.mentor.id,
        data: updateData,
        oldEmail: widget.mentor.email,
        newEmail: email,
      );

      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: const Color(0xFF10B981),
            behavior: SnackBarBehavior.floating,
            content: Text(
              '✅ Successfully updated mentor details for $name ($assignedClass)',
              style: GoogleFonts.inter(color: Colors.white, fontWeight: FontWeight.bold),
            ),
          ),
        );
        widget.onSaved();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to save changes: $e';
          _isSaving = false;
        });
      }
    }
  }

  Future<void> _confirmDeleteMentor() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Row(
          children: [
            const Icon(Icons.warning_amber_rounded, color: Colors.red, size: 28),
            const SizedBox(width: 10),
            Text('Delete Mentor Account?', style: GoogleFonts.outfit(fontWeight: FontWeight.bold)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Are you sure you want to permanently delete faculty account for "${widget.mentor.name}" (${widget.mentor.email})?',
              style: GoogleFonts.inter(fontSize: 13),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.amber.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.amber.shade200),
              ),
              child: Row(
                children: [
                  const Icon(Icons.info_outline, color: Colors.amber, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'All ${widget.assignedStudentCount} assigned student records will be safely retained and unassigned for reassignment.',
                      style: GoogleFonts.inter(fontSize: 12, color: Colors.amber.shade900),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red.shade700,
              foregroundColor: Colors.white,
            ),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Confirm Delete'),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    setState(() {
      _isDeleting = true;
      _errorMessage = null;
    });

    try {
      await SupabaseService.deleteMentor(
        widget.mentor.id,
        email: widget.mentor.email,
      );

      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: Colors.red.shade700,
            behavior: SnackBarBehavior.floating,
            content: Text(
              '🗑️ Mentor "${widget.mentor.name}" has been removed.',
              style: GoogleFonts.inter(color: Colors.white, fontWeight: FontWeight.bold),
            ),
          ),
        );
        widget.onSaved();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to delete mentor: $e';
          _isDeleting = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      child: Container(
        width: 620,
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.9,
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── HEADER ───────────────────────────────────────
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AppTheme.primaryNavy.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.manage_accounts_rounded, color: AppTheme.primaryNavy, size: 26),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Text(
                            'Edit Mentor Profile & Metadata',
                            style: GoogleFonts.outfit(fontSize: 19, fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: AppTheme.accentGold.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              'ADMIN OVERRIDE',
                              style: GoogleFonts.inter(fontSize: 10, fontWeight: FontWeight.bold, color: AppTheme.accentGold),
                            ),
                          ),
                        ],
                      ),
                      Text(
                        'Direct full administrative control over credentials, class assignment, and contact metadata.',
                        style: GoogleFonts.inter(fontSize: 12, color: Colors.grey.shade600),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: (_isSaving || _isDeleting) ? null : () => Navigator.pop(context),
                ),
              ],
            ),
            const Divider(height: 24),

            // ── SCROLLABLE FORM BODY ─────────────────────────
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Class & Student Mapping Info Banner
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                      decoration: BoxDecoration(
                        color: const Color(0xFF0F1C3F).withOpacity(0.05),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: AppTheme.primaryNavy.withOpacity(0.15)),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.sync_rounded, color: AppTheme.primaryNavy, size: 20),
                          const SizedBox(width: 10),
                          Expanded(
                            child: RichText(
                              text: TextSpan(
                                style: GoogleFonts.inter(fontSize: 12, color: Colors.black87),
                                children: [
                                  const TextSpan(text: 'Current Class: '),
                                  TextSpan(
                                    text: widget.mentor.assignedClass ?? 'CSE 5A',
                                    style: const TextStyle(fontWeight: FontWeight.bold, color: AppTheme.primaryNavy),
                                  ),
                                  TextSpan(text: ' • Linked to '),
                                  TextSpan(
                                    text: '${widget.assignedStudentCount} students',
                                    style: const TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF10B981)),
                                  ),
                                  const TextSpan(text: '. Updating email or class will sync students automatically.'),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Row 1: Full Name & Official Email
                    Row(
                      children: [
                        Expanded(
                          child: _buildField(
                            label: 'Full Name *',
                            controller: _nameCtrl,
                            hint: 'e.g. Dr. Gunjan',
                            icon: Icons.person_rounded,
                          ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: _buildField(
                            label: 'Faculty Login Email *',
                            controller: _emailCtrl,
                            hint: 'e.g. gunjan@mru.edu.in',
                            icon: Icons.email_outlined,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 14),

                    // Row 2: Password / Reset PIN & Phone Number
                    Row(
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Password / Reset Password *',
                                style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600),
                              ),
                              const SizedBox(height: 5),
                              TextField(
                                controller: _passCtrl,
                                obscureText: _obscurePass,
                                decoration: InputDecoration(
                                  prefixIcon: const Icon(Icons.lock_outline, size: 18),
                                  suffixIcon: IconButton(
                                    icon: Icon(_obscurePass ? Icons.visibility_off : Icons.visibility, size: 18),
                                    onPressed: () => setState(() => _obscurePass = !_obscurePass),
                                  ),
                                  hintText: 'Enter new password',
                                  isDense: true,
                                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: _buildField(
                            label: 'Phone Number',
                            controller: _phoneCtrl,
                            hint: 'e.g. 9876543210',
                            icon: Icons.phone_outlined,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 14),

                    // Assigned Class / Section + Quick chips
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Assigned Class / Section *',
                              style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600),
                            ),
                            Text(
                              'Select chip or type custom',
                              style: GoogleFonts.inter(fontSize: 11, color: Colors.grey.shade500),
                            ),
                          ],
                        ),
                        const SizedBox(height: 5),
                        TextField(
                          controller: _classCtrl,
                          decoration: InputDecoration(
                            prefixIcon: const Icon(Icons.class_outlined, size: 18),
                            hintText: 'e.g. CSE 5A, CSE 4A',
                            isDense: true,
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                          ),
                        ),
                        const SizedBox(height: 6),
                        Wrap(
                          spacing: 6,
                          runSpacing: 4,
                          children: _quickClasses.map((cls) {
                            final isSelected = _classCtrl.text.trim().toUpperCase() == cls;
                            return ChoiceChip(
                              label: Text(cls, style: TextStyle(fontSize: 11, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal)),
                              selected: isSelected,
                              selectedColor: AppTheme.accentGold.withOpacity(0.3),
                              onSelected: (_) {
                                setState(() => _classCtrl.text = cls);
                              },
                            );
                          }).toList(),
                        ),
                      ],
                    ),
                    const SizedBox(height: 14),

                    // Row 3: Designation & Department
                    Row(
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Designation',
                                style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600),
                              ),
                              const SizedBox(height: 5),
                              TextField(
                                controller: _desigCtrl,
                                decoration: InputDecoration(
                                  prefixIcon: const Icon(Icons.badge_outlined, size: 18),
                                  hintText: 'e.g. Assistant Professor',
                                  isDense: true,
                                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: _buildField(
                            label: 'Department',
                            controller: _deptCtrl,
                            hint: 'e.g. Dept. of Computer Science & Technology',
                            icon: Icons.account_balance_outlined,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Wrap(
                      spacing: 6,
                      runSpacing: 4,
                      children: _quickDesignations.map((desig) {
                        return ActionChip(
                          label: Text(desig, style: const TextStyle(fontSize: 10)),
                          onPressed: () => setState(() => _desigCtrl.text = desig),
                        );
                      }).toList(),
                    ),
                    const SizedBox(height: 14),

                    // Row 4: Office Location & Office Hours
                    Row(
                      children: [
                        Expanded(
                          child: _buildField(
                            label: 'Office Location',
                            controller: _officeCtrl,
                            hint: 'e.g. Room 304, Block B',
                            icon: Icons.location_on_outlined,
                          ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: _buildField(
                            label: 'Office Consultation Hours',
                            controller: _hoursCtrl,
                            hint: 'e.g. Mon - Fri: 2:00 PM - 4:00 PM',
                            icon: Icons.access_time_outlined,
                          ),
                        ),
                      ],
                    ),

                    if (_errorMessage != null) ...[
                      const SizedBox(height: 14),
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: Colors.red.shade50,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.red.shade200),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.error_outline, color: Colors.red, size: 18),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _errorMessage!,
                                style: GoogleFonts.inter(color: Colors.red.shade900, fontSize: 12),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],

                    const SizedBox(height: 20),
                    const Divider(),
                    const SizedBox(height: 8),

                    // Danger Zone: Delete Mentor
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Danger Zone',
                              style: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.bold, color: Colors.red.shade700),
                            ),
                            Text(
                              'Remove mentor account safely without losing student records',
                              style: GoogleFonts.inter(fontSize: 11, color: Colors.grey.shade600),
                            ),
                          ],
                        ),
                        OutlinedButton.icon(
                          onPressed: (_isSaving || _isDeleting) ? null : _confirmDeleteMentor,
                          icon: _isDeleting
                              ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.red))
                              : const Icon(Icons.delete_forever_rounded, size: 16, color: Colors.red),
                          label: Text(
                            'Delete Mentor',
                            style: GoogleFonts.inter(color: Colors.red.shade700, fontWeight: FontWeight.bold, fontSize: 12),
                          ),
                          style: OutlinedButton.styleFrom(
                            side: BorderSide(color: Colors.red.shade300),
                            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),
            const Divider(height: 1),
            const SizedBox(height: 16),

            // ── ACTION BUTTONS ────────────────────────────────
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: (_isSaving || _isDeleting) ? null : () => Navigator.pop(context),
                  child: Text('Cancel', style: GoogleFonts.inter(color: Colors.grey.shade700)),
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  onPressed: (_isSaving || _isDeleting) ? null : _saveChanges,
                  icon: _isSaving
                      ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                      : const Icon(Icons.save_rounded, size: 18),
                  label: Text(
                    _isSaving ? 'Saving Changes...' : 'Save All Changes',
                    style: GoogleFonts.inter(fontWeight: FontWeight.bold),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryNavy,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
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

  Widget _buildField({
    required String label,
    required TextEditingController controller,
    String? hint,
    IconData? icon,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 5),
        TextField(
          controller: controller,
          decoration: InputDecoration(
            prefixIcon: icon != null ? Icon(icon, size: 18) : null,
            hintText: hint,
            isDense: true,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
            contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
          ),
        ),
      ],
    );
  }
}
