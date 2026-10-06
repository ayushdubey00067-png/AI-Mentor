// lib/screens/admin/admin_dashboard_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/auth_provider.dart';
import '../../services/supabase_service.dart';
import '../../utils/app_theme.dart';
import '../../widgets/edit_mentor_dialog.dart';
import '../../widgets/edit_section_semester_dialog.dart';
import '../../widgets/student_detail_dialog.dart';
import '../../widgets/upload_attendance_dialog.dart';
import '../auth_screen.dart';

class AdminDashboardScreen extends StatefulWidget {
  const AdminDashboardScreen({super.key});

  @override
  State<AdminDashboardScreen> createState() => _AdminDashboardScreenState();
}

class _AdminDashboardScreenState extends State<AdminDashboardScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tab;
  List<UserModel> _mentors = [];
  List<UserModel> _students = [];
  bool _isLoading = true;
  String _studentSearch = '';

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
    _loadData();
  }

  @override
  void dispose() {
    _tab.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    try {
      final mentors = await SupabaseService.getAllMentors();
      final students = await SupabaseService.getAllStudents();
      if (mounted) {
        setState(() {
          _mentors = mentors;
          _students = students;
          _isLoading = false;
        });
      }
    } catch (_) {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showAddMentorDialog() {
    final nameCtrl = TextEditingController();
    final emailCtrl = TextEditingController();
    final passCtrl = TextEditingController();
    final deptCtrl = TextEditingController(text: 'Dept. of Computer Science & Technology');
    final desigCtrl = TextEditingController(text: 'Assistant Professor');
    final phoneCtrl = TextEditingController();
    final classCtrl = TextEditingController(text: 'CSE 4A');
    final officeCtrl = TextEditingController(text: 'Room 304, Block B');

    bool isSubmitting = false;
    String? errorText;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) => Dialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Container(
            width: 500,
            padding: const EdgeInsets.all(24),
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: AppTheme.primaryNavy.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: const Icon(Icons.person_add_alt_1_rounded, color: AppTheme.primaryNavy),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Provision New Mentor',
                              style: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                            Text(
                              'Register faculty account & map to academic class/section',
                              style: GoogleFonts.inter(fontSize: 12, color: Colors.grey.shade600),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: isSubmitting ? null : () => Navigator.pop(ctx),
                      ),
                    ],
                  ),
                  const Divider(height: 24),

                  _buildField('Full Name', nameCtrl, hint: 'e.g. Dr. Anupriya Sharma'),
                  const SizedBox(height: 12),
                  _buildField('Faculty Email (Login)', emailCtrl, hint: 'e.g. anupriya.cse@mru.ac.in'),
                  const SizedBox(height: 12),
                  _buildField('Password', passCtrl, hint: 'e.g. mentor123 / phone number', isPassword: true),
                  const SizedBox(height: 12),
                  _buildField('Assigned Class / Section', classCtrl, hint: 'e.g. CSE 4A, CSE 5A'),
                  const SizedBox(height: 12),
                  _buildField('Department', deptCtrl, hint: 'e.g. Dept. of Computer Science & Technology'),
                  const SizedBox(height: 12),
                  _buildField('Designation', desigCtrl, hint: 'e.g. Assistant Professor'),
                  const SizedBox(height: 12),
                  _buildField('Phone Number', phoneCtrl, hint: 'e.g. 9876543210'),
                  const SizedBox(height: 12),
                  _buildField('Office Location', officeCtrl, hint: 'e.g. Room 304, Block B'),

                  if (errorText != null) ...[
                    const SizedBox(height: 12),
                    Text(errorText!, style: const TextStyle(color: Colors.red, fontSize: 12)),
                  ],

                  const SizedBox(height: 20),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: isSubmitting ? null : () => Navigator.pop(ctx),
                        child: const Text('Cancel'),
                      ),
                      const SizedBox(width: 8),
                      ElevatedButton(
                        onPressed: isSubmitting
                            ? null
                            : () async {
                                if (nameCtrl.text.trim().isEmpty ||
                                    emailCtrl.text.trim().isEmpty ||
                                    passCtrl.text.trim().isEmpty) {
                                  setDialogState(() => errorText = 'Name, Email, and Password are required.');
                                  return;
                                }

                                setDialogState(() {
                                  isSubmitting = true;
                                  errorText = null;
                                });

                                try {
                                  await SupabaseService.createMentor(
                                    name: nameCtrl.text.trim(),
                                    email: emailCtrl.text.trim(),
                                    password: passCtrl.text.trim(),
                                    department: deptCtrl.text.trim(),
                                    designation: desigCtrl.text.trim(),
                                    phone: phoneCtrl.text.trim(),
                                    assignedClass: classCtrl.text.trim(),
                                    officeLocation: officeCtrl.text.trim(),
                                  );

                                  if (mounted) {
                                    Navigator.pop(ctx);
                                    _loadData();
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                        backgroundColor: const Color(0xFF10B981),
                                        content: Text('Successfully registered mentor ${nameCtrl.text}!'),
                                      ),
                                    );
                                  }
                                } catch (e) {
                                  setDialogState(() {
                                    isSubmitting = false;
                                    errorText = 'Failed to create mentor: $e';
                                  });
                                }
                              },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.primaryNavy,
                          foregroundColor: Colors.white,
                        ),
                        child: isSubmitting
                            ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                            : const Text('Create Mentor Account'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildField(String label, TextEditingController ctrl, {String? hint, bool isPassword = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600)),
        const SizedBox(height: 4),
        TextField(
          controller: ctrl,
          obscureText: isPassword,
          decoration: InputDecoration(
            hintText: hint,
            isDense: true,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
            contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final adminUser = auth.currentUser;
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppTheme.primaryNavy,
        foregroundColor: Colors.white,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: AppTheme.accentGold,
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.admin_panel_settings_rounded, color: AppTheme.primaryNavy, size: 20),
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Acadly Admin Portal',
                  style: GoogleFonts.outfit(fontSize: 17, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                Text(
                  adminUser?.email ?? 'admin@mru.ac.in',
                  style: GoogleFonts.inter(fontSize: 11, color: Colors.white70),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded, color: Colors.white),
            tooltip: 'Refresh Data',
            onPressed: _loadData,
          ),
          const SizedBox(width: 8),
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.white,
                side: const BorderSide(color: Color(0xFFF87171), width: 1.5),
                backgroundColor: const Color(0xFFEF4444).withOpacity(0.15),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
              icon: const Icon(Icons.logout_rounded, size: 16, color: Color(0xFFFCA5A5)),
              label: Text(
                'Log Out',
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                  color: Colors.white,
                ),
              ),
              onPressed: () async {
                await auth.logout();
                if (mounted) {
                  Navigator.of(context).pushReplacement(
                    MaterialPageRoute(builder: (_) => const AuthScreen()),
                  );
                }
              },
            ),
          ),
        ],

        bottom: TabBar(
          controller: _tab,
          indicatorColor: AppTheme.accentGold,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.white60,
          tabs: [
            Tab(
              icon: const Icon(Icons.supervisor_account),
              text: 'Faculty Mentors (${_mentors.length})',
            ),
            Tab(
              icon: const Icon(Icons.school),
              text: 'All Students (${_students.length})',
            ),
          ],
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : TabBarView(
              controller: _tab,
              children: [
                _buildMentorsTab(isDark),
                _buildStudentsTab(isDark),
              ],
            ),
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: AppTheme.accentGold,
        foregroundColor: AppTheme.primaryNavy,
        onPressed: _showAddMentorDialog,
        icon: const Icon(Icons.person_add_alt_1),
        label: Text(
          'Add New Mentor',
          style: GoogleFonts.inter(fontWeight: FontWeight.bold),
        ),
      ),
    );
  }

  Widget _buildMentorsTab(bool isDark) {
    if (_mentors.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.people_outline, size: 64, color: Colors.grey),
            const SizedBox(height: 12),
            Text('No mentors registered yet', style: GoogleFonts.outfit(fontSize: 16)),
            const SizedBox(height: 6),
            const Text('Click "+ Add New Mentor" to create a faculty account.'),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _mentors.length,
      itemBuilder: (ctx, i) {
        final m = _mentors[i];
        final assignedCount = _students.where((s) => s.mentorEmail?.toLowerCase() == m.email.toLowerCase()).length;

        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          elevation: 1.5,
          child: InkWell(
            borderRadius: BorderRadius.circular(14),
            onTap: () {
              EditMentorDialog.show(
                context,
                mentor: m,
                assignedStudentCount: assignedCount,
                onSaved: _loadData,
              );
            },
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  CircleAvatar(
                    radius: 26,
                    backgroundColor: AppTheme.primaryNavy,
                    child: Text(
                      m.name.isNotEmpty ? m.name[0].toUpperCase() : 'M',
                      style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18),
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              m.name,
                              style: GoogleFonts.outfit(fontSize: 17, fontWeight: FontWeight.bold),
                            ),
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 3),
                              decoration: BoxDecoration(
                                color: AppTheme.accentGold.withOpacity(0.15),
                                borderRadius: BorderRadius.circular(6),
                                border: Border.all(color: AppTheme.accentGold.withOpacity(0.4)),
                              ),
                              child: Text(
                                m.assignedClass ?? 'CSE 5A',
                                style: GoogleFonts.inter(
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                  color: AppTheme.accentGold,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '${m.designation ?? 'Faculty'} • ${m.department ?? 'Computer Science & Technology'}',
                          style: GoogleFonts.inter(fontSize: 13, color: Colors.grey.shade600),
                        ),
                        const SizedBox(height: 8),
                        Wrap(
                          spacing: 14,
                          runSpacing: 4,
                          children: [
                            _buildIconText(Icons.email_outlined, m.email),
                            if (m.phone != null && m.phone!.isNotEmpty) _buildIconText(Icons.phone_outlined, m.phone!),
                            if (m.officeLocation != null && m.officeLocation!.isNotEmpty) _buildIconText(Icons.location_on_outlined, m.officeLocation!),
                            if (m.officeHours != null && m.officeHours!.isNotEmpty) _buildIconText(Icons.access_time_outlined, m.officeHours!),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                        decoration: BoxDecoration(
                          color: Colors.blue.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '$assignedCount Students',
                          style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.blue.shade700),
                        ),
                      ),
                      const SizedBox(height: 8),
                      IconButton.filledTonal(
                        icon: const Icon(Icons.edit_outlined, size: 18),
                        tooltip: 'Edit Mentor Profile & Metadata',
                        style: IconButton.styleFrom(
                          backgroundColor: AppTheme.accentGold.withOpacity(0.15),
                          foregroundColor: AppTheme.primaryNavy,
                        ),
                        onPressed: () {
                          EditMentorDialog.show(
                            context,
                            mentor: m,
                            assignedStudentCount: assignedCount,
                            onSaved: _loadData,
                          );
                        },
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildStudentsTab(bool isDark) {
    final filtered = _students.where((s) {
      if (_studentSearch.isEmpty) return true;
      final q = _studentSearch.toLowerCase();
      return (s.rollNumber ?? '').toLowerCase().contains(q) ||
          s.name.toLowerCase().contains(q) ||
          s.email.toLowerCase().contains(q) ||
          (s.section ?? '').toLowerCase().contains(q);
    }).toList();

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    hintText: 'Search by student name, roll number, section, or email...',
                    prefixIcon: const Icon(Icons.search),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                  ),
                  onChanged: (val) => setState(() => _studentSearch = val),
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton.icon(
                onPressed: () {
                  final user = context.read<AuthProvider>().currentUser;
                  if (user != null) {
                    UploadAttendanceDialog.show(
                      context,
                      currentUser: user,
                      onUploadSuccess: _loadData,
                    );
                  }
                },
                icon: const Icon(Icons.fact_check_outlined, size: 18),
                label: const Text('Upload Attendance Report'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF10B981),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: filtered.isEmpty
              ? const Center(child: Text('No matching students found.'))
              : ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemCount: filtered.length,
                  itemBuilder: (ctx, i) {
                    final s = filtered[i];
                      return Card(
                        margin: const EdgeInsets.only(bottom: 8),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        child: ListTile(
                          onTap: () {
                            StudentDetailDialog.show(
                              context,
                              student: s,
                              onDataChanged: _loadData,
                            );
                          },
                          leading: CircleAvatar(
                            backgroundColor: Colors.grey.shade200,
                            child: Text(
                              s.name.isNotEmpty ? s.name[0].toUpperCase() : 'S',
                              style: const TextStyle(fontWeight: FontWeight.bold, color: AppTheme.primaryNavy),
                            ),
                          ),
                          title: Row(
                            children: [
                              Text(s.name, style: GoogleFonts.inter(fontWeight: FontWeight.bold, fontSize: 14)),
                              const SizedBox(width: 8),
                              Text('(${s.rollNumber ?? 'N/A'})', style: const TextStyle(fontSize: 12, color: Colors.grey)),
                            ],
                          ),
                          subtitle: Text(
                            '${s.email} • Mobile (Pass): ${s.phone ?? s.mobileNo ?? 'N/A'}\nSection: ${s.section ?? 'A'} | Sem: ${s.semester ?? '4'} | Mentor: ${s.mentorEmail ?? 'None'}',
                            style: const TextStyle(fontSize: 12),
                          ),
                          trailing: IconButton(
                            icon: const Icon(Icons.edit_outlined, color: AppTheme.accentGold),
                            tooltip: 'Edit Section & Semester',
                            onPressed: () {
                              EditSectionSemesterDialog.show(
                                context,
                                student: s,
                                onSaved: _loadData,
                              );
                            },
                          ),
                        ),
                      );
                  },
                ),
        ),
      ],
    );
  }

  Widget _buildIconText(IconData icon, String text) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 13, color: Colors.grey.shade600),
        const SizedBox(width: 4),
        Text(text, style: GoogleFonts.inter(fontSize: 12, color: Colors.grey.shade700)),
      ],
    );
  }
}
