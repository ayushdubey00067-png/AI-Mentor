// lib/screens/student/student_documents_screen.dart
import 'dart:convert';
import 'dart:async';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../../models/models.dart';
import '../../services/ai_service.dart';
import '../../services/auth_provider.dart';
import '../../services/supabase_service.dart';
import '../../utils/app_theme.dart';

class StudentDocumentsScreen extends StatefulWidget {
  const StudentDocumentsScreen({super.key});
  @override
  State<StudentDocumentsScreen> createState() =>
      _StudentDocumentsScreenState();
}

class _StudentDocumentsScreenState extends State<StudentDocumentsScreen> {
  List<StudentDocument> _docs      = [];
  bool   _loading    = false;
  bool   _uploading  = false;
  String _uploadStatus = '';
  int    _uploadStep   = 0; // 0=idle 1=saving 2=extracting 3=chunking 4=done

  static const List<Map<String, dynamic>> _docTypes = [
    {'value': 'timetable',         'label': 'Timetable',          'emoji': '📅', 'color': Color(0xFF3B82F6)},
    {'value': 'academic_calendar', 'label': 'Academic Calendar',  'emoji': '🗓️', 'color': Color(0xFF8B5CF6)},
    {'value': 'syllabus',          'label': 'Syllabus',           'emoji': '📚', 'color': Color(0xFF10B981)},
    {'value': 'marksheet',         'label': 'Marksheet / Marks',  'emoji': '📊', 'color': Color(0xFFF59E0B)},
    {'value': 'attendance',        'label': 'Attendance Sheet',   'emoji': '✅', 'color': Color(0xFF06B6D4)},
    {'value': 'assignment',        'label': 'Assignment',         'emoji': '📝', 'color': Color(0xFFEF4444)},
    {'value': 'other',             'label': 'Policy / Other',     'emoji': '🏛️', 'color': Color(0xFF6B7280)},
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadDocs());
  }

  Future<void> _loadDocs() async {
    final auth = context.read<AuthProvider>();
    setState(() => _loading = true);
    final docs =
        await SupabaseService.getStudentDocuments(auth.currentUser!.id);
    if (mounted) setState(() { _docs = docs; _loading = false; });
  }

  // ══════════════════════════════════════════════════════════
  // UPLOAD DIALOG
  // ══════════════════════════════════════════════════════════
  Future<void> _showUploadDialog() async {
    String selectedType = 'timetable';
    final titleCtrl = TextEditingController();

    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setS) => Container(
          padding: EdgeInsets.only(
              bottom: MediaQuery.of(ctx).viewInsets.bottom),
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              Container(width: 44, height: 4,
                  decoration: BoxDecoration(color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(2))),
              const SizedBox(height: 18),
              Text('Upload Document', style: GoogleFonts.playfairDisplay(
                  fontSize: 20, fontWeight: FontWeight.w700,
                  color: const Color(0xFF111827))),
              const SizedBox(height: 4),
              Text('AI will read & index it for smart answers',
                  style: GoogleFonts.lato(fontSize: 13,
                      color: const Color(0xFF6B7280))),
              const SizedBox(height: 18),

              // Type selector
              Align(alignment: Alignment.centerLeft,
                child: Text('Document Type', style: GoogleFonts.lato(
                    fontSize: 13, fontWeight: FontWeight.w600,
                    color: const Color(0xFF6B7280)))),
              const SizedBox(height: 10),
              SizedBox(height: 94,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: _docTypes.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 8),
                  itemBuilder: (_, i) {
                    final t   = _docTypes[i];
                    final sel = selectedType == t['value'];
                    return GestureDetector(
                      onTap: () =>
                          setS(() => selectedType = t['value'] as String),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 180),
                        width: 82,
                        padding: const EdgeInsets.symmetric(vertical: 8),
                        decoration: BoxDecoration(
                          color: sel
                              ? (t['color'] as Color).withOpacity(0.12)
                              : const Color(0xFFF9FAFB),
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                              color: sel
                                  ? t['color'] as Color
                                  : const Color(0xFFE5E7EB),
                              width: sel ? 2 : 1)),
                        child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(t['emoji'] as String,
                                  style: const TextStyle(fontSize: 22)),
                              const SizedBox(height: 5),
                              Text(t['label'] as String,
                                  style: GoogleFonts.lato(
                                      fontSize: 9,
                                      fontWeight: FontWeight.w600,
                                      color: sel
                                          ? t['color'] as Color
                                          : const Color(0xFF6B7280)),
                                  textAlign: TextAlign.center,
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis),
                            ]),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 16),

              // Title field
              Align(alignment: Alignment.centerLeft,
                child: Text('Title (Optional)', style: GoogleFonts.lato(
                    fontSize: 13, fontWeight: FontWeight.w600,
                    color: const Color(0xFF6B7280)))),
              const SizedBox(height: 8),
              TextField(
                controller: titleCtrl,
                textCapitalization: TextCapitalization.sentences,
                decoration: InputDecoration(
                  hintText: 'e.g. Semester 5 Timetable',
                  hintStyle: GoogleFonts.lato(
                      color: const Color(0xFF9CA3AF), fontSize: 14),
                  filled: true, fillColor: const Color(0xFFF9FAFB),
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide:
                          const BorderSide(color: Color(0xFFE5E7EB))),
                  enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide:
                          const BorderSide(color: Color(0xFFE5E7EB))),
                  focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: const BorderSide(
                          color: AppTheme.primary, width: 2)),
                  contentPadding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 12),
                ),
              ),
              const SizedBox(height: 20),

              // Source buttons
              Align(alignment: Alignment.centerLeft,
                child: Text('Choose Source', style: GoogleFonts.lato(
                    fontSize: 13, fontWeight: FontWeight.w600,
                    color: const Color(0xFF6B7280)))),
              const SizedBox(height: 12),
              Row(children: [
                Expanded(child: _srcBtn(
                  Icons.camera_alt_rounded, 'Camera',
                  const Color(0xFF3B82F6),
                  onTap: () async {
                    Navigator.pop(ctx);
                    await _pickImage(type: selectedType,
                        title: titleCtrl.text.trim(),
                        source: ImageSource.camera);
                  },
                )),
                const SizedBox(width: 10),
                Expanded(child: _srcBtn(
                  Icons.photo_library_rounded, 'Gallery',
                  const Color(0xFF8B5CF6),
                  onTap: () async {
                    Navigator.pop(ctx);
                    await _pickImage(type: selectedType,
                        title: titleCtrl.text.trim(),
                        source: ImageSource.gallery);
                  },
                )),
                const SizedBox(width: 10),
                Expanded(child: _srcBtn(
                  Icons.upload_file_rounded, 'File',
                  const Color(0xFF10B981),
                  onTap: () async {
                    Navigator.pop(ctx);
                    await _pickFile(type: selectedType,
                        title: titleCtrl.text.trim());
                  },
                )),
              ]),
            ]),
          ),
        ),
      ),
    );
  }

  Widget _srcBtn(IconData icon, String label, Color color,
      {required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
            color: color.withOpacity(0.08),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: color.withOpacity(0.3))),
        child: Column(children: [
          Icon(icon, color: color, size: 26),
          const SizedBox(height: 6),
          Text(label, style: GoogleFonts.lato(
              fontSize: 12, fontWeight: FontWeight.w600, color: color)),
        ]),
      ),
    );
  }

  // ══════════════════════════════════════════════════════════
  // PICK IMAGE
  // ══════════════════════════════════════════════════════════
  Future<void> _pickImage({
    required String type,
    required String title,
    required ImageSource source,
  }) async {
    try {
      final picked = await ImagePicker().pickImage(
          source: source, maxWidth: 1920, maxHeight: 1920, imageQuality: 85);
      if (picked == null) return;
      final bytes    = await picked.readAsBytes();
      final ext      = picked.name.split('.').last.toLowerCase();
      final mimeType = ext == 'png' ? 'image/png' : 'image/jpeg';
      await _processAndUpload(
        type: type, title: title, fileName: picked.name,
        mimeType: mimeType, fileSize: bytes.length,
        base64: base64Encode(bytes),
      );
    } catch (e) {
      _showSnack('Image pick failed: $e', isError: true);
    }
  }

  // ══════════════════════════════════════════════════════════
  // PICK FILE
  // ══════════════════════════════════════════════════════════
  Future<void> _pickFile({
    required String type, required String title,
  }) async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['jpg', 'jpeg', 'png', 'pdf'],
        withData: true,
      );
      if (result == null || result.files.isEmpty) return;
      final file     = result.files.first;
      if (file.bytes == null) return;
      final ext      = (file.extension ?? 'jpg').toLowerCase();
      final mimeType = ext == 'pdf'
          ? 'application/pdf'
          : ext == 'png' ? 'image/png' : 'image/jpeg';
      await _processAndUpload(
        type: type, title: title, fileName: file.name,
        mimeType: mimeType, fileSize: file.size,
        base64: base64Encode(file.bytes!),
      );
    } catch (e) {
      _showSnack('File pick failed: $e', isError: true);
    }
  }

  // ══════════════════════════════════════════════════════════
  // MAIN PIPELINE: UPLOAD → OCR → CHUNK → EMBED
  // ══════════════════════════════════════════════════════════
  Future<void> _processAndUpload({
    required String type,   required String title,
    required String fileName, required String mimeType,
    required int fileSize,  required String base64,
  }) async {
    final auth      = context.read<AuthProvider>();
    final docTitle  = title.isEmpty ? _defaultTitle(type) : title;

    setState(() { _uploading = true; _uploadStep = 1;
        _uploadStatus = 'Saving document...'; });

    try {
      // ── Step 1: Save to Supabase ───────────────────────────
      final doc = await SupabaseService.uploadDocument(
        studentId: auth.currentUser!.id,
        docType:   type, title: docTitle,
        fileName:  fileName, mimeType: mimeType,
        fileSize:  fileSize, contentBase64: base64,
      );
      debugPrint('✅ Document saved: ${doc.id}');

      // ── Step 2: Gemini Vision OCR ──────────────────────────
      setState(() { _uploadStep = 2;
          _uploadStatus = '🤖 AI reading document (OCR)...'; });

      String extractedText = '';
      try {
        extractedText = await AIService.analyzeDocumentImage(
          base64Data: base64, mimeType: mimeType, docType: type,
        );
        debugPrint('✅ OCR: ${extractedText.length} chars extracted');

        if (extractedText.isNotEmpty) {
          await SupabaseService.updateDocumentExtractedText(
              doc.id, extractedText);
        }
      } catch (e) {
        debugPrint('⚠️ OCR failed: $e — proceeding without text');
      }

      // ── Step 3 & 4: Batch Chunk + Embed (RAG) ─────────────────
      if (extractedText.isNotEmpty) {
        // Mandatory Cool-down to let the RPM quota breathe
        setState(() { _uploadStatus = '⏳ Cooling down AI quota (10s)...'; });
        await Future.delayed(const Duration(seconds: 10));

        setState(() { 
          _uploadStep = 3;
          _uploadStatus = '✂️ Analyzing content structure...'; 
        });

        final chunks = _splitIntoChunks(extractedText, size: 800);
        debugPrint('✅ RAG: Created ${chunks.length} chunks from ${extractedText.length} chars');

        setState(() { 
          _uploadStep = 4;
          _uploadStatus = '⚡ Mapping ${chunks.length} segments to AI search index...'; 
        });

        try {
          final embeddings = await AIService.createBatchEmbeddings(chunks);
          
          if (embeddings.isNotEmpty) {
             setState(() { _uploadStatus = '💾 Saving AI search index...'; });
             await SupabaseService.saveDocumentChunks(
              documentId: doc.id,
              studentId:  auth.currentUser!.id,
              chunks:     chunks.sublist(0, embeddings.length),
              embeddings: embeddings,
            );
            debugPrint('✅ RAG: ${embeddings.length} chunks indexed successfully');
          } else {
             debugPrint('⚠️ RAG: No embeddings generated');
          }
        } catch (e) {
          debugPrint('⚠️ Indexing Error: $e');
          _showSnack('AI search index partially failed. You can retry via the 🔄 icon.', isError: true, duration: 6);
        }
      }

      // ── Done ───────────────────────────────────────────────
      await _loadDocs();
      setState(() { _uploading = false; _uploadStep = 0; _uploadStatus = ''; });

      if (extractedText.isNotEmpty) {
        _showSnack('✅ Document uploaded & indexed! Ask AI anything about it.', isError: false, duration: 4);
      } else {
        _showSnack('⚠️ Document saved, but AI was too busy to read it. Please click the 🔄 retry icon later.', isError: true, duration: 6);
      }

    } catch (e) {
      setState(() { _uploading = false; _uploadStep = 0; _uploadStatus = ''; });
      
      // Error Shield: Convert technical Gemini errors into student-friendly messages
      String friendlyMsg = 'Upload failed: $e';
      if (e.toString().contains('OCR_FAILED') || e.toString().contains('429')) {
        friendlyMsg = '⚠️ The AI document reader is currently busy. Your document is saved, but you may need to click the 🔄 retry icon in a few minutes.';
      } else if (e.toString().contains('404')) {
        friendlyMsg = '⚠️ AI Configuration error. Our team has been notified. Please try again later.';
      }
      
      _showSnack(friendlyMsg, isError: true, duration: 6);
    }
  }

  // ── Chunk text into large pieces for 30+ page support ──
  List<String> _splitIntoChunks(String text, {int size = 800}) {
    if (text.trim().isEmpty) return [];
    
    // Normalize whitespace
    final cleanText = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    final words     = cleanText.split(' ');
    final chunks    = <String>[];
    const overlap   = 30;

    if (words.length <= size) {
      return [cleanText];
    }

    for (int i = 0; i < words.length; i += size - overlap) {
      final end   = (i + size).clamp(0, words.length);
      final chunk = words.sublist(i, end).join(' ').trim();
      
      // Only add if it has meaningful content
      if (chunk.length > 5) {
        chunks.add(chunk);
      }
      
      if (end >= words.length) break;
    }

    return chunks.isEmpty ? [cleanText] : chunks;
  }

  String _defaultTitle(String type) =>
      _docTypes.firstWhere((t) => t['value'] == type,
          orElse: () => {'label': 'Document'})['label'] as String;

  Future<void> _deleteDoc(StudentDocument doc) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text('Delete Document', style: GoogleFonts.playfairDisplay(
            fontSize: 18, fontWeight: FontWeight.w700)),
        content: Text(
            'Delete "${doc.title}"?\nThis will also remove its AI search index.',
            style: GoogleFonts.lato(fontSize: 14)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirm == true) {
      await SupabaseService.deleteDocument(doc.id);
      await _loadDocs();
      _showSnack('Document deleted', isError: false);
    }
  }

  // ── Manual Retry Indexing ─────────────────────────────────
  Future<void> _retryIndexing(StudentDocument partialDoc) async {
    setState(() { _uploading = true; _uploadStep = 1; 
        _uploadStatus = 'Fetching document content...'; });
    
    try {
      // 1. Fetch full doc (with Base64)
      final doc = await SupabaseService.getDocumentWithContent(partialDoc.id);
      if (doc == null || doc.contentBase64 == null) {
        throw Exception('Document content missing.');
      }

      // 2. OCR (if needed)
      String extracted = doc.extractedText ?? '';
      if (extracted.isEmpty) {
        setState(() { _uploadStep = 2; _uploadStatus = '🤖 Retrying AI OCR...'; });
        extracted = await AIService.analyzeDocumentImage(
          base64Data: doc.contentBase64!,
          mimeType:   doc.mimeType,
          docType:    doc.docType,
        );
        if (extracted.isNotEmpty) {
          await SupabaseService.updateDocumentExtractedText(doc.id, extracted);
        }
      }

      // 3. Batch Index
      if (extracted.isNotEmpty) {
        // Cool-down
        setState(() { _uploadStatus = '⏳ Waiting for AI (Cooling)...'; });
        await Future.delayed(const Duration(seconds: 5));

        setState(() { _uploadStep = 3; 
            _uploadStatus = '⚡ Re-indexing chunks (Batch API)...'; });
        final chunks = _splitIntoChunks(extracted);
        final embeddings = await AIService.createBatchEmbeddings(chunks);
        
        if (embeddings.isNotEmpty) {
          await SupabaseService.saveDocumentChunks(
            documentId: doc.id, studentId: doc.studentId!,
            chunks: chunks.sublist(0, embeddings.length),
            embeddings: embeddings,
          );
        }
      }

      await _loadDocs();
      _showSnack('✅ Document re-indexed successfully!', isError: false);
    } catch (e) {
      _showSnack('Retry failed: $e', isError: true);
    } finally {
      setState(() { _uploading = false; _uploadStep = 0; _uploadStatus = ''; });
    }
  }

  void _showSnack(String msg,
      {required bool isError, int duration = 3}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg, style: GoogleFonts.lato(color: Colors.white)),
      backgroundColor: isError ? Colors.red : const Color(0xFF10B981),
      duration: Duration(seconds: duration),
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
    ));
  }

  // ══════════════════════════════════════════════════════════
  // BUILD
  // ══════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF0F4FF),
      appBar: AppBar(
        backgroundColor: AppTheme.primary,
        title: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('My Documents', style: GoogleFonts.playfairDisplay(
              fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white)),
          Text('Upload → AI reads & answers from them',
              style: GoogleFonts.lato(fontSize: 11, color: Colors.white70)),
        ]),
      ),
      body: _uploading
          ? _uploadingView()
          : _loading
              ? const Center(child: CircularProgressIndicator())
              : _docs.isEmpty
                  ? _emptyState()
                  : _docsList(),
      floatingActionButton: _uploading
          ? null
          : FloatingActionButton.extended(
              onPressed: _showUploadDialog,
              backgroundColor: AppTheme.primary,
              icon: const Icon(Icons.upload_rounded, color: Colors.white),
              label: Text('Upload', style: GoogleFonts.lato(
                  color: Colors.white, fontWeight: FontWeight.w600)),
            ),
    );
  }

  // ── Upload progress view ──────────────────────────────────
  Widget _uploadingView() {
    final steps = [
      '1. Save document',
      '2. AI reads document (OCR)',
      '3. Split into chunks',
      '4. Create search index',
    ];
    return Center(
      child: Padding(padding: const EdgeInsets.all(40),
        child: Column(mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation(AppTheme.primary)),
            const SizedBox(height: 28),
            Text(_uploadStatus, textAlign: TextAlign.center,
                style: GoogleFonts.lato(fontSize: 16,
                    fontWeight: FontWeight.w700, color: AppTheme.primary)),
            const SizedBox(height: 20),
            ...List.generate(steps.length, (i) {
              final done    = i < _uploadStep - 1;
              final current = i == _uploadStep - 1;
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(children: [
                  Icon(
                    done ? Icons.check_circle_rounded
                        : current ? Icons.radio_button_checked_rounded
                            : Icons.radio_button_unchecked_rounded,
                    size: 18,
                    color: done ? const Color(0xFF10B981)
                        : current ? AppTheme.primary
                            : const Color(0xFFD1D5DB),
                  ),
                  const SizedBox(width: 10),
                  Text(steps[i], style: GoogleFonts.lato(
                      fontSize: 13,
                      fontWeight: current ? FontWeight.w700 : FontWeight.normal,
                      color: done || current
                          ? const Color(0xFF111827)
                          : const Color(0xFF9CA3AF))),
                ]),
              );
            }),
            const SizedBox(height: 16),
            Text(
              '⏳ Please wait — AI is processing.\nDo not close this screen.',
              textAlign: TextAlign.center,
              style: GoogleFonts.lato(fontSize: 12,
                  color: const Color(0xFF6B7280), height: 1.5)),
          ]),
      ),
    );
  }

  Widget _emptyState() => Center(
    child: SingleChildScrollView(
      padding: const EdgeInsets.all(40),
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Container(width: 100, height: 100,
          decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.08),
              shape: BoxShape.circle),
          child: const Icon(Icons.cloud_upload_outlined,
              size: 48, color: AppTheme.primary)),
        const SizedBox(height: 24),
        Text('No Documents Yet', style: GoogleFonts.playfairDisplay(
            fontSize: 22, fontWeight: FontWeight.w700,
            color: const Color(0xFF111827))),
        const SizedBox(height: 12),
        Text(
          'Upload your documents and AI will answer:\n\n'
          '📅 "What time is Monday Math class?"\n'
          '📊 "How much did I score in Physics?"\n'
          '🗓️ "When is my Chemistry exam?"\n'
          '📚 "What topics are in Unit 3?"\n'
          '✅ "How many classes can I miss?"',
          textAlign: TextAlign.center,
          style: GoogleFonts.lato(fontSize: 14,
              color: const Color(0xFF6B7280), height: 1.7)),
        const SizedBox(height: 80),
      ]),
    ),
  );

  Widget _docsList() => RefreshIndicator(
    onRefresh: _loadDocs,
    child: ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
      children: [
        // RAG active banner
        Container(
          padding: const EdgeInsets.all(12),
          margin: const EdgeInsets.only(bottom: 14),
          decoration: BoxDecoration(
            color: const Color(0xFFEFF6FF),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0xFFBFDBFE))),
          child: Row(children: [
            const Icon(Icons.auto_awesome, color: Color(0xFF3B82F6), size: 16),
            const SizedBox(width: 8),
            Expanded(child: Text(
              'RAG Active — just chat normally. AI will search your '
              'documents automatically and answer accurately.',
              style: GoogleFonts.lato(fontSize: 12,
                  color: const Color(0xFF1E40AF), height: 1.4))),
          ]),
        ),
        ..._buildGroupedList(),
      ],
    ),
  );

  List<Widget> _buildGroupedList() {
    final grouped = <String, List<StudentDocument>>{};
    for (final d in _docs) grouped.putIfAbsent(d.docType, () => []).add(d);

    final widgets = <Widget>[];
    for (final entry in grouped.entries) {
      final ti = _docTypes.firstWhere(
          (t) => t['value'] == entry.key,
          orElse: () => _docTypes.last);
      widgets.add(Padding(
        padding: const EdgeInsets.only(top: 4, bottom: 8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: (ti['color'] as Color).withOpacity(0.1),
            borderRadius: BorderRadius.circular(20)),
          child: Text(
            '${ti['emoji']} ${ti['label']} (${entry.value.length})',
            style: GoogleFonts.lato(fontSize: 13,
                fontWeight: FontWeight.w700, color: ti['color'] as Color)),
        ),
      ));
      for (final doc in entry.value) widgets.add(_docCard(doc, ti));
      widgets.add(const SizedBox(height: 6));
    }
    return widgets;
  }

  Widget _docCard(StudentDocument doc, Map<String, dynamic> ti) {
    final color   = ti['color'] as Color;
    final indexed = doc.extractedText?.isNotEmpty == true;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: Colors.white, borderRadius: BorderRadius.circular(16),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05),
            blurRadius: 8, offset: const Offset(0, 3))]),
      child: Padding(padding: const EdgeInsets.all(14),
        child: Row(children: [
          Container(width: 50, height: 50,
            decoration: BoxDecoration(color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12)),
            child: Center(child: Text(ti['emoji'] as String,
                style: const TextStyle(fontSize: 22)))),
          const SizedBox(width: 12),
          Expanded(child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(doc.title, style: GoogleFonts.lato(fontSize: 14,
                  fontWeight: FontWeight.w700, color: const Color(0xFF111827))),
              Text(doc.fileName, style: GoogleFonts.lato(
                  fontSize: 12, color: const Color(0xFF6B7280)),
                  overflow: TextOverflow.ellipsis),
              const SizedBox(height: 5),
              Row(children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: indexed
                        ? const Color(0xFFF0FDF4)
                        : const Color(0xFFFFF7ED),
                    borderRadius: BorderRadius.circular(8)),
                  child: Text(
                    indexed ? '✅ AI Indexed' : '⚠️ Not indexed',
                    style: GoogleFonts.lato(fontSize: 10,
                        fontWeight: FontWeight.w600,
                        color: indexed
                            ? const Color(0xFF16A34A)
                            : const Color(0xFFD97706)))),
                const SizedBox(width: 8),
                Text(DateFormat('MMM d').format(doc.createdAt.toLocal()),
                    style: GoogleFonts.lato(fontSize: 11,
                        color: const Color(0xFF9CA3AF))),
              ]),
            ])),
          GestureDetector(
            onTap: () => _deleteDoc(doc),
            child: Container(width: 34, height: 34,
              decoration: BoxDecoration(
                  color: const Color(0xFFFEF2F2),
                  borderRadius: BorderRadius.circular(10)),
              child: const Icon(Icons.delete_outline_rounded,
                  color: Color(0xFFEF4444), size: 18))),
          if (!indexed) ...[
            const SizedBox(width: 8),
            GestureDetector(
              onTap: () => _retryIndexing(doc),
              child: Container(width: 34, height: 34,
                decoration: BoxDecoration(
                    color: const Color(0xFFEFF6FF),
                    borderRadius: BorderRadius.circular(10)),
                child: const Icon(Icons.refresh_rounded,
                    color: Color(0xFF3B82F6), size: 18))),
          ],
        ]),
      ),
    );
  }
}