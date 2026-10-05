// lib/widgets/document_viewer_dialog.dart
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../models/models.dart';
import '../services/academic_document_chunker.dart';
import '../services/ai_service.dart';
import '../services/pdf_extraction_service.dart';
import '../services/supabase_service.dart';
import '../utils/app_theme.dart';

class DocumentViewerDialog extends StatefulWidget {
  final StudentDocument document;

  final VoidCallback? onUpdated;

  const DocumentViewerDialog({
    super.key,
    required this.document,
    this.onUpdated,
  });

  static Future<void> show(
    BuildContext context,
    StudentDocument document, {
    VoidCallback? onUpdated,
  }) {
    return showDialog(
      context: context,
      barrierDismissible: true,
      builder: (ctx) => DocumentViewerDialog(
        document: document,
        onUpdated: onUpdated,
      ),
    );
  }

  @override
  State<DocumentViewerDialog> createState() => _DocumentViewerDialogState();
}

class _DocumentViewerDialogState extends State<DocumentViewerDialog> {
  StudentDocument? _fullDoc;
  bool _isLoading = true;
  String _searchQuery = '';
  final TextEditingController _searchCtrl = TextEditingController();
  final ScrollController _scrollCtrl = ScrollController();

  // Progressive Chunk OCR State
  bool _isOcrRunning = false;
  bool _cancelOcrRequested = false;
  int _ocrCurrentPage = 0;
  int _ocrTotalPages = 0;
  String _ocrStatusMessage = '';

  static const Map<String, Map<String, dynamic>> _typeMeta = {
    'timetable':         {'label': 'Timetable',          'emoji': '📅', 'color': Color(0xFF3B82F6)},
    'academic_calendar': {'label': 'Academic Calendar',  'emoji': '🗓️', 'color': Color(0xFF8B5CF6)},
    'syllabus':          {'label': 'Syllabus',           'emoji': '📚', 'color': Color(0xFF10B981)},
    'marksheet':         {'label': 'Marksheet / Result', 'emoji': '📊', 'color': Color(0xFFF59E0B)},
    'attendance':        {'label': 'Attendance Register','emoji': '✅', 'color': Color(0xFF06B6D4)},
    'assignment':        {'label': 'Assignment Notice',  'emoji': '📝', 'color': Color(0xFFEF4444)},
    'other':             {'label': 'Circular / Other',   'emoji': '🏛️', 'color': Color(0xFF6B7280)},
  };

  @override
  void initState() {
    super.initState();
    _loadFullDoc();
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadFullDoc() async {
    if (widget.document.extractedText != null &&
        widget.document.extractedText!.isNotEmpty) {
      if (mounted) {
        setState(() {
          _fullDoc = widget.document;
          _isLoading = false;
        });
      }
      return;
    }

    try {
      final doc = await SupabaseService.getDocumentWithContent(widget.document.id);
      if (mounted) {
        setState(() {
          _fullDoc = doc ?? widget.document;
          _isLoading = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _fullDoc = widget.document;
          _isLoading = false;
        });
      }
    }
  }

  String _formatFileSize(int? bytes) {
    if (bytes == null || bytes <= 0) return '';
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  Future<void> _startProgressiveOcr({int batchSize = 5}) async {
    final doc = _fullDoc ?? widget.document;
    setState(() {
      _isOcrRunning = true;
      _cancelOcrRequested = false;
      _ocrStatusMessage = 'Loading document file...';
    });

    try {
      final bytes = await SupabaseService.getDocumentBytes(doc);
      if (bytes == null || bytes.isEmpty) {
        if (mounted) {
          setState(() {
            _isOcrRunning = false;
            _ocrStatusMessage = 'Document binary content not available.';
          });
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Document file could not be retrieved from Supabase.'),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }

      // Count existing pages in extracted text to determine startPage
      final existingText = doc.extractedText ?? '';
      final pageMatches = RegExp(r'---\s*PAGE\s+(\d+)\s*---', caseSensitive: false).allMatches(existingText);
      int startPage = 0;
      if (pageMatches.isNotEmpty) {
        startPage = int.tryParse(pageMatches.last.group(1) ?? '0') ?? 0;
      }

      String currentFullText = existingText;

      await PDFExtractionService.runProgressiveVisionOcr(
        pdfBytes: bytes,
        docType: doc.docType,
        startPage: startPage,
        maxPagesToProcess: batchSize,
        isCancelled: () => _cancelOcrRequested,
        onStatusUpdate: (msg) {
          if (mounted) {
            setState(() => _ocrStatusMessage = msg);
          }
        },
        onPageComplete: (cur, total, pageText) async {
          if (!mounted) return;
          final String newSection = '--- PAGE $cur ---\n$pageText\n\n';
          currentFullText = currentFullText.trim().isEmpty
              ? newSection
              : '$currentFullText\n$newSection';

          setState(() {
            _ocrCurrentPage = cur;
            _ocrTotalPages = total;
            _fullDoc = (_fullDoc ?? widget.document).copyWith(extractedText: currentFullText);
          });

          // Save progressively to Supabase database so progress is never lost
          await SupabaseService.updateDocumentExtractedText(doc.id, currentFullText);

          // Chunk and embed newly extracted page into pgvector
          try {
            List<String> pageChunks = [];
            try {
              final jsonDecoded = jsonDecode(pageText);
              if (jsonDecoded is Map<String, dynamic>) {
                // Save structured Native JSON to database
                await SupabaseService.appendDocumentPageJson(
                  docId: doc.id,
                  pageNumber: cur,
                  pageJson: jsonDecoded,
                  totalPages: total,
                );

                pageChunks = AcademicDocumentChunker.generateSemanticCardsFromJson(
                  docType: doc.docType,
                  docTitle: '${doc.title} (Page $cur)',
                  jsonMap: jsonDecoded,
                  academicYear: doc.academicYear ?? '2026-2027',
                  semester: doc.term,
                );
              }
            } catch (_) {}

            if (pageChunks.isEmpty) {
              pageChunks = AcademicDocumentChunker.createSemanticChunks(
                docType: doc.docType,
                docTitle: '${doc.title} (Page $cur)',
                rawText: pageText,
              );
            }

            if (pageChunks.isNotEmpty) {
              final embeddings = await AIService.createBatchEmbeddings(pageChunks);
              if (embeddings.isNotEmpty) {
                await SupabaseService.saveDocumentChunks(
                  documentId: doc.id,
                  docType: doc.docType,
                  academicYear: doc.academicYear ?? '2026-2027',
                  semester: doc.term,
                  targetScope: doc.targetScope,
                  targetRollNo: doc.targetRollNo,
                  chunks: pageChunks.sublist(0, embeddings.length),
                  embeddings: embeddings,
                );
              }
            }
          } catch (chunkErr) {
            debugPrint('Notice during progressive chunking: $chunkErr');
          }

          // Notify parent screens (Mentor dashboard or Student document view) to refresh indexed badge
          if (widget.onUpdated != null) {
            widget.onUpdated!();
          }
        },
      );

      if (mounted) {
        setState(() {
          _isOcrRunning = false;
          _ocrStatusMessage = 'Finished batch!';
        });
        if (widget.onUpdated != null) {
          widget.onUpdated!();
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isOcrRunning = false;
          _ocrStatusMessage = 'Error during OCR: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final doc = _fullDoc ?? widget.document;
    final meta = _typeMeta[doc.docType] ?? {
      'label': doc.docType,
      'emoji': '📄',
      'color': const Color(0xFF6B7280),
    };
    final Color badgeColor = meta['color'] as Color;
    final isClassScope = doc.targetScope == 'class';
    final contentText = doc.extractedText?.trim() ?? '';

    return Dialog(
      backgroundColor: Colors.transparent,
      insetPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
      child: Center(
        child: Container(
          width: 760,
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.88,
          ),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(20),
            boxShadow: const [
              BoxShadow(
                color: Color(0x22000000),
                blurRadius: 24,
                offset: Offset(0, 8),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header
              Container(
                padding: const EdgeInsets.fromLTRB(20, 16, 16, 16),
                decoration: const BoxDecoration(
                  color: Color(0xFFF9FAFB),
                  borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
                  border: Border(bottom: BorderSide(color: Color(0xFFE5E7EB))),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: badgeColor.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Center(
                        child: Text(
                          meta['emoji'] as String,
                          style: const TextStyle(fontSize: 22),
                        ),
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            doc.title,
                            style: GoogleFonts.lato(
                              fontSize: 17,
                              fontWeight: FontWeight.w700,
                              color: const Color(0xFF111827),
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          const SizedBox(height: 3),
                          Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                                decoration: BoxDecoration(
                                  color: badgeColor.withOpacity(0.1),
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(
                                  meta['label'] as String,
                                  style: GoogleFonts.lato(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: badgeColor,
                                  ),
                                ),
                              ),
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                                decoration: BoxDecoration(
                                  color: isClassScope ? const Color(0xFFEFF6FF) : const Color(0xFFFEF3C7),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(
                                    color: isClassScope ? const Color(0xFFBFDBFE) : const Color(0xFFFDE68A),
                                  ),
                                ),
                                child: Text(
                                  isClassScope ? '🌐 Entire Class' : '👤 ${doc.targetRollNo ?? "Personal"}',
                                  style: GoogleFonts.lato(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: isClassScope ? const Color(0xFF1E40AF) : const Color(0xFFB45309),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close_rounded, color: Color(0xFF6B7280)),
                      onPressed: () => Navigator.pop(context),
                      splashRadius: 20,
                    ),
                  ],
                ),
              ),

              // Metadata bar
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                color: const Color(0xFFF3F4F6),
                child: Row(
                  children: [
                    const Icon(Icons.insert_drive_file_outlined, size: 15, color: Color(0xFF6B7280)),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        doc.fileName,
                        style: GoogleFonts.lato(
                          fontSize: 12,
                          color: const Color(0xFF4B5563),
                          fontWeight: FontWeight.w500,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (doc.fileSize != null && doc.fileSize! > 0) ...[
                      Text(
                        _formatFileSize(doc.fileSize),
                        style: GoogleFonts.lato(
                          fontSize: 11,
                          color: const Color(0xFF9CA3AF),
                        ),
                      ),
                      const SizedBox(width: 10),
                    ],
                    Text(
                      DateFormat('MMM d, yyyy').format(doc.createdAt.toLocal()),
                      style: GoogleFonts.lato(
                        fontSize: 11,
                        color: const Color(0xFF9CA3AF),
                      ),
                    ),
                  ],
                ),
              ),

              // Active OCR Chunk Progress Banner
              if (_isOcrRunning)
                Container(
                  margin: const EdgeInsets.fromLTRB(20, 10, 20, 0),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0xFFEFF6FF),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: const Color(0xFFBFDBFE)),
                  ),
                  child: Row(
                    children: [
                      const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2, valueColor: AlwaysStoppedAnimation(Color(0xFF2563EB))),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _ocrStatusMessage,
                              style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700, color: const Color(0xFF1E40AF)),
                            ),
                            const SizedBox(height: 4),
                            LinearProgressIndicator(
                              value: _ocrTotalPages > 0 ? _ocrCurrentPage / _ocrTotalPages : null,
                              backgroundColor: const Color(0xFFDBEAFE),
                              valueColor: const AlwaysStoppedAnimation(Color(0xFF2563EB)),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 12),
                      OutlinedButton(
                        onPressed: () => setState(() => _cancelOcrRequested = true),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.red,
                          side: const BorderSide(color: Colors.red),
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        ),
                        child: const Text('Stop', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700)),
                      ),
                    ],
                  ),
                ),

              // Search bar
              if (contentText.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 12, 20, 4),
                  child: TextField(
                    controller: _searchCtrl,
                    onChanged: (v) => setState(() => _searchQuery = v.trim().toLowerCase()),
                    style: GoogleFonts.lato(fontSize: 13),
                    decoration: InputDecoration(
                      hintText: 'Search inside document (e.g. roll number, subject, grade)...',
                      hintStyle: GoogleFonts.lato(fontSize: 12, color: const Color(0xFF9CA3AF)),
                      prefixIcon: const Icon(Icons.search_rounded, size: 18, color: Color(0xFF9CA3AF)),
                      suffixIcon: _searchCtrl.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear, size: 16),
                              onPressed: () {
                                _searchCtrl.clear();
                                setState(() => _searchQuery = '');
                              },
                            )
                          : null,
                      isDense: true,
                      contentPadding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
                      filled: true,
                      fillColor: const Color(0xFFF9FAFB),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                        borderSide: const BorderSide(color: Color(0xFFE5E7EB)),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                        borderSide: const BorderSide(color: Color(0xFFE5E7EB)),
                      ),
                    ),
                  ),
                ),

              // Document Content Reader
              Expanded(
                child: _isLoading
                    ? const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            CircularProgressIndicator(strokeWidth: 2.5),
                            SizedBox(height: 12),
                            Text('Loading document content...', style: TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
                          ],
                        ),
                      )
                    : contentText.isEmpty
                        ? Center(
                            child: Padding(
                              padding: const EdgeInsets.all(32),
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Icon(Icons.document_scanner_outlined, size: 48, color: Color(0xFFD1D5DB)),
                                  const SizedBox(height: 12),
                                  Text(
                                    'No OCR text extracted yet',
                                    style: GoogleFonts.lato(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700,
                                      color: const Color(0xFF4B5563),
                                    ),
                                  ),
                                  const SizedBox(height: 6),
                                  Text(
                                    'This is a scanned paper document. Click below to start progressive chunk-by-chunk OCR indexing.',
                                    textAlign: TextAlign.center,
                                    style: GoogleFonts.lato(fontSize: 12, color: const Color(0xFF9CA3AF)),
                                  ),
                                  const SizedBox(height: 18),
                                  if (!_isOcrRunning)
                                    ElevatedButton.icon(
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: const Color(0xFF0284C7),
                                        foregroundColor: Colors.white,
                                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                                      ),
                                      onPressed: () => _startProgressiveOcr(batchSize: 5),
                                      icon: const Icon(Icons.auto_awesome, size: 16),
                                      label: Text(
                                        '⚡ Start OCR Indexing (Next 5 Pages)',
                                        style: GoogleFonts.lato(fontSize: 13, fontWeight: FontWeight.w700),
                                      ),
                                    ),
                                ],
                              ),
                            ),
                          )
                        : Container(
                            margin: const EdgeInsets.fromLTRB(20, 8, 20, 16),
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: const Color(0xFFFAFAFA),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: const Color(0xFFE5E7EB)),
                            ),
                            child: Scrollbar(
                              controller: _scrollCtrl,
                              thumbVisibility: true,
                              child: SingleChildScrollView(
                                controller: _scrollCtrl,
                                child: SelectableText(
                                  _filterContent(contentText, _searchQuery),
                                  style: GoogleFonts.firaCode(
                                    fontSize: 12.5,
                                    height: 1.6,
                                    color: const Color(0xFF1F2937),
                                  ),
                                ),
                              ),
                            ),
                          ),
              ),

              // Bottom action footer
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                decoration: const BoxDecoration(
                  color: Color(0xFFF9FAFB),
                  borderRadius: BorderRadius.vertical(bottom: Radius.circular(20)),
                  border: Border(top: BorderSide(color: Color(0xFFE5E7EB))),
                ),
                child: Row(
                  children: [
                    if (contentText.isNotEmpty) ...[
                      const Icon(Icons.auto_awesome, size: 14, color: Color(0xFF10B981)),
                      const SizedBox(width: 6),
                      Text(
                        'AI Indexed (${contentText.length} chars)',
                        style: GoogleFonts.lato(
                          fontSize: 12,
                          color: const Color(0xFF059669),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(width: 14),
                      if (!_isOcrRunning)
                        TextButton.icon(
                          onPressed: () => _startProgressiveOcr(batchSize: 5),
                          icon: const Icon(Icons.add_circle_outline, size: 14, color: Color(0xFF0284C7)),
                          label: Text(
                            'Index Next 5 Pages',
                            style: GoogleFonts.lato(fontSize: 12, fontWeight: FontWeight.w700, color: const Color(0xFF0284C7)),
                          ),
                        ),
                    ],
                    const Spacer(),
                    if (contentText.isNotEmpty)
                      OutlinedButton.icon(
                        onPressed: () {
                          Clipboard.setData(ClipboardData(text: contentText));
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Document text copied to clipboard'),
                              duration: Duration(seconds: 2),
                              behavior: SnackBarBehavior.floating,
                            ),
                          );
                        },
                        icon: const Icon(Icons.copy_rounded, size: 15),
                        label: const Text('Copy Text'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: const Color(0xFF374151),
                          side: const BorderSide(color: Color(0xFFD1D5DB)),
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        ),
                      ),
                    const SizedBox(width: 10),
                    ElevatedButton(
                      onPressed: () => Navigator.pop(context),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.primary,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ),
                      child: const Text('Close'),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _filterContent(String fullText, String query) {
    if (query.isEmpty) return fullText;
    final lines = fullText.split('\n');
    final matchedLines = <String>[];
    for (int i = 0; i < lines.length; i++) {
      if (lines[i].toLowerCase().contains(query)) {
        final start = (i - 1 >= 0) ? i - 1 : 0;
        final end = (i + 1 < lines.length) ? i + 1 : lines.length - 1;
        for (int j = start; j <= end; j++) {
          if (!matchedLines.contains(lines[j])) {
            matchedLines.add(lines[j]);
          }
        }
        matchedLines.add('---');
      }
    }
    return matchedLines.isNotEmpty
        ? matchedLines.join('\n')
        : 'No occurrences found for "$query".\n\nShowing full content:\n\n$fullText';
  }
}
