// lib/services/document_queue_service.dart
import 'dart:convert';
import 'dart:ui';
import 'package:flutter/foundation.dart';
import 'package:syncfusion_flutter_pdf/pdf.dart';
import '../models/models.dart';
import 'academic_document_chunker.dart';
import 'ai_service.dart';
import 'supabase_service.dart';

class DocumentOcrJob {
  final String documentId;
  final String docTitle;
  final String docType;
  final String academicYear;
  final String semester;
  final String targetScope;
  final String? targetRollNo;

  int currentPage;
  int totalPages;
  String status; // 'pending', 'processing', 'paused', 'completed', 'failed'
  String statusMessage;
  String? errorMessage;
  double progress; // 0.0 to 1.0
  bool cancelRequested;

  DocumentOcrJob({
    required this.documentId,
    required this.docTitle,
    required this.docType,
    required this.academicYear,
    required this.semester,
    required this.targetScope,
    this.targetRollNo,
    this.currentPage = 0,
    this.totalPages = 0,
    this.status = 'pending',
    this.statusMessage = 'Ready',
    this.errorMessage,
    this.progress = 0.0,
    this.cancelRequested = false,
  });
}

class DocumentQueueService extends ChangeNotifier {
  final Map<String, DocumentOcrJob> _jobs = {};

  DocumentOcrJob? getJob(String docId) => _jobs[docId];

  bool isProcessing(String docId) {
    final j = _jobs[docId];
    return j != null && j.status == 'processing';
  }

  bool isPaused(String docId) {
    final j = _jobs[docId];
    return j != null && j.status == 'paused';
  }

  bool isCompleted(String docId) {
    final j = _jobs[docId];
    return j != null && j.status == 'completed';
  }

  /// Starts or resumes background OCR processing for a document
  Future<void> startOcr(StudentDocument doc, {bool forceReindex = false}) async {
    final docId = doc.id;
    if (isProcessing(docId)) {
      debugPrint('⚠️ Document $docId is already running OCR.');
      return;
    }

    final job = DocumentOcrJob(
      documentId: docId,
      docTitle: doc.title,
      docType: doc.docType,
      academicYear: doc.academicYear ?? '2026-2027',
      semester: doc.term,
      targetScope: doc.targetScope,
      targetRollNo: doc.targetRollNo,
      status: 'processing',
      statusMessage: 'Loading document...',
    );
    _jobs[docId] = job;
    notifyListeners();

    try {
      // 1. Fetch document binary data
      final bytes = await SupabaseService.getDocumentBytes(doc);
      if (bytes == null || bytes.isEmpty) {
        job.status = 'failed';
        job.errorMessage = 'Document binary file not found';
        job.statusMessage = 'Failed: File missing';
        notifyListeners();
        return;
      }

      final isPdf = doc.mimeType == 'application/pdf' || doc.fileName.toLowerCase().endsWith('.pdf');
      PdfDocument? pdfDoc;
      int totalPages = 1;

      if (isPdf) {
        pdfDoc = PdfDocument(inputBytes: bytes);
        totalPages = pdfDoc.pages.count;
      }

      job.totalPages = totalPages;

      // Determine starting page: check how many pages already extracted
      int startPage = 0;
      if (!forceReindex) {
        if (doc.extractedJson != null && doc.extractedJson!.isNotEmpty) {
          final existingPages = doc.extractedJson!.keys
              .where((k) => k.startsWith('page_'))
              .map((k) => int.tryParse(k.replaceFirst('page_', '')) ?? 0)
              .toList();
          if (existingPages.isNotEmpty) {
            existingPages.sort();
            startPage = existingPages.last;
          }
        } else if (doc.ocrProgress != null && doc.ocrProgress!['current'] != null) {
          startPage = (doc.ocrProgress!['current'] as num).toInt();
        }

        // If already fully completed
        if (startPage >= totalPages && totalPages > 0) {
          job.currentPage = totalPages;
          job.progress = 1.0;
          job.status = 'completed';
          job.statusMessage = 'All $totalPages pages already indexed';
          await SupabaseService.updateDocumentOcrStatus(
            docId: docId,
            status: 'completed',
            progress: {'current': totalPages, 'total': totalPages},
          );
          pdfDoc?.dispose();
          notifyListeners();
          return;
        }
      }

      job.currentPage = startPage;
      job.progress = totalPages > 0 ? (startPage / totalPages) : 0.0;
      notifyListeners();

      // Update DB to mark as processing
      await SupabaseService.updateDocumentOcrStatus(
        docId: docId,
        status: 'processing',
        progress: {'current': startPage, 'total': totalPages},
      );

      // Process page by page
      for (int i = startPage; i < totalPages; i++) {
        final pageNum = i + 1;

        // Check if user requested pause / stop
        if (job.cancelRequested) {
          job.status = 'paused';
          job.statusMessage = 'Paused at Page $i of $totalPages';
          await SupabaseService.updateDocumentOcrStatus(
            docId: docId,
            status: 'paused',
            progress: {'current': i, 'total': totalPages},
          );
          pdfDoc?.dispose();
          notifyListeners();
          debugPrint('⏹ OCR gracefully stopped at page $i for doc $docId');
          return;
        }

        job.statusMessage = 'Rendering page $pageNum of $totalPages...';
        notifyListeners();
        await Future.delayed(Duration.zero);

        // Prepare page base64 data
        String pageBase64 = '';
        String pageMime = doc.mimeType;

        if (isPdf && pdfDoc != null) {
          final singleDoc = PdfDocument();
          final page = pdfDoc.pages[i];
          singleDoc.pageSettings.size = page.size;
          if (page.size.width > page.size.height) {
            singleDoc.pageSettings.orientation = PdfPageOrientation.landscape;
          } else {
            singleDoc.pageSettings.orientation = PdfPageOrientation.portrait;
          }
          singleDoc.pages.add().graphics.drawPdfTemplate(page.createTemplate(), Offset.zero);
          final singleBytes = singleDoc.saveSync();
          singleDoc.dispose();
          pageBase64 = base64Encode(singleBytes);
          pageMime = 'application/pdf';
        } else {
          pageBase64 = base64Encode(bytes);
        }

        // Retry loop for OCR with exponential backoff
        Map<String, dynamic>? pageJson;
        int attempts = 0;
        const int maxAttempts = 3;

        while (attempts < maxAttempts && pageJson == null) {
          attempts++;
          try {
            job.statusMessage = 'Extracting Native JSON (Page $pageNum of $totalPages)...';
            notifyListeners();

            pageJson = await AIService.extractDocumentNativeJson(
              base64Data: pageBase64,
              mimeType: pageMime,
              docType: doc.docType,
            );
          } catch (err) {
            debugPrint('⚠️ OCR page $pageNum attempt $attempts failed: $err');
            if (attempts < maxAttempts) {
              job.statusMessage = 'Page $pageNum retry $attempts of $maxAttempts (waiting ${attempts * 2}s)...';
              notifyListeners();
              await Future.delayed(Duration(seconds: attempts * 2));
            } else {
              // Failed after 3 retries
              job.status = 'failed';
              job.errorMessage = 'Failed to extract page $pageNum after $maxAttempts attempts: $err';
              job.statusMessage = 'Failed on page $pageNum';
              await SupabaseService.updateDocumentOcrStatus(
                docId: docId,
                status: 'failed',
                progress: {'current': i, 'total': totalPages},
              );
              pdfDoc?.dispose();
              notifyListeners();
              return;
            }
          }
        }

        if (pageJson != null) {
          // 1. Atomic database upsert into extracted_json JSONB
          job.statusMessage = 'Saving Page $pageNum structured records...';
          notifyListeners();

          await SupabaseService.appendDocumentPageJson(
            docId: docId,
            pageNumber: pageNum,
            pageJson: pageJson,
            totalPages: totalPages,
          );

          // 2. Generate Semantic Cards for Category Table Vector RAG
          job.statusMessage = 'Building AI Semantic Cards for Page $pageNum...';
          notifyListeners();

          final semanticCards = AcademicDocumentChunker.generateSemanticCardsFromJson(
            docType: doc.docType,
            docTitle: doc.title,
            jsonMap: pageJson,
            academicYear: doc.academicYear ?? '2026-2027',
            semester: doc.term,
          );

          if (semanticCards.isNotEmpty) {
            job.statusMessage = 'Embedding ${semanticCards.length} cards (Page $pageNum)...';
            notifyListeners();

            try {
              final embeddings = await AIService.createBatchEmbeddings(semanticCards);
              await SupabaseService.saveDocumentChunks(
                documentId: docId,
                docType: doc.docType,
                academicYear: doc.academicYear ?? '2026-2027',
                semester: doc.term,
                targetScope: doc.targetScope,
                targetRollNo: doc.targetRollNo,
                chunks: semanticCards,
                embeddings: embeddings,
              );
            } catch (embErr) {
              debugPrint('⚠️ Warning generating embeddings for page $pageNum: $embErr');
            }
          }

          // Advance progress
          job.currentPage = pageNum;
          job.progress = pageNum / totalPages;
          job.statusMessage = 'Page $pageNum of $totalPages completed';
          notifyListeners();
        }

        // Small pause between pages to prevent rate limits
        await Future.delayed(const Duration(milliseconds: 400));
      }

      pdfDoc?.dispose();

      // All pages complete!
      job.status = 'completed';
      job.currentPage = totalPages;
      job.progress = 1.0;
      job.statusMessage = 'Completed ($totalPages pages indexed)';
      await SupabaseService.updateDocumentOcrStatus(
        docId: docId,
        status: 'completed',
        progress: {'current': totalPages, 'total': totalPages},
      );
      notifyListeners();
      debugPrint('🎉 Successfully completed Native JSON OCR for doc $docId ($totalPages pages)');
    } catch (e) {
      debugPrint('❌ DocumentQueueService exception: $e');
      job.status = 'failed';
      job.errorMessage = e.toString();
      job.statusMessage = 'Error: $e';
      await SupabaseService.updateDocumentOcrStatus(docId: docId, status: 'failed');
      notifyListeners();
    }
  }

  /// Stops / pauses ongoing OCR safely at the next page boundary
  void stopOcr(String docId) {
    final job = _jobs[docId];
    if (job != null && job.status == 'processing') {
      job.cancelRequested = true;
      job.statusMessage = 'Stopping after current page...';
      notifyListeners();
      debugPrint('⏹ Stop requested for doc $docId');
    }
  }

  /// Resumes OCR for a paused or failed document
  Future<void> resumeOcr(StudentDocument doc) async {
    final job = _jobs[doc.id];
    if (job != null) {
      job.cancelRequested = false;
      job.errorMessage = null;
    }
    await startOcr(doc);
  }
}
