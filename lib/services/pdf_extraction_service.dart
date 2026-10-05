// lib/services/pdf_extraction_service.dart
import 'dart:convert';
import 'dart:ui';
import 'package:flutter/foundation.dart';
import 'package:syncfusion_flutter_pdf/pdf.dart';
import 'ai_service.dart';

class PDFExtractionService {
  /// Extracts all text from a PDF document byte array page by page.
  /// Yields execution to the Dart event loop between pages to prevent UI freezing.
  /// [onProgress] callback reports (currentPage, totalPages).
  static Future<String> extractTextFromPdfBytes(
    Uint8List bytes, {
    void Function(int currentPage, int totalPages)? onProgress,
  }) async {
    try {
      final PdfDocument loadedDocument = PdfDocument(inputBytes: bytes);
      final int totalPages = loadedDocument.pages.count;
      debugPrint('📄 Loaded PDF with $totalPages pages (${(bytes.length / (1024 * 1024)).toStringAsFixed(2)} MB)');

      final PdfTextExtractor extractor = PdfTextExtractor(loadedDocument);

      // Fast check on first 2 pages: if no digital text glyphs exist, it is a pure scanned image PDF
      final int sampleCount = totalPages > 2 ? 2 : totalPages;
      bool hasDigitalText = false;
      for (int i = 0; i < sampleCount; i++) {
        final sample = extractor.extractText(startPageIndex: i, endPageIndex: i);
        if (sample.trim().isNotEmpty) {
          hasDigitalText = true;
          break;
        }
      }

      if (!hasDigitalText) {
        debugPrint('⚠️ Scanned PDF detected: pages contain scanned image bitmaps rather than digital text. Skipping loop to prevent browser freeze.');
        loadedDocument.dispose();
        return '';
      }

      final StringBuffer fullTextBuffer = StringBuffer();
      for (int i = 0; i < totalPages; i++) {
        // Extract text for specific page (0-indexed)
        try {
          final String pageText = extractor.extractText(startPageIndex: i, endPageIndex: i);
          if (pageText.trim().isNotEmpty) {
            fullTextBuffer.writeln('--- PAGE ${i + 1} ---');
            fullTextBuffer.writeln(pageText.trim());
            fullTextBuffer.writeln();
          }
        } catch (pageError) {
          debugPrint('⚠️ Warning extracting page ${i + 1}: $pageError');
        }

        if (onProgress != null) {
          onProgress(i + 1, totalPages);
        }

        // Yield control to the Flutter UI event loop so the browser remains responsive
        await Future.delayed(Duration.zero);
      }

      loadedDocument.dispose();
      final String extracted = fullTextBuffer.toString().trim();
      debugPrint('✅ PDF Extraction complete: ${extracted.length} characters across $totalPages pages.');
      return extracted;
    } catch (e) {
      debugPrint('❌ PDFExtractionService error: $e');
      return '';
    }
  }

  /// Checks if a PDF contains native digital text or is a scanned image
  static Future<bool> isDigitalPdf(Uint8List bytes) async {
    try {
      final PdfDocument doc = PdfDocument(inputBytes: bytes);
      final extractor = PdfTextExtractor(doc);
      // Check first 3 pages
      final checkPages = doc.pages.count > 3 ? 3 : doc.pages.count;
      final sample = extractor.extractText(startPageIndex: 0, endPageIndex: checkPages - 1);
      doc.dispose();
      return sample.trim().length > 50;
    } catch (_) {
      return false;
    }
  }

  /// Extracts and runs Vision OCR on a multi-page PDF page-by-page / chunk-by-chunk.
  /// Converts each single page into a small standalone PDF template and passes it to Gemini Vision.
  /// Calls [onPageComplete] with (pageIndex, totalPages, pageText).
  static Future<void> runProgressiveVisionOcr({
    required Uint8List pdfBytes,
    required String docType,
    int startPage = 0,
    int maxPagesToProcess = 5,
    required void Function(int currentPage, int totalPages, String pageText) onPageComplete,
    required void Function(String status) onStatusUpdate,
    bool Function()? isCancelled,
  }) async {
    try {
      final PdfDocument sourceDoc = PdfDocument(inputBytes: pdfBytes);
      final int totalPages = sourceDoc.pages.count;
      final int endPage = (startPage + maxPagesToProcess < totalPages)
          ? startPage + maxPagesToProcess
          : totalPages;

      for (int i = startPage; i < endPage; i++) {
        if (isCancelled != null && isCancelled()) {
          debugPrint('Progressive OCR cancelled by user at page ${i + 1}');
          break;
        }

        onStatusUpdate('Rendering page ${i + 1} of $totalPages...');
        await Future.delayed(Duration.zero);

        final PdfDocument singleDoc = PdfDocument();
        final page = sourceDoc.pages[i];
        singleDoc.pageSettings.size = page.size;
        if (page.size.width > page.size.height) {
          singleDoc.pageSettings.orientation = PdfPageOrientation.landscape;
        } else {
          singleDoc.pageSettings.orientation = PdfPageOrientation.portrait;
        }
        singleDoc.pages.add().graphics.drawPdfTemplate(page.createTemplate(), Offset.zero);
        final List<int> singleBytes = singleDoc.saveSync();
        singleDoc.dispose();

        onStatusUpdate('Running Gemini Vision OCR on page ${i + 1} of $totalPages...');
        final String b64 = base64Encode(singleBytes);

        try {
          final pageJson = await AIService.extractDocumentNativeJson(
            base64Data: b64,
            mimeType: 'application/pdf',
            docType: docType,
          );

          if (pageJson.isNotEmpty) {
            final formattedText = jsonEncode(pageJson);
            onPageComplete(i + 1, totalPages, formattedText);
          }
        } catch (ocrErr) {
          debugPrint('⚠️ Error in Native JSON OCR for page ${i + 1}: $ocrErr');
        }

        await Future.delayed(const Duration(milliseconds: 300));
      }

      sourceDoc.dispose();
    } catch (e) {
      debugPrint('❌ runProgressiveVisionOcr failure: $e');
    }
  }
}
