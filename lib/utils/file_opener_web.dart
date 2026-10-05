// lib/utils/file_opener_web.dart
import 'dart:convert';
import 'dart:html' as html;

void openRawDocument({
  required String base64Content,
  required String fileName,
  required String mimeType,
}) {
  try {
    final cleanBase64 = base64Content.contains(',')
        ? base64Content.split(',').last
        : base64Content;
    final bytes = base64Decode(cleanBase64);
    final effectiveMime = mimeType.isNotEmpty
        ? mimeType
        : fileName.toLowerCase().endsWith('.pdf')
            ? 'application/pdf'
            : 'application/octet-stream';
    final blob = html.Blob([bytes], effectiveMime);
    final url = html.Url.createObjectUrlFromBlob(blob);
    html.window.open(url, '_blank');
    Future.delayed(const Duration(minutes: 10), () {
      html.Url.revokeObjectUrl(url);
    });
  } catch (_) {
    try {
      final effectiveMime = mimeType.isNotEmpty ? mimeType : 'application/pdf';
      final cleanBase64 = base64Content.contains(',')
          ? base64Content.split(',').last
          : base64Content;
      final anchor = html.AnchorElement(href: 'data:$effectiveMime;base64,$cleanBase64')
        ..target = '_blank'
        ..download = fileName;
      anchor.click();
    } catch (_) {}
  }
}

void openRawBytes({
  required List<int> bytes,
  required String fileName,
  required String mimeType,
}) {
  try {
    final effectiveMime = mimeType.isNotEmpty
        ? mimeType
        : fileName.toLowerCase().endsWith('.pdf')
            ? 'application/pdf'
            : 'application/octet-stream';
    final blob = html.Blob([bytes], effectiveMime);
    final url = html.Url.createObjectUrlFromBlob(blob);
    html.window.open(url, '_blank');
    Future.delayed(const Duration(minutes: 10), () {
      html.Url.revokeObjectUrl(url);
    });
  } catch (_) {}
}
