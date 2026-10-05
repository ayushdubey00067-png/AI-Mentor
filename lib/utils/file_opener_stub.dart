// lib/utils/file_opener_stub.dart
import 'package:flutter/foundation.dart';

void openRawDocument({
  required String base64Content,
  required String fileName,
  required String mimeType,
}) {
  debugPrint('openRawDocument stub called for $fileName (mime: $mimeType)');
}

void openRawBytes({
  required List<int> bytes,
  required String fileName,
  required String mimeType,
}) {
  debugPrint('openRawBytes stub called for $fileName (mime: $mimeType)');
}
