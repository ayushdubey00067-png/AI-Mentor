
List<String> _splitIntoChunksV2(String text, {int size = 300}) {
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

void main() {
  print("--- Test 1 (Very short) ---");
  print(_splitIntoChunksV2("Hi."));

  print("\n--- Test 2 (Empty) ---");
  print(_splitIntoChunksV2("   "));

  print("\n--- Test 3 (Exactly size) ---"); // size=300 tokens
  String text300 = List.generate(300, (i) => "w$i").join(" ");
  var res3 = _splitIntoChunksV2(text300);
  print("Chunks: ${res3.length}, Words in first: ${res3[0].split(' ').length}");

  print("\n--- Test 4 (Over size) ---"); // 500 words
  String text500 = List.generate(500, (i) => "w$i").join(" ");
  var res4 = _splitIntoChunksV2(text500);
  print("Chunks: ${res4.length}");
  print("Chunk 1 length: ${res4[0].split(' ').length}");
  print("Chunk 2 length: ${res4[1].split(' ').length}");
  print("Overlap test: ${res4[1].startsWith('w${300-30}')}");
}
