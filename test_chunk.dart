
List<String> _splitIntoChunksV1(String text, {int size = 300}) {
  if (text.trim().isEmpty) return [];
  final words   = text.split(RegExp(r'\s+'));
  final chunks  = <String>[];
  const overlap = 30;

  for (int i = 0; i < words.length; i += size - overlap) {
    final end   = (i + size).clamp(0, words.length);
    final chunk = words.sublist(i, end).join(' ').trim();
    if (chunk.length > 30) chunks.add(chunk);
    if (end >= words.length) break;
  }

  return chunks.isEmpty
      ? [text.substring(0, text.length.clamp(0, 2000))]
      : chunks;
}

void main() {
  String testText = "This is a very short text.";
  print("Test 1 (Short): ${_splitIntoChunksV1(testText)}");

  String longText = "";
  for(int i=0; i<500; i++) longText += "word$i ";
  print("Test 2 (Long): Chunks: ${_splitIntoChunksV1(longText).length}");
  print("First chunk length: ${_splitIntoChunksV1(longText)[0].split(' ').length} words");
  
  String midText = "";
  for(int i=0; i<40; i++) midText += "word$i ";
  print("Test 3 (Mid): Chunks: ${_splitIntoChunksV1(midText)}");
}
