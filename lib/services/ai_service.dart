import 'dart:convert';
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/models.dart';
import '../utils/constants.dart';
import '../utils/ai_config.dart';
import 'document_json_schemas.dart';

class AIService {
  static final _supabase = Supabase.instance.client;

  // ── CORE GEMINI CALL (via Supabase Edge Function) ──────────
  static Future<HttpResponseMock> _invokeFunction(
    String task,
    String model,
    Map<String, dynamic> body,
  ) async {
    try {
      final res = await _supabase.functions.invoke(
        kSupabaseChatFunction,
        body: {
          'model': model,
          ...body,
        },
        headers: {'x-gemini-task': task},
      ).timeout(const Duration(seconds: 60));

      if (res.status == 200) {
        return HttpResponseMock(res.status, jsonEncode(res.data));
      }

      debugPrint('❌ Supabase Function Error: Status=${res.status}');

      // Handle specific errors
      if (res.status == 429) {
        throw Exception('RATE_LIMIT: AI is currently overloaded.');
      }
      if (res.status == 503) {
        throw Exception('MODEL_OVERLOADED_503: Model $model is experiencing high demand.');
      }
      if (res.status == 404) {
        throw Exception(
            'NOT_FOUND: Edge function "chat" not found. Did you run "supabase functions deploy chat"?');
      }
      if (res.status == 500) {
        final errorData = res.data as Map<String, dynamic>?;
        if (errorData?['error'] == 'MISSING_API_KEY') {
          throw Exception(
              'CONFIG_ERROR: API Keys not set in Supabase secrets. Run "supabase secrets set GEMINI_API_KEYS=...".');
        }
      }

      throw Exception('AI_SERVICE_ERROR: ${res.status}');
    } catch (e) {
      debugPrint('❌ _invokeFunction Exception: $e');
      if (e is FunctionException) {
        if (e.status == 429) throw Exception('RATE_LIMIT: Model $model hit quota.');
        if (e.status == 503) throw Exception('MODEL_OVERLOADED_503: Model $model high demand.');
      }
      if (e is TimeoutException) {
        throw Exception('NETWORK_ERROR: Request timed out');
      }
      rethrow;
    }
  }

  // ══════════════════════════════════════════════════════════
  // EMBEDDING
  // ══════════════════════════════════════════════════════════
  static Future<List<double>> createEmbedding(String text) async {
    final trimmed = text.length > 6000 ? text.substring(0, 6000) : text;
    try {
      final res = await _invokeFunction(
        'embedContent',
        kGeminiEmbedModel,
        {
          'content': {
            'parts': [
              {'text': trimmed}
            ]
          },
          'taskType': 'RETRIEVAL_QUERY',
          'outputDimensionality': 768,
        },
      );

      final data = jsonDecode(res.body);
      final values = (data['embedding']['values'] as List);
      return values.map((v) => (v as num).toDouble()).toList();
    } catch (e) {
      debugPrint('❌ createEmbedding: $e');
      rethrow;
    }
  }

  static Future<List<List<double>>> createBatchEmbeddings(
    List<String> chunks, {
    void Function(int currentBatch, int totalBatches)? onProgress,
  }) async {
    if (chunks.isEmpty) return [];
    final List<List<double>> allEmbeddings = [];
    const int batchSize = 20;
    final int totalBatches = (chunks.length / batchSize).ceil();

    for (int b = 0; b < chunks.length; b += batchSize) {
      final batchIndex = (b / batchSize).floor() + 1;
      final int end = (b + batchSize < chunks.length) ? b + batchSize : chunks.length;
      final batchChunks = chunks.sublist(b, end);

      if (onProgress != null) {
        onProgress(batchIndex, totalBatches);
      }

      try {
        final List<Map<String, dynamic>> requests = [];
        for (final chunk in batchChunks) {
          final trimmed = chunk.length > 6000 ? chunk.substring(0, 6000) : chunk;
          requests.add({
            'model': 'models/$kGeminiEmbedModel',
            'taskType': 'RETRIEVAL_DOCUMENT',
            'outputDimensionality': 768,
            'content': {
              'parts': [
                {'text': trimmed}
              ]
            }
          });
        }

        final res = await _invokeFunction(
          'batchEmbedContents',
          kGeminiEmbedModel,
          {'requests': requests},
        );

        final data = jsonDecode(res.body);
        final List embeddingsList = data['embeddings'] as List? ?? [];
        for (final e in embeddingsList) {
          final vals = (e['values'] as List);
          allEmbeddings.add(vals.map((v) => (v as num).toDouble()).toList());
        }

        if (b + batchSize < chunks.length) {
          await Future.delayed(const Duration(milliseconds: 100));
        }
      } catch (e) {
        debugPrint('❌ createBatchEmbeddings error on batch $batchIndex: $e');
        if (allEmbeddings.isEmpty) rethrow;
      }
    }

    return allEmbeddings;
  }

  // ══════════════════════════════════════════════════════════
  // ANALYZE IMAGE (Vision)
  // ══════════════════════════════════════════════════════════
  static Future<String> analyzeDocumentImage({
    required String base64Data,
    required String mimeType,
    required String docType,
    String? customPrompt,
  }) async {
    final prompt = customPrompt ??
        'You are extracting text from a college document.\n'
            'Document type: $docType\n\n'
            'Extract ALL text completely and accurately.';

    try {
      final res = await _invokeFunction(
        'generateContent',
        AIConfig.visionModel, // Vision capable model
        {
          'contents': [
            {
              'parts': [
                {
                  'inline_data': {
                    'mime_type': mimeType,
                    'data': base64Data,
                  }
                },
                {'text': prompt},
              ]
            }
          ],
          'generationConfig': {
            'maxOutputTokens': 8192,
            'temperature': 0.0,
          },
        },
      );

      final data = jsonDecode(res.body);
      final text =
          data['candidates']?[0]['content']?['parts']?[0]['text'] as String?;
      return text?.trim() ?? '';
    } catch (e) {
      debugPrint('❌ analyzeDocumentImage: $e');
      rethrow;
    }
  }

  /// Extracts structured Native JSON directly from document images using Gemini JSON mode
  static Future<Map<String, dynamic>> extractDocumentNativeJson({
    required String base64Data,
    required String mimeType,
    required String docType,
  }) async {
    final prompt = DocumentJsonSchemas.getPromptForDocType(docType);

    try {
      final res = await _invokeFunction(
        'generateContent',
        AIConfig.visionModel,
        {
          'contents': [
            {
              'parts': [
                {
                  'inline_data': {
                    'mime_type': mimeType,
                    'data': base64Data,
                  }
                },
                {'text': prompt},
              ]
            }
          ],
          'generationConfig': {
            'response_mime_type': 'application/json',
            'maxOutputTokens': 8192,
            'temperature': 0.0,
          },
        },
      );

      final data = jsonDecode(res.body);
      final text = data['candidates']?[0]['content']?['parts']?[0]['text'] as String?;
      if (text == null || text.trim().isEmpty) {
        throw Exception('Empty JSON output from vision model');
      }

      final parsed = jsonDecode(text.trim());
      if (parsed is Map<String, dynamic>) {
        return parsed;
      } else if (parsed is List) {
        return {'items': parsed};
      }
      return {'data': parsed};
    } catch (e) {
      debugPrint('❌ extractDocumentNativeJson error: $e');
      rethrow;
    }
  }

  // ══════════════════════════════════════════════════════════
  // STUDENT/MENTOR CHAT
  // ══════════════════════════════════════════════════════════
  static Future<String> sendStudentMessage({
    required List<MessageModel> history,
    required String newMessage,
    String? studentName,
    String? rollNo,
    String? dept,
    String? program,
    String? branch,
    String? semester,
    String? section,
    List<String>? skills,
    List<String>? interests,
    String? ragContext,
    void Function(String partialText)? onStreamChunk,
  }) async {
    final systemPrompt = buildStudentPrompt(
      name: studentName,
      rollNo: rollNo,
      dept: dept,
      program: program,
      branch: branch,
      semester: semester,
      section: section,
      skills: skills,
      interests: interests,
      ragContext: ragContext,
    );
    final fullResponse = await _callGemini(
      systemPrompt: systemPrompt,
      contents: ContextWindowManager.buildSlidingWindow(
        rawHistory: history,
        newMessage: newMessage,
        maxTurns: AIConfig.maxHistoryTurns,
      ),
      temperature: AIConfig.studentTemperature,
    );

    // Provide word-by-word streaming effect if consumer requests it
    if (onStreamChunk != null && fullResponse.isNotEmpty) {
      await _streamText(fullResponse, onStreamChunk);
    }

    return fullResponse;
  }

  static Future<String> sendMentorMessage({
    required List<Map<String, dynamic>> history,
    required String newMessage,
    required String mentorName,
    String? designation,
    String? dept,
    List<String>? expertise,
    int? totalStudents,
    int? activeChats,
    String? ragContext,
    void Function(String partialText)? onStreamChunk,
  }) async {
    final systemPrompt = buildMentorPrompt(
      mentorName: mentorName,
      designation: designation,
      dept: dept,
      expertise: expertise,
      totalStudents: totalStudents,
      activeChats: activeChats,
      ragContext: ragContext,
    );

    // Sliding window for mentor chat (keep last 10 messages)
    final recentHistory = history.length > AIConfig.maxHistoryTurns
        ? history.sublist(history.length - AIConfig.maxHistoryTurns)
        : history;

    final contents = <Map<String, dynamic>>[];
    for (final msg in recentHistory) {
      final role = msg['role'] == 'assistant' ? 'model' : 'user';
      final content = msg['content'] as String? ?? '';
      if (content.trim().isEmpty) continue;
      contents.add({
        'role': role,
        'parts': [
          {'text': content}
        ]
      });
    }
    contents.add({
      'role': 'user',
      'parts': [
        {'text': newMessage}
      ]
    });

    final fullResponse = await _callGemini(
      systemPrompt: systemPrompt,
      contents: contents,
      temperature: AIConfig.mentorTemperature,
    );

    if (onStreamChunk != null && fullResponse.isNotEmpty) {
      await _streamText(fullResponse, onStreamChunk);
    }

    return fullResponse;
  }

  /// Smooth client-side batched word streaming effect
  static Future<void> _streamText(String fullText, void Function(String) onStreamChunk) async {
    final words = fullText.split(' ');
    final buffer = StringBuffer();
    const int batchSize = 3;
    for (int i = 0; i < words.length; i += batchSize) {
      final end = (i + batchSize < words.length) ? i + batchSize : words.length;
      for (int j = i; j < end; j++) {
        if (buffer.isNotEmpty) buffer.write(' ');
        buffer.write(words[j]);
      }
      onStreamChunk(buffer.toString());
      // Smooth typing delay without starving the browser UI loop
      await Future.delayed(const Duration(milliseconds: 16));
    }
  }

  static Future<String> _callGemini({
    required String systemPrompt,
    required List<Map<String, dynamic>> contents,
    double temperature = 0.6,
  }) async {
    final modelsToTry = [kGeminiChatModel, ...kGeminiFallbacks];
    String? lastError;

    for (final model in modelsToTry) {
      try {
        debugPrint('🤖 Attempting AI call with model: $model');
        final res = await _invokeFunction(
          'generateContent',
          model,
          {
            'systemPrompt': systemPrompt,
            'contents': contents,
            'generationConfig': {
              'maxOutputTokens': AIConfig.maxOutputTokens,
              'temperature': temperature,
              'topP': 0.9,
              'thinkingConfig': {'thinkingBudget': 0},
            },
            'safetySettings': [
              {
                'category': 'HARM_CATEGORY_HARASSMENT',
                'threshold': 'BLOCK_ONLY_HIGH'
              },
              {
                'category': 'HARM_CATEGORY_HATE_SPEECH',
                'threshold': 'BLOCK_ONLY_HIGH'
              },
            ],
          },
        );

        final data = jsonDecode(res.body);
        final text =
            data['candidates']?[0]['content']?['parts']?[0]['text'] as String?;
        return text?.trim() ?? '⚠️ AI response was empty.';
      } catch (e) {
        lastError = e.toString();
        // Catch 429 quota exhaustion or 503 high demand spikes and rotate to fallback model
        if (lastError.contains('RATE_LIMIT') ||
            lastError.contains('429') ||
            lastError.contains('503') ||
            lastError.contains('MODEL_OVERLOADED') ||
            lastError.contains('RESOURCE_EXHAUSTED') ||
            lastError.contains('UNAVAILABLE')) {
          debugPrint('⏳ Model $model unavailable (429/503). Rotating to fallback model...');
          continue;
        }
        // If it's not a rate limit / demand error, rethrow immediately
        rethrow;
      }
    }

    throw Exception(lastError ?? 'All models failed to respond.');
  }

  static Future<Map<String, String>?> extractStudentDetails(
      String message) async {
    final modelsToTry = [kGeminiChatModel, ...kGeminiFallbacks];

    for (final model in modelsToTry) {
      try {
        final res = await _invokeFunction(
          'generateContent',
          model,
          {
            'systemPrompt':
                'Extract student profile details from this message as JSON. '
                    'Fields: name, program, branch, semester. '
                    'Use empty string "" if unknown. Return ONLY valid JSON, no markdown.',
            'contents': [
              {
                'role': 'user',
                'parts': [
                  {'text': message}
                ]
              }
            ],
            'generationConfig': {
              'responseMimeType': 'application/json',
              'temperature': 0.1,
            },
          },
        );

        final data = jsonDecode(res.body);
        final raw = data['candidates']?[0]['content']?['parts']?[0]['text'];
        if (raw == null) return null;

        final cleaned =
            raw.trim().replaceAll('```json', '').replaceAll('```', '').trim();
        final map = jsonDecode(cleaned) as Map<String, dynamic>;

        return {
          'name': (map['name'] ?? '').toString(),
          'program': (map['program'] ?? '').toString(),
          'branch': (map['branch'] ?? '').toString(),
          'semester': (map['semester'] ?? '').toString(),
        };
      } catch (e) {
        debugPrint('extractStudentDetails failed on $model: $e');
        continue;
      }
    }
    return null;
  }

  static String generateGreeting(String? userName,
          {String? branch, String? semester}) =>
      buildFirstMessage(userName, branch: branch, semester: semester);

  static String friendlyError(String e) {
    if (e.contains('RATE_LIMIT')) {
      return '⏳ **AI is busy.** Please wait a minute and try again.';
    }
    if (e.contains('NETWORK_ERROR')) {
      return '🌐 **Network issue.** Check your connection.';
    }
    if (e.contains('NOT_FOUND')) {
      return '🚀 **Backend not ready.** Function "chat" is not deployed yet.';
    }
    if (e.contains('CONFIG_ERROR')) {
      return '🔑 **Config Error.** Gemini API keys are missing in Supabase secrets.';
    }

    // Fallback: show the actual error for easier debugging
    return '❌ **Something went wrong.**\n\nDetail: ${e.replaceAll('Exception:', '').trim()}';
  }
}

// Simple wrapper to match expected behavior
class HttpResponseMock {
  final int statusCode;
  final String body;
  HttpResponseMock(this.statusCode, this.body);
}
