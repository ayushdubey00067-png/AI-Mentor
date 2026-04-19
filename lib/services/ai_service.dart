import 'dart:convert';
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/models.dart';
import '../utils/constants.dart';

class AIService {
  static final _supabase = Supabase.instance.client;

  // ── CORE GEMINI CALL (via Supabase Edge Function) ──────────
  static Future<http_Response_Mock> _invokeFunction(
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
        return http_Response_Mock(res.status, jsonEncode(res.data));
      }

      debugPrint('❌ Supabase Function Error: Status=${res.status}');

      // Handle specific errors
      if (res.status == 429) {
        throw Exception('RATE_LIMIT: AI is currently overloaded.');
      }
      if (res.status == 404) {
        throw Exception('NOT_FOUND: Edge function "chat" not found. Did you run "supabase functions deploy chat"?');
      }
      if (res.status == 500) {
        final errorData = res.data as Map<String, dynamic>?;
        if (errorData?['error'] == 'MISSING_API_KEY') {
          throw Exception('CONFIG_ERROR: API Keys not set in Supabase secrets. Run "supabase secrets set GEMINI_API_KEYS=...".');
        }
      }

      throw Exception('AI_SERVICE_ERROR: ${res.status}');
    } catch (e) {
      debugPrint('❌ _invokeFunction Exception: $e');
      if (e is FunctionException && e.status == 429) {
        throw Exception('RATE_LIMIT: Model $model hit quota.');
      }
      if (e is TimeoutException) throw Exception('NETWORK_ERROR: Request timed out');
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

  static Future<List<List<double>>> createBatchEmbeddings(List<String> chunks) async {
    if (chunks.isEmpty) return [];
    try {
      final List<Map<String, dynamic>> requests = [];
      for (final chunk in chunks) {
        final trimmed = chunk.length > 6000 ? chunk.substring(0, 6000) : chunk;
        requests.add({
          'model': 'models/$kGeminiEmbedModel',
          'taskType': 'RETRIEVAL_DOCUMENT',
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
      final List embeddingsList = data['embeddings'];
      return embeddingsList.map((e) {
        final vals = (e['values'] as List);
        return vals.map((v) => (v as num).toDouble()).toList();
      }).toList();
    } catch (e) {
      debugPrint('❌ createBatchEmbeddings: $e');
      rethrow;
    }
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
        'gemini-2.5-flash', // Vision capable model
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
          'generationConfig': {'maxOutputTokens': 4096},
        },
      );

      final data = jsonDecode(res.body);
      final text = data['candidates']?[0]['content']?['parts']?[0]['text'] as String?;
      return text?.trim() ?? '';
    } catch (e) {
      debugPrint('❌ analyzeDocumentImage: $e');
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
    List<String>? skills,
    List<String>? interests,
    String? ragContext,
  }) async {
    final systemPrompt = buildStudentPrompt(
      name: studentName,
      rollNo: rollNo,
      dept: dept,
      program: program,
      branch: branch,
      semester: semester,
      skills: skills,
      interests: interests,
      ragContext: ragContext,
    );
    return _callGemini(
      systemPrompt: systemPrompt,
      contents: _buildContents(history, newMessage),
    );
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

    final contents = <Map<String, dynamic>>[];
    for (final msg in history) {
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

    return _callGemini(systemPrompt: systemPrompt, contents: contents);
  }

  static Future<String> _callGemini({
    required String systemPrompt,
    required List<Map<String, dynamic>> contents,
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
            'safetySettings': [
              {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_ONLY_HIGH'},
              {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH'},
            ],
          },
        );

        final data = jsonDecode(res.body);
        final text = data['candidates']?[0]['content']?['parts']?[0]['text'] as String?;
        return text?.trim() ?? '⚠️ AI response was empty.';
      } catch (e) {
        lastError = e.toString();
        // Catch 429 status code or our custom RATE_LIMIT string
        if (lastError.contains('RATE_LIMIT') || lastError.contains('429') || lastError.contains('RESOURCE_EXHAUSTED')) {
          debugPrint('⏳ Model $model hit rate limit. Trying fallback...');
          continue;
        }
        // If it's not a rate limit error, rethrow immediately
        rethrow;
      }
    }

    throw Exception(lastError ?? 'All models failed to respond.');
  }

  // ── HELPERS ───────────────────────────────────────────────

  static List<Map<String, dynamic>> _buildContents(
      List<MessageModel> history, String newMessage) {
    final contents = <Map<String, dynamic>>[];
    final recentHistory = history.length > 30 ? history.sublist(history.length - 30) : history;

    for (final m in recentHistory) {
      if (!m.isUser && !m.isAssistant) continue;
      contents.add({
        'role': m.isUser ? 'user' : 'model',
        'parts': [
          {'text': m.content.trim()}
        ]
      });
    }
    contents.add({
      'role': 'user',
      'parts': [
        {'text': newMessage}
      ]
    });
    return contents;
  }

  static Future<Map<String, String>?> extractStudentDetails(String message) async {
    final modelsToTry = [kGeminiChatModel, ...kGeminiFallbacks];
    
    for (final model in modelsToTry) {
      try {
        final res = await _invokeFunction(
          'generateContent',
          model,
          {
            'contents': [
              {
                'parts': [
                  {
                    'text': 'Extract student details from: "$message"\n'
                        'Return JSON: {"name":"","program":"","branch":"","semester":""}'
                  }
                ]
              }
            ],
          },
        );
        final data = jsonDecode(res.body);
        final text = data['candidates'][0]['content']['parts'][0]['text'] as String;
        final clean = text.replaceAll('```json', '').replaceAll('```', '').trim();
        final det = jsonDecode(clean) as Map<String, dynamic>;
        return det.map((k, v) => MapEntry(k, v.toString()));
      } catch (e) {
        final errStr = e.toString();
        if (errStr.contains('RATE_LIMIT') || errStr.contains('429') || errStr.contains('RESOURCE_EXHAUSTED')) {
          debugPrint('⏳ Extraction: Model $model hit rate limit. Trying fallback...');
          continue;
        }
        return null;
      }
    }
    return null;
  }

  static String generateGreeting() => kFirstMessage;

  static String friendlyError(String e) {
    if (e.contains('RATE_LIMIT')) return '⏳ **AI is busy.** Please wait a minute and try again.';
    if (e.contains('NETWORK_ERROR')) return '🌐 **Network issue.** Check your connection.';
    if (e.contains('NOT_FOUND')) return '🚀 **Backend not ready.** Function "chat" is not deployed yet.';
    if (e.contains('CONFIG_ERROR')) return '🔑 **Config Error.** Gemini API keys are missing in Supabase secrets.';
    
    // Fallback: show the actual error for easier debugging
    return '❌ **Something went wrong.**\n\nDetail: ${e.replaceAll('Exception:', '').trim()}';
  }
}

// Simple wrapper to match expected behavior
class http_Response_Mock {
  final int statusCode;
  final String body;
  http_Response_Mock(this.statusCode, this.body);
}