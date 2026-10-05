// lib/utils/ai_config.dart
import '../models/models.dart';

/// Centralized AI Configuration Hub for Acadly
/// Manages model routing, sliding context windows, token budgeting,
/// and persona generation for Students and Mentors.
class AIConfig {
  // ── MODELS ──────────────────────────────────────────────────
  static const String primaryChatModel = 'gemini-3.5-flash';
  static const List<String> fallbackChatModels = [
    'gemini-3.1-flash-lite',
    'gemini-flash-latest',
    'gemini-3.8-flash',
  ];
  static const String visionModel = 'gemini-3.1-flash-lite';
  static const String embedModel = 'gemini-embedding-001';
  static const int embeddingDims = 768;

  // ── SLIDING CONTEXT WINDOW LIMITS ──────────────────────────
  /// Max historical turns (user + assistant pairs) retained in active prompt.
  /// 10 messages = 5 full conversational turns.
  static const int maxHistoryTurns = 10;
  
  /// Generation safety budgets
  static const int maxOutputTokens = 4096;
  static const double studentTemperature = 0.6; // Supportive & natural
  static const double mentorTemperature = 0.2;  // Strict, analytical & data-driven

  // ── SUPABASE TABLES ─────────────────────────────────────────
  static const String usersTable = 'users';
  static const String conversationsTable = 'conversations';
  static const String messagesTable = 'messages';
  static const String interventionsTable = 'mentor_interventions';
  static const String issuesTable = 'issue_reports';
  static const String academicDocsTable = 'academic_documents';
  static const String chunksTable = 'document_chunks';
  static const String attendanceTable = 'attendance';
  static const String resultsTable = 'academic_results';
  static const String schedulesTable = 'schedules';
}

/// Token-efficient Sliding Context Window Manager
class ContextWindowManager {
  /// Builds optimized Gemini contents array applying the sliding window
  static List<Map<String, dynamic>> buildSlidingWindow({
    required List<MessageModel> rawHistory,
    required String newMessage,
    int maxTurns = AIConfig.maxHistoryTurns,
  }) {
    final contents = <Map<String, dynamic>>[];

    // Extract the most recent conversation slice
    final recentHistory = rawHistory.length > maxTurns
        ? rawHistory.sublist(rawHistory.length - maxTurns)
        : rawHistory;

    for (final m in recentHistory) {
      if (!m.isUser && !m.isAssistant) continue;
      final text = m.content.trim();
      if (text.isEmpty) continue;

      contents.add({
        'role': m.isUser ? 'user' : 'model',
        'parts': [{'text': text}]
      });
    }

    // Append the active user query
    contents.add({
      'role': 'user',
      'parts': [{'text': newMessage.trim()}]
    });

    return contents;
  }
}
