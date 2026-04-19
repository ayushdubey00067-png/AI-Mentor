// lib/services/supabase_service.dart
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/models.dart';
import '../utils/constants.dart';

class SupabaseService {
  static final SupabaseClient _db = Supabase.instance.client;

  // ══════════════════════════════════════════════════════════
  // AUTH
  // ══════════════════════════════════════════════════════════

  static Future<UserModel?> login(String email, String password) async {
    final e = email.trim().toLowerCase(), p = password.trim();
    try {
      final rows = await _db.from(kUsersTable).select('id').eq('email', e);
      if (rows.isEmpty) return null;
      final row = await _db.from(kUsersTable).select()
          .eq('email', e).eq('password_hash', p).maybeSingle();
      if (row == null) return null;
      await _db.from(kUsersTable)
          .update({'last_active': DateTime.now().toIso8601String()})
          .eq('id', row['id']);
      return UserModel.fromMap(row);
    } on PostgrestException catch (ex) {
      if (ex.message.contains('permission') || ex.message.contains('RLS'))
        throw Exception('RLS_BLOCKED: Run supabase_schema.sql');
      throw Exception('DB error: ${ex.message}');
    }
  }

  static Future<UserModel> register({
    required String email, required String password, required String name,
    required String role, String? program, String? branch,
    String? semester, String? mentorEmail, String? rollNumber,
    String? department, String? designation, String? phone,
  }) async {
    final e = email.trim().toLowerCase();
    try {
      if (role == 'student' && mentorEmail != null && mentorEmail.trim().isNotEmpty) {
        final m = await _db.from(kUsersTable).select('id')
            .eq('email', mentorEmail.trim().toLowerCase()).eq('role', 'mentor');
        if (m.isEmpty) throw Exception(
            'invalid_mentor: No mentor with email "${mentorEmail.trim()}"');
      }
      final exists = await _db.from(kUsersTable).select('id').eq('email', e);
      if (exists.isNotEmpty) throw Exception('duplicate_email: Already registered.');
      final row = await _db.from(kUsersTable).insert({
        'email': e, 'password_hash': password.trim(),
        'name': name.trim(), 'role': role,
        'program':  (role=='student' && program?.trim().isNotEmpty==true)  ? program!.trim()  : null,
        'branch':   (role=='student' && branch?.trim().isNotEmpty==true)   ? branch!.trim()   : null,
        'semester': (role=='student' && semester?.trim().isNotEmpty==true) ? semester!.trim() : null,
        'mentor_email': (role=='student' && mentorEmail?.trim().isNotEmpty==true)
            ? mentorEmail!.trim().toLowerCase() : null,
        'roll_number': (role=='student' && rollNumber?.trim().isNotEmpty==true) ? rollNumber!.trim() : null,
        'department': department?.trim(),
        'designation': (role=='mentor') ? designation?.trim() : null,
        'phone': phone?.trim(),
      }).select().single();
      return UserModel.fromMap(row);
    } on PostgrestException catch (ex) {
      if (ex.code == '23505') throw Exception('duplicate_email: Already registered.');
      throw Exception('Register failed: ${ex.message}');
    } catch (ex) {
      if (ex.toString().contains('duplicate_email') ||
          ex.toString().contains('invalid_mentor')) rethrow;
      throw Exception('Register error: $ex');
    }
  }

  static Future<List<UserModel>> getAllUsers() async {
    try {
      final rows = await _db.from(kUsersTable).select()
          .order('created_at', ascending: false);
      return (rows as List).map((e) => UserModel.fromMap(e)).toList();
    } catch (_) { return []; }
  }

  static Future<List<UserModel>> getMyStudents(String mentorEmail) async {
    try {
      final rows = await _db.from(kUsersTable).select()
          .eq('role', 'student').eq('mentor_email', mentorEmail.toLowerCase())
          .order('created_at', ascending: false);
      return (rows as List).map((e) => UserModel.fromMap(e)).toList();
    } catch (_) { return []; }
  }

  // ══════════════════════════════════════════════════════════
  // CONVERSATIONS
  // ══════════════════════════════════════════════════════════

  static Future<ConversationModel> createConversation(
      String studentId, {String? mentorEmail}) async {
    final row = await _db.from(kConversationsTable).insert({
      'student_id': studentId, 'title': 'New Conversation',
      'is_first_message_done': false, 'student_details_collected': false,
      'status': 'active',
      if (mentorEmail != null) 'mentor_email': mentorEmail.toLowerCase(),
    }).select().single();
    return ConversationModel.fromMap(row);
  }

  static Future<List<ConversationModel>> getStudentConversations(String studentId) async {
    final rows = await _db.from(kConversationsTable).select()
        .eq('student_id', studentId).order('updated_at', ascending: false);
    return (rows as List).map((e) => ConversationModel.fromMap(e)).toList();
  }

  static Future<List<ConversationModel>> getMentorConversations(String mentorEmail) async {
    try {
      final rows = await _db.from(kConversationsTable).select()
          .eq('mentor_email', mentorEmail.toLowerCase())
          .order('updated_at', ascending: false);
      return (rows as List).map((e) => ConversationModel.fromMap(e)).toList();
    } catch (_) { return []; }
  }

  static Future<List<ConversationModel>> getAllConversations() async {
    final rows = await _db.from(kConversationsTable).select()
        .order('updated_at', ascending: false);
    return (rows as List).map((e) => ConversationModel.fromMap(e)).toList();
  }

  static Future<void> updateConversation(String id, Map<String, dynamic> data) async {
    data['updated_at'] = DateTime.now().toIso8601String();
    await _db.from(kConversationsTable).update(data).eq('id', id);
  }

  static Future<void> markFirstMessageDone(String id) async =>
      updateConversation(id, {'is_first_message_done': true});

  static Future<void> saveStudentDetails({
    required String conversationId, required String name,
    required String program, required String branch, required String semester,
    String? rollNo,
  }) async => updateConversation(conversationId, {
    'student_details_collected': true, 'student_name': name,
    'student_program': program, 'student_branch': branch,
    'student_semester': semester, 
    if (rollNo != null) 'student_roll_no': rollNo,
    'title': '$name - $program',
  });

  static Future<void> updateConversationStatus(String id, String status) async =>
      updateConversation(id, {'status': status});

  static Future<void> deleteConversation(String id) async {
    // Delete all messages first (though DB should have cascade, we ensure it here)
    await _db.from(kMessagesTable).delete().eq('conversation_id', id);
    // Delete the conversation
    await _db.from(kConversationsTable).delete().eq('id', id);
  }

  // ══════════════════════════════════════════════════════════
  // MESSAGES
  // ══════════════════════════════════════════════════════════

  static Future<MessageModel> sendMessage({
    required String conversationId, required String content,
    required String senderRole, String? senderId, bool isAiGenerated = false,
  }) async {
    final row = await _db.from(kMessagesTable).insert({
      'conversation_id': conversationId, 'content': content,
      'sender_role': senderRole, if (senderId != null) 'sender_id': senderId,
      'is_ai_generated': isAiGenerated,
    }).select().single();
    return MessageModel.fromMap(row);
  }

  static Future<List<MessageModel>> getMessages(String conversationId) async {
    final rows = await _db.from(kMessagesTable).select()
        .eq('conversation_id', conversationId).order('created_at', ascending: true);
    return (rows as List).map((e) => MessageModel.fromMap(e)).toList();
  }

  static Future<int> getMessageCount(String conversationId) async {
    try {
      final rows = await _db.from(kMessagesTable).select('id')
          .eq('conversation_id', conversationId).eq('sender_role', 'user');
      return rows.length;
    } catch (_) { return 0; }
  }

  // ══════════════════════════════════════════════════════════
  // STUDENT DOCUMENTS
  // ══════════════════════════════════════════════════════════

  static Future<StudentDocument> uploadDocument({
    required String studentId, required String docType,
    required String title, required String fileName,
    required String mimeType, required int fileSize,
    required String contentBase64, String? extractedText,
  }) async {
    final row = await _db.from(kDocumentsTable).insert({
      'student_id': studentId, 'doc_type': docType, 'title': title,
      'file_name': fileName, 'mime_type': mimeType, 'file_size': fileSize,
      'content_base64': contentBase64,
      if (extractedText != null) 'extracted_text': extractedText,
    }).select().single();
    return StudentDocument.fromMap(row);
  }

  static Future<void> updateDocumentExtractedText(
      String docId, String text) async {
    await _db.from(kDocumentsTable)
        .update({'extracted_text': text}).eq('id', docId);
  }

  static Future<List<StudentDocument>> getStudentDocuments(String studentId) async {
    try {
      final rows = await _db.from(kDocumentsTable).select()
          .eq('student_id', studentId).order('created_at', ascending: false);
      return (rows as List).map((e) {
        final m = Map<String, dynamic>.from(e);
        m.remove('content_base64'); // strip for list
        return StudentDocument.fromMap(m);
      }).toList();
    } catch (_) { return []; }
  }

  static Future<StudentDocument?> getDocumentWithContent(String docId) async {
    try {
      final row = await _db.from(kDocumentsTable).select().eq('id', docId).single();
      return StudentDocument.fromMap(row);
    } catch (_) { return null; }
  }

  static Future<void> deleteDocument(String docId) async {
    // Also delete chunks
    await _db.from(kChunksTable).delete().eq('document_id', docId);
    await _db.from(kDocumentsTable).delete().eq('id', docId);
  }

  // ══════════════════════════════════════════════════════════
  // RAG — DOCUMENT CHUNKS (vector embeddings)
  // ══════════════════════════════════════════════════════════

  /// Save text chunks with their embeddings to Supabase
  static Future<void> saveDocumentChunks({
    required String documentId,
    required String studentId,
    required List<String> chunks,
    required List<List<double>> embeddings,
  }) async {
    debugPrint('💾 Saving ${chunks.length} chunks for doc $documentId (Batch size: 20)');
    
    // Batch the inserts to avoid large payload timeouts
    const int batchSize = 20;
    for (int i = 0; i < chunks.length; i += batchSize) {
      final List<Map<String, dynamic>> batchRows = [];
      final int end = (i + batchSize < chunks.length) ? i + batchSize : chunks.length;
      
      for (int j = i; j < end; j++) {
        batchRows.add({
          'document_id': documentId,
          'student_id':  studentId,
          'chunk_text':  chunks[j],
          'chunk_index': j,
          'embedding':   embeddings[j],
        });
      }
      
      try {
        await _db.from(kChunksTable).insert(batchRows);
        debugPrint('... saved batch ${i ~/ batchSize + 1} (${batchRows.length} chunks)');
        
        // Safety delay to prevent statement timeouts on large files
        if (i + batchSize < chunks.length) {
          await Future.delayed(const Duration(milliseconds: 500));
        }
      } on PostgrestException catch (e) {
        debugPrint('❌ DB Error during chunk insert: ${e.message} (Code: ${e.code})');
        if (e.message.contains('dimension')) {
          throw Exception('DB_DIMENSION_MISMATCH: AI embedding dimension ($kEmbeddingDims) does not match your database table. Run migration if needed.');
        }
        throw Exception('DB_CHUNK_ERROR: Failed to save search index (batch ${i ~/ batchSize + 1}).');
      } catch (e) {
        debugPrint('❌ saveDocumentChunks batch error: $e');
        rethrow;
      }
    }
    
    debugPrint('✅ Saved ${chunks.length} total chunks');
  }

  /// Vector similarity search — find most relevant chunks
  static Future<List<String>> searchSimilarChunks({
    required String studentId,
    required List<double> queryEmbedding,
    int limit = 5,
    double minSimilarity = 0.5,
  }) async {
    try {
      debugPrint('🔍 RAG: Searching chunks for student $studentId');

      final res = await _db.rpc('match_document_chunks', params: {
        'query_embedding':  queryEmbedding,
        'match_student_id': studentId,
        'match_count':      limit,
        'min_similarity':   minSimilarity,
      });

      final chunks = (res as List)
          .map((r) => r['chunk_text'] as String)
          .where((t) => t.trim().isNotEmpty)
          .toList();

      debugPrint('✅ RAG: Found ${chunks.length} relevant chunks');
      return chunks;
    } catch (e) {
      debugPrint('⚠️ RAG search failed (no chunks yet?): $e');
      return [];
    }
  }

  /// Delete all chunks for a document
  static Future<void> deleteDocumentChunks(String documentId) async {
    await _db.from(kChunksTable).delete().eq('document_id', documentId);
  }

  // ══════════════════════════════════════════════════════════
  // ISSUE REPORTS
  // ══════════════════════════════════════════════════════════

  static Future<IssueReport> submitIssue({
    required String studentId, required String? studentName,
    required String? studentEmail, required String? studentProgram,
    required String? studentBranch, required String? studentSemester,
    required String? studentRollNo,
    required String? mentorEmail, required String category,
    required String title, required String description, required String priority,
  }) async {
    final row = await _db.from(kIssuesTable).insert({
      'student_id': studentId, 'student_name': studentName,
      'student_email': studentEmail, 'student_program': studentProgram,
      'student_branch': studentBranch, 'student_semester': studentSemester,
      'student_roll_no': studentRollNo,
      'mentor_email': mentorEmail?.toLowerCase(),
      'category': category, 'title': title,
      'description': description, 'priority': priority, 'status': 'open',
    }).select().single();
    return IssueReport.fromMap(row);
  }

  static Future<List<IssueReport>> getStudentIssues(String studentId) async {
    try {
      final rows = await _db.from(kIssuesTable).select()
          .eq('student_id', studentId).order('created_at', ascending: false);
      return (rows as List).map((e) => IssueReport.fromMap(e)).toList();
    } catch (_) { return []; }
  }

  static Future<List<IssueReport>> getMentorIssues(String mentorEmail) async {
    try {
      final rows = await _db.from(kIssuesTable).select()
          .eq('mentor_email', mentorEmail.toLowerCase())
          .order('created_at', ascending: false);
      return (rows as List).map((e) => IssueReport.fromMap(e)).toList();
    } catch (_) { return []; }
  }

  static Future<void> respondToIssue({
    required String issueId, required String response, required String newStatus,
  }) async {
    await _db.from(kIssuesTable).update({
      'mentor_response': response, 'status': newStatus,
      'mentor_responded_at': DateTime.now().toIso8601String(),
      'updated_at': DateTime.now().toIso8601String(),
    }).eq('id', issueId);
  }

  static Future<void> updateIssueStatus(String issueId, String status) async {
    await _db.from(kIssuesTable).update({
      'status': status, 'updated_at': DateTime.now().toIso8601String(),
    }).eq('id', issueId);
  }

  // ══════════════════════════════════════════════════════════
  // PROGRESS REPORTS
  // ══════════════════════════════════════════════════════════

  static Future<StudentProgressReport> getStudentProgress(UserModel student) async {
    final convs   = await getStudentConversations(student.id);
    final issues  = await getStudentIssues(student.id);
    int totalMsg  = 0;
    for (final c in convs) totalMsg += await getMessageCount(c.id);
    return StudentProgressReport(
      student: student, conversations: convs, issues: issues,
      totalMessages: totalMsg,
      activeConversations:  convs.where((c) => c.status == 'active').length,
      resolvedConversations:convs.where((c) => c.status == 'resolved').length,
      flaggedConversations: convs.where((c) => c.status == 'flagged').length,
      lastActive: convs.isNotEmpty ? convs.first.updatedAt : null,
    );
  }

  // ══════════════════════════════════════════════════════════
  // REALTIME
  // ══════════════════════════════════════════════════════════

  static RealtimeChannel subscribeToMessages(
      String conversationId, Function(MessageModel) onMessage) {
    return _db.channel('messages:$conversationId')
        .onPostgresChanges(
          event: PostgresChangeEvent.insert, schema: 'public',
          table: kMessagesTable,
          filter: PostgresChangeFilter(type: PostgresChangeFilterType.eq,
              column: 'conversation_id', value: conversationId),
          callback: (p) => onMessage(MessageModel.fromMap(p.newRecord)),
        ).subscribe();
  }

  // ══════════════════════════════════════════════════════════
  // MENTOR INTERVENTIONS
  // ══════════════════════════════════════════════════════════

  static Future<void> logMentorIntervention({
    required String conversationId, required String mentorId,
    required String type, String? note,
  }) async {
    await _db.from(kInterventionsTable).insert({
      'conversation_id': conversationId, 'mentor_id': mentorId,
      'intervention_type': type, if (note != null) 'note': note,
    });
  }

  // ══════════════════════════════════════════════════════════
  // ACADEMIC DATA LOOKUPS [NEW]
  // ══════════════════════════════════════════════════════════

  static Future<List<Map<String, dynamic>>> getAcademicRecord(String studentId, {String? rollNo}) async {
    try {
      final List<Map<String, dynamic>> allRecords = [];

      // 1. Fetch from Attendance table
      if (rollNo != null && rollNo.isNotEmpty) {
        final rows = await _db.from(kAcademicRecordsTable).select()
            .ilike('student_roll_no', rollNo.trim());
        if (rows.isNotEmpty) {
          allRecords.addAll(List<Map<String, dynamic>>.from(rows).map((r) => {...r, 'record_type': 'attendance'}));
        }
      }

      // 2. Fetch from Academic Results table (as backup/extra data)
      if (rollNo != null && rollNo.isNotEmpty) {
        final resultRows = await _db.from(kAcademicResultsTable).select()
            .ilike('student_roll_no', rollNo.trim());
        if (resultRows.isNotEmpty) {
          allRecords.addAll(List<Map<String, dynamic>>.from(resultRows).map((r) => {...r, 'record_type': 'result'}));
        }
      }

      // 3. Fallback by student_id if nothing found by roll number
      if (allRecords.isEmpty) {
        final rows = await _db.from(kAcademicRecordsTable).select().eq('student_id', studentId);
        if (rows.isNotEmpty) {
          allRecords.addAll(List<Map<String, dynamic>>.from(rows).map((r) => {...r, 'record_type': 'attendance'}));
        }
      }

      return allRecords;
    } catch (e) {
      debugPrint('⚠️ getAcademicRecord failed: $e');
      return [];
    }
  }

  static Future<List<dynamic>> getTimetable(String studentId, String day) async {
    try {
      final row = await _db.from(kSchedulesTable).select()
          .eq('student_id', studentId).eq('day_of_week', day).maybeSingle();
      if (row == null) return [];
      return row['schedule_json'] as List<dynamic>;
    } catch (_) {
      // Retry without specific day to see if there is any timetable at all
      try {
        final all = await _db.from(kSchedulesTable).select('day_of_week')
            .eq('student_id', studentId);
        return all; // Returns list of available days
      } catch (__) { return []; }
    }
  }

  static Future<UserModel?> findStudentByQuery(String query, {String? mentorEmail}) async {
    try {
      final q = query.trim().toLowerCase();
      // Search by Email, Name, or Roll Number
      var builder = _db.from(kUsersTable).select()
          .eq('role', 'student')
          .or('email.ilike.%$q%,name.ilike.%$q%,roll_number.ilike.%$q%');
      
      if (mentorEmail != null) {
        builder = builder.eq('mentor_email', mentorEmail.toLowerCase());
      }
      
      final rows = await builder.limit(1).maybeSingle();
      if (rows == null) return null;
      return UserModel.fromMap(rows);
    } catch (e) {
      debugPrint('⚠️ findStudentByQuery failed: $e');
      return null;
    }
  }

  static Future<void> updateUserProfile(UserModel user) async {
    await _db.from(kUsersTable).update(user.toMap()).eq('id', user.id);
  }
}