// lib/services/supabase_service.dart
import 'dart:convert';
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
      if (ex.message.contains('permission') || ex.message.contains('RLS')) {
        throw Exception('RLS_BLOCKED: Run supabase_schema.sql');
      }
      throw Exception('DB error: ${ex.message}');
    }
  }

  static Future<UserModel> register({
    required String email, required String password, required String name,
    required String role, String? program, String? branch,
    String? semester, String? section, String? mentorEmail, String? rollNumber,
    String? department, String? designation, String? phone,
  }) async {
    final e = email.trim().toLowerCase();
    try {
      if (role == 'student' && mentorEmail != null && mentorEmail.trim().isNotEmpty) {
        final m = await _db.from(kUsersTable).select('id')
            .eq('email', mentorEmail.trim().toLowerCase()).eq('role', 'mentor');
        if (m.isEmpty) {
          throw Exception(
            'invalid_mentor: No mentor with email "${mentorEmail.trim()}"');
        }
      }
      final exists = await _db.from(kUsersTable).select('id').eq('email', e);
      if (exists.isNotEmpty) throw Exception('duplicate_email: Already registered.');
      final insertData = <String, dynamic>{
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
      };
      if (role == 'student' && section?.trim().isNotEmpty == true) {
        insertData['section'] = section!.trim();
      }
      final row = await _db.from(kUsersTable).insert(insertData).select().single();
      return UserModel.fromMap(row);
    } on PostgrestException catch (ex) {
      if (ex.code == '23505') throw Exception('duplicate_email: Already registered.');
      throw Exception('Register failed: ${ex.message}');
    } catch (ex) {
      if (ex.toString().contains('duplicate_email') ||
          ex.toString().contains('invalid_mentor')) {
        rethrow;
      }
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
    String studentId, {
    String? mentorEmail,
    String? studentName,
    String? studentRollNo,
    String? studentProgram,
    String? studentBranch,
    String? studentSemester,
    bool? studentDetailsCollected,
  }) async {
    final hasDetails = studentDetailsCollected ??
        (studentName != null && studentName.isNotEmpty);
    final row = await _db.from(kConversationsTable).insert({
      'student_id': studentId,
      'title': 'New Conversation',
      'is_first_message_done': false,
      'student_details_collected': hasDetails,
      'status': 'active',
      if (mentorEmail != null) 'mentor_email': mentorEmail.toLowerCase(),
      if (studentName != null) 'student_name': studentName,
      if (studentRollNo != null) 'student_roll_no': studentRollNo,
      if (studentProgram != null) 'student_program': studentProgram,
      if (studentBranch != null) 'student_branch': studentBranch,
      if (studentSemester != null) 'student_semester': studentSemester,
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
  // ══════════════════════════════════════════════════════════
  // ACADEMIC DOCUMENTS (MENTOR PUBLISHED)
  // ══════════════════════════════════════════════════════════

  static Future<StudentDocument> uploadDocument({
    String? studentId,
    String? uploadedBy,
    required String docType,
    required String title,
    required String fileName,
    required String mimeType,
    required int fileSize,
    String? contentBase64,
    Uint8List? rawBytes,
    String? storagePath,
    String? extractedText,
    String targetScope = 'class',
    String? targetRollNo,
    String? program,
    String? branch,
    String? semester,
    String academicYear = '2025-2026',
  }) async {
    final uploader = uploadedBy ?? studentId;
    final termVal = (semester != null && (semester.toLowerCase() == 'odd' || semester.toLowerCase() == 'even'))
        ? semester.toLowerCase()
        : 'odd';

    // Single-instance replacement for class-wide documents in the same academic year & term
    if (targetScope == 'class' && uploader != null) {
      try {
        final existing = await _db.from(kDocumentsTable)
            .select('id')
            .eq('uploaded_by', uploader)
            .eq('doc_type', docType)
            .eq('target_scope', 'class')
            .eq('academic_year', academicYear)
            .eq('semester', termVal);
        for (final oldRow in (existing as List)) {
          final oldId = oldRow['id'] as String;
          await deleteDocument(oldId);
          debugPrint('🔄 Replaced previous $docType document ($oldId) for $academicYear $termVal');
        }
      } catch (e) {
        debugPrint('Notice during old document replacement: $e');
      }
    }

    String? finalStoragePath = storagePath;
    String? safeBase64 = contentBase64;

    // If file is > 1.5MB or rawBytes is provided without base64, upload directly to Supabase Storage Bucket
    if (rawBytes != null && (fileSize > 1.5 * 1024 * 1024 || safeBase64 == null)) {
      try {
        final cleanFileName = fileName.replaceAll(RegExp(r'[^a-zA-Z0-9._-]'), '_');
        final remotePath = '${docType}_${DateTime.now().millisecondsSinceEpoch}_$cleanFileName';

        await _db.storage.from('academic_documents').uploadBinary(
          remotePath,
          rawBytes,
          fileOptions: FileOptions(
            contentType: mimeType,
            upsert: true,
          ),
        );
        finalStoragePath = remotePath;
        safeBase64 = null; // Don't bloat PostgreSQL text column!
        debugPrint('📦 Uploaded ${(fileSize / (1024 * 1024)).toStringAsFixed(2)}MB file directly to Supabase Storage: $remotePath');
      } catch (storageErr) {
        debugPrint('⚠️ Storage bucket notice (will fallback): $storageErr');
      }
    }

    // Insert lightweight metadata row into academic_documents (PostgreSQL statement executes in ~20ms)
    final row = await _db.from(kDocumentsTable).insert({
      if (uploader != null) 'uploaded_by': uploader,
      'doc_type': docType,
      'title': title,
      'file_name': fileName,
      'mime_type': mimeType,
      'file_size': fileSize,
      if (safeBase64 != null && safeBase64.isNotEmpty) 'content_base64': safeBase64,
      if (finalStoragePath != null) 'storage_path': finalStoragePath,
      if (extractedText != null) 'extracted_text': extractedText,
      'target_scope': targetScope,
      if (targetRollNo != null && targetRollNo.isNotEmpty)
        'target_roll_no': targetRollNo,
      if (program != null && program.isNotEmpty) 'program': program,
      if (branch != null && branch.isNotEmpty) 'branch': branch,
      'semester': termVal,
      'academic_year': academicYear,
    }).select().single();

    // Ensure shadow record in student_documents to satisfy legacy foreign keys on chunk tables
    try {
      await _db.from('student_documents').upsert({
        'id': row['id'],
        'student_id': uploader,
        'doc_type': docType,
        'title': title,
        'file_name': fileName,
        'mime_type': mimeType,
      }).catchError((_) => null);
    } catch (_) {}

    return StudentDocument.fromMap(row);
  }

  /// Loads raw document binary bytes from either storage bucket or base64
  static Future<Uint8List?> getDocumentBytes(StudentDocument doc) async {
    if (doc.storagePath != null && doc.storagePath!.isNotEmpty) {
      try {
        final bytes = await _db.storage.from('academic_documents').download(doc.storagePath!);
        return bytes;
      } catch (e) {
        debugPrint('⚠️ Error downloading document bytes from storage: $e');
      }
    }
    if (doc.contentBase64 != null && doc.contentBase64!.isNotEmpty) {
      try {
        return base64Decode(doc.contentBase64!);
      } catch (_) {}
    }
    return null;
  }

  static Future<void> updateDocumentExtractedText(
      String docId, String text) async {
    await _db.from(kDocumentsTable)
        .update({'extracted_text': text}).eq('id', docId);
  }

  /// Get all documents uploaded by a specific mentor
  static Future<List<StudentDocument>> getMentorDocuments(String mentorId) async {
    try {
      final rows = await _db.from(kDocumentsTable).select()
          .eq('uploaded_by', mentorId)
          .order('created_at', ascending: false);
      return (rows as List).map((e) {
        final m = Map<String, dynamic>.from(e);
        m.remove('content_base64');
        return StudentDocument.fromMap(m);
      }).toList();
    } catch (_) { return []; }
  }

  /// Get verified academic documents available to a student (class scope + personal)
  static Future<List<StudentDocument>> getStudentAccessibleDocuments({
    required String studentId,
    String? rollNo,
    String? program,
    String? branch,
  }) async {
    try {
      var query = _db.from(kDocumentsTable).select();
      if (rollNo != null && rollNo.isNotEmpty) {
        query = query.or('target_scope.eq.class,target_roll_no.eq.$rollNo');
      } else {
        query = query.eq('target_scope', 'class');
      }
      final rows = await query.order('created_at', ascending: false);
      return (rows as List).map((e) {
        final m = Map<String, dynamic>.from(e);
        m.remove('content_base64');
        return StudentDocument.fromMap(m);
      }).toList();
    } catch (_) { return []; }
  }

  /// Deprecated alias pointing to getStudentAccessibleDocuments
  static Future<List<StudentDocument>> getStudentDocuments(String studentId) async {
    return getStudentAccessibleDocuments(studentId: studentId);
  }

  static Future<StudentDocument?> getDocumentWithContent(String docId) async {
    try {
      final row = await _db.from(kDocumentsTable).select().eq('id', docId).single();
      return StudentDocument.fromMap(row);
    } catch (_) { return null; }
  }

  static Future<void> deleteDocument(String docId) async {
    // Clean up from all 6 dedicated category tables as well as legacy tables
    await Future.wait([
      _db.from('marksheet_chunks').delete().eq('document_id', docId).catchError((_) => null),
      _db.from('attendance_chunks').delete().eq('document_id', docId).catchError((_) => null),
      _db.from('syllabus_chunks').delete().eq('document_id', docId).catchError((_) => null),
      _db.from('calendar_chunks').delete().eq('document_id', docId).catchError((_) => null),
      _db.from('assignment_chunks').delete().eq('document_id', docId).catchError((_) => null),
      _db.from('circular_chunks').delete().eq('document_id', docId).catchError((_) => null),
      _db.from(kChunksTable).delete().eq('document_id', docId).catchError((_) => null),
    ]);
    await _db.from(kDocumentsTable).delete().eq('id', docId);
  }

  /// Atomically appends a page's Native JSON into student_documents.extracted_json
  static Future<void> appendDocumentPageJson({
    required String docId,
    required int pageNumber,
    required Map<String, dynamic> pageJson,
    int? totalPages,
  }) async {
    try {
      await _db.rpc('append_document_page_json', params: {
        'p_doc_id': docId,
        'p_page_number': pageNumber,
        'p_page_json': pageJson,
      });

      if (totalPages != null && totalPages > 0) {
        await _db.from(kDocumentsTable).update({
          'ocr_progress': {
            'current': pageNumber,
            'total': totalPages,
          }
        }).eq('id', docId);
      }
    } catch (e) {
      debugPrint('❌ appendDocumentPageJson error: $e');
      rethrow;
    }
  }

  /// Updates document OCR lifecycle state and progress metrics
  static Future<void> updateDocumentOcrStatus({
    required String docId,
    required String status, // 'pending', 'processing', 'completed', 'paused', 'failed'
    Map<String, dynamic>? progress,
  }) async {
    try {
      final updateData = <String, dynamic>{
        'ocr_status': status,
        'updated_at': DateTime.now().toIso8601String(),
      };
      if (progress != null) {
        updateData['ocr_progress'] = progress;
      }
      await _db.from(kDocumentsTable).update(updateData).eq('id', docId);
    } catch (e) {
      debugPrint('⚠️ updateDocumentOcrStatus error: $e');
    }
  }

  // ══════════════════════════════════════════════════════════
  // RAG — 6-CATEGORY DOMAIN-PARTITIONED CHUNKS & VECTOR SEARCH
  // ══════════════════════════════════════════════════════════

  static String _resolveCategoryTable(String docType) {
    switch (docType.toLowerCase()) {
      case 'marksheet':
      case 'result':
      case 'academic_results':
        return 'marksheet_chunks';
      case 'attendance':
      case 'attendance_register':
        return 'attendance_chunks';
      case 'syllabus':
      case 'curriculum':
        return 'syllabus_chunks';
      case 'academic_calendar':
      case 'calendar':
      case 'datesheet':
        return 'calendar_chunks';
      case 'assignment':
      case 'project':
        return 'assignment_chunks';
      case 'circular':
      case 'notice':
      case 'other':
      case 'others':
      default:
        return 'circular_chunks';
    }
  }

  static Future<void> saveDocumentChunks({
    required String documentId,
    String? docType,
    String? studentId,
    String academicYear = '2026-2027',
    String semester = 'odd',
    String targetScope = 'class',
    String? targetRollNo,
    String? subjectCode,
    String? subjectName,
    required List<String> chunks,
    required List<List<double>> embeddings,
  }) async {
    final targetCategoryTable = _resolveCategoryTable(docType ?? 'other');
    debugPrint('💾 Saving ${chunks.length} chunks into [$targetCategoryTable] & [$kChunksTable] for doc $documentId (Year: $academicYear, Term: $semester, Scope: $targetScope)');
    
    // Ensure shadow record in student_documents exists to satisfy legacy foreign keys
    try {
      await _db.from('student_documents').upsert({
        'id': documentId,
        'student_id': studentId,
        'doc_type': docType ?? 'other',
        'title': 'Document $documentId',
        'file_name': 'document.pdf',
        'mime_type': 'application/pdf',
      }).catchError((_) => null);
    } catch (_) {}

    const int batchSize = 20;
    for (int i = 0; i < chunks.length; i += batchSize) {
      final List<Map<String, dynamic>> categoryRows = [];
      final List<Map<String, dynamic>> legacyRows = [];
      final int end = (i + batchSize < chunks.length) ? i + batchSize : chunks.length;
      
      for (int j = i; j < end; j++) {
        final emb = embeddings[j];
        final safeEmbedding = emb.length > 768 ? emb.sublist(0, 768) : emb;

        // Specialized category record
        final Map<String, dynamic> row = {
          'document_id': documentId,
          'academic_year': academicYear,
          'term': semester,
          'chunk_text': chunks[j],
          'chunk_index': j,
          'embedding': safeEmbedding,
        };

        if (targetCategoryTable == 'marksheet_chunks') {
          row['target_scope'] = targetScope;
          if (targetRollNo != null && targetRollNo.isNotEmpty) {
            row['target_roll_no'] = targetRollNo;
          }
        } else if (targetCategoryTable == 'attendance_chunks') {
          row['target_scope'] = targetScope;
          if (targetRollNo != null && targetRollNo.isNotEmpty) {
            row['target_roll_no'] = targetRollNo;
          }
          if (subjectCode != null) row['subject_code'] = subjectCode;
        } else if (targetCategoryTable == 'syllabus_chunks') {
          if (subjectCode != null) row['subject_code'] = subjectCode;
          if (subjectName != null) row['subject_name'] = subjectName;
        } else if (targetCategoryTable == 'assignment_chunks') {
          if (subjectCode != null) row['subject_code'] = subjectCode;
        }

        categoryRows.add(row);

        // Legacy baseline row
        legacyRows.add({
          'document_id': documentId,
          'chunk_text': chunks[j],
          'chunk_index': j,
          'embedding': safeEmbedding,
        });
      }
      
      try {
        // 1. Insert into dedicated category table
        await _db.from(targetCategoryTable).insert(categoryRows);
        // 2. Insert into unified legacy table for full redundancy
        await _db.from(kChunksTable).insert(legacyRows).catchError((_) => null);

        debugPrint('... saved batch ${i ~/ batchSize + 1} (${categoryRows.length} chunks) to $targetCategoryTable');
        
        if (i + batchSize < chunks.length) {
          await Future.delayed(const Duration(milliseconds: 300));
        }
      } catch (e) {
        debugPrint('❌ saveDocumentChunks error on $targetCategoryTable: $e');
        rethrow;
      }
    }
    
    debugPrint('✅ Successfully saved ${chunks.length} chunks to category table [$targetCategoryTable]');
  }

  /// High-precision Category-Partitioned similarity search with temporal and scope pre-filtering
  static Future<List<String>> searchSimilarChunks({
    String? studentId,
    String? rollNo,
    String? docType,
    String? academicYear,
    String? term,
    String? targetScope,
    required List<double> queryEmbedding,
    int limit = 8,
    double minSimilarity = 0.20,
  }) async {
    try {
      final safeQueryEmb = queryEmbedding.length > 768
          ? queryEmbedding.sublist(0, 768)
          : queryEmbedding;

      final categoryName = docType != null && docType.isNotEmpty ? docType : 'all';

      // 1. Call match_category_chunks RPC with strict domain filtering
      try {
        final res = await _db.rpc('match_category_chunks', params: {
          'category_name': categoryName,
          'query_embedding': safeQueryEmb,
          if (academicYear != null && academicYear.isNotEmpty) 'filter_year': academicYear,
          if (term != null && term.isNotEmpty) 'filter_term': term,
          if (targetScope != null && targetScope.isNotEmpty) 'filter_scope': targetScope,
          if (rollNo != null && rollNo.isNotEmpty) 'filter_roll_no': rollNo,
          'match_count': limit,
          'min_similarity': minSimilarity,
        });

        final chunks = (res as List)
            .map((r) => r['chunk_text'] as String)
            .where((t) => t.trim().isNotEmpty)
            .toList();

        if (chunks.isNotEmpty) {
          debugPrint('✅ Category RAG ($categoryName, Year: $academicYear, Term: $term): Found ${chunks.length} chunks');
          return chunks;
        }
      } catch (e) {
        debugPrint('⚠️ match_category_chunks notice: $e');
      }

      // 2. Fallback to match_document_chunks if category search was empty
      final resOld = await _db.rpc('match_document_chunks', params: {
        'query_embedding': safeQueryEmb,
        if (studentId != null && studentId.isNotEmpty) 'match_student_id': studentId,
        'match_count': limit,
        'min_similarity': minSimilarity,
      });

      final chunks = (resOld as List)
          .map((r) => r['chunk_text'] as String)
          .where((t) => t.trim().isNotEmpty)
          .toList();

      return chunks;
    } catch (e) {
      debugPrint('⚠️ RAG search failed: $e');
      return [];
    }
  }

  static Future<void> deleteDocumentChunks(String documentId) async {
    await Future.wait([
      _db.from('marksheet_chunks').delete().eq('document_id', documentId).catchError((_) => null),
      _db.from('attendance_chunks').delete().eq('document_id', documentId).catchError((_) => null),
      _db.from('syllabus_chunks').delete().eq('document_id', documentId).catchError((_) => null),
      _db.from('calendar_chunks').delete().eq('document_id', documentId).catchError((_) => null),
      _db.from('assignment_chunks').delete().eq('document_id', documentId).catchError((_) => null),
      _db.from('circular_chunks').delete().eq('document_id', documentId).catchError((_) => null),
      _db.from(kChunksTable).delete().eq('document_id', documentId).catchError((_) => null),
    ]);
  }

  // ══════════════════════════════════════════════════════════
  // ISSUE REPORTS
  // ══════════════════════════════════════════════════════════

  static Future<IssueReport> submitIssue({
    required String studentId, required String? studentName,
    required String? studentEmail, required String? studentProgram,
    required String? studentBranch, required String? studentSemester,
    required String? mentorEmail, required String category,
    required String title, required String description, required String priority,
  }) async {
    final row = await _db.from(kIssuesTable).insert({
      'student_id': studentId, 'student_name': studentName,
      'student_email': studentEmail, 'student_program': studentProgram,
      'student_branch': studentBranch, 'student_semester': studentSemester,
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
    for (final c in convs) {
      totalMsg += await getMessageCount(c.id);
    }
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

      // Resolve student's roll number if not directly supplied
      String? resolvedRollNo = rollNo?.trim();
      if ((resolvedRollNo == null || resolvedRollNo.isEmpty) && studentId.isNotEmpty) {
        try {
          final userRow = await _db.from(kUsersTable).select('roll_number').eq('id', studentId).maybeSingle();
          if (userRow != null && userRow['roll_number'] != null) {
            resolvedRollNo = userRow['roll_number'].toString().trim();
          }
        } catch (_) {}
      }

      if (resolvedRollNo == null || resolvedRollNo.isEmpty) {
        return [];
      }

      // 1. Fetch from Attendance table (linked by student_roll_no)
      final rows = await _db.from(kAcademicRecordsTable).select()
          .ilike('student_roll_no', resolvedRollNo);
      if (rows.isNotEmpty) {
        allRecords.addAll(List<Map<String, dynamic>>.from(rows).map((r) => {...r, 'record_type': 'attendance'}));
      }

      // 2. Fetch from Academic Results table (linked by student_roll_no)
      final resultRows = await _db.from(kAcademicResultsTable).select()
          .ilike('student_roll_no', resolvedRollNo);
      if (resultRows.isNotEmpty) {
        allRecords.addAll(List<Map<String, dynamic>>.from(resultRows).map((r) => {...r, 'record_type': 'result'}));
      }

      return allRecords;
    } catch (e) {
      debugPrint('⚠️ getAcademicRecord failed: $e');
      return [];
    }
  }

  static Future<List<dynamic>> getTimetable(String studentId, String day, {String? rollNo}) async {
    try {
      String? resolvedRollNo = rollNo?.trim();
      if ((resolvedRollNo == null || resolvedRollNo.isEmpty) && studentId.isNotEmpty) {
        try {
          final userRow = await _db.from(kUsersTable).select('roll_number').eq('id', studentId).maybeSingle();
          if (userRow != null && userRow['roll_number'] != null) {
            resolvedRollNo = userRow['roll_number'].toString().trim();
          }
        } catch (_) {}
      }

      if (resolvedRollNo == null || resolvedRollNo.isEmpty) return [];

      final rows = await _db.from(kSchedulesTable).select()
          .ilike('student_roll_no', resolvedRollNo)
          .ilike('day_of_week', day);
      return rows;
    } catch (e) {
      debugPrint('⚠️ getTimetable failed: $e');
      return [];
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