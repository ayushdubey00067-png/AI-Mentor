// lib/models/models.dart

class UserModel {
  final String id;
  final String email;
  final String name;
  final String role;
  final String? program;
  final String? branch;
  final String? semester;
  final String? mentorEmail;
  final String? rollNumber;
  final String? phone;
  final String? department;
  final List<String>? skills;
  final List<String>? hobbies;
  final List<String>? careerInterests;
  final String? designation;
  final List<String>? expertise;
  final String? officeLocation;
  final String? officeHours;
  final String? profileUrl;
  final DateTime createdAt;
  final DateTime lastActive;

  UserModel({
    required this.id,
    required this.email,
    required this.name,
    required this.role,
    this.program,
    this.branch,
    this.semester,
    this.mentorEmail,
    this.rollNumber,
    this.phone,
    this.department,
    this.skills,
    this.hobbies,
    this.careerInterests,
    this.designation,
    this.expertise,
    this.officeLocation,
    this.officeHours,
    this.profileUrl,
    required this.createdAt,
    required this.lastActive,
  });

  bool get isMentor => role == 'mentor';
  bool get isStudent => role == 'student';

  factory UserModel.fromMap(Map<String, dynamic> map) {
    return UserModel(
      id: map['id'] ?? '',
      email: map['email'] ?? '',
      name: map['name'] ?? '',
      role: map['role'] ?? 'student',
      program: map['program'],
      branch: map['branch'],
      semester: map['semester'],
      mentorEmail: map['mentor_email'],
      rollNumber: map['roll_number'],
      phone: map['phone'],
      department: map['department'],
      skills: map['skills'] != null ? List<String>.from(map['skills']) : null,
      hobbies: map['hobbies'] != null ? List<String>.from(map['hobbies']) : null,
      careerInterests: map['career_interests'] != null ? List<String>.from(map['career_interests']) : null,
      designation: map['designation'],
      expertise: map['expertise'] != null ? List<String>.from(map['expertise']) : null,
      officeLocation: map['office_location'],
      officeHours: map['office_hours'],
      profileUrl: map['profile_url'],
      createdAt: DateTime.parse(map['created_at'] ?? DateTime.now().toIso8601String()),
      lastActive: DateTime.parse(map['last_active'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'email': email,
      'name': name,
      'role': role,
      'program': program,
      'branch': branch,
      'semester': semester,
      'mentor_email': mentorEmail,
      'roll_number': rollNumber,
      'phone': phone,
      'department': department,
      'skills': skills,
      'hobbies': hobbies,
      'career_interests': careerInterests,
      'designation': designation,
      'expertise': expertise,
      'office_location': officeLocation,
      'office_hours': officeHours,
      'profile_url': profileUrl,
      'created_at': createdAt.toIso8601String(),
      'last_active': lastActive.toIso8601String(),
    };
  }
}

class ConversationModel {
  final String id;
  final String studentId;
  final String title;
  final bool isFirstMessageDone;
  final bool studentDetailsCollected;
  final String? studentName;
  final String? studentProgram;
  final String? studentBranch;
  final String? studentSemester;
  final String? studentRollNo;
  final String? studentDept;
  final List<String>? studentSkills;
  final List<String>? studentInterests;
  final String? mentorEmail;
  final String status;
  final DateTime createdAt;
  final DateTime updatedAt;

  ConversationModel({
    required this.id,
    required this.studentId,
    required this.title,
    required this.isFirstMessageDone,
    required this.studentDetailsCollected,
    this.studentName,
    this.studentProgram,
    this.studentBranch,
    this.studentSemester,
    this.studentRollNo,
    this.studentDept,
    this.studentSkills,
    this.studentInterests,
    this.mentorEmail,
    required this.status,
    required this.createdAt,
    required this.updatedAt,
  });

  factory ConversationModel.fromMap(Map<String, dynamic> map) {
    return ConversationModel(
      id: map['id'] ?? '',
      studentId: map['student_id'] ?? '',
      title: map['title'] ?? 'New Conversation',
      isFirstMessageDone: map['is_first_message_done'] ?? false,
      studentDetailsCollected: map['student_details_collected'] ?? false,
      studentName: map['student_name'],
      studentProgram: map['student_program'],
      studentBranch: map['student_branch'],
      studentSemester: map['student_semester'],
      studentRollNo: map['student_roll_no'],
      studentDept: map['student_dept'],
      studentSkills: map['student_skills'] != null ? List<String>.from(map['student_skills']) : null,
      studentInterests: map['student_interests'] != null ? List<String>.from(map['student_interests']) : null,
      mentorEmail: map['mentor_email'],
      status: map['status'] ?? 'active',
      createdAt: DateTime.parse(map['created_at'] ?? DateTime.now().toIso8601String()),
      updatedAt: DateTime.parse(map['updated_at'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'student_id': studentId,
      'title': title,
      'is_first_message_done': isFirstMessageDone,
      'student_details_collected': studentDetailsCollected,
      'student_name': studentName,
      'student_program': studentProgram,
      'student_branch': studentBranch,
      'student_semester': studentSemester,
      'student_roll_no': studentRollNo,
      'student_dept': studentDept,
      'student_skills': studentSkills,
      'student_interests': studentInterests,
      'mentor_email': mentorEmail,
      'status': status,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }

  bool get isResolved => status == 'resolved';
  bool get isActive   => status == 'active';
  bool get isFlagged  => status == 'flagged';
}

class MessageModel {
  final String id;
  final String conversationId;
  final String senderRole;
  final String? senderId;
  final String content;
  final bool isAiGenerated;
  final DateTime createdAt;

  MessageModel({
    required this.id,
    required this.conversationId,
    required this.senderRole,
    this.senderId,
    required this.content,
    this.isAiGenerated = false,
    required this.createdAt,
  });

  factory MessageModel.fromMap(Map<String, dynamic> map) {
    return MessageModel(
      id: map['id'] ?? '',
      conversationId: map['conversation_id'] ?? '',
      senderRole: map['sender_role'] ?? 'user',
      senderId: map['sender_id'],
      content: map['content'] ?? '',
      isAiGenerated: map['is_ai_generated'] ?? false,
      createdAt: DateTime.parse(map['created_at'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'conversation_id': conversationId,
      'sender_role': senderRole,
      'sender_id': senderId,
      'content': content,
      'is_ai_generated': isAiGenerated,
      'created_at': createdAt.toIso8601String(),
    };
  }

  bool get isUser => senderRole == 'user';
  bool get isAssistant => senderRole == 'assistant' || isAiGenerated;
  bool get isMentor => senderRole == 'mentor';
}

class StudentDocument {
  final String id;
  final String studentId;
  final String docType;
  final String title;
  final String fileName;
  final String mimeType;
  final int? fileSize;
  final String? contentBase64;
  final String? extractedText;
  final String? storagePath;
  final DateTime createdAt;

  StudentDocument({
    required this.id,
    required this.studentId,
    required this.docType,
    required this.title,
    required this.fileName,
    required this.mimeType,
    this.fileSize,
    this.contentBase64,
    this.extractedText,
    this.storagePath,
    required this.createdAt,
  });

  factory StudentDocument.fromMap(Map<String, dynamic> map) {
    return StudentDocument(
      id: map['id'] ?? '',
      studentId: map['student_id'] ?? '',
      docType: map['doc_type'] ?? 'other',
      title: map['title'] ?? '',
      fileName: map['file_name'] ?? '',
      mimeType: map['mime_type'] ?? '',
      fileSize: map['file_size'],
      contentBase64: map['content_base64'],
      extractedText: map['extracted_text'],
      storagePath: map['storage_path'],
      createdAt: DateTime.parse(map['created_at'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'student_id': studentId,
      'doc_type': docType,
      'title': title,
      'file_name': fileName,
      'mime_type': mimeType,
      'file_size': fileSize,
      'content_base64': contentBase64,
      'extracted_text': extractedText,
      'storage_path': storagePath,
      'created_at': createdAt.toIso8601String(),
    };
  }

  static String typeLabel(String type) {
    switch (type) {
      case 'timetable': return 'Timetable';
      case 'academic_calendar': return 'Academic Calendar';
      case 'syllabus': return 'Syllabus';
      case 'marksheet': return 'Marksheet';
      case 'attendance': return 'Attendance';
      case 'assignment': return 'Assignment';
      default: return 'Document';
    }
  }
}

class IssueReport {
  final String id;
  final String studentId;
  final String? studentName;
  final String? studentEmail;
  final String? studentProgram;
  final String? studentBranch;
  final String? studentSemester;
  final String? mentorEmail;
  final String category;
  final String title;
  final String description;
  final String priority;
  final String status;
  final String? mentorResponse;
  final DateTime? mentorRespondedAt;
  final DateTime createdAt;
  final DateTime updatedAt;

  IssueReport({
    required this.id,
    required this.studentId,
    this.studentName,
    this.studentEmail,
    this.studentProgram,
    this.studentBranch,
    this.studentSemester,
    this.mentorEmail,
    required this.category,
    required this.title,
    required this.description,
    required this.priority,
    required this.status,
    this.mentorResponse,
    this.mentorRespondedAt,
    required this.createdAt,
    required this.updatedAt,
  });

  factory IssueReport.fromMap(Map<String, dynamic> map) {
    return IssueReport(
      id: map['id'] ?? '',
      studentId: map['student_id'] ?? '',
      studentName: map['student_name'],
      studentEmail: map['student_email'],
      studentProgram: map['student_program'],
      studentBranch: map['student_branch'],
      studentSemester: map['student_semester'],
      mentorEmail: map['mentor_email'],
      category: map['category'] ?? 'other',
      title: map['title'] ?? '',
      description: map['description'] ?? '',
      priority: map['priority'] ?? 'medium',
      status: map['status'] ?? 'open',
      mentorResponse: map['mentor_response'],
      mentorRespondedAt: map['mentor_responded_at'] != null 
          ? DateTime.parse(map['mentor_responded_at']) : null,
      createdAt: DateTime.parse(map['created_at'] ?? DateTime.now().toIso8601String()),
      updatedAt: DateTime.parse(map['updated_at'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'student_id': studentId,
      'student_name': studentName,
      'student_email': studentEmail,
      'student_program': studentProgram,
      'student_branch': studentBranch,
      'student_semester': studentSemester,
      'mentor_email': mentorEmail,
      'category': category,
      'title': title,
      'description': description,
      'priority': priority,
      'status': status,
      'mentor_response': mentorResponse,
      'mentor_responded_at': mentorRespondedAt?.toIso8601String(),
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }

  bool get isOpen     => status == 'open';
  bool get isInProgress => status == 'in_progress';
  bool get isResolved => status == 'resolved';
  bool get hasResponse => mentorResponse != null && mentorResponse!.isNotEmpty;

  bool get isUrgent => priority == 'urgent';
  bool get isHigh   => priority == 'high';
  bool get isMedium => priority == 'medium';

  static String categoryLabel(String cat) {
    switch (cat) {
      case 'academic':     return 'Academic';
      case 'attendance':   return 'Attendance';
      case 'examination':  return 'Examination';
      case 'hostel':      return 'Hostel';
      case 'financial':    return 'Financial';
      case 'placement':    return 'Placement';
      case 'personal':     return 'Personal';
      default: return cat[0].toUpperCase() + cat.substring(1);
    }
  }

  static Map<String, dynamic> statusInfo(String status) {
    switch (status) {
      case 'open':        return {'label': 'Open', 'emoji': '⭕'};
      case 'in_progress': return {'label': 'Analyzing', 'emoji': '⏳'};
      case 'resolved':    return {'label': 'Resolved', 'emoji': '✅'};
      default:            return {'label': status, 'emoji': '📄'};
    }
  }

  static Map<String, dynamic> priorityInfo(String p) {
    switch (p) {
      case 'urgent': return {'label': 'Urgent', 'emoji': '🔥'};
      case 'high':   return {'label': 'High', 'emoji': '⚡'};
      case 'medium': return {'label': 'Medium', 'emoji': '🔸'};
      default:       return {'label': 'Low', 'emoji': '🔹'};
    }
  }
}

class StudentProgressReport {
  final UserModel student;
  final List<ConversationModel> conversations;
  final List<IssueReport> issues;
  final int totalMessages;
  final int activeConversations;
  final int resolvedConversations;
  final int flaggedConversations;
  final DateTime? lastActive;

  StudentProgressReport({
    required this.student,
    required this.conversations,
    required this.issues,
    required this.totalMessages,
    required this.activeConversations,
    required this.resolvedConversations,
    required this.flaggedConversations,
    this.lastActive,
  });

  double get engagementScore {
    if (totalMessages == 0) return 0;
    double score = totalMessages * 0.5;
    score += resolvedConversations * 20;
    score -= flaggedConversations * 10;
    return score.clamp(0, 100);
  }

  String get engagementLabel {
    final score = engagementScore;
    if (score >= 80) return 'Exceptional';
    if (score >= 60) return 'High';
    if (score >= 40) return 'Solid';
    if (score >= 20) return 'Moderate';
    return 'Low';
  }

  int get openIssues => issues.where((i) => i.isOpen).length;
}