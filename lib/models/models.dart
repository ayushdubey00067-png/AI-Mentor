// lib/models/models.dart

class UserModel {
  final String id;
  final String email;
  final String name;
  final String role; // 'admin', 'mentor', 'student'
  final String? program;
  final String? branch;
  final String? semester;
  final String? section;
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
  
  // Rich Student Directory Fields
  final String? officialEmail;
  final String? personalEmail;
  final String? mobileNo;
  final String? gender;
  final String? fatherName;
  final String? fatherMobile;
  final String? motherName;
  final String? motherMobile;
  final String? studentClass;
  final String? domicileState;
  final String? pincode;
  final String? applicationNo;
  final String? admissionDate;
  final String? status;
  final String? assignedClass; // for mentors e.g. 'CSE 4A'
  final String? baseSemester;
  final String? baseYear;
  final String? passwordHash;

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
    this.section,
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
    this.officialEmail,
    this.personalEmail,
    this.mobileNo,
    this.gender,
    this.fatherName,
    this.fatherMobile,
    this.motherName,
    this.motherMobile,
    this.studentClass,
    this.domicileState,
    this.pincode,
    this.applicationNo,
    this.admissionDate,
    this.status,
    this.assignedClass,
    this.baseSemester,
    this.baseYear,
    this.passwordHash,
    required this.createdAt,
    required this.lastActive,
  });

  bool get isAdmin => role == 'admin';
  bool get isMentor => role == 'mentor';
  bool get isStudent => role == 'student';

  factory UserModel.fromMap(Map<String, dynamic> map) {
    final role = map['role'] ?? 'student';
    final rawName = map['full_name'] ?? map['name'] ?? '';
    final rawEmail = map['official_email'] ?? map['email'] ?? '';
    final rawPhone = map['mobile_no'] ?? map['phone'];

    return UserModel(
      id: map['id'] ?? '',
      email: rawEmail,
      name: rawName,
      role: role,
      program: map['program'],
      branch: map['branch'],
      semester: map['semester']?.toString(),
      section: map['section'] ?? (role == 'student' ? 'A' : null),
      mentorEmail: map['mentor_email'],
      rollNumber: map['roll_number'],
      phone: rawPhone,
      department: map['department'],
      skills: map['skills'] != null ? List<String>.from(map['skills']) : null,
      hobbies: map['hobbies'] != null ? List<String>.from(map['hobbies']) : null,
      careerInterests: map['career_interests'] != null ? List<String>.from(map['career_interests']) : null,
      designation: map['designation'],
      expertise: map['expertise'] != null ? List<String>.from(map['expertise']) : null,
      officeLocation: map['office_location'],
      officeHours: map['office_hours'],
      profileUrl: map['profile_url'],
      officialEmail: map['official_email'] ?? (role == 'student' ? rawEmail : null),
      personalEmail: map['personal_email'],
      mobileNo: rawPhone,
      gender: map['gender'],
      fatherName: map['father_name'],
      fatherMobile: map['father_mobile'],
      motherName: map['mother_name'],
      motherMobile: map['mother_mobile'],
      studentClass: map['student_class'],
      domicileState: map['domicile_state'],
      pincode: map['pincode']?.toString(),
      applicationNo: map['application_no'],
      admissionDate: map['admission_date'],
      status: map['status'] ?? 'active',
      assignedClass: map['assigned_class'],
      baseSemester: map['base_semester']?.toString(),
      baseYear: map['base_year'],
      passwordHash: map['password_hash'],
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
      'section': section,
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
      'official_email': officialEmail,
      'personal_email': personalEmail,
      'mobile_no': mobileNo,
      'gender': gender,
      'father_name': fatherName,
      'father_mobile': fatherMobile,
      'mother_name': motherName,
      'mother_mobile': motherMobile,
      'student_class': studentClass,
      'domicile_state': domicileState,
      'pincode': pincode,
      'application_no': applicationNo,
      'admission_date': admissionDate,
      'status': status,
      'assigned_class': assignedClass,
      'base_semester': baseSemester,
      'base_year': baseYear,
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
  final String? studentId;
  final String? uploadedBy;
  final String docType;
  final String title;
  final String fileName;
  final String mimeType;
  final int? fileSize;
  final String? contentBase64;
  final String? extractedText;
  final String? storagePath;
  final String targetScope; // 'class' or 'individual'
  final String? targetRollNo;
  final String? program;
  final String? branch;
  final String? semester;
  final String? academicYear;
  final String? academicTerm;
  final Map<String, dynamic>? extractedJson;
  final String ocrStatus; // 'pending', 'processing', 'completed', 'paused', 'failed'
  final Map<String, dynamic>? ocrProgress;
  final DateTime createdAt;

  StudentDocument({
    required this.id,
    this.studentId,
    this.uploadedBy,
    required this.docType,
    required this.title,
    required this.fileName,
    required this.mimeType,
    this.fileSize,
    this.contentBase64,
    this.extractedText,
    this.storagePath,
    this.targetScope = 'class',
    this.targetRollNo,
    this.program,
    this.branch,
    this.semester,
    this.academicYear = '2026-2027',
    this.academicTerm,
    this.extractedJson,
    this.ocrStatus = 'pending',
    this.ocrProgress,
    required this.createdAt,
  });

  bool get isOcrCompleted => ocrStatus == 'completed';
  bool get isOcrProcessing => ocrStatus == 'processing';
  bool get isOcrPending => ocrStatus == 'pending' || ocrStatus.isEmpty;
  bool get isOcrPaused => ocrStatus == 'paused';

  String get term {
    if (academicTerm != null && academicTerm!.isNotEmpty) {
      return academicTerm!.toLowerCase();
    }
    if (semester != null && (semester!.toLowerCase() == 'odd' || semester!.toLowerCase() == 'even')) {
      return semester!.toLowerCase();
    }
    return 'odd';
  }

  factory StudentDocument.fromMap(Map<String, dynamic> map) {
    return StudentDocument(
      id: map['id'] ?? '',
      studentId: map['student_id'],
      uploadedBy: map['uploaded_by'],
      docType: map['doc_type'] ?? 'other',
      title: map['title'] ?? '',
      fileName: map['file_name'] ?? '',
      mimeType: map['mime_type'] ?? '',
      fileSize: map['file_size'],
      contentBase64: map['content_base64'],
      extractedText: map['extracted_text'],
      storagePath: map['storage_path'],
      targetScope: map['target_scope'] ?? 'class',
      targetRollNo: map['target_roll_no'],
      program: map['program'],
      branch: map['branch'],
      semester: map['semester'],
      academicYear: map['academic_year'] ?? '2026-2027',
      academicTerm: map['academic_term'] ?? map['semester'],
      extractedJson: map['extracted_json'] is Map ? Map<String, dynamic>.from(map['extracted_json']) : null,
      ocrStatus: map['ocr_status'] ?? (map['extracted_text'] != null && (map['extracted_text'] as String).isNotEmpty ? 'completed' : 'pending'),
      ocrProgress: map['ocr_progress'] is Map ? Map<String, dynamic>.from(map['ocr_progress']) : null,
      createdAt: DateTime.parse(map['created_at'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      if (studentId != null) 'student_id': studentId,
      if (uploadedBy != null) 'uploaded_by': uploadedBy,
      'doc_type': docType,
      'title': title,
      'file_name': fileName,
      'mime_type': mimeType,
      'file_size': fileSize,
      'content_base64': contentBase64,
      'extracted_text': extractedText,
      'storage_path': storagePath,
      'target_scope': targetScope,
      if (targetRollNo != null) 'target_roll_no': targetRollNo,
      if (program != null) 'program': program,
      if (branch != null) 'branch': branch,
      if (semester != null) 'semester': semester,
      'academic_year': academicYear,
      if (extractedJson != null) 'extracted_json': extractedJson,
      'ocr_status': ocrStatus,
      if (ocrProgress != null) 'ocr_progress': ocrProgress,
      'created_at': createdAt.toIso8601String(),
    };
  }

  StudentDocument copyWith({
    String? id,
    String? studentId,
    String? uploadedBy,
    String? docType,
    String? title,
    String? fileName,
    String? mimeType,
    int? fileSize,
    String? contentBase64,
    String? extractedText,
    String? storagePath,
    String? targetScope,
    String? targetRollNo,
    String? program,
    String? branch,
    String? semester,
    String? academicYear,
    String? academicTerm,
    Map<String, dynamic>? extractedJson,
    String? ocrStatus,
    Map<String, dynamic>? ocrProgress,
    DateTime? createdAt,
  }) {
    return StudentDocument(
      id: id ?? this.id,
      studentId: studentId ?? this.studentId,
      uploadedBy: uploadedBy ?? this.uploadedBy,
      docType: docType ?? this.docType,
      title: title ?? this.title,
      fileName: fileName ?? this.fileName,
      mimeType: mimeType ?? this.mimeType,
      fileSize: fileSize ?? this.fileSize,
      contentBase64: contentBase64 ?? this.contentBase64,
      extractedText: extractedText ?? this.extractedText,
      storagePath: storagePath ?? this.storagePath,
      targetScope: targetScope ?? this.targetScope,
      targetRollNo: targetRollNo ?? this.targetRollNo,
      program: program ?? this.program,
      branch: branch ?? this.branch,
      semester: semester ?? this.semester,
      academicYear: academicYear ?? this.academicYear,
      academicTerm: academicTerm ?? this.academicTerm,
      extractedJson: extractedJson ?? this.extractedJson,
      ocrStatus: ocrStatus ?? this.ocrStatus,
      ocrProgress: ocrProgress ?? this.ocrProgress,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  bool get isClassScope => targetScope == 'class';
  bool get isIndividualScope => targetScope == 'individual';

  static String typeLabel(String type) {
    switch (type) {
      case 'timetable': return 'Timetable';
      case 'academic_calendar': return 'Academic Calendar';
      case 'syllabus': return 'Syllabus';
      case 'marksheet': return 'Marksheet';
      case 'attendance': return 'Attendance';
      case 'notes': return 'Lecture Notes';
      case 'assignment': return 'Assignment';
      case 'notice': return 'Official Notice';
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

// ══════════════════════════════════════════════════════════
// ATTENDANCE & MULTI-SEMESTER MODELS
// ══════════════════════════════════════════════════════════

class AttendanceRecord {
  final String id;
  final String studentRollNo;
  final String semester;
  final String subjectCode;
  final String subjectName;
  final String courseType;
  final String? facultyName;
  final double attendancePercentage;
  final int totalClasses;
  final int attendedClasses;
  final String monitoringCycle;
  final String academicYear;
  final String? lastUpdatedBy;
  final DateTime lastUpdated;

  AttendanceRecord({
    required this.id,
    required this.studentRollNo,
    required this.semester,
    required this.subjectCode,
    required this.subjectName,
    this.courseType = 'CORE',
    this.facultyName,
    required this.attendancePercentage,
    this.totalClasses = 0,
    this.attendedClasses = 0,
    this.monitoringCycle = 'SECOND MONITORING',
    this.academicYear = '2025-2026',
    this.lastUpdatedBy,
    required this.lastUpdated,
  });

  bool get isDefaulter => attendancePercentage < 75.0;

  factory AttendanceRecord.fromMap(Map<String, dynamic> map) {
    return AttendanceRecord(
      id: map['id'] ?? '',
      studentRollNo: map['student_roll_no'] ?? '',
      semester: map['semester']?.toString() ?? '5',
      subjectCode: map['subject_code'] ?? '',
      subjectName: map['subject_name'] ?? '',
      courseType: map['course_type'] ?? 'CORE',
      facultyName: map['faculty_name'],
      attendancePercentage: (map['attendance_percentage'] != null)
          ? double.tryParse(map['attendance_percentage'].toString()) ?? 0.0
          : 0.0,
      totalClasses: map['total_classes'] ?? 0,
      attendedClasses: map['attended_classes'] ?? 0,
      monitoringCycle: map['monitoring_cycle'] ?? 'SECOND MONITORING',
      academicYear: map['academic_year'] ?? '2025-2026',
      lastUpdatedBy: map['last_updated_by'],
      lastUpdated: DateTime.parse(map['last_updated'] ?? DateTime.now().toIso8601String()),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'student_roll_no': studentRollNo,
      'semester': semester,
      'subject_code': subjectCode,
      'subject_name': subjectName,
      'course_type': courseType,
      'faculty_name': facultyName,
      'attendance_percentage': attendancePercentage,
      'total_classes': totalClasses,
      'attended_classes': attendedClasses,
      'monitoring_cycle': monitoringCycle,
      'academic_year': academicYear,
      'last_updated_by': lastUpdatedBy,
      'last_updated': lastUpdated.toIso8601String(),
    };
  }
}

class StudentAttendanceSummary {
  final String id;
  final String studentRollNo;
  final String semester;
  final String monitoringCycle;
  final double overallPercentage;
  final int defaulterSubjectCount;
  final bool isCritical;
  final DateTime lastUpdated;

  StudentAttendanceSummary({
    required this.id,
    required this.studentRollNo,
    required this.semester,
    required this.monitoringCycle,
    required this.overallPercentage,
    this.defaulterSubjectCount = 0,
    this.isCritical = false,
    required this.lastUpdated,
  });

  factory StudentAttendanceSummary.fromMap(Map<String, dynamic> map) {
    return StudentAttendanceSummary(
      id: map['id'] ?? '',
      studentRollNo: map['student_roll_no'] ?? '',
      semester: map['semester']?.toString() ?? '5',
      monitoringCycle: map['monitoring_cycle'] ?? 'SECOND MONITORING',
      overallPercentage: (map['overall_percentage'] != null)
          ? double.tryParse(map['overall_percentage'].toString()) ?? 0.0
          : 0.0,
      defaulterSubjectCount: map['defaulter_subject_cnt'] ?? 0,
      isCritical: map['is_critical'] ?? false,
      lastUpdated: DateTime.parse(map['last_updated'] ?? DateTime.now().toIso8601String()),
    );
  }
}

class AttendanceReportModel {
  final String id;
  final String className;
  final String section;
  final String semester;
  final String? department;
  final String? branch;
  final String cycleName;
  final String? dateRange;
  final double minCriteria;
  final String uploadedBy;
  final int totalStudents;
  final int criticalCount;
  final DateTime createdAt;

  AttendanceReportModel({
    required this.id,
    required this.className,
    required this.section,
    required this.semester,
    this.department,
    this.branch,
    required this.cycleName,
    this.dateRange,
    this.minCriteria = 75.0,
    required this.uploadedBy,
    this.totalStudents = 0,
    this.criticalCount = 0,
    required this.createdAt,
  });

  factory AttendanceReportModel.fromMap(Map<String, dynamic> map) {
    return AttendanceReportModel(
      id: map['id'] ?? '',
      className: map['class_name'] ?? '',
      section: map['section'] ?? 'A',
      semester: map['semester']?.toString() ?? '5',
      department: map['department'],
      branch: map['branch'],
      cycleName: map['cycle_name'] ?? '',
      dateRange: map['date_range'],
      minCriteria: (map['min_criteria'] != null)
          ? double.tryParse(map['min_criteria'].toString()) ?? 75.0
          : 75.0,
      uploadedBy: map['uploaded_by'] ?? '',
      totalStudents: map['total_students'] ?? 0,
      criticalCount: map['critical_count'] ?? 0,
      createdAt: DateTime.parse(map['created_at'] ?? DateTime.now().toIso8601String()),
    );
  }
}

class SubjectColumnInfo {
  final String subjectName;
  final String subjectCode;
  final String courseType;
  final String facultyName;
  final int colIndex;

  SubjectColumnInfo({
    required this.subjectName,
    required this.subjectCode,
    required this.courseType,
    required this.facultyName,
    required this.colIndex,
  });
}

class StudentAttendanceRow {
  final int srNo;
  final String rollNo;
  final String studentName;
  final Map<String, double?> subjectPercentages; // Key: subjectCode -> percentage (or null if not enrolled)
  final double overallPercentage;
  final int defaulterCount;
  final bool isCritical;

  StudentAttendanceRow({
    required this.srNo,
    required this.rollNo,
    required this.studentName,
    required this.subjectPercentages,
    required this.overallPercentage,
    required this.defaulterCount,
    required this.isCritical,
  });
}

class ParsedAttendanceReport {
  final String branch;
  final String department;
  final String className;
  final String section;
  final String semester;
  final String cycleName;
  final String dateRange;
  final double minCriteria;
  final String programCoordinator;
  final List<SubjectColumnInfo> subjects;
  final List<StudentAttendanceRow> studentRows;
  final int totalStudents;
  final int criticalCount;
  final bool isFormatValid;
  final String? validationError;

  ParsedAttendanceReport({
    required this.branch,
    required this.department,
    required this.className,
    required this.section,
    required this.semester,
    required this.cycleName,
    required this.dateRange,
    required this.minCriteria,
    required this.programCoordinator,
    required this.subjects,
    required this.studentRows,
    required this.totalStudents,
    required this.criticalCount,
    this.isFormatValid = true,
    this.validationError,
  });
}