// lib/services/document_json_schemas.dart

/// Strict JSON Schemas and prompts for Native JSON OCR across all 6 Academic Document Domains
class DocumentJsonSchemas {
  /// Returns the specialized prompt that instructs Gemini Vision to output 100% valid Native JSON
  static String getPromptForDocType(String docType) {
    switch (docType.toLowerCase()) {
      case 'marksheet':
      case 'result':
      case 'academic_results':
        return _marksheetPrompt;
      case 'attendance':
      case 'attendance_register':
        return _attendancePrompt;
      case 'syllabus':
      case 'curriculum':
        return _syllabusPrompt;
      case 'academic_calendar':
      case 'calendar':
      case 'datesheet':
        return _calendarPrompt;
      case 'assignment':
      case 'project':
        return _assignmentPrompt;
      case 'circular':
      case 'notice':
      case 'other':
      case 'others':
      default:
        return _circularPrompt;
    }
  }

  static const String _marksheetPrompt = """
You are an institutional examination marksheet optical extractor for Manav Rachna University.
Analyze this marksheet page image and extract ALL data into pure, valid JSON with ZERO hallucination.
Transcribe ONLY the literal text printed in the headers, column boxes, and student rows.

OUTPUT JSON SCHEMA:
{
  "university_name": "Full name of university printed at top (e.g. MANAV RACHNA UNIVERSITY)",
  "school_name": "School or Faculty name (e.g. SCHOOL OF ENGINEERING)",
  "programme_name": "Degree, branch and specializations printed (e.g. BACHELOR OF TECHNOLOGY IN COMPUTER SCIENCE AND ENGINEERING)",
  "result_session": "Result session printed at top (e.g. MAY-2026)",
  "semester": "Semester name verbatim (e.g. SECOND)",
  "batch": "Academic batch verbatim (e.g. 2025 - 29)",
  "date": "Date printed at bottom (e.g. 15-06-26)",
  "courses": [
    {
      "code": "Exact course code printed in the top row of table header (e.g. 4.5PH02000, 4.5CS04E00)",
      "title": "Exact course title verbatim from header box (e.g. QUANTUM MECHANICS FOR ENGINEERS)",
      "credits": 3
    }
  ],
  "students": [
    {
      "s_no": 3,
      "roll_no": "Exact alphanumeric student roll number (e.g. 2K25CSUN01003)",
      "name": "Full student name verbatim (e.g. ABHISHEK VATS)",
      "father_name": "Father name verbatim (e.g. VIJAY VATS)",
      "grades": {
        "CourseCode1": "Literal grade obtained (e.g. O, A+, A, B+, B, C, P, F, AB, DB, MP)"
      },
      "sgpa": 6.29,
      "result": "PASS or FAIL (Mark FAIL if any grade is F, AB, DB, or if SGPA is below 4.0, otherwise PASS)"
    }
  ]
}

CRITICAL RULES:
1. Extract EVERY SINGLE student row present on this page from top to bottom without skipping.
2. If row numbers skip (e.g. rows 1 and 2 are empty/blank and students start at row 3), extract rows exactly where printed.
3. Map every grade in the student's row to its exact column course code header from left to right.
4. Calculate or transcribe 'result' strictly: if any grade is 'F', 'AB', 'DB', or 'MP', result is 'FAIL', otherwise 'PASS'.
5. Return ONLY valid raw JSON conforming strictly to this schema without markdown code blocks.
""";

  static const String _attendancePrompt = """
You are an institutional attendance register extractor.
Extract all attendance records on this page into pure, valid JSON.

OUTPUT JSON SCHEMA:
{
  "institution": "University / College name",
  "subject": {
    "code": "Subject code",
    "title": "Subject name"
  },
  "attendance_period": "Date range or month",
  "students": [
    {
      "roll_no": "Student roll number",
      "name": "Student full name",
      "total_classes": 40,
      "attended_classes": 32,
      "percentage": 80.0,
      "is_defaulter": false
    }
  ]
}
""";

  static const String _syllabusPrompt = """
You are an academic course syllabus parser.
Extract the syllabus structure from this page into pure, valid JSON.

OUTPUT JSON SCHEMA:
{
  "course_code": "Course Code",
  "course_title": "Course Title",
  "academic_year": "Academic Year",
  "semester": "Semester",
  "credits": 4,
  "modules": [
    {
      "module_number": 1,
      "title": "Module / Unit Title",
      "topics": ["Topic 1", "Topic 2", "Topic 3"],
      "recommended_readings": ["Book / Reference"]
    }
  ],
  "evaluation_criteria": "Grading / Exam weightage description"
}
""";

  static const String _calendarPrompt = """
You are an institutional academic calendar extractor.
Extract all dates, academic deadlines, and holiday schedules into pure, valid JSON.

OUTPUT JSON SCHEMA:
{
  "institution": "University Name",
  "academic_year": "e.g. 2026-2027",
  "term": "odd or even",
  "events": [
    {
      "date": "Date or date range",
      "day_of_week": "Day",
      "activity": "Activity or event description",
      "is_holiday": false,
      "category": "academic or examination or holiday"
    }
  ]
}
""";

  static const String _assignmentPrompt = """
You are an academic assignment guideline extractor.
Extract the assignment questions, deadlines, and submission rules into pure, valid JSON.

OUTPUT JSON SCHEMA:
{
  "subject_code": "Course Code",
  "subject_title": "Course Title",
  "assignment_title": "Assignment Title",
  "deadline": "Submission deadline date",
  "total_marks": 20,
  "questions": [
    {
      "question_number": 1,
      "question_text": "Text of question",
      "max_marks": 5
    }
  ],
  "submission_instructions": "Guidelines for submission"
}
""";

  static const String _circularPrompt = """
You are an institutional circular and policy notice extractor.
Extract this document into pure, valid JSON.

OUTPUT JSON SCHEMA:
{
  "reference_number": "Notice / Circular Ref No",
  "date_issued": "Date of circular",
  "title": "Title of notice",
  "issuing_authority": "Office / Dean / Registrar",
  "summary": "Key summary of notice",
  "key_points": ["Point 1", "Point 2"],
  "target_audience": "students or faculty or all"
}
""";
}
