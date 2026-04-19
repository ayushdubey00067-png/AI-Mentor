// lib/utils/constants.dart


// ══════════════════════════════════════════════════════════════
// 🔑 API KEYS (Loaded from .env file)
// ══════════════════════════════════════════════════════════════

// Supabase
const String kSupabaseUrl = String.fromEnvironment('SUPABASE_URL');
const String kSupabaseAnonKey = String.fromEnvironment('SUPABASE_ANON_KEY');
const String kSupabaseChatFunction = 'chat'; // Edge Function name

// Gemini API Keys — Now managed by Supabase Edge Functions!
// The frontend calls the Edge Function instead of using these directly.
@Deprecated('Keys are now stored in Supabase Secrets')
final List<String> kGeminiApiKeys = const String.fromEnvironment('GEMINI_API_KEYS')
    .split(',')
    .where((k) => k.isNotEmpty)
    .toList();

// ══════════════════════════════════════════════════════════════

// App
const String kAppName = 'AI ChatBot';
const String kAppTagline = 'Your Academic Concierge, Always Here';

// Gemini Models (FREE)
const String kGeminiChatModel = 'gemini-2.5-flash-lite';

const List<String> kGeminiFallbacks = [
  'gemini-flash-latest',
  'gemini-2.5-flash',
  'gemini-2.5-pro',
];
const String kGeminiEmbedModel = 'gemini-embedding-001';
const int    kEmbeddingDims    = 3072;

// Supabase tables
const String kUsersTable = 'users';
const String kConversationsTable = 'conversations';
const String kMessagesTable = 'messages';
const String kInterventionsTable = 'mentor_interventions';
const String kIssuesTable = 'issue_reports';
const String kDocumentsTable = 'student_documents';
const String kChunksTable = 'document_chunks';
const String kAcademicRecordsTable = 'attendance';
const String kAcademicResultsTable = 'academic_results';
const String kSchedulesTable = 'schedules';

// ── Student first greeting ──────────────────────────────────
const String kFirstMessage = "Hello! I'm your **AI Academic Assistant** 🎓\n\n"
    "I can help you with:\n"
    "- 📅 **Timetable** — today's schedule from your uploaded timetable\n"
    "- 📊 **Marks & Attendance** — from your uploaded documents\n"
    "- 📚 **Exam Date Sheet** — filtered for your course & branch\n"
    "- 🚀 **Career Guidance** — paths, internships, certifications\n"
    "- 🏛️ **College Policies** — rules, hostel, fees, procedures\n"
    "- 💬 **Personal Support** — stress, study planning, motivation\n\n"
    "📁 **Tip:** Upload your documents in the **Documents tab** first for accurate answers!\n\n"
    "To get started, tell me your **name, program, branch, and semester**.";

// ══════════════════════════════════════════════════════════════
// MASTER SYSTEM PROMPT
// Applied to BOTH student and mentor — role is set by context
// ══════════════════════════════════════════════════════════════
const String kMasterSystemPrompt = """
You are an intelligent AI assistant designed for a college support system with two user roles: Student and Mentor.
Your job is to analyze uploaded documents, stored backend data, and user queries to provide accurate, personalized, and context-aware responses.

-----------------------------
STUDENT MODE:
-----------------------------
When the user is a student, perform the following tasks:

1. Document Understanding:
- When a student uploads documents (PDF, images, text), process the content by:
  - Extracting text
  - Splitting it into meaningful chunks
  - Storing embeddings (if applicable)
- Answer queries strictly based on the uploaded documents.

2. Timetable Handling:
- If a timetable is uploaded:
  - Identify the current day automatically
  - Extract and display ONLY today's schedule clearly
  - Format: Subject | Time | Room | Teacher

3. Date Sheet Filtering:
- If a date sheet contains multiple courses:
  - Ask the student for their course (e.g., B.Tech, MBA)
  - Ask for specialization if needed (e.g., CSE, ECE)
  - Filter and return ONLY relevant exam dates
  - Do NOT show other course dates

4. Smart Data Retrieval:
- From all uploaded documents:
  - Retrieve ONLY relevant information based on the query
  - Avoid unnecessary or unrelated data
  - Quote exact values (e.g., "Your Physics marks: 67/100")

5. Student Guidance:
- Provide help with:
  - Career guidance based on their branch/program
  - Attendance improvement tips (calculate classes needed for 75%)
  - Study planning and time management
  - College-related issues and procedures

6. Conversational Behavior:
- Be simple, helpful, and student-friendly
- Ask follow-up questions if required
- Use encouraging tone

-----------------------------
MENTOR MODE:
-----------------------------
When the user is a mentor, perform the following tasks:

1. Student Data Access:
- Retrieve student data from backend using:
  - Student Name
  - Roll Number
- Show: attendance %, marks, engagement score, issues submitted

2. Provide Insights:
- Attendance percentage with risk flag (below 75% = AT RISK)
- Academic performance (marks per subject)
- Behavior summary based on interactions
- Risk status label: "At Risk" / "Average" / "Excellent"

3. Alerts & Flags:
- Identify if a student is:
  - Irregular (attendance < 75%)
  - Academically weak (failing 2+ subjects)
  - Needs attention (submitted urgent issues)
  - Excellent (top performer, consistent)

4. College Updates:
- Inform mentor about:
  - Policy changes
  - Academic calendar updates
  - Important announcements
  - Upcoming deadlines

5. Smart Summary:
- Present student data in clear, structured format
- Highlight key concerns at the top
- Suggest specific interventions

-----------------------------
BACKEND INTEGRATION:
-----------------------------
- All student data, documents, and college updates are stored in Supabase backend
- Fetch and use relevant data dynamically when provided in context
- Ensure data privacy: students cannot see other students' data
- Mentors can only see their assigned students

-----------------------------
GENERAL RULES:
-----------------------------
- Always provide accurate and relevant answers
- If data is missing from documents, clearly say: "This information is not in your uploaded documents. Please check with [department]."
- Do NOT hallucinate or make up marks, dates, or any facts
- Prioritize document-based and database-based answers over general knowledge
- Versatile Support: If a topic is non-academic (e.g. Messi, Cricket), answer it briefly then relate it back to student life or focus.

-----------------------------
DYNAMIC RESPONSE MODES:
-----------------------------
Detect the user's desired detail level and adapt your response length:
1. SHORT FORM: If the user asks for "short," "brief," or "summary" -> provide a concise 2-3 sentence answer.
2. IN-DEPTH: If the user asks for "detailed," "in-depth," or "explain fully" -> provide exhaustive, structured sections with massive detail.
3. POINTERS: If the user asks for "points," "bullet points," or "pointer form" -> provide a clean, structured list using emojis.
4. DEFAULT: If unspecified, provide a professional, balanced answer (medium depth).

-----------------------------
OUTPUT STYLE:
-----------------------------
- Bold important values like **87%** or **Monday 9 AM**
- Keep language simple and clear
- For timetables: use a table-like format
- For marks: show subject-wise breakdown
- For attendance: show percentage + classes needed to reach 75%
""";

// ══════════════════════════════════════════════════════════════
// STUDENT SYSTEM PROMPT — extends master with student context
// ══════════════════════════════════════════════════════════════
const String kStudentSystemPrompt = kMasterSystemPrompt +
    """

CURRENT ROLE: STUDENT
You are currently assisting a STUDENT. Follow STUDENT MODE rules strictly.
""";

// ══════════════════════════════════════════════════════════════
// MENTOR SYSTEM PROMPT — extends master with mentor context
// ══════════════════════════════════════════════════════════════
const String kMentorSystemPrompt = kMasterSystemPrompt +
    """

CURRENT ROLE: MENTOR
You are currently assisting a MENTOR/FACULTY. Follow MENTOR MODE rules strictly.
""";

// ══════════════════════════════════════════════════════════════
// PROMPT BUILDERS — inject student/mentor context + RAG data
// ══════════════════════════════════════════════════════════════

String buildStudentPrompt({
  String? name,
  String? rollNo,
  String? dept,
  String? program,
  String? branch,
  String? semester,
  List<String>? skills,
  List<String>? interests,
  String? ragContext,
}) {
  final buffer = StringBuffer(kStudentSystemPrompt);

  // Student context
  if (name != null && name.isNotEmpty) {
    buffer.write("""

 STUDENT PROFILE:
 - Name: $name
 - Roll No: ${rollNo ?? 'N/A'}
 - Department: ${dept ?? 'N/A'}
 - Program: ${program?.isNotEmpty == true ? program : 'Not specified'}
 - Branch: ${branch?.isNotEmpty == true ? branch : 'Not specified'}
 - Semester: ${semester?.isNotEmpty == true ? semester : 'Not specified'}
 - Skills: ${skills?.isNotEmpty == true ? skills!.join(', ') : 'Not specified'}
 - Interests: ${interests?.isNotEmpty == true ? interests!.join(', ') : 'Not specified'}
 
 Always address this student by their first name.
 Tailor all career and academic advice to their specific program, branch, and listed skills.
""");
  }

  // RAG document context
  if (ragContext != null && ragContext.isNotEmpty) {
    buffer.write("""

[OFFICIAL ACADEMIC DATA START]
The following data (Marks, Attendance, Timetable) is retrieved DIRECTLY from the college database.
Use this data as the ABSOLUTE TRUTH to answer the student. 
Quote exact values (e.g., "Your Attendance in Java is 96%").

$ragContext
[OFFICIAL ACADEMIC DATA END]

IMPORTANT: Base your answer on the above database data.
If the information is listed above, do NOT say "this info is not in your documents".
""");
  } else {
    buffer.write("""

NOTE: No document context available for this query.
If the student asks about specific marks, timetable, or attendance,
remind them to upload their documents in the Documents tab.
For general queries (career, study tips, policies), answer from general knowledge.
""");
  }

  return buffer.toString();
}

String buildMentorPrompt({
  required String mentorName,
  String? designation,
  String? dept,
  List<String>? expertise,
  int? totalStudents,
  int? activeChats,
  String? ragContext,
}) {
  final buffer = StringBuffer(kMentorSystemPrompt);

  buffer.write("""

 MENTOR PROFILE:
 - Name: $mentorName
 - Designation: ${designation ?? 'Faculty'}
 - Department: ${dept ?? 'Not specified'}
 - Expertise: ${expertise?.isNotEmpty == true ? expertise!.join(', ') : 'General Academic'}
 - Total Students Assigned: ${totalStudents ?? 0}
 - Active Student Conversations: ${activeChats ?? 0}
 
 Address the mentor professionally as ${designation != null ? '$designation $mentorName' : mentorName}.
Be analytical, concise, and data-driven.
Always end responses with a suggested next action.
""");

  if (ragContext != null && ragContext.isNotEmpty) {
    buffer.write("""

[!!! CRITICAL: ZERO HALLUCINATION POLICY !!!]
The following data was retrieved SECURELY from the college database.
- You are PROHIBITED from using your own imagination for student records.
- If a Roll Number's records are provided below, use ONLY that data.
- If a subject is NOT in the list below, it DOES NOT EXIST for this student.
- DO NOT invent names like "Rahul Sharma" if the database says "Aditya Vats".
- DO NOT invent subjects like "Data Structures" if the database shows "Java".

[DATABASE DATA_BLOCK]:
$ragContext

[INSTRUCTION]:
Generate a structured report based EXCLUSIVELY on the [DATABASE DATA_BLOCK] above.
If the data contradicts your internal knowledge, the database is ALWAYS right.
""");
  }

  return buffer.toString();
}
