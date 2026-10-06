-- ============================================================
-- AI ChatBot v12 — ATTENDANCE INGESTION & MULTI-SEMESTER SUBJECT ARCHITECTURE
-- Multi-Format Reports (.csv, .xlsx, .xls) + Subject Disambiguation
-- ============================================================

-- 1. ATTENDANCE MONITORING REPORTS (Metadata)
CREATE TABLE IF NOT EXISTS attendance_reports (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  class_name       TEXT NOT NULL,
  section          TEXT NOT NULL,
  semester         TEXT NOT NULL,
  department       TEXT,
  branch           TEXT,
  cycle_name       TEXT NOT NULL,
  date_range       TEXT,
  min_criteria     DECIMAL(5,2) DEFAULT 75.0,
  uploaded_by      TEXT NOT NULL,
  total_students   INTEGER DEFAULT 0,
  critical_count   INTEGER DEFAULT 0,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- 2. SUBJECT-WISE ATTENDANCE RECORDS (Multi-Semester Scoped)
CREATE TABLE IF NOT EXISTS attendance (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  student_roll_no       TEXT NOT NULL,
  semester              TEXT NOT NULL DEFAULT '5',
  subject_code          TEXT NOT NULL,
  subject_name          TEXT NOT NULL,
  course_type           TEXT DEFAULT 'CORE',
  faculty_name          TEXT,
  attendance_percentage DECIMAL(5,2) NOT NULL DEFAULT 0,
  total_classes         INTEGER DEFAULT 0,
  attended_classes      INTEGER DEFAULT 0,
  monitoring_cycle      TEXT DEFAULT 'SECOND MONITORING',
  academic_year         TEXT DEFAULT '2025-2026',
  last_updated_by       TEXT,
  last_updated          TIMESTAMPTZ DEFAULT NOW()
);

-- Add missing columns if attendance table already existed from earlier schema
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS semester TEXT NOT NULL DEFAULT '5';
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS course_type TEXT DEFAULT 'CORE';
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS faculty_name TEXT;
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS monitoring_cycle TEXT DEFAULT 'SECOND MONITORING';
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS academic_year TEXT DEFAULT '2025-2026';
ALTER TABLE attendance ADD COLUMN IF NOT EXISTS last_updated_by TEXT;

-- Composite uniqueness on (student_roll_no, semester, subject_code, monitoring_cycle)
DO $$
BEGIN
  BEGIN
    ALTER TABLE attendance DROP CONSTRAINT IF EXISTS uq_attendance_student_sem_sub_cycle;
  EXCEPTION WHEN OTHERS THEN NULL;
  END;
END $$;

ALTER TABLE attendance ADD CONSTRAINT uq_attendance_student_sem_sub_cycle 
UNIQUE (student_roll_no, semester, subject_code, monitoring_cycle);

-- 3. STUDENT ATTENDANCE SUMMARY (ERP Aggregates)
CREATE TABLE IF NOT EXISTS student_attendance_summary (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  student_roll_no       TEXT NOT NULL,
  semester              TEXT NOT NULL DEFAULT '5',
  monitoring_cycle      TEXT NOT NULL DEFAULT 'SECOND MONITORING',
  overall_percentage    DECIMAL(5,2) NOT NULL DEFAULT 0,
  defaulter_subject_cnt INTEGER DEFAULT 0,
  is_critical           BOOLEAN DEFAULT FALSE,
  last_updated          TIMESTAMPTZ DEFAULT NOW()
);

DO $$
BEGIN
  BEGIN
    ALTER TABLE student_attendance_summary DROP CONSTRAINT IF EXISTS uq_att_summary_roll_sem_cycle;
  EXCEPTION WHEN OTHERS THEN NULL;
  END;
END $$;

ALTER TABLE student_attendance_summary ADD CONSTRAINT uq_att_summary_roll_sem_cycle 
UNIQUE (student_roll_no, semester, monitoring_cycle);

-- 4. BATCH UPSERT RPC FUNCTION
CREATE OR REPLACE FUNCTION batch_upsert_attendance(
  report_meta JSONB,
  records_data JSONB,
  summaries_data JSONB
)
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  elem JSONB;
  rep_id UUID;
  upserted_subjects INT := 0;
  upserted_students INT := 0;
  v_semester TEXT;
  v_section TEXT;
  v_cycle TEXT;
  v_uploader TEXT;
BEGIN
  v_semester := COALESCE(report_meta->>'semester', '5');
  v_section  := COALESCE(report_meta->>'section', 'A');
  v_cycle    := COALESCE(report_meta->>'cycle_name', 'SECOND ATTENDANCE MONITORING REPORT');
  v_uploader := COALESCE(report_meta->>'uploaded_by', 'system@mru.ac.in');

  -- 1. Insert or update Report Header
  INSERT INTO attendance_reports (
    class_name, section, semester, department, branch,
    cycle_name, date_range, min_criteria, uploaded_by,
    total_students, critical_count
  ) VALUES (
    COALESCE(report_meta->>'class_name', 'BTech CSE Sem ' || v_semester),
    v_section,
    v_semester,
    report_meta->>'department',
    report_meta->>'branch',
    v_cycle,
    report_meta->>'date_range',
    COALESCE((report_meta->>'min_criteria')::DECIMAL, 75.0),
    v_uploader,
    COALESCE((report_meta->>'total_students')::INT, 0),
    COALESCE((report_meta->>'critical_count')::INT, 0)
  ) RETURNING id INTO rep_id;

  -- 2. Upsert Individual Subject Attendance Records
  FOR elem IN SELECT * FROM jsonb_array_elements(records_data)
  LOOP
    -- Auto-provision student shadow record if roll number doesn't exist yet
    INSERT INTO students (roll_number, full_name, official_email, mobile_no, password_hash, section, semester)
    VALUES (
      TRIM(elem->>'student_roll_no'),
      COALESCE(TRIM(elem->>'student_name'), 'Student ' || TRIM(elem->>'student_roll_no')),
      LOWER(TRIM(elem->>'student_roll_no')) || '@mru.ac.in',
      '9999999999',
      '9999999999',
      v_section,
      v_semester
    ) ON CONFLICT (roll_number) DO NOTHING;

    -- Upsert Subject Attendance
    INSERT INTO attendance (
      student_roll_no,
      semester,
      subject_code,
      subject_name,
      course_type,
      faculty_name,
      attendance_percentage,
      monitoring_cycle,
      academic_year,
      last_updated_by
    ) VALUES (
      TRIM(elem->>'student_roll_no'),
      COALESCE(TRIM(elem->>'semester'), v_semester),
      TRIM(elem->>'subject_code'),
      TRIM(elem->>'subject_name'),
      COALESCE(TRIM(elem->>'course_type'), 'CORE'),
      TRIM(elem->>'faculty_name'),
      (elem->>'attendance_percentage')::DECIMAL,
      COALESCE(TRIM(elem->>'monitoring_cycle'), v_cycle),
      COALESCE(TRIM(elem->>'academic_year'), '2025-2026'),
      v_uploader
    )
    ON CONFLICT (student_roll_no, semester, subject_code, monitoring_cycle)
    DO UPDATE SET
      subject_name          = EXCLUDED.subject_name,
      course_type           = EXCLUDED.course_type,
      faculty_name          = EXCLUDED.faculty_name,
      attendance_percentage = EXCLUDED.attendance_percentage,
      last_updated_by       = EXCLUDED.last_updated_by,
      last_updated          = NOW();

    upserted_subjects := upserted_subjects + 1;
  END LOOP;

  -- 3. Upsert Student Summaries
  FOR elem IN SELECT * FROM jsonb_array_elements(summaries_data)
  LOOP
    INSERT INTO student_attendance_summary (
      student_roll_no,
      semester,
      monitoring_cycle,
      overall_percentage,
      defaulter_subject_cnt,
      is_critical
    ) VALUES (
      TRIM(elem->>'student_roll_no'),
      COALESCE(TRIM(elem->>'semester'), v_semester),
      COALESCE(TRIM(elem->>'monitoring_cycle'), v_cycle),
      (elem->>'overall_percentage')::DECIMAL,
      COALESCE((elem->>'defaulter_subject_cnt')::INT, 0),
      COALESCE((elem->>'is_critical')::BOOLEAN, FALSE)
    )
    ON CONFLICT (student_roll_no, semester, monitoring_cycle)
    DO UPDATE SET
      overall_percentage    = EXCLUDED.overall_percentage,
      defaulter_subject_cnt = EXCLUDED.defaulter_subject_cnt,
      is_critical           = EXCLUDED.is_critical,
      last_updated          = NOW();

    upserted_students := upserted_students + 1;
  END LOOP;

  RETURN jsonb_build_object(
    'success', true,
    'report_id', rep_id,
    'total_subjects_updated', upserted_subjects,
    'total_students_updated', upserted_students
  );
END;
$$;

-- 5. Indexes
CREATE INDEX IF NOT EXISTS idx_attendance_roll_sem ON attendance(student_roll_no, semester);
CREATE INDEX IF NOT EXISTS idx_attendance_sub_code ON attendance(subject_code);
CREATE INDEX IF NOT EXISTS idx_attendance_cycle    ON attendance(monitoring_cycle);
CREATE INDEX IF NOT EXISTS idx_att_summary_roll    ON student_attendance_summary(student_roll_no);
CREATE INDEX IF NOT EXISTS idx_att_reports_class   ON attendance_reports(class_name, section);

-- 6. Permissions
ALTER TABLE attendance_reports          DISABLE ROW LEVEL SECURITY;
ALTER TABLE attendance                  DISABLE ROW LEVEL SECURITY;
ALTER TABLE student_attendance_summary  DISABLE ROW LEVEL SECURITY;

GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL ROUTINES IN SCHEMA public TO anon, authenticated;
