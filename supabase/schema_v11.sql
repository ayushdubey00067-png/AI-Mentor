-- ============================================================
-- AI ChatBot v11 — 3-TIER ARCHITECTURE SCHEMA
-- Admins + Mentors + Students + Batch Upsert RPC + Calendar Engine
-- ============================================================

-- 1. Enable Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. ADMINS TABLE
CREATE TABLE IF NOT EXISTS admins (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email         TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name          TEXT NOT NULL,
  role          TEXT NOT NULL DEFAULT 'admin',
  phone         TEXT,
  department    TEXT DEFAULT 'Academic Administration',
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Seed default Admin account if not exists
INSERT INTO admins (email, password_hash, name, role)
VALUES ('admin@mru.ac.in', 'admin123', 'Academic Administrator', 'admin')
ON CONFLICT (email) DO NOTHING;

-- 3. MENTORS TABLE
CREATE TABLE IF NOT EXISTS mentors (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email           TEXT UNIQUE NOT NULL,
  password_hash   TEXT NOT NULL,
  name            TEXT NOT NULL,
  role            TEXT NOT NULL DEFAULT 'mentor',
  department      TEXT DEFAULT 'Computer Science & Technology',
  designation     TEXT DEFAULT 'Assistant Professor',
  phone           TEXT,
  assigned_class  TEXT DEFAULT 'CSE 4A', -- e.g., 'CSE 4A', 'CSE 5A'
  office_location TEXT,
  office_hours    TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  last_active     TIMESTAMPTZ DEFAULT NOW()
);

-- 4. STUDENTS TABLE
CREATE TABLE IF NOT EXISTS students (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  roll_number     TEXT UNIQUE NOT NULL,
  full_name       TEXT NOT NULL,
  gender          TEXT,
  official_email  TEXT UNIQUE NOT NULL,
  personal_email  TEXT,
  mobile_no       TEXT NOT NULL,
  password_hash   TEXT NOT NULL,
  father_name     TEXT DEFAULT 'NA',
  father_mobile   TEXT DEFAULT 'NA',
  mother_name     TEXT DEFAULT 'NA',
  mother_mobile   TEXT DEFAULT 'NA',
  section         TEXT NOT NULL DEFAULT 'A',
  student_class   TEXT DEFAULT 'BTech CSE Sem 4',
  program         TEXT DEFAULT 'B.Tech',
  branch          TEXT DEFAULT 'Computer Science & Engineering',
  department      TEXT DEFAULT 'Dept. of Computer Science & Technology',
  semester        TEXT NOT NULL DEFAULT '4',
  base_semester   TEXT DEFAULT '4',
  base_year       TEXT DEFAULT '2025-2026',
  domicile_state  TEXT DEFAULT 'NA',
  pincode         TEXT DEFAULT 'NA',
  application_no  TEXT DEFAULT 'NA',
  admission_date  TEXT DEFAULT 'NA',
  status          TEXT DEFAULT 'active',
  mentor_email    TEXT REFERENCES mentors(email) ON DELETE SET NULL,
  role            TEXT NOT NULL DEFAULT 'student',
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  last_active     TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Seamless Migration of existing data from users table (if present)
DO $$
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users') THEN
    -- Migrate existing mentors
    INSERT INTO mentors (id, email, password_hash, name, department, designation, phone, office_location, office_hours, created_at, last_active)
    SELECT id, email, password_hash, name, department, designation, phone, office_location, office_hours, created_at, last_active
    FROM users
    WHERE role = 'mentor'
    ON CONFLICT (email) DO NOTHING;

    -- Migrate existing students
    INSERT INTO students (
      id, roll_number, full_name, official_email, mobile_no, password_hash,
      program, branch, semester, section, mentor_email, department, created_at, last_active
    )
    SELECT 
      id, 
      COALESCE(roll_number, 'ROLL_' || SUBSTRING(id::text, 1, 8)),
      name, 
      email, 
      COALESCE(phone, '9999999999'),
      password_hash,
      COALESCE(program, 'B.Tech'), 
      COALESCE(branch, 'Computer Science & Engineering'), 
      COALESCE(semester, '4'), 
      COALESCE(section, 'A'), 
      mentor_email, 
      department, 
      created_at, 
      last_active
    FROM users
    WHERE role = 'student'
    ON CONFLICT (roll_number) DO NOTHING;
  END IF;
END $$;

-- 6. High-Performance Batch Upsert RPC Function
CREATE OR REPLACE FUNCTION batch_upsert_students(students_data JSONB)
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  elem JSONB;
  inserted_count INT := 0;
BEGIN
  FOR elem IN SELECT * FROM jsonb_array_elements(students_data)
  LOOP
    INSERT INTO students (
      roll_number,
      full_name,
      gender,
      official_email,
      personal_email,
      mobile_no,
      password_hash,
      father_name,
      father_mobile,
      mother_name,
      mother_mobile,
      section,
      student_class,
      program,
      branch,
      department,
      semester,
      base_semester,
      base_year,
      domicile_state,
      pincode,
      application_no,
      admission_date,
      status,
      mentor_email
    )
    VALUES (
      TRIM(elem->>'roll_number'),
      TRIM(elem->>'full_name'),
      COALESCE(TRIM(elem->>'gender'), 'NA'),
      LOWER(TRIM(elem->>'official_email')),
      LOWER(TRIM(elem->>'personal_email')),
      TRIM(elem->>'mobile_no'),
      TRIM(elem->>'mobile_no'), -- Password defaults to 10-digit mobile number
      COALESCE(TRIM(elem->>'father_name'), 'NA'),
      COALESCE(TRIM(elem->>'father_mobile'), 'NA'),
      COALESCE(TRIM(elem->>'mother_name'), 'NA'),
      COALESCE(TRIM(elem->>'mother_mobile'), 'NA'),
      COALESCE(TRIM(elem->>'section'), 'A'),
      COALESCE(TRIM(elem->>'student_class'), 'BTech CSE Sem 4'),
      COALESCE(TRIM(elem->>'program'), 'B.Tech'),
      COALESCE(TRIM(elem->>'branch'), 'Computer Science & Engineering'),
      COALESCE(TRIM(elem->>'department'), 'Dept. of Computer Science & Technology'),
      COALESCE(TRIM(elem->>'semester'), '4'),
      COALESCE(TRIM(elem->>'semester'), '4'),
      COALESCE(TRIM(elem->>'base_year'), '2025-2026'),
      COALESCE(TRIM(elem->>'domicile_state'), 'NA'),
      COALESCE(TRIM(elem->>'pincode'), 'NA'),
      COALESCE(TRIM(elem->>'application_no'), 'NA'),
      COALESCE(TRIM(elem->>'admission_date'), 'NA'),
      COALESCE(TRIM(elem->>'status'), 'active'),
      LOWER(TRIM(elem->>'mentor_email'))
    )
    ON CONFLICT (roll_number) DO UPDATE SET
      full_name      = EXCLUDED.full_name,
      gender         = EXCLUDED.gender,
      official_email = EXCLUDED.official_email,
      personal_email = EXCLUDED.personal_email,
      mobile_no      = EXCLUDED.mobile_no,
      father_name    = EXCLUDED.father_name,
      father_mobile  = EXCLUDED.father_mobile,
      mother_name    = EXCLUDED.mother_name,
      mother_mobile  = EXCLUDED.mother_mobile,
      section        = EXCLUDED.section,
      student_class  = EXCLUDED.student_class,
      domicile_state = EXCLUDED.domicile_state,
      pincode        = EXCLUDED.pincode,
      application_no = EXCLUDED.application_no,
      admission_date = EXCLUDED.admission_date,
      status         = EXCLUDED.status,
      mentor_email   = COALESCE(EXCLUDED.mentor_email, students.mentor_email),
      last_active    = NOW();

    inserted_count := inserted_count + 1;
  END LOOP;

  RETURN jsonb_build_object(
    'success', true,
    'total_processed', inserted_count
  );
END;
$$;

-- 7. Indexes
CREATE INDEX IF NOT EXISTS idx_students_roll        ON students(roll_number);
CREATE INDEX IF NOT EXISTS idx_students_official_em ON students(official_email);
CREATE INDEX IF NOT EXISTS idx_students_personal_em ON students(personal_email);
CREATE INDEX IF NOT EXISTS idx_students_mentor      ON students(mentor_email);
CREATE INDEX IF NOT EXISTS idx_students_sec_sem     ON students(section, semester);
CREATE INDEX IF NOT EXISTS idx_mentors_email        ON mentors(email);
CREATE INDEX IF NOT EXISTS idx_admins_email         ON admins(email);

-- 8. Disable RLS & Grant Permissions
ALTER TABLE admins   DISABLE ROW LEVEL SECURITY;
ALTER TABLE mentors  DISABLE ROW LEVEL SECURITY;
ALTER TABLE students DISABLE ROW LEVEL SECURITY;

GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL ROUTINES IN SCHEMA public TO anon, authenticated;

SELECT '✅ v11 3-Tier Schema (Admins, Mentors, Students) Ready!' AS status;
