-- Q memory schema. Applied at startup; every statement is idempotent.

CREATE TABLE IF NOT EXISTS members (
  id            TEXT PRIMARY KEY,
  household_id  TEXT NOT NULL,
  name          TEXT NOT NULL,
  aliases       JSONB NOT NULL DEFAULT '[]',   -- "my son", "dad", "grandma"
  age_years     NUMERIC,
  conditions    JSONB NOT NULL DEFAULT '[]'    -- seeded only, e.g. ["hypertension"]
);

CREATE TABLE IF NOT EXISTS sessions (
  id                  TEXT PRIMARY KEY,
  household_id        TEXT NOT NULL,
  status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'closed', 'consolidated')),
  member_hint_id      TEXT REFERENCES members(id),  -- not an override
  pending_episode_id  TEXT,                          -- what the pending question is about
  summary             TEXT,                          -- written by consolidation
  started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_active_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at           TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS messages (
  id          BIGSERIAL PRIMARY KEY,
  session_id  TEXT NOT NULL REFERENCES sessions(id),
  turn_id     TEXT NOT NULL,
  role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content     TEXT NOT NULL,
  member_id   TEXT REFERENCES members(id),
  episode_id  TEXT,
  decision    TEXT,                                  -- assistant rows: emergency, doctor, ...
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT messages_session_turn_role UNIQUE (session_id, turn_id, role)
);

-- Upgrade the earlier global turn key without dropping any messages.
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_turn_id_role_key;
CREATE UNIQUE INDEX IF NOT EXISTS messages_session_turn_role ON messages (session_id, turn_id, role);

CREATE TABLE IF NOT EXISTS episodes (
  id                  TEXT PRIMARY KEY,
  member_id           TEXT NOT NULL REFERENCES members(id),
  complaint           TEXT NOT NULL,
  status              TEXT NOT NULL CHECK (status IN
                        ('triaging', 'monitoring', 'doctor', 'emergency', 'resolved_reported')),
  slots               JSONB NOT NULL DEFAULT '{}',  -- triage details gathered so far
  questions_asked     INT NOT NULL DEFAULT 0,
  check_back_after    TIMESTAMPTZ,                  -- set when home monitoring starts
  related_episode_id  TEXT REFERENCES episodes(id), -- a return after reported resolution
  opened_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Protocol-driven intake (2026-10-28): which protocol an episode follows and its answers so far.
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS protocol_id TEXT;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS findings JSONB NOT NULL DEFAULT '{}';
-- Other symptoms of the same illness and whether each is still there (2026-10-29), e.g. a fever
-- that came with a cough and has gone: {symptom: {status: active|resolved, quote, at}}.
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS associated JSONB NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS member_facts (
  id                 BIGSERIAL PRIMARY KEY,
  member_id          TEXT NOT NULL REFERENCES members(id),
  session_id         TEXT REFERENCES sessions(id),     -- null for seeded or clinician facts
  episode_id         TEXT REFERENCES episodes(id),
  kind               TEXT NOT NULL CHECK (kind IN
                       ('condition', 'symptom', 'observation', 'medication', 'allergy', 'preference', 'note')),
  subject            TEXT NOT NULL CHECK (subject ~ '^[a-z0-9]+(_[a-z0-9]+)*$'),  -- fever, blood_pressure
  text               TEXT NOT NULL,                    -- the evidence, as reported
  effective_at       TIMESTAMPTZ,                      -- when it was true or measured, if known
  status             TEXT NOT NULL DEFAULT 'provisional'
                       CHECK (status IN ('provisional', 'consolidated', 'verified')),
  source             TEXT NOT NULL CHECK (source IN ('user_report', 'seeded', 'clinician')),
  source_message_id  BIGINT REFERENCES messages(id),
  superseded_by      BIGINT REFERENCES member_facts(id),
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (status <> 'verified' OR source <> 'user_report')  -- a user report is never verified
);

CREATE INDEX IF NOT EXISTS members_household ON members (household_id);
CREATE INDEX IF NOT EXISTS sessions_household_status ON sessions (household_id, status, last_active_at);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session_id, id);
CREATE INDEX IF NOT EXISTS messages_episode ON messages (episode_id, id) WHERE episode_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS episodes_member_status ON episodes (member_id, status);
CREATE INDEX IF NOT EXISTS member_facts_member ON member_facts (member_id, subject) WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS member_facts_session ON member_facts (session_id);
