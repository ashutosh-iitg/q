"""SQL for the store: one read round trip and one write transaction per turn."""


LOAD_CONTEXT_SQL = """
WITH s AS (
  INSERT INTO sessions (id, household_id) VALUES (%(session_id)s, %(household_id)s)
  ON CONFLICT (id) DO UPDATE SET last_active_at = now(), status = 'active'
    WHERE sessions.household_id = EXCLUDED.household_id
  RETURNING id, member_hint_id, pending_episode_id, (xmax = 0) AS is_new
),
m AS (SELECT * FROM members WHERE household_id = %(household_id)s)
SELECT
  (SELECT row_to_json(s) FROM s) AS session,
  (SELECT coalesce(json_agg(m ORDER BY m.name), '[]') FROM m) AS members,
  (SELECT coalesce(json_agg(f ORDER BY f.id), '[]')
     FROM member_facts f JOIN m ON m.id = f.member_id
    WHERE f.superseded_by IS NULL) AS facts,
  (SELECT coalesce(json_agg(e ORDER BY e.updated_at DESC), '[]')
     FROM episodes e JOIN m ON m.id = e.member_id
    WHERE e.status <> 'resolved_reported'
       OR e.updated_at > now() - make_interval(days => %(resolved_days)s)) AS episodes,
  (SELECT coalesce(json_agg(x ORDER BY x.id), '[]')
     FROM (SELECT * FROM messages WHERE session_id = %(session_id)s
           ORDER BY id DESC LIMIT %(recent)s) x) AS messages,
  (SELECT coalesce(json_agg(y ORDER BY y.closed_at), '[]')
     FROM (SELECT id AS session_id, summary, closed_at FROM sessions
            WHERE household_id = %(household_id)s AND status = 'consolidated' AND summary IS NOT NULL
            ORDER BY closed_at DESC LIMIT %(summaries)s) y) AS summaries
"""


UPSERT_MEMBER_SQL = """
INSERT INTO members (id, household_id, name, aliases, age_years, conditions)
VALUES (%(id)s, %(household_id)s, %(name)s, %(aliases)s, %(age_years)s, %(conditions)s)
ON CONFLICT (id) DO UPDATE SET household_id = EXCLUDED.household_id, name = EXCLUDED.name,
  aliases = EXCLUDED.aliases, age_years = EXCLUDED.age_years, conditions = EXCLUDED.conditions
"""


INSERT_MESSAGE_SQL = """
INSERT INTO messages (session_id, turn_id, role, content, member_id, episode_id, decision)
VALUES (%(session_id)s, %(turn_id)s, %(role)s, %(content)s, %(member_id)s, %(episode_id)s, %(decision)s)
ON CONFLICT (session_id, turn_id, role) DO NOTHING
RETURNING id
"""


UPSERT_EPISODE_SQL = """
INSERT INTO episodes (id, member_id, complaint, status, slots, questions_asked,
                      check_back_after, related_episode_id, protocol_id, findings, associated, urgency,
                      handoff_id)
VALUES (%(id)s, %(member_id)s, %(complaint)s, %(status)s, %(slots)s, %(questions_asked_delta)s,
        %(check_back_after)s, %(related_episode_id)s, %(protocol_id)s, coalesce(%(findings)s::jsonb, '{}'),
        coalesce(%(associated)s::jsonb, '{}'), %(urgency)s, %(handoff_id)s)
ON CONFLICT (id) DO UPDATE SET
  status = EXCLUDED.status,
  slots = episodes.slots || EXCLUDED.slots,
  protocol_id = coalesce(EXCLUDED.protocol_id, episodes.protocol_id),
  findings = coalesce(%(findings)s::jsonb, episodes.findings),
  associated = coalesce(%(associated)s::jsonb, episodes.associated),
  urgency = coalesce(EXCLUDED.urgency, episodes.urgency),
  handoff_id = coalesce(EXCLUDED.handoff_id, episodes.handoff_id),
  questions_asked = episodes.questions_asked + EXCLUDED.questions_asked,
  check_back_after = EXCLUDED.check_back_after,
  updated_at = now()
  WHERE episodes.member_id = EXCLUDED.member_id
RETURNING id
"""


INSERT_FACT_SQL = """
INSERT INTO member_facts (member_id, session_id, episode_id, kind, subject, text, effective_at,
                          status, source, source_message_id)
VALUES (%(member_id)s, %(session_id)s, %(episode_id)s, %(kind)s, %(subject)s, %(text)s, %(effective_at)s,
        %(status)s, 'user_report', %(source_message_id)s)
RETURNING id
"""


CLOSE_IDLE_SESSIONS_SQL = """
WITH closed AS (
  UPDATE sessions SET status = 'closed', closed_at = now()
  WHERE household_id = %(household_id)s AND id <> %(active_session_id)s AND status = 'active'
    AND last_active_at < now() - make_interval(mins => %(idle_minutes)s)
  RETURNING id
)
SELECT id FROM closed
UNION  -- also retry sessions whose earlier consolidation failed
SELECT id FROM sessions
WHERE household_id = %(household_id)s AND id <> %(active_session_id)s AND status = 'closed'
"""


CLOSE_SESSION_SQL = """
UPDATE sessions SET status = 'closed', closed_at = now()
WHERE id = %(session_id)s AND household_id = %(household_id)s AND status = 'active'
"""


LOAD_CONSOLIDATION_SQL = """
WITH s AS (SELECT * FROM sessions WHERE id = %(session_id)s AND status = 'closed'),
m AS (SELECT members.* FROM members JOIN s USING (household_id))
SELECT
  (SELECT row_to_json(s) FROM s) AS session,
  (SELECT coalesce(json_agg(m ORDER BY m.name), '[]') FROM m) AS members,
  (SELECT coalesce(json_agg(f ORDER BY f.id), '[]')
     FROM member_facts f JOIN m ON m.id = f.member_id
    WHERE f.superseded_by IS NULL) AS facts,
  (SELECT coalesce(json_agg(e ORDER BY e.updated_at DESC), '[]')
     FROM episodes e JOIN m ON m.id = e.member_id
    WHERE e.status <> 'resolved_reported'
       OR e.updated_at > now() - make_interval(days => %(resolved_days)s)
       OR e.id IN (SELECT episode_id FROM messages WHERE session_id = %(session_id)s)) AS episodes,
  (SELECT coalesce(json_agg(x ORDER BY x.id), '[]')
     FROM messages x WHERE x.session_id = %(session_id)s) AS messages,
  (SELECT coalesce(json_agg(y ORDER BY y.closed_at), '[]')
     FROM (SELECT p.id AS session_id, p.summary, p.closed_at FROM sessions p JOIN s USING (household_id)
            WHERE p.status = 'consolidated' AND p.summary IS NOT NULL AND p.id <> s.id
            ORDER BY p.closed_at DESC LIMIT %(summaries)s) y) AS summaries
"""


MARK_CONSOLIDATED_SQL = """
UPDATE sessions SET status = 'consolidated', summary = %(summary)s
WHERE id = %(session_id)s AND status = 'closed'
"""


PROMOTE_FACT_SQL = """
UPDATE member_facts SET status = 'consolidated', subject = %(subject)s
WHERE id = %(fact_id)s AND session_id = %(session_id)s AND status = 'provisional' AND superseded_by IS NULL
"""


MARK_DUPLICATE_SQL = """
UPDATE member_facts f SET superseded_by = kept.id
FROM member_facts kept
WHERE f.id = %(fact_id)s AND f.session_id = %(session_id)s AND f.superseded_by IS NULL
  AND kept.id = %(kept_id)s AND kept.member_id = f.member_id AND kept.superseded_by IS NULL
"""


REPLACE_FACT_SQL = """
UPDATE member_facts old SET superseded_by = new.id
FROM member_facts new
WHERE new.id = %(fact_id)s AND new.session_id = %(session_id)s
  AND old.id = %(replaced_id)s AND old.member_id = new.member_id AND old.superseded_by IS NULL
"""


SUPERSEDE_FACT_SQL = """
UPDATE member_facts SET superseded_by = %(new_id)s
WHERE id = %(old_id)s AND member_id = %(member_id)s AND superseded_by IS NULL
"""


UPDATE_SESSION_SQL = """
UPDATE sessions SET member_hint_id = coalesce(%(member_id)s, member_hint_id),
  pending_episode_id = %(pending_episode_id)s, last_active_at = now()
WHERE id = %(session_id)s AND household_id = %(household_id)s
"""
