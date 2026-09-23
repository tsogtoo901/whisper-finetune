import os, io, csv, math, datetime, zipfile, unicodedata, secrets, threading, concurrent.futures, json
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_file, abort, Response, stream_with_context)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, boto3
from botocore.client import Config

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production-please")

B2_KEY_ID      = os.environ.get("B2_KEY_ID", "")
B2_APP_KEY     = os.environ.get("B2_APP_KEY", "")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "mnkh-dataset")
B2_ENDPOINT    = os.environ.get("B2_ENDPOINT", "")
DB_PATH        = os.environ.get("DB_PATH", "instance/mnkh.db")
# Public origin used to build buyer-facing links (the /d/<token> landing page).
# Defaults to the current serving host. Point this at a branded custom domain
# (e.g. https://mongoliandata.com) by setting the env var once it is attached to
# Railway — no code change needed. Trailing slash stripped so we can append paths.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://mgl-speech-portal.us").rstrip("/")
RATE_PER_HOUR  = 20000  # ₮ per hour

# OpenAI configuration. Key is set per-environment (Railway env var).
# Model is hardcoded for now — gpt-4o gives strong Mongolian output.
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o")

def get_openai_client():
    """Lazy import so the dependency is optional in dev environments without a key.
    Returns None if the key is unset, so callers can show a friendly error."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        return None

def get_b2():
    return boto3.client("s3", endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID, aws_secret_access_key=B2_APP_KEY,
        config=Config(signature_version="s3v4"))

def get_db():
    # SQLite needs explicit concurrency settings; the defaults assume a single writer and
    # raise "database is locked" the moment two workers contend. WAL lets readers and one
    # writer coexist without blocking each other, and busy_timeout makes the client wait
    # for a lock for up to 15s instead of erroring instantly. Together they turn a hot
    # contention failure into a brief queue delay, which is what we want under gunicorn's
    # multi-worker setup. NORMAL sync is the WAL-recommended durability level — full fsync
    # on every commit isn't necessary and was a major source of contention.
    db = sqlite3.connect(DB_PATH, timeout=15.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA busy_timeout=15000")
    return db

# ══ Conversational speech collection — constants (Stage A) ═══════════════════
# State machine (§8.3). Only these transitions are legal; _conv_transition enforces.
CONV_STATES = ("created","recording","uploaded","drafting","drafted","in_edit",
               "in_review","approved","rejected","aborted","incomplete")
CONV_TRANSITIONS = {
    "created":    {"recording","aborted"},
    "recording":  {"uploaded","aborted","incomplete"},
    "incomplete": {"uploaded","aborted"},           # re-upload from the same page (§4.5)
    "uploaded":   {"drafting","aborted"},
    "drafting":   {"drafted","uploaded"},           # draft job may fail → back to uploaded
    "drafted":    {"in_edit"},
    "rejected":   {"in_edit"},                      # only drafted/rejected may enter in_edit
    "in_edit":    {"in_review"},
    "in_review":  {"approved","rejected"},
    "approved":   set(),
    "aborted":    set(),
}
# Statuses that COUNT toward caps/pairing (§6): everything except the three
# non-participation outcomes. In-progress sessions count too, so caps cannot be
# gamed by opening several sessions before any upload completes.
CONV_COUNTING_STATUSES = ("created","recording","uploaded","drafting","drafted",
                          "in_edit","in_review","approved")
CONV_CAP_MINUTES        = 600.0   # 10 h lifetime cap per speaker on conversations (separate from read speech)
# Hours-cap switch. OFF for now (testing / early collection). Set Railway env var
# CONV_CAP_ENABLED=1 to enforce the cap again and show remaining minutes in the UI.
# Partner-diversity rules below are NOT affected by this switch.
CONV_CAP_ENABLED        = os.environ.get("CONV_CAP_ENABLED", "0").strip() == "1"
# Read-speech lifetime cap (separate 10 h budget). Same switch pattern; OFF until the
# conversational pipeline is fully operational — enabling it now would block speakers who
# already exceed 10 h of read speech before they can move on to conversations.
RS_CAP_MINUTES          = 600.0
RS_CAP_ENABLED          = os.environ.get("RS_CAP_ENABLED", "0").strip() == "1"
CONV_PLANNED_MINUTES    = 20.0    # planning unit for cap checks at creation (§6)
CONV_MIN_MINUTES        = 8.0     # recording rules (§4.4)
CONV_MAX_MINUTES        = 25.0
CONV_PARTNER_SHARE_MAX  = 0.25    # ≤25% of minutes with one partner once ≥3 sessions
CONV_SAME_PARTNER_MAX   = 2       # ≤2 sessions with same partner until 3 distinct partners
CONV_DISTINCT_PARTNERS  = 3

# ── DRAFT Mongolian texts — authored by the assistant at the admin's instruction
# (2026-09-08) for correction by a native speaker during the test run. NOT certified.
CONV_PRIVACY_WARNING_MN = (
    "[ДРАФТ — засварлах шаардлагатай] Бусад хүний бодит нэр, утасны дугаар, гэрийн хаяг, "
    "регистрийн дугаар, ажлын газрын нэрийг ярианы явцад хэлэхгүй байхыг анхаарна уу. "
    "Ердийн, чөлөөтэй яриа өрнүүлнэ үү.")
CONV_START_INSTRUCTION_MN = (
    "[ДРАФТ] Эхлээд «Холбогдох» дараад хамтрагчаа сонсоно уу. Дараа нь аль нэг нь "
    "«Бичлэг эхлүүлэх» дарахад хоёр талд нэгэн зэрэг бичлэг эхэлнэ — нэг зэрэг дарах шаардлагагүй.")
CONV_FILLER_SEED_DRAFT = ["өө", "ээ", "аа", "мм", "хмм", "тэгээд", "яахав", "юу гэдэг юм"]

# Admin-editable system prompt for topic generation (seeded once into conv_topic_settings).
CONV_DRAFT_MODEL   = os.environ.get("CONV_DRAFT_MODEL", "whisper-1")   # OpenAI transcription model
CONV_AUTO_DRAFT    = os.environ.get("CONV_AUTO_DRAFT", "1") != "0"        # draft automatically after upload
CONV_JOB_API_KEY   = os.environ.get("CONV_JOB_API_KEY", "")               # external draft job (spec §7); empty = disabled
CONV_DRAFT_CHUNK_SEC = 600                                                # 10-min chunks at 16 kHz (~19 MB) per API call
CONV_EDIT_DEADLINE_DAYS = 7
# [ДРАФТ — засварлах шаардлагатай] speaker self-correction rules (shown on the edit screen)
CONV_SPEAKER_RULES_MN = [
    "Хэлсэн үгийг ЯГ БАЙГААГААР нь бичнэ. Өө, ээ, аа, давталт, дутуу хэлсэн үг (хэ-) зэргийг засаж хаяхгүй, хэвээр үлдээнэ.",
    "Дүрмийн алдаа, ярианы хэллэгийг «зөв» болгож засахгүй — юу хэлснийг бичнэ, юу хэлэх ёстой байсныг биш.",
    "Ярианы ДУНД ханиалга, инээд, чанга чимээ орвол яг тэр газарт [ханиалга] / [инээд] / [чимээ] гэж тэмдэглэнэ (доорх товчнуудаар). Өөр тэмдэг хэрэглэхгүй.",
    "Хамрын татах, амьсгал, жижиг сэрчигнээн зэргийг огт тэмдэглэхгүй — алгасна.",
    "Мөр бүхэлдээ инээд, ханиалга, чимээ бол юу ч бичихгүй, «Яриа биш» гэж тэмдэглэнэ.",
    "Гуравдагч хүний (өрөөнд байгаа өөр хүний) үгийг ХЭЗЭЭ Ч бичихгүй. Ярьж байгаа хүнтэй давхцвал мөрийг «Дуу муу», зөвхөн тэр хүн сонсогдвол «Яриа биш» гэж тэмдэглэнэ.",
    "Хамтрагчийн дуу чихэвчнээс бүдэг сонсогдож байвал алгасна — бичихгүй.",
    "Бусад хүний бодит нэр, утас, хаяг, регистр сонсогдвол тэр мөрийг «Нууц» гэж тэмдэглэнэ (текстийг устгах шаардлагагүй).",
    "Сайн сонсогдохгүй үг байвал мөрийг «Тодорхойгүй» гэж тэмдэглэнэ.",
    "Мөр бүрийг сонсоод зөв бол «Зөв» дарж, буруу бол засаад дараагийн мөр рүү шилжинэ.",
]
CONV_EVENT_TOKENS = ["[ханиалга]", "[инээд]", "[чимээ]"]   # the ONLY inline non-speech markers; stripped or kept at delivery per buyer

# ICE servers for in-page calling. Mobile carriers usually need a TURN relay.
#   Cloudflare Calls TURN: set CF_TURN_KEY_ID + CF_TURN_API_TOKEN (short-lived creds generated here).
#   Static TURN (Metered/Twilio/coturn): TURN_URLS (comma-separated), TURN_USERNAME, TURN_CREDENTIAL.
#   Neither set: STUN only (works on many Wi-Fi links, often NOT on cellular).
CF_TURN_KEY_ID    = os.environ.get("CF_TURN_KEY_ID", "")
CF_TURN_API_TOKEN = os.environ.get("CF_TURN_API_TOKEN", "")
TURN_URLS         = [u.strip() for u in os.environ.get("TURN_URLS", "").split(",") if u.strip()]
TURN_USERNAME     = os.environ.get("TURN_USERNAME", "")
TURN_CREDENTIAL   = os.environ.get("TURN_CREDENTIAL", "")
STUN_URLS         = ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]
_ICE_CACHE = {"servers": None, "exp": 0.0}

CONV_INVITE_DAYS       = 7     # a pending invitation lapses after this
CONV_MAX_OPEN_INVITES  = 5     # per speaker, outgoing pending

# Loudspeaker-leak check (post-upload): with earphones, a track drops to room noise
# (25-40 dB below own speech) whenever the partner speaks; through a loudspeaker the
# partner's voice lands in it only ~10 dB below own speech. Measured as that contrast,
# per side, alignment-aware — works even in pause-free conversations.
CONV_LEAK_CONTRAST_DB    = 15.0    # a track must drop at least this far below its own speech level while ONLY the partner speaks
CONV_LEAK_ANALYZE_SEC    = 600     # analyze up to the first 10 minutes (leak is persistent)
CONV_LEAK_MIN_FRAMES     = 150     # need >= 15 s of partner-speaking frames to score a side

# Test-phase gate for speaker-facing conversation pages. Set CONV_TEST_PASSWORD="" to
# remove the gate without a code change.
CONV_TEST_PASSWORD = os.environ.get("CONV_TEST_PASSWORD", "conversation")

CONV_TOPIC_GEN_DEFAULT_PROMPT = (
    "You generate conversation topics for a Mongolian speech dataset. Two adult native "
    "speakers of Khalkha Mongolian will talk to each other over a phone/video call for "
    "8-25 minutes on the topic. Output ONLY a JSON array of objects with keys "
    "\"title\" and \"prompt\", both written in natural, everyday Mongolian (Cyrillic). "
    "Requirements: (1) topics must create back-and-forth — planning, deciding, comparing, "
    "mild disagreement, giving each other advice — not two monologues; (2) everyday, "
    "safe, apolitical, non-religious, nothing sexual or violent; (3) NEVER invite the "
    "speakers to discuss other identifiable people's private details (health, money, "
    "family problems, workplaces, addresses); keep the focus on the speakers' own "
    "opinions and generic experiences; (4) the prompt gives a concrete task or set of "
    "questions (2-4 sentences) so the pair never runs out of things to say; (5) vary "
    "domains: food, travel, weekend plans, work in general, weather and seasons, hobbies, "
    "childhood memories, city life, sports, technology, shopping, transport, education, "
    "traditions. No numbering, no commentary, JSON only.")

def init_db():
    os.makedirs("instance", exist_ok=True)
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            speaker_id TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL DEFAULT 'contributor',
            frozen INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_id TEXT UNIQUE NOT NULL,
            full_name TEXT, age INTEGER, gender TEXT, region TEXT,
            dialect TEXT DEFAULT 'Khalkha',
            native_language TEXT DEFAULT 'Mongolian',
            education_level TEXT, phone TEXT,
            photo_b2_key TEXT,
            bank_name TEXT, iban TEXT, receiver_name TEXT,
            completed INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_id TEXT NOT NULL,
            clip_number INTEGER NOT NULL,
            filename TEXT NOT NULL,
            text_mn TEXT NOT NULL,
            text_en TEXT,
            speech_type TEXT NOT NULL DEFAULT 'RS',
            subcategory TEXT DEFAULT '',
            UNIQUE(speaker_id, clip_number)
        );
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_id TEXT NOT NULL,
            prompt_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            b2_key TEXT,
            duration_seconds REAL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            reject_note TEXT,
            compensated INTEGER DEFAULT 0,
            submitted_at TEXT DEFAULT (datetime('now')),
            reviewed_at TEXT,
            FOREIGN KEY(prompt_id) REFERENCES prompts(id)
        );
        CREATE TABLE IF NOT EXISTS deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            buyer_name TEXT NOT NULL,
            buyer_id INTEGER,
            clip_count INTEGER DEFAULT 0,
            total_seconds REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(buyer_id) REFERENCES buyers(id)
        );
        CREATE TABLE IF NOT EXISTS delivery_clips (
            delivery_id INTEGER NOT NULL,
            clip_id INTEGER NOT NULL,
            PRIMARY KEY(delivery_id, clip_id),
            FOREIGN KEY(delivery_id) REFERENCES deliveries(id),
            FOREIGN KEY(clip_id) REFERENCES clips(id)
        );
        -- Buyers: tracks who has received which clips so we can deliver the same
        -- audio to multiple buyers (each gets a fresh, unseen copy) without ever
        -- sending the same clip twice to the same buyer.
        CREATE TABLE IF NOT EXISTS buyers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            contact TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        -- Editor system tables (Stage 1)
        CREATE TABLE IF NOT EXISTS editor_contributors (
            editor_user_id INTEGER NOT NULL,
            contributor_speaker_id TEXT NOT NULL,
            assigned_at TEXT DEFAULT (datetime('now')),
            assigned_by TEXT DEFAULT 'admin',
            PRIMARY KEY(editor_user_id, contributor_speaker_id),
            FOREIGN KEY(editor_user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS editor_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            editor_user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(editor_user_id) REFERENCES users(id)
        );
        -- Chat between a contributor and their CURRENT editor (resolved live via
        -- editor_contributors), plus the admin who can monitor any thread and join in.
        -- Thread key = contributor_speaker_id, so reassigning a contributor to a new
        -- editor seamlessly hands the thread to that editor with full history.
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contributor_speaker_id TEXT NOT NULL,
            sender_role TEXT NOT NULL,           -- 'contributor' | 'editor' | 'admin'
            sender_user_id INTEGER,
            body TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            read_by_contributor INTEGER DEFAULT 0,
            read_by_editor INTEGER DEFAULT 0,
            read_by_admin INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS contributor_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            editor_user_id INTEGER NOT NULL,
            note TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            decided_at TEXT,
            FOREIGN KEY(editor_user_id) REFERENCES users(id)
        );
        -- Per-editor LLM prompt generation settings. Admin sets these from the
        -- Editors page so each editor's contributors get topically uniform prompts.
        -- Editor sees only a button — admin owns the configuration entirely.
        CREATE TABLE IF NOT EXISTS editor_generation_settings (
            editor_user_id INTEGER PRIMARY KEY,
            speech_type TEXT NOT NULL DEFAULT 'RS',
            subcategory TEXT NOT NULL DEFAULT '',
            system_prompt TEXT NOT NULL DEFAULT '',
            batch_size INTEGER NOT NULL DEFAULT 50,
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(editor_user_id) REFERENCES users(id)
        );
        -- Payment transactions: one row per "mark paid" action per recipient.
        -- Inserted by mark_compensated, mark_editor_paid, and mark_all_paid.
        -- Read by the /earnings page so contributors and editors can see their
        -- own all-time payout history and total earnings.
        CREATE TABLE IF NOT EXISTS payment_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_role TEXT NOT NULL,
            amount INTEGER NOT NULL,
            clip_count INTEGER NOT NULL DEFAULT 0,
            duration_seconds REAL NOT NULL DEFAULT 0,
            paid_at TEXT DEFAULT (datetime('now')),
            paid_by_username TEXT,
            method TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_payment_tx_user
            ON payment_transactions(user_id, paid_at);
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            position TEXT NOT NULL,
            is_adult INTEGER NOT NULL DEFAULT 0,
            age TEXT,
            email TEXT,
            city TEXT,
            facebook TEXT,
            hours_per_week TEXT,
            prior_experience TEXT,
            heard_from TEXT,
            message TEXT,
            submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
            ip_address TEXT,
            user_agent TEXT,
            status TEXT NOT NULL DEFAULT 'new',
            admin_notes TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_applications_status
            ON applications(status, submitted_at DESC);
    """)
    # Migrations - existing
    try: db.execute("ALTER TABLE prompts ADD COLUMN subcategory TEXT DEFAULT ''")
    except Exception: pass
    for col, typ in [("users","frozen INTEGER DEFAULT 0"),
                     ("clips","duration_seconds REAL DEFAULT 0"),
                     ("clips","compensated INTEGER DEFAULT 0"),
                     ("profiles","bank_name TEXT"),
                     ("profiles","iban TEXT"),
                     ("profiles","receiver_name TEXT")]:
        try: db.execute(f"ALTER TABLE {col} ADD COLUMN {typ}")
        except Exception: pass
    # Migrations - editor system (Stage 1)
    # Per-user hourly rate. Defaults: contributor=20000 (legacy), admin/editor=0
    # For new contributors created after this migration, the new_contributor route assigns 35000.
    # Existing contributors keep their grandfathered 20000 rate until admin changes it.
    for col, typ in [("users", "hourly_rate INTEGER DEFAULT 20000"),
                     ("users", "archived INTEGER DEFAULT 0"),
                     ("users", "archived_at TEXT"),
                     ("clips", "editor_user_id INTEGER"),
                     ("clips", "contributor_rate_at_approval INTEGER"),
                     ("clips", "editor_rate_at_approval INTEGER"),
                     ("clips", "editor_paid INTEGER DEFAULT 0"),
                     ("users", "editor_penalty_pct INTEGER DEFAULT 0"),
                     ("users", "editor_penalty_reason TEXT"),
                     ("users", "editor_penalty_at TEXT"),
                     ("clips", "editor_reviewed_at TEXT"),
                     ("clips", "admin_reviewed_at TEXT"),
                     ("prompts", "created_by_user_id INTEGER"),
                     ("prompts", "text_normalized TEXT"),
                     ("profiles", "consent_b2_key TEXT"),
                     ("profiles", "consent_uploaded_at TEXT"),
                     ("profiles", "recording_device TEXT"),
                     ("profiles", "dob TEXT"),
                     ("profiles", "consent_affirmed_at TEXT"),
                     ("profiles", "consent_affirm_ip TEXT"),
                     ("profiles", "consent_method TEXT"),
                     ("profiles", "consent_version TEXT"),
                     ("profiles", "ocr_count_day TEXT"),
                     ("profiles", "ocr_count_n INTEGER DEFAULT 0"),
                     ("profiles", "locked INTEGER DEFAULT 0"),
                     ("profiles", "locked_at TEXT"),
                     ("profiles", "locked_by TEXT"),
                     ("deliveries", "buyer_id INTEGER"),
                     ("deliveries", "build_status TEXT DEFAULT 'pending'"),
                     ("deliveries", "build_error TEXT"),
                     ("deliveries", "build_progress INTEGER DEFAULT 0"),
                     ("deliveries", "zip_b2_key TEXT"),
                     ("deliveries", "zip_size_bytes INTEGER"),
                     ("deliveries", "download_token TEXT")]:
        try: db.execute(f"ALTER TABLE {col} ADD COLUMN {typ}")
        except Exception: pass

    # One-time wipe of the legacy "lock-in" deliveries data.
    # Old model: every approved clip could only be in ONE delivery (lock-in).
    # New model: same clip can be sold to multiple buyers as fresh. Old test rows
    # are incompatible (no buyer_id) and harmless to drop since they're all from testing.
    # Wipe runs ONCE — controlled by a sentinel row in a tiny one-time-flags table.
    db.execute("CREATE TABLE IF NOT EXISTS _one_time_flags (key TEXT PRIMARY KEY, applied_at TEXT)")
    flag = db.execute("SELECT 1 FROM _one_time_flags WHERE key='wipe_legacy_deliveries_v1'").fetchone()
    if not flag:
        legacy_count = db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
        db.execute("DELETE FROM delivery_clips")
        db.execute("DELETE FROM deliveries")
        db.execute("INSERT INTO _one_time_flags (key, applied_at) VALUES (?, datetime('now'))",
                   ("wipe_legacy_deliveries_v1",))
        if legacy_count > 0:
            print(f"[INIT] Wiped {legacy_count} legacy delivery row(s) — new buyer-aware model takes over")

    # Second one-time wipe for the pre-built-zip migration. Existing deliveries
    # (from the buyer-aware-but-sync-zip era) don't have zip_b2_key or download_token,
    # so they can't serve buyer links and have no pre-built zip to download. Since
    # admin is still in testing mode and has explicitly OK'd wiping between iterations,
    # we drop them here. Future deliveries will be fully token-and-zip aware.
    flag2 = db.execute("SELECT 1 FROM _one_time_flags WHERE key='wipe_for_prebuilt_zips_v1'").fetchone()
    if not flag2:
        legacy_count = db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
        db.execute("DELETE FROM delivery_clips")
        db.execute("DELETE FROM deliveries")
        db.execute("INSERT INTO _one_time_flags (key, applied_at) VALUES (?, datetime('now'))",
                   ("wipe_for_prebuilt_zips_v1",))
        if legacy_count > 0:
            print(f"[INIT] Wiped {legacy_count} delivery row(s) for pre-built-zip migration")

    # Unique index on download_token. Skipped when token is NULL (partial index)
    # so rows in 'pending' state without a token yet don't collide.
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_deliveries_download_token "
               "ON deliveries(download_token) WHERE download_token IS NOT NULL")
    # Index for fast cross-speaker duplicate lookups. Not UNIQUE because we want
    # the dedupe report to surface existing duplicates rather than fail at startup.
    db.execute("CREATE INDEX IF NOT EXISTS idx_prompts_text_norm ON prompts(text_normalized)")

    # Indexes on clips. The clips table had none, so the admin dashboard's per-contributor
    # COUNT/SUM-by-status subqueries and the todo_count NOT EXISTS check were each doing a
    # full table scan of clips — tens of millions of row reads per page load. These turn
    # those into index lookups. All IF NOT EXISTS, so safe and idempotent.
    db.execute("CREATE INDEX IF NOT EXISTS idx_clips_prompt ON clips(prompt_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_clips_speaker_status ON clips(speaker_id, status)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_clips_editor_status ON clips(editor_user_id, status)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status)")

    # Background bulk-job tracking (bulk approve runs in a thread; progress lives in the
    # DB so it survives Cloudflare timeouts and is visible from any gunicorn worker).
    db.execute(
        "CREATE TABLE IF NOT EXISTS bulk_jobs ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  kind TEXT NOT NULL,"
        "  scope TEXT,"
        "  total INTEGER DEFAULT 0, done INTEGER DEFAULT 0,"
        "  approved INTEGER DEFAULT 0, moved INTEGER DEFAULT 0, reconciled INTEGER DEFAULT 0,"
        "  missing INTEGER DEFAULT 0, errors INTEGER DEFAULT 0,"
        "  status TEXT DEFAULT 'running',"
        "  error TEXT,"
        "  result_json TEXT,"
        "  started_at TEXT DEFAULT (datetime('now')),"
        "  finished_at TEXT)")
    # Append-only consent affirmation trail (agreement v2.0+). One row per affirmation
    # event that ACTUALLY happened — rows are never synthesized, updated, or deleted
    # (E.1: every prior version's record is preserved). The only later write allowed is
    # stamping download_copy_delivered_at on the speaker's own first download.
    db.execute(
        "CREATE TABLE IF NOT EXISTS consent_records ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  speaker_id TEXT NOT NULL,"
        "  user_id INTEGER,"
        "  agreement_version TEXT NOT NULL,"
        "  text_sha256 TEXT NOT NULL,"
        "  affirmation_wording TEXT,"
        "  checkbox_affirmed INTEGER NOT NULL DEFAULT 0,"
        "  affirmed_at TEXT DEFAULT (datetime('now')),"
        "  session_ref TEXT,"
        "  ip TEXT,"
        "  device_info TEXT,"
        "  id_verification_result TEXT,"
        "  pdf_b2_key TEXT,"
        "  download_copy_delivered_at TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_consent_records_speaker "
               "ON consent_records(speaker_id)")
    # Append-only audit of admin corrections to the ID-OCR'd name (e.g. the Ү/У
    # misread). The speaker never edits their own name; only an admin can, against
    # the ID image, and every change is recorded with who/when/old/new.
    # ── Conversational speech collection (Stage A) — all additive, RS untouched ──
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_topics ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  title_mn TEXT NOT NULL,"
        "  prompt_mn TEXT NOT NULL,"
        "  status TEXT NOT NULL DEFAULT 'draft',"      # draft | active | retired
        "  source TEXT NOT NULL DEFAULT 'generated',"  # generated | manual
        "  use_count INTEGER NOT NULL DEFAULT 0,"
        "  created_at TEXT DEFAULT (datetime('now')),"
        "  updated_at TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_topic_settings ("
        "  id INTEGER PRIMARY KEY CHECK (id=1),"
        "  system_prompt TEXT NOT NULL,"
        "  updated_at TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_sessions ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_code TEXT UNIQUE NOT NULL,"
        "  topic_id INTEGER,"
        "  status TEXT NOT NULL DEFAULT 'created',"
        "  speaker_a_id TEXT NOT NULL,"
        "  speaker_b_id TEXT NOT NULL,"
        "  invite_code TEXT,"
        "  created_by TEXT,"
        "  created_at TEXT DEFAULT (datetime('now')),"
        "  recording_started_at TEXT,"
        "  uploaded_at TEXT,"
        "  drafting_started_at TEXT,"
        "  drafted_at TEXT,"
        "  editor_id INTEGER,"
        "  reviewer_id INTEGER,"
        "  reject_reason TEXT,"
        "  approved_at TEXT,"
        "  aborted_at TEXT,"
        "  incomplete_at TEXT,"
        "  duration_sec REAL,"
        "  model_version TEXT,"
        "  alignment_offset_json TEXT,"
        "  draft_error TEXT,"
        "  leak_corr REAL,"
        "  abort_reason TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_conv_sessions_a ON conv_sessions(speaker_a_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_conv_sessions_b ON conv_sessions(speaker_b_id)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_tracks ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL,"
        "  speaker_id TEXT NOT NULL,"
        "  channel TEXT NOT NULL,"                     # A | B
        "  b2_key TEXT,"
        "  client_start_ms INTEGER,"
        "  client_stop_ms INTEGER,"
        "  duration_sec REAL,"
        "  size_bytes INTEGER,"
        "  rms REAL,"
        "  check_status TEXT DEFAULT 'pending',"       # pending | ok | rejected
        "  check_error TEXT,"
        "  uploaded_at TEXT,"
        "  upload_id TEXT,"
        "  edit_status TEXT DEFAULT 'pending',"
        "  edit_done_at TEXT,"
        "  clock_skew_ms INTEGER,"
        "  gain_db REAL,"
        "  UNIQUE(session_id, channel))")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_affirmations ("   # append-only (§9.3)
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL,"
        "  speaker_id TEXT NOT NULL,"
        "  affirmed_at TEXT DEFAULT (datetime('now')),"
        "  ip TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_pairing_overrides ("   # append-only (§6)
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL,"
        "  admin_username TEXT,"
        "  reason TEXT NOT NULL,"
        "  violations TEXT,"
        "  created_at TEXT DEFAULT (datetime('now')))")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_signals ("       # WebRTC signaling mailbox (short-lived)
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL,"
        "  from_channel TEXT NOT NULL,"
        "  kind TEXT NOT NULL,"
        "  payload TEXT,"
        "  created_at TEXT DEFAULT (datetime('now')))")
    db.execute("CREATE INDEX IF NOT EXISTS idx_conv_signals ON conv_signals(session_id, id)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_invites ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  code TEXT UNIQUE NOT NULL,"                 # internal reference only (retired as a user-facing code)
        "  creator_speaker_id TEXT NOT NULL,"
        "  invitee_speaker_id TEXT,"
        "  status TEXT NOT NULL DEFAULT 'pending',"   # pending | accepted | declined | cancelled | expired
        "  responded_at TEXT,"
        "  topic_id INTEGER,"
        "  created_at TEXT DEFAULT (datetime('now')),"
        "  expires_at TEXT,"
        "  used_session_id INTEGER,"
        "  kind TEXT DEFAULT 'invite')")             # 'invite' (7-day, legacy) | 'call' (rings ~90 s)
    _ci_cols = {r["name"] for r in db.execute("PRAGMA table_info(conv_invites)").fetchall()}
    if _ci_cols and "kind" not in _ci_cols:
        db.execute("ALTER TABLE conv_invites ADD COLUMN kind TEXT DEFAULT 'invite'")
    # Pay bookkeeping (mirrors clips.*_rate_at_approval / compensated / editor_paid). Rates are
    # frozen at admin approval; paid flags flip when admin settles (same buttons as read speech).
    _cs_cols = {r["name"] for r in db.execute("PRAGMA table_info(conv_sessions)").fetchall()}
    for col, typ in [("rate_a_at_approval", "INTEGER"), ("rate_b_at_approval", "INTEGER"),
                     ("editor_rate_at_approval", "INTEGER"), ("paid_a", "INTEGER DEFAULT 0"),
                     ("paid_b", "INTEGER DEFAULT 0"), ("editor_paid", "INTEGER DEFAULT 0")]:
        if _cs_cols and col not in _cs_cols:
            db.execute(f"ALTER TABLE conv_sessions ADD COLUMN {col} {typ}")
    # Which topics each editor's circle may use (editor_user_id 0 = the admin's own circle,
    # i.e. speakers not assigned to any editor). A circle with no rows falls back to all
    # active topics, so nothing dead-ends before admin has assigned batches.
    db.execute("CREATE TABLE IF NOT EXISTS conv_topic_groups ("
               "  topic_id INTEGER NOT NULL,"
               "  editor_user_id INTEGER NOT NULL DEFAULT 0,"
               "  assigned_at TEXT DEFAULT (datetime('now')),"
               "  PRIMARY KEY(topic_id, editor_user_id))")
    if _ci_cols and "topic_id" not in _ci_cols:
        db.execute("ALTER TABLE conv_invites ADD COLUMN topic_id INTEGER")
    if _ci_cols and "invitee_speaker_id" not in _ci_cols:
        db.execute("ALTER TABLE conv_invites ADD COLUMN invitee_speaker_id TEXT")
        db.execute("ALTER TABLE conv_invites ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'")
        db.execute("ALTER TABLE conv_invites ADD COLUMN responded_at TEXT")
        # legacy code-based invites: consumed ones -> accepted, open ones -> expired
        db.execute("UPDATE conv_invites SET status=CASE WHEN used_session_id IS NOT NULL THEN 'accepted' ELSE 'expired' END")
    _ct_cols = {r["name"] for r in db.execute("PRAGMA table_info(conv_tracks)").fetchall()}
    if _ct_cols and "upload_id" not in _ct_cols:
        db.execute("ALTER TABLE conv_tracks ADD COLUMN upload_id TEXT")
    if _ct_cols and "edit_status" not in _ct_cols:
        db.execute("ALTER TABLE conv_tracks ADD COLUMN edit_status TEXT DEFAULT 'pending'")
        db.execute("ALTER TABLE conv_tracks ADD COLUMN edit_done_at TEXT")
    if _ct_cols and "clock_skew_ms" not in _ct_cols:
        db.execute("ALTER TABLE conv_tracks ADD COLUMN clock_skew_ms INTEGER")
    if _ct_cols and "gain_db" not in _ct_cols:
        db.execute("ALTER TABLE conv_tracks ADD COLUMN gain_db REAL")
    _cs_cols = {r["name"] for r in db.execute("PRAGMA table_info(conv_sessions)").fetchall()}
    if _cs_cols and "draft_error" not in _cs_cols:
        db.execute("ALTER TABLE conv_sessions ADD COLUMN draft_error TEXT")
    if _cs_cols and "leak_corr" not in _cs_cols:
        db.execute("ALTER TABLE conv_sessions ADD COLUMN leak_corr REAL")
        db.execute("ALTER TABLE conv_sessions ADD COLUMN abort_reason TEXT")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_utterances ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL,"
        "  channel TEXT NOT NULL,"
        "  speaker_id TEXT NOT NULL,"
        "  seq INTEGER NOT NULL,"
        "  start_sec REAL NOT NULL,"
        "  end_sec REAL NOT NULL,"
        "  draft_text TEXT,"
        "  text TEXT,"
        "  source TEXT NOT NULL DEFAULT 'draft',"      # draft | edited
        "  pii INTEGER NOT NULL DEFAULT 0,"
        "  unclear INTEGER NOT NULL DEFAULT 0,"
        "  nonspeech_only INTEGER NOT NULL DEFAULT 0,"
        "  bad_audio INTEGER NOT NULL DEFAULT 0,"
        "  accepted INTEGER NOT NULL DEFAULT 0,"
        "  edited_by TEXT,"
        "  edited_at TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_conv_utt_session ON conv_utterances(session_id, channel, seq)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS conv_filler_tokens ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  token TEXT UNIQUE NOT NULL,"
        "  active INTEGER NOT NULL DEFAULT 1)")
    # Seed the admin-editable generation prompt and DRAFT filler list once.
    if not db.execute("SELECT 1 FROM conv_topic_settings WHERE id=1").fetchone():
        db.execute("INSERT INTO conv_topic_settings (id, system_prompt, updated_at) "
                   "VALUES (1, ?, datetime('now'))", (CONV_TOPIC_GEN_DEFAULT_PROMPT,))
    if not db.execute("SELECT 1 FROM conv_filler_tokens LIMIT 1").fetchone():
        for tok in CONV_FILLER_SEED_DRAFT:
            db.execute("INSERT OR IGNORE INTO conv_filler_tokens (token) VALUES (?)", (tok,))

    db.execute(
        "CREATE TABLE IF NOT EXISTS name_corrections ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  speaker_id TEXT NOT NULL,"
        "  old_name TEXT,"
        "  new_name TEXT NOT NULL,"
        "  corrected_by TEXT,"
        "  reason TEXT,"
        "  corrected_at TEXT DEFAULT (datetime('now')))")

    # Migration for DBs created before result_json existed (used by the ghost scan job
    # to persist its findings so the page reads results instead of re-scanning live).
    _bj_cols = {r["name"] for r in db.execute("PRAGMA table_info(bulk_jobs)").fetchall()}
    if "result_json" not in _bj_cols:
        db.execute("ALTER TABLE bulk_jobs ADD COLUMN result_json TEXT")
    # Threads don't survive a restart/deploy: any job still 'running' at startup is dead.
    # Its per-batch commits are durable, so simply re-running the job resumes the rest.
    db.execute("UPDATE bulk_jobs SET status='failed', "
               "  error='Interrupted by a server restart/deploy. Already-committed batches are saved — run bulk approve again to finish the rest.', "
               "  finished_at=datetime('now') WHERE status='running'")

    # One-time backfill: populate text_normalized for any existing prompts
    # that don't have it yet. Cheap to run — only touches rows where the column is NULL.
    null_rows = db.execute("SELECT id, text_mn FROM prompts WHERE text_normalized IS NULL").fetchall()
    if null_rows:
        for row in null_rows:
            normalized = normalize_prompt_text(row["text_mn"] or "")
            db.execute("UPDATE prompts SET text_normalized=? WHERE id=?", (normalized, row["id"]))
        print(f"[INIT] Backfilled text_normalized for {len(null_rows)} existing prompts")

    # Stage 4 - one-time backfill: lock the current rate of existing approved clips that
    # have a NULL frozen rate. This protects past clips' compensation from future rate changes.
    # We lock to the CONTRIBUTOR's current hourly_rate (which is 20000 by migration default,
    # so legacy clips lock at 20000/hr exactly as today).
    db.execute(
        "UPDATE clips SET contributor_rate_at_approval = ("
        "  SELECT u.hourly_rate FROM users u WHERE u.speaker_id = clips.speaker_id) "
        "WHERE status='approved' AND contributor_rate_at_approval IS NULL"
    )
    # Editors did not exist before Stage 1, so editor_rate_at_approval stays NULL on legacy clips.
    # That's correct — legacy approved clips have no editor and no editor payout.

    # One-time backfill: translate any Mongolian region/education/dialect values
    # currently in DB to their English equivalents. Runs every startup but is idempotent
    # — only updates rows where the current value matches a known Mongolian label.
    for mn, en in _REGION_MN_TO_EN.items():
        db.execute("UPDATE profiles SET region=? WHERE region=?", (en, mn))
    for mn, en in _EDUCATION_MN_TO_EN.items():
        db.execute("UPDATE profiles SET education_level=? WHERE education_level=?", (en, mn))
    for mn, en in _DIALECT_MN_TO_EN.items():
        db.execute("UPDATE profiles SET dialect=? WHERE dialect=?", (en, mn))

    # Backfill recording_device for existing contributors:
    #   SPK01 = USB microphone (Elgato Wave:3 — the project's founding contributor)
    #   Everyone else defaults to "Mobile phone".
    # Admin can adjust per-contributor via the profile editor.
    db.execute(
        "UPDATE profiles SET recording_device='USB microphone' "
        "WHERE speaker_id='MNKH_SPK01' AND (recording_device IS NULL OR recording_device='')"
    )
    db.execute(
        "UPDATE profiles SET recording_device='Mobile phone' "
        "WHERE (recording_device IS NULL OR recording_device='') "
        "  AND speaker_id LIKE 'MNKH_SPK%' AND speaker_id != 'MNKH_SPK01'"
    )

    cur = db.execute("SELECT id FROM users WHERE role='admin' LIMIT 1")
    if not cur.fetchone():
        db.execute("INSERT INTO users (username,password,speaker_id,role) VALUES (?,?,?,?)",
            ("admin", generate_password_hash("admin123"), "ADMIN", "admin"))
    db.commit(); db.close()

def calc_comp(secs, rate=None):
    """Compute compensation: ceil seconds to whole minutes, then ceil the money.
    If rate is None, uses the legacy default (RATE_PER_HOUR = 20000)."""
    if not secs or secs <= 0: return 0
    if rate is None: rate = RATE_PER_HOUR
    return math.ceil(math.ceil(secs/60) * rate / 60)

def _payroll_pending(db):
    """Timestamp (UTC) of the most recent payroll download NOT yet followed by an
    "All Paid" settle, else None. While set, any new admin approval adds pay that is
    NOT in the downloaded payroll — the approval UI warns so the recorded settle can
    never silently exceed the bank transfer (the drift found in the Aug-26 incident)."""
    r = db.execute("SELECT applied_at FROM _one_time_flags WHERE key='payroll_pending'").fetchone()
    return r["applied_at"] if r else None

def apply_editor_penalty(amount, pct):
    """Reduce an editor payout by the active penalty percentage (admin-set, cleared on
    settle). Integer MNT, rounded down — the reduction can never overshoot. pct outside
    0..100 is clamped; 0/None means no penalty."""
    try:
        p = int(pct or 0)
    except (TypeError, ValueError):
        p = 0
    p = max(0, min(100, p))
    if not amount or p == 0:
        return amount or 0
    return (amount * (100 - p)) // 100

def calc_comp_grouped(clips, rate_field, fallback_rate):
    """Sum compensation across clips that may have different frozen rates.
    Groups clips by rate, then applies calc_comp ONCE per rate-group.
    This matches the legacy behavior (one ceil per batch) instead of
    inflating totals by ceiling each clip individually.

    `clips` — iterable of dict-like rows with at least 'duration_seconds' and rate_field.
    `rate_field` — name of the column holding each clip's frozen rate (may be None).
    `fallback_rate` — rate to use when frozen rate is NULL.
    """
    by_rate = {}
    for c in clips:
        rate = c[rate_field] if c[rate_field] is not None else fallback_rate
        by_rate[rate] = by_rate.get(rate, 0) + (c["duration_seconds"] or 0)
    return sum(calc_comp(secs, rate) for rate, secs in by_rate.items())

def to_pdf_bytes(file_storage):
    """Convert an uploaded file to PDF bytes.
    PDF passes through unchanged. Images (JPG, PNG, HEIC, WEBP, etc.) are
    converted to a single-page PDF. Returns bytes, or raises ValueError.

    Max accepted size: 15 MB. Output PDFs are usually much smaller.
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("No file uploaded.")
    raw = file_storage.read()
    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("File is too large. Maximum size is 15 MB.")
    if len(raw) < 200:
        raise ValueError("File is too small or empty.")
    # PDF magic header is %PDF
    if raw[:4] == b"%PDF":
        return raw
    # Otherwise treat as image
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(raw))
        # Some formats (HEIC) need pillow-heif; if Pillow can't open, error out
        # Convert to RGB so PDF saving works (PDF doesn't support RGBA/P modes well)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="PDF", resolution=150.0)
        out.seek(0)
        return out.read()
    except Exception as e:
        raise ValueError(f"Could not read image file: {e}")

# ══ Consent agreement v2.0 (Илтгэгчийн зөвшөөрөл болон эрх шилжүүлэх гэрээ) ═════
# The Mongolian list below is the GOVERNING certified text (E.3) and the ONLY text
# that is hashed. It must match the certified file exactly after canonical
# serialization — NEVER edit it here; suspected typos are flagged to the admin.
# The English list is the certified convenience translation: display-only, not hashed.
CURRENT_CONSENT_VERSION = "2.0"

CONSENT_V2_TITLE_MN = "ИЛТГЭГЧИЙН ЗӨВШӨӨРӨЛ БОЛОН ЭРХ ШИЛЖҮҮЛЭХ ГЭРЭЭ"
CONSENT_V2_TITLE_EN = "SPEAKER CONSENT AND RIGHTS GRANT AGREEMENT"

# Certified v2.0 text — embedded verbatim from the reissued certified files
# (Монгол_хувилбар.docx / Англи_хувилбар.docx). Never edit here.
CONSENT_AGREEMENT_V2_MN = [
    "(Энэхүү гэрээний англи хэлээрх хувилбар нь зөвхөн талуудын ойлголцлыг баталгаажуулах, ойлгоход хялбар болгох зорилготой орчуулга болно. Талуудын хооронд үүсэх эрх зүйн харилцаа, тэдгээрийн эрх, үүргийг зохицуулахад монгол хэлээрх хувилбар хүчинтэй үйлчилнэ — E.3 хэсгийг үзнэ үү.)",
    "A ХЭСЭГ — ТАЛУУД БА ОРОЛЦОО",
    "A.1. Гэрээний талууд. Энэхүү гэрээг нэг талаас Монгол Улсын иргэн […] овогтой […]. (Төрсөн он сар өдөр: […]) (цаашид “Илтгэгч” гэх) болон Монгол Улсад бүртгэлтэй хуулийн этгээд болох ГУРВАНБИЛЭГ ХХК (регистрийн дугаар: 2043149; албан ёсны хаяг: Монгол Улс, Улаанбаатар хот, Баянгол дүүрэг; 18-р хороо, 4-р хороолол, өөрийн байр 10; холбоо барих: hi@gurvanbileg.com / +976-72014897; хувийн мэдээллийн нууцлал, хамгаалалтай асуудлаар холбоо барих: privacy@gurvanbileg.com) (цаашид “Компани” гэх) нарын хооронд байгуулав. Компани нь энэхүү гэрээний дагуу Илтгэгчтэй цорын ганц гэрээлэгч тал бөгөөд Илтгэгчийн хувийн мэдээллийг боловсруулж, хадгалагч байхын дээр энэхүү гэрээгээр Илтгэгчээс шилжүүлж буй эрхийн цор ганц хүлээн авагч мөн.",
    "A.2. Сайн дурын оролцоо; миний өөрийн дуу хоолой. Компанийн платформоор дамжуулан миний илгээх бүх дуу хоолойн бичлэг (цаашид хамтад нь “Бичлэгүүд” гэх) нь Илтгэгч миний өөрийн дуу хоолой бөгөөд миний бүрэн зөвшөөрөлтэйгөөр, сайн дурын үндсэн дээр хийгдсэн гэдгийг би баталж байна. Илтгэгч миний бие энэхүү дуу хоолойн бичлэгийг боловсруулах, хадгалах зөвшөөрлийг аливаа дарамт шахалтгүйгээр, өөрийн хүсэл зоригийн үндсэн дээр өгч байгааг баталж байна.",
    "A.3. Бие даасан оролцоо ба хөдөлмөрийн харилцаа үүсэхгүй байх тухай. Энэхүү гэрээний үндсэн дээр болон түүнийг хэрэгжүүлэхтэй холбоотойгоор талуудын хооронд хөдөлмөр эрхлэлтийн харилцаа үүсэхгүй буюу Компани нь Илтгэгчийн ажил олгогч болохгүйг Илтгэгч бүрэн хүлээн зөвшөөрч байна. Энэхүү гэрээний дагуу Илтгэгч миний бие Компанид дуу хоолойн бичлэгээ өгөх эсэх, тийм бол дуу хоолойн бичлэгээ хэзээ өгөх тухай өөрөө бие даан шийдвэрлэнэ. Мөн энэхүү гэрээний дагуу Илтгэгч миний бие тогтсон цагийн хуваариар ажиллахгүйн дээр ажиллах цагийн доод хэмжээг мөрдөхгүй. Түүнчлэн Компаниас Илтгэгчид байнга ажил олгогдох баталгаа байхгүйн дээр Илтгэгчийн зүгээс Компанид байнга дуу хоолойн бичлэг илгээх баталгаа байхгүйг талууд хүлээн зөвшөөрч байна. Илтгэгч нь Компанид дуу хоолойн бичлэг илгээхэд шаардлагатай тоног төхөөрөмж, техник хэрэгслийг өөрөө хариуцах буюу зөвхөн өөрийн тоног төхөөрөмжийг ашиглана. Илтгэгч нь Компанид өөрийн дуу хоолойн бичлэгийг илгээх харилцааг хүссэн үедээ зогсоож болно. Энэхүү гэрээний дагуу Илтгэгчид цалин, хөдөлмөрийн хөлс олгохгүй бөгөөд талуудын харилцан тохирсны дагуу нэгж дуу хоолойн бичлэг тус бүрд тохирсон төлбөрийг төлнө. Илтгэгч миний бие энэхүү гэрээг байгуулах, түүнээс үүдэн гарах эрх зүйн харилцаанд орох бүрэн чадамжтай болохоо баталж байна.",
    "A.4. Нас болон иргэний хувийн мэдээлэл баталгаажуулалт. Илтгэгч миний бие 18 насанд хүрсэн бөгөөд энэ нь төрийн байгууллагаас олгосон албан ёсны, хүчин төгөлдөр иргэний баримт бичгээр нотлогдохыг баталж байна. Миний төрөөс олгосон баримт бичигт байгаа овог нэр, төрсөн огноо, хүйсийн мэдээллийг албан ёсны, үнэн зөв бүртгэл гэж үзэж, миний гараар оруулсан зөрүүтэй мэдээллийг түүгээр орлуулах бөгөөд төлбөрийг зөвхөн баталгаажсан иргэний баримт бичигт нэр нь бичигдсэн хүнд хийх боломжтой гэдгийг би ойлгож байна. Түүнчлэн Илтгэгчийн танин баталгаажуулах баримт бичгийг шалгаж, баталгаажуулахад автоматжуулсан технологи, хэрэгсэл ашиглахыг миний бие бүрэн ойлгож байна.",
    "A.5. Гүйцэтгэлд тавих чанарын шаардлага ба гэрээний төлбөр. Би зөвхөн төслийн стандартын шаардлагыг хангасан дуу хоолойн Бичлэгүүдийг илгээх бөгөөд дуу хоолойн бичлэг үүсгэх, түүнийг Компанид илгээхдээ үнэнч, шударгаар хандаж, шалгагдаж зөвшөөрөгдсөн Бичлэгүүдийн төлбөрийг энэхүү төслийн төлбөрийн нөхцөлийн дагуу авахыг бүрэн хүлээн зөвшөөрч байна. Илтгэгчийн илгээсэн дуу хоолойн Бичлэгийг төслийн стандартын шаардлагыг хангасан гэж үзэж, төлбөр төлөх эсэх нь энэхүү гэрээний B хэсэгт заасан Бичлэгт хамаарах эрх шилжих харилцаанаас тусдаа зохицуулагдах бөгөөд B хэсэгт заасан Бичлэгт хамаарах эрх шилжих зохицуулалт нь Илтгэгчийн илгээсэн бүх Бичлэгт хамаарна.",
    "B ХЭСЭГ — БИЧЛЭГТ ХАМААРАХ ЭРХ ШИЛЖИХ",
    "B.1. Эрх шилжилт. Илтгэгч миний бие Компанид илгээх бүхий л дуу хоолойн Бичлэгт хамааран дагалдах нэр болон зурагтай дүйцэх материал; зохиогчийн эрх; гүйцэтгэгчийн (хамаарах) эрх; оюуны өмчийн эрх, оюуны өмчийн эд хөрөнгийн (өмчлөх болон захиран зарцуулах) эрх, ашиг сонирхол болон Бичлэгийг түгээх, нийтлэх амины эрхийг багтаасан бүрэн эрхийг Компанид олгож буйгаа үүгээр хүлээн зөвшөөрч байна. Дээрх эрх олголтыг хязгаарлахгүйгээр Компани нь Бичлэгүүд болон холбогдох төслийн аливаа материалыг хадгалах, боловсруулах, засварлах, өөрчлөх, ашиглах, хуулбарлан олшруулах, түгээх, үүсмэл бүтээл бэлтгэх, нийтэд толилуулах болон үзүүлэх, лиценз олгох, олон шатлалаар дэд лицензүүд олгох, худалдах, мөн одоо мэдэгдэж байгаа эсхүл цаашид бий болох аливаа технологи, формат, платформд оруулах зэрэг үйлдэлд хамаарах цуцлах боломжгүй, дэлхий даяар хүчинтэй, хугацаагүй, роялтигүй эрхтэй бөгөөд эдгээр үйлдлийг бусдад хийхийг зөвшөөрөх эрхтэй болохыг би үл маргах журмаар хүлээн зөвшөөрч байна. Эдгээр эрхийг дуу ярианы өгөгдлийн сан, хиймэл оюуны сургалтын өгөгдлийн сан, судалгаа, бүтээгдэхүүн хөгжүүлэлт, чанарын хяналт, арилжааны өгөгдлийн бүтээгдэхүүн зэрэг аливаа зорилгоор, нэмэлт төлбөргүй, хугацааны болон газар зүйн хязгаарлалтгүйгээр хэрэгжүүлж болно.",
    "B.2. Шилжүүлэх эрхэд хамаарах эрхийн ангилал. Энэхүү гэрээний B.1-д заасны дагуу Илтгэгчээс Компанид илгээх бүхий л Бичлэгт дагалдан Компанид шилжих эрхэд Бичлэгийг биет хэлбэрт буулгах, дамжуулах, хуулбарлан олшруулах, нийтэд түгээх, нийтлэх, түрээслүүлэх, олон нийтийн сүлжээгээр дамжуулан нийтэд хүртээмжтэй байршуулахтай холбоотой төслийн гүйцэтгэгч миний онцгой эрхүүд болон миний эзэмшиж болох фонограммтай холбоотой аливаа эрх холбогдох хууль тогтоомжийн дагуу боломжит дээд хэмжээгээр хамаарахыг Илтгэгч миний бие хүлээн зөвшөөрч байна. Хэрэв холбогдох хууль тогтоомжийн дагуу эдгээр эрхийн аль нэгийг Компанид шилжүүлэх боломжгүй бол Илтгэгч миний бие тухайн эрхийг ашиглах лицензийг Компанид олгохоо үүгээр баталж байгаа бөгөөд ийнхүү олгох лизенц нь онцгой, цуцлах боломжгүй, хугацаагүй, улс хоорондын хилийн хязгааргүй (дэлхий даяар хүчин төгөлдөр), гуравдагч этгээдэд дэд лиценз олгох байдлаар шилжүүлэх боломжтой байна.",
    "B.3. Бичлэгүүдэд хамаарах зохиогчийн амины (эдийн бус) эрх. Холбогдох хууль тогтоомжоор зөвшөөрөгдөх дээд хэмжээнд, Бичлэгүүд болон Бичлэгүүдээс үүссэн материалтай холбоотой миний эзэмшиж болох аливаа амины (эдийн бус) эрхийг Компани, түүний хэрэглэгчид, лиценз эзэмшигчид, дэд лиценз эзэмшигчид болон эрх залгамжлагчдын эсрэг хэрэгжүүлэхгүй, тэдгээрийн эсрэг аливаа шаардлага гаргахгүй гэдгээ би зөвшөөрч байна.",
    "B.4. Хиймэл оюуны зөвшөөрөгдсөн хэрэглээ. Бичлэгүүдийг хиймэл оюун ухаан болон машин сургалтын системийг хөгжүүлэх, сургах, турших, үнэлэх (үүнд яриа таних, бичвэрийг ярианд хөрвүүлэх, яриа боловсруулах технологид ашиглаж болох бөгөөд Компанийн хэрэглэгчдэд лицензээр олгох өгөгдлийн санд оруулж болно)-д ашиглаж болохыг Илтгэгч миний бие бүрэн ойлгож, зөвшөөрч байна. Энд заасан хиймэл оюуны зөвшөөрөгдсөн хэрэглээнд миний дуу хоолойтой төстэй байж болох дуу хоолой зэрэг нийлэг яриа үүсгэх ажиллагаа мөн адил багтана.",
    "B.5. Хориглосон хэрэглээ. Энэхүү гэрээний дагуу Компани нь өөрийн хэрэглэгчид болон лиценз эзэмшигчдээс Бичлэгүүдийг дараах зорилгоор ашиглахгүй байхыг шаардах үүрэг хүлээнэ:",
    "- залилан мэхлэх;",
    "- Илтгэгчийн нэрийг ашиглаж дүр эсгэх;",
    "- Илтгэгчид хамаатуулсан төөрөгдүүлсэн сурталчилгаа эсхүл дэмжлэгийн мэдэгдэл хийх;",
    "- хууль бус хяналт, тандалт хийх;",
    "- Илтгэгчийг гүтгэх, гутаан доромжлох, худал мэдээлэл тараах;",
    "- Илтгэгчийн нэр хүндэд хохирол учруулах байдлаар ашиглах;",
    "- хууль бус ялгаварлан гадуурхалтад ашиглахгүй байх.",
    "B.6. Унших текст. Илтгэгчийн дуу хоолойн бичлэг хийхдээ унших текст, бичмэл материалыг Компаниас хангах бөгөөд эдгээр текст, бичвэрт холбогдох аливаа эрхийн асуудлыг Компани хариуцахын дээр энэхүү гэрээгээр зохицуулах харилцаа нь дуу хоолойн бичлэг хийхэд унших текстийн агуулга болон түүнд холбогдох оюуны өмчийн болон зохиогчийн эрхтэй холбоотой харилцаанд хамаарахгүй, гагцхүү Илтгэгчийн текст уншиж, дуу хоолойн бичлэг хийх гүйцэтгэл болон түүний дуу хоолойг хамарч байгааг Илтгэгч миний бие бүрэн ойлгож байна.",
    "B.7. Үл маргах (нэхэмжлэл, шаардлага гаргахгүй байх) зарчим. Холбогдох хууль тогтоомжид харшилсан, эсхүл энэхүү гэрээний Б.5-д хориглосон хэрэглээнд хамаарахаас бусад тохиолдолд Бичлэгийг, эсхүл Бичлэгээс үүссэн буюу түүнтэй өөр байдлаар холбоотой материалыг ашигласантай холбогдуулан Компани, түүний лиценз эзэмшигч, дамжуулан лиценз эзэмшигч болон эрх шилжүүлэн авсан этгээдийн эсрэг аливаа улс, нутаг дэвсгэрийн эрх бүхий шүүх, байгууллагад ямар нэгэн нэхэмжлэл, шаардлага, хэрэг, маргаан үүсгэх, гаргах, дэмжих, эсхүл бусад этгээдээр ийм нэхэмжлэл, шаардлага гаргуулахыг зөвшөөрөхгүй. Үүнд хувийн нууцад халдсан, сурталчилгааны болон олон нийтэд танигдах эрх, бусад иргэний эрх зөрчигдсөн гэх үндэслэлээр гаргах нэхэмжлэл, мөн миний дуу хоолой эсхүл түүнтэй адилтгах дуу хоолойг ашигласантай холбоотой нэхэмжлэл нэгэн адил хамаарна.",
    "B.8. Бичлэг болон түүнд хамаарах агуулгыг шилжүүлэх. Компани нь Бичлэгүүд болон тэдгээрийг агуулсан өгөгдлийн санг өөрийн хамаарал бүхий этгээдүүд болон хэрэглэгчдэд шилжүүлэх, тэдгээрийг ашиглах эрх бүхий лиценз болон дэд лиценз олгож болох бөгөөд ийнхүү Бичлэг болон түүнд хамаарах агуулгыг шилжүүлэн авсан хэрэглэгч, лиценз эзэмшигчид нь хүлээн авсан өгөгдлийн сан болон бүтээгдэхүүний нэг хэсэг болгон Бичлэгүүдийг цааш ашиглах, дэд лиценз олгох болохыг Илтгэгч миний бие бүрэн ойлгож, зөвшөөрч байна.",
    "B.9. Шилжүүлсэн эрхийн хүчин төгөлдөр байдал. Энэхүү гэрээний дагуу Илтгэгч миний дуу хоолойн бичлэг болон түүнтэй хамт Компанид шилжүүлсэн оюуны өмчийн, зохиогчийн, эд хөрөнгийн эрхүүд (гэрээний B.1, B.2-т дэлгэрэнгүй дурдсан) нь байнгын буюу хугацаагүй хүчин төгөлдөр байх бөгөөд дараах тохиолдолд мөн адил хүчин төгөлдөр хэвээр үлдэнэ:",
    "\t(a) Илтгэгч миний бие төсөлд оролцохоо зогсоох буюу цаашид Компанид дуу хоолойн бичлэг илгээхгүй байхаар шийдвэрлэсэн;",
    "\t(b) энэхүү гэрээ аливаа шалтгаанаар дуусгавар болсон буюу хүчин төгөлдөр бус болсон, талууд гэрээнээс татгалзсан, эсхүл цуцалсан;",
    "\t(c) талууд гэрээг шинээр байгуулсан, талуудын хоорондын эрх зүйн харилцаанд өөрчлөлт орсон, эсхүл гэрээнд нэмэлт, өөрчлөлт оруулсан.",
    "Мөн илтгэгчээс өмнө нь хийж дуусгаж, Компанид илгээсэн бүхий л дуу хоолойн Бичлэгүүд болон түүнд хамаарах аливаа эрхүүд байнгын, хугацаагүй хүчин төгөлдөр хэвээр байх бөгөөд энэ заалтад дурдсан хүчин төгөлдөр хэвээр байх эрхэд нэгэн адил хамаарна.",
    "C ХЭСЭГ — ХУВИЙН МЭДЭЭЛЭЛ БА НУУЦЛАЛЫН БАТАЛГАА",
    "C.1. Мэдээлэл хариуцагч. Энэхүү гэрээний дагуу Илтгэгчээс Компанид илгээх Бичлэгүүд болон түүнд хамаарах мэдээлэл, Илтгэгчийн хувийн мэдээллийг хариуцагч нь ГУРВАНБИЛЭГ ХХК байна (дэлгэрэнгүйг A.1 хэсгээс үзнэ үү).",
    "C.2. Компанийн цуглуулах мэдээлэл. Компани дараах мэдээллийг цуглуулж, боловсруулна: ",
    "- Төрийн эрх бүхий байгууллагаас олгож, баталгаажсан иргэний баримт бичиг дэх Илтгэгч миний овог, нэр, төрсөн огноо, хүйс; ",
    "- Төрийн эрх бүхий байгууллагаас олгож, баталгаажсан иргэний баримт бичиг (регистрийн дугаар, иргэний бүртгэлийн дугаарыг дарсан/нуусан байх); ",
    "- миний дуу хоолойн бичлэгүүд (биометрик мэдээлэл); ",
    "- бичлэгийн мета өгөгдөл; ",
    "- Компанийн платформ дахь миний бүртгэл, сешн, үйлдлийн лог, IP хаяг болон төхөөрөмжийн мэдээлэл; ",
    "- миний ярианы хэв шинжийг илэрхийлэх мэдээлэл (хурд, хэмнэл, зөвшөөрснийг илэрхийлэх хэллэг зэрэг); ",
    "- миний банкны эсхүл төлбөр авсан дансны мэдээлэл.",
    "C.3. Мэдээлэл цуглуулах зорилго ба эрх зүйн үндэслэл. Энэхүү мэдээллийг энэ гэрээний дагуу, өөрийн хүсэл зоригийн үндсэн дээр өгсөн миний зөвшөөрөлд үндэслэн дараах зорилгоор боловсруулна: ",
    "- иргэний таних мэдээлэл болон насыг баталгаажуулах;",
    "- зөвшөөрлийг баримтжуулах;",
    "- дуу хоолой, ярианы өгөгдлийн сан үүсгэх;",
    "- ажлын гүйцэтгэлийн чанарыг хянаж, үнэлэх;",
    "- төлбөр боловсруулах;",
    "- хууль тогтоомжоор тавигдах холбогдох шаардлагыг хангуулах.",
    "C.4. Мэдээлэл хадгалах хугацаа. Компани дараах мэдээллийг хадгална: (a) миний иргэнийг таних баримт бичгийн зургийг баталгаажуулснаас хойш 5 жилийн турш хадгалж, уг хугацаа дуусмагц устгана (зургийг устгасны дараа ч баталгаажуулалтын үр дүнг хадгална); (b) миний энэхүү гэрээгээр олгосон аливаа зөвшөөрөлтэй холбоотой бүртгэлийг миний Бичлэгүүдийг агуулсан өгөгдлийн санг арилжааны зорилгоор хадгалж, ашиглаж байгаа бүх хугацаанд, үүн дээр нэмээд 10 жилийн турш хадгална; (c) миний Бичлэгүүдийг энэхүү гэрээнд заасан хугацааны турш хадгална.",
    "C.5. Худалдан авагчид шилжүүлэх мэдээлэл. Миний Бичлэгүүдийг агуулсан өгөгдлийн санг лицензээр дамжуулан хүлээн авсан хэрэглэгчид дараах мэдээллийг хүлээн авч болохыг Илтгэгч миний бие үүгээр хүлээн зөвшөөрч байна:",
    "- миний Бичлэгүүд;",
    "- Бичлэгийн бичвэрт хөрвүүлсэн байдал болон түүний орчуулга;",
    "- нууц нэршил бүхий Илтгэгчийг таних тусгай дугаар, код;",
    "- миний хүйс, насны бүлэг, иргэний харьяалал;",
    "- зөвшөөрөл авсныг нотлох гэрчилгээ (насанд хүрсэн нь баталгаажсан төлөв, баталгаажуулсан огноо болон арга, гэрээний хувилбар, баталгаажуулалтын огноо-цаг болон арга, эрх олголтын хураангуй). ",
    "Илтгэгчийн иргэнийг таних баримт бичгийн зургийг хэрэглэгчдэд ердийн нөхцөлд өгөхгүй. Компани нь Илтгэгчийн иргэнийг таних баримт бичгийг нотлох үндсэн баримтыг аюулгүй хадгалж, зөвхөн үндэслэл бүхий аудит, хяналт шалгалт, маргаан эсхүл хууль зүйн бусад шаардлагатай үед, Илтгэгчийн нууцлалын аюулгүй байдлыг бүрэн хангаж, зөвхөн зайлшгүй шаардлагатай хэмжээнд мэдээллийг хязгаарлан задруулж болно.",
    "C.6. Мэдээлэлтэй холбоотой Илтгэгчийн эрх. Илтгэгч Компанид хадгалагдаж буй өөрийн хувийн мэдээлэлтэй танилцах, түүнд хандах, мэдээллээ залруулах хүсэлт гаргах эрхтэй бөгөөд цаашид мэдээлэл цуглуулахтай холбоотой зөвшөөрлөө Компанийн платформ дахь өөрийн бүртгэлээр дамжуулан хүссэн үедээ буцааж болно. Зөвшөөрлөө буцаах нь өмнө нь илгээсэн Бичлэгүүдэд олгосон эрхийн хүчин төгөлдөр байдалд нөлөөлөхгүй бөгөөд зөвшөөрөл буцаахаас өмнө хийгдсэн мэдээлэл боловсруулалт хууль ёсны хүчин төгөлдөр хэвээр байх бөгөөд Компанид хадгалагдана. Хэрэв би энэхүү гэрээний C.2-т заасан мэдээллийг өгөхөөс татгалзсан тохиолдолд Компанийн төсөлд оролцох боломжгүй болохыг ойлгож байна. Би гомдлоо Компанийн нууцлалын асуудлаар холбоо барих хаягт (A.1) эсхүл Монгол Улсын эрх бүхий байгууллагад гаргаж болно.",
    "C.7. Мэдээллийг нийтэд задруулах. Миний хувийн таних мэдээллийг нийтэд ил болгохгүй. Миний Бичлэгүүд болон энэхүү гэрээний C.5-д заасан нууц нэршил бүхий мэдээллийг хэрэглэгчдэд нийлүүлэх өгөгдлийн санд оруулж болохыг Илтгэгч миний бие үүгээр хүлээн зөвшөөрч байна.",
    "D ХЭСЭГ — БИОМЕТРИК МЭДЭЭЛЭЛ БА МЭДЭЭЛЛИЙГ ГАДААД УЛС ДАХЬ ХЭРЭГЛЭГЧИД ШИЛЖҮҮЛЭХ ЗӨВШӨӨРӨЛ",
    "D.1. Биометрик мэдээлэл. Миний дуу хоолойн бичлэгүүд нь намайг таних боломжтой биометрик мэдээлэл гэдгийг Илтгэгч миний бие ойлгож байгаа бөгөөд энэхүү гэрээний C.3 хэсэгт заасан зорилгоор Компани уг мэдээллийг цуглуулах, хадгалах, боловсруулахыг би бүрэн хүлээн зөвшөөрч байна.",
    "D.2. Мэдээллийг гадаад улс дахь харилцагчид шилжүүлэх. Энэхүү гэрээний С.5-д заасан мэдээлэл, тухайлбал миний дуу хоолойн Бичлэг болон түүнд хамаарах мэдээлэл, тэдгээрийг бичвэр болгон хөрвүүлсэн хэлбэр болон энэхүү гэрээнд заасан аливаа эрх, зөвшөөрлийг олгосныг нотлох мэдээлэл, Илтгэгчийг таних нууц нэршил, код бүхий мэдээлэл (төрийн эрх бүхий байгууллагаас олгосон иргэнийг таних баримт бичгийн зургийг оруулахгүйгээр) зэргийг Компанийн Монгол Улсын нутаг дэвсгэрээс гадна байрлах харилцагчдад (үүнд Америкийн Нэгдсэн Улс болон тухайн нөхцөлд хамаарах бусад улс/бүс нутагт байрлах, энэхүү гэрээний Б.4-т заасан зорилгоор ярианы өгөгдлийн санд хамаарах өгөгдлийн сан болон түүний мэдээллийг ашиглах эрх бүхий лиценз авсан компани, байгууллагууд хамаарна) дамжуулахыг бүрэн зөвшөөрч байна. Эдгээр харилцагч нь Компаниар дамжуулан хүлээн авсан мэдээллийн хувьд бие даасан мэдээлэл хариуцагч байдлаар ашиглаж, үйл ажиллагаа явуулж болох бөгөөд энэхүү гэрээний Б.8-д заасан нөхцөлийн дагуу тухайн өгөгдлийн санг цааш бусад гуравдагч этгээдэд дахин дамжуулж, лиценз олгож болохыг Илтгэгч миний бие бүрэн ойлгож байна. Компани болон түүнээс мэдээлэл хүлээн авсан харилцагч нь мэдээллийг бусад гуравдагч этгээдэд дамжуулахдаа мэдээллийн аюулгүй байдал болон ашиглалтад тавих хязгаарлалтыг тусгасан бичгээр байгуулсан гэрээ, хэлцлийн үндсэн дээр дамжуулах бөгөөд Илтгэгч миний бие мэдээлэл хүлээн авагчдын ангиллын талаар Компанийн хувийн мэдээлэл хамгаалалтын асуудал хариуцсан ажилтнаас мэдээлэл авах эрхтэй. Илтгэгч миний бие Компанид шилжүүлсэн ирээдүйд мэдээлэл хадгалах, ашиглах, дамжуулах зөвшөөрлийг буцаасан, цуцалсан нь ийнхүү зөвшөөрлийг буцааж, цуцлахаас өмнөх хугацаанд хууль ёсны дагуу хадгалж, боловсруулж, дамжуулсан мэдээллийн хууль ёсны бөгөөд хүчин төгөлдөр байдалд нөлөөлөхгүй болохыг үүгээр зөвшөөрч байна.",
    "E ХЭСЭГ — ЕРӨНХИЙ НӨХЦӨЛ",
    "E.1. Энэхүү гэрээ нь түүнийг байгуулахаас өмнөх хугацаанд Илтгэгчээс Компанид илгээсэн дуу хоолойн Бичлэгт нэгэн адил хамаарах бөгөөд талуудын хооронд байгуулсан өмнөх хэлэлцээр, тохиролцоог орлох буюу тэдгээрээс давуу хүчинтэй үйлчилнэ. Энэхүү гэрээ нь түүнийг байгуулахаас өмнө Илтгэгч болон Компанийн хооронд дуу хоолойн Бичлэг илгээх, түүнтэй холбоотой аливаа зөвшөөрлийг олгох хүрээнд байгуулсан бүхий л хэлэлцээр, тохиролцоог орлох буюу тэдгээрээс давуу хүчинтэй үйлчилнэ. Илтгэгч миний бие энэхүү гэрээг байгуулснаар түүнээс өмнөх хугацаанд талуудын хоорондын аливаа тохиролцоо, зөвшөөрлийн дагуу Компанид илгээсэн бүхий л дуу хоолойн Бичлэгтэй холбоотой энэхүү гэрээний B хэсэгт заасан эрхүүдийг дахин баталгаажуулж байгаа бөгөөд энэхүү гэрээг байгуулахаас өмнө илгээсэн дуу хоолойн Бичлэгтэй холбоотойгоор Компани энэхүү гэрээний B хэсэгт заасан бүхий л эрхийг эдлэх болохыг үүгээр баталж байна. Энэхүү гэрээг байгуулахаас өмнө миний зүгээс Компанид олгосон аливаа зөвшөөрлийн хувилбаруудыг Компани миний зөвшөөрлийг нотлох бүртгэлийн нэг хэсэг болгон хадгалж болно.",
    "E.2. Гэрээнд нэмэлт, өөрчлөлт оруулах болон талууд эрх зүйн харилцааг үргэлжлүүлэх үндэслэл. Компани энэхүү гэрээнд нэмэлт, өөрчлөлт оруулах, шинээр байгуулах санаачилгыг хэдийд ч гаргаж болно. Гэрээнд аливаа нэмэлт, өөрчлөлт, шинэчилсэн найруулга оруулах тохиолдолд Компани шинэчилсэн гэрээний бүрэн эхийг, түүнд орсон ач холбогдол бүхий өөрчлөлтийн хураангуйн хамт Илтгэгчид танилцуулах бөгөөд Илтгэгч цаашид Компанид дуу хоолойн Бичлэг илгээхийн өмнө талууд гэрээний нэмэлт, өөрчлөлт, шинэчилсэн хувилбарыг баталгаажуулж, Илтгэгчийн зүгээс Компанид дуу хоолойн Бичлэг болон түүнд хамаарах мэдээллийг хадгалах, боловсруулах, ашиглах, дамжуулах зөвшөөрлийг дахин олгосон байна. Гэрээнд орсон аливаа нэмэлт, өөрчлөлт, шинэчилсэн хувилбар нь түүнийг баталгаажуулахын өмнө Илтгэгчийн зүгээс Компанид илгээсэн дуу хоолойн Бичлэг болон түүнд хамаарах мэдээллийг хадгалах, боловсруулах, ашиглах, дамжуулах хүрээнд Компанид олгосон эрх, зөвшөөрлийг аливаа хэмжээгээр бууруулж, хязгаарлаж, хүчингүй болгохгүй болохыг талууд хүлээн зөвшөөрч байна.",
    "E.3. Гэрээ байгуулах хэл. Компани нь энэхүү гэрээний монгол болон англи хэлээрх хувилбарыг Илтгэгчид танилцуулна. Энэхүү гэрээний дагуу талуудын хооронд үүсэх эрх зүйн харилцааг зохицуулахад зөвхөн монгол хэлээрх хувилбарыг мөрдлөг болгоно. Энэхүү гэрээний англи хэлээрх хувилбар нь зөвхөн талуудын хоорондын харилцааг илүү ойлгомжтой, баталгаатай болгох зорилго бүхий орчуулга болно. Энэхүү гэрээний монгол болон англи хэлээрх хувилбарын хооронд агуулгын зөрүү, зөрчил гарсан тохиолдолд монгол хэлээрх хувилбарыг баримтална. Энэхүү гэрээг байгуулахын өмнө Компанийн зүгээс гэрээний монгол хэлээрх хувилбарыг бүрэн эхээр танилцуулж, талууд санал нэгдэн гэрээг баталгаажуулсан болохыг Илтгэгч миний бие үүгээр баталж байна.",
    "E.4. Гэрээнд хэрэглэх хууль ба маргаан шийдвэрлэх журам. Талуудын хооронд байгуулсан энэхүү гэрээ болон түүнийг хэрэгжүүлэхтэй холбоотой аливаа харилцаанд Монгол Улсын хууль тогтоомжийг мөрдлөг болгоно. Энэхүү гэрээ болон түүнийг хэрэгжүүлэхтэй холбоотойгоор талуудын хооронд үүссэн аливаа маргааныг талууд юун түрүүнд харилцан зөвшилцөх замаар шийдвэрлэхийг эрмэлзэнэ. Талууд маргааныг ийнхүү харилцан зөвшилцөх замаар шийдвэрлэж чадаагүй тохиолдолд Монгол Улсын эрх бүхий шүүх, эсхүл арбитрын журмаар шийдвэрлүүлнэ. Энэхүү гэрээгээр Илтгэгчээс Компанид илгээсэн дуу хоолойн Бичлэг болон түүнд хамаарах мэдээлэл, түүнчлэн тэдгээрийг хадгалах, боловсруулах, ашиглах, дамжуулахтай холбоотой гэрээнд заасан эрх, зөвшөөрлийг ашиглах, дамжуулахтай холбоотой Компани болон түүний харилцагчийн хооронд байгуулах аливаа гэрээ, хэлцэлд бусад улс, тэр дундаа Америкийн Нэгдсэн Улсын Калифорни мужийн хууль тогтоомжийг мөрдлөг болгох боломжтой болохыг Илтгэгч миний бие ойлгож, хүлээн зөвшөөрч байна.",
    "E.5. Мэдээлэл ашиглах зөвшөөрөл олгосон тухай бүртгэл. Энэхүү гэрээг байгуулснаар Компани нь энэхүү гэрээ, Илтгэгчийн төрийн эрх бүхий байгууллагаас олгож, баталгаажсан иргэнийг таних баримт бичгийн зураг (регистрийн дугаар болон иргэний улсын бүртгэлийн дугаарыг халхалж, нууцалсан байх), мэдээлэл хадгалах, боловсруулах, ашиглах, дамжуулахтай холбоотой үйлдэл хийх зөвшөөрөл олгосныг нотлох баримт бичиг (тухайлбал, зөвшөөрлийг баталгаажуулсан арга, огноо, цагийн тэмдэглэгээ, IP хаяг, гэрээний хувилбар болон зөвшөөрөл хүчин төгөлдөр болсон огноо) зэргийг агуулсан зөвшөөрөл олгосны бүртгэлийг PDF өргөтгөлтэй цахим файл хэлбэрээр үүсгэнэ. Компани нь уг зөвшөөрөл олгосны бүртгэлийг Илтгэгчээс зохих журмын дагуу зөвшөөрөл авсныг нотлох баримт болгон аюулгүй байдлыг хангасан нөхцөлд хадгалахыг Илтгэгч миний бие бүрэн ойлгож, хүлээн зөвшөөрч байна.",
    "БАТАЛГААЖУУЛАЛТ",
    "Зөвшөөрлийг баталгаажуулах нүдийг сонгон тэмдэглэж, илгээснээр Илтгэгч миний бие энэхүү гэрээний монгол хэлээрх хувилбарыг бүрэн уншиж, танилцсан бөгөөд гэрээний агуулга, түүнээс үүдэн гарах эрх зүйн үр дагаврыг бүрэн ойлгож, зөвшөөрсөн бөгөөд энэхүү гэрээг байгуулах, хэрэгжүүлэхтэй холбоотой миний зүгээс өгсөн бүхий л мэдээлэл үнэн зөв болохын дээр миний бие дээр дурдсан гэрээний бүхий л нөхцөлийг өөрийн хүсэл зоригийн үндсэн дээр хүлээн зөвшөөрсөн болохоо үүгээр баталгаажуулж байна.",
    "(Систем дараах мэдээллийг бүртгэнэ: баталгаажсан Төрийн эрх бүхий байгууллагаас олгож, баталгаажсан Илтгэгчийн иргэнийг таних баримт бичиг; гэрээний бүрэн эх болон түүний монгол хэлээрх эх бичвэрийн криптограф хэш; зөвшөөрлийг баталгаажуулсан бичвэр болон зөвшөөрлийг баталгаажуулах нүдийг сонгож, идэвхжүүлсэн эсэх төлөв; огноо-цагийн тэмдэглэгээ; хэрэглэгчийн цахим бүртгэл болон систем дэх үйлдлийн бүртгэл; IP хаяг болон төхөөрөмжийн мэдээлэл; иргэнийг таних мэдээллийн баталгаажуулалтын үр дүн; Илтгэгчид эдгээр мэдээллийн татаж авах боломжит хуулбарыг хүргэсэн тухай мэдээлэл.)",
]
CONSENT_AGREEMENT_V2_EN = [
    "(English convenience translation. The Mongolian version governs — see Section E.3.)",
    "PART A — PARTIES AND PARTICIPATION",
    "A.1. Contracting party. This agreement is between [full name], a citizen of Mongolia, born [date of birth] (the \"Speaker\"), and GURVANBILEG LLC, a limited liability company registered in Mongolia (registration number: 2043149), registered address: Building 10, 4th khoroolol, 18th khoroo, Bayangol District, Ulaanbaatar, Mongolia; contact: hi@gurvanbileg.com / +976 72014897; privacy contact: privacy@gurvanbileg.com (the \"Company\"). The Company is the sole contracting party, data controller, and recipient of the rights granted under this agreement.",
    "A.2. Voluntary participation; my own voice. I confirm that all voice recordings I submit through the Company’s platform (together, the \"Recordings\") are of my own voice and were made voluntarily, with my full consent. I give this consent freely.",
    "A.3. Independent participation; no employment. I understand that this agreement does not create an employment relationship. I choose whether and when to participate; I have no fixed schedule, no minimum hours, and no guaranteed work; I use my own equipment; I may stop at any time; and I am paid per accepted recording rather than a salary. I represent that I have full legal capacity to enter this agreement.",
    "A.4. Age and identity verification. I confirm that I am at least 18 years old, as verified by the government-issued photo ID I have uploaded. I understand that my name, date of birth, and sex as shown on my government ID will be treated as the authoritative record and will replace any information I entered manually, and that payment may only be made to the person named on the verified ID. I understand that automated tools are used to verify my ID.",
    "A.5. Quality and payment. I understand that I will only submit Recordings that meet the project’s required standards, that I will work honestly, and that I will receive payment for accepted Recordings according to the payment terms of this project. Whether a Recording is accepted for payment is a separate matter from the rights granted in Part B, which apply to all Recordings I submit.",
    "PART B — GRANT OF RIGHTS",
    "B.1. Assignment. I hereby grant and assign to the Company all right, title, and interest, including rights of publicity; name, image and likeness; copyright; performer’s (related) rights; and all other intellectual property rights, in and to all Recordings. Without limiting the foregoing grant, I acknowledge that the Company shall have the irrevocable, worldwide, perpetual, royalty-free right, and right to authorize others, to store, process, edit, alter, use, reproduce, distribute, prepare derivative works of, publicly perform and display, license, sublicense (through multiple tiers), sell, and include the Recordings and any related project materials in any and all technologies, formats and platforms now known or hereafter developed, for any purpose, including for speech datasets, AI training datasets, research, product development, quality control, and commercial data products, without any additional payment, time limit, or geographic limitation.",
    "B.2. Statutory rights categories. For clarity, the grant in B.1 includes, to the maximum extent transferable under applicable law, my exclusive rights as a performer concerning fixation, transmission, reproduction, distribution, rental, and making the Recordings available through networks, and any phonogram-related rights I may hold. Where any such right cannot be assigned under applicable law, I grant the Company an exclusive, irrevocable, perpetual, worldwide, transferable, and sublicensable license to that right.",
    "B.3. Moral rights. To the maximum extent permitted by applicable law, I agree not to assert any moral rights I may have in the Recordings or material derived from the Recordings against the Company, its customers, licensees, sub-licensees, or assigns.",
    "B.4. Permitted AI uses. I understand and agree that the Recordings may be used for the development, training, testing, and evaluation of artificial-intelligence and machine-learning systems, including speech recognition, text-to-speech, and speech-processing technologies, and may be included in datasets licensed to the Company’s customers. This includes the creation of synthetic speech, including voices that may resemble my voice.",
    "B.5. Prohibited uses. The Company will contractually require its customers and licensees not to use the Recordings for fraud, impersonation of me, deceptive endorsements attributed to me, unlawful surveillance, defamatory or reputation-damaging uses, or unlawful discrimination.",
    "B.6. Prompt text. I understand that the texts I read are provided by the Company, that the Company is responsible for the rights in those texts, and that this agreement concerns my performance and voice, not authorship of the prompt texts.",
    "B.7. No claims. I will not assert, maintain, or consent to others bringing any claim, action, suit or demand of any kind whatsoever, in any jurisdiction, against the Company or its licensees, sub-licensees, or assigns, including claims grounded upon invasion of privacy, rights of publicity or other civil rights, or for the use of my voice or a sound-alike, in connection with the use of my Recordings or material derived from or otherwise related to the Recordings, except for uses prohibited by Section B.5 or by law.",
    "B.8. Onward transfer. I understand that the Company may assign, license, and sublicense the Recordings and datasets containing them to its affiliates and customers, and that those customers and licensees may further use and sublicense the Recordings as part of the datasets and products they obtain.",
    "B.9. Survival of granted rights. The rights I grant in this agreement over my Recordings are permanent and survive: (a) my decision to stop participating in the project; (b) the termination of this agreement for any reason; and (c) any later updates or new versions of this agreement. Recordings I previously completed remain valid and covered by this grant.",
    "PART C — PERSONAL DATA AND PRIVACY NOTICE",
    "C.1. Data controller. The data controller is GURVANBILEG LLC (details in Section A.1).",
    "C.2. What is collected. The Company collects and processes: my full name, date of birth, and sex (from my government ID); an image of my government ID with the registration number redacted by me; my voice recordings (which are biometric information); recording metadata; my account and session information on the Company's platform, activity logs, IP address, and device information; information reflecting my speech characteristics (such as pace, rhythm, consent-expressing phrases); my affirmation records (timestamp, method, agreement version); and my bank or payment account information.",
    "C.3. Purpose and legal basis. This information is processed, on the basis of my consent given in this agreement, for: identity and age verification; consent documentation; production, quality control, and delivery of speech datasets; payment processing; and legal compliance.",
    "C.4. Retention. The Company retains: (a) my ID image for 5 years following verification, after which it is deleted while the verification result is retained; (b) my consent records for as long as datasets containing my Recordings are commercially maintained plus 10 years; and (c) my Recordings for the duration described in this agreement.",
    "C.5. What buyers receive. Customers who license datasets containing my Recordings receive: my Recordings; transcripts and translations; a pseudonymous speaker identifier; my sex, age band, and country; and a consent-evidence certificate (verified-adult status, verification date and method, agreement version, affirmation timestamp and method, and rights-grant summary). My ID image is not routinely provided to customers. The Company retains the underlying identity evidence securely and may disclose it only during a justified audit, dispute, or legal requirement, under confidentiality and minimized to what is necessary.",
    "C.6. My rights. I may request access to and correction of my personal information, and I may withdraw my consent for future collection at any time by submitting a request through my account on the Company's platform. Withdrawal does not affect the validity of the rights granted over Recordings already submitted, and processing carried out before withdrawal remains lawful. If I decline to provide the information in C.2, I cannot participate in the project. I may direct complaints to the Company’s privacy contact (A.1) or to the competent Mongolian authority.",
    "C.7. Public disclosure. My personal identity will not be published. My Recordings and the pseudonymous data in C.5 will be included in datasets provided to customers.",
    "PART D — BIOMETRIC DATA AND CROSS-BORDER TRANSFER CONSENT",
    "D.1. Biometric data. I understand that my voice recordings are biometric information capable of identifying me, and I expressly consent to their collection, storage, and processing by the Company for the purposes in Section C.3. ",
    "D.2. Cross-border transfer. I expressly consent to the transfer of the data described in Section C.5 (my Recordings, transcripts, and the pseudonymous consent-evidence data — not my ID image) to the Company’s customers located outside Mongolia, including in the United States of America and other countries or regions as applicable, being companies and organizations that license speech datasets for the purposes described in Section B.4. I understand that these customers may act as independent controllers of the transferred data, that they may further sublicense the datasets as described in Section B.8, that transfers are made under written agreements containing security and use restrictions, and that I may obtain information about the categories of recipients from the Company’s privacy contact. Withdrawal of my consent does not affect datasets already lawfully delivered.",
    "PART E — GENERAL TERMS",
    "E.1. Application to prior submissions; superseding effect. This agreement supersedes and replaces all earlier versions of the consent agreement between me and the Company. On the date I affirm this agreement, I confirm and re-grant the rights described in Part B in and to all Recordings I previously submitted under any earlier version. Earlier consent versions are preserved by the Company as part of my consent record.",
    "E.2. Future updates; continued participation. The Company may update this agreement from time to time. I will be shown any updated version in full, with a summary of material changes, and asked to affirm it before continuing to record. No update will reduce or revoke the rights I have already granted over Recordings previously submitted.",
    "E.3. Language. This agreement is presented to me in Mongolian and English. The Mongolian version is the legally controlling version between me and the Company; the English version is a convenience translation. If the two versions conflict, the Mongolian version prevails. I confirm that the complete Mongolian version was displayed to me before affirmation.",
    "E.4. Governing law and disputes. This agreement between me and the Company is governed by the laws of Mongolia. Disputes between me and the Company shall be resolved by negotiation, then the competent courts of Mongolia or arbitration. I understand that agreements between the Company and its customers may be governed by other laws, including the laws of the State of California, U.S.A.",
    "E.5. Consent record. A consent record (PDF) containing this agreement, my redacted ID image, and my affirmation details (method, timestamp, IP address, agreement version, and effective consent date) will be created and retained securely by the Company as evidence that my consent was properly obtained.",
    "AFFIRMATION",
    "By ticking the affirmation box and submitting, I confirm that the complete Mongolian version of this agreement was displayed to me, that I have read and understood it, that all information I have provided is true, and that I agree to all of the terms above.",
    "(System captures: verified speaker identity; agreement version and hash of the displayed Mongolian text; affirmation wording and checkbox status; timestamp; account/session ID; IP address and device information; ID-verification result; and delivery of a downloadable copy to the Speaker.)",
]
# Баталгаажуулалт — the affirmation wording rendered with the checkbox and recorded on
# every consent_records row. Part of the certified text; embedded at finalize.
CONSENT_V2_AFFIRMATION_MN = "Зөвшөөрлийг баталгаажуулах нүдийг сонгон тэмдэглэж, илгээснээр Илтгэгч миний бие энэхүү гэрээний монгол хэлээрх хувилбарыг бүрэн уншиж, танилцсан бөгөөд гэрээний агуулга, түүнээс үүдэн гарах эрх зүйн үр дагаврыг бүрэн ойлгож, зөвшөөрсөн бөгөөд энэхүү гэрээг байгуулах, хэрэгжүүлэхтэй холбоотой миний зүгээс өгсөн бүхий л мэдээлэл үнэн зөв болохын дээр миний бие дээр дурдсан гэрээний бүхий л нөхцөлийг өөрийн хүсэл зоригийн үндсэн дээр хүлээн зөвшөөрсөн болохоо үүгээр баталгаажуулж байна."

# Shown to existing speakers when the version gate blocks them (admin-supplied, verbatim).
# NOTE: the summary-of-material-changes box was removed at the admin's explicit decision
# (2026-08-23); the gate message below is the sole change-notice carrier.
CONSENT_GATE_MESSAGE_MN = "Манай төсөлтэй хамтран ажиллаж байгаа танд баярлалаа. Манай төсөл анхан шатандаа яваа болохоор шинэчлэл сайжруулалт ойр ойрхон хийгдэж байна. Сүүлийн шинэчлэлийн хүрээнд: Profile дотор 'дуу хоолойн өгөгдөл ашиглах зөвшөөрөл' шинэчлэгдсэн тул та 'Дахин баталгаажуулах' шаардлагатай. Шинэ хувилбар нь хувийн нууцлалд илүү ээлтэй болж онцгой бус шаардлагаар иргэний үнэмлэхийн мэдээллийг гадаад байгууллагуудруу дэлгэхгүй байхаар болж илүү нарийвчлалтай болж сайжирч байгаа юм. Үүний дараа та хэвийн ажилаа үргэлжлүүлэх боломжтой."

# Master switch for the v2.0 re-affirmation gate (record/submit blocking).
# ON: speakers whose consent_version != CURRENT_CONSENT_VERSION are blocked from
# recording and redirected to re-affirm. Set False to disable instantly.
CONSENT_GATE_ENABLED = True

def _canonical_consent_text(paragraphs):
    """Deterministic serialization of the agreement for hashing: each paragraph with
    normalized newlines and no leading/trailing whitespace, joined by exactly one blank
    line. The hash is computed over the TEMPLATE (A.1's […] slots in literal form),
    never over a personalized render — so every speaker records the same hash."""
    parts = [(p or "").replace("\r\n", "\n").replace("\r", "\n").strip() for p in paragraphs]
    return "\n\n".join(parts)

def consent_text_sha256(paragraphs):
    import hashlib
    return hashlib.sha256(_canonical_consent_text(paragraphs).encode("utf-8")).hexdigest()

# Set at finalize to the SHA-256 of the canonical certified Mongolian template.
CURRENT_CONSENT_TEXT_SHA256 = "5b44b4f7ff1672d4b3c20519cb3ff5d25affb334d28c352bd5af01983d96213d"

def _consent_selfcheck():
    """Startup drift guard: every consent record's validity depends on the embedded
    Mongolian template matching the finalized hash. Logs loudly on mismatch (does not
    crash the app); the gate must not be enabled while this warns."""
    live = consent_text_sha256(CONSENT_AGREEMENT_V2_MN)
    if live != CURRENT_CONSENT_TEXT_SHA256:
        print("[CONSENT] WARNING: embedded v2.0 Mongolian text hash != CURRENT_CONSENT_TEXT_SHA256 "
              f"(live={live[:16]}…, expected={str(CURRENT_CONSENT_TEXT_SHA256)[:16]}…). "
              "Text is unfinalized or has drifted — do NOT enable the consent gate.")
_consent_selfcheck()

def _personalize_consent_paragraphs(paragraphs, full_name, dob):
    """Fill A.1's three literal […] slots from ID-VERIFIED values only (never
    self-entered): slot1 = patronymic, slot2 = given name (full_name is stored as
    '<patronymic> <given>' from the ID), slot3 = date of birth. If verified values are
    absent (first-time consent, before ID capture), the literal template is shown."""
    if not full_name or not dob:
        return list(paragraphs)
    parts = full_name.strip().split()
    if len(parts) >= 2:
        patronymic, given = " ".join(parts[:-1]), parts[-1]
    else:
        patronymic = given = full_name.strip()
    out = []
    for p in paragraphs:
        if p.strip().startswith("A.1.") and p.count("[…]") >= 3:
            q = p.replace("[…]", patronymic, 1).replace("[…]", given, 1).replace("[…]", dob, 1)
            out.append(q)
        else:
            out.append(p)
    return out

# Buyer-facing rights-grant summary for the consent-evidence certificate (C.5).
# Drafted strictly from certified Part B; approved verbatim by the admin 2026-08-23.
RIGHTS_GRANT_SUMMARY_EN = [
    "Rights granted by the speaker. Under a signed digital consent agreement (v2.0), the "
    "speaker granted the Company an irrevocable, worldwide, perpetual, royalty-free "
    "assignment of all rights in their voice recordings — including copyright, performer's "
    "rights, and name/image/likeness rights — with full authority to store, process, "
    "reproduce, distribute, license, and sublicense through multiple tiers, and to use the "
    "recordings for AI/ML development, training, testing, and evaluation (including speech "
    "recognition, text-to-speech, and speech processing) and in commercial data products, "
    "in any format or platform now known or later developed, with no additional payment, "
    "time limit, or geographic restriction. The grant includes the creation of synthetic "
    "speech, including voices resembling the speaker's, and it survives the speaker "
    "leaving the project or the agreement ending.",
    "Required limit (binding on all downstream users). Recipients and sublicensees may not "
    "use the recordings for: fraud; impersonation of the speaker; deceptive endorsements "
    "attributed to the speaker; unlawful surveillance; defamatory or reputation-damaging "
    "use; or unlawful discrimination. (Consent agreement §B.5.)",
]

def _age_band(age):
    """C.5 permits buyers to receive the speaker's AGE GROUP (насны бүлэг), not exact age."""
    try:
        a = int(age)
    except (TypeError, ValueError):
        return ""
    if a < 18:
        return "under-18"   # must never occur; surfaced loudly rather than hidden
    for lo, hi in ((18, 25), (26, 35), (36, 45), (46, 55), (56, 65)):
        if a <= hi:
            return f"{lo}-{hi}"
    return "66+"

def _consent_display_blocks(paragraphs):
    """Split agreement paragraphs into header/body blocks for display: section headers
    ('A ХЭСЭГ — …', 'PART A — …', 'БАТАЛГААЖУУЛАЛТ') render as headings."""
    import re as _re
    out = []
    for p in paragraphs:
        s = (p or "").strip()
        is_h = bool(_re.match(r"^([A-E]\s+ХЭСЭГ|PART\s+[A-E])", s)) or s in (
            "БАТАЛГААЖУУЛАЛТ", "CONFIRMATION", "ATTESTATION", "AFFIRMATION")
        out.append({"h": is_h, "x": p})
    return out

# Shown once to each pre-existing contributor who consented on PAPER (consent_method != 'digital'),
# when they try to record. They must re-affirm via the new digital consent flow on /profile,
# after which consent_method becomes 'digital' and this gate stops firing for them.
# Contributors onboarded after the digital flow launched have consent_method='digital' from
# day one, so they never see this message.
MIGRATION_MESSAGE_MN = (
    "Манай төсөлтэй хамтран ажиллаж байгаа танд баярлалаа. Манай төсөл анхан шатандаа яваа "
    "болохоор шинэчлэл сайжруулалт ойр ойрхон хийгдэж байгаад хүлцэл өчье. Сүүлийн шинэчлэлийн "
    "хүрээнд: Profile дотор дуу хоолойн өгөгдлийн зөвшөөрөл автомат горимд шилжихээс өмнө "
    "цаасаар баталгаажуулж элссэн уншигчид иргэний үнэмлэхээ шинэ автомат горимын дагуу давтан "
    "баталгаажуулах шаардлагатай болоод байна. Шинэ автомат горим нь хувийн нууцлалд илүү "
    "ээлтэй процесс болж байгаа юм. Үүний дараа та хэвийн ажилаа үргэлжлүүлэх боломжтой."
)

def _needs_digital_migration(db, speaker_id):
    """True if this contributor already has a consent on file but it isn't digital yet —
    i.e., the legacy paper-consent path. Used to gate /record and /submit_clip until they
    re-affirm via the new automated ID flow."""
    p = db.execute(
        "SELECT consent_b2_key, COALESCE(consent_method,'') AS cm FROM profiles WHERE speaker_id=?",
        (speaker_id,)).fetchone()
    return bool(p and p["consent_b2_key"] and p["cm"] != "digital")

def _needs_consent_reaffirmation(db, speaker_id):
    """Agreement-VERSION gate (v2.0): the speaker HAS consent on file but affirmed an
    older version → block /record and /submit_clip until they re-affirm the current
    certified text. Dormant while CONSENT_GATE_ENABLED is False. New signups affirm
    v2.0 from day one (consent_version is stamped at submit) and never match."""
    if not CONSENT_GATE_ENABLED:
        return False
    p = db.execute(
        "SELECT consent_b2_key, COALESCE(consent_version,'') AS v "
        "FROM profiles WHERE speaker_id=?", (speaker_id,)).fetchone()
    return bool(p and p["consent_b2_key"] and p["v"] != CURRENT_CONSENT_VERSION)

def _profile_is_locked(speaker_id):
    """True if an admin has locked this contributor's profile after reviewing it. While
    locked, the profile fields and the consent/ID flow are frozen so a contributor can't
    silently overwrite already-delivered identity/consent data."""
    if not speaker_id:
        return False
    db = get_db()
    row = db.execute(
        "SELECT COALESCE(locked,0) AS locked FROM profiles WHERE speaker_id=?",
        (speaker_id,)).fetchone()
    db.close()
    return bool(row and row["locked"])

def _consent_font(size):
    """Load a Cyrillic-capable TTF, preferring the bundled copy in static/."""
    from PIL import ImageFont
    for p in (os.path.join(app.root_path, "static", "DejaVuSans.ttf"),
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

def _build_consent_pdf(name, dob, id_bytes, request, first_consent_date=None):
    """Compose a consent record PDF (pure Pillow, no extra deps):
       page(s) 1..n: affirmation record + full bilingual agreement text,
       final page: the contributor's (already-redacted) ID image."""
    from PIL import Image, ImageDraw
    import datetime as _dt

    W, H = 1240, 1754          # A4 @ ~150 DPI
    MARGIN = 95
    LINE_GAP = 9

    f_title = _consent_font(30)
    f_head  = _consent_font(19)
    f_body  = _consent_font(19)
    f_small = _consent_font(16)

    pages = []
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    y = MARGIN

    def _wrap(text, font, max_w):
        out = []
        for para_line in text.split("\n"):
            cur = ""
            for w in para_line.split():
                test = (cur + " " + w).strip()
                if draw.textlength(test, font=font) <= max_w:
                    cur = test
                else:
                    if cur: out.append(cur)
                    cur = w
            out.append(cur)
        return out or [""]

    def write(text, font, gap_after=16):
        nonlocal img, draw, y
        for line in _wrap(text, font, W - 2 * MARGIN):
            lh = font.size + LINE_GAP
            if y + lh > H - MARGIN:
                pages.append(img)
                img = Image.new("RGB", (W, H), "white")
                draw = ImageDraw.Draw(img)
                y = MARGIN
            draw.text((MARGIN, y), line, fill="black", font=font)
            y += lh
        y += gap_after

    ts = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    sid = session.get("speaker_id", "")

    write(CONSENT_V2_TITLE_MN, f_title, 6)
    write(CONSENT_V2_TITLE_EN, f_small, 22)
    write("— ЗӨВШӨӨРӨЛ ОЛГОСНЫ БҮРТГЭЛ / CONSENT RECORD —", f_head, 10)
    write(f"Нэр / Name:  {name}", f_body, 3)
    write(f"Төрсөн огноо / DOB:  {dob or '—'}", f_body, 3)
    write(f"Speaker ID:  {sid}", f_body, 3)
    write(f"Гэрээний хувилбар / Agreement version:  {CURRENT_CONSENT_VERSION}", f_body, 3)
    write(f"Эх бичвэрийн SHA-256 / Text SHA-256:  {consent_text_sha256(CONSENT_AGREEMENT_V2_MN)}", f_small, 3)
    write("Баталгаажуулсан арга / Method:  Digital in-app consent (government ID verified)", f_body, 3)
    # The EFFECTIVE consent date is always the ORIGINAL one; a re-affirmation records
    # its own timestamp separately and never advances the effective date (E.1 / #6).
    if first_consent_date:
        write(f"Зөвшөөрөл хүчинтэй огноо / Effective consent date:  {first_consent_date[:10]}", f_body, 3)
        write(f"Энэ баталгаажуулалт / This affirmation:  {ts}", f_body, 3)
    else:
        write(f"Зөвшөөрөл хүчинтэй огноо / Effective consent date:  {ts[:10]}", f_body, 3)
        write(f"Энэ баталгаажуулалт / This affirmation:  {ts}", f_body, 3)
    write(f"IP:  {ip}", f_body, 14)
    write(f"[x]  {CONSENT_V2_AFFIRMATION_MN}", f_body, 22)
    # Full PERSONALIZED Mongolian agreement (A.1 filled from ID-verified values):
    for para in _personalize_consent_paragraphs(CONSENT_AGREEMENT_V2_MN, name, dob):
        write(para, f_body, 14)
    y += 6
    write("— English convenience translation (for reference; the Mongolian version governs — E.3) —", f_small, 10)
    for para in CONSENT_AGREEMENT_V2_EN:
        write(para, f_small, 12)

    pages.append(img)

    # Final page: the redacted ID image
    try:
        id_img = Image.open(io.BytesIO(id_bytes))
        if id_img.mode not in ("RGB", "L"):
            id_img = id_img.convert("RGB")
        page = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(page)
        d.text((MARGIN, MARGIN), "Иргэний үнэмлэх / Government ID", fill="black", font=f_head)
        avail_w, avail_h = W - 2 * MARGIN, H - 2 * MARGIN - 70
        iw, ih = id_img.size
        scale = min(avail_w / iw, avail_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        id_img = id_img.resize((nw, nh))
        page.paste(id_img, (MARGIN + (avail_w - nw) // 2, MARGIN + 70))
        pages.append(page)
    except Exception as e:
        print(f"[CONSENT] ID image embed failed: {e}")

    out = io.BytesIO()
    pages[0].save(out, format="PDF", save_all=True,
                  append_images=pages[1:], resolution=150.0)
    out.seek(0)
    return out.read()


def _build_consent_certificate_pdf(sid, cert):
    """Buyer-facing CONSENT EVIDENCE CERTIFICATE for one speaker (agreement C.5):
    pseudonymous attestation of verified consent — 18+ status, verification method,
    agreement version + governing-text hash, affirmation record, and the approved
    rights-grant summary. Contains NO name, date of birth, or identity-document
    image. Pure Pillow, same machinery as _build_consent_pdf."""
    from PIL import Image, ImageDraw
    import datetime as _dt

    W, H = 1240, 1754
    MARGIN = 95
    LINE_GAP = 9
    f_title = _consent_font(30)
    f_head  = _consent_font(20)
    f_body  = _consent_font(18)
    f_small = _consent_font(15)

    pages = []
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    y = MARGIN

    def _wrap(text, font, max_w):
        out = []
        for para_line in text.split("\n"):
            cur = ""
            for w_ in para_line.split():
                test = (cur + " " + w_).strip()
                if draw.textlength(test, font=font) <= max_w:
                    cur = test
                else:
                    if cur: out.append(cur)
                    cur = w_
            out.append(cur)
        return out or [""]

    def write(text, font, gap_after=14):
        nonlocal img, draw, y
        for line in _wrap(text, font, W - 2 * MARGIN):
            lh = font.size + LINE_GAP
            if y + lh > H - MARGIN:
                pages.append(img)
                img = Image.new("RGB", (W, H), "white")
                draw = ImageDraw.Draw(img)
                y = MARGIN
            draw.text((MARGIN, y), line, fill="black", font=font)
            y += lh
        y += gap_after

    write("CONSENT EVIDENCE CERTIFICATE", f_title, 4)
    write("MNKH Mongolian Speech Dataset — per-speaker consent attestation", f_small, 20)
    write(f"Speaker (pseudonymous ID):  {sid}", f_head, 12)
    write("Age verification:  18+ — verified against a government-issued photo ID", f_body, 3)
    write("Verification method:  Digital in-app consent with automated government-ID verification", f_body, 3)
    write(f"Effective consent date:  {cert.get('effective_date','')}", f_body, 3)
    write(f"Agreement version:  {cert.get('version','')}", f_body, 3)
    write(f"Agreement text SHA-256 (governing Mongolian text):  {cert.get('text_sha256','')}", f_small, 3)
    write(f"Affirmation:  {cert.get('affirmed_at','')} UTC — checkbox affirmation in the "
          f"speaker's authenticated account session", f_body, 20)
    write("RIGHTS GRANTED BY THE SPEAKER", f_head, 8)
    for para in RIGHTS_GRANT_SUMMARY_EN:
        write(para, f_body, 10)
    write("DATA PROTECTION", f_head, 8)
    write("This delivery contains no speaker name, date of birth, or identity-document image. "
          "The Company retains the full ID-verified consent record (agreement, redacted "
          "identity document, and affirmation details) internally, and may disclose it only "
          "where a justified audit, dispute, or legal requirement demands, in minimized form, "
          "per the consent agreement (C.5).", f_body, 20)
    write("Issued by MONGOLIANDATA LLC (California, USA) and Гурванбилэг ХХК "
          "(Ulaanbaatar, Mongolia)", f_small, 2)
    write(f"Generated {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", f_small, 0)

    pages.append(img)
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True,
                  append_images=pages[1:], resolution=150)
    return buf.getvalue()

def next_speaker_id(db):
    """Return the next sequential MNKH_SPK## ID by looking at existing IDs.
    Always returns max+1 — does not fill gaps from deletions, to avoid reusing
    IDs that may be referenced elsewhere. Pads to at least 2 digits.
    Operates on whichever connection is passed in (caller manages commit/close)."""
    import re
    rows = db.execute(
        "SELECT speaker_id FROM users WHERE speaker_id LIKE 'MNKH_SPK%'"
    ).fetchall()
    pattern = re.compile(r'^MNKH_SPK(\d+)$')
    max_n = 0
    for r in rows:
        m = pattern.match(r["speaker_id"])
        if m:
            n = int(m.group(1))
            if n > max_n: max_n = n
    next_n = max_n + 1
    return f"MNKH_SPK{next_n:02d}"

def fmt_hours(secs):
    secs=int(secs or 0); h,r=divmod(secs,3600); m=r//60
    if h>0 and m>0: return f"{h}h {m}m"
    if h>0: return f"{h}h"
    if m>0: return f"{m}m"
    return f"{secs}s"

def fmt_dur(secs):
    secs=int(secs or 0); h,r=divmod(secs,3600); m,s=divmod(r,60)
    return f"{h}:{m:02d}:{s:02d}" if h>0 else f"{m}:{s:02d}"

def normalize_prompt_text(text):
    """Normalize a prompt for duplicate detection.

    Rules: lowercase letters/digits only, all punctuation removed, internal whitespace collapsed to single spaces.
    This catches the common cases: "Сайн уу", "Сайн уу!", "САЙН УУ?", "  Сайн   уу  ", "Сайн, уу." all
    normalize to "сайн уу" — they're the same sentence dressed differently.

    Crucially, it does NOT catch legitimate Mongolian inflectional variation: "хүн" / "хүний" / "хүнд"
    remain distinct, so we don't false-positive on valid linguistic diversity.
    """
    if not text: return ""
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            chars.append(ch.lower())
        elif ch.isspace():
            chars.append(" ")
        # Everything else (punctuation, symbols) is dropped
    return " ".join("".join(chars).split())

# ─────────────────────────────────────────────────────────────────────────────
# Bilingual taxonomy for contributor profiles.
# DB stores ENGLISH values. Profile form shows "Mongolian (English)" labels.
# Delivery metadata CSVs emit the English values directly — buyers get clean,
# normalized data with no translation step on their side.
# ─────────────────────────────────────────────────────────────────────────────

REGION_CHOICES = [
    # (db_value_english, display_label_mongolian)
    ("Ulaanbaatar",    "Улаанбаатар"),
    ("Darkhan",        "Дархан"),
    ("Erdenet",        "Эрдэнэт"),
    ("Choibalsan",     "Чойбалсан"),
    ("Ulgii",          "Өлгий"),
    ("Khovd",          "Ховд"),
    ("Moron",          "Мөрөн"),
    ("Dalanzadgad",    "Даланзадгад"),
    ("Arvaikheer",     "Арвайхээр"),
    ("Ondorkhaan",     "Өндөрхаан"),
    ("Sukhbaatar",     "Сүхбаатар"),
    ("Bayankhongor",   "Баянхонгор"),
    ("Altai",          "Алтай"),
    ("Uliastai",       "Улиастай"),
    ("Tsetserleg",     "Цэцэрлэг"),
    ("Mandalgovi",     "Мандалговь"),
    ("Sainshand",      "Сайншанд"),
    ("Other",          "Бусад"),
]

EDUCATION_CHOICES = [
    ("Primary school",         "Бага сургууль"),
    ("Secondary school",       "Дунд сургууль"),
    ("Vocational/Technical",   "Мэргэжлийн техникийн сургууль"),
    ("Bachelor's",             "Бакалавр"),
    ("Master's",               "Магистр"),
    ("Doctorate",              "Доктор"),
]

DIALECT_CHOICES = [
    ("Khalkha",    "Халх"),
    ("Buryat",     "Буриад"),
    ("Oirat",      "Ойрад"),
    ("Dariganga",  "Дарьганга"),
    ("Khorchin",   "Хорчин"),
    ("Other",      "Бусад"),
]

# Reverse maps used during the one-time backfill: translate any Mongolian values
# currently stored in the DB to their English equivalents.
_REGION_MN_TO_EN = {mn: en for en, mn in REGION_CHOICES}
_EDUCATION_MN_TO_EN = {mn: en for en, mn in EDUCATION_CHOICES}
_DIALECT_MN_TO_EN = {mn: en for en, mn in DIALECT_CHOICES}

# Recording device taxonomy — admin-editable per contributor.
# We use a small fixed set to keep buyer-facing data normalized.
RECORDING_DEVICE_CHOICES = [
    "Mobile phone",
    "USB microphone",
    "Headset microphone",
    "Other",
]

def login_required(f):
    @wraps(f)
    def d(*a,**k):
        if "user_id" not in session: return redirect(url_for("login"))
        return f(*a,**k)
    return d

def admin_required(f):
    @wraps(f)
    def d(*a,**k):
        if session.get("role")!="admin": abort(403)
        return f(*a,**k)
    return d

def can_manage_speaker(speaker_id):
    """Returns True if current user is admin, or an editor who owns this contributor."""
    role = session.get("role")
    if role == "admin": return True
    if role == "editor":
        db = get_db()
        row = db.execute(
            "SELECT 1 FROM editor_contributors WHERE editor_user_id=? AND contributor_speaker_id=?",
            (session["user_id"], speaker_id)
        ).fetchone()
        db.close()
        return bool(row)
    return False

def manager_required(f):
    """Allows admin OR an editor who owns the speaker_id passed as URL arg."""
    @wraps(f)
    def d(*a,**k):
        speaker_id = k.get("speaker_id")
        if speaker_id is None: abort(403)
        if not can_manage_speaker(speaker_id): abort(403)
        return f(*a,**k)
    return d

def admin_or_login_required(f):
    """Allows any logged-in admin or editor — ownership must be checked inside the handler."""
    @wraps(f)
    def d(*a,**k):
        if "user_id" not in session: return redirect(url_for("login"))
        if session.get("role") not in ("admin","editor"): abort(403)
        return f(*a,**k)
    return d

def _post_action_redirect():
    """Return the right dashboard for current user role."""
    return url_for("editor_home") if session.get("role") == "editor" else url_for("admin_dashboard")

def profile_required(f):
    """For contributors: require profile completion AND consent upload before recording.
    For editors: do NOT gate. A new editor lands on her dashboard immediately and
    can start working. Her earnings card on the dashboard flags missing payment
    info visually (red value + ? tooltip) so she knows to fix it in /profile,
    but nothing blocks her from doing real work first.
    Admins are always allowed."""
    @wraps(f)
    def d(*a,**k):
        role = session.get("role")
        if role == "contributor":
            db = get_db()
            p = db.execute(
                "SELECT completed, consent_b2_key FROM profiles WHERE speaker_id=?",
                (session["speaker_id"],)
            ).fetchone()
            db.close()
            if not p or not p["completed"]:
                flash("Бичлэг хийхийн өмнө профайлаа бүрэн бөглөнө үү.","info")
                return redirect(url_for("profile"))
            if not p["consent_b2_key"]:
                flash("Бичлэг хийхийн өмнө зөвшөөрлийн маягтыг хэвлэж, гарын үсэг зурж, аппликэйшн руу буцааж байршуулна уу.","info")
                return redirect(url_for("profile"))
        # Editors fall through — no gating. Visual nudge handled on editor_home.
        return f(*a,**k)
    return d

# ── Auth ──────────────────────────────────────────────────────────────────────
def landing_url_for_role(role):
    """Where each role lands after login."""
    if role == "admin":      return url_for("admin_dashboard")
    if role == "editor":     return url_for("editor_home")
    return url_for("contributor_home")

# ── Public info pages (handbooks, apply, FAQ) ─────────────────────────────────
# These are linked from the hamburger menu on the login page and are intentionally
# accessible without authentication so prospective contributors and editors can read them.
@app.route("/handbook/contributor")
def contributor_handbook():
    return render_template("contributor_handbook.html")

@app.route("/handbook/editor")
def editor_handbook():
    return render_template("editor_handbook.html")

@app.route("/apply")
def apply():
    return render_template("apply.html")

@app.route("/apply/submit", methods=["POST"])
def apply_submit():
    """Public endpoint — receives job application form submissions."""
    # Honeypot check: if hidden field is filled, silently reject (bots only)
    if request.form.get("website", "").strip():
        return redirect(url_for("apply_thanks"))

    full_name = (request.form.get("full_name") or "").strip()
    phone     = (request.form.get("phone") or "").strip()
    position  = (request.form.get("position") or "").strip()
    is_adult  = 1 if request.form.get("is_adult") == "yes" else 0

    # Required field validation
    if not full_name or not phone or not position or not is_adult:
        return render_template("apply.html",
            error="Бүтэн нэр, утасны дугаар, ажлын байр болон насанд хүрсэн баталгаажуулалт шаардлагатай.",
            form_data=request.form
        ), 400

    if position not in ("contributor", "editor", "both"):
        return render_template("apply.html",
            error="Хүчингүй ажлын байр сонгогдсон байна.",
            form_data=request.form
        ), 400

    db = get_db()
    db.execute("""
        INSERT INTO applications
            (full_name, phone, position, is_adult, age, email, city, facebook,
             hours_per_week, prior_experience, heard_from, message,
             ip_address, user_agent, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
    """, (
        full_name, phone, position, is_adult,
        (request.form.get("age") or "").strip() or None,
        (request.form.get("email") or "").strip() or None,
        (request.form.get("city") or "").strip() or None,
        (request.form.get("facebook") or "").strip() or None,
        (request.form.get("hours_per_week") or "").strip() or None,
        (request.form.get("prior_experience") or "").strip() or None,
        (request.form.get("heard_from") or "").strip() or None,
        (request.form.get("message") or "").strip() or None,
        request.headers.get("X-Forwarded-For", request.remote_addr or ""),
        (request.headers.get("User-Agent") or "")[:500],
    ))
    db.commit()
    return redirect(url_for("apply_thanks"))

@app.route("/apply/thanks")
def apply_thanks():
    return render_template("apply_thanks.html")

@app.route("/admin/applications")
@login_required
@admin_required
def admin_applications():
    """Admin view of all job applications."""
    status_filter = request.args.get("status", "active")
    db = get_db()
    if status_filter == "all":
        rows = db.execute("SELECT * FROM applications ORDER BY submitted_at DESC").fetchall()
    elif status_filter == "active":
        rows = db.execute(
            "SELECT * FROM applications WHERE status IN ('new','contacted') "
            "ORDER BY submitted_at DESC"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM applications WHERE status=? ORDER BY submitted_at DESC",
            (status_filter,)
        ).fetchall()

    counts = {}
    for row in db.execute(
        "SELECT status, COUNT(*) AS n FROM applications GROUP BY status"
    ).fetchall():
        counts[row["status"]] = row["n"]
    counts["all"] = sum(counts.values())
    counts["active"] = counts.get("new", 0) + counts.get("contacted", 0)

    return render_template(
        "admin_applications.html",
        applications=rows,
        status_filter=status_filter,
        counts=counts
    )

@app.route("/admin/applications/<int:app_id>/update", methods=["POST"])
@login_required
@admin_required
def admin_application_update(app_id):
    """Update an application's status and/or admin notes."""
    new_status = request.form.get("status", "").strip()
    admin_notes = (request.form.get("admin_notes") or "").strip()
    valid_statuses = ("new", "contacted", "hired", "rejected", "archived")
    if new_status and new_status not in valid_statuses:
        abort(400)
    db = get_db()
    if new_status:
        db.execute(
            "UPDATE applications SET status=?, admin_notes=? WHERE id=?",
            (new_status, admin_notes, app_id)
        )
    else:
        db.execute(
            "UPDATE applications SET admin_notes=? WHERE id=?",
            (admin_notes, app_id)
        )
    db.commit()
    return redirect(request.referrer or url_for("admin_applications"))

@app.route("/admin/applications/<int:app_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_application_delete(app_id):
    """Permanently delete an application."""
    db = get_db()
    db.execute("DELETE FROM applications WHERE id=?", (app_id,))
    db.commit()
    return redirect(url_for("admin_applications"))

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/", methods=["GET","POST"])
def login():
    if "user_id" in session:
        return redirect(landing_url_for_role(session.get("role","contributor")))
    error=None
    if request.method=="POST":
        u=request.form.get("username","").strip()
        p=request.form.get("password","")
        db=get_db()
        user=db.execute("SELECT * FROM users WHERE username=?",(u,)).fetchone()
        db.close()
        if user and not user["frozen"] and not user["archived"] and check_password_hash(user["password"],p):
            session.update(user_id=user["id"],username=user["username"],
                           speaker_id=user["speaker_id"],role=user["role"])
            return redirect(landing_url_for_role(user["role"]))
        if user and user["archived"]:
            error="Таны аккаунт идвэхгүй удсан тул хасагдах жагсаалтанд орсон байна."
        elif user and user["frozen"]:
            error="Your account has been temporarily frozen. Contact the project coordinator."
        else:
            error="Invalid username or password."
    return render_template("login.html",error=error)

@app.route("/editor")
@login_required
@profile_required
def editor_home():
    """Editor dashboard — only their own contributors visible."""
    if session.get("role") != "editor": abort(403)
    editor_user_id = session["user_id"]
    db = get_db()
    # Get all contributors assigned to this editor
    contributors = db.execute(
        "SELECT u.*, "
        "  (SELECT COUNT(*) FROM prompts WHERE speaker_id=u.speaker_id) as total_prompts, "
        "  (SELECT COUNT(*) FROM prompts p WHERE p.speaker_id=u.speaker_id "
        "    AND NOT EXISTS (SELECT 1 FROM clips c WHERE c.prompt_id=p.id AND c.status != 'rejected')) as todo_count, "
        "  (SELECT COUNT(*) FROM clips WHERE speaker_id=u.speaker_id AND status='approved') as approved, "
        "  (SELECT COUNT(*) FROM clips WHERE speaker_id=u.speaker_id AND status='pending') as pending, "
        "  (SELECT COUNT(*) FROM clips WHERE speaker_id=u.speaker_id AND status='rejected') as rejected, "
        "  (SELECT COALESCE(SUM(duration_seconds),0) FROM clips WHERE speaker_id=u.speaker_id AND status='approved') as approved_sec, "
        "  (SELECT COALESCE(SUM(duration_seconds),0) FROM clips WHERE speaker_id=u.speaker_id AND status='approved' AND compensated=0) as uncomp_sec, "
        "  (p.completed AND COALESCE(p.consent_b2_key,'')<>'') as profile_complete, "
        "  CASE "
        "    WHEN (p.completed AND COALESCE(p.consent_b2_key,'')<>'' "
        f"          AND COALESCE(p.consent_version,'') = '{CURRENT_CONSENT_VERSION}') THEN 'current' "
        "    WHEN (p.completed AND COALESCE(p.consent_b2_key,'')<>'') THEN 'stale' "
        "    ELSE 'none' END as consent_state "
        "FROM users u "
        "JOIN editor_contributors ec ON ec.contributor_speaker_id = u.speaker_id "
        "LEFT JOIN profiles p ON p.speaker_id = u.speaker_id "
        "WHERE ec.editor_user_id = ? AND u.role='contributor' AND COALESCE(u.archived,0)=0 "
        "ORDER BY u.created_at DESC",
        (editor_user_id,)
    ).fetchall()
    # Latest admin message for this editor
    msg_row = db.execute(
        "SELECT message, created_at FROM editor_messages WHERE editor_user_id=? "
        "ORDER BY created_at DESC LIMIT 1", (editor_user_id,)
    ).fetchone()
    # Pending contributor request from this editor (if any)
    pending_request = db.execute(
        "SELECT * FROM contributor_requests WHERE editor_user_id=? AND status='pending' "
        "ORDER BY created_at DESC LIMIT 1", (editor_user_id,)
    ).fetchone()
    # Editor's contributor cap = 20 + each previously approved request
    approved_extras = db.execute(
        "SELECT COUNT(*) as n FROM contributor_requests WHERE editor_user_id=? AND status='approved'",
        (editor_user_id,)
    ).fetchone()["n"]
    # Editor's own hourly rate
    me = db.execute("SELECT hourly_rate, editor_penalty_pct, editor_penalty_reason FROM users WHERE id=?", (editor_user_id,)).fetchone()
    # Editor's review queue: pending clips from their assigned contributors
    my_queue_count = db.execute(
        "SELECT COUNT(*) as n FROM clips c "
        "JOIN editor_contributors ec ON ec.contributor_speaker_id = c.speaker_id "
        "WHERE c.status='pending' AND ec.editor_user_id = ?",
        (editor_user_id,)
    ).fetchone()["n"]
    # Editor's own unpaid earnings: clips THEY approved that admin finalized & not yet paid
    editor_unpaid_clips = db.execute(
        "SELECT duration_seconds, editor_rate_at_approval "
        "FROM clips WHERE editor_user_id=? AND status='approved' AND COALESCE(editor_paid,0)=0",
        (editor_user_id,)
    ).fetchall()
    # Clips the editor approved but admin overruled (admin rejected during final review).
    # Distinguishing condition: status='rejected' AND admin_reviewed_at IS NOT NULL
    # (a clip the editor rejected directly would have admin_reviewed_at NULL since admin never saw it)
    rejected_by_admin_count = db.execute(
        "SELECT COUNT(*) as n FROM clips "
        "WHERE editor_user_id=? AND status='rejected' "
        "  AND admin_reviewed_at IS NOT NULL",
        (editor_user_id,)
    ).fetchone()["n"]
    # Profile completeness check — drives the red-earnings-with-tooltip warning.
    # Editor's profile is "complete" when full_name/phone/bank_name/iban are all
    # set (see profile route, line ~916). We detect missing payment info as
    # "no profile row" OR "completed=0" OR explicitly-blank bank_name/iban —
    # any of these means we don't have what we need to pay her, so flag it.
    prof_row = db.execute(
        "SELECT completed, bank_name, iban FROM profiles WHERE speaker_id=?",
        (session["speaker_id"],)
    ).fetchone()
    payment_info_missing = (
        not prof_row
        or not prof_row["completed"]
        or not (prof_row["bank_name"] or "").strip()
        or not (prof_row["iban"] or "").strip()
    )
    db.close()

    contributor_count = len(contributors)
    cap = 20 + approved_extras
    can_self_add = contributor_count < cap

    # Aggregate stats across all my contributors (computed here, not in Jinja)
    total_approved_sec = sum(c["approved_sec"] or 0 for c in contributors)
    # Editor's audio metrics come from THEIR approved clips, not just from their contributors' totals
    total_uncomp_sec = sum(uc["duration_seconds"] or 0 for uc in editor_unpaid_clips)
    editor_rate = me["hourly_rate"] if me else 70000
    # Group by frozen editor_rate_at_approval (or fallback to current rate), ceil per group
    my_uncomp_amount = calc_comp_grouped(
        editor_unpaid_clips, "editor_rate_at_approval", editor_rate
    )
    # Active quality penalty (admin-set) reduces the payout until the next settle.
    my_uncomp_amount = apply_editor_penalty(my_uncomp_amount, me["editor_penalty_pct"])

    # Conversations: my circle's sessions waiting for my editing, and unpaid approved ones.
    _cdb = get_db()
    conv_queue_count = _cdb.execute(
        "SELECT COUNT(*) n FROM conv_sessions WHERE editor_id=? AND status IN ('drafted','rejected','in_edit')",
        (editor_user_id,)).fetchone()["n"]
    conv_rows = _conv_pay_rows_editor(_cdb, editor_user_id)
    conv_uncomp_amount = apply_editor_penalty(_conv_pay_amount(conv_rows, editor_rate), me["editor_penalty_pct"] if me else 0)
    conv_uncomp_sec = sum(r["duration_seconds"] for r in conv_rows)
    _cdb.close()
    return render_template("editor_home.html",
        conv_queue_count=conv_queue_count, conv_uncomp_amount=conv_uncomp_amount, conv_uncomp_sec=conv_uncomp_sec,
        chat_unread=chat_unread_count(),
        contributors=contributors,
        contributor_count=contributor_count,
        cap=cap,
        can_self_add=can_self_add,
        directive=msg_row["message"] if msg_row else None,
        directive_date=msg_row["created_at"][:10] if msg_row else None,
        pending_request=pending_request,
        editor_rate=editor_rate,
        total_approved_sec=total_approved_sec,
        total_uncomp_sec=total_uncomp_sec,
        my_uncomp_amount=my_uncomp_amount,
        my_queue_count=my_queue_count,
        rejected_by_admin_count=rejected_by_admin_count,
        payment_info_missing=payment_info_missing,
        username=session.get("username"),
        fmt_hours=fmt_hours, calc_comp=calc_comp)

@app.route("/editor/rejected")
@login_required
@profile_required
def editor_rejected():
    """List clips the editor approved but the admin overruled (rejected at final review).
    The editor needs to know about these so they can coach the contributor and
    apply the same standards in their own first-stage reviews going forward."""
    if session.get("role") != "editor": abort(403)
    editor_user_id = session["user_id"]
    db = get_db()
    clips = db.execute(
        "SELECT c.id, c.filename, c.speaker_id, c.reject_note, c.duration_seconds, "
        "       c.admin_reviewed_at, c.editor_reviewed_at, "
        "       p.text_mn, p.text_en, p.speech_type, p.clip_number, "
        "       u.username AS contributor_username "
        "FROM clips c "
        "JOIN prompts p ON p.id = c.prompt_id "
        "LEFT JOIN users u ON u.speaker_id = c.speaker_id "
        "WHERE c.editor_user_id = ? "
        "  AND c.status = 'rejected' "
        "  AND c.admin_reviewed_at IS NOT NULL "
        "ORDER BY c.admin_reviewed_at DESC",
        (editor_user_id,)
    ).fetchall()
    db.close()
    return render_template("editor_rejected.html", clips=clips, fmt_dur=fmt_dur)

@app.route("/editor/new_contributor", methods=["GET","POST"])
@login_required
@profile_required
def editor_new_contributor():
    if session.get("role") != "editor": abort(403)
    editor_user_id = session["user_id"]
    db = get_db()
    current_count = db.execute(
        "SELECT COUNT(*) as n FROM editor_contributors WHERE editor_user_id=?",
        (editor_user_id,)
    ).fetchone()["n"]
    approved_extras = db.execute(
        "SELECT COUNT(*) as n FROM contributor_requests WHERE editor_user_id=? AND status='approved'",
        (editor_user_id,)
    ).fetchone()["n"]
    cap = 20 + approved_extras
    if current_count >= cap:
        db.close()
        flash(f"You have reached your {cap}-contributor limit. Request approval from admin to add more.","error")
        return redirect(url_for("editor_home"))

    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","").strip()
        # Editor cannot specify speaker_id — system assigns it sequentially.
        # We retry on collision to handle the rare race where two editors submit at the same instant.
        if not u or not p:
            flash("Username and password are required.","error")
            db.close()
            return redirect(url_for("editor_new_contributor"))
        if len(p) < 6:
            flash("Password must be at least 6 characters.","error")
            db.close()
            return redirect(url_for("editor_new_contributor"))
        last_err = None
        for attempt in range(5):
            s = next_speaker_id(db)
            try:
                db.execute(
                    "INSERT INTO users (username,password,speaker_id,role,hourly_rate) VALUES (?,?,?,?,?)",
                    (u, generate_password_hash(p), s, "contributor", 35000)
                )
                db.execute(
                    "INSERT INTO editor_contributors (editor_user_id,contributor_speaker_id,assigned_by) VALUES (?,?,?)",
                    (editor_user_id, s, "editor")
                )
                db.commit()
                db.close()
                flash(f"Contributor '{u}' created with speaker ID {s} and assigned to you.","success")
                return redirect(url_for("editor_home"))
            except sqlite3.IntegrityError as e:
                # Two possibilities: username collision (won't help to retry) or speaker_id race
                msg = str(e).lower()
                if "username" in msg or "users.username" in msg:
                    db.close()
                    flash("That username already exists. Please choose a different one.","error")
                    return redirect(url_for("editor_new_contributor"))
                # Otherwise it's likely a speaker_id race — rollback and retry
                db.rollback()
                last_err = e
        db.close()
        flash("Could not assign a speaker ID after several attempts. Please try again.","error")
        return redirect(url_for("editor_new_contributor"))

    # GET: predict the next ID for display (purely informational; not enforced)
    predicted_id = next_speaker_id(db)
    db.close()
    return render_template("editor_new_contributor.html",
        username=session.get("username"),
        slots_remaining=cap-current_count,
        predicted_speaker_id=predicted_id)

@app.route("/editor/request_more", methods=["POST"])
@login_required
@profile_required
def editor_request_more():
    """Editor requests admin approval to add a 21st+ contributor."""
    if session.get("role") != "editor": abort(403)
    editor_user_id = session["user_id"]
    note = request.form.get("note","").strip()[:500]
    db = get_db()
    # Don't create duplicate pending requests
    existing = db.execute(
        "SELECT id FROM contributor_requests WHERE editor_user_id=? AND status='pending'",
        (editor_user_id,)
    ).fetchone()
    if existing:
        db.close()
        flash("You already have a pending request. Please wait for admin response.","info")
        return redirect(url_for("editor_home"))
    db.execute(
        "INSERT INTO contributor_requests (editor_user_id,note,status) VALUES (?,?,?)",
        (editor_user_id, note or None, "pending")
    )
    db.commit(); db.close()
    flash("Request submitted to admin.","success")
    return redirect(url_for("editor_home"))

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/admin/bulk_delete_prompts", methods=["POST"])
@login_required
@admin_or_login_required
@profile_required
def bulk_delete_prompts():
    speaker_id = request.form.get("speaker_id","")
    if not can_manage_speaker(speaker_id): abort(403)
    ids = request.form.getlist("prompt_ids")
    if not ids:
        flash("No prompts selected.","error")
        return redirect(url_for("add_prompts", speaker_id=speaker_id))
    db = get_db()
    deleted = 0
    for pid in ids:
        p = db.execute("SELECT * FROM prompts WHERE id=? AND speaker_id=?",(pid,speaker_id)).fetchone()
        if not p: continue
        clip = db.execute("SELECT status FROM clips WHERE prompt_id=?",(pid,)).fetchone()
        if clip and clip["status"] == "approved": continue  # never delete approved
        if clip: db.execute("DELETE FROM clips WHERE prompt_id=?",(pid,))
        db.execute("DELETE FROM prompts WHERE id=?",(pid,))
        deleted += 1
    db.commit(); db.close()
    flash(f"Deleted {deleted} prompt(s). Approved prompts are protected and were skipped.","success")
    return redirect(url_for("add_prompts", speaker_id=speaker_id))

@app.route("/admin/change_password", methods=["POST"])
@login_required
@admin_required
def change_password():
    current = request.form.get("current_password","")
    new_pw  = request.form.get("new_password","").strip()
    confirm = request.form.get("confirm_password","").strip()
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    if not check_password_hash(user["password"], current):
        flash("Current password is incorrect.", "error")
        db.close()
        return redirect(url_for("admin_dashboard"))
    if len(new_pw) < 8:
        flash("New password must be at least 8 characters.", "error")
        db.close()
        return redirect(url_for("admin_dashboard"))
    if new_pw != confirm:
        flash("New passwords do not match.", "error")
        db.close()
        return redirect(url_for("admin_dashboard"))
    db.execute("UPDATE users SET password=? WHERE id=?",
        (generate_password_hash(new_pw), session["user_id"]))
    db.commit()
    db.close()
    flash("Password changed successfully.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/change_password", methods=["POST"])
@login_required
def change_password_user():
    """Password change for contributors and editors. Posts back to /profile.
    Admins use /admin/change_password from their dashboard modal instead."""
    role = session.get("role")
    if role == "admin":
        return redirect(url_for("change_password"))
    current = request.form.get("current_password","")
    new_pw  = request.form.get("new_password","").strip()
    confirm = request.form.get("confirm_password","").strip()
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    if not user or not check_password_hash(user["password"], current):
        db.close()
        if role == "editor":
            flash("Current password is incorrect.","error")
        else:
            flash("Одоогийн нууц үг буруу байна.","error")
        return redirect(url_for("profile"))
    if len(new_pw) < 6:
        db.close()
        if role == "editor":
            flash("New password must be at least 6 characters.","error")
        else:
            flash("Шинэ нууц үг хамгийн багадаа 6 тэмдэгт байх ёстой.","error")
        return redirect(url_for("profile"))
    if new_pw != confirm:
        db.close()
        if role == "editor":
            flash("New passwords do not match.","error")
        else:
            flash("Шинэ нууц үгүүд тохирохгүй байна.","error")
        return redirect(url_for("profile"))
    db.execute("UPDATE users SET password=? WHERE id=?",
        (generate_password_hash(new_pw), session["user_id"]))
    db.commit()
    db.close()
    if role == "editor":
        flash("Password changed successfully.","success")
    else:
        flash("Нууц үг амжилттай солигдлоо.","success")
    return redirect(url_for("profile"))

# ── Earnings history ──────────────────────────────────────────────────────────
@app.route("/earnings")
@login_required
def earnings_history():
    """Shows the current user's payment history. Available to contributors and
    editors — admins are bounced to their dashboard since they're not paid per clip."""
    role = session.get("role")
    if role == "admin":
        return redirect(url_for("admin_dashboard"))
    db = get_db()
    transactions = db.execute(
        "SELECT id, amount, clip_count, duration_seconds, paid_at, paid_by_username, method "
        "FROM payment_transactions WHERE user_id=? "
        "ORDER BY paid_at DESC",
        (session["user_id"],)
    ).fetchall()
    grand_total = sum(t["amount"] or 0 for t in transactions)
    total_clips = sum(t["clip_count"] or 0 for t in transactions)
    db.close()
    return render_template("earnings_history.html",
                           transactions=transactions,
                           grand_total=grand_total,
                           total_clips=total_clips,
                           role=role)

# ── Profile ───────────────────────────────────────────────────────────────────
@app.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    role = session.get("role")
    if role == "admin": return redirect(url_for("admin_dashboard"))
    sid = session["speaker_id"]
    db = get_db()
    prof = db.execute("SELECT * FROM profiles WHERE speaker_id=?",(sid,)).fetchone()
    # True only while the speaker is actually blocked by the consent-version gate:
    # drives the urgent (red, pulsing) styling of the re-affirm button.
    needs_reaffirm = _needs_consent_reaffirmation(db, sid)

    if request.method == "POST":
        # Locked by admin after review: silently ignore any edit. Fields are frozen in the
        # UI; this is the server-side backstop. No message shown, by design.
        if prof and prof["locked"]:
            db.close()
            return redirect(url_for("profile"))
        fn = request.form.get("full_name","").strip()
        phn = request.form.get("phone","").strip()
        bnk = request.form.get("bank_name","").strip()
        iban = request.form.get("iban","").strip().upper()
        iban = "".join(iban.split())   # drop any spaces (IBANs are often typed grouped)
        if iban == "MN":               # untouched 'MN' prefix prefill = no IBAN entered
            iban = ""
        rcv = request.form.get("receiver_name","").strip()

        # IBAN must be exactly 'MN' followed by 18 digits (20 characters total). Blank is
        # still allowed at this layer (the per-role required checks below decide whether
        # blank is acceptable for that role). 'MN'-only is normalized to blank above.
        if iban and not (len(iban) == 20 and iban[:2] == "MN"
                         and all(c in "0123456789" for c in iban[2:])):
            flash("Дансны дугаар (IBAN) нь 'MN' үсгээр эхэлж, ард нь яг 18 оронтой тоо байх ёстой (нийт 20 тэмдэгт).", "error")
            db.close()
            return redirect(url_for("profile"))

        if role == "editor":
            # Editors need payment info; photo is OPTIONAL
            if not all([fn, phn, bnk, iban]):
                flash("Please complete all required fields (name, phone, bank, IBAN).","error")
                db.close()
                return redirect(url_for("profile"))
            # Optional photo upload
            photo_key = prof["photo_b2_key"] if prof else None
            photo_file = request.files.get("photo")
            if photo_file and photo_file.filename:
                ext = photo_file.filename.rsplit(".",1)[-1].lower()
                if ext not in ("jpg","jpeg","png","webp"):
                    flash("Photo must be JPG, PNG or WebP.","error"); db.close()
                    return redirect(url_for("profile"))
                try:
                    b2 = get_b2(); photo_key = f"photos/{sid}_photo.{ext}"
                    b2.put_object(Bucket=B2_BUCKET_NAME, Key=photo_key,
                                  Body=photo_file.read(), ContentType=f"image/{ext}")
                except Exception as e:
                    print(f"Photo: {e}"); flash("Could not upload photo.","error"); db.close()
                    return redirect(url_for("profile"))
            if prof:
                db.execute(
                    "UPDATE profiles SET full_name=?, phone=?, bank_name=?, iban=?, "
                    "  receiver_name=?, photo_b2_key=?, completed=1, updated_at=datetime('now') "
                    "WHERE speaker_id=?",
                    (fn, phn, bnk, iban, rcv, photo_key, sid))
            else:
                db.execute(
                    "INSERT INTO profiles (speaker_id, full_name, phone, bank_name, iban, "
                    "  receiver_name, photo_b2_key, completed) VALUES (?,?,?,?,?,?,?,1)",
                    (sid, fn, phn, bnk, iban, rcv, photo_key))
            db.commit(); db.close()
            flash("Profile saved successfully.","success")
            return redirect(url_for("editor_home"))

        # Contributor flow — name/age/gender and the bank beneficiary name come from the
        # VERIFIED ID via the consent flow, NOT manual entry. The profile form only collects
        # location, language, education, phone, and bank account. We must NOT write
        # full_name/age/gender/receiver_name here, or we'd clobber the ID-sourced values.
        reg = request.form.get("region","").strip()
        dia = request.form.get("dialect","Khalkha").strip()
        nat = request.form.get("native_language","Mongolian").strip()
        edu = request.form.get("education_level","").strip()
        if not all([reg, edu]):
            flash("Бүх заавал бөглөх талбарыг бөглөнө үү.","error"); db.close()
            return redirect(url_for("profile"))
        if prof:
            db.execute("""UPDATE profiles SET region=?,dialect=?,native_language=?,
                education_level=?,phone=?,bank_name=?,iban=?,
                completed=1,updated_at=datetime('now') WHERE speaker_id=?""",
                (reg, dia, nat, edu, phn, bnk, iban, sid))
        else:
            db.execute("""INSERT INTO profiles (speaker_id,region,dialect,native_language,
                education_level,phone,bank_name,iban,completed)
                VALUES (?,?,?,?,?,?,?,?,1)""",
                (sid, reg, dia, nat, edu, phn, bnk, iban))
        db.commit(); db.close()
        flash("Профайл амжилттай хадгалагдлаа!","success")
        return redirect(url_for("contributor_home"))

    db.close()
    return render_template("profile.html", prof=prof, role=role,
                           locked=bool(prof and prof["locked"]),
                           regions=REGION_CHOICES, educations=EDUCATION_CHOICES,
                           dialects=DIALECT_CHOICES,
                           consent_title_mn=CONSENT_V2_TITLE_MN,
                           consent_title_en=CONSENT_V2_TITLE_EN,
                           consent_mn=_consent_display_blocks(_personalize_consent_paragraphs(
                               CONSENT_AGREEMENT_V2_MN,
                               (prof["full_name"] if prof else None),
                               (prof["dob"] if prof else None))),
                           consent_en=_consent_display_blocks(CONSENT_AGREEMENT_V2_EN),
                           consent_affirmation_mn=CONSENT_V2_AFFIRMATION_MN,
                           consent_version=CURRENT_CONSENT_VERSION,
                           needs_reaffirm=needs_reaffirm)

@app.route("/consent/extract", methods=["POST"])
@login_required
def consent_extract():
    """Receives the redacted ID image, runs OpenAI vision to read name + DOB,
    returns JSON for the contributor to confirm/correct. Stores nothing."""
    if session.get("role") != "contributor":
        abort(403)
    if _profile_is_locked(session.get("speaker_id")):
        return jsonify({"ok": False, "error": "locked"}), 403
    f = request.files.get("id_image")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "no_image"}), 400

    # Server-side anti-abuse: cap OCR calls per contributor per day. A client-side limit
    # alone can be bypassed (direct POST / page refresh); this is the real protection
    # against an account spamming the (paid) OpenAI vision endpoint. Resets daily.
    OCR_DAILY_CAP = 50
    sid = session.get("speaker_id")
    today = datetime.date.today().isoformat()
    _db = get_db()
    _row = _db.execute("SELECT ocr_count_day, ocr_count_n FROM profiles WHERE speaker_id=?",
                       (sid,)).fetchone()
    used = (_row["ocr_count_n"] or 0) if (_row and _row["ocr_count_day"] == today) else 0
    if used >= OCR_DAILY_CAP:
        _db.close()
        return jsonify({"ok": False, "capped": True,
                        "error": "Өдрийн оролдлогын хязгаарт хүрлээ."}), 429
    if _row:
        _db.execute("UPDATE profiles SET ocr_count_day=?, ocr_count_n=? WHERE speaker_id=?",
                    (today, used + 1, sid))
    else:
        _db.execute("INSERT INTO profiles (speaker_id, ocr_count_day, ocr_count_n) VALUES (?,?,1)",
                    (sid, today))
    _db.commit(); _db.close()

    client = get_openai_client()
    if client is None:
        # No OpenAI configured — let the user fill in manually
        return jsonify({"ok": True, "name": "", "dob": "", "sex": "",
                        "readable": True, "is_id": True, "note": "no_ocr"})

    import base64
    raw = f.read()
    if len(raw) > 15 * 1024 * 1024:
        return jsonify({"ok": False, "error": "too_large"}), 400
    b64 = base64.b64encode(raw).decode("ascii")
    mime = f.mimetype or "image/jpeg"

    prompt = (
        "You are reading a Mongolian national ID card (Иргэний үнэмлэх) for an "
        "adult-only voice dataset project. The registration number "
        "(Регистрийн дугаар / Registration number) may be blacked out — that is "
        "expected, ignore it.\n\n"
        "The card has THREE name fields printed in this order:\n"
        "  1. 'Овог / Family name' — the clan name. IGNORE THIS FIELD COMPLETELY.\n"
        "  2. 'Эцэг/эх-ийн нэр / Surname' — the patronymic. USE THIS.\n"
        "  3. 'Нэр / Given name' — the given name. USE THIS.\n\n"
        "Build the full name as: \"<Surname> <Given name>\" — i.e. the patronymic "
        "(Эцэг/эх-ийн нэр / Surname) followed by the given name (Нэр / Given name), "
        "in Cyrillic exactly as printed. Do NOT include the Family name (Овог).\n"
        "Example: if Family name=Билгүүд, Surname=Эрдэнэбаатар, Given name=Цогтбилэг, "
        "then name = \"Эрдэнэбаатар Цогтбилэг\".\n\n"
        "Also read the date of birth (Төрсөн он, сар, өдөр / Date of birth).\n"
        "Also read the sex (Хүйс / Sex): return exactly \"Male\" if it shows "
        "Эрэгтэй/Male, or \"Female\" if it shows Эмэгтэй/Female.\n"
        "Respond with STRICT JSON only, no markdown:\n"
        '{"name": "<Surname Given name in Cyrillic, or empty string>", '
        '"dob": "<YYYY-MM-DD, or empty string>", '
        '"sex": "<Male | Female | empty string>", '
        '"is_government_id": <true/false>, '
        '"face_present": <true/false>, '
        '"readable": <true/false>}'
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            max_tokens=300,
            temperature=0,
        )
        txt = (resp.choices[0].message.content or "").strip()
        # Strip code fences if the model added them
        if txt.startswith("```"):
            txt = txt.split("```")[1] if "```" in txt[3:] else txt
            txt = txt.replace("json", "", 1).strip("`\n ")
        import json as _json
        data = _json.loads(txt)
        sex = (data.get("sex") or "").strip().capitalize()
        if sex not in ("Male", "Female"):
            sex = ""
        return jsonify({
            "ok": True,
            "name": (data.get("name") or "").strip(),
            "dob": (data.get("dob") or "").strip(),
            "sex": sex,
            "is_id": bool(data.get("is_government_id", True)),
            "face_present": bool(data.get("face_present", True)),
            "readable": bool(data.get("readable", True)),
        })
    except Exception as e:
        print(f"[CONSENT OCR] error for {session.get('username')}: {e}")
        # Soft-fail: let the contributor proceed and fill fields manually
        return jsonify({"ok": True, "name": "", "dob": "", "sex": "",
                        "readable": False, "is_id": True, "note": "ocr_failed"})


@app.route("/consent/submit", methods=["POST"])
@login_required
def consent_submit():
    """Final digital-consent submission: redacted ID image + confirmed name/DOB
    + the on-screen affirmation. Builds a composite consent PDF and stores it."""
    if session.get("role") != "contributor":
        abort(403)
    sid = session["speaker_id"]

    # Locked by admin after review: the consent/ID record is frozen. Silently ignore.
    if _profile_is_locked(sid):
        return redirect(url_for("profile"))

    if request.form.get("affirmed") != "yes":
        flash("Зөвшөөрлийн нөхцөлийг зөвшөөрөх шаардлагатай.", "error")
        return redirect(url_for("profile"))

    name = (request.form.get("id_name") or "").strip()
    dob  = (request.form.get("id_dob") or "").strip()
    gender = (request.form.get("id_gender") or "").strip()
    if gender not in ("Male", "Female", "Other"):
        gender = ""
    f = request.files.get("id_image")
    if not name or not f or not f.filename:
        flash("Нэр болон иргэний үнэмлэхний зураг шаардлагатай.", "error")
        return redirect(url_for("profile"))

    # Derive age from the verified date of birth (so age is sourced from the ID too).
    age_val = None
    try:
        _d = datetime.datetime.strptime(dob[:10], "%Y-%m-%d").date()
        _t = datetime.date.today()
        age_val = _t.year - _d.year - ((_t.month, _t.day) < (_d.month, _d.day))
        if not (0 < age_val < 120):
            age_val = None
    except Exception:
        age_val = None

    # Look up any existing (original) consent date FIRST. Re-submitting consent updates
    # the document/format — it does not restart consent — so the effective date must
    # never move forward, or prior clips would wrongly appear to predate consent.
    db = get_db()
    existing = db.execute(
        "SELECT consent_uploaded_at FROM profiles WHERE speaker_id=?", (sid,)
    ).fetchone()
    db.close()
    first_consent_date = (existing["consent_uploaded_at"]
                          if existing and existing["consent_uploaded_at"] else None)

    try:
        id_bytes = f.read()
        consent_pdf = _build_consent_pdf(name, dob, id_bytes, request,
                                         first_consent_date=first_consent_date)
    except Exception as e:
        print(f"[CONSENT] PDF build error: {e}")
        flash("Зөвшөөрлийн файл боловсруулахад алдаа гарлаа. Дахин оролдоно уу.", "error")
        return redirect(url_for("profile"))

    # Versioned, timestamped key: every affirmation writes a NEW object; prior

    # consent PDFs are never overwritten (E.1).

    _ver_ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    b2_key = f"consent/{sid}/consent_v{CURRENT_CONSENT_VERSION}_{_ver_ts}.pdf"
    try:
        b2 = get_b2()
        b2.put_object(Bucket=B2_BUCKET_NAME, Key=b2_key,
                      Body=consent_pdf, ContentType="application/pdf")
    except Exception as e:
        print(f"[CONSENT] B2 error: {e}")
        flash("Файлыг хадгалахад алдаа гарлаа. Дахин оролдоно уу.", "error")
        return redirect(url_for("profile"))

    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    db = get_db()
    prof = db.execute("SELECT 1 FROM profiles WHERE speaker_id=?", (sid,)).fetchone()
    if prof:
        # The verified ID is the source of truth: its name/age/gender OVERWRITE whatever
        # was on file (clearing nicknames/typos), and the read-only bank beneficiary name
        # is set to the ID name so payment only goes to the person who recorded.
        # consent_uploaded_at is preserved so re-consent never moves the effective date.
        db.execute(
            "UPDATE profiles SET consent_b2_key=?, "
            "  consent_uploaded_at=CASE WHEN COALESCE(consent_uploaded_at,'')='' "
            "    THEN datetime('now') ELSE consent_uploaded_at END, "
            "  consent_affirmed_at=datetime('now'), consent_affirm_ip=?, "
            "  consent_method='digital', consent_version=?, "
            "  dob=CASE WHEN ?<>'' THEN ? ELSE dob END, "
            "  age=CASE WHEN ? IS NOT NULL THEN ? ELSE age END, "
            "  gender=CASE WHEN ?<>'' THEN ? ELSE gender END, "
            "  full_name=CASE WHEN ?<>'' THEN ? ELSE full_name END, "
            "  receiver_name=CASE WHEN ?<>'' THEN ? ELSE receiver_name END, "
            "  updated_at=datetime('now') WHERE speaker_id=?",
            (b2_key, ip, CURRENT_CONSENT_VERSION, dob, dob, age_val, age_val, gender, gender,
             name, name, name, name, sid))
    else:
        db.execute(
            "INSERT INTO profiles (speaker_id, consent_b2_key, consent_uploaded_at, "
            "  consent_affirmed_at, consent_affirm_ip, consent_method, consent_version, "
            "  dob, age, gender, full_name, receiver_name) "
            "VALUES (?, ?, datetime('now'), datetime('now'), ?, 'digital', ?, ?, ?, ?, ?, ?)",
            (sid, b2_key, ip, CURRENT_CONSENT_VERSION, dob, age_val, gender, name, name))
    # Append-only affirmation trail (never updated or deleted). The hash stored is the
    # live hash of the exact Mongolian TEMPLATE this build displays — identical for
    # every speaker, and equal to CURRENT_CONSENT_TEXT_SHA256 once finalized.
    db.execute(
        "INSERT INTO consent_records (speaker_id, user_id, agreement_version, text_sha256, "
        "  affirmation_wording, checkbox_affirmed, session_ref, ip, device_info, "
        "  id_verification_result, pdf_b2_key) "
        "VALUES (?,?,?,?,?,1,?,?,?,?,?)",
        (sid, session.get("user_id"), CURRENT_CONSENT_VERSION,
         consent_text_sha256(CONSENT_AGREEMENT_V2_MN),
         CONSENT_V2_AFFIRMATION_MN,
         f"uid:{session.get('user_id')}/{session.get('username','')}",
         ip, (request.user_agent.string or "")[:300],
         "passed — in-app government-ID capture + automated OCR (name/DOB/sex read from ID)",
         b2_key))
    db.commit(); db.close()
    flash("Зөвшөөрөл амжилттай баталгаажлаа. Гэрээний хуулбараа профайл хуудаснаас татаж авах боломжтой.", "success")
    return redirect(url_for("profile"))


@app.route("/consent/my_pdf")
@login_required
def consent_my_pdf():
    """The speaker downloads their own consent record PDF (E.5 / Баталгаажуулалт:
    a downloadable copy must be available to them). Stamps
    download_copy_delivered_at on their latest consent_records row — the only write
    ever made to an existing consent record."""
    if session.get("role") != "contributor":
        abort(403)
    sid = session["speaker_id"]
    db = get_db()
    p = db.execute("SELECT consent_b2_key FROM profiles WHERE speaker_id=?", (sid,)).fetchone()
    if not (p and p["consent_b2_key"]):
        db.close(); abort(404)
    db.execute(
        "UPDATE consent_records SET download_copy_delivered_at="
        "  COALESCE(download_copy_delivered_at, datetime('now')) "
        "WHERE id=(SELECT id FROM consent_records WHERE speaker_id=? ORDER BY id DESC LIMIT 1)",
        (sid,))
    db.commit(); db.close()
    try:
        b2 = get_b2()
        url = b2.generate_presigned_url(
            "get_object", Params={"Bucket": B2_BUCKET_NAME, "Key": p["consent_b2_key"]},
            ExpiresIn=600)
        return redirect(url)
    except Exception as e:
        print(f"[CONSENT] my_pdf presign failed for {sid}: {e}")
        flash("Татахад алдаа гарлаа. Дахин оролдоно уу.", "error")
        return redirect(url_for("profile"))


@app.route("/upload_consent", methods=["POST"])
@login_required
def upload_consent():
    """Disabled. The file-upload consent fallback has been removed — consent is now only
    via the in-app camera + ID verification flow (/consent/submit), which records
    consent_method='digital'. Kept as a hard block so a stale page, bookmark, or direct
    POST can't create a non-digital consent record."""
    flash("Бэлэн файл оруулах сонголтыг идэвхгүй болгосон. Камер ашиглан зөвшөөрлөө баталгаажуулна уу.", "error")
    return redirect(url_for("profile"))

@app.route("/admin/view_consent/<speaker_id>")
@login_required
@manager_required
def admin_view_consent(speaker_id):
    """Admin or owning editor downloads/views a contributor's consent PDF."""
    db = get_db()
    prof = db.execute("SELECT consent_b2_key FROM profiles WHERE speaker_id=?", (speaker_id,)).fetchone()
    db.close()
    if not prof or not prof["consent_b2_key"]:
        flash("No consent on file for this contributor.","error")
        return redirect(url_for("view_profile", speaker_id=speaker_id))
    try:
        b2 = get_b2()
        url = b2.generate_presigned_url(
            "get_object",
            Params={"Bucket": B2_BUCKET_NAME, "Key": prof["consent_b2_key"]},
            ExpiresIn=3600
        )
        return redirect(url)
    except Exception as e:
        print(f"Consent presign error: {e}")
        flash("Could not retrieve consent file.","error")
        return redirect(url_for("view_profile", speaker_id=speaker_id))

# ── Contributor ───────────────────────────────────────────────────────────────
@app.route("/home")
@login_required
def contributor_home():
    if session["role"]=="admin": return redirect(url_for("admin_dashboard"))
    sid=session["speaker_id"]; db=get_db()
    # Profile-completeness check: contributor can only actually record once
    # they've finished the profile AND uploaded the signed consent. Used to
    # decide whether the Record buttons on this page are live links or
    # bubble-style notices telling them to finish the profile first.
    prof_check = db.execute(
        "SELECT completed, consent_b2_key FROM profiles WHERE speaker_id=?",
        (sid,)).fetchone()
    can_record = bool(prof_check and prof_check["completed"] and prof_check["consent_b2_key"])
    prompts=db.execute(
        "SELECT p.*,c.status,c.reject_note,c.id as clip_id,c.duration_seconds "
        "FROM prompts p LEFT JOIN clips c ON c.prompt_id=p.id AND c.speaker_id=p.speaker_id "
        "  AND c.status != 'missing' "
        "WHERE p.speaker_id=? ORDER BY p.clip_number",(sid,)).fetchall()
    # Pull the frozen rate alongside each approved clip for accurate per-clip earnings
    me = db.execute("SELECT hourly_rate FROM users WHERE speaker_id=?", (sid,)).fetchone()
    current_rate = me["hourly_rate"] if me else RATE_PER_HOUR
    ac=db.execute(
        "SELECT duration_seconds, compensated, contributor_rate_at_approval "
        "FROM clips WHERE speaker_id=? AND status='approved'",(sid,)).fetchall()
    db.close()
    total_sec=sum(c["duration_seconds"] or 0 for c in ac)
    comp_sec=sum(c["duration_seconds"] or 0 for c in ac if c["compensated"])
    uncomp_sec=total_sec-comp_sec
    # Sum unpaid earnings by grouping clips per frozen rate, then ceil ONCE per group.
    # Avoids over-counting that would happen if we ceiled each clip individually.
    unpaid_clips = [c for c in ac if not c["compensated"]]
    uncomp_amount = calc_comp_grouped(unpaid_clips, "contributor_rate_at_approval", current_rate)
    total=len(prompts); approved=sum(1 for p in prompts if p["status"]=="approved")
    # Count both pending and editor_approved as "in review" for the contributor's view
    pending=sum(1 for p in prompts if p["status"] in ("pending","editor_approved"))
    rejected=sum(1 for p in prompts if p["status"]=="rejected")
    # Conversation sessions waiting on this speaker (created / recording / incomplete),
    # for the home-page banner. Separate short-lived connection; RS logic untouched.
    _cdb = get_db()
    _conv_sweep_incomplete(_cdb)
    _sid = session["speaker_id"]
    # Admin-paired sessions still to be recorded (speaker-made attempts are hidden: a new call
    # always starts fresh, so a leftover one is never something to "resume").
    conv_waiting = [dict(r) | {"kind": "record"} for r in _cdb.execute(
        "SELECT s.id, s.session_code, s.status, t.title_mn AS topic_title FROM conv_sessions s "
        "LEFT JOIN conv_topics t ON t.id=s.topic_id "
        "WHERE (s.speaker_a_id=? OR s.speaker_b_id=?) AND s.status IN ('created','recording','incomplete') "
        "  AND s.created_by NOT IN (s.speaker_a_id, s.speaker_b_id) "
        "ORDER BY s.id DESC", (_sid, _sid)).fetchall()]
    # a call ringing me right now
    conv_waiting += [{"id": r["id"], "session_code": r["creator_speaker_id"], "status": "дуудлага", "topic_title": None, "kind": "invite"}
                     for r in _cdb.execute(
        "SELECT id, creator_speaker_id FROM conv_invites WHERE invitee_speaker_id=? AND status='pending' AND expires_at > datetime('now') "
        "ORDER BY id DESC", (_sid,)).fetchall()]
    # (speakers no longer correct transcripts — the circle's editor does — so no 'edit' banner)
    _cdb.close()
    return render_template("contributor_home.html",
        conv_waiting=conv_waiting,
        chat_unread=chat_unread_count(),
        prompts=prompts,total=total,approved=approved,pending=pending,rejected=rejected,
        total_sec=total_sec,comp_sec=comp_sec,uncomp_sec=uncomp_sec,
        uncomp_amount=uncomp_amount, current_rate=current_rate,
        can_record=can_record,
        fmt_hours=fmt_hours,fmt_dur=fmt_dur,calc_compensation=calc_comp)

@app.route("/record/<int:prompt_id>")
@login_required
@profile_required
def record(prompt_id):
    if session["role"]=="admin": return redirect(url_for("admin_dashboard"))
    db=get_db()
    # Migration gate: pre-existing contributors who consented on paper must re-affirm via
    # the new digital flow before they can keep recording. One-time, per contributor.
    if _needs_digital_migration(db, session["speaker_id"]):
        db.close()
        flash(MIGRATION_MESSAGE_MN, "warning")
        return redirect(url_for("profile"))
    if _needs_consent_reaffirmation(db, session["speaker_id"]):
        db.close()
        flash(CONSENT_GATE_MESSAGE_MN, "warning")
        return redirect(url_for("profile"))
    if RS_CAP_ENABLED:
        used = db.execute("SELECT COALESCE(SUM(duration_seconds),0) s FROM clips WHERE speaker_id=? AND status IN ('approved','pending','editor_approved')",
                          (session["speaker_id"],)).fetchone()["s"] or 0
        if used / 60.0 >= RS_CAP_MINUTES:
            db.close()
            flash(f"[ДРАФТ] Та уншлагын бичлэгийн {RS_CAP_MINUTES/60:.0f} цагийн хязгаартаа хүрсэн байна. Баярлалаа! Одооноос ярианы бичлэгт оролцох боломжтой.", "info")
            return redirect(url_for("contributor_home"))
    prompt=db.execute("SELECT * FROM prompts WHERE id=? AND speaker_id=?",
        (prompt_id,session["speaker_id"])).fetchone()
    if not prompt: abort(404)
    existing=db.execute("SELECT * FROM clips WHERE prompt_id=? AND speaker_id=?",
        (prompt_id,session["speaker_id"])).fetchone()
    db.close()
    if existing and existing["status"] in ("pending","editor_approved","approved"):
        flash("Энэ клип аль хэдийн илгээгдсэн байна.","info")
        return redirect(url_for("contributor_home"))
    return render_template("record.html",prompt=prompt,existing=existing)

@app.route("/submit_clip", methods=["POST"])
@login_required
def submit_clip():
    if session["role"]=="admin": abort(403)
    prompt_id=request.form.get("prompt_id",type=int)
    dur_sec=request.form.get("duration_seconds",type=float) or 0
    audio=request.files.get("audio")
    if not prompt_id or not audio:
        return jsonify({"ok":False,"error":"Missing data"}),400
    db=get_db()
    # Defense-in-depth: even if someone POSTs here directly, refuse the upload until they've
    # migrated to digital consent. The page-level gate in /record is the primary UX.
    if _needs_digital_migration(db, session["speaker_id"]):
        db.close()
        return jsonify({"ok":False,"error":"consent_migration_required",
                        "message":MIGRATION_MESSAGE_MN}), 403
    if _needs_consent_reaffirmation(db, session["speaker_id"]):
        db.close()
        return jsonify({"ok":False,"error":"consent_reaffirmation_required",
                        "message":CONSENT_GATE_MESSAGE_MN}), 403
    prompt=db.execute("SELECT * FROM prompts WHERE id=? AND speaker_id=?",
        (prompt_id,session["speaker_id"])).fetchone()
    if not prompt: db.close(); return jsonify({"ok":False,"error":"Not found"}),404
    # Clear ALL stale prior clips for this prompt (rejected, or audio-lost 'missing')
    # before inserting the fresh recording — not just one — so re-recording can never
    # leave duplicate rows behind. record() already blocks re-recording while a
    # pending/editor_approved/approved clip exists, so only stale rows are removed here.
    db.execute(
        "DELETE FROM clips WHERE prompt_id=? AND speaker_id=? AND status IN ('rejected','missing')",
        (prompt_id, session["speaker_id"]))
    raw=audio.read(); filename=prompt["filename"]
    # Guard against empty / glitched recordings entering the pipeline. The ONLY validation
    # on this path. Authoritative check = the actual audio length read from the WAV itself
    # (settings-agnostic, so a genuinely short real clip is never falsely rejected). If the
    # bytes don't parse as WAV, fall back to a size floor — an empty/header-only WAV is
    # ~50-200 bytes, while any real sub-second clip is many KB.
    MIN_DURATION = 0.25   # seconds
    MIN_BYTES = 1024
    real_dur = None
    try:
        import wave as _wave
        with _wave.open(io.BytesIO(raw), "rb") as _w:
            _fr = _w.getframerate()
            real_dur = (_w.getnframes() / _fr) if _fr else 0.0
    except Exception:
        real_dur = None
    is_empty = (real_dur < MIN_DURATION) if real_dur is not None else (len(raw) < MIN_BYTES)
    if is_empty:
        db.close()
        print(f"[SUBMIT] rejected empty recording: speaker={session['speaker_id']} "
              f"prompt={prompt_id} bytes={len(raw)} wav_dur={real_dur}")
        return jsonify({"ok": False, "error": "empty_recording",
                        "message": "Бичлэг амжилтгүй боллоо — дуу хоолой илрээгүй байна. Дахин бичнэ үү."}), 400
    b2_key=f"pending/{session['speaker_id']}/{filename}"
    b2_ok=False
    try:
        b2=get_b2()
        b2.put_object(Bucket=B2_BUCKET_NAME,Key=b2_key,Body=raw,ContentType="audio/wav")
        b2_ok=True
    except Exception as e:
        print(f"B2: {e}"); b2_key=None
    db.execute("INSERT INTO clips (speaker_id,prompt_id,filename,b2_key,duration_seconds,status) VALUES (?,?,?,?,?,?)",
        (session["speaker_id"],prompt_id,filename,b2_key,dur_sec,"pending"))
    db.commit(); db.close()
    return jsonify({"ok":True,"b2_ok":b2_ok})

# ── Admin core ────────────────────────────────────────────────────────────────
@app.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    db=get_db()
    contributors=db.execute(
        "SELECT u.*,"
        "  (SELECT COUNT(*) FROM prompts WHERE speaker_id=u.speaker_id) as total_prompts,"
        "  (SELECT COUNT(*) FROM prompts p WHERE p.speaker_id=u.speaker_id "
        "    AND NOT EXISTS (SELECT 1 FROM clips c WHERE c.prompt_id=p.id AND c.status != 'rejected')) as todo_count,"
        "  (SELECT COUNT(*) FROM clips WHERE speaker_id=u.speaker_id AND status='approved') as approved,"
        "  (SELECT COUNT(*) FROM clips WHERE speaker_id=u.speaker_id AND status='pending') as pending,"
        "  (SELECT COUNT(*) FROM clips WHERE speaker_id=u.speaker_id AND status='rejected') as rejected,"
        "  (SELECT COALESCE(SUM(duration_seconds),0) FROM clips WHERE speaker_id=u.speaker_id AND status='approved') as approved_sec,"
        "  (SELECT COALESCE(SUM(duration_seconds),0) FROM clips WHERE speaker_id=u.speaker_id AND status='approved' AND compensated=0) as uncomp_sec,"
        "  (p.completed AND COALESCE(p.consent_b2_key,'')<>'') as profile_complete,"
        # Three-state consent badge: 'current' = complete + v2.0 affirmed (green),
        # 'stale' = complete but pre-v2.0, gated awaiting re-affirmation (amber),
        # 'none' = incomplete profile or no consent on file (red).
        "  CASE "
        "    WHEN (p.completed AND COALESCE(p.consent_b2_key,'')<>'' "
        f"          AND COALESCE(p.consent_version,'') = '{CURRENT_CONSENT_VERSION}') THEN 'current' "
        "    WHEN (p.completed AND COALESCE(p.consent_b2_key,'')<>'') THEN 'stale' "
        "    ELSE 'none' END as consent_state,"
        "  COALESCE(p.locked,0) as profile_locked,"
        "  (SELECT editor_user_id FROM editor_contributors WHERE contributor_speaker_id=u.speaker_id LIMIT 1) as assigned_editor_id "
        "FROM users u LEFT JOIN profiles p ON p.speaker_id=u.speaker_id "
        "WHERE u.role='contributor' AND COALESCE(u.archived,0)=0 ORDER BY u.created_at DESC"
    ).fetchall()
    # Compute each contributor's unpaid amount accurately using each clip's frozen rate
    # (or fallback to user's current hourly_rate if NULL).
    contributors = [dict(c) for c in contributors]
    for c in contributors:
        unpaid_clips = db.execute(
            "SELECT duration_seconds, contributor_rate_at_approval "
            "FROM clips WHERE speaker_id=? AND status='approved' AND compensated=0",
            (c["speaker_id"],)
        ).fetchall()
        c["uncomp_amount"] = calc_comp_grouped(
            unpaid_clips, "contributor_rate_at_approval",
            c["hourly_rate"] or RATE_PER_HOUR
        )
    editors=db.execute(
        "SELECT u.id, u.username, u.hourly_rate, "
        "  COALESCE(u.editor_penalty_pct,0) AS editor_penalty_pct, u.editor_penalty_reason, "
        "  (SELECT COUNT(*) FROM editor_contributors WHERE editor_user_id=u.id) as contributor_count "
        "FROM users u WHERE u.role='editor' AND COALESCE(u.frozen,0)=0 AND COALESCE(u.archived,0)=0 ORDER BY u.username"
    ).fetchall()
    # Compute each editor's unpaid earnings using each clip's frozen editor rate
    editors = [dict(e) for e in editors]
    for e in editors:
        unpaid_editor_clips = db.execute(
            "SELECT duration_seconds, editor_rate_at_approval "
            "FROM clips WHERE editor_user_id=? AND status='approved' AND COALESCE(editor_paid,0)=0",
            (e["id"],)
        ).fetchall()
        e["uncomp_amount_base"] = calc_comp_grouped(
            unpaid_editor_clips, "editor_rate_at_approval",
            e["hourly_rate"] or 70000
        )
        e["uncomp_amount"] = apply_editor_penalty(
            e["uncomp_amount_base"], e["editor_penalty_pct"])
        e["uncomp_sec"] = sum(uc["duration_seconds"] or 0 for uc in unpaid_editor_clips)
    # Admin's pending queue: pending clips from contributors NOT assigned to any editor
    queue_count=db.execute(
        "SELECT COUNT(*) as n FROM clips WHERE status='pending' "
        "  AND speaker_id NOT IN (SELECT contributor_speaker_id FROM editor_contributors)"
    ).fetchone()["n"]
    # Admin's final-review queue: clips editors have approved, awaiting admin's final decision
    final_count=db.execute(
        "SELECT COUNT(*) as n FROM clips WHERE status='editor_approved'"
    ).fetchone()["n"]
    # Pending job applications (new + contacted, i.e. still needing admin attention)
    applications_count=db.execute(
        "SELECT COUNT(*) as n FROM applications WHERE status IN ('new','contacted')"
    ).fetchone()["n"]

    # Last bulk "All paid" action — shown next to the button so admin can gauge the payout cycle
    last_bulk = db.execute(
        "SELECT paid_at FROM payment_transactions WHERE method='bulk_all_paid' "
        "ORDER BY paid_at DESC LIMIT 1"
    ).fetchone()
    last_all_paid = None
    if last_bulk and last_bulk["paid_at"]:
        try:
            dt = datetime.datetime.strptime(last_bulk["paid_at"], "%Y-%m-%d %H:%M:%S")
            days_ago = (datetime.datetime.utcnow() - dt).days
            if days_ago <= 0:
                rel = "today"
            elif days_ago == 1:
                rel = "1 day ago"
            else:
                rel = f"{days_ago} days ago"
            last_all_paid = f"{dt.strftime('%Y-%m-%d')} · {rel}"
        except (ValueError, TypeError):
            last_all_paid = last_bulk["paid_at"][:10]

    # Group contributors by editor (or unassigned -> 'admin' group)
    # Each group gets a stable color index so the visual grouping is consistent.
    editor_map = {e["id"]: dict(e) for e in editors}
    # Group colors: purple for the admin/unassigned group, light green for ALL editors.
    ADMIN_COLOR  = ("#f1ebfb", "#7c3aed")   # light purple bg, purple accent
    EDITOR_COLOR = ("#eafaf1", "#1a7f5a")   # light green bg, green accent

    groups = []
    # Preload generation settings for ALL admin/editor users so we can attach
    # the current "Working on: ..." config to each group header without N+1 queries.
    # Keyed by user_id. Admin's own settings live under their admin user_id.
    gen_settings_rows = db.execute(
        "SELECT editor_user_id, speech_type, subcategory, system_prompt, batch_size "
        "FROM editor_generation_settings"
    ).fetchall()
    gen_settings_map = {r["editor_user_id"]: dict(r) for r in gen_settings_rows}
    def _gs_or_none(uid):
        s = gen_settings_map.get(uid)
        # Treat "row exists but system_prompt empty" as not-configured for display purposes
        if s and not (s.get("system_prompt") or "").strip():
            return None
        return s
    admin_user_id = session.get("user_id")
    # Always show the admin/unassigned group first
    unassigned = [c for c in contributors if not c["assigned_editor_id"]]
    groups.append({
        "kind": "admin",
        "label": "Admin (unassigned)",
        "editor_id": None,
        "admin_user_id": admin_user_id,   # used for Generation settings + Унших ажил өгөх button
        "gen_settings": _gs_or_none(admin_user_id),
        "bg": ADMIN_COLOR[0],
        "accent": ADMIN_COLOR[1],
        "contributors": unassigned,
        "directive": None,
    })
    # Editor groups ordered by current unpaid balance, highest first.
    # The admin/unassigned group was appended above this loop, so it stays pinned on top
    # regardless of editor unpaid amounts.
    sorted_editors = sorted(editors, key=lambda x: -(x.get("uncomp_amount") or 0))
    for e in sorted_editors:
        ec = [c for c in contributors if c["assigned_editor_id"] == e["id"]]
        # Latest admin directive for this editor
        msg_row = db.execute(
            "SELECT message FROM editor_messages WHERE editor_user_id=? "
            "ORDER BY created_at DESC LIMIT 1", (e["id"],)
        ).fetchone()
        groups.append({
            "kind": "editor",
            "label": e["username"],
            "editor_id": e["id"],
            "hourly_rate": e["hourly_rate"] or 70000,
            "uncomp_amount": e.get("uncomp_amount", 0),
            "uncomp_amount_base": e.get("uncomp_amount_base", 0),
            "penalty_pct": e.get("editor_penalty_pct", 0) or 0,
            "penalty_reason": e.get("editor_penalty_reason") or "",
            "uncomp_sec": e.get("uncomp_sec", 0),
            "gen_settings": _gs_or_none(e["id"]),
            "bg": EDITOR_COLOR[0],
            "accent": EDITOR_COLOR[1],
            "contributors": ec,
            "directive": msg_row["message"] if msg_row else None,
        })

    # Conversations: awaiting admin approval, waiting for admin's own editing (unassigned
    # circle), and unpaid conversation pay per contributor row and per editor group.
    conv_review_count = db.execute("SELECT COUNT(*) n FROM conv_sessions WHERE status='in_review'").fetchone()["n"]
    conv_edit_count = db.execute("SELECT COUNT(*) n FROM conv_sessions WHERE editor_id IS NULL AND status IN ('drafted','rejected','in_edit')").fetchone()["n"]
    for c in contributors:
        rows = _conv_pay_rows_speaker(db, c["speaker_id"])
        c["conv_uncomp_amount"] = _conv_pay_amount(rows, c["hourly_rate"] or RATE_PER_HOUR)
        c["conv_uncomp_sec"] = sum(r["duration_seconds"] for r in rows)
    for g in groups:
        if g["kind"] == "editor":
            rows = _conv_pay_rows_editor(db, g["editor_id"])
            g["conv_uncomp_amount"] = apply_editor_penalty(_conv_pay_amount(rows, g["hourly_rate"]), g["penalty_pct"])
            g["conv_uncomp_sec"] = sum(r["duration_seconds"] for r in rows)
    # Pending contributor requests count (for header badge)
    pending_requests = db.execute(
        "SELECT COUNT(*) as n FROM contributor_requests WHERE status='pending'"
    ).fetchone()["n"]
    db.close()

    _db_pp = get_db(); _payroll_pending_dash = _payroll_pending(_db_pp); _db_pp.close()
    return render_template("admin_dashboard.html",
        conv_review_count=conv_review_count, conv_edit_count=conv_edit_count,
        payroll_pending=_payroll_pending_dash,
        chat_unread=chat_unread_count(),
        contributors=contributors,
        editors=editors,
        groups=groups,
        gen_settings_for_template=gen_settings_map,
        queue_count=queue_count,
        final_count=final_count,
        applications_count=applications_count,
        last_all_paid=last_all_paid,
        pending_requests=pending_requests,
        fmt_hours=fmt_hours, calc_comp=calc_comp)

@app.route("/admin/assign_contributor", methods=["POST"])
@login_required
@admin_required
def assign_contributor():
    """Assign a contributor to an editor, or move to admin (unassigned)."""
    speaker_id = request.form.get("speaker_id","").strip()
    editor_id_raw = request.form.get("editor_id","").strip()
    if not speaker_id:
        flash("Missing contributor.","error")
        return redirect(url_for("admin_dashboard"))
    db = get_db()
    contributor = db.execute(
        "SELECT id FROM users WHERE speaker_id=? AND role='contributor'",
        (speaker_id,)
    ).fetchone()
    if not contributor:
        db.close(); flash("Contributor not found.","error")
        return redirect(url_for("admin_dashboard"))
    # Always remove any existing assignment first
    db.execute(
        "DELETE FROM editor_contributors WHERE contributor_speaker_id=?",
        (speaker_id,)
    )
    if editor_id_raw and editor_id_raw != "0":
        try: editor_id = int(editor_id_raw)
        except ValueError:
            db.close(); flash("Invalid editor.","error")
            return redirect(url_for("admin_dashboard"))
        editor = db.execute(
            "SELECT id, username FROM users WHERE id=? AND role='editor' AND COALESCE(archived,0)=0",
            (editor_id,)
        ).fetchone()
        if not editor:
            db.close(); flash("Editor not found.","error")
            return redirect(url_for("admin_dashboard"))
        db.execute(
            "INSERT INTO editor_contributors (editor_user_id,contributor_speaker_id,assigned_by) VALUES (?,?,?)",
            (editor_id, speaker_id, "admin")
        )
        flash(f"Assigned {speaker_id} to {editor['username']}.","success")
    else:
        flash(f"Moved {speaker_id} to Admin (unassigned).","success")
    db.commit(); db.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/send_message/<int:editor_id>", methods=["POST"])
@login_required
@admin_required
def send_message(editor_id):
    """Admin sends a directive/message to an editor.
    The newest message replaces visibility of older ones on the editor dashboard."""
    db = get_db()
    editor = db.execute("SELECT id,username FROM users WHERE id=? AND role='editor'", (editor_id,)).fetchone()
    if not editor:
        db.close(); flash("Editor not found.","error")
        return redirect(url_for("admin_dashboard"))
    msg = request.form.get("message","").strip()[:2000]
    if not msg:
        db.close(); flash("Message cannot be empty.","error")
        return redirect(url_for("admin_dashboard"))
    db.execute(
        "INSERT INTO editor_messages (editor_user_id,message) VALUES (?,?)",
        (editor_id, msg)
    )
    db.commit(); db.close()
    flash(f"Message sent to {editor['username']}.","success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete_message/<int:editor_id>", methods=["POST"])
@login_required
@admin_required
def delete_message(editor_id):
    """Delete ALL directive messages admin has sent to this editor.
    The dashboard only shows the latest, so deleting all is equivalent to clearing
    what's currently displayed. Simple model: there's either a current directive or there isn't."""
    db = get_db()
    editor = db.execute("SELECT username FROM users WHERE id=? AND role='editor'", (editor_id,)).fetchone()
    if not editor:
        db.close(); abort(404)
    db.execute("DELETE FROM editor_messages WHERE editor_user_id=?", (editor_id,))
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) deleted message(s) for editor "
          f"{editor['username']} (id={editor_id}) at {datetime.datetime.now().isoformat()}")
    flash(f"Message to {editor['username']} deleted.","success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/requests")
@login_required
@admin_required
def list_requests():
    """Show pending editor requests for additional contributor slots."""
    db = get_db()
    requests = db.execute(
        "SELECT cr.*, u.username AS editor_username, "
        "  (SELECT COUNT(*) FROM editor_contributors WHERE editor_user_id=cr.editor_user_id) as current_count "
        "FROM contributor_requests cr "
        "JOIN users u ON u.id = cr.editor_user_id "
        "ORDER BY CASE cr.status WHEN 'pending' THEN 0 ELSE 1 END, cr.created_at DESC"
    ).fetchall()
    pending_requests_count = sum(1 for r in requests if r["status"] == "pending")
    db.close()
    return render_template("list_requests.html", requests=requests,
                           pending_requests_count=pending_requests_count)

@app.route("/admin/decide_request/<int:request_id>", methods=["POST"])
@login_required
@admin_required
def decide_request(request_id):
    decision = request.form.get("decision","")
    if decision not in ("approved","denied"): abort(400)
    db = get_db()
    req = db.execute("SELECT * FROM contributor_requests WHERE id=?", (request_id,)).fetchone()
    if not req or req["status"] != "pending":
        db.close(); flash("Request not found or already decided.","error")
        return redirect(url_for("list_requests"))
    db.execute(
        "UPDATE contributor_requests SET status=?, decided_at=datetime('now') WHERE id=?",
        (decision, request_id)
    )
    db.commit(); db.close()
    flash(f"Request {decision}.","success")
    return redirect(url_for("list_requests"))

@app.route("/admin/new_contributor", methods=["GET","POST"])
@login_required
@admin_required
def new_contributor():
    db = get_db()
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","").strip()
        # Admin no longer specifies speaker_id — system assigns the next MNKH_SPK##
        # sequentially. Retry on race in case two creations submit at the same instant.
        if not u or not p:
            db.close()
            flash("Username and password are required.","error")
            return redirect(url_for("new_contributor"))
        for attempt in range(5):
            s = next_speaker_id(db)
            try:
                db.execute(
                    "INSERT INTO users (username,password,speaker_id,role,hourly_rate) VALUES (?,?,?,?,?)",
                    (u, generate_password_hash(p), s, "contributor", 35000)
                )
                db.commit()
                db.close()
                flash(f"Contributor '{u}' created ({s}).","success")
                return redirect(url_for("admin_dashboard"))
            except sqlite3.IntegrityError as e:
                msg = str(e).lower()
                if "username" in msg or "users.username" in msg:
                    db.close()
                    flash("That username already exists. Please choose a different one.","error")
                    return redirect(url_for("new_contributor"))
                # Speaker ID race — rollback and retry
                db.rollback()
        db.close()
        flash("Could not assign a speaker ID after several attempts. Please try again.","error")
        return redirect(url_for("new_contributor"))
    # GET: predict the next ID for display
    predicted_id = next_speaker_id(db)
    db.close()
    return render_template("new_contributor.html", predicted_speaker_id=predicted_id)

# ── Editor management (Stage 1) ───────────────────────────────────────────────
def editor_required(f):
    @wraps(f)
    def d(*a,**k):
        if session.get("role") != "editor": abort(403)
        return f(*a,**k)
    return d

@app.route("/admin/editors")
@login_required
@admin_required
def list_editors():
    db = get_db()
    editors = db.execute(
        "SELECT u.*, "
        "  (SELECT COUNT(*) FROM editor_contributors WHERE editor_user_id=u.id) AS contributor_count "
        "FROM users u WHERE u.role='editor' AND COALESCE(u.archived,0)=0 ORDER BY u.created_at DESC"
    ).fetchall()
    pending_requests_count = db.execute(
        "SELECT COUNT(*) AS n FROM contributor_requests WHERE status='pending'"
    ).fetchone()["n"]
    db.close()
    return render_template("list_editors.html", editors=editors,
                           pending_requests_count=pending_requests_count)

@app.route("/admin/new_editor", methods=["GET","POST"])
@login_required
@admin_required
def new_editor():
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","").strip()
        if not u or not p:
            flash("Username and password are required.","error")
            return redirect(url_for("new_editor"))
        if len(p) < 6:
            flash("Password must be at least 6 characters.","error")
            return redirect(url_for("new_editor"))
        # Editors don't have a speaker_id like contributors do — use a synthetic identifier
        # so the UNIQUE constraint on users.speaker_id doesn't conflict.
        synthetic_id = f"EDITOR_{u.upper()}"
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username,password,speaker_id,role,hourly_rate) VALUES (?,?,?,?,?)",
                (u, generate_password_hash(p), synthetic_id, "editor", 70000)
            )
            db.commit()
            flash(f"Editor '{u}' created.","success")
        except sqlite3.IntegrityError:
            flash("Username already exists.","error")
        finally:
            db.close()
        return redirect(url_for("list_editors"))
    return render_template("new_editor.html")

@app.route("/admin/freeze/<int:user_id>", methods=["POST"])
@login_required
@admin_or_login_required
@profile_required
def freeze_contributor(user_id):
    db=get_db()
    user=db.execute("SELECT * FROM users WHERE id=? AND role='contributor'",(user_id,)).fetchone()
    if not user: db.close(); abort(404)
    # Editors can only freeze contributors they own
    if not can_manage_speaker(user["speaker_id"]):
        db.close(); abort(403)
    new_state=0 if user["frozen"] else 1
    db.execute("UPDATE users SET frozen=? WHERE id=?",(new_state,user_id))
    db.commit(); db.close()
    action="unfrozen" if new_state==0 else "frozen"
    flash(f"Account {action}.","success")
    return redirect(_post_action_redirect())

@app.route("/manager/reset_contributor_password/<int:user_id>", methods=["POST"])
@login_required
@admin_or_login_required
@profile_required
def reset_contributor_password(user_id):
    """Admin OR an editor who owns this contributor can override their password.
    Editors can ONLY reset passwords for their own roster — never another editor's
    contributors, and never editors or admins."""
    new_pw = request.form.get("new_password","").strip()
    if len(new_pw) < 6:
        flash("Password must be at least 6 characters.","error")
        return redirect(_post_action_redirect())
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id=? AND role='contributor'",
        (user_id,)
    ).fetchone()
    if not user:
        db.close(); abort(404)
    # Permission check — only admin or this contributor's editor
    if not can_manage_speaker(user["speaker_id"]):
        db.close(); abort(403)
    db.execute(
        "UPDATE users SET password=? WHERE id=?",
        (generate_password_hash(new_pw), user_id)
    )
    db.commit(); db.close()
    # Audit log: who reset whose password (visible in Railway deploy logs)
    print(f"[AUDIT] {session.get('username')} ({session.get('role')}) reset password for contributor "
          f"{user['username']} (speaker_id={user['speaker_id']}) at {datetime.datetime.now().isoformat()}")
    flash(f"Password updated for {user['username']}.","success")
    return redirect(_post_action_redirect())

@app.route("/admin/reset_editor_password/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def reset_editor_password(user_id):
    """Admin-only: reset an editor's password. Editors cannot reset other
    editors' passwords."""
    new_pw = request.form.get("new_password","").strip()
    if len(new_pw) < 6:
        flash("Password must be at least 6 characters.","error")
        return redirect(url_for("list_editors"))
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id=? AND role='editor'",
        (user_id,)
    ).fetchone()
    if not user:
        db.close(); abort(404)
    db.execute(
        "UPDATE users SET password=? WHERE id=?",
        (generate_password_hash(new_pw), user_id)
    )
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) reset password for editor "
          f"{user['username']} (id={user_id}) at {datetime.datetime.now().isoformat()}")
    flash(f"Password updated for editor {user['username']}.","success")
    return redirect(url_for("list_editors"))

# ─────────────────────────────────────────────────────────────────────────────
# Archive system
#
# Archiving is the standard way to retire a worker. Unlike delete, archiving:
#   • Preserves the user row and all related data forever (clips, profiles, payroll
#     history). This is essential because the dataset's provenance includes the
#     identity of who recorded and reviewed each clip.
#   • Locks the user out of login (login.html shows a specific error).
#   • Removes the user from active dashboards, lists, and payroll exports.
#   • Surfaces the user only on the dedicated "Archived workers" page (/admin/archived).
#   • Is reversible — admin can unarchive a worker if they return.
#
# Distinction from `frozen`: frozen is a temporary "pause" — visible in dashboard,
# still in payroll, can be unfrozen. Archive is a permanent retirement.
#
# Editors cannot archive contributors directly; they "Remove" (unassign) them
# back to the admin pool, and admin makes the archive decision. This is a
# deliberate audit-trail choice.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/archive_contributor/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def archive_contributor(user_id):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id=? AND role='contributor'",
        (user_id,)
    ).fetchone()
    if not user:
        db.close(); abort(404)
    if user["archived"]:
        db.close()
        flash(f"{user['username']} is already archived.","error")
        return redirect(url_for("admin_dashboard"))
    # Unassign from any editor when archiving (don't leave dangling assignments)
    db.execute(
        "DELETE FROM editor_contributors WHERE contributor_speaker_id=?",
        (user["speaker_id"],)
    )
    db.execute(
        "UPDATE users SET archived=1, archived_at=datetime('now') WHERE id=?",
        (user_id,)
    )
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) archived contributor "
          f"{user['username']} (speaker_id={user['speaker_id']}, id={user_id}) "
          f"at {datetime.datetime.now().isoformat()}")
    flash(f"{user['username']} has been archived. They can be restored from the Archived Workers page.","success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/archive_editor/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def archive_editor(user_id):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id=? AND role='editor'",
        (user_id,)
    ).fetchone()
    if not user:
        db.close(); abort(404)
    if user["archived"]:
        db.close()
        flash(f"{user['username']} is already archived.","error")
        return redirect(url_for("list_editors"))
    # Block archive if the editor still has contributors assigned. The admin
    # must unassign them first so each contributor's reassignment is a deliberate decision.
    contributor_count = db.execute(
        "SELECT COUNT(*) AS n FROM editor_contributors WHERE editor_user_id=?",
        (user_id,)
    ).fetchone()["n"]
    if contributor_count > 0:
        db.close()
        flash(
            f"Cannot archive {user['username']}: they still have {contributor_count} "
            f"contributor(s) assigned. Reassign or remove those contributors first.",
            "error"
        )
        return redirect(url_for("list_editors"))
    db.execute(
        "UPDATE users SET archived=1, archived_at=datetime('now') WHERE id=?",
        (user_id,)
    )
    # Clean up directives and pending requests so they don't sit forever
    db.execute("DELETE FROM editor_messages WHERE editor_user_id=?", (user_id,))
    db.execute(
        "UPDATE contributor_requests SET status='archived' WHERE editor_user_id=? AND status='pending'",
        (user_id,)
    )
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) archived editor "
          f"{user['username']} (id={user_id}) at {datetime.datetime.now().isoformat()}")
    flash(f"Editor {user['username']} has been archived.","success")
    return redirect(url_for("list_editors"))

@app.route("/admin/unarchive/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def unarchive_user(user_id):
    """Restore an archived contributor or editor back to active status."""
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id=? AND role IN ('contributor','editor')",
        (user_id,)
    ).fetchone()
    if not user:
        db.close(); abort(404)
    if not user["archived"]:
        db.close()
        flash(f"{user['username']} is not archived.","error")
        return redirect(url_for("list_archived"))
    db.execute("UPDATE users SET archived=0, archived_at=NULL WHERE id=?", (user_id,))
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) UNARCHIVED {user['role']} "
          f"{user['username']} (id={user_id}) at {datetime.datetime.now().isoformat()}")
    flash(f"{user['username']} has been restored to active status.","success")
    return redirect(url_for("list_archived"))

@app.route("/admin/archived")
@login_required
@admin_required
def list_archived():
    """Show all archived contributors and editors with their last-active info
    and any unpaid balance still owed."""
    db = get_db()
    contributors = db.execute(
        "SELECT u.id, u.username, u.speaker_id, u.hourly_rate, u.created_at, u.archived_at, "
        "  p.full_name, p.phone, p.bank_name, p.iban, "
        "  (SELECT COUNT(*) FROM clips WHERE speaker_id=u.speaker_id AND status='approved') AS approved_clips, "
        "  (SELECT COALESCE(SUM(duration_seconds),0) FROM clips WHERE speaker_id=u.speaker_id AND status='approved') AS approved_sec, "
        "  (SELECT COALESCE(SUM(duration_seconds),0) FROM clips WHERE speaker_id=u.speaker_id AND status='approved' AND compensated=0) AS uncomp_sec "
        "FROM users u "
        "LEFT JOIN profiles p ON p.speaker_id=u.speaker_id "
        "WHERE u.role='contributor' AND COALESCE(u.archived,0)=1 "
        "ORDER BY u.archived_at DESC NULLS LAST"
    ).fetchall()
    contributors = [dict(c) for c in contributors]
    for c in contributors:
        unpaid_clips = db.execute(
            "SELECT duration_seconds, contributor_rate_at_approval "
            "FROM clips WHERE speaker_id=? AND status='approved' AND compensated=0",
            (c["speaker_id"],)
        ).fetchall()
        c["uncomp_amount"] = calc_comp_grouped(
            unpaid_clips, "contributor_rate_at_approval",
            c["hourly_rate"] or RATE_PER_HOUR
        )
    editors = db.execute(
        "SELECT u.id, u.username, u.hourly_rate, u.created_at, u.archived_at, "
        "  COALESCE(u.editor_penalty_pct,0) AS editor_penalty_pct, "
        "  p.full_name, p.phone, p.bank_name, p.iban, "
        "  (SELECT COUNT(*) FROM clips WHERE editor_user_id=u.id AND status='approved') AS reviewed_clips, "
        "  (SELECT COALESCE(SUM(duration_seconds),0) FROM clips WHERE editor_user_id=u.id AND status='approved' AND COALESCE(editor_paid,0)=0) AS uncomp_sec "
        "FROM users u "
        "LEFT JOIN profiles p ON p.speaker_id=u.speaker_id "
        "WHERE u.role='editor' AND COALESCE(u.archived,0)=1 "
        "ORDER BY u.archived_at DESC NULLS LAST"
    ).fetchall()
    editors = [dict(e) for e in editors]
    for e in editors:
        unpaid_editor_clips = db.execute(
            "SELECT duration_seconds, editor_rate_at_approval "
            "FROM clips WHERE editor_user_id=? AND status='approved' AND COALESCE(editor_paid,0)=0",
            (e["id"],)
        ).fetchall()
        e["uncomp_amount"] = apply_editor_penalty(calc_comp_grouped(
            unpaid_editor_clips, "editor_rate_at_approval",
            e["hourly_rate"] or 70000
        ), e["editor_penalty_pct"])
    pending_requests_count = db.execute(
        "SELECT COUNT(*) AS n FROM contributor_requests WHERE status='pending'"
    ).fetchone()["n"]
    db.close()
    return render_template(
        "archived.html",
        contributors=contributors, editors=editors, fmt_hours=fmt_hours,
        pending_requests_count=pending_requests_count
    )

@app.route("/admin/delete_contributor/<speaker_id>", methods=["POST"])
@login_required
@admin_required
def delete_contributor(speaker_id):
    """Permanently and irreversibly erase a contributor: account, profile,
    prompts, clips, chat, payout history, and every associated object in B2
    storage. Intended for clearing out test accounts. Guarded so it cannot be
    run on a contributor whose clips were already delivered to a buyer."""
    sid = speaker_id
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE speaker_id=? AND role='contributor'", (sid,)).fetchone()
    if not user:
        db.close(); abort(404)
    user_id = user["id"]

    # Safety 1: admin must retype the exact speaker_id (also enforced client-side).
    if (request.form.get("confirm_speaker_id") or "").strip() != sid:
        db.close()
        flash(f"Deletion aborted: confirmation text did not match {sid}.", "error")
        return redirect(url_for("view_profile", speaker_id=sid))

    # Safety 2: refuse if any of this contributor's clips were sent to a buyer.
    # A delivered clip is part of a corpus the buyer already received; deleting it
    # here would corrupt delivery history and cannot un-sell it.
    delivered = db.execute(
        "SELECT DISTINCT d.name FROM delivery_clips dc "
        "JOIN clips c ON c.id = dc.clip_id "
        "JOIN deliveries d ON d.id = dc.delivery_id "
        "WHERE c.speaker_id=?", (sid,)
    ).fetchall()
    if delivered:
        names = ", ".join(r["name"] for r in delivered)
        db.close()
        flash(f"Cannot delete {sid}: their clips are part of delivery batch(es): {names}. "
              f"Delete or rebuild those batches first.", "error")
        return redirect(url_for("view_profile", speaker_id=sid))

    # Gather every B2 key tied to this contributor BEFORE removing the DB rows.
    prof = db.execute("SELECT consent_b2_key, photo_b2_key FROM profiles WHERE speaker_id=?", (sid,)).fetchone()
    clip_keys = [r["b2_key"] for r in db.execute(
        "SELECT b2_key FROM clips WHERE speaker_id=? AND b2_key IS NOT NULL", (sid,)).fetchall()]
    keys = set(clip_keys)
    if prof:
        if prof["consent_b2_key"]: keys.add(prof["consent_b2_key"])
        if prof["photo_b2_key"]:   keys.add(prof["photo_b2_key"])

    # Remove all DB rows (children first; FKs aren't enforced, so order is for clarity).
    db.execute("DELETE FROM delivery_clips WHERE clip_id IN (SELECT id FROM clips WHERE speaker_id=?)", (sid,))
    db.execute("DELETE FROM clips WHERE speaker_id=?", (sid,))
    db.execute("DELETE FROM prompts WHERE speaker_id=?", (sid,))
    db.execute("DELETE FROM chat_messages WHERE contributor_speaker_id=?", (sid,))
    db.execute("DELETE FROM editor_contributors WHERE contributor_speaker_id=?", (sid,))
    db.execute("DELETE FROM payment_transactions WHERE user_id=?", (user_id,))
    db.execute("DELETE FROM profiles WHERE speaker_id=?", (sid,))
    db.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.commit(); db.close()

    # Purge B2 (after the DB delete; best-effort and idempotent — a 404 just means
    # the object is already gone). A prefix sweep catches orphaned objects left by
    # earlier copy-before-delete moves. Note: the bucket's versioning lifecycle may
    # retain hidden prior versions ~100 days; the contributor disappears from every
    # queue, count, and download immediately regardless.
    b2_deleted = b2_failed = 0
    try:
        b2 = get_b2()
        try:
            for prefix in (f"pending/{sid}/", f"editor_approved/{sid}/",
                           f"approved/{sid}/", f"consent/{sid}/"):
                token = None
                while True:
                    kw = {"Bucket": B2_BUCKET_NAME, "Prefix": prefix}
                    if token: kw["ContinuationToken"] = token
                    resp = b2.list_objects_v2(**kw)
                    for o in resp.get("Contents", []):
                        keys.add(o["Key"])
                    if resp.get("IsTruncated"):
                        token = resp.get("NextContinuationToken")
                    else:
                        break
        except Exception as e:
            print(f"[DELETE_CONTRIBUTOR] prefix sweep failed for {sid}: {e}")
        for k in keys:
            try:
                b2.delete_object(Bucket=B2_BUCKET_NAME, Key=k); b2_deleted += 1
            except Exception as e:
                b2_failed += 1
                print(f"[DELETE_CONTRIBUTOR] B2 delete failed key={k}: {e}")
    except Exception as e:
        b2_failed = len(keys)
        print(f"[DELETE_CONTRIBUTOR] B2 client error for {sid}: {e}")

    msg = (f"Contributor {sid} permanently deleted. "
           f"{len(clip_keys)} clip record(s) removed; {b2_deleted} storage object(s) purged.")
    cat = "success"
    if b2_failed:
        msg += f" {b2_failed} object(s) could not be purged from storage — see server logs."
        cat = "warning"
    flash(msg, cat)
    return redirect(url_for("admin_dashboard"))

@app.route("/editor/unassign_contributor/<speaker_id>", methods=["POST"])
@login_required
@profile_required
def editor_unassign_contributor(speaker_id):
    """Editor removes a contributor from their roster.
    The contributor account remains intact and moves to the admin (unassigned) group."""
    if session.get("role") != "editor": abort(403)
    if not can_manage_speaker(speaker_id): abort(403)
    db = get_db()
    db.execute(
        "DELETE FROM editor_contributors WHERE editor_user_id=? AND contributor_speaker_id=?",
        (session["user_id"], speaker_id)
    )
    db.commit(); db.close()
    flash(f"Removed {speaker_id} from your roster. They are now unassigned.","success")
    return redirect(url_for("editor_home"))

@app.route("/admin/add_prompts/<speaker_id>", methods=["GET","POST"])
@login_required
@manager_required
@profile_required
def add_prompts(speaker_id):
    is_editor = session.get("role") == "editor"

    if request.method=="POST":
        lines=request.form.get("prompts","").strip().split("\n")
        db=get_db()
        cur_max=db.execute("SELECT COALESCE(MAX(clip_number),0) FROM prompts WHERE speaker_id=?",
            (speaker_id,)).fetchone()[0]

        # For editors: read speech_type/subcategory from THEIR generation settings (admin-controlled),
        # not from form data. Editors don't choose; admin pre-configures per editor.
        if is_editor:
            settings = db.execute(
                "SELECT speech_type, subcategory FROM editor_generation_settings WHERE editor_user_id=?",
                (session.get("user_id"),)
            ).fetchone()
            if settings:
                stype = settings["speech_type"] or "RS"
                subcat = (settings["subcategory"] or "").strip()
            else:
                stype = "RS"
                subcat = ""
        else:
            # Admin path. If the form includes speech_type (manual paste), use form values.
            # Otherwise (LLM draft save flow for unassigned contributor), fall back to
            # the admin's own stored generation settings — same model as editor.
            form_stype = request.form.get("speech_type")
            if form_stype is not None:
                stype = form_stype or "RS"
                subcat = request.form.get("subcategory","").strip()
            else:
                admin_settings = db.execute(
                    "SELECT speech_type, subcategory FROM editor_generation_settings WHERE editor_user_id=?",
                    (session.get("user_id"),)
                ).fetchone()
                if admin_settings:
                    stype = admin_settings["speech_type"] or "RS"
                    subcat = (admin_settings["subcategory"] or "").strip()
                else:
                    stype = "RS"
                    subcat = ""

        # Cross-speaker dedup
        existing_normalized = set()
        for row in db.execute("SELECT text_normalized FROM prompts WHERE text_normalized IS NOT NULL AND text_normalized != ''").fetchall():
            existing_normalized.add(row["text_normalized"])

        added = 0
        skipped_db = 0
        skipped_batch = 0
        batch_normalized = set()

        for line in lines:
            parts=line.strip().split("|"); mn=parts[0].strip()
            en=parts[1].strip() if len(parts)>1 else ""
            if not mn: continue

            normalized = normalize_prompt_text(mn)
            if not normalized:
                continue
            if normalized in existing_normalized:
                skipped_db += 1
                continue
            if normalized in batch_normalized:
                skipped_batch += 1
                continue

            cur_max += 1
            spk_code = speaker_id.replace("MNKH_","") if speaker_id.startswith("MNKH_") else speaker_id
            filename = f"MNKH_{spk_code}_{stype}_{cur_max:04d}.wav"
            db.execute(
                "INSERT OR IGNORE INTO prompts (speaker_id,clip_number,filename,text_mn,text_en,speech_type,subcategory,text_normalized) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (speaker_id,cur_max,filename,mn,en,stype,subcat,normalized)
            )
            batch_normalized.add(normalized)
            existing_normalized.add(normalized)
            added += 1

        db.commit(); db.close()

        # Editors get a minimal success flash — dedup counts are hidden from them per project decision.
        # Admins still get the full breakdown so they can spot quality issues during manual paste.
        if is_editor:
            flash(f"{added} текст хадгалагдлаа.", "success" if added > 0 else "error")
        else:
            msg_parts = [f"Added {added} prompt(s) for {speaker_id}"]
            if skipped_db > 0:
                msg_parts.append(f"{skipped_db} skipped (already in dataset)")
            if skipped_batch > 0:
                msg_parts.append(f"{skipped_batch} skipped (duplicate within paste)")
            flash(". ".join(msg_parts) + ".", "success" if added > 0 else "error")

        if (skipped_db + skipped_batch) > 0:
            print(f"[DEDUP] {session.get('username')} adding prompts for {speaker_id}: "
                  f"added={added} skipped_db={skipped_db} skipped_batch={skipped_batch}")

        return redirect(url_for("add_prompts",speaker_id=speaker_id))

    db=get_db()
    # Load non-approved prompts only
    existing=db.execute(
        "SELECT p.*,c.status FROM prompts p "
        "LEFT JOIN clips c ON c.prompt_id=p.id AND c.speaker_id=p.speaker_id "
        "WHERE p.speaker_id=? AND (c.status IS NULL OR c.status != 'approved') "
        "ORDER BY p.clip_number",(speaker_id,)).fetchall()

    # editor_settings: shown to editors as their generation config.
    # admin_can_generate: True when ADMIN is viewing an UNASSIGNED contributor AND admin's
    # own generation settings are configured. Drives whether the LLM button appears for admin.
    editor_settings = None
    admin_can_generate = False
    if is_editor:
        row = db.execute(
            "SELECT speech_type, subcategory, system_prompt, batch_size "
            "FROM editor_generation_settings WHERE editor_user_id=?",
            (session.get("user_id"),)
        ).fetchone()
        editor_settings = dict(row) if row else None
        if editor_settings and not (editor_settings.get("system_prompt") or "").strip():
            editor_settings = None
    else:
        # Admin path. Check if THIS contributor is unassigned — admins only generate
        # for unassigned contributors (assigned ones belong to their editor's flow).
        assigned = db.execute(
            "SELECT editor_user_id FROM editor_contributors WHERE contributor_speaker_id=? LIMIT 1",
            (speaker_id,)
        ).fetchone()
        if not assigned:
            admin_row = db.execute(
                "SELECT system_prompt FROM editor_generation_settings WHERE editor_user_id=?",
                (session.get("user_id"),)
            ).fetchone()
            if admin_row and (admin_row["system_prompt"] or "").strip():
                admin_can_generate = True
    db.close()
    return render_template("add_prompts.html",speaker_id=speaker_id,existing=existing,
                           editor_settings=editor_settings,
                           admin_can_generate=admin_can_generate)

@app.route("/admin/set_generation_settings/<int:editor_id>", methods=["POST"])
@login_required
@admin_required
def set_generation_settings(editor_id):
    """Admin sets the LLM prompt generation config for a specific editor OR for
    themselves (used when generating prompts for unassigned contributors).
    All four fields (speech_type, subcategory, system_prompt, batch_size) are stored
    together and applied to every 'Унших текст үүсгэх' click by that user."""
    speech_type   = request.form.get("speech_type","RS").strip().upper()
    subcategory   = request.form.get("subcategory","").strip()
    system_prompt = request.form.get("system_prompt","").strip()
    # The "next" param tells us which page to redirect to. Defaults to admin_dashboard
    # since that's where Generation settings now lives per the latest UI move.
    next_url = request.form.get("next") or url_for("admin_dashboard")
    try:
        batch_size = int(request.form.get("batch_size","50"))
    except ValueError:
        batch_size = 50
    if batch_size < 1: batch_size = 1
    if batch_size > 200: batch_size = 200

    if speech_type not in ("RS","CS","CQ","ND","EE"):
        flash("Invalid speech type.","error")
        return redirect(next_url)

    db = get_db()
    # Accept either an editor OR the calling admin themself
    user = db.execute(
        "SELECT username, role FROM users WHERE id=? AND role IN ('editor','admin')",
        (editor_id,)
    ).fetchone()
    if not user:
        db.close(); abort(404)
    # If targeting an admin user, that admin must be the one making the request.
    # No admin-edits-another-admin's-settings — keeps audit clean.
    if user["role"] == "admin" and editor_id != session.get("user_id"):
        db.close()
        flash("You can only edit your own generation settings.","error")
        return redirect(next_url)

    db.execute(
        "INSERT INTO editor_generation_settings (editor_user_id, speech_type, subcategory, system_prompt, batch_size, updated_at) "
        "VALUES (?,?,?,?,?,datetime('now')) "
        "ON CONFLICT(editor_user_id) DO UPDATE SET "
        "  speech_type=excluded.speech_type, "
        "  subcategory=excluded.subcategory, "
        "  system_prompt=excluded.system_prompt, "
        "  batch_size=excluded.batch_size, "
        "  updated_at=datetime('now')",
        (editor_id, speech_type, subcategory, system_prompt, batch_size)
    )
    db.commit(); db.close()
    target_label = "your own" if user["role"] == "admin" else f"{user['username']}'s"
    print(f"[GEN] admin updated {target_label} generation settings "
          f"(id={editor_id}, type={speech_type}, sub={subcategory!r}, batch={batch_size}) "
          f"at {datetime.datetime.now().isoformat()}")
    flash(f"Generation settings saved for {user['username']}.","success")
    return redirect(next_url)

@app.route("/admin/generate_prompts/<speaker_id>", methods=["POST"])
@login_required
@manager_required
@profile_required
def generate_prompts(speaker_id):
    """Generate a batch of prompts via OpenAI for the current editor's contributor.

    Reads the user's stored generation settings, calls the LLM, and returns
    a JSON array of {mn, en} objects. The actual saving happens via the
    regular POST to /admin/add_prompts/<speaker_id> with the chosen prompts.

    Allowed for: editors (for their assigned contributors), admins (for contributors
    not assigned to any editor — admins act as the editor for unassigned contributors).
    """
    role = session.get("role")
    user_id = session.get("user_id")
    if role not in ("editor", "admin"):
        return jsonify({"error": "Unauthorized."}), 403

    db = get_db()
    # If admin: confirm the contributor is unassigned. Admins must not bypass
    # an existing editor's queue by generating prompts on their behalf.
    if role == "admin":
        assigned = db.execute(
            "SELECT editor_user_id FROM editor_contributors WHERE contributor_speaker_id=? LIMIT 1",
            (speaker_id,)
        ).fetchone()
        if assigned:
            db.close()
            return jsonify({"error": "This contributor is assigned to an editor. Only that editor can generate prompts for them."}), 403

    settings = db.execute(
        "SELECT speech_type, subcategory, system_prompt, batch_size "
        "FROM editor_generation_settings WHERE editor_user_id=?",
        (user_id,)
    ).fetchone()
    db.close()

    if not settings or not (settings["system_prompt"] or "").strip():
        return jsonify({"error": "Админ одоогоор таньд даалгавар олгоогүй байна."}), 400

    client = get_openai_client()
    if not client:
        return jsonify({"error": "Системийн алдаа гарлаа түр хүлээгээд дахин оролдн уу."}), 500

    # We append a strict format instruction to the admin's system prompt so the model's
    # output is mechanically parseable. The admin's prompt body provides topic+style;
    # this suffix guarantees structure.
    format_suffix = (
        f"\n\nGenerate exactly {settings['batch_size']} sentences. "
        f"Each sentence on its own line. "
        f"Each line MUST be formatted as: MONGOLIAN_TEXT | ENGLISH_TRANSLATION "
        f"with the pipe character separating the two. "
        f"Do not number the lines. Do not add any preface, explanation, or wrapping text — "
        f"only the formatted lines, one per line."
    )
    full_prompt = settings["system_prompt"] + format_suffix

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": f"Generate {settings['batch_size']} prompts now."}
            ],
            temperature=0.8,  # diversity is important; we don't want repetitive sentences
            max_tokens=6000,  # ~50 prompts of ~80 tokens each + buffer
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[GEN] OpenAI error for editor {session.get('username')}: {e}")
        return jsonify({"error": "Системийн алдаа гарлаа түр хүлээгээд дахин оролдн уу."}), 500

    # Parse the output: split on newlines, then on the pipe character.
    prompts = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line: continue
        # Strip any leading bullet/number cruft like "1.", "1)", "-", "*"
        while line and (line[0].isdigit() or line[0] in ".)-*•"):
            line = line[1:].strip()
        if "|" not in line: continue
        mn, _, en = line.partition("|")
        mn = mn.strip(); en = en.strip()
        if mn:
            prompts.append({"mn": mn, "en": en})

    if not prompts:
        # Model didn't follow format — surface a system-error message to the editor.
        # Admin can investigate by looking at Railway logs (we print the raw text below).
        print(f"[GEN] No parseable prompts in LLM output for {session.get('username')}: {text[:500]}")
        return jsonify({"error": "Системийн алдаа гарлаа түр хүлээгээд дахин оролдн уу."}), 500

    print(f"[GEN] Generated {len(prompts)} prompts for editor {session.get('username')} "
          f"(speaker={speaker_id}, type={settings['speech_type']}, sub={settings['subcategory']!r})")
    return jsonify({"prompts": prompts, "count": len(prompts)})

@app.route("/admin/dedupe_report")
@login_required
@admin_required
def dedupe_report():
    """Find all groups of prompts that normalize to the same text — these are duplicates
    across the dataset. Shows them grouped so admin can see what's repeated and decide
    whether to clean up."""
    db = get_db()
    # Find every normalized text value that appears in >1 prompt row.
    dup_keys = db.execute(
        "SELECT text_normalized, COUNT(*) as n "
        "FROM prompts "
        "WHERE text_normalized IS NOT NULL AND text_normalized != '' "
        "GROUP BY text_normalized HAVING COUNT(*) > 1 "
        "ORDER BY n DESC, text_normalized"
    ).fetchall()
    groups = []
    for k in dup_keys:
        rows = db.execute(
            "SELECT p.id, p.speaker_id, p.clip_number, p.text_mn, p.text_en, p.speech_type, "
            "       p.subcategory, p.filename, "
            "       (SELECT status FROM clips WHERE prompt_id=p.id LIMIT 1) AS clip_status "
            "FROM prompts p "
            "WHERE p.text_normalized=? "
            "ORDER BY p.id",
            (k["text_normalized"],)
        ).fetchall()
        groups.append({
            "normalized": k["text_normalized"],
            "count": k["n"],
            "prompts": [dict(r) for r in rows],
        })
    db.close()
    total_dup_prompts = sum(g["count"] for g in groups)
    extra_prompts    = total_dup_prompts - len(groups)  # how many would be removed if we kept one per group
    return render_template(
        "dedupe_report.html",
        groups=groups,
        total_groups=len(groups),
        total_dup_prompts=total_dup_prompts,
        extra_prompts=extra_prompts,
    )

@app.route("/admin/dedupe_resolve", methods=["POST"])
@login_required
@admin_required
def dedupe_resolve():
    """Clean up duplicate prompts: for each group of duplicates, keep ONE and delete
    the others — but only delete prompts that have NO clip attached (deleting a prompt
    with a recorded clip would orphan audio data).

    Rule: keep the earliest prompt (lowest id) when none have clips. When some duplicates
    have clips and others don't, keep the one with the clip and delete the empty ones.
    When MULTIPLE duplicates have clips, do nothing for that group (admin must resolve manually).
    """
    db = get_db()
    deleted = 0
    skipped_unsafe = 0
    kept = 0
    dup_keys = db.execute(
        "SELECT text_normalized "
        "FROM prompts "
        "WHERE text_normalized IS NOT NULL AND text_normalized != '' "
        "GROUP BY text_normalized HAVING COUNT(*) > 1"
    ).fetchall()
    for k in dup_keys:
        rows = db.execute(
            "SELECT p.id, p.speaker_id, "
            "       EXISTS(SELECT 1 FROM clips WHERE prompt_id=p.id) AS has_clip "
            "FROM prompts p "
            "WHERE p.text_normalized=? "
            "ORDER BY p.id",
            (k["text_normalized"],)
        ).fetchall()
        with_clips = [r for r in rows if r["has_clip"]]
        without_clips = [r for r in rows if not r["has_clip"]]

        if len(with_clips) > 1:
            # Multiple recorded duplicates — too risky to auto-resolve. Skip this group.
            skipped_unsafe += 1
            continue

        # Decide which row to keep: prefer one with a clip; otherwise the earliest
        if with_clips:
            keep_id = with_clips[0]["id"]
        else:
            keep_id = rows[0]["id"]   # earliest by id
        kept += 1
        # Delete all duplicates in this group that have NO clip and are NOT the keep row.
        # (The keep row might itself be a no-clip prompt — that's fine, it stays.)
        for r in without_clips:
            if r["id"] != keep_id:
                db.execute("DELETE FROM prompts WHERE id=?", (r["id"],))
                deleted += 1
    db.commit(); db.close()
    print(f"[DEDUP] {session.get('username')} ran dedupe_resolve: deleted={deleted} "
          f"kept={kept} skipped_unsafe={skipped_unsafe} at {datetime.datetime.now().isoformat()}")
    msg = f"Removed {deleted} duplicate prompt(s)."
    if skipped_unsafe > 0:
        msg += f" {skipped_unsafe} group(s) skipped — multiple recorded clips with the same text, please resolve manually."
    flash(msg, "success" if deleted > 0 or skipped_unsafe == 0 else "error")
    return redirect(url_for("dedupe_report"))

@app.route("/admin/dedupe_delete_selected", methods=["POST"])
@login_required
@admin_required
def dedupe_delete_selected():
    """Hard-delete every prompt in the submitted checkbox list.

    For each prompt:
      • Delete the audio file from B2 (best-effort — if B2 errors, we still proceed
        with the DB cleanup so the admin's intent is honored)
      • Remove any delivery_clips rows pointing at it
      • Remove the clips row
      • Remove the prompts row itself

    Per the project decision: no special handling for clips already shipped in
    deliveries — there are no real deliveries yet, so we don't gate on this.
    """
    ids = request.form.getlist("prompt_ids")
    try:
        ids = [int(x) for x in ids]
    except (TypeError, ValueError):
        flash("Invalid prompt selection.", "error")
        return redirect(url_for("dedupe_report"))

    if not ids:
        flash("No prompts selected.", "error")
        return redirect(url_for("dedupe_report"))

    db = get_db()
    deleted_prompts = 0
    deleted_clips = 0
    b2_failures = 0

    # Get B2 client once and reuse — saves re-auth on every loop
    try:
        b2 = get_b2()
    except Exception as e:
        print(f"[DEDUP] B2 client init failed: {e}")
        b2 = None

    for prompt_id in ids:
        # Look up the prompt and any clip that goes with it
        prompt_row = db.execute("SELECT id, speaker_id, filename FROM prompts WHERE id=?", (prompt_id,)).fetchone()
        if not prompt_row:
            continue  # Already gone or never existed
        clips_rows = db.execute("SELECT id, b2_key FROM clips WHERE prompt_id=?", (prompt_id,)).fetchall()

        # Step 1: Best-effort B2 file deletion for each clip
        if b2:
            for c in clips_rows:
                if c["b2_key"]:
                    try:
                        b2.delete_object(Bucket=B2_BUCKET_NAME, Key=c["b2_key"])
                    except Exception as e:
                        print(f"[DEDUP] B2 delete failed for {c['b2_key']}: {e}")
                        b2_failures += 1

        # Step 2: Remove delivery_clips bridge rows for any of these clips
        for c in clips_rows:
            db.execute("DELETE FROM delivery_clips WHERE clip_id=?", (c["id"],))

        # Step 3: Remove the clips rows
        if clips_rows:
            db.execute("DELETE FROM clips WHERE prompt_id=?", (prompt_id,))
            deleted_clips += len(clips_rows)

        # Step 4: Remove the prompt itself
        db.execute("DELETE FROM prompts WHERE id=?", (prompt_id,))
        deleted_prompts += 1

    db.commit(); db.close()

    print(f"[DEDUP] {session.get('username')} manually deleted {deleted_prompts} duplicate prompt(s), "
          f"{deleted_clips} clip(s), {b2_failures} B2 deletion error(s), "
          f"at {datetime.datetime.now().isoformat()}")

    msg = f"Deleted {deleted_prompts} prompt(s)"
    if deleted_clips > 0:
        msg += f" and {deleted_clips} associated clip(s)"
    if b2_failures > 0:
        msg += f" — {b2_failures} B2 file(s) could not be removed (will need manual B2 cleanup)"
    msg += "."
    flash(msg, "success")
    return redirect(url_for("dedupe_report"))

@app.route("/admin/delete_prompt/<int:prompt_id>", methods=["POST"])
@login_required
@admin_or_login_required
@profile_required
def delete_prompt(prompt_id):
    db=get_db()
    p=db.execute("SELECT * FROM prompts WHERE id=?",(prompt_id,)).fetchone()
    if not p: db.close(); abort(404)
    if not can_manage_speaker(p["speaker_id"]):
        db.close(); abort(403)
    # Only allow delete if no approved clip
    clip=db.execute("SELECT status FROM clips WHERE prompt_id=?",(prompt_id,)).fetchone()
    if clip and clip["status"]=="approved":
        flash("Cannot delete a prompt with an approved clip.","error")
        db.close()
        return redirect(url_for("add_prompts",speaker_id=p["speaker_id"]))
    if clip: db.execute("DELETE FROM clips WHERE prompt_id=?",(prompt_id,))
    db.execute("DELETE FROM prompts WHERE id=?",(prompt_id,))
    db.commit(); db.close()
    flash("Prompt deleted.","success")
    return redirect(request.referrer or _post_action_redirect())

@app.route("/admin/edit_prompt/<int:prompt_id>", methods=["GET","POST"])
@login_required
@admin_or_login_required
@profile_required
def edit_prompt(prompt_id):
    db=get_db()
    p=db.execute("SELECT * FROM prompts WHERE id=?",(prompt_id,)).fetchone()
    if not p: db.close(); abort(404)
    if not can_manage_speaker(p["speaker_id"]):
        db.close(); abort(403)
    if request.method=="POST":
        mn=request.form.get("text_mn","").strip()
        en=request.form.get("text_en","").strip()
        if not mn:
            flash("Mongolian text is required.","error")
        else:
            # Treat an edit like a new entry: re-fingerprint the Mongolian and re-run the
            # same duplicate guard the entry path uses. Reject the save if the edited text
            # now matches another prompt anywhere in the pool (cross-speaker). Exclude this
            # prompt's own row. Always rewrite text_normalized so it can't go stale.
            normalized = normalize_prompt_text(mn)
            dup = None
            if normalized:
                dup = db.execute(
                    "SELECT id FROM prompts WHERE text_normalized=? AND id<>? LIMIT 1",
                    (normalized, prompt_id)).fetchone()
            if dup:
                flash("Энэ текст өгөгдлийн санд аль хэдийн байгаа тул өөрчлөлт хадгалагдсангүй.","error")
            else:
                db.execute("UPDATE prompts SET text_mn=?,text_en=?,text_normalized=? WHERE id=?",
                           (mn,en,normalized,prompt_id))
                db.commit(); flash("Prompt updated.","success")
        db.close()
        return redirect(url_for("add_prompts",speaker_id=p["speaker_id"]))
    db.close()
    return render_template("edit_prompt.html",prompt=p)

# ── Review queue (Stage 3 - two-stage approval) ─────────────────────────────
@app.route("/admin/queue")
@login_required
@admin_required
def review_queue():
    """Admin's queue: pending clips from UNASSIGNED contributors.
    Clips from contributors assigned to editors go to /editor/queue first."""
    db=get_db()
    clips=db.execute(
        "SELECT c.*,p.text_mn,p.text_en,p.clip_number,p.speech_type "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "WHERE c.status='pending' "
        "  AND c.speaker_id NOT IN (SELECT contributor_speaker_id FROM editor_contributors) "
        "ORDER BY c.submitted_at ASC"
    ).fetchall()
    db.close()
    return render_template("review_queue.html",clips=clips,fmt_dur=fmt_dur,
        queue_kind="pending", page_title="Pending review",
        page_sub="Clips from contributors directly under admin (no editor assigned).")

@app.route("/admin/final_queue")
@login_required
@admin_required
def final_queue():
    """Admin's final-review queue: clips that editors have already approved."""
    db=get_db()
    clips=db.execute(
        "SELECT c.*,p.text_mn,p.text_en,p.clip_number,p.speech_type, "
        "       u.username AS editor_username "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "LEFT JOIN users u ON u.id=c.editor_user_id "
        "WHERE c.status='editor_approved' "
        "ORDER BY c.editor_reviewed_at ASC"
    ).fetchall()
    pp = _payroll_pending(db)
    db.close()
    return render_template("review_queue.html",clips=clips,fmt_dur=fmt_dur,
        payroll_pending=pp,
        queue_kind="editor_approved", page_title="Final review",
        page_sub="Clips editors have approved. You give the final approval before they enter the dataset.")

@app.route("/admin/review/<int:clip_id>")
@login_required
@admin_required
def review_clip(clip_id):
    db=get_db()
    clip=db.execute(
        "SELECT c.*,p.text_mn,p.text_en,p.clip_number,p.speech_type,p.id as prompt_id, "
        "       u.username AS editor_username "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "LEFT JOIN users u ON u.id=c.editor_user_id "
        "WHERE c.id=?",(clip_id,)).fetchone()
    if not clip: abort(404)
    db.close()
    audio_url=None
    if clip["b2_key"]:
        try:
            b2=get_b2()
            audio_url=b2.generate_presigned_url("get_object",
                Params={"Bucket":B2_BUCKET_NAME,"Key":clip["b2_key"]},ExpiresIn=3600)
        except Exception as e: print(f"Presign: {e}")
    return render_template("review_clip.html",clip=clip,audio_url=audio_url,
        fmt_dur=fmt_dur, reviewer_role="admin")

@app.route("/admin/decide/<int:clip_id>", methods=["POST"])
@login_required
@admin_required
def decide_clip(clip_id):
    """Admin's decision on a clip — handles both 'pending' (direct) and 'editor_approved' (final stage)."""
    decision=request.form.get("decision")
    reject_note=request.form.get("reject_note","").strip()
    if decision not in ("approved","rejected"): abort(400)
    db=get_db()
    clip=db.execute("SELECT * FROM clips WHERE id=?",(clip_id,)).fetchone()
    if not clip: db.close(); abort(404)
    current_status = clip["status"]
    back = url_for("final_queue") if current_status == "editor_approved" else url_for("review_queue")
    # If the action came from a specific page (e.g. a spot-check sample), return there.
    # Only accept a local relative path to avoid open-redirects.
    nxt = (request.form.get("next") or "").strip()
    if nxt.startswith("/") and not nxt.startswith("//"):
        back = nxt

    if decision == "approved":
        # Ensure the audio actually lives under approved/ BEFORE marking the clip
        # approved, and write the resulting key in the SAME update so status and
        # b2_key can never drift. (The old code set status outside the B2 try, so a
        # half-failed move left approved clips pointing at a stale pending/ key.)
        final_key = clip["b2_key"]
        if clip["b2_key"]:
            try:
                final_key, st = _ensure_approved_any(get_b2(), clip["b2_key"])
            except Exception as e:
                print(f"B2 move (admin approve) clip {clip_id}: {e}")
                db.close()
                flash("Couldn't move the audio to approved storage (B2 error). Clip left unchanged — try again.", "error")
                return redirect(back)
            if st == "missing":
                db.close()
                flash("Couldn't approve: the audio file is missing from storage. Clip left unchanged for investigation.", "error")
                return redirect(back)
        # Lock the rates at the moment of final approval. Editor rate only if the clip has an editor.
        contrib_rate_row = db.execute(
            "SELECT hourly_rate FROM users WHERE speaker_id=?", (clip["speaker_id"],)
        ).fetchone()
        contrib_rate = contrib_rate_row["hourly_rate"] if contrib_rate_row else RATE_PER_HOUR
        editor_rate = None
        if clip["editor_user_id"]:
            ed_row = db.execute(
                "SELECT hourly_rate FROM users WHERE id=?", (clip["editor_user_id"],)
            ).fetchone()
            editor_rate = ed_row["hourly_rate"] if ed_row else 70000
        db.execute(
            "UPDATE clips SET status='approved', b2_key=?, reject_note=NULL, "
            "  reviewed_at=datetime('now'), admin_reviewed_at=datetime('now'), "
            "  contributor_rate_at_approval=?, editor_rate_at_approval=? "
            "WHERE id=?",
            (final_key, contrib_rate, editor_rate, clip_id))
        db.commit(); db.close()
        _dbw = get_db(); _pp = _payroll_pending(_dbw); _dbw.close()
        if _pp:
            flash(f"⚠ Note: payroll was downloaded {_pp} UTC and All Paid has not been "
                  f"clicked — this approval is NOT in that payroll.", "warning")
        flash("Clip approved.", "success")
        return redirect(back)

    # decision == "rejected"
    if clip["b2_key"] and "editor_approved/" in clip["b2_key"]:
        # Move the file back out of editor_approved/ so it doesn't linger there.
        try:
            b2 = get_b2()
            new_key = clip["b2_key"].replace("editor_approved/", "pending/", 1)
            b2.copy_object(Bucket=B2_BUCKET_NAME,
                CopySource={"Bucket": B2_BUCKET_NAME, "Key": clip["b2_key"]}, Key=new_key)
            b2.delete_object(Bucket=B2_BUCKET_NAME, Key=clip["b2_key"])
            db.execute("UPDATE clips SET b2_key=? WHERE id=?", (new_key, clip_id))
        except Exception as e:
            print(f"B2 move (admin reject) clip {clip_id}: {e}")
    db.execute(
        "UPDATE clips SET status='rejected', reject_note=?, "
        "  reviewed_at=datetime('now'), admin_reviewed_at=datetime('now') "
        "WHERE id=?",
        (reject_note, clip_id))
    db.commit(); db.close()
    flash("Clip rejected.", "success")
    return redirect(back)

def _b2_exists(b2, key):
    """True if an object exists at `key`. Any error (incl. 404) -> False.
    B2-only; safe to call from worker threads."""
    if not key:
        return False
    try:
        b2.head_object(Bucket=B2_BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False

def _ensure_approved_file(b2, b2_key):
    """Idempotently ensure a clip's audio lives under approved/. Returns (final_key, status):
        'moved'   - file was in editor_approved/, now copied to approved/ and original deleted
        'already' - file was already in approved/ (a prior interrupted run); only DB needs fixing
        'missing' - file in NEITHER location (do NOT approve; needs investigation)
        'noop'    - key empty or already an approved/ key; leave as-is
    Performs ONLY B2 calls — never touches the database, so it is safe in worker threads.
    Order is chosen so the common case (source still present) costs the fewest calls."""
    if not b2_key or "editor_approved/" not in b2_key:
        return (b2_key, 'noop')
    approved_key = b2_key.replace("editor_approved/", "approved/", 1)
    if _b2_exists(b2, b2_key):
        # Source still present -> normal move: copy first, then delete (never lose audio).
        b2.copy_object(Bucket=B2_BUCKET_NAME,
                       CopySource={"Bucket": B2_BUCKET_NAME, "Key": b2_key},
                       Key=approved_key)
        b2.delete_object(Bucket=B2_BUCKET_NAME, Key=b2_key)
        return (approved_key, 'moved')
    # Source gone -> was it already moved by a previous interrupted run?
    if _b2_exists(b2, approved_key):
        return (approved_key, 'already')
    return (None, 'missing')

def _ensure_approved_any(b2, b2_key):
    """Like _ensure_approved_file, but accepts a pending/ OR editor_approved/ source
    (admin can approve a clip directly from pending), or an already-approved/ key.
    Idempotent, copy-before-delete, never loses audio. B2-only; no DB access.
    Returns (final_key, status): moved | already | missing | noop."""
    if not b2_key:
        return (b2_key, 'noop')
    if b2_key.startswith("approved/"):
        return (b2_key, 'already') if _b2_exists(b2, b2_key) else (None, 'missing')
    if b2_key.startswith("pending/"):
        approved_key = b2_key.replace("pending/", "approved/", 1)
    elif b2_key.startswith("editor_approved/"):
        approved_key = b2_key.replace("editor_approved/", "approved/", 1)
    else:
        return (b2_key, 'noop')
    if _b2_exists(b2, b2_key):
        b2.copy_object(Bucket=B2_BUCKET_NAME,
                       CopySource={"Bucket": B2_BUCKET_NAME, "Key": b2_key},
                       Key=approved_key)
        b2.delete_object(Bucket=B2_BUCKET_NAME, Key=b2_key)
        return (approved_key, 'moved')
    # Source gone — maybe a prior interrupted run already moved it.
    if _b2_exists(b2, approved_key):
        return (approved_key, 'already')
    return (None, 'missing')

def _ensure_pending_file(b2, b2_key):
    """Mirror of _ensure_approved_file for the REJECT path: move the file from
    editor_approved/ back to pending/ so the contributor can re-record.
    Same idempotency contract (copy-then-delete; reconciles after an interrupted run).
    Returns (final_key, status) with same vocabulary as _ensure_approved_file."""
    if not b2_key or "editor_approved/" not in b2_key:
        return (b2_key, 'noop')
    pending_key = b2_key.replace("editor_approved/", "pending/", 1)
    if _b2_exists(b2, b2_key):
        b2.copy_object(Bucket=B2_BUCKET_NAME,
                       CopySource={"Bucket": B2_BUCKET_NAME, "Key": b2_key},
                       Key=pending_key)
        b2.delete_object(Bucket=B2_BUCKET_NAME, Key=b2_key)
        return (pending_key, 'moved')
    if _b2_exists(b2, pending_key):
        return (pending_key, 'already')
    return (None, 'missing')

def _format_bulk_msg(counts, scope=None):
    """Build the user-facing flash message for a bulk-approve run."""
    prefix = f"Bulk-approved {counts['approved']} clip(s)"
    if scope:
        prefix += f" for {scope}"
    prefix += "."
    detail = []
    if counts["moved"]:      detail.append(f"{counts['moved']} files moved to approved/")
    if counts["reconciled"]: detail.append(f"{counts['reconciled']} reconciled from a prior interrupted run")
    if counts["missing"]:    detail.append(f"{counts['missing']} skipped — file missing, needs investigation")
    if counts["errors"]:     detail.append(f"{counts['errors']} skipped — B2 error")
    if detail:
        prefix += " (" + "; ".join(detail) + ".)"
    return prefix

def _bulk_approve_clips(db, clips, log_tag="", job_id=None):
    """Core bulk-approve body, shared by the full-queue route and the per-speaker route.
    Takes a pre-fetched list of editor_approved clips and approves them with the same
    parallel B2 work + per-batch commits + idempotent file-reconciliation guarantees.
    If job_id is given, per-batch progress is written to bulk_jobs in the SAME commit
    as the batch itself, so the progress display can never disagree with the data.

    Returns a counts dict ({approved, moved, reconciled, missing, errors}); the caller
    handles redirects and flashes.
    """
    b2 = get_b2()
    BATCH_SIZE = 25      # commit boundary — bounds the most any interruption can lose
    MAX_WORKERS = 10     # concurrent B2 operations
    approved = moved = reconciled = missing = errors = 0

    def _b2_work(c):
        """Worker-thread body: B2 only, NO database access."""
        try:
            # _ensure_approved_any (not _ensure_approved_file): bulk approve can act on
            # clips still at pending status, and the old helper nooped pending/ keys —
            # marking clips approved while their audio never moved out of pending/.
            final_key, st = _ensure_approved_any(b2, c["b2_key"])
            return (c, final_key, st)
        except Exception as e:
            print(f"[BULK APPROVE{' ' + log_tag if log_tag else ''}] B2 error clip {c['id']}: {e}")
            return (c, None, 'error')

    for i in range(0, len(clips), BATCH_SIZE):
        batch = clips[i:i + BATCH_SIZE]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = list(ex.map(_b2_work, batch))
        for (c, final_key, st) in results:
            if st in ("missing", "error"):
                if st == "missing": missing += 1
                else: errors += 1
                # Leave the clip as editor_approved for investigation; never approve a
                # clip whose audio we couldn't confirm.
                continue
            # Freeze per-clip contributor and editor rates at this approval moment
            contrib_rate_row = db.execute(
                "SELECT hourly_rate FROM users WHERE speaker_id=?", (c["speaker_id"],)
            ).fetchone()
            contrib_rate = contrib_rate_row["hourly_rate"] if contrib_rate_row else RATE_PER_HOUR
            editor_rate = None
            if c["editor_user_id"]:
                ed_row = db.execute(
                    "SELECT hourly_rate FROM users WHERE id=?", (c["editor_user_id"],)
                ).fetchone()
                editor_rate = ed_row["hourly_rate"] if ed_row else 70000
            db.execute(
                "UPDATE clips SET status='approved', b2_key=?, "
                "  reviewed_at=datetime('now'), admin_reviewed_at=datetime('now'), "
                "  contributor_rate_at_approval=?, editor_rate_at_approval=? "
                "WHERE id=?",
                (final_key, contrib_rate, editor_rate, c["id"]))
            approved += 1
            if st == "moved": moved += 1
            elif st == "already": reconciled += 1
        if job_id is not None:
            db.execute(
                "UPDATE bulk_jobs SET done=?, approved=?, moved=?, reconciled=?, "
                "  missing=?, errors=? WHERE id=?",
                (min(i + BATCH_SIZE, len(clips)), approved, moved, reconciled,
                 missing, errors, job_id))
        db.commit()   # durable, resumable progress after each batch

    return {"approved": approved, "moved": moved, "reconciled": reconciled,
            "missing": missing, "errors": errors}

def _run_bulk_approve_job(job_id, speaker_id=None):
    """Background-thread body for a bulk approve. Opens its own DB connection
    (SQLite connections are per-thread), re-selects the still-editor_approved clips,
    runs the standard bulk core with progress reporting, and finalizes the job row.
    Any crash marks the job 'failed' with the error visible on the progress page —
    already-committed batches stay saved, so re-running simply finishes the rest."""
    db = get_db()
    try:
        if speaker_id:
            clips = db.execute(
                "SELECT id, b2_key, speaker_id, editor_user_id FROM clips "
                "WHERE status='editor_approved' AND speaker_id=?", (speaker_id,)).fetchall()
        else:
            clips = db.execute(
                "SELECT id, b2_key, speaker_id, editor_user_id FROM clips "
                "WHERE status='editor_approved'").fetchall()
        db.execute("UPDATE bulk_jobs SET total=? WHERE id=?", (len(clips), job_id))
        db.commit()
        _bulk_approve_clips(db, clips,
                            log_tag=(f"speaker {speaker_id}" if speaker_id else "full queue"),
                            job_id=job_id)
        db.execute("UPDATE bulk_jobs SET status='done', done=total, finished_at=datetime('now') "
                   "WHERE id=?", (job_id,))
        db.commit()
    except Exception as e:
        import traceback
        print(f"[BULK JOB {job_id}] failed:\n" + traceback.format_exc())
        try:
            db.execute("UPDATE bulk_jobs SET status='failed', error=?, finished_at=datetime('now') "
                       "WHERE id=?", (str(e), job_id))
            db.commit()
        except Exception:
            pass
    finally:
        db.close()

def _start_bulk_approve_job(scope_label, speaker_id=None):
    """Create the job row and spawn the worker thread. Returns the job id, or the id of
    an already-running bulk job (only one runs at a time — they'd contend on B2 and DB)."""
    db = get_db()
    running = db.execute(
        "SELECT id FROM bulk_jobs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
    if running:
        db.close()
        return running["id"], False
    cur = db.execute("INSERT INTO bulk_jobs (kind, scope) VALUES ('approve', ?)", (scope_label,))
    job_id = cur.lastrowid
    db.commit(); db.close()
    t = threading.Thread(target=_run_bulk_approve_job, args=(job_id, speaker_id), daemon=True)
    t.start()
    return job_id, True

@app.route("/admin/bulk_approve_editor_queue", methods=["POST"])
@login_required
@admin_required
def bulk_approve_editor_queue():
    """Approve every clip currently in the editor_approved queue — as a BACKGROUND job.
    Large queues exceed Cloudflare's ~100s window; running in a thread with DB-backed
    progress means the admin always sees a live progress page instead of a timeout."""
    db = get_db()
    n = db.execute("SELECT COUNT(*) AS n FROM clips WHERE status='editor_approved'").fetchone()["n"]
    db.close()
    if not n:
        flash("No clips in the editor queue.", "info")
        return redirect(url_for("final_queue"))
    job_id, started = _start_bulk_approve_job("full queue")
    if not started:
        flash("A bulk job is already running — showing its progress.", "info")
    return redirect(url_for("bulk_job_page", job_id=job_id))

@app.route("/admin/bulk_approve_speaker/<speaker_id>", methods=["POST"])
@login_required
@admin_required
def bulk_approve_speaker(speaker_id):
    """Approve every editor-approved clip for ONE contributor — as a BACKGROUND job.
    Same guarantees as the full-queue version, scoped to one speaker."""
    db = get_db()
    n = db.execute("SELECT COUNT(*) AS n FROM clips WHERE status='editor_approved' AND speaker_id=?",
                   (speaker_id,)).fetchone()["n"]
    db.close()
    if not n:
        flash(f"No editor-approved clips for {speaker_id}.", "info")
        return redirect(url_for("final_queue"))
    job_id, started = _start_bulk_approve_job(speaker_id, speaker_id=speaker_id)
    if not started:
        flash("A bulk job is already running — showing its progress.", "info")
    return redirect(url_for("bulk_job_page", job_id=job_id))

@app.route("/admin/bulk_jobs/<int:job_id>")
@login_required
@admin_required
def bulk_job_page(job_id):
    db = get_db()
    job = db.execute("SELECT * FROM bulk_jobs WHERE id=?", (job_id,)).fetchone()
    db.close()
    if not job:
        abort(404)
    return render_template("bulk_job.html", job=job)

@app.route("/admin/bulk_jobs/<int:job_id>/status")
@login_required
@admin_required
def bulk_job_status(job_id):
    db = get_db()
    job = db.execute("SELECT * FROM bulk_jobs WHERE id=?", (job_id,)).fetchone()
    db.close()
    if not job:
        return jsonify({"ok": False}), 404
    return jsonify({"ok": True, **{k: job[k] for k in job.keys()}})

# ----------------------------------------------------------------------------
# Bulk REJECT — same parallel + idempotent + resumable pattern as bulk approve,
# but moves files back to pending/ and flips status to 'rejected' with a
# REQUIRED admin-supplied note that's shown to every affected contributor.
# ----------------------------------------------------------------------------

def _bulk_reject_clips(db, clips, note, log_tag=""):
    """Reject editor-approved clips in bulk. Moves each file back to pending/, sets
    status='rejected' with the admin-supplied note (visible to the contributor), and
    stamps the review timestamps.
    Returns counts dict: {rejected, moved, reconciled, missing, errors}."""
    b2 = get_b2()
    BATCH_SIZE = 25
    MAX_WORKERS = 10
    rejected = moved = reconciled = missing = errors = 0

    def _b2_work(c):
        try:
            final_key, st = _ensure_pending_file(b2, c["b2_key"])
            return (c, final_key, st)
        except Exception as e:
            print(f"[BULK REJECT{' ' + log_tag if log_tag else ''}] B2 error clip {c['id']}: {e}")
            return (c, None, 'error')

    for i in range(0, len(clips), BATCH_SIZE):
        batch = clips[i:i + BATCH_SIZE]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results = list(ex.map(_b2_work, batch))
        for (c, final_key, st) in results:
            if st in ("missing", "error"):
                if st == "missing": missing += 1
                else: errors += 1
                continue
            db.execute(
                "UPDATE clips SET status='rejected', b2_key=?, reject_note=?, "
                "  reviewed_at=datetime('now'), admin_reviewed_at=datetime('now') "
                "WHERE id=?",
                (final_key, note, c["id"]))
            rejected += 1
            if st == "moved": moved += 1
            elif st == "already": reconciled += 1
        db.commit()
    return {"rejected": rejected, "moved": moved, "reconciled": reconciled,
            "missing": missing, "errors": errors}

def _format_bulk_reject_msg(counts):
    msg = f"Bulk-rejected {counts['rejected']} clip(s)."
    detail = []
    if counts["moved"]:      detail.append(f"{counts['moved']} files moved back to pending/")
    if counts["reconciled"]: detail.append(f"{counts['reconciled']} reconciled from a prior interrupted run")
    if counts["missing"]:    detail.append(f"{counts['missing']} skipped — file missing")
    if counts["errors"]:     detail.append(f"{counts['errors']} skipped — B2 error")
    if detail:
        msg += " (" + "; ".join(detail) + ".)"
    return msg

@app.route("/admin/bulk_reject_editor_queue", methods=["POST"])
@login_required
@admin_required
def bulk_reject_editor_queue():
    """Reject every clip currently in the editor_approved queue — moves files back to
    pending/ and marks them rejected with the admin-supplied note. Destructive: the
    front-end requires the note + a typed 'REJECT' confirmation before this POSTs."""
    note = (request.form.get("reject_note") or "").strip()
    if not note:
        flash("A rejection reason is required for bulk reject.", "error")
        return redirect(url_for("final_queue"))
    db = get_db()
    clips = db.execute(
        "SELECT id, b2_key, speaker_id FROM clips WHERE status='editor_approved'"
    ).fetchall()
    if not clips:
        db.close(); flash("No clips in the editor queue.", "info")
        return redirect(url_for("final_queue"))
    counts = _bulk_reject_clips(db, clips, note=note, log_tag="full queue")
    db.close()
    flash(_format_bulk_reject_msg(counts),
          "info" if (counts["missing"] or counts["errors"]) else "success")
    return redirect(url_for("final_queue"))

@app.route("/admin/spot_check_queue")
@login_required
@admin_required
def spot_check_queue():
    """Show admin a random ~10% sample of the editor_approved queue.
    The form shows each sampled clip with its audio. The admin reviews these,
    then can bulk-approve the rest."""
    import random
    db=get_db()
    all_clips=db.execute(
        "SELECT c.*,p.text_mn,p.text_en,p.clip_number,p.speech_type, "
        "       u.username AS editor_username "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "LEFT JOIN users u ON u.id=c.editor_user_id "
        "WHERE c.status='editor_approved' "
        "ORDER BY c.editor_reviewed_at ASC"
    ).fetchall()
    db.close()
    if not all_clips:
        flash("No clips in the editor-approved queue.","info")
        return redirect(url_for("admin_dashboard"))
    # Sample at least 1 clip, up to 10% (rounded up)
    sample_size = max(1, math.ceil(len(all_clips) * 0.10))
    sample_size = min(sample_size, len(all_clips))
    sampled = random.sample(list(all_clips), sample_size)
    sampled_ids = {c["id"] for c in sampled}
    # Generate presigned URLs for the sampled clips
    b2 = get_b2()
    sampled_with_urls = []
    for c in sampled:
        url = None
        if c["b2_key"]:
            try:
                url = b2.generate_presigned_url("get_object",
                    Params={"Bucket":B2_BUCKET_NAME,"Key":c["b2_key"]},ExpiresIn=3600)
            except Exception as e: print(f"Spot-check presign: {e}")
        sampled_with_urls.append({**dict(c), "audio_url": url})
    _db_pp = get_db(); _payroll_pending_cached = _payroll_pending(_db_pp); _db_pp.close()
    return render_template("spot_check.html",
        payroll_pending=_payroll_pending_cached,
        sampled=sampled_with_urls,
        total_count=len(all_clips),
        sample_count=len(sampled),
        fmt_dur=fmt_dur)

@app.route("/admin/bulk_reject_speaker/<speaker_id>", methods=["POST"])
@login_required
@admin_required
def bulk_reject_speaker(speaker_id):
    """Reject every editor-approved clip for ONE contributor — moves files back to
    pending/ and marks them rejected with the admin-supplied note (shown to the
    contributor). Same parallel + idempotent + resumable guarantees as the full-queue
    reject, scoped to one speaker. Destructive: the page requires the note + a typed
    'REJECT' confirmation before POSTing."""
    note = (request.form.get("reject_note") or "").strip()
    if not note:
        flash("A rejection reason is required.", "error")
        return redirect(url_for("spot_check_speaker", speaker_id=speaker_id))
    db = get_db()
    clips = db.execute(
        "SELECT id, b2_key, speaker_id FROM clips "
        "WHERE status='editor_approved' AND speaker_id=?", (speaker_id,)
    ).fetchall()
    if not clips:
        db.close(); flash(f"No editor-approved clips for {speaker_id}.", "info")
        return redirect(url_for("final_queue"))
    counts = _bulk_reject_clips(db, clips, note=note, log_tag=f"speaker {speaker_id}")
    db.close()
    flash(_format_bulk_reject_msg(counts),
          "info" if (counts["missing"] or counts["errors"]) else "success")
    return redirect(url_for("final_queue"))

@app.route("/admin/spot_check_speaker/<speaker_id>")
@login_required
@admin_required
def spot_check_speaker(speaker_id):
    """Show a random ~10% sample of ONE contributor's editor_approved clips, with audio,
    so the admin can judge that speaker's batch before approving or rejecting all of it.
    Pairs with bulk_approve_speaker (approve all) and bulk_reject_speaker (reject all)."""
    import random
    db = get_db()
    all_clips = db.execute(
        "SELECT c.*, p.text_mn, p.text_en, p.clip_number, p.speech_type, "
        "       u.username AS editor_username "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "LEFT JOIN users u ON u.id=c.editor_user_id "
        "WHERE c.status='editor_approved' AND c.speaker_id=? "
        "ORDER BY c.editor_reviewed_at ASC", (speaker_id,)
    ).fetchall()
    db.close()
    if not all_clips:
        flash(f"No editor-approved clips for {speaker_id}.", "info")
        return redirect(url_for("admin_dashboard"))
    sample_size = min(len(all_clips), max(1, math.ceil(len(all_clips) * 0.10)))
    sampled = random.sample(list(all_clips), sample_size)
    b2 = get_b2()
    sampled_with_urls = []
    for c in sampled:
        url = None
        if c["b2_key"]:
            try:
                url = b2.generate_presigned_url("get_object",
                    Params={"Bucket": B2_BUCKET_NAME, "Key": c["b2_key"]}, ExpiresIn=3600)
            except Exception as e:
                print(f"[SPOTCHECK] presign clip {c['id']}: {e}")
        sampled_with_urls.append({**dict(c), "audio_url": url})
    _db_pp = get_db(); _payroll_pending_cached = _payroll_pending(_db_pp); _db_pp.close()
    return render_template("spot_check_speaker.html",
        payroll_pending=_payroll_pending_cached,
        speaker_id=speaker_id,
        sampled=sampled_with_urls,
        total_count=len(all_clips),
        sample_count=len(sampled),
        fmt_dur=fmt_dur)

# ── Editor's review routes ────────────────────────────────────────────────────
@app.route("/editor/queue")
@login_required
@profile_required
def editor_queue():
    """Editor's queue: pending clips ONLY from contributors assigned to them."""
    if session.get("role") != "editor": abort(403)
    editor_user_id = session["user_id"]
    db=get_db()
    clips=db.execute(
        "SELECT c.*,p.text_mn,p.text_en,p.clip_number,p.speech_type "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "JOIN editor_contributors ec ON ec.contributor_speaker_id = c.speaker_id "
        "WHERE c.status='pending' AND ec.editor_user_id = ? "
        "ORDER BY c.submitted_at ASC",
        (editor_user_id,)
    ).fetchall()
    db.close()
    return render_template("review_queue.html",clips=clips,fmt_dur=fmt_dur,
        queue_kind="editor_pending", page_title="My review queue",
        page_sub="Clips from your contributors waiting for your review.")

@app.route("/editor/review/<int:clip_id>")
@login_required
@profile_required
def editor_review_clip(clip_id):
    if session.get("role") != "editor": abort(403)
    editor_user_id = session["user_id"]
    db=get_db()
    clip=db.execute(
        "SELECT c.*,p.text_mn,p.text_en,p.clip_number,p.speech_type,p.id as prompt_id "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "WHERE c.id=?",(clip_id,)).fetchone()
    if not clip: db.close(); abort(404)
    # Ownership check: editor can only review clips from their own contributors
    if not can_manage_speaker(clip["speaker_id"]):
        db.close(); abort(403)
    if clip["status"] != "pending":
        db.close()
        flash("This clip is no longer in the pending state.","info")
        return redirect(url_for("editor_queue"))
    db.close()
    audio_url=None
    if clip["b2_key"]:
        try:
            b2=get_b2()
            audio_url=b2.generate_presigned_url("get_object",
                Params={"Bucket":B2_BUCKET_NAME,"Key":clip["b2_key"]},ExpiresIn=3600)
        except Exception as e: print(f"Presign: {e}")
    return render_template("review_clip.html",clip=clip,audio_url=audio_url,
        fmt_dur=fmt_dur, reviewer_role="editor")

@app.route("/editor/decide/<int:clip_id>", methods=["POST"])
@login_required
@profile_required
def editor_decide_clip(clip_id):
    """Editor's decision: approve sends to editor_approved (admin sees next),
    reject sends back to contributor with feedback."""
    if session.get("role") != "editor": abort(403)
    editor_user_id = session["user_id"]
    decision = request.form.get("decision")
    reject_note = request.form.get("reject_note","").strip()
    if decision not in ("approved","rejected"): abort(400)
    db=get_db()
    clip=db.execute("SELECT * FROM clips WHERE id=?",(clip_id,)).fetchone()
    if not clip: db.close(); abort(404)
    if not can_manage_speaker(clip["speaker_id"]):
        db.close(); abort(403)
    if clip["status"] != "pending":
        db.close()
        flash("This clip is no longer in the pending state.","info")
        return redirect(url_for("editor_queue"))

    if decision == "approved":
        # Move B2 file from pending/ to editor_approved/.
        # FAILURE SEMANTICS (fixes the grey-player bug): if the COPY fails, abort the
        # approval entirely — the clip stays pending and the editor retries. If the copy
        # succeeds but the DELETE of the pending original fails, keep the pointer on the
        # editor_approved/ copy (the copy exists!) and let the stray pending file linger
        # harmlessly. The old code reset the pointer to pending/ on ANY error while still
        # marking the clip editor_approved — a divergent state that later turned into
        # dead players when bulk approve consumed the pending copy.
        new_key = clip["b2_key"]
        if clip["b2_key"] and "pending/" in clip["b2_key"]:
            b2 = get_b2()
            moved_key = clip["b2_key"].replace("pending/", "editor_approved/", 1)
            try:
                b2.copy_object(Bucket=B2_BUCKET_NAME,
                    CopySource={"Bucket": B2_BUCKET_NAME, "Key": clip["b2_key"]}, Key=moved_key)
            except Exception as e:
                print(f"B2 editor-approve copy failed (clip {clip_id}): {e}")
                db.close()
                flash("Storage error while approving — nothing was changed. Please try again.", "error")
                return redirect(url_for("editor_queue"))
            new_key = moved_key
            try:
                b2.delete_object(Bucket=B2_BUCKET_NAME, Key=clip["b2_key"])
            except Exception as e:
                print(f"B2 editor-approve delete-after-copy failed (clip {clip_id}; "
                      f"stray pending copy lingers, harmless): {e}")
        db.execute(
            "UPDATE clips SET status='editor_approved', editor_user_id=?, "
            "  editor_reviewed_at=datetime('now'), b2_key=? WHERE id=?",
            (editor_user_id, new_key, clip_id))
        flash("Clip approved and forwarded to admin for final review.","success")
    else:
        # Rejection from editor: same as before — sends clip back to contributor
        db.execute(
            "UPDATE clips SET status='rejected', reject_note=?, "
            "  editor_user_id=?, reviewed_at=datetime('now'), "
            "  editor_reviewed_at=datetime('now') WHERE id=?",
            (reject_note, editor_user_id, clip_id))
        flash("Clip rejected and sent back to the contributor.","success")
    db.commit(); db.close()
    return redirect(url_for("editor_queue"))

@app.route("/admin/mark_compensated/<speaker_id>", methods=["POST"])
@login_required
@admin_required
def mark_compensated(speaker_id):
    db=get_db()
    # First: record the payment transaction (before flipping the flag).
    user = db.execute(
        "SELECT id, hourly_rate FROM users WHERE speaker_id=? AND role='contributor'",
        (speaker_id,)
    ).fetchone()
    if user:
        unpaid_clips = db.execute(
            "SELECT duration_seconds, contributor_rate_at_approval "
            "FROM clips WHERE speaker_id=? AND status='approved' AND COALESCE(compensated,0)=0",
            (speaker_id,)
        ).fetchall()
        if unpaid_clips:
            fallback = user["hourly_rate"] if user["hourly_rate"] else 35000
            amount = calc_comp_grouped(unpaid_clips, "contributor_rate_at_approval", fallback)
            if amount > 0:
                total_sec = sum(c["duration_seconds"] or 0 for c in unpaid_clips)
                db.execute(
                    "INSERT INTO payment_transactions "
                    "(user_id, user_role, amount, clip_count, duration_seconds, paid_by_username, method) "
                    "VALUES (?, 'contributor', ?, ?, ?, ?, 'individual')",
                    (user["id"], amount, len(unpaid_clips), total_sec, session.get("username"))
                )
    db.execute("UPDATE clips SET compensated=1 WHERE speaker_id=? AND status='approved' AND compensated=0",(speaker_id,))
    conv_amt = _conv_settle_speaker(db, speaker_id, "conv_individual")   # conversations too
    db.commit(); db.close()
    flash(f"All pending compensation marked as paid for {speaker_id}." + (f" (incl. ₮{conv_amt:,} for conversations)" if conv_amt else ""),"success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/mark_editor_paid/<int:editor_id>", methods=["POST"])
@login_required
@admin_required
def mark_editor_paid(editor_id):
    """Mark all of this editor's unpaid approved clips as editor-paid.
    This does NOT affect the contributor's compensated flag — that's separate."""
    db=get_db()
    editor=db.execute("SELECT id, username, hourly_rate, COALESCE(editor_penalty_pct,0) AS editor_penalty_pct FROM users WHERE id=? AND role='editor'", (editor_id,)).fetchone()
    if not editor: db.close(); abort(404)
    # Record the payment transaction before flipping the flag.
    unpaid_clips = db.execute(
        "SELECT duration_seconds, editor_rate_at_approval "
        "FROM clips WHERE editor_user_id=? AND status='approved' AND COALESCE(editor_paid,0)=0",
        (editor_id,)
    ).fetchall()
    if unpaid_clips:
        fallback = editor["hourly_rate"] if editor["hourly_rate"] else 0
        amount = apply_editor_penalty(
            calc_comp_grouped(unpaid_clips, "editor_rate_at_approval", fallback),
            editor["editor_penalty_pct"])
        if amount > 0:
            total_sec = sum(c["duration_seconds"] or 0 for c in unpaid_clips)
            db.execute(
                "INSERT INTO payment_transactions "
                "(user_id, user_role, amount, clip_count, duration_seconds, paid_by_username, method) "
                "VALUES (?, 'editor', ?, ?, ?, ?, 'individual')",
                (editor_id, amount, len(unpaid_clips), total_sec, session.get("username"))
            )
    db.execute(
        "UPDATE clips SET editor_paid=1 "
        "WHERE editor_user_id=? AND status='approved' AND COALESCE(editor_paid,0)=0",
        (editor_id,)
    )
    _conv_settle_editor(db, editor_id, "conv_individual")   # conversations too (same penalty applies)
    # Settling this editor ends the penalty cycle: the reduction applied to THIS payout;
    # the next cycle starts clean.
    db.execute("UPDATE users SET editor_penalty_pct=0, editor_penalty_reason=NULL, "
               "editor_penalty_at=NULL WHERE id=?", (editor_id,))
    db.commit(); db.close()
    penalty_note = (f" (after {editor['editor_penalty_pct']}% quality penalty — now cleared)"
                    if editor["editor_penalty_pct"] else "")
    flash(f"Editor '{editor['username']}' marked as paid.{penalty_note}","success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/editor_penalty/<int:editor_id>", methods=["POST"])
@login_required
@admin_required
def set_editor_penalty(editor_id):
    """Set (or clear, with 0) a percentage pay reduction on one editor for the CURRENT
    payment cycle — a quality penalty for pushing unedited/unreviewed work downstream.
    Applies everywhere the editor's unpaid amount appears (their own earnings view, the
    admin dashboard, the payroll export, and the settle transaction) and clears
    automatically when that editor is settled (individual "Mark paid" or bulk
    "All Paid")."""
    db = get_db()
    editor = db.execute(
        "SELECT id, username FROM users WHERE id=? AND role='editor'",
        (editor_id,)).fetchone()
    if not editor:
        db.close(); abort(404)
    try:
        pct = int(request.form.get("penalty_pct", "").strip())
    except (TypeError, ValueError):
        db.close()
        flash("Penalty must be a whole number between 0 and 100.", "error")
        return redirect(url_for("admin_dashboard"))
    if pct < 0 or pct > 100:
        db.close()
        flash("Penalty must be between 0 and 100 percent.", "error")
        return redirect(url_for("admin_dashboard"))
    reason = (request.form.get("penalty_reason") or "").strip()
    if pct == 0:
        db.execute("UPDATE users SET editor_penalty_pct=0, editor_penalty_reason=NULL, "
                   "editor_penalty_at=NULL WHERE id=?", (editor_id,))
        db.commit(); db.close()
        flash(f"Penalty removed for editor '{editor['username']}'.", "success")
    else:
        db.execute("UPDATE users SET editor_penalty_pct=?, editor_penalty_reason=?, "
                   "editor_penalty_at=datetime('now') WHERE id=?",
                   (pct, reason or None, editor_id))
        db.commit(); db.close()
        flash(f"{pct}% pay reduction applied to editor '{editor['username']}' for this "
              f"payment cycle. It will clear automatically when they are next marked paid.",
              "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/mark_all_paid", methods=["POST"])
@login_required
@admin_required
def mark_all_paid():
    """Bulk-mark every outstanding payment as paid in one action.
    Used at the end of a payment cycle after bulk bank transfer.

    Affects:
      • Contributors (not archived): clips.compensated = 1
      • Editors      (not archived): clips.editor_paid = 1

    Archived workers' clips are NOT touched — they were already excluded from
    active payroll exports, so they're outside the cycle. If you need to settle
    with an archived worker, do it via SQL or unarchive them first.

    The flash message shows the counts so admin can sanity-check before walking
    away. The Excel payroll export they generated BEFORE pressing this serves as
    the audit record of exactly who got paid what.
    """
    db = get_db()

    # Snapshot the totals about to be marked, for the flash message
    contrib_summary = db.execute(
        "SELECT COUNT(DISTINCT c.speaker_id) AS workers, "
        "       COUNT(*) AS clips, "
        "       COALESCE(SUM(c.duration_seconds),0) AS sec "
        "FROM clips c "
        "JOIN users u ON u.speaker_id = c.speaker_id "
        "WHERE c.status='approved' AND COALESCE(c.compensated,0)=0 "
        "  AND u.role='contributor' AND COALESCE(u.archived,0)=0"
    ).fetchone()
    editor_summary = db.execute(
        "SELECT COUNT(DISTINCT c.editor_user_id) AS workers, "
        "       COUNT(*) AS clips, "
        "       COALESCE(SUM(c.duration_seconds),0) AS sec "
        "FROM clips c "
        "JOIN users u ON u.id = c.editor_user_id "
        "WHERE c.status='approved' AND COALESCE(c.editor_paid,0)=0 "
        "  AND u.role='editor' AND COALESCE(u.archived,0)=0"
    ).fetchone()

    # Record payment transactions per user BEFORE flipping the paid flags.
    # One row per non-archived user with outstanding earnings — so each user's
    # /earnings page sees a clean history entry tied to this bulk action.
    admin_user = session.get("username")
    # Contributors
    contrib_users = db.execute(
        "SELECT id, speaker_id, hourly_rate FROM users "
        "WHERE role='contributor' AND COALESCE(archived,0)=0"
    ).fetchall()
    for cu in contrib_users:
        rows = db.execute(
            "SELECT duration_seconds, contributor_rate_at_approval "
            "FROM clips WHERE speaker_id=? AND status='approved' AND COALESCE(compensated,0)=0",
            (cu["speaker_id"],)
        ).fetchall()
        if not rows: continue
        fallback = cu["hourly_rate"] if cu["hourly_rate"] else 35000
        amount = calc_comp_grouped(rows, "contributor_rate_at_approval", fallback)
        if amount <= 0: continue
        total_sec = sum(r["duration_seconds"] or 0 for r in rows)
        db.execute(
            "INSERT INTO payment_transactions "
            "(user_id, user_role, amount, clip_count, duration_seconds, paid_by_username, method) "
            "VALUES (?, 'contributor', ?, ?, ?, ?, 'bulk_all_paid')",
            (cu["id"], amount, len(rows), total_sec, admin_user)
        )
    # Editors
    editor_users = db.execute(
        "SELECT id, hourly_rate, COALESCE(editor_penalty_pct,0) AS editor_penalty_pct FROM users "
        "WHERE role='editor' AND COALESCE(archived,0)=0"
    ).fetchall()
    for eu in editor_users:
        rows = db.execute(
            "SELECT duration_seconds, editor_rate_at_approval "
            "FROM clips WHERE editor_user_id=? AND status='approved' AND COALESCE(editor_paid,0)=0",
            (eu["id"],)
        ).fetchall()
        if not rows: continue
        fallback = eu["hourly_rate"] if eu["hourly_rate"] else 0
        amount = apply_editor_penalty(
            calc_comp_grouped(rows, "editor_rate_at_approval", fallback),
            eu["editor_penalty_pct"])
        if amount <= 0: continue
        total_sec = sum(r["duration_seconds"] or 0 for r in rows)
        db.execute(
            "INSERT INTO payment_transactions "
            "(user_id, user_role, amount, clip_count, duration_seconds, paid_by_username, method) "
            "VALUES (?, 'editor', ?, ?, ?, ?, 'bulk_all_paid')",
            (eu["id"], amount, len(rows), total_sec, admin_user)
        )

    # Conversations: same cycle, same people (non-archived), own transaction rows.
    for cu in contrib_users:
        _conv_settle_speaker(db, cu["speaker_id"], "conv_bulk_all_paid")
    for eu in editor_users:
        _conv_settle_editor(db, eu["id"], "conv_bulk_all_paid")

    # Mark contributor compensation paid for non-archived contributors only.
    # We join via the users table to filter archived workers; SQLite supports
    # UPDATE with subquery EXISTS.
    db.execute(
        "UPDATE clips SET compensated=1 "
        "WHERE status='approved' AND COALESCE(compensated,0)=0 "
        "  AND speaker_id IN ("
        "    SELECT speaker_id FROM users "
        "    WHERE role='contributor' AND COALESCE(archived,0)=0"
        "  )"
    )

    # Mark editor compensation paid for non-archived editors only.
    db.execute(
        "UPDATE clips SET editor_paid=1 "
        "WHERE status='approved' AND COALESCE(editor_paid,0)=0 "
        "  AND editor_user_id IN ("
        "    SELECT id FROM users "
        "    WHERE role='editor' AND COALESCE(archived,0)=0"
        "  )"
    )
    # "All Paid" ends the payment cycle: every active editor's quality penalty has now
    # been applied to this settlement, so all penalties reset for the new cycle.
    db.execute(
        "UPDATE users SET editor_penalty_pct=0, editor_penalty_reason=NULL, "
        "  editor_penalty_at=NULL "
        "WHERE role='editor' AND COALESCE(editor_penalty_pct,0)>0")
    # Settle closes the payroll window: the downloaded payroll and the recorded
    # transactions now describe the same moment.
    db.execute("DELETE FROM _one_time_flags WHERE key='payroll_pending'")

    db.commit(); db.close()

    print(f"[AUDIT] {session.get('username')} (admin) marked all paid: "
          f"{contrib_summary['workers']} contributors ({contrib_summary['clips']} clips, "
          f"{int(contrib_summary['sec'])}s), "
          f"{editor_summary['workers']} editors ({editor_summary['clips']} clips, "
          f"{int(editor_summary['sec'])}s) "
          f"at {datetime.datetime.now().isoformat()}")

    msg_parts = []
    if contrib_summary["workers"] > 0:
        msg_parts.append(f"{contrib_summary['workers']} contributor(s) ({contrib_summary['clips']} clips)")
    if editor_summary["workers"] > 0:
        msg_parts.append(f"{editor_summary['workers']} editor(s) ({editor_summary['clips']} clips)")
    if msg_parts:
        flash("Marked all outstanding payments as paid: " + ", ".join(msg_parts) + ".", "success")
    else:
        flash("No outstanding payments to mark — everyone is already paid.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/change_rate", methods=["POST"])
@login_required
@admin_required
def change_rate():
    """Change a contributor's or editor's hourly rate. Past approved clips keep their
    frozen rate; only future final-approvals will use the new rate."""
    user_id = request.form.get("user_id","").strip()
    new_rate_raw = request.form.get("new_rate","").strip()
    if not user_id or not new_rate_raw:
        flash("Missing user or rate.","error")
        return redirect(url_for("admin_dashboard"))
    try:
        user_id = int(user_id)
        new_rate = int(new_rate_raw)
    except ValueError:
        flash("Invalid input.","error")
        return redirect(url_for("admin_dashboard"))
    if new_rate < 0 or new_rate > 10_000_000:
        flash("Rate is out of allowed range.","error")
        return redirect(url_for("admin_dashboard"))
    db = get_db()
    user = db.execute(
        "SELECT id, username, role, hourly_rate FROM users WHERE id=? AND role IN ('contributor','editor')",
        (user_id,)
    ).fetchone()
    if not user:
        db.close()
        flash("User not found.","error")
        return redirect(url_for("admin_dashboard"))
    db.execute("UPDATE users SET hourly_rate=? WHERE id=?", (new_rate, user_id))
    db.commit(); db.close()
    flash(
        f"{user['role'].capitalize()} '{user['username']}' rate changed from "
        f"₮{user['hourly_rate']:,}/hr to ₮{new_rate:,}/hr. "
        f"Past approved clips keep their original rate; new approvals will use ₮{new_rate:,}.",
        "success"
    )
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/set_recording_device/<speaker_id>", methods=["POST"])
@login_required
@admin_required
def set_recording_device(speaker_id):
    """Admin sets the recording device type for a contributor. This value is
    exported in the delivery metadata so buyers know which capture hardware
    produced each clip."""
    device = request.form.get("recording_device","").strip()
    if device not in RECORDING_DEVICE_CHOICES:
        flash("Invalid recording device.","error")
        return redirect(url_for("view_profile", speaker_id=speaker_id))
    db = get_db()
    prof = db.execute("SELECT speaker_id FROM profiles WHERE speaker_id=?", (speaker_id,)).fetchone()
    if not prof:
        db.close()
        flash("Profile not found.","error")
        return redirect(url_for("admin_dashboard"))
    db.execute(
        "UPDATE profiles SET recording_device=?, updated_at=datetime('now') WHERE speaker_id=?",
        (device, speaker_id)
    )
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) set recording_device='{device}' for "
          f"{speaker_id} at {datetime.datetime.now().isoformat()}")
    flash(f"Recording device set to '{device}' for {speaker_id}.","success")
    return redirect(url_for("view_profile", speaker_id=speaker_id))

@app.route("/admin/lock_profile/<speaker_id>", methods=["POST"])
@login_required
@admin_required
def lock_profile(speaker_id):
    """Lock or unlock a contributor's profile after review. While locked, the contributor
    can't edit profile fields or re-run the consent/ID flow, so already-delivered identity
    and consent data can't be silently overwritten. Set lock=1 to lock, lock=0 to unlock."""
    lock = request.form.get("lock") == "1"
    db = get_db()
    prof = db.execute("SELECT speaker_id FROM profiles WHERE speaker_id=?", (speaker_id,)).fetchone()
    if not prof:
        db.close()
        flash("Profile not found.", "error")
        return redirect(url_for("admin_dashboard"))
    if lock:
        db.execute(
            "UPDATE profiles SET locked=1, locked_at=datetime('now'), locked_by=?, "
            "  updated_at=datetime('now') WHERE speaker_id=?",
            (session.get("username"), speaker_id))
    else:
        db.execute(
            "UPDATE profiles SET locked=0, locked_at=NULL, locked_by=NULL, "
            "  updated_at=datetime('now') WHERE speaker_id=?",
            (speaker_id,))
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) {'LOCKED' if lock else 'UNLOCKED'} profile "
          f"{speaker_id} at {datetime.datetime.now().isoformat()}")
    flash(f"Profile {'locked' if lock else 'unlocked'} for {speaker_id}.", "success")
    ref = request.referrer
    if ref and ref.startswith(request.host_url):
        return redirect(ref)
    return redirect(url_for("view_profile", speaker_id=speaker_id))

@app.route("/admin/correct_name/<speaker_id>", methods=["POST"])
@login_required
@admin_required
def correct_name(speaker_id):
    """Admin-only correction of the ID-OCR'd name for the rare OCR misread (e.g.
    Ү read as У). The speaker cannot edit their own name; this keeps names
    admin/ID-sourced, not self-entered. Append-only audit row; the corrected
    full_name flows into the consent PDF / A.1 personalization on the speaker's
    next affirmation. Does not alter any existing consent record."""
    new_name = (request.form.get("full_name") or "").strip()
    reason = (request.form.get("reason") or "").strip()
    if not new_name:
        flash("Corrected name cannot be empty.", "error")
        return redirect(url_for("view_profile", speaker_id=speaker_id))
    db = get_db()
    prof = db.execute("SELECT full_name FROM profiles WHERE speaker_id=?", (speaker_id,)).fetchone()
    if not prof:
        db.close(); abort(404)
    old_name = prof["full_name"] or ""
    if new_name == old_name:
        db.close()
        flash("No change — the name is already that value.", "info")
        return redirect(url_for("view_profile", speaker_id=speaker_id))
    # Update both the contract name and the payout receiver name (kept in sync today),
    # and record the correction append-only.
    db.execute("UPDATE profiles SET full_name=?, receiver_name=? WHERE speaker_id=?",
               (new_name, new_name, speaker_id))
    db.execute("INSERT INTO name_corrections (speaker_id, old_name, new_name, corrected_by, reason) "
               "VALUES (?,?,?,?,?)",
               (speaker_id, old_name, new_name, session.get("username"), reason or None))
    db.commit(); db.close()
    print(f"[AUDIT] {session.get('username')} (admin) corrected name for {speaker_id}: "
          f"{old_name!r} -> {new_name!r}")
    flash(f"Name corrected to '{new_name}'.", "success")
    return redirect(url_for("view_profile", speaker_id=speaker_id))


@app.route("/admin/view_profile/<speaker_id>")
@login_required
@manager_required
@profile_required
def view_profile(speaker_id):
    db=get_db()
    prof=db.execute("SELECT * FROM profiles WHERE speaker_id=?",(speaker_id,)).fetchone()
    db.close()
    photo_url=None
    if prof and prof["photo_b2_key"]:
        try:
            b2=get_b2()
            photo_url=b2.generate_presigned_url("get_object",
                Params={"Bucket":B2_BUCKET_NAME,"Key":prof["photo_b2_key"]},ExpiresIn=3600)
        except Exception as e: print(f"Photo presign: {e}")
    return render_template("view_profile.html",prof=prof,speaker_id=speaker_id,photo_url=photo_url,
                           recording_devices=RECORDING_DEVICE_CHOICES)

# ── Ghost-clip diagnostic & cleanup ───────────────────────────────────────────
def _head_state(b2, key):
    """'present' | 'missing' (definitive 404) | 'unknown' (any other error).
    Only a real 404 counts as missing, so a transient B2 problem never gets a
    clip mislabelled."""
    try:
        b2.head_object(Bucket=B2_BUCKET_NAME, Key=key)
        return "present"
    except Exception as e:
        resp = getattr(e, "response", None)
        if isinstance(resp, dict):
            if resp.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return "missing"
            if str(resp.get("Error", {}).get("Code")) in ("404", "NoSuchKey", "NotFound"):
                return "missing"
        return "unknown"


def _scan_approved_ghosts():
    """Parallel, read-only. Returns (scanned_count, ghosts, unknown), where a ghost
    is an approved clip whose RECORDED b2_key returns a definitive 404.
    One B2 client per worker THREAD (thread-local), not per clip: constructing a
    boto3 client per clip (~10k constructions) is slow, memory-heavy, and client
    construction is not thread-safe — a likely source of mid-scan crashes."""
    db = get_db()
    rows = db.execute(
        "SELECT c.id, c.speaker_id, c.filename, c.b2_key, "
        "       (SELECT COUNT(*) FROM delivery_clips dc WHERE dc.clip_id=c.id) AS in_deliveries "
        "FROM clips c WHERE c.status='approved' AND c.b2_key IS NOT NULL ORDER BY c.id"
    ).fetchall()
    db.close()

    _tl = threading.local()

    def classify(c):
        try:
            b2 = getattr(_tl, "b2", None)
            if b2 is None:
                b2 = _tl.b2 = get_b2()
            return (c, _head_state(b2, c["b2_key"]))
        except Exception:
            return (c, "unknown")

    ghosts, unknown = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for c, state in pool.map(classify, rows):
            if state == "missing":
                ghosts.append(c)
            elif state == "unknown":
                unknown.append(c)
    return (len(rows), ghosts, unknown)


def _run_ghost_scan_job(job_id):
    """Background-thread body for the ghost scan. HEAD-checks every approved clip's key
    against storage with a thread-local B2 client per worker, writes live progress into
    bulk_jobs (done / missing[=ghosts found] / errors[=unknown]) every batch, and on
    completion persists the ghost clip-id list into result_json so the page can render
    results without ever re-scanning. Read-only against storage."""
    db = get_db()
    try:
        rows = db.execute(
            "SELECT id, b2_key FROM clips WHERE status='approved' AND b2_key IS NOT NULL "
            "ORDER BY id").fetchall()
        db.execute("UPDATE bulk_jobs SET total=? WHERE id=?", (len(rows), job_id)); db.commit()
        _tl = threading.local()
        def classify(c):
            try:
                b2 = getattr(_tl, "b2", None)
                if b2 is None:
                    b2 = _tl.b2 = get_b2()
                return (c["id"], _head_state(b2, c["b2_key"]))
            except Exception:
                return (c["id"], "unknown")
        ghost_ids = []; unknown = 0; done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for cid, state in pool.map(classify, rows):
                done += 1
                if state == "missing":
                    ghost_ids.append(cid)
                elif state == "unknown":
                    unknown += 1
                if done % 250 == 0:
                    db.execute("UPDATE bulk_jobs SET done=?, missing=?, errors=? WHERE id=?",
                               (done, len(ghost_ids), unknown, job_id)); db.commit()
        db.execute(
            "UPDATE bulk_jobs SET done=?, missing=?, errors=?, status='done', "
            "  result_json=?, finished_at=datetime('now') WHERE id=?",
            (done, len(ghost_ids), unknown,
             json.dumps({"scanned": len(rows), "ghost_ids": ghost_ids, "unknown": unknown}),
             job_id))
        db.commit()
    except Exception as e:
        import traceback
        print(f"[GHOST SCAN JOB {job_id}] failed:\n" + traceback.format_exc())
        try:
            db.execute("UPDATE bulk_jobs SET status='failed', error=?, finished_at=datetime('now') "
                       "WHERE id=?", (str(e), job_id)); db.commit()
        except Exception:
            pass
    finally:
        db.close()

def _start_ghost_scan_job():
    """Create the ghost-scan job row and spawn the worker thread. Only one background
    job runs at a time (they contend on B2/threads, and a scan running during a bulk
    move could see transient 404s). Returns (job_id, started, running_kind)."""
    db = get_db()
    running = db.execute(
        "SELECT id, kind FROM bulk_jobs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
    if running:
        kind = running["kind"]; rid = running["id"]; db.close()
        return (rid, False, kind)
    cur = db.execute("INSERT INTO bulk_jobs (kind, scope) VALUES ('ghost_scan', 'approved pool')")
    job_id = cur.lastrowid
    db.commit(); db.close()
    threading.Thread(target=_run_ghost_scan_job, args=(job_id,), daemon=True).start()
    return (job_id, True, 'ghost_scan')

@app.route("/admin/ghost_clips/scan", methods=["POST"])
@login_required
@admin_required
def ghost_clips_scan():
    """Start a ghost scan as a background job (the approved pool is too large to check
    inside a single request). Redirects to the live progress page."""
    job_id, started, kind = _start_ghost_scan_job()
    if not started and kind != 'ghost_scan':
        flash("Another background job is running — wait for it to finish, then start the scan.", "info")
        return redirect(url_for("ghost_clips"))
    return redirect(url_for("ghost_scan_progress", job_id=job_id))

@app.route("/admin/ghost_clips/scan/<int:job_id>")
@login_required
@admin_required
def ghost_scan_progress(job_id):
    db = get_db()
    job = db.execute("SELECT * FROM bulk_jobs WHERE id=? AND kind='ghost_scan'", (job_id,)).fetchone()
    db.close()
    if not job:
        abort(404)
    return render_template("ghost_scan_job.html", job=job)


@app.route("/admin/ghost_clips")
@login_required
@admin_required
def ghost_clips():
    """READ-ONLY. Shows the results of the most recent completed ghost scan (persisted
    by the background job) — it does NOT scan on load. From here you start a scan, watch
    its progress, and run cleanup against the saved findings."""
    db = get_db()
    running = db.execute(
        "SELECT id FROM bulk_jobs WHERE kind='ghost_scan' AND status='running' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    last = db.execute(
        "SELECT * FROM bulk_jobs WHERE kind='ghost_scan' AND status='done' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    scanned = unknown = 0; finished_at = None; ghosts = []; affected = []
    if last and last["result_json"]:
        data = json.loads(last["result_json"])
        scanned = data.get("scanned", 0)
        unknown = data.get("unknown", 0)
        finished_at = last["finished_at"]
        gids = data.get("ghost_ids", [])
        if gids:
            qm = ",".join("?" * len(gids))
            ghosts = db.execute(
                f"SELECT c.id, c.speaker_id, c.filename, c.b2_key, c.status, "
                f"  (SELECT COUNT(*) FROM delivery_clips dc WHERE dc.clip_id=c.id) AS in_deliveries "
                f"FROM clips c WHERE c.id IN ({qm}) ORDER BY c.id", gids).fetchall()
            for r in db.execute(
                f"SELECT DISTINCT d.name FROM delivery_clips dc "
                f"JOIN deliveries d ON d.id=dc.delivery_id WHERE dc.clip_id IN ({qm}) "
                f"ORDER BY d.name", gids).fetchall():
                affected.append(r["name"])
    db.close()
    return render_template(
        "ghost_clips.html",
        has_scan=bool(last),
        running_job_id=(running["id"] if running else None),
        scanned=scanned,
        ghosts=ghosts,
        ghost_in_delivery=sum(1 for g in ghosts if g["in_deliveries"]),
        unknown=unknown,
        finished_at=finished_at,
        affected_deliveries=affected,
    )


@app.route("/admin/ghost_clips/reconcile", methods=["POST"])
@login_required
@admin_required
def ghost_clips_reconcile():
    """Reconcile the ghosts found by the most recent completed scan. For each, re-check
    the canonical locations (approved/ → editor_approved/ → pending/): if the audio
    exists somewhere the b2_key was stale → fix the pointer (RECOVER); if it's gone
    everywhere → mark status='missing'. Uses the SAVED ghost list (small) rather than
    re-scanning all approved clips, so it can't time out. Defensive per clip: never
    marks missing a clip whose audio is actually present. Requires typed confirmation."""
    if (request.form.get("confirm") or "").strip().upper() != "CLEANUP":
        flash("Cleanup aborted: confirmation text did not match CLEANUP.", "error")
        return redirect(url_for("ghost_clips"))

    db = get_db()
    last = db.execute(
        "SELECT id, result_json FROM bulk_jobs WHERE kind='ghost_scan' AND status='done' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    if not last or not last["result_json"]:
        db.close()
        flash("No completed scan to reconcile. Run a scan first.", "info")
        return redirect(url_for("ghost_clips"))
    scan_job_id = last["id"]
    gids = json.loads(last["result_json"]).get("ghost_ids", [])
    ghost_rows = []
    if gids:
        qm = ",".join("?" * len(gids))
        ghost_rows = db.execute(
            f"SELECT id, speaker_id, filename, b2_key, status FROM clips "
            f"WHERE id IN ({qm})", gids).fetchall()
    db.close()

    recovered = marked = skipped = 0
    for c in ghost_rows:
        if c["status"] != "approved":
            continue  # already handled since the scan
        spk, fn, cur = c["speaker_id"], c["filename"], c["b2_key"]
        try:
            b2 = get_b2()
            if _head_state(b2, cur) == "present":
                continue  # not actually a ghost anymore
            found = None
            for k in (f"approved/{spk}/{fn}", f"editor_approved/{spk}/{fn}", f"pending/{spk}/{fn}"):
                if k == cur:
                    continue
                if _head_state(b2, k) == "present":
                    found = k
                    break
            d = get_db()
            if found:
                d.execute("UPDATE clips SET b2_key=? WHERE id=? AND status='approved'", (found, c["id"]))
                recovered += 1
                print(f"[GHOST_RECONCILE] recovered clip {c['id']}: {cur} -> {found}")
            else:
                d.execute("UPDATE clips SET status='missing' WHERE id=? AND status='approved'", (c["id"],))
                marked += 1
                print(f"[GHOST_RECONCILE] marked missing clip {c['id']} ({cur}) — not found anywhere")
            d.commit(); d.close()
        except Exception as e:
            skipped += 1
            print(f"[GHOST_RECONCILE] skipped clip {c['id']} ({cur}) due to error: {e}")

    # Supersede the scan so the page prompts for a fresh scan rather than showing
    # ghosts that were just recovered/marked.
    d = get_db()
    d.execute("UPDATE bulk_jobs SET status='superseded' WHERE id=?", (scan_job_id,))
    d.commit(); d.close()

    msg = (f"Ghost cleanup complete. Recovered {recovered} clip(s) with a stale key "
           f"(pointer fixed); marked {marked} clip(s) as missing (audio gone everywhere). "
           f"Run a fresh scan to confirm.")
    if skipped:
        msg += f" {skipped} skipped on a transient error — re-run after scanning."
    flash(msg, "warning" if skipped else "success")
    return redirect(url_for("ghost_clips"))


# ── Stale storage keys ────────────────────────────────────────────────────────
def _stale_key_rows(db):
    """Approved clips whose recorded b2_key is not under approved/. Caused by the old
    bulk-approve helper nooping pending/ sources: the clip was marked approved but its
    audio never moved. Cheap SQL; no B2 calls."""
    return db.execute(
        "SELECT id, speaker_id, filename, b2_key FROM clips "
        "WHERE status='approved' AND b2_key IS NOT NULL AND b2_key NOT LIKE 'approved/%' "
        "ORDER BY speaker_id, id").fetchall()

@app.route("/admin/stale_keys")
@login_required
@admin_required
def stale_keys():
    """READ-ONLY report. Approved clips whose audio still lives outside approved/
    (typically pending/). They deliver fine today, but anything that later touches
    that pending/ file silently turns them into ghosts — move them now."""
    db = get_db()
    rows = _stale_key_rows(db)
    db.close()
    per_spk = {}
    for r in rows:
        per_spk[r["speaker_id"]] = per_spk.get(r["speaker_id"], 0) + 1
    return render_template("stale_keys.html",
                           n=len(rows),
                           per_speaker=sorted(per_spk.items(), key=lambda x: -x[1]),
                           sample=rows[:50])

@app.route("/admin/stale_keys/reconcile", methods=["POST"])
@login_required
@admin_required
def stale_keys_reconcile():
    """Move each stale-keyed approved clip's audio to approved/ and update the key in
    the same statement. Copy-before-delete, idempotent, batched commits, parallel B2.
    Clips whose audio is missing everywhere are left untouched for the ghost tool."""
    if (request.form.get("confirm") or "").strip().upper() != "MOVE":
        flash("Aborted: confirmation text did not match MOVE.", "error")
        return redirect(url_for("stale_keys"))
    db = get_db()
    rows = _stale_key_rows(db)
    b2 = get_b2()
    BATCH = 25; WORKERS = 10
    moved = already = missing_n = errors = 0

    def _work(c):
        try:
            return (c, *_ensure_approved_any(b2, c["b2_key"]))
        except Exception as e:
            print(f"[STALEKEY] B2 error clip {c['id']}: {e}")
            return (c, None, "error")

    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(_work, batch))
        for (c, final_key, st) in results:
            if st in ("moved", "already") and final_key:
                db.execute(
                    "UPDATE clips SET b2_key=? WHERE id=? AND status='approved'",
                    (final_key, c["id"]))
                if st == "moved": moved += 1
                else: already += 1
            elif st == "missing":
                missing_n += 1   # leave as-is; the ghost tool flags + handles these
            elif st == "error":
                errors += 1
        db.commit()
    db.close()
    msg = f"Moved {moved} file(s) to approved/; fixed {already} pointer(s) already there."
    if missing_n: msg += f" {missing_n} had no audio anywhere — run the Ghost clips tool for those."
    if errors: msg += f" {errors} hit storage errors — re-run to retry."
    flash(msg, "success" if not (missing_n or errors) else "error")
    return redirect(url_for("stale_keys"))


# ── Storage key consistency ───────────────────────────────────────────────────
_STATUS_PREFIX = {"pending": "pending/", "editor_approved": "editor_approved/",
                  "approved": "approved/"}

def _key_mismatch_rows(db):
    """Clips whose b2_key prefix disagrees with their status (the 'grey player' state:
    e.g. an editor_approved clip whose pointer still says pending/). Cheap SQL only."""
    return db.execute(
        "SELECT id, speaker_id, filename, status, b2_key FROM clips "
        "WHERE b2_key IS NOT NULL AND ("
        "  (status='pending' AND b2_key NOT LIKE 'pending/%') OR "
        "  (status='editor_approved' AND b2_key NOT LIKE 'editor_approved/%') OR "
        "  (status='approved' AND b2_key NOT LIKE 'approved/%'))"
        "ORDER BY status, speaker_id, id").fetchall()

@app.route("/admin/key_consistency")
@login_required
@admin_required
def key_consistency():
    """READ-ONLY. Lists pointer/status mismatches and live-checks (HEAD) where each
    clip's audio actually exists across the three locations, so the repair is
    fully informed. Approved-status mismatches are the stale-keys tool's job
    (files must MOVE); this page repoints only pending/editor_approved clips."""
    db = get_db()
    rows = _key_mismatch_rows(db)
    db.close()
    b2 = get_b2()
    detailed = []
    for r in rows[:200]:
        tail = f"{r['speaker_id']}/{r['filename']}"
        locs = {}
        for pfx in ("pending/", "editor_approved/", "approved/"):
            try:
                locs[pfx] = _b2_exists(b2, pfx + tail)
            except Exception:
                locs[pfx] = None   # unknown (B2 error)
        try:
            key_ok = _b2_exists(b2, r["b2_key"])
        except Exception:
            key_ok = None
        detailed.append({**dict(r), "locs": locs, "key_ok": key_ok})
    n_repointable = sum(1 for d in detailed if d["status"] in ("pending", "editor_approved"))
    n_approved = sum(1 for r in rows if r["status"] == "approved")
    return render_template("key_consistency.html",
                           n=len(rows), detailed=detailed,
                           n_repointable=n_repointable, n_approved=n_approved)

@app.route("/admin/key_consistency/repoint", methods=["POST"])
@login_required
@admin_required
def key_consistency_repoint():
    """Repair: for each pending/editor_approved clip whose pointer mismatches its status,
    repoint b2_key to a location where the audio VERIFIABLY exists (HEAD-checked).
    Preference order = the location matching the clip's status first. NO files are
    copied, moved, or deleted, and no status changes — pointer updates only.
    Approved-status mismatches are intentionally skipped (stale-keys tool moves those).
    Idempotent; safe to re-run."""
    if (request.form.get("confirm") or "").strip().upper() != "REPOINT":
        flash("Aborted: confirmation text did not match REPOINT.", "error")
        return redirect(url_for("key_consistency"))
    db = get_db()
    rows = [r for r in _key_mismatch_rows(db)
            if r["status"] in ("pending", "editor_approved")]
    b2 = get_b2()
    prefer = {"editor_approved": ("editor_approved/", "pending/"),
              "pending": ("pending/", "editor_approved/")}
    repointed = nowhere = errors = 0
    for r in rows:
        tail = f"{r['speaker_id']}/{r['filename']}"
        try:
            new_key = None
            for pfx in prefer[r["status"]]:
                if _b2_exists(b2, pfx + tail):
                    new_key = pfx + tail
                    break
            if new_key:
                db.execute("UPDATE clips SET b2_key=? WHERE id=? AND status=?",
                           (new_key, r["id"], r["status"]))
                repointed += 1
            else:
                nowhere += 1   # audio in neither non-approved location — leave for ghost handling
        except Exception as e:
            print(f"[KEYFIX] clip {r['id']}: {e}")
            errors += 1
    db.commit(); db.close()
    msg = f"Repointed {repointed} clip(s) to where their audio actually is."
    if nowhere: msg += f" {nowhere} had no audio at pending/ or editor_approved/ — those need re-record or ghost handling."
    if errors: msg += f" {errors} hit storage errors — re-run to retry."
    flash(msg, "success" if not (nowhere or errors) else "error")
    return redirect(url_for("key_consistency"))
# ── Duplicate-clip diagnostic + cleanup ───────────────────────────────────────
# Status priority for choosing which clip to keep when a prompt has several.
_CLIP_STATUS_PRIORITY = {"approved": 0, "editor_approved": 1, "pending": 2,
                         "rejected": 3, "missing": 4}

def _duplicate_clip_groups(db):
    """Return one group per prompt that carries >1 clip row. Each group is a dict:
    {prompt_id, speaker_id, clip_number, text_mn, speech_type, clips:[rows], keep_id}.
    keep_id is the clip to retain: best status, newest id as tiebreak. Pure DB; this
    is the single source of truth shared by the read-only report and the cleanup, so
    the two can never disagree about what would be kept."""
    dup = db.execute(
        "SELECT prompt_id, COUNT(*) AS n FROM clips GROUP BY prompt_id HAVING COUNT(*) > 1"
    ).fetchall()
    groups = []
    if not dup:
        return groups
    ids = [r["prompt_id"] for r in dup]
    qm = ",".join("?" * len(ids))
    rows = db.execute(
        f"SELECT c.id, c.prompt_id, c.speaker_id, c.status, c.b2_key, c.submitted_at, "
        f"       c.compensated, c.editor_paid, p.clip_number, p.text_mn, p.speech_type "
        f"FROM clips c JOIN prompts p ON p.id=c.prompt_id "
        f"WHERE c.prompt_id IN ({qm}) ORDER BY c.prompt_id, c.id", ids).fetchall()
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    for pid, clips in by_prompt.items():
        keep = sorted(
            clips,
            key=lambda c: (_CLIP_STATUS_PRIORITY.get(c["status"], 9), -(c["id"] or 0))
        )[0]
        first = clips[0]
        groups.append({
            "prompt_id": pid,
            "speaker_id": first["speaker_id"],
            "clip_number": first["clip_number"],
            "text_mn": first["text_mn"],
            "speech_type": first["speech_type"],
            "clips": clips,
            "keep_id": keep["id"],
        })
    groups.sort(key=lambda g: len(g["clips"]), reverse=True)
    return groups

@app.route("/admin/duplicate_clips")
@login_required
@admin_required
def duplicate_clips():
    """READ-ONLY diagnostic. Finds prompts that carry more than one clip row. The
    clips table has no one-per-prompt guard and submit_clip historically cleared
    only stale clips, so duplicates can accumulate. Shows the scale and, per prompt,
    which clip a cleanup would keep. Pure DB — no B2 calls; changes nothing."""
    db = get_db()
    groups = _duplicate_clip_groups(db)
    db.close()
    total_clips = sum(len(g["clips"]) for g in groups)
    multi_approved = sum(
        1 for g in groups if sum(1 for c in g["clips"] if c["status"] == "approved") > 1
    )
    return render_template(
        "duplicate_clips.html",
        n_prompts=len(groups),
        total_clips=total_clips,
        redundant=total_clips - len(groups),
        multi_approved=multi_approved,
        groups=groups,
    )

@app.route("/admin/duplicate_clips/reconcile", methods=["POST"])
@login_required
@admin_required
def duplicate_clips_reconcile():
    """Collapse each duplicated prompt down to a single clip — the same one the
    report previews as KEEP. Removes ONLY the redundant DB rows; never deletes a B2
    object (duplicate approved clips share one physical file, so deleting the removed
    row's audio would destroy the kept clip's audio). Any paid status on a removed
    row is carried onto the kept row so no payment record is lost. Typed 'DEDUPE'."""
    if (request.form.get("confirm") or "").strip().upper() != "DEDUPE":
        flash("Cleanup aborted: confirmation text did not match DEDUPE.", "error")
        return redirect(url_for("duplicate_clips"))
    db = get_db()
    groups = _duplicate_clip_groups(db)
    prompts_cleaned = rows_removed = 0
    for g in groups:
        keep_id = g["keep_id"]
        removed = [c for c in g["clips"] if c["id"] != keep_id]
        if not removed:
            continue
        # Preserve paid status: if any removed row was paid, the kept row inherits it.
        if any((c["compensated"] or 0) for c in removed):
            db.execute("UPDATE clips SET compensated=1 WHERE id=?", (keep_id,))
        if any((c["editor_paid"] or 0) for c in removed):
            db.execute("UPDATE clips SET editor_paid=1 WHERE id=?", (keep_id,))
        rid = [c["id"] for c in removed]
        qm = ",".join("?" * len(rid))
        db.execute(f"DELETE FROM clips WHERE id IN ({qm})", rid)
        rows_removed += len(removed)
        prompts_cleaned += 1
    db.commit(); db.close()
    flash(f"De-duplicated {prompts_cleaned} prompt(s); removed {rows_removed} redundant clip row(s). "
          f"No audio files were touched; paid status was preserved on the kept clips.", "success")
    return redirect(url_for("duplicate_clips"))


# ── Troubleshoot hub ──────────────────────────────────────────────────────────
@app.route("/admin/troubleshoot")
@login_required
@admin_required
def troubleshoot():
    """Single entry point for the data-integrity tools. The cheap, pure-SQL checks
    (duplicate clips, duplicate prompts) are computed inline so the page shows health
    at a glance. The ghost scan does live storage checks across every approved clip,
    so it is launched on demand from here rather than run on every page load."""
    db = get_db()
    # Duplicate clips — prompts carrying more than one clip row (shared helper).
    dcg = _duplicate_clip_groups(db)
    dup_clip_prompts = len(dcg)
    dup_clip_redundant = sum(len(g["clips"]) for g in dcg) - dup_clip_prompts
    dup_clip_multi_approved = sum(
        1 for g in dcg if sum(1 for c in g["clips"] if c["status"] == "approved") > 1)
    # Duplicate prompts — same normalized text across prompt rows (dedupe_report's check).
    dt = db.execute(
        "SELECT COUNT(*) AS groups, COALESCE(SUM(n),0) AS total FROM ("
        "  SELECT COUNT(*) AS n FROM prompts "
        "  WHERE text_normalized IS NOT NULL AND text_normalized != '' "
        "  GROUP BY text_normalized HAVING COUNT(*) > 1)"
    ).fetchone()
    dup_text_groups = dt["groups"] or 0
    dup_text_extra = (dt["total"] or 0) - dup_text_groups
    # Ghost scan scope — cheap count only; the scan itself runs on the ghost page.
    approved_count = db.execute(
        "SELECT COUNT(*) AS n FROM clips WHERE status='approved'").fetchone()["n"]
    # Stale storage keys — approved clips whose key isn't under approved/. Cheap SQL.
    stale_count = db.execute(
        "SELECT COUNT(*) AS n FROM clips WHERE status='approved' "
        "AND b2_key IS NOT NULL AND b2_key NOT LIKE 'approved/%'").fetchone()["n"]
    # Pointer/status mismatches for non-approved clips (grey-player state). Cheap SQL.
    mismatch_count = db.execute(
        "SELECT COUNT(*) AS n FROM clips WHERE b2_key IS NOT NULL AND ("
        "  (status='pending' AND b2_key NOT LIKE 'pending/%') OR "
        "  (status='editor_approved' AND b2_key NOT LIKE 'editor_approved/%'))").fetchone()["n"]
    db.close()
    return render_template(
        "troubleshoot.html",
        dup_clip_prompts=dup_clip_prompts,
        dup_clip_redundant=dup_clip_redundant,
        dup_clip_multi_approved=dup_clip_multi_approved,
        dup_text_groups=dup_text_groups,
        dup_text_extra=dup_text_extra,
        approved_count=approved_count,
        stale_count=stale_count,
        mismatch_count=mismatch_count,
    )


# ── Delivery Manager ──────────────────────────────────────────────────────────
# ── Deliveries v2: buyer-aware, build-from-pool model ─────────────────────────
#
# Mental model:
# - Approved clips form a shared pool. The pool isn't depleted by deliveries —
#   a clip can be sold to many buyers independently as fresh inventory.
# - Each buyer has their own delivery history. When admin builds a batch for
#   buyer X, the wizard shows ONLY clips that haven't been delivered to X before.
# - delivery_clips remains the source of truth for "was this clip ever in a
#   delivery to this buyer?"
# - Buyers are deletable (cascade-deletes their deliveries) for testing convenience.

def next_buyer_code():
    """Auto-generate the next BUY_NNN code by scanning existing buyers.
    Mirrors next_speaker_id() — find max numeric suffix, increment."""
    db = get_db()
    rows = db.execute("SELECT code FROM buyers WHERE code LIKE 'BUY_%'").fetchall()
    db.close()
    max_n = 0
    for r in rows:
        try:
            n = int(r["code"].replace("BUY_", ""))
            if n > max_n: max_n = n
        except ValueError:
            continue
    return f"BUY_{max_n + 1:03d}"

def inventory_by_category(buyer_id=None):
    """Return list of {speech_type, subcategory, clip_count, total_seconds}
    rows aggregated from APPROVED clips.

    When buyer_id is provided, excludes clips that buyer has already received
    in any prior delivery. When None, returns the full pool inventory.

    Subcategory defaults to 'general' when empty so the UI shows consistent rows
    instead of NULL/empty cells.
    """
    db = get_db()
    if buyer_id:
        rows = db.execute(
            "SELECT COALESCE(NULLIF(p.subcategory,''),'general') AS subcategory, "
            "       p.speech_type, "
            "       COUNT(c.id) AS clip_count, "
            "       COALESCE(SUM(c.duration_seconds), 0) AS total_seconds "
            "FROM clips c "
            "JOIN prompts p ON p.id = c.prompt_id "
            "WHERE c.status = 'approved' "
            "  AND c.id NOT IN ("
            "    SELECT dc.clip_id FROM delivery_clips dc "
            "    JOIN deliveries d ON d.id = dc.delivery_id "
            "    WHERE d.buyer_id = ?"
            "  ) "
            "GROUP BY p.speech_type, subcategory "
            "ORDER BY p.speech_type, subcategory",
            (buyer_id,)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT COALESCE(NULLIF(p.subcategory,''),'general') AS subcategory, "
            "       p.speech_type, "
            "       COUNT(c.id) AS clip_count, "
            "       COALESCE(SUM(c.duration_seconds), 0) AS total_seconds "
            "FROM clips c "
            "JOIN prompts p ON p.id = c.prompt_id "
            "WHERE c.status = 'approved' "
            "GROUP BY p.speech_type, subcategory "
            "ORDER BY p.speech_type, subcategory"
        ).fetchall()
    db.close()
    return [dict(r) for r in rows]

def inventory_grand_total(buyer_id=None):
    """Return (total_seconds, clip_count) tuple for the approved pool.
    When buyer_id given, excludes clips already delivered to that buyer."""
    db = get_db()
    if buyer_id:
        row = db.execute(
            "SELECT COALESCE(SUM(c.duration_seconds), 0) AS total_seconds, "
            "       COUNT(c.id) AS clip_count "
            "FROM clips c "
            "WHERE c.status = 'approved' "
            "  AND c.id NOT IN ("
            "    SELECT dc.clip_id FROM delivery_clips dc "
            "    JOIN deliveries d ON d.id = dc.delivery_id "
            "    WHERE d.buyer_id = ?"
            "  )",
            (buyer_id,)
        ).fetchone()
    else:
        row = db.execute(
            "SELECT COALESCE(SUM(c.duration_seconds), 0) AS total_seconds, "
            "       COUNT(c.id) AS clip_count "
            "FROM clips c "
            "WHERE c.status = 'approved'"
        ).fetchone()
    db.close()
    return (row["total_seconds"], row["clip_count"])


# ── Delivery zip build worker ────────────────────────────────────────────────
#
# The zip-build runs in a daemon thread spawned by the build-batch route. Admin's
# POST returns immediately; the thread does the work over ~5 minutes and updates
# the deliveries.build_status field as it progresses.
#
# Why a thread and not a queue: simplicity. We don't need cross-process or
# cross-server coordination — Railway runs one Flask process, the work is bounded,
# and a failed build can be retried by the admin via a single button click. If
# Railway restarts mid-build, the in-progress row stays 'building' and admin
# can hit Retry. We treat that as a feature (no silent failures).
#
# What the worker does:
#   1. Re-fetch the delivery + its clips + speaker profiles
#   2. Update build_status='building', progress=0
#   3. Build the zip in memory (no streaming complexity since we're not the
#      HTTP response — we have all the time in the world)
#   4. Upload the zip to B2 at deliveries/zips/<batch_name>.zip
#   5. Update build_status='ready' with zip_b2_key + size + progress=100
#   6. Buyer download routes (via signed URL) become live.
#
# Two cooperation points where we abort cleanly if admin deleted the delivery:
#   - Before building (if row is already gone, do nothing)
#   - After uploading the zip (if row was deleted during the build, delete the
#     orphan zip from B2 so storage doesn't bloat)

def _build_delivery_zip(delivery_id):
    """Build the zip for a single delivery and upload to B2. Called from a
    background thread. All DB / B2 clients are thread-local so this is safe to
    run concurrently with HTTP request handlers."""
    SPEECH_TYPE_FOLDERS = {
        "RS": "read_speech",
        "CS": "conversational",
        "CQ": "commands_queries",
        "ND": "numbers_dates",
        "EE": "emotional_expressive",
    }

    db = get_db()
    delivery = db.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    if not delivery:
        db.close()
        print(f"[BUILD] worker started for delivery {delivery_id} but row no longer exists — aborting")
        return

    try:
        # Move to 'building' state
        db.execute("UPDATE deliveries SET build_status='building', build_progress=0, build_error=NULL WHERE id=?",
                   (delivery_id,))
        db.commit()

        clips = db.execute(
            "SELECT c.*, p.text_mn, p.text_en, p.speech_type, p.subcategory, p.clip_number, "
            "       pr.gender, pr.age, pr.region, pr.dialect, pr.native_language, pr.recording_device "
            "FROM clips c "
            "JOIN prompts p ON p.id=c.prompt_id "
            "LEFT JOIN profiles pr ON pr.speaker_id=c.speaker_id "
            "JOIN delivery_clips dc ON dc.clip_id=c.id "
            "WHERE dc.delivery_id=? ORDER BY c.speaker_id, p.clip_number",
            (delivery_id,)
        ).fetchall()

        speaker_ids = list(dict.fromkeys(c["speaker_id"] for c in clips))
        speaker_profiles = {}
        consent_meta = {}
        for sid in speaker_ids:
            p = db.execute("SELECT * FROM profiles WHERE speaker_id=?", (sid,)).fetchone()
            speaker_profiles[sid] = p
            consent_meta[sid] = db.execute(
                "SELECT agreement_version, text_sha256, affirmed_at FROM consent_records "
                "WHERE speaker_id=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
        db.close()

        # ── CONSENT HARD GATE (defense-in-depth) ────────────────────────────
        # Batch selection already excludes speakers without the current-version
        # consent, but the zip boundary enforces the invariant unconditionally:
        # if ANY clip in this delivery belongs to a speaker who has not affirmed
        # agreement v{CURRENT}, the build FAILS LOUDLY and no zip is produced —
        # never a silently smaller zip. This can only trigger on a batch built
        # before the selection filter existed or on a manually altered row; the
        # remedy is re-affirmation or rebuilding the batch.
        _non_consented = sorted(
            sid for sid in speaker_ids
            if not speaker_profiles.get(sid)
            or ((speaker_profiles[sid]["consent_version"] or "") != CURRENT_CONSENT_VERSION))
        if _non_consented:
            raise RuntimeError(
                f"CONSENT GATE: {len(_non_consented)} speaker(s) in this batch have not "
                f"affirmed consent v{CURRENT_CONSENT_VERSION}: "
                + ", ".join(_non_consented[:10])
                + (" …" if len(_non_consented) > 10 else "")
                + ". No zip was produced. Have them re-affirm, or rebuild the batch.")

        folder = f"MNKH_{delivery['buyer_name'].replace(' ','_')}_{delivery['name'].replace(' ','_')}"

        # Build in memory. For a 330-clip batch with ~600KB WAVs we're talking
        # about ~200MB peak RAM. Railway's free/hobby tier has 512MB-8GB; this
        # is fine. If batches ever cross ~1GB we'll need to write to /tmp first.
        buf = io.BytesIO()
        zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)

        # ── 1. Utterance metadata CSV ────────────────────────────────────
        csv_buf = io.StringIO(); csv_buf.write("\ufeff")
        w = csv.writer(csv_buf)
        w.writerow(["filename","transcription_mn","translation_en","speaker_id",
                    "speech_type","subcategory","duration_seconds","gender","age_band","region",
                    "dialect","native_language","recording_device","recording_environment"])
        for c in clips:
            w.writerow([c["filename"], c["text_mn"], c["text_en"] or "", c["speaker_id"],
                         c["speech_type"], c["subcategory"] or "general",
                         round(c["duration_seconds"] or 0, 2),
                         c["gender"] or "", _age_band(c["age"]), c["region"] or "",
                         c["dialect"] or "Khalkha",
                         c["native_language"] or "Mongolian",
                         c["recording_device"] or "Mobile phone",
                         "reviewed for low background noise"])
        zf.writestr(f"{folder}/metadata/utterance_metadata.csv", csv_buf.getvalue().encode("utf-8"))

        # ── 2. Speaker metadata CSV ──────────────────────────────────────
        spk_buf = io.StringIO(); spk_buf.write("\ufeff")
        sw = csv.writer(spk_buf)
        # Pseudonymous per C.5/C.7: no full_name; exact age becomes an age band.
        sw.writerow(["speaker_id","age_band","gender","region","dialect",
                     "native_language","education_level","recording_device",
                     "clip_count","total_duration_seconds"])
        for sid in speaker_ids:
            p = speaker_profiles.get(sid)
            spk_clips = [c for c in clips if c["speaker_id"] == sid]
            spk_dur = sum(c["duration_seconds"] or 0 for c in spk_clips)
            sw.writerow([sid,
                _age_band(p["age"]) if p else "",
                p["gender"] if p else "",
                p["region"] if p else "",
                p["dialect"] if p else "Khalkha",
                p["native_language"] if p else "Mongolian",
                p["education_level"] if p else "",
                (p["recording_device"] if p else None) or "Mobile phone",
                len(spk_clips),
                round(spk_dur, 2)])
        zf.writestr(f"{folder}/metadata/speaker_metadata.csv", spk_buf.getvalue().encode("utf-8"))

        # ── 3. Consent evidence: pseudonymous CSV + generated certificates (C.5) ──
        # Buyers receive consent-evidence CERTIFICATES, not the speakers' consent PDFs:
        # no full name, no date of birth, no identity-document image leaves the Company.
        # Selection guarantees every delivered speaker affirmed the current agreement
        # version, so every certificate asserts a grant its speaker actually made.
        con_buf = io.StringIO(); con_buf.write("\ufeff")
        cw = csv.writer(con_buf)
        cw.writerow(["speaker_id","consent_status","age_verified","verification_method",
                     "agreement_version","agreement_text_sha256","consent_effective_date",
                     "affirmation_timestamp","certificate_file"])
        for sid in speaker_ids:
            p = speaker_profiles.get(sid)
            cr = consent_meta.get(sid)
            ver = ((p["consent_version"] if p else "") or "")
            ok = bool(p and p["consent_b2_key"] and ver == CURRENT_CONSENT_VERSION)
            cw.writerow([sid,
                f"Confirmed (v{ver} digital consent)" if ok else "NOT CONFIRMED — excluded-state, investigate",
                "18+ (government photo ID)" if ok else "",
                "Digital in-app consent with automated government-ID verification" if ok else "",
                ver,
                (cr["text_sha256"] if cr else CURRENT_CONSENT_TEXT_SHA256) if ok else "",
                (p["consent_uploaded_at"][:10] if p and p["consent_uploaded_at"] else "") if ok else "",
                (cr["affirmed_at"] if cr else (p["consent_affirmed_at"] if p else "")) if ok else "",
                f"certificates/{sid}.pdf" if ok else ""])
        zf.writestr(f"{folder}/consent/consent_records.csv", con_buf.getvalue().encode("utf-8"))

        b2_main = get_b2()
        for sid in speaker_ids:
            p = speaker_profiles.get(sid)
            cr = consent_meta.get(sid)
            if not (p and p["consent_b2_key"]):
                continue
            try:
                cert = {
                    "version": (p["consent_version"] or ""),
                    "text_sha256": (cr["text_sha256"] if cr else CURRENT_CONSENT_TEXT_SHA256),
                    "effective_date": (p["consent_uploaded_at"][:10] if p["consent_uploaded_at"] else ""),
                    "affirmed_at": (cr["affirmed_at"] if cr else (p["consent_affirmed_at"] or "")),
                }
                zf.writestr(f"{folder}/consent/certificates/{sid}.pdf",
                            _build_consent_certificate_pdf(sid, cert))
            except Exception as e:
                print(f"[BUILD] skip certificate {sid}: {e}")

        # Progress: 20% after metadata + consent PDFs done
        _update_progress(delivery_id, 20)

        # ── 4. README ───────────────────────────────────────────────────
        # Pseudonymous roster (C.5/C.7): speaker IDs only — never real names.
        speaker_lines = "\n".join(f"  {sid}" for sid in speaker_ids)
        readme = (f"MNKH Speech Dataset — {delivery['name']}\n"
                  f"{'='*50}\n"
                  f"Buyer:          {delivery['buyer_name']}\n"
                  f"Created:        {delivery['created_at'][:10]}\n"
                  f"Clips:          {delivery['clip_count']}\n"
                  f"Total duration: {fmt_hours(delivery['total_seconds'])}\n"
                  f"Speakers:       {len(speaker_ids)}\n"
                  f"{speaker_lines}\n\n"
                  f"AUDIO FORMAT\n"
                  f"------------\n"
                  f"Format:      WAV PCM\n"
                  f"Sample rate: 48,000 Hz\n"
                  f"Bit depth:   16-bit\n"
                  f"Channels:    Mono\n"
                  f"Max length:  60 seconds per clip\n\n"
                  f"LANGUAGE\n"
                  f"--------\n"
                  f"Language:  Mongolian\n"
                  f"Dialect:   Khalkha (Standard Mongolian)\n"
                  f"Script:    Cyrillic\n\n"
                  f"QUALITY\n"
                  f"-------\n"
                  f"All clips: 100% real human voices — no AI/synthesized audio\n"
                  f"All clips: Human-reviewed by editor at least once\n"
                  f"Many clips: Human-reviewed twice (editor + admin)\n\n"
                  f"FOLDER STRUCTURE\n"
                  f"----------------\n"
                  f"audio/<speech_type>/<subcategory>/<speaker>/<filename>.wav\n"
                  f"metadata/utterance_metadata.csv — per-clip metadata\n"
                  f"metadata/speaker_metadata.csv   — per-speaker profile\n"
                  f"consent/consent_records.csv     — per-speaker consent evidence (pseudonymous)\n"
                  f"consent/certificates/           — per-speaker CONSENT EVIDENCE CERTIFICATES: verified 18+ status, verification method, agreement version + SHA-256 of the governing consent text, affirmation record, and the speaker's rights grant (incl. AI/ML training and sublicensing) with the binding prohibited-uses flow-down. No speaker names, birth dates, or identity documents are included; the Company retains full ID-verified consent records internally and can produce them under lawful audit.\n")
        zf.writestr(f"{folder}/README.txt", readme)

        # ── 5. Audio files — parallel B2 fetches ────────────────────────
        # Same parallelism trick as before: 12 concurrent fetches.
        # Critical: check periodically that the delivery still exists, so we don't
        # waste 5 minutes building something admin already deleted.
        fetch_clips = [c for c in clips if c["b2_key"]]

        def fetch_one(idx_clip):
            idx, c = idx_clip
            try:
                b2_local = get_b2()
                obj = b2_local.get_object(Bucket=B2_BUCKET_NAME, Key=c["b2_key"])
                return (idx, c, obj["Body"].read(), None)
            except Exception as e:
                return (idx, c, None, str(e))

        indexed = list(enumerate(fetch_clips))
        completed = 0
        total_to_fetch = len(indexed)
        # Progress range during audio fetching: 20 -> 95
        progress_start = 20
        progress_end = 95
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for fut in concurrent.futures.as_completed(
                {pool.submit(fetch_one, ic): ic[0] for ic in indexed}
            ):
                idx, c, audio_bytes, err = fut.result()
                completed += 1
                if audio_bytes is None:
                    print(f"[BUILD] skip {c['filename']}: {err}")
                else:
                    spk = c["speaker_id"].replace("MNKH_","") if c["speaker_id"].startswith("MNKH_") else c["speaker_id"]
                    stype_folder = SPEECH_TYPE_FOLDERS.get(c["speech_type"], c["speech_type"].lower())
                    subcat = (c["subcategory"] or "general").strip().lower().replace(" ","_") or "general"
                    path = f"{folder}/audio/{stype_folder}/{subcat}/{spk}/{c['filename']}"
                    zf.writestr(path, audio_bytes)
                # Update progress every 10 clips (or last clip)
                if completed % 10 == 0 or completed == total_to_fetch:
                    pct = progress_start + int((progress_end - progress_start) * completed / max(total_to_fetch, 1))
                    _update_progress(delivery_id, pct)
                    # Also check if admin deleted us — if so, abort the build
                    if not _delivery_still_exists(delivery_id):
                        print(f"[BUILD] delivery {delivery_id} was deleted mid-build — aborting")
                        zf.close()
                        return  # nothing to upload; no zip in B2 yet

        zf.close()
        zip_bytes = buf.getvalue()
        zip_size = len(zip_bytes)
        _update_progress(delivery_id, 95)

        # ── 6. Upload to B2 ─────────────────────────────────────────────
        zip_b2_key = f"deliveries/zips/{delivery['name']}.zip"
        b2_main.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=zip_b2_key,
            Body=zip_bytes,
            ContentType="application/zip",
        )

        # ── 7. Mark ready ───────────────────────────────────────────────
        # First check if the delivery still exists (admin may have deleted after
        # we uploaded the zip — racing). If gone, delete our orphan zip.
        db2 = get_db()
        still_there = db2.execute("SELECT 1 FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        if not still_there:
            db2.close()
            print(f"[BUILD] delivery {delivery_id} deleted just before final mark-ready; cleaning up orphan zip {zip_b2_key}")
            try:
                b2_main.delete_object(Bucket=B2_BUCKET_NAME, Key=zip_b2_key)
            except Exception as e:
                print(f"[BUILD] orphan zip delete failed: {e}")
            return
        db2.execute(
            "UPDATE deliveries SET build_status='ready', build_progress=100, "
            "                      zip_b2_key=?, zip_size_bytes=? "
            "WHERE id=?",
            (zip_b2_key, zip_size, delivery_id)
        )
        db2.commit(); db2.close()
        print(f"[BUILD] delivery {delivery_id} built: {zip_size:,} bytes uploaded to {zip_b2_key}")

    except Exception as e:
        # Capture error so admin can see it on the dashboard and retry
        err = f"{type(e).__name__}: {e}"
        print(f"[BUILD] delivery {delivery_id} FAILED: {err}")
        try:
            db_err = get_db()
            db_err.execute(
                "UPDATE deliveries SET build_status='failed', build_error=? WHERE id=?",
                (err[:500], delivery_id)
            )
            db_err.commit(); db_err.close()
        except Exception as e2:
            print(f"[BUILD] could not record failure to DB: {e2}")


def _update_progress(delivery_id, pct):
    """Set build_progress on a delivery from inside the worker thread.
    Swallows errors — progress is purely cosmetic, must not crash the build."""
    try:
        db = get_db()
        db.execute("UPDATE deliveries SET build_progress=? WHERE id=?", (pct, delivery_id))
        db.commit(); db.close()
    except Exception as e:
        print(f"[BUILD] progress update failed for {delivery_id}: {e}")


def _delivery_still_exists(delivery_id):
    """Check from inside the worker whether admin deleted us. Used to abort
    cleanly during long-running builds."""
    try:
        db = get_db()
        row = db.execute("SELECT 1 FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        db.close()
        return row is not None
    except Exception:
        return True  # err on side of continuing — failed check shouldn't kill the build


def _spawn_build_worker(delivery_id):
    """Start the background build thread. Daemon so it doesn't keep the process
    alive past app shutdown. Each thread has its own DB + B2 client."""
    t = threading.Thread(target=_build_delivery_zip, args=(delivery_id,), daemon=True)
    t.start()
    return t


@app.route("/admin/deliveries")
@login_required
@admin_required
def deliveries():
    """Master deliveries dashboard:
      - Grand total inventory (all approved clips)
      - Per-category breakdown (so admin knows what's on the shelf)
      - Buyers list (with summary of past deliveries)
      - Recent deliveries list (downloadable)
    """
    total_sec, total_clips = inventory_grand_total(buyer_id=None)
    categories = inventory_by_category(buyer_id=None)

    db = get_db()
    buyers = db.execute(
        "SELECT b.*, "
        "       (SELECT COUNT(*) FROM deliveries d WHERE d.buyer_id = b.id) AS batch_count, "
        "       (SELECT COALESCE(SUM(d.total_seconds),0) FROM deliveries d WHERE d.buyer_id = b.id) AS total_sent_seconds "
        "FROM buyers b "
        "ORDER BY b.created_at DESC"
    ).fetchall()
    recent_deliveries = db.execute(
        "SELECT d.*, b.code AS buyer_code, b.name AS buyer_display_name "
        "FROM deliveries d "
        "LEFT JOIN buyers b ON b.id = d.buyer_id "
        "ORDER BY d.created_at DESC LIMIT 50"
    ).fetchall()
    db.close()

    return render_template(
        "deliveries.html",
        total_sec=total_sec,
        total_clips=total_clips,
        categories=categories,
        buyers=buyers,
        recent_deliveries=recent_deliveries,
        fmt_hours=fmt_hours,
        public_base_url=PUBLIC_BASE_URL,
    )

@app.route("/admin/new_buyer", methods=["POST"])
@login_required
@admin_required
def new_buyer():
    name = request.form.get("name","").strip()
    contact = request.form.get("contact","").strip() or None
    notes = request.form.get("notes","").strip() or None
    if not name:
        flash("Buyer name is required.","error")
        return redirect(url_for("deliveries"))
    code = next_buyer_code()
    db = get_db()
    db.execute("INSERT INTO buyers (code, name, contact, notes) VALUES (?,?,?,?)",
               (code, name, contact, notes))
    db.commit(); db.close()
    print(f"[BUYER] admin created buyer {code} ({name}) at {datetime.datetime.now().isoformat()}")
    flash(f"Created buyer {code}: {name}", "success")
    return redirect(url_for("deliveries"))

@app.route("/admin/delete_buyer/<int:buyer_id>", methods=["POST"])
@login_required
@admin_required
def delete_buyer(buyer_id):
    """Cascade-delete a buyer: removes their deliveries and the delivery_clips
    bridge rows along with the buyer themself. Audit log captures the wipe."""
    db = get_db()
    buyer = db.execute("SELECT code, name FROM buyers WHERE id=?", (buyer_id,)).fetchone()
    if not buyer:
        db.close(); abort(404)
    # Collect this buyer's deliveries + their B2 zip keys so we can clean both up
    deliv_rows = db.execute(
        "SELECT id, zip_b2_key FROM deliveries WHERE buyer_id=?", (buyer_id,)
    ).fetchall()
    deliv_ids = [r["id"] for r in deliv_rows]
    b2_keys_to_delete = [r["zip_b2_key"] for r in deliv_rows if r["zip_b2_key"]]
    n_clips = 0
    for did in deliv_ids:
        n_clips += db.execute("SELECT COUNT(*) FROM delivery_clips WHERE delivery_id=?", (did,)).fetchone()[0]
        db.execute("DELETE FROM delivery_clips WHERE delivery_id=?", (did,))
    db.execute("DELETE FROM deliveries WHERE buyer_id=?", (buyer_id,))
    db.execute("DELETE FROM buyers WHERE id=?", (buyer_id,))
    db.commit(); db.close()
    # Best-effort B2 zip cleanup. Orphans are logged but don't block.
    b2_status = []
    if b2_keys_to_delete:
        try:
            b2 = get_b2()
            for key in b2_keys_to_delete:
                try:
                    b2.delete_object(Bucket=B2_BUCKET_NAME, Key=key)
                    b2_status.append(f"deleted {key}")
                except Exception as e:
                    b2_status.append(f"ORPHAN {key}: {e}")
        except Exception as e:
            b2_status.append(f"B2 client error: {e}")
    print(f"[BUYER] admin deleted buyer {buyer['code']} ({buyer['name']}) "
          f"+ {len(deliv_ids)} deliveries + {n_clips} bridge rows + "
          f"{len(b2_keys_to_delete)} B2 zip(s): {'; '.join(b2_status) if b2_status else 'no zips'} "
          f"at {datetime.datetime.now().isoformat()}")
    flash(f"Deleted {buyer['code']} ({buyer['name']}) and all their deliveries.","success")
    return redirect(url_for("deliveries"))

@app.route("/admin/build_batch")
@login_required
@admin_required
def build_batch():
    """Show the build-batch wizard for a specific buyer.
    Renders the available-for-this-buyer pool so admin can spec a delivery."""
    buyer_id_raw = request.args.get("buyer_id","").strip()
    try:
        buyer_id = int(buyer_id_raw)
    except (ValueError, TypeError):
        flash("Please pick a buyer first.","error")
        return redirect(url_for("deliveries"))
    db = get_db()
    buyer = db.execute("SELECT * FROM buyers WHERE id=?", (buyer_id,)).fetchone()
    db.close()
    if not buyer:
        flash("Buyer not found.","error")
        return redirect(url_for("deliveries"))

    available = inventory_by_category(buyer_id=buyer_id)
    available_total_sec, available_total_clips = inventory_grand_total(buyer_id=buyer_id)

    # Pull the "already sent" per-category so admin can see history alongside availability
    db = get_db()
    already_sent_rows = db.execute(
        "SELECT COALESCE(NULLIF(p.subcategory,''),'general') AS subcategory, "
        "       p.speech_type, "
        "       COUNT(c.id) AS clip_count, "
        "       COALESCE(SUM(c.duration_seconds), 0) AS total_seconds "
        "FROM clips c "
        "JOIN prompts p ON p.id = c.prompt_id "
        "JOIN delivery_clips dc ON dc.clip_id = c.id "
        "JOIN deliveries d ON d.id = dc.delivery_id "
        "WHERE d.buyer_id = ? "
        "GROUP BY p.speech_type, subcategory "
        "ORDER BY p.speech_type, subcategory",
        (buyer_id,)
    ).fetchall()
    db.close()
    sent_lookup = {(r["speech_type"], r["subcategory"]): {
        "clip_count": r["clip_count"], "total_seconds": r["total_seconds"]
    } for r in already_sent_rows}

    return render_template(
        "build_batch.html",
        buyer=buyer,
        available=available,
        available_total_sec=available_total_sec,
        available_total_clips=available_total_clips,
        sent_lookup=sent_lookup,
        fmt_hours=fmt_hours,
    )

@app.route("/admin/build_batch", methods=["POST"])
@login_required
@admin_required
def build_batch_submit():
    """Create a delivery for this buyer matching the requested spec.

    Form fields:
      buyer_id: int
      name (optional): override the auto-generated batch name
      hours_<speech_type>_<subcategory>: float — requested hours per category

    For each requested category, pick approved clips (oldest first, FIFO) excluding
    any already delivered to this buyer, accumulating until target seconds reached.
    The final batch duration may slightly exceed the request (last clip pushes over).
    """
    buyer_id_raw = request.form.get("buyer_id","").strip()
    try:
        buyer_id = int(buyer_id_raw)
    except (ValueError, TypeError):
        flash("Invalid buyer.","error")
        return redirect(url_for("deliveries"))
    db = get_db()
    buyer = db.execute("SELECT * FROM buyers WHERE id=?", (buyer_id,)).fetchone()
    if not buyer:
        db.close()
        flash("Buyer not found.","error")
        return redirect(url_for("deliveries"))

    # Parse the form: every field starting with "minutes_" maps to a (speech_type, subcategory).
    # Subcategory is encoded after the speech type so we can roundtrip exactly.
    # Format: minutes_<SPEECH_TYPE>__<subcategory>
    # The double underscore separator avoids collisions if subcategory contains underscores.
    # Minimum unit is 1 minute (integer) — fractions not allowed by design.
    requested = {}  # (speech_type, subcategory) -> seconds
    for key, val in request.form.items():
        if not key.startswith("minutes_"): continue
        body = key[len("minutes_"):]
        if "__" not in body: continue
        st, sub = body.split("__", 1)
        try:
            # int() rejects '0.5' but accepts '0'. Float-then-int would silently
            # truncate fractional input which we'd rather treat as a form error.
            minutes = int(val)
        except (ValueError, TypeError):
            continue
        if minutes <= 0:
            continue
        seconds = minutes * 60
        requested[(st, sub)] = seconds

    if not requested:
        db.close()
        flash("Specify at least one category and a positive number of minutes.","error")
        return redirect(url_for("build_batch", buyer_id=buyer_id))

    # Now pick clips per category. For each requested (st, sub), pull approved clips
    # NOT already delivered to this buyer, ordered by created_at ASC (FIFO),
    # accumulating until duration >= target.
    selected_clip_ids = []
    selected_total_sec = 0.0
    insufficient = []  # categories where we could not meet the target
    for (st, sub), target_sec in requested.items():
        # Subcategory in DB can be NULL/empty for "general". Translate accordingly.
        sub_filter_sql = ("AND COALESCE(NULLIF(p.subcategory,''),'general') = ?")
        rows = db.execute(
            "SELECT c.id, c.duration_seconds "
            "FROM clips c "
            "JOIN prompts p ON p.id = c.prompt_id "
            # Only clips whose speaker has affirmed the CURRENT agreement version are
            # deliverable: the certificate asserts the v2.0 Part B grant, so shipping a
            # clip from a non-re-affirmed speaker would assert a grant that speaker
            # hasn't made.
            "JOIN profiles pr ON pr.speaker_id = c.speaker_id "
            "WHERE c.status = 'approved' "
            "  AND pr.consent_version = ? "
            f"  AND p.speech_type = ? "
            f"  {sub_filter_sql} "
            "  AND c.id NOT IN ("
            "    SELECT dc.clip_id FROM delivery_clips dc "
            "    JOIN deliveries d ON d.id = dc.delivery_id "
            "    WHERE d.buyer_id = ?"
            "  ) "
            "ORDER BY c.submitted_at ASC, c.id ASC",
            (CURRENT_CONSENT_VERSION, st, sub, buyer_id)
        ).fetchall()
        accumulated = 0.0
        picked_here = []
        for r in rows:
            picked_here.append(r["id"])
            accumulated += (r["duration_seconds"] or 0)
            if accumulated >= target_sec:
                break
        if accumulated < target_sec:
            insufficient.append((st, sub, int(target_sec/60), accumulated/60))
        selected_clip_ids.extend(picked_here)
        selected_total_sec += accumulated

    if not selected_clip_ids:
        db.close()
        flash("No clips available to deliver under the requested spec.","error")
        return redirect(url_for("build_batch", buyer_id=buyer_id))

    # Auto-generate a batch name. Admin can rename later if they want.
    today = datetime.date.today().isoformat()
    # Find next sequence number for this buyer/day
    existing = db.execute(
        "SELECT name FROM deliveries WHERE buyer_id=? AND name LIKE ?",
        (buyer_id, f"{buyer['code']}_{today}_%")
    ).fetchall()
    seq = len(existing) + 1
    auto_name = f"{buyer['code']}_{today}_{seq:03d}"
    name = (request.form.get("name","").strip() or auto_name)

    # Generate the buyer-facing download token. 12 hex chars = 48 bits of entropy.
    # Unguessable in practice; not enumerable from short URL.
    download_token = secrets.token_hex(6)

    db.execute(
        "INSERT INTO deliveries (name, buyer_name, buyer_id, clip_count, total_seconds, "
        "                        build_status, build_progress, download_token) "
        "VALUES (?, ?, ?, ?, ?, 'pending', 0, ?)",
        (name, buyer["name"], buyer_id, len(selected_clip_ids), selected_total_sec, download_token)
    )
    delivery_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    for cid in selected_clip_ids:
        db.execute("INSERT INTO delivery_clips (delivery_id, clip_id) VALUES (?, ?)",
                   (delivery_id, cid))
    excluded_v2 = db.execute(
        "SELECT COUNT(*) FROM clips c JOIN profiles pr ON pr.speaker_id=c.speaker_id "
        "WHERE c.status='approved' AND COALESCE(pr.consent_version,'') <> ?",
        (CURRENT_CONSENT_VERSION,)).fetchone()[0]
    db.commit(); db.close()
    if excluded_v2:
        flash(f"Note: {excluded_v2} approved clip(s) were not eligible because their speaker "
              f"has not yet re-affirmed consent v{CURRENT_CONSENT_VERSION}. They become "
              f"deliverable automatically once the speaker re-affirms.", "info")
    print(f"[DELIVERY] admin built batch {name} for buyer {buyer['code']} ({buyer['name']}): "
          f"{len(selected_clip_ids)} clips, {selected_total_sec:.1f}s, "
          f"insufficient_categories={len(insufficient)}, token={download_token} "
          f"at {datetime.datetime.now().isoformat()}")

    # Spawn the background zip-build worker. POST returns immediately so admin's
    # UI doesn't hang for the 5-minute build — they see the new batch on the
    # dashboard with status "Building..." and can keep working.
    _spawn_build_worker(delivery_id)

    msg = f"Batch '{name}' queued — building in background. Check back in a few minutes."
    if insufficient:
        details = "; ".join(f"{st}/{sub} requested {req}min got {got:.1f}min"
                           for st, sub, req, got in insufficient)
        msg += f" Note: some categories had less than requested ({details})."
    flash(msg, "success")
    return redirect(url_for("deliveries"))

@app.route("/admin/delete_delivery/<int:delivery_id>", methods=["POST"])
@login_required
@admin_required
def delete_delivery(delivery_id):
    """Delete a delivery batch. Cascades to delivery_clips bridge rows AND the
    pre-built zip on B2. The actual approved clips remain in the pool, available
    for future deliveries (including re-delivery to the same buyer since the
    lock is gone too).

    On B2-delete failure we log the orphan key and continue — better to let
    admin's intent succeed (delete from dashboard) than to block on a transient
    storage issue."""
    db = get_db()
    delivery = db.execute("SELECT name, buyer_name, zip_b2_key FROM deliveries WHERE id=?",
                          (delivery_id,)).fetchone()
    if not delivery:
        db.close(); abort(404)
    n_clips = db.execute("SELECT COUNT(*) FROM delivery_clips WHERE delivery_id=?", (delivery_id,)).fetchone()[0]
    db.execute("DELETE FROM delivery_clips WHERE delivery_id=?", (delivery_id,))
    db.execute("DELETE FROM deliveries WHERE id=?", (delivery_id,))
    db.commit(); db.close()
    # Best-effort B2 cleanup. If the zip never finished building, zip_b2_key is NULL.
    b2_delete_status = "n/a (no zip yet)"
    if delivery["zip_b2_key"]:
        try:
            b2 = get_b2()
            b2.delete_object(Bucket=B2_BUCKET_NAME, Key=delivery["zip_b2_key"])
            b2_delete_status = "deleted"
        except Exception as e:
            b2_delete_status = f"ORPHAN: {delivery['zip_b2_key']} ({e})"
    print(f"[DELIVERY] admin deleted batch {delivery['name']} (buyer={delivery['buyer_name']}, "
          f"clips={n_clips}, b2_zip={b2_delete_status}) at {datetime.datetime.now().isoformat()}")
    flash(f"Deleted batch '{delivery['name']}' ({n_clips} clips).", "success")
    return redirect(url_for("deliveries"))

@app.route("/admin/deliveries/download/<int:delivery_id>")
@login_required
@admin_required
def download_delivery(delivery_id):
    """Admin-facing download. The zip was pre-built when the batch was created
    and lives on B2. We just generate a signed URL and redirect."""
    db = get_db()
    d = db.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    db.close()
    if not d:
        abort(404)
    if d["build_status"] != "ready" or not d["zip_b2_key"]:
        flash(f"Batch zip not ready yet (status: {d['build_status']}).", "error")
        return redirect(url_for("deliveries"))
    # Generate a short-lived (1 hour) signed URL for admin's own download.
    # Buyer-facing path uses a separate, longer-lived URL.
    try:
        b2 = get_b2()
        url = b2.generate_presigned_url(
            "get_object",
            Params={"Bucket": B2_BUCKET_NAME, "Key": d["zip_b2_key"]},
            ExpiresIn=3600,
        )
        return redirect(url, code=302)
    except Exception as e:
        print(f"[DELIVERY] generate_presigned_url failed for delivery {delivery_id}: {e}")
        flash("Could not generate download URL. Check Railway logs.", "error")
        return redirect(url_for("deliveries"))


@app.route("/admin/retry_build/<int:delivery_id>", methods=["POST"])
@login_required
@admin_required
def retry_build(delivery_id):
    """Manually re-queue a build for a delivery that failed or got stuck.
    Wipes any previous zip on B2 (in case a partial upload happened) and
    fires off a fresh worker thread."""
    db = get_db()
    d = db.execute("SELECT name, zip_b2_key FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    if not d:
        db.close(); abort(404)
    # Best-effort wipe of any old zip — fresh start
    if d["zip_b2_key"]:
        try:
            b2 = get_b2()
            b2.delete_object(Bucket=B2_BUCKET_NAME, Key=d["zip_b2_key"])
        except Exception as e:
            print(f"[BUILD] retry: failed to clean old zip for {delivery_id}: {e}")
    db.execute(
        "UPDATE deliveries SET build_status='pending', build_progress=0, "
        "                      build_error=NULL, zip_b2_key=NULL, zip_size_bytes=NULL "
        "WHERE id=?", (delivery_id,)
    )
    db.commit(); db.close()
    _spawn_build_worker(delivery_id)
    print(f"[BUILD] admin re-queued build for delivery {delivery_id} ({d['name']})")
    flash(f"Build re-queued for '{d['name']}'.", "success")
    return redirect(url_for("deliveries"))


@app.route("/admin/reset_token/<int:delivery_id>", methods=["POST"])
@login_required
@admin_required
def reset_token(delivery_id):
    """Generate a new download token, invalidating the old one. Use when admin
    wants to revoke an old link or after sending the wrong link to someone."""
    db = get_db()
    d = db.execute("SELECT name FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    if not d:
        db.close(); abort(404)
    new_token = secrets.token_hex(6)
    db.execute("UPDATE deliveries SET download_token=? WHERE id=?", (new_token, delivery_id))
    db.commit(); db.close()
    print(f"[DELIVERY] admin reset token for delivery {delivery_id} ({d['name']}): new token = {new_token}")
    flash(f"New link generated for '{d['name']}'. Old link is now invalid.", "success")
    return redirect(url_for("deliveries"))


# ── Buyer-facing routes (no authentication; token-gated) ─────────────────────
#
# Lives at /d/<token> on whatever domain Flask is serving. In production this
# should be reached at https://mongoliandata.com/d/<token> via custom domain
# pointing at Railway; the code is host-agnostic so the same routes work on
# the admin domain too (useful for testing before DNS is configured).

@app.route("/d/<token>")
def buyer_landing(token):
    """Public landing page. Shows the dataset's data sheet and a download button.
    No login required; the token is the access credential."""
    db = get_db()
    d = db.execute(
        "SELECT d.*, b.code AS buyer_code, b.name AS buyer_real_name "
        "FROM deliveries d LEFT JOIN buyers b ON b.id = d.buyer_id "
        "WHERE d.download_token=?", (token,)
    ).fetchone()
    if not d:
        db.close()
        return render_template("buyer_not_found.html"), 404

    # Gather per-category breakdown for the data sheet
    breakdown = db.execute(
        "SELECT COALESCE(NULLIF(p.subcategory,''),'general') AS subcategory, "
        "       p.speech_type, "
        "       COUNT(c.id) AS clip_count, "
        "       COALESCE(SUM(c.duration_seconds), 0) AS total_seconds "
        "FROM clips c "
        "JOIN prompts p ON p.id=c.prompt_id "
        "JOIN delivery_clips dc ON dc.clip_id=c.id "
        "WHERE dc.delivery_id=? "
        "GROUP BY p.speech_type, subcategory "
        "ORDER BY p.speech_type, subcategory",
        (d["id"],)
    ).fetchall()

    # Speaker demographics aggregate
    speakers = db.execute(
        "SELECT DISTINCT c.speaker_id, pr.gender, pr.age, pr.region, pr.dialect, pr.native_language, "
        "       pr.consent_method, pr.consent_b2_key "
        "FROM clips c "
        "LEFT JOIN profiles pr ON pr.speaker_id=c.speaker_id "
        "JOIN delivery_clips dc ON dc.clip_id=c.id "
        "WHERE dc.delivery_id=?",
        (d["id"],)
    ).fetchall()
    db.close()

    # Aggregate demographics for display
    n_speakers = len(speakers)
    genders = {}
    regions = {}
    ages = []
    for s in speakers:
        if s["gender"]:
            genders[s["gender"]] = genders.get(s["gender"], 0) + 1
        if s["region"]:
            regions[s["region"]] = regions.get(s["region"], 0) + 1
        if s["age"]:
            try:
                ages.append(int(s["age"]))
            except (ValueError, TypeError):
                pass
    age_range = (min(ages), max(ages)) if ages else (None, None)

    # Consent provenance statement — derived from the actual consent_method of the
    # speakers in THIS delivery, so the buyer-facing claim can never overstate.
    # 'digital' = the in-app ID flow; a document with no recorded method = legacy
    # upload; no document = no consent on file (surfaced honestly, ties to the
    # still-open question of whether build_batch should gate on consent).
    n_digital = n_legacy = n_no_consent = 0
    for s in speakers:
        if s["consent_b2_key"]:
            if (s["consent_method"] or "").strip().lower() == "digital":
                n_digital += 1
            else:
                n_legacy += 1
        else:
            n_no_consent += 1
    def _sp(n):
        return "speaker" if n == 1 else "speakers"
    _parts = []
    if n_digital and not n_legacy:
        _lead = "The speaker" if n_digital == 1 else f"All {n_digital} speakers"
        _parts.append(f"{_lead} completed in-app digital consent (agreement v{CURRENT_CONSENT_VERSION}), "
                      f"age-verified 18+ against a government-issued photo ID. Pseudonymous "
                      f"consent-evidence certificates are included; no identity documents or "
                      f"personal identifiers ship with the data.")
    elif n_legacy and not n_digital:
        _lead = "The speaker" if n_legacy == 1 else f"All {n_legacy} speakers"
        _parts.append(f"{_lead} provided documented consent authorizing commercial use "
                      f"of their voice.")
    elif n_digital and n_legacy:
        _parts.append(f"{n_digital} of {n_digital + n_legacy} speakers completed in-app digital "
                      f"consent verified against a government-issued photo ID; the remaining "
                      f"{n_legacy} provided documented consent via an earlier consent flow.")
    if n_no_consent:
        _verb = "does" if n_no_consent == 1 else "do"
        _parts.append(f"{n_no_consent} {_sp(n_no_consent)} {_verb} not yet have consent "
                      f"documentation on file.")
    consent_statement = " ".join(_parts) if _parts else \
        "Per-speaker consent documentation is included in the delivery."

    # Pick a sample clip. Prefer one whose prompt actually has an English
    # translation and is recent (more likely a complete, valid recording than the
    # earliest test clip), falling back to the most recent clip otherwise. The
    # presign mirrors the editor/admin review players exactly (they play reliably);
    # we deliberately do NOT pass a response-content-type override, which B2 may
    # reject on a presigned URL.
    sample_audio_url = None
    sample_clip = None
    if d["build_status"] == "ready":
        try:
            db2 = get_db()
            sample_row = db2.execute(
                "SELECT c.*, p.text_mn, p.text_en FROM clips c "
                "JOIN prompts p ON p.id=c.prompt_id "
                "JOIN delivery_clips dc ON dc.clip_id=c.id "
                "WHERE dc.delivery_id=? AND c.b2_key IS NOT NULL "
                "ORDER BY (CASE WHEN p.text_en IS NOT NULL AND TRIM(p.text_en) <> '' "
                "          THEN 0 ELSE 1 END), c.id DESC "
                "LIMIT 1",
                (d["id"],)
            ).fetchone()
            db2.close()
            if sample_row:
                b2 = get_b2()
                sample_audio_url = b2.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": B2_BUCKET_NAME, "Key": sample_row["b2_key"]},
                    ExpiresIn=3600,
                )
                sample_clip = sample_row
        except Exception as e:
            print(f"[BUYER] sample audio URL generation failed: {e}")

    def _fmt_mb(b):
        if not b: return "—"
        mb = b / (1024 * 1024)
        if mb >= 1000:
            return f"{mb/1024:.1f} GB"
        return f"{mb:.0f} MB"

    return render_template(
        "buyer_landing.html",
        delivery=d,
        breakdown=breakdown,
        n_speakers=n_speakers,
        genders=genders,
        regions=regions,
        age_range=age_range,
        sample_audio_url=sample_audio_url,
        sample_clip=sample_clip,
        consent_statement=consent_statement,
        fmt_hours=fmt_hours,
        fmt_mb=_fmt_mb,
    )


@app.route("/d/<token>/download")
def buyer_download(token):
    """Buyer's actual download trigger. Looks up the delivery by token,
    generates a 7-day signed B2 URL, and 302-redirects. The browser
    downloads directly from B2's CDN — Flask is out of the loop in
    milliseconds."""
    db = get_db()
    d = db.execute(
        "SELECT id, name, build_status, zip_b2_key FROM deliveries WHERE download_token=?",
        (token,)
    ).fetchone()
    db.close()
    if not d:
        abort(404)
    if d["build_status"] != "ready" or not d["zip_b2_key"]:
        # Still building or failed — send buyer back to landing page
        return redirect(url_for("buyer_landing", token=token))
    try:
        b2 = get_b2()
        url = b2.generate_presigned_url(
            "get_object",
            Params={"Bucket": B2_BUCKET_NAME, "Key": d["zip_b2_key"]},
            ExpiresIn=7 * 24 * 3600,  # 7 days
        )
        print(f"[BUYER] download initiated: token={token}, delivery={d['name']}")
        return redirect(url, code=302)
    except Exception as e:
        print(f"[BUYER] signed URL generation failed: {e}")
        abort(500)


# ── Payroll export ─────────────────────────────────────────────────────────────
@app.route("/admin/export/manifest.csv")
@login_required
@admin_required
def training_data_export():
    """Training Data Export (INTERNAL — distinct from buyer Deliveries): streams
    manifest.csv with exactly three columns — file (full B2 object key), transcript
    (Mongolian text_mn only), speaker_id — covering every Final-Review-approved clip.
    Read-only SELECT streamed row-by-row under WAL, so editors keep working while it
    runs. UTF-8 with BOM so Cyrillic opens cleanly in Excel."""
    def generate():
        buf = io.StringIO()
        w = csv.writer(buf)
        yield "\ufeff"
        w.writerow(["file", "transcript", "speaker_id"])
        yield buf.getvalue(); buf.seek(0); buf.truncate(0)
        db = get_db()
        try:
            cur = db.execute(
                "SELECT c.b2_key, p.text_mn, c.speaker_id "
                "FROM clips c JOIN prompts p ON p.id = c.prompt_id "
                "WHERE c.status = 'approved' "
                "ORDER BY c.speaker_id, c.id")
            for r in cur:
                w.writerow([r["b2_key"] or "", r["text_mn"] or "", r["speaker_id"]])
                yield buf.getvalue(); buf.seek(0); buf.truncate(0)
        finally:
            db.close()
    return Response(generate(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="manifest.csv"'})


@app.route("/admin/export_payroll")
@login_required
@admin_required
def export_payroll():
    db=get_db()
    # Fetch contributor unpaid clips with their frozen rates
    contrib_clips = db.execute(
        "SELECT u.id as user_id, u.username, u.speaker_id, u.hourly_rate, "
        "       c.duration_seconds, c.contributor_rate_at_approval, "
        "       p.full_name, p.bank_name, p.iban, p.receiver_name "
        "FROM users u "
        "JOIN clips c ON c.speaker_id=u.speaker_id AND c.status='approved' AND c.compensated=0 "
        "LEFT JOIN profiles p ON p.speaker_id=u.speaker_id "
        "WHERE u.role='contributor' AND COALESCE(u.archived,0)=0 "
        "ORDER BY u.speaker_id"
    ).fetchall()
    # Group clips by user. For each user, group by rate and ceil ONCE per rate group.
    contrib_payroll = {}
    for c in contrib_clips:
        uid = c["user_id"]
        if uid not in contrib_payroll:
            contrib_payroll[uid] = {
                "username": c["username"], "speaker_id": c["speaker_id"],
                "full_name": c["full_name"], "bank_name": c["bank_name"],
                "iban": c["iban"], "_rate_buckets": {}, "_user_rate": c["hourly_rate"] or RATE_PER_HOUR,
            }
        rate = c["contributor_rate_at_approval"] or c["hourly_rate"] or RATE_PER_HOUR
        contrib_payroll[uid]["_rate_buckets"][rate] = (
            contrib_payroll[uid]["_rate_buckets"].get(rate, 0) + (c["duration_seconds"] or 0)
        )
    # Sum each user's per-rate buckets into the final amount
    for entry in contrib_payroll.values():
        entry["amount"] = sum(
            calc_comp(secs, rate) for rate, secs in entry["_rate_buckets"].items()
        )
        entry.pop("_rate_buckets")
        entry.pop("_user_rate")

    # Editors — fetch unpaid clips with frozen editor rate
    editor_clips = db.execute(
        "SELECT u.id as user_id, u.username, u.hourly_rate, "
        "       COALESCE(u.editor_penalty_pct,0) AS editor_penalty_pct, "
        "       c.duration_seconds, c.editor_rate_at_approval, "
        "       p.full_name, p.bank_name, p.iban "
        "FROM users u "
        "JOIN clips c ON c.editor_user_id=u.id AND c.status='approved' AND COALESCE(c.editor_paid,0)=0 "
        "LEFT JOIN profiles p ON p.speaker_id=u.speaker_id "
        "WHERE u.role='editor' AND COALESCE(u.archived,0)=0 "
        "ORDER BY u.username"
    ).fetchall()
    editor_payroll = {}
    for c in editor_clips:
        uid = c["user_id"]
        if uid not in editor_payroll:
            editor_payroll[uid] = {
                "username": c["username"], "speaker_id": f"EDITOR_{c['username'].upper()}",
                "full_name": c["full_name"], "bank_name": c["bank_name"],
                "iban": c["iban"], "_rate_buckets": {},
                "_penalty_pct": c["editor_penalty_pct"],
            }
        rate = c["editor_rate_at_approval"] or c["hourly_rate"] or 70000
        editor_payroll[uid]["_rate_buckets"][rate] = (
            editor_payroll[uid]["_rate_buckets"].get(rate, 0) + (c["duration_seconds"] or 0)
        )
    for entry in editor_payroll.values():
        entry["amount"] = apply_editor_penalty(sum(
            calc_comp(secs, rate) for rate, secs in entry["_rate_buckets"].items()
        ), entry["_penalty_pct"])
        entry.pop("_rate_buckets"); entry.pop("_penalty_pct")
    db.close()

    # Combine contributors + editors into one list of payment rows
    all_rows = list(contrib_payroll.values()) + list(editor_payroll.values())
    # Filter out zero-amount rows just in case
    all_rows = [r for r in all_rows if r["amount"] > 0]

    import openpyxl, re
    template_path = os.path.join(os.path.dirname(__file__), "BulkTranTemplate.xlsx")
    buf = io.BytesIO()
    with open(template_path, "rb") as f:
        buf.write(f.read())
    buf.seek(0)
    wb = openpyxl.load_workbook(buf)
    ws = wb.active
    def extract_bank_code(bank_name_str):
        if not bank_name_str: return ""
        m = re.search(r'\((\d+)\)', bank_name_str)
        return m.group(1) if m else ""
    for r_idx, r in enumerate(all_rows, 2):
        amount      = r["amount"]
        bank_name   = r["bank_name"] or ""
        bank_code   = extract_bank_code(bank_name)
        iban        = r["iban"] or ""
        full_name   = r["full_name"] or ""
        col_a = "10" if bank_code == "040000" else "20"
        ws.cell(row=r_idx, column=1,  value=col_a)
        ws.cell(row=r_idx, column=2,  value="MN550004000417006334")
        ws.cell(row=r_idx, column=3,  value="MNT")
        ws.cell(row=r_idx, column=4,  value=bank_code)
        ws.cell(row=r_idx, column=5,  value=iban)
        ws.cell(row=r_idx, column=6,  value=full_name)
        ws.cell(row=r_idx, column=7,  value="MNT")
        ws.cell(row=r_idx, column=8,  value="Audio File Compensation")
        ws.cell(row=r_idx, column=9,  value="1")
        ws.cell(row=r_idx, column=10, value=amount)
    out = io.BytesIO()
    # Downloading the payroll opens the settlement window: approvals from now until
    # "All Paid" would NOT be in this file — the approval UI warns while this is set.
    db2 = get_db()
    db2.execute("REPLACE INTO _one_time_flags (key, applied_at) VALUES ('payroll_pending', datetime('now'))")
    db2.commit(); db2.close()
    wb.save(out)
    out.seek(0)
    return send_file(out,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"BulkPayment_{datetime.date.today()}.xlsx")

@app.route("/admin/reconcile_consent_dates", methods=["POST"])
@login_required
@admin_required
def reconcile_consent_dates():
    """One-time maintenance: correct consent_uploaded_at to the TRUE first-consent date
    by reading the OLDEST version of each contributor's consent.pdf from B2 version
    history. Needed because early re-consents (before date-preservation existed)
    overwrote the original date. Only ever moves a date EARLIER, never later — so it is
    safe and idempotent (re-running changes nothing once corrected)."""
    db = get_db()
    rows = db.execute(
        "SELECT speaker_id, consent_b2_key, consent_uploaded_at FROM profiles "
        "WHERE COALESCE(consent_b2_key,'')<>''").fetchall()
    b2 = get_b2()
    checked = corrected = errors = 0
    for i, r in enumerate(rows, 1):
        sid = r["speaker_id"]; key = r["consent_b2_key"]
        checked += 1
        try:
            oldest = None
            paginator = b2.get_paginator("list_object_versions")
            for page in paginator.paginate(Bucket=B2_BUCKET_NAME, Prefix=key):
                for v in page.get("Versions", []):
                    if v.get("Key") != key:
                        continue
                    lm = v.get("LastModified")
                    if lm and (oldest is None or lm < oldest):
                        oldest = lm
            if oldest is None:
                continue
            oldest_str = oldest.strftime("%Y-%m-%d %H:%M:%S")
            cur = r["consent_uploaded_at"] or ""
            # Only move BACK to the true earliest; never forward (protects already-correct dates).
            if (not cur) or (oldest_str < cur):
                db.execute("UPDATE profiles SET consent_uploaded_at=? WHERE speaker_id=?",
                           (oldest_str, sid))
                corrected += 1
        except Exception as e:
            print(f"[CONSENT RECONCILE] {sid}: {e}")
            errors += 1
        if i % 25 == 0:
            db.commit()   # periodic commit -> resumable if interrupted
    db.commit(); db.close()
    flash(f"Consent dates reconciled — checked {checked}, corrected {corrected}, "
          f"errors {errors}.", "info" if errors else "success")
    return redirect(url_for("admin_dashboard"))

# ============================================================================
# Contributor <-> Editor <-> Admin chat
# Thread key is the contributor's speaker_id. The contributor always messages
# "Editor"; the system routes to whoever is currently assigned (resolved live),
# so reassignment is seamless. Admin can monitor every thread and post as "Admin".
# ============================================================================

def _current_editor_id(db, sid):
    """The contributor's currently-assigned editor user id, or None (unassigned)."""
    row = db.execute(
        "SELECT editor_user_id FROM editor_contributors WHERE contributor_speaker_id=? LIMIT 1",
        (sid,)).fetchone()
    return row["editor_user_id"] if row else None

def _chat_can_access(db, sid):
    """True if the current session may read/post in contributor `sid`'s thread."""
    role = session.get("role")
    if role == "admin":
        return True
    if role == "editor":
        return _current_editor_id(db, sid) == session.get("user_id")
    # contributor: only their own thread
    return session.get("speaker_id") == sid

def chat_unread_count():
    """Unread count for the badge, tailored to the current session's role.
    Returns 0 on any error so a chat hiccup never breaks a dashboard."""
    try:
        db = get_db()
        role = session.get("role")
        if role == "contributor":
            n = db.execute(
                "SELECT COUNT(*) AS n FROM chat_messages WHERE contributor_speaker_id=? "
                "AND sender_role!='contributor' AND read_by_contributor=0",
                (session.get("speaker_id"),)).fetchone()["n"]
        elif role == "editor":
            n = db.execute(
                "SELECT COUNT(*) AS n FROM chat_messages m "
                "JOIN editor_contributors ec ON ec.contributor_speaker_id=m.contributor_speaker_id "
                "WHERE ec.editor_user_id=? AND m.sender_role!='editor' AND m.read_by_editor=0",
                (session.get("user_id"),)).fetchone()["n"]
        elif role == "admin":
            # Actionable for admin: new contributor messages in UNASSIGNED threads
            # (admin's own group). Other groups are monitored on the chat page itself.
            n = db.execute(
                "SELECT COUNT(*) AS n FROM chat_messages m WHERE m.sender_role='contributor' "
                "AND m.read_by_admin=0 AND m.contributor_speaker_id NOT IN "
                "(SELECT contributor_speaker_id FROM editor_contributors)").fetchone()["n"]
        else:
            n = 0
        db.close()
        return n
    except Exception as e:
        print(f"[CHAT] unread count error: {e}")
        return 0

@app.route("/chat")
@login_required
def chat():
    db = get_db()
    role = session.get("role")
    threads = []   # for editor/admin: list of contributor threads
    me_sid = None  # for contributor: their own thread

    if role == "contributor":
        me_sid = session.get("speaker_id")
        ed_id = _current_editor_id(db, me_sid)
        ed_name = "Admin"
        if ed_id:
            ed_name = "Editor"
        db.close()
        return render_template("chat.html", role=role, me_sid=me_sid,
                               counterpart=ed_name, threads=None)

    if role == "editor":
        rows = db.execute(
            "SELECT u.speaker_id, u.username, "
            "  (SELECT COUNT(*) FROM chat_messages m WHERE m.contributor_speaker_id=u.speaker_id "
            "    AND m.sender_role!='editor' AND m.read_by_editor=0) AS unread, "
            "  (SELECT MAX(created_at) FROM chat_messages m WHERE m.contributor_speaker_id=u.speaker_id) AS last_at "
            "FROM users u JOIN editor_contributors ec ON ec.contributor_speaker_id=u.speaker_id "
            "WHERE ec.editor_user_id=? AND u.role='contributor' AND COALESCE(u.archived,0)=0 "
            "ORDER BY (last_at IS NULL), last_at DESC, u.username",
            (session.get("user_id"),)).fetchall()
        threads = [dict(r) for r in rows]
        db.close()
        return render_template("chat.html", role=role, me_sid=None,
                               counterpart=None, threads=threads, groups=None)

    if role == "admin":
        # All threads that have at least one message OR an assigned contributor,
        # grouped by editor (with an 'Unassigned' group), for full monitoring.
        rows = db.execute(
            "SELECT u.speaker_id, u.username, "
            "  (SELECT editor_user_id FROM editor_contributors ec WHERE ec.contributor_speaker_id=u.speaker_id LIMIT 1) AS editor_id, "
            "  (SELECT username FROM users e WHERE e.id=(SELECT editor_user_id FROM editor_contributors ec WHERE ec.contributor_speaker_id=u.speaker_id LIMIT 1)) AS editor_name, "
            "  (SELECT COUNT(*) FROM chat_messages m WHERE m.contributor_speaker_id=u.speaker_id "
            "    AND m.sender_role!='admin' AND m.read_by_admin=0) AS unread, "
            "  (SELECT COUNT(*) FROM chat_messages m WHERE m.contributor_speaker_id=u.speaker_id) AS msg_count, "
            "  (SELECT MAX(created_at) FROM chat_messages m WHERE m.contributor_speaker_id=u.speaker_id) AS last_at "
            "FROM users u WHERE u.role='contributor' AND COALESCE(u.archived,0)=0 "
            "ORDER BY (last_at IS NULL), last_at DESC, u.username").fetchall()
        groups = {}
        for r in rows:
            d = dict(r)
            key = d["editor_name"] if d["editor_id"] else "Unassigned"
            groups.setdefault(key, []).append(d)
        db.close()
        return render_template("chat.html", role=role, me_sid=None,
                               counterpart=None, threads=None, groups=groups)

    db.close()
    abort(403)

@app.route("/chat/thread/<speaker_id>")
@login_required
def chat_thread(speaker_id):
    """JSON messages for a thread; marks them read for the viewer."""
    db = get_db()
    if not _chat_can_access(db, speaker_id):
        db.close(); abort(403)
    role = session.get("role")
    rows = db.execute(
        "SELECT id, sender_role, sender_user_id, body, created_at "
        "FROM chat_messages WHERE contributor_speaker_id=? ORDER BY created_at, id",
        (speaker_id,)).fetchall()
    # Mark read for this viewer
    col = {"contributor": "read_by_contributor", "editor": "read_by_editor",
           "admin": "read_by_admin"}.get(role)
    if col:
        db.execute(
            f"UPDATE chat_messages SET {col}=1 WHERE contributor_speaker_id=? "
            f"AND sender_role!=? AND {col}=0", (speaker_id, role))
        db.commit()
    msgs = [{"id": r["id"], "role": r["sender_role"], "body": r["body"],
             "at": r["created_at"]} for r in rows]
    db.close()
    return jsonify({"ok": True, "messages": msgs})

@app.route("/chat/send", methods=["POST"])
@login_required
def chat_send():
    role = session.get("role")
    sid = (request.form.get("speaker_id") or "").strip()
    body = (request.form.get("body") or "").strip()
    if role == "contributor":
        sid = session.get("speaker_id")   # contributors can only post to their own thread
    if not sid or not body:
        return jsonify({"ok": False, "error": "empty"}), 400
    if len(body) > 4000:
        body = body[:4000]
    db = get_db()
    if not _chat_can_access(db, sid):
        db.close(); abort(403)
    # The sender has, by definition, already seen their own message.
    rbc = 1 if role == "contributor" else 0
    rbe = 1 if role == "editor" else 0
    rba = 1 if role == "admin" else 0
    db.execute(
        "INSERT INTO chat_messages (contributor_speaker_id, sender_role, sender_user_id, "
        "  body, read_by_contributor, read_by_editor, read_by_admin) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, role, session.get("user_id"), body, rbc, rbe, rba))
    db.commit(); db.close()
    return jsonify({"ok": True})

# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL SPEECH COLLECTION — Stage A Part 1 (admin foundation)
# Data model, state machine, caps/pairing enforcement, topic generation + review,
# admin session creation. Speaker-facing recording/upload arrives in Part 2.
# Nothing here touches RS behavior.
# ══════════════════════════════════════════════════════════════════════════════
import random as _random

def _conv_next_session_code(db):
    """CONV_000001-style, sequential, never reused (rows are never deleted)."""
    # Highest number ever issued, not "the latest row" — the latter can re-issue a code
    # (seen on the live DB: two sessions sharing CONV_000008), which would make two sessions
    # write to the same B2 audio path.
    r = db.execute("SELECT MAX(CAST(SUBSTR(session_code, 6) AS INTEGER)) AS n FROM conv_sessions").fetchone()
    n = (r["n"] or 0) + 1
    return f"CONV_{n:06d}"

def _conv_invite_code():
    """6 chars, unambiguous alphabet (no 0/O/1/I)."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(6))

def _conv_transition(db, session_id, new_status, **stamps):
    """Enforce the §8.3 state machine. Raises ValueError on an illegal move.
    Optional stamps set timestamp columns in the same UPDATE."""
    row = db.execute("SELECT status FROM conv_sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        raise ValueError(f"session {session_id} not found")
    cur = row["status"]
    if new_status not in CONV_TRANSITIONS.get(cur, set()):
        raise ValueError(f"illegal transition {cur} -> {new_status}")
    sets = ["status=?"]; params = [new_status]
    for col, val in stamps.items():
        sets.append(f"{col}=?"); params.append(val)
    params.append(session_id)
    db.execute(f"UPDATE conv_sessions SET {', '.join(sets)} WHERE id=?", params)

def _conv_session_minutes(row):
    """Minutes a session contributes: measured duration if known, else the planning unit."""
    return (row["duration_sec"] / 60.0) if row["duration_sec"] else CONV_PLANNED_MINUTES

def _conv_speaker_stats(db, sid):
    """Counting sessions for one speaker: total minutes, minutes per partner,
    session count, distinct partners. Excludes aborted/incomplete/rejected (§6)."""
    qm = ",".join("?" * len(CONV_COUNTING_STATUSES))
    rows = db.execute(
        f"SELECT speaker_a_id, speaker_b_id, duration_sec FROM conv_sessions "
        f"WHERE (speaker_a_id=? OR speaker_b_id=?) AND status IN ({qm})",
        (sid, sid, *CONV_COUNTING_STATUSES)).fetchall()
    total = 0.0; per_partner = {}
    for r in rows:
        partner = r["speaker_b_id"] if r["speaker_a_id"] == sid else r["speaker_a_id"]
        m = _conv_session_minutes(r)
        total += m
        per_partner[partner] = per_partner.get(partner, 0.0) + m
    counts = {}
    for r in rows:
        partner = r["speaker_b_id"] if r["speaker_a_id"] == sid else r["speaker_a_id"]
        counts[partner] = counts.get(partner, 0) + 1
    return {"total_min": total, "per_partner_min": per_partner,
            "sessions": len(rows), "partner_sessions": counts,
            "distinct_partners": len(per_partner)}

def _conv_pairing_check(db, a, b):
    """Apply §6 to a proposed new session between a and b, assuming a planned
    20-minute session. Returns (ok, violations[list of str])."""
    v = []
    if a == b:
        return False, ["A speaker cannot be paired with themselves."]
    for sid, other in ((a, b), (b, a)):
        s = _conv_speaker_stats(db, sid)
        # Hard cap (only when CONV_CAP_ENABLED)
        if CONV_CAP_ENABLED and s["total_min"] + CONV_PLANNED_MINUTES > CONV_CAP_MINUTES:
            v.append(f"{sid} would exceed the {CONV_CAP_MINUTES/60:.0f} h cap "
                     f"({s['total_min']:.0f} min used; a session needs {CONV_PLANNED_MINUTES:.0f} min).")
        # Same-partner limit until 3 distinct partners
        with_other = s["partner_sessions"].get(other, 0)
        if s["distinct_partners"] < CONV_DISTINCT_PARTNERS and with_other >= CONV_SAME_PARTNER_MAX:
            v.append(f"{sid} already has {with_other} sessions with {other}; needs "
                     f"{CONV_DISTINCT_PARTNERS} distinct partners before more with the same one.")
        # Partner-share limit once ≥3 sessions (evaluated after adding this session)
        if s["sessions"] + 1 >= CONV_DISTINCT_PARTNERS:
            new_total = s["total_min"] + CONV_PLANNED_MINUTES
            new_with = s["per_partner_min"].get(other, 0.0) + CONV_PLANNED_MINUTES
            if new_with > CONV_PARTNER_SHARE_MAX * new_total + 1e-9:
                v.append(f"{sid} would have {100*new_with/new_total:.0f}% of their minutes with "
                         f"{other} (limit {int(CONV_PARTNER_SHARE_MAX*100)}%).")
    return (len(v) == 0), v

def _conv_circle_of(db, speaker_id):
    """The editor circle a speaker belongs to: the editor's user id from editor_contributors
    (read-speech assignment), or 0 for speakers not assigned to any editor (the admin acts
    as their editor). Only speakers of the same circle may call each other."""
    r = db.execute("SELECT editor_user_id FROM editor_contributors WHERE contributor_speaker_id=? LIMIT 1",
                   (speaker_id,)).fetchone()
    return int(r["editor_user_id"]) if r else 0

def _conv_group_topic_ids(db, circle):
    """Active topics assigned to this circle by admin; empty list = no batch assigned yet
    (callers fall back to all active topics)."""
    return [r["topic_id"] for r in db.execute(
        "SELECT g.topic_id FROM conv_topic_groups g JOIN conv_topics t ON t.id=g.topic_id "
        "WHERE g.editor_user_id=? AND t.status='active' ORDER BY t.title_mn", (int(circle or 0),)).fetchall()]

def _conv_group_topics(db, circle):
    """Topic rows (id, title_mn) the dial page offers this circle."""
    ids = _conv_group_topic_ids(db, circle)
    if ids:
        qm = ",".join("?" * len(ids))
        return db.execute(f"SELECT id, title_mn FROM conv_topics WHERE id IN ({qm}) ORDER BY title_mn", ids).fetchall()
    return db.execute("SELECT id, title_mn FROM conv_topics WHERE status='active' ORDER BY title_mn").fetchall()

def _conv_pick_topic(db, a, b):
    """Round-robin over the circle's active topics (all active topics if the circle has no
    batch yet), avoiding any topic either speaker has done (counting statuses) if possible;
    else least-used."""
    ids = _conv_group_topic_ids(db, _conv_circle_of(db, a))
    if ids:
        qm = ",".join("?" * len(ids))
        active = db.execute(f"SELECT id FROM conv_topics WHERE status='active' AND id IN ({qm}) "
                            "ORDER BY use_count ASC, id ASC", ids).fetchall()
    else:
        active = db.execute("SELECT id FROM conv_topics WHERE status='active' "
                            "ORDER BY use_count ASC, id ASC").fetchall()
    if not active:
        return None
    qm = ",".join("?" * len(CONV_COUNTING_STATUSES))
    done = {r["topic_id"] for r in db.execute(
        f"SELECT topic_id FROM conv_sessions WHERE (speaker_a_id IN (?,?) OR speaker_b_id IN (?,?)) "
        f"AND status IN ({qm})", (a, b, a, b, *CONV_COUNTING_STATUSES)).fetchall()}
    for t in active:
        if t["id"] not in done:
            return t["id"]
    return active[0]["id"]

def _conv_eligible_speakers(db):
    """Enrolled contributors on the current consent version (same gate as RS)."""
    return db.execute(
        "SELECT u.speaker_id, u.username FROM users u JOIN profiles p ON p.speaker_id=u.speaker_id "
        "WHERE u.role='contributor' AND COALESCE(u.frozen,0)=0 AND COALESCE(u.archived,0)=0 "
        "  AND p.completed=1 AND COALESCE(p.consent_b2_key,'')<>'' "
        "  AND COALESCE(p.consent_version,'')=? ORDER BY u.speaker_id",
        (CURRENT_CONSENT_VERSION,)).fetchall()

def _conv_generate_topics(n):
    """Ask OpenAI for n draft topics using the admin-editable system prompt.
    Returns list of {title, prompt}. Robust to fenced/loose JSON."""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI is not configured (OPENAI_API_KEY missing).")
    db = get_db()
    sp = db.execute("SELECT system_prompt FROM conv_topic_settings WHERE id=1").fetchone()
    db.close()
    system_prompt = sp["system_prompt"] if sp else CONV_TOPIC_GEN_DEFAULT_PROMPT
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Generate {int(n)} topics now. JSON array only."}],
        temperature=0.9,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = raw.strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < 0:
        raise RuntimeError("Model did not return a JSON array.")
    items = json.loads(raw[start:end+1])
    out = []
    for it in items:
        t = (it.get("title") or "").strip(); p = (it.get("prompt") or "").strip()
        if t and p:
            out.append({"title": t, "prompt": p})
    return out

def _conv_session_row(db, sess_id):
    return db.execute(
        "SELECT s.*, t.title_mn AS topic_title, t.prompt_mn AS topic_prompt "
        "FROM conv_sessions s LEFT JOIN conv_topics t ON t.id=s.topic_id WHERE s.id=?",
        (sess_id,)).fetchone()

# ── Admin: topics ─────────────────────────────────────────────────────────────
@app.route("/admin/conv/topics")
@login_required
@admin_required
def conv_topics():
    db = get_db()
    topics = db.execute("SELECT * FROM conv_topics ORDER BY "
                        "CASE status WHEN 'draft' THEN 0 WHEN 'active' THEN 1 ELSE 2 END, id DESC").fetchall()
    sp = db.execute("SELECT system_prompt FROM conv_topic_settings WHERE id=1").fetchone()
    db.close()
    counts = {k: sum(1 for t in topics if t["status"] == k) for k in ("draft", "active", "retired")}
    return render_template("conv_topics.html", topics=topics, counts=counts,
                           system_prompt=(sp["system_prompt"] if sp else CONV_TOPIC_GEN_DEFAULT_PROMPT),
                           openai_ready=(get_openai_client() is not None))

@app.route("/admin/conv/topics/generate", methods=["POST"])
@login_required
@admin_required
def conv_topics_generate():
    try:
        n = max(1, min(40, int(request.form.get("n", "20"))))
    except ValueError:
        n = 20
    try:
        items = _conv_generate_topics(n)
    except Exception as e:
        flash(f"Topic generation failed: {e}", "error")
        return redirect(url_for("conv_topics"))
    db = get_db()
    for it in items:
        db.execute("INSERT INTO conv_topics (title_mn, prompt_mn, status, source) VALUES (?,?,'draft','generated')",
                   (it["title"], it["prompt"]))
    db.commit(); db.close()
    flash(f"Generated {len(items)} draft topic(s). Review, edit, then approve the ones you want live.", "success")
    return redirect(url_for("conv_topics"))

@app.route("/admin/conv/topics/settings", methods=["POST"])
@login_required
@admin_required
def conv_topics_settings():
    sp = (request.form.get("system_prompt") or "").strip()
    if len(sp) < 40:
        flash("System prompt too short.", "error")
        return redirect(url_for("conv_topics"))
    db = get_db()
    db.execute("INSERT INTO conv_topic_settings (id, system_prompt, updated_at) VALUES (1, ?, datetime('now')) "
               "ON CONFLICT(id) DO UPDATE SET system_prompt=excluded.system_prompt, updated_at=excluded.updated_at", (sp,))
    db.commit(); db.close()
    flash("Generation prompt saved.", "success")
    return redirect(url_for("conv_topics"))

@app.route("/admin/conv/topics/add", methods=["POST"])
@login_required
@admin_required
def conv_topics_add():
    t = (request.form.get("title_mn") or "").strip(); p = (request.form.get("prompt_mn") or "").strip()
    if not t or not p:
        flash("Title and prompt are both required.", "error")
        return redirect(url_for("conv_topics"))
    db = get_db()
    db.execute("INSERT INTO conv_topics (title_mn, prompt_mn, status, source) VALUES (?,?,'active','manual')", (t, p))
    db.commit(); db.close()
    flash("Topic added to the active pool.", "success")
    return redirect(url_for("conv_topics"))

@app.route("/admin/conv/topics/<int:topic_id>/update", methods=["POST"])
@login_required
@admin_required
def conv_topic_update(topic_id):
    action = (request.form.get("action") or "").strip()
    t = (request.form.get("title_mn") or "").strip(); p = (request.form.get("prompt_mn") or "").strip()
    db = get_db()
    row = db.execute("SELECT * FROM conv_topics WHERE id=?", (topic_id,)).fetchone()
    if not row:
        db.close(); abort(404)
    if action == "discard":
        if row["status"] == "draft" and not db.execute(
                "SELECT 1 FROM conv_sessions WHERE topic_id=?", (topic_id,)).fetchone():
            db.execute("DELETE FROM conv_topics WHERE id=?", (topic_id,))   # drafts only, never used
        else:
            db.execute("UPDATE conv_topics SET status='retired', updated_at=datetime('now') WHERE id=?", (topic_id,))
        db.commit(); db.close()
        flash("Topic discarded.", "info"); return redirect(url_for("conv_topics"))
    if not t or not p:
        db.close(); flash("Title and prompt are both required.", "error")
        return redirect(url_for("conv_topics"))
    new_status = {"approve": "active", "save": row["status"], "retire": "retired",
                  "reactivate": "active"}.get(action, row["status"])
    db.execute("UPDATE conv_topics SET title_mn=?, prompt_mn=?, status=?, updated_at=datetime('now') WHERE id=?",
               (t, p, new_status, topic_id))
    db.commit(); db.close()
    flash({"approve": "Topic approved — now in the active pool.", "retire": "Topic retired.",
           "reactivate": "Topic reactivated.", "save": "Topic saved."}.get(action, "Saved."), "success")
    return redirect(url_for("conv_topics"))

# ── Admin: sessions ───────────────────────────────────────────────────────────
@app.route("/admin/conv/topic_groups", methods=["GET", "POST"])
@login_required
@admin_required
def conv_topic_groups():
    """Assign a batch of topics to each editor circle. Call initiators pick from their circle's
    batch (or auto-rotate within it). Circle 0 = admin's own (unassigned) speakers."""
    db = get_db()
    editors = db.execute("SELECT id, username FROM users WHERE role='editor' AND COALESCE(archived,0)=0 ORDER BY username").fetchall()
    if request.method == "POST":
        try:
            circle = int(request.form.get("circle") or 0)
        except ValueError:
            circle = 0
        ids = []
        for v in request.form.getlist("topic_ids"):
            try: ids.append(int(v))
            except ValueError: pass
        db.execute("DELETE FROM conv_topic_groups WHERE editor_user_id=?", (circle,))
        for tid in ids:
            db.execute("INSERT OR IGNORE INTO conv_topic_groups (topic_id, editor_user_id) VALUES (?,?)", (tid, circle))
        db.commit(); db.close()
        flash(f"{len(ids)} topic(s) assigned to that circle.", "success")
        return redirect(url_for("conv_topic_groups", circle=circle))
    try:
        circle = int(request.args.get("circle") or 0)
    except ValueError:
        circle = 0
    topics = db.execute("SELECT id, title_mn, use_count FROM conv_topics WHERE status='active' ORDER BY title_mn").fetchall()
    chosen = set(_conv_group_topic_ids(db, circle))
    counts = {r["editor_user_id"]: r["n"] for r in db.execute(
        "SELECT g.editor_user_id, COUNT(*) n FROM conv_topic_groups g JOIN conv_topics t ON t.id=g.topic_id "
        "WHERE t.status='active' GROUP BY g.editor_user_id").fetchall()}
    db.close()
    return render_template("conv_topic_groups.html", editors=editors, circle=circle, topics=topics, chosen=chosen, counts=counts)

@app.route("/admin/conv/sessions")
@login_required
@admin_required
def conv_sessions():
    _conv_housekeeping()
    db = get_db()
    # Default view hides failed attempts (aborted / incomplete): every unanswered-then-dropped
    # call leaves an aborted row, so they pile up fast. "?status=all" shows everything;
    # the per-status chips still work as before.
    status = (request.args.get("status") or "").strip()
    q = ("SELECT s.*, t.title_mn AS topic_title FROM conv_sessions s "
         "LEFT JOIN conv_topics t ON t.id=s.topic_id ")
    params = ()
    if status == "all":
        pass
    elif status:
        q += "WHERE s.status=? "; params = (status,)
    else:
        q += "WHERE s.status NOT IN ('aborted','incomplete') "
    q += "ORDER BY s.id DESC LIMIT 300"
    sessions = db.execute(q, params).fetchall()
    counts = {r["status"]: r["n"] for r in db.execute(
        "SELECT status, COUNT(*) n FROM conv_sessions GROUP BY status").fetchall()}
    speakers = _conv_eligible_speakers(db)
    active_topics = db.execute("SELECT id, title_mn FROM conv_topics WHERE status='active' ORDER BY title_mn").fetchall()
    # remaining minutes per eligible speaker (for the picker)
    remaining = {s["speaker_id"]: max(0.0, CONV_CAP_MINUTES - _conv_speaker_stats(db, s["speaker_id"])["total_min"])
                 for s in speakers} if CONV_CAP_ENABLED else {}
    db.close()
    return render_template("conv_sessions.html", sessions=sessions, counts=counts, status=status,
                           speakers=speakers, active_topics=active_topics, remaining=remaining,
                           states=CONV_STATES, cap_enabled=CONV_CAP_ENABLED)

@app.route("/admin/conv/sessions/create", methods=["POST"])
@login_required
@admin_required
def conv_session_create():
    a = (request.form.get("speaker_a") or "").strip(); b = (request.form.get("speaker_b") or "").strip()
    topic_sel = (request.form.get("topic_id") or "auto").strip()
    override_reason = (request.form.get("override_reason") or "").strip()
    db = get_db()
    elig = {s["speaker_id"] for s in _conv_eligible_speakers(db)}
    if a not in elig or b not in elig:
        db.close(); flash("Both speakers must be enrolled and on the current consent version.", "error")
        return redirect(url_for("conv_sessions"))
    ok, violations = _conv_pairing_check(db, a, b)
    if not ok and not override_reason:
        db.close()
        flash("Cannot create session: " + " ".join(violations) +
              " (An admin override with a reason is possible.)", "error")
        return redirect(url_for("conv_sessions"))
    if topic_sel == "auto":
        topic_id = _conv_pick_topic(db, a, b)
    else:
        try: topic_id = int(topic_sel)
        except ValueError: topic_id = None
    if not topic_id:
        db.close(); flash("No active topics — approve at least one topic first.", "error")
        return redirect(url_for("conv_sessions"))
    code = _conv_next_session_code(db)
    ca, cb = _conv_circle_of(db, a), _conv_circle_of(db, b)
    editor_id = ca if (ca and ca == cb) else None      # cross-circle admin pairing → admin edits it
    cur = db.execute(
        "INSERT INTO conv_sessions (session_code, topic_id, status, speaker_a_id, speaker_b_id, invite_code, created_by, editor_id) "
        "VALUES (?,?,'created',?,?,?,?,?)",
        (code, topic_id, a, b, _conv_invite_code(), session.get("username"), editor_id))
    sess_id = cur.lastrowid
    db.execute("UPDATE conv_topics SET use_count=use_count+1 WHERE id=?", (topic_id,))
    if not ok:
        db.execute("INSERT INTO conv_pairing_overrides (session_id, admin_username, reason, violations) VALUES (?,?,?,?)",
                   (sess_id, session.get("username"), override_reason, " | ".join(violations)))
        print(f"[CONV OVERRIDE] {session.get('username')} created {code} despite: {violations} — reason: {override_reason}")
    db.commit(); db.close()
    flash(f"Session {code} created for {a} + {b}." + (" (pairing rule overridden — logged)" if not ok else ""), "success")
    return redirect(url_for("conv_session_detail", sess_id=sess_id))

@app.route("/admin/conv/sessions/<int:sess_id>")
@login_required
@admin_required
def conv_session_detail(sess_id):
    db = get_db()
    s = _conv_session_row(db, sess_id)
    if not s:
        db.close(); abort(404)
    tracks = db.execute("SELECT * FROM conv_tracks WHERE session_id=? ORDER BY channel", (sess_id,)).fetchall()
    overrides = db.execute("SELECT * FROM conv_pairing_overrides WHERE session_id=? ORDER BY id", (sess_id,)).fetchall()
    affirmations = db.execute("SELECT * FROM conv_affirmations WHERE session_id=? ORDER BY id", (sess_id,)).fetchall()
    stats = {sid: _conv_speaker_stats(db, sid) for sid in (s["speaker_a_id"], s["speaker_b_id"])}
    n_utt_a = db.execute("SELECT COUNT(*) FROM conv_utterances WHERE session_id=? AND channel='A'", (sess_id,)).fetchone()[0]
    n_utt_b = db.execute("SELECT COUNT(*) FROM conv_utterances WHERE session_id=? AND channel='B'", (sess_id,)).fetchone()[0]
    n_done = db.execute("SELECT COUNT(*) FROM conv_tracks WHERE session_id=? AND edit_status='done'", (sess_id,)).fetchone()[0]
    ed = db.execute("SELECT username FROM users WHERE id=?", (s["editor_id"],)).fetchone() if s["editor_id"] else None
    db.close()
    return render_template("conv_session_detail.html", s=s, tracks=tracks, overrides=overrides,
                           n_utt=n_utt_a + n_utt_b, n_utt_a=n_utt_a, n_utt_b=n_utt_b, n_done=n_done,
                           editor_name=(ed["username"] if ed else None), calc_comp=calc_comp,
                           affirmations=affirmations, stats=stats, cap_min=CONV_CAP_MINUTES, cap_enabled=CONV_CAP_ENABLED,
                           can_abort=("aborted" in CONV_TRANSITIONS.get(s["status"], set())))

@app.route("/admin/conv/sessions/<int:sess_id>/abort", methods=["POST"])
@login_required
@admin_required
def conv_session_abort(sess_id):
    db = get_db()
    try:
        _conv_on_abort(db, sess_id, "admin abort")
        db.commit(); flash("Session aborted (does not count toward caps or pairing).", "info")
    except ValueError as e:
        flash(f"Cannot abort: {e}", "error")
    db.close()
    return redirect(url_for("conv_session_detail", sess_id=sess_id))

# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL — Stage A Part 2 (speaker-facing: invites, session page,
# affirmation, direct-to-B2 multipart upload, background track checks)
# ══════════════════════════════════════════════════════════════════════════════
CONV_SAMPLE_RATE   = 48000
CONV_PART_BYTES    = 8 * 1024 * 1024      # multipart part size (>= B2's 5 MB minimum)
CONV_SILENCE_RMS   = 100.0                # 16-bit scale; below this the track is "silent"
CONV_INCOMPLETE_MIN = 30                  # one track missing this long -> incomplete (§4.5)
CONV_INVITE_HOURS  = 24

def _conv_track_key(session_code, speaker_id):
    return f"conv/sessions/{session_code}/{speaker_id}.wav"          # §10.1 raw layout

def _conv_is_eligible(db, speaker_id):
    return any(s["speaker_id"] == speaker_id for s in _conv_eligible_speakers(db))

def _conv_participant(db, sess_id, speaker_id):
    """Session row + this speaker's channel, or (None, None) if not a participant."""
    s = _conv_session_row(db, sess_id)
    if not s:
        return None, None
    if speaker_id == s["speaker_a_id"]:
        return s, "A"
    if speaker_id == s["speaker_b_id"]:
        return s, "B"
    return None, None

def _conv_sweep_incomplete(db):
    """§4.5: a recording session with one track uploaded and the other still missing
    after 30 min becomes 'incomplete' (re-upload allowed). Cheap; run lazily on page loads."""
    rows = db.execute(
        "SELECT s.id FROM conv_sessions s WHERE s.status='recording' AND EXISTS ("
        "  SELECT 1 FROM conv_tracks t WHERE t.session_id=s.id AND t.uploaded_at IS NOT NULL "
        "    AND t.uploaded_at <= datetime('now', ?)) "
        "AND (SELECT COUNT(*) FROM conv_tracks t2 WHERE t2.session_id=s.id AND t2.uploaded_at IS NOT NULL) < 2",
        (f"-{CONV_INCOMPLETE_MIN} minutes",)).fetchall()
    for r in rows:
        try:
            _conv_transition(db, r["id"], "incomplete",
                             incomplete_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        except ValueError:
            pass
    if rows:
        db.commit()

def _conv_energy_envelope(b2, key, size, max_sec, frame=4800):
    """RMS per 100 ms frame (48 kHz) for the first max_sec seconds, via ranged reads."""
    import array, math
    env = []; off = 44; read = 2 * 1024 * 1024; carry = array.array('h')
    end_total = min(size, 44 + max_sec * 96000)
    try:
        import audioop
    except Exception:
        audioop = None
    while off < end_total:
        end = min(end_total - 1, off + read - 1)
        raw = b2.get_object(Bucket=B2_BUCKET_NAME, Key=key, Range=f"bytes={off}-{end}")["Body"].read()
        off = end + 1
        if len(raw) % 2: raw = raw[:-1]
        a = array.array('h'); a.frombytes(raw)
        if carry: a = carry + a; carry = array.array('h')
        n = (len(a) // frame) * frame
        for i in range(0, n, frame):
            seg = a[i:i + frame]
            if audioop:
                env.append(audioop.rms(seg.tobytes(), 2))
            else:
                env.append(math.sqrt(sum(x * x for x in seg) / frame))
        carry = a[n:]
    return env

def _conv_leak_scores(env_a, env_b, offset_b_ms):
    """Per side: contrast in dB = (own level while speaking) - (own track level while ONLY
    the partner speaks; 40th percentile so legitimate overlap doesn't dominate). High
    contrast = clean earphones; low contrast = the partner is in the track (loudspeaker)
    or the room is too noisy to use. Returns (contrast_a, contrast_b, n)."""
    import math
    def db_(v): return 20 * math.log10(v + 1.0)
    def pct(vals, p):
        s = sorted(vals); return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))] if s else 0.0
    d = int(round(offset_b_ms / 100.0))
    pairs = [(db_(env_a[k + d]), db_(env_b[k])) for k in range(len(env_b)) if 0 <= k + d < len(env_a)]
    if len(pairs) < CONV_LEAK_MIN_FRAMES:
        return None, None, len(pairs)
    la = [p[0] for p in pairs]; lb = [p[1] for p in pairs]
    fa0, oa = pct(la, 0.10), pct(la, 0.90)
    fb0, ob = pct(lb, 0.10), pct(lb, 0.90)
    a_talking = [(x, y) for x, y in pairs if x > fa0 + 0.6 * (oa - fa0)]
    b_talking = [(x, y) for x, y in pairs if y > fb0 + 0.6 * (ob - fb0)]
    score_a = score_b = None
    if len(a_talking) >= CONV_LEAK_MIN_FRAMES and len(b_talking) >= CONV_LEAK_MIN_FRAMES:
        own_a = pct([x for x, _ in a_talking], 0.50)          # A's level when A speaks
        a_while_b = pct([x for x, _ in b_talking], 0.40)      # A's level when B speaks
        own_b = pct([y for _, y in b_talking], 0.50)
        b_while_a = pct([y for _, y in a_talking], 0.40)
        score_a = own_a - a_while_b
        score_b = own_b - b_while_a
    return score_a, score_b, min(len(a_talking), len(b_talking))

def _conv_leak_check(db, sess_id):
    """Both tracks passed individually: compare them. Returns (leak: bool, corr, n)."""
    tracks = {t["channel"]: t for t in db.execute("SELECT * FROM conv_tracks WHERE session_id=?", (sess_id,)).fetchall()}
    a, b = tracks.get("A"), tracks.get("B")
    if not (a and b and a["b2_key"] and b["b2_key"]):
        return False, None, 0
    b2 = get_b2()
    env_a = _conv_energy_envelope(b2, a["b2_key"], int(a["size_bytes"] or 0), CONV_LEAK_ANALYZE_SEC)
    env_b = _conv_energy_envelope(b2, b["b2_key"], int(b["size_bytes"] or 0), CONV_LEAK_ANALYZE_SEC)
    off = 0
    if a["client_start_ms"] and b["client_start_ms"]:
        off = (b["client_start_ms"] + int(b["clock_skew_ms"] or 0)) - a["client_start_ms"]
        off = max(-60000, min(60000, off))
    sa, sb, n = _conv_leak_scores(env_a, env_b, off)
    scores = [s for s in (sa, sb) if s is not None]
    if not scores:
        return False, None, n          # inconclusive -> do not block
    worst = min(scores)                # lowest contrast = dirtiest track
    return worst < CONV_LEAK_CONTRAST_DB, worst, n

def _conv_check_track(track_id):
    """Background: verify an uploaded track straight from B2 (never through a request).
    Header via stdlib wave on a ranged read; RMS over sampled ranged reads (array-based,
    version-proof); size/duration consistency. Marks the track ok/rejected and promotes
    the session to 'uploaded' once both tracks pass."""
    import wave, array, math
    db = get_db()
    try:
        t = db.execute("SELECT * FROM conv_tracks WHERE id=?", (track_id,)).fetchone()
        if not t or not t["b2_key"]:
            return
        b2 = get_b2()
        head = b2.head_object(Bucket=B2_BUCKET_NAME, Key=t["b2_key"])
        size = int(head.get("ContentLength") or 0)
        first = b2.get_object(Bucket=B2_BUCKET_NAME, Key=t["b2_key"], Range="bytes=0-4095")["Body"].read()
        err = None; dur = None; rms = None
        try:
            with wave.open(io.BytesIO(first), "rb") as w:
                fr, ch, sw, nframes = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            if (fr, ch, sw) != (CONV_SAMPLE_RATE, 1, 2):
                err = f"[ДРАФТ] Бичлэгийн формат буруу ({fr} Hz / {ch} ch / {sw*8}-bit) — хуудсыг дахин ачаалж бичнэ үү"
            else:
                dur = nframes / float(fr)
                # 15 s grace under the minimum: the second phone starts a moment after the
                # Start signal and both stop together, so its track is slightly shorter.
                if not (CONV_MIN_MINUTES * 60 - 15 <= dur <= CONV_MAX_MINUTES * 60 + 5):
                    err = (f"[ДРАФТ] Бичлэгийн урт {dur/60:.1f} мин — {CONV_MIN_MINUTES:.0f}–{CONV_MAX_MINUTES:.0f} "
                           f"минутын хооронд байх ёстой")
                elif abs(size - (44 + nframes * 2)) > 4096:
                    err = "[ДРАФТ] Файл бүрэн илгээгдээгүй байна — дахин бичнэ үү"
        except Exception as e:
            err = f"[ДРАФТ] Бичлэгийн файл уншигдсангүй — дахин бичнэ үү ({e})"
        if err is None:
            # RMS over up to 24 sampled 1 MB windows spread across the data region.
            data_start, data_end = 44, size
            span = max(1, data_end - data_start)
            win = 1024 * 1024; nwin = min(24, max(1, span // win))
            ssum = 0.0; n = 0
            for k in range(nwin):
                off = data_start + (span * k) // nwin
                off -= off % 2
                end = min(data_end - 1, off + win - 1)
                if end <= off: continue
                chunk = b2.get_object(Bucket=B2_BUCKET_NAME, Key=t["b2_key"], Range=f"bytes={off}-{end}")["Body"].read()
                if len(chunk) % 2: chunk = chunk[:-1]
                a = array.array("h"); a.frombytes(chunk)
                try:
                    import audioop
                    ssum += (audioop.rms(chunk, 2) ** 2) * len(a); n += len(a)
                except Exception:
                    ssum += sum(x * x for x in a); n += len(a)
            rms = math.sqrt(ssum / n) if n else 0.0
            if rms < CONV_SILENCE_RMS:
                err = f"[ДРАФТ] Бичлэгт дуу хоолой илрээгүй (микрофон ажиллаагүй байж магадгүй, RMS {rms:.0f})"
        if err:
            db.execute("UPDATE conv_tracks SET check_status='rejected', check_error=?, duration_sec=?, rms=?, size_bytes=? WHERE id=?",
                       (err, dur, rms, size, track_id))
            db.commit()
            print(f"[CONV CHECK] track {track_id} rejected: {err}")
            return
        db.execute("UPDATE conv_tracks SET check_status='ok', check_error=NULL, duration_sec=?, rms=?, size_bytes=? WHERE id=?",
                   (dur, rms, size, track_id))
        db.commit()
        # both tracks ok -> session uploaded (from recording or incomplete)
        oks = db.execute("SELECT duration_sec FROM conv_tracks WHERE session_id=? AND check_status='ok'",
                         (t["session_id"],)).fetchall()
        if len(oks) == 2:
            # Loudspeaker-leak check across the pair (the guarantee behind the earphone rule).
            try:
                leak, corr, nfr = _conv_leak_check(db, t["session_id"])
            except Exception as e:
                print(f"[CONV CHECK] leak check failed for session {t['session_id']} (skipping): {e}")
                leak, corr, nfr = False, None, 0
            db.execute("UPDATE conv_sessions SET leak_corr=? WHERE id=?", (corr, t["session_id"]))
            if leak:
                reason = (f"[ДРАФТ] Бичлэгт хамтрагчийн дуу эсвэл хэт их дэвсгэр чимээ орсон байна "
                          f"(ялгаа {corr:.0f} dB). Чихэвч зүүж, чимээгүй газар шинэ яриа хийнэ үү.")
                db.execute("UPDATE conv_tracks SET check_status='rejected', check_error=? WHERE session_id=?", (reason, t["session_id"]))
                db.execute("UPDATE conv_sessions SET abort_reason=? WHERE id=?", (reason, t["session_id"]))
                try:
                    _conv_on_abort(db, t["session_id"], f"loudspeaker leak corr={corr:.2f} over {nfr} frames")
                except ValueError as e:
                    print(f"[CONV CHECK] could not abort leaked session {t['session_id']}: {e}")
                db.commit()
                return
            sess_dur = sum(r["duration_sec"] for r in oks) / 2.0
            try:
                _conv_transition(db, t["session_id"], "uploaded",
                                 uploaded_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                 duration_sec=sess_dur)
                db.commit()
                _conv_maybe_autodraft(t["session_id"])
            except ValueError as e:
                print(f"[CONV CHECK] session {t['session_id']} not promoted: {e}")
    except Exception as e:
        import traceback; print(f"[CONV CHECK] track {track_id} failed:\n" + traceback.format_exc())
        try:
            db.execute("UPDATE conv_tracks SET check_status='rejected', check_error=? WHERE id=?",
                       (f"check failed: {e}", track_id)); db.commit()
        except Exception:
            pass
    finally:
        db.close()

def conv_gate_required(f):
    """During testing, speaker-facing conversation pages require a one-time password
    (stored in the session). Admin/reviewer pages are not gated. Empty password = open."""
    @wraps(f)
    def d(*a, **k):
        if CONV_TEST_PASSWORD and not session.get("conv_access_ok"):
            if request.path.endswith(("/status", "/signal", "/ice", "/affirm", "/upload/init", "/upload/sign",
                                      "/upload/complete", "/abort", "/edit/done")) or "/utt/" in request.path \
                    or request.path.startswith("/conv/call"):
                return jsonify({"ok": False, "error": "gate"}), 403
            return redirect(url_for("conv_gate", next=request.path))
        return f(*a, **k)
    return d

@app.route("/conv/gate", methods=["GET", "POST"])
@login_required
def conv_gate():
    if not CONV_TEST_PASSWORD or session.get("conv_access_ok"):
        return redirect(url_for("conv_home"))
    nxt = request.values.get("next") or "/conv"
    if not (nxt.startswith("/") and not nxt.startswith("//")):
        nxt = "/conv"
    if request.method == "POST":
        if (request.form.get("password") or "").strip() == CONV_TEST_PASSWORD:
            session["conv_access_ok"] = True
            return redirect(nxt)
        flash("Нууц үг буруу байна.", "error")
    return render_template("conv_gate.html", next=nxt)

# ── Speaker: home (my sessions, hours, create/join) ──────────────────────────
@app.route("/conv")
@login_required
@conv_gate_required
def conv_home():
    _conv_housekeeping()
    if session.get("role") != "contributor":
        abort(403)
    sid = session["speaker_id"]
    db = get_db()
    _conv_sweep_incomplete(db)
    eligible = _conv_is_eligible(db, sid)
    stats = _conv_speaker_stats(db, sid)
    # Only conversations that got through recording are listed. Failed attempts (aborted /
    # incomplete) and in-flight speaker-made sessions (created / recording) stay hidden — a
    # new call always starts a new session. Admin-created sessions still show while pending.
    sessions = db.execute(
        "SELECT s.*, t.title_mn AS topic_title FROM conv_sessions s LEFT JOIN conv_topics t ON t.id=s.topic_id "
        "WHERE (s.speaker_a_id=? OR s.speaker_b_id=?) AND s.status NOT IN ('aborted','incomplete') "
        "AND NOT (s.status IN ('created','recording') AND s.created_by IN (s.speaker_a_id, s.speaker_b_id)) "
        "ORDER BY s.id DESC LIMIT 50", (sid, sid)).fetchall()
    incoming = db.execute(
        "SELECT id, creator_speaker_id, created_at, expires_at FROM conv_invites WHERE invitee_speaker_id=? AND status='pending' "
        "AND expires_at > datetime('now') ORDER BY id DESC", (sid,)).fetchall()
    outgoing = db.execute(
        "SELECT id, invitee_speaker_id, created_at, expires_at FROM conv_invites WHERE creator_speaker_id=? AND status='pending' "
        "AND expires_at > datetime('now') ORDER BY id DESC", (sid,)).fetchall()
    active_topics = _conv_group_topics(db, _conv_circle_of(db, sid))   # this circle's batch (all active if none assigned)
    # show the chosen topic on incoming invites
    topic_titles = {t["id"]: t["title_mn"] for t in active_topics}
    incoming = [dict(r) | {"topic_title": topic_titles.get(r["topic_id"])} for r in db.execute(
        "SELECT id, creator_speaker_id, created_at, expires_at, topic_id FROM conv_invites WHERE invitee_speaker_id=? AND status='pending' "
        "AND expires_at > datetime('now') ORDER BY id DESC", (sid,)).fetchall()]
    db.close()
    return render_template("conv_home.html", eligible=eligible, stats=stats, cap_min=CONV_CAP_MINUTES, cap_enabled=CONV_CAP_ENABLED,
                           sessions=sessions, incoming=incoming, outgoing=outgoing, my_sid=sid,
                           active_topics=active_topics,
                           # the call/record UI now lives on this page
                           warning_mn=CONV_PRIVACY_WARNING_MN, min_min=int(CONV_MIN_MINUTES), max_min=int(CONV_MAX_MINUTES),
                           sample_rate=CONV_SAMPLE_RATE, part_bytes=CONV_PART_BYTES, ring_sec=CONV_CALL_RING_SEC)

@app.route("/conv/invite", methods=["POST"])
@login_required
@conv_gate_required
def conv_invite_send():
    """Invite a partner by speaker ID. Creates a PENDING invitation only — a session is
    created when the invitee accepts (never without their agreement, since a session
    reserves cap minutes for both). Fails fast with the reason if the pair is not
    allowed today."""
    if session.get("role") != "contributor":
        abort(403)
    me = session["speaker_id"]
    target = (request.form.get("invitee") or "").strip().upper().replace(" ", "")
    db = get_db()
    if not _conv_is_eligible(db, me):
        db.close(); flash("Та одоогоор ярианы бичлэгт оролцох боломжгүй байна (профайл болон зөвшөөрлөө шалгана уу).", "warning")
        return redirect(url_for("conv_home"))
    if not target or target == me:
        db.close(); flash("Хамтрагчийн ID-г зөв оруулна уу (өөрийгөө урих боломжгүй).", "error")
        return redirect(url_for("conv_home"))
    if not db.execute("SELECT 1 FROM users WHERE speaker_id=? AND role='contributor'", (target,)).fetchone():
        db.close(); flash(f"{target} гэсэн оролцогч олдсонгүй.", "error")
        return redirect(url_for("conv_home"))
    if not _conv_is_eligible(db, target):
        db.close(); flash(f"{target} одоогоор ярианы бичлэгт оролцох боломжгүй байна (зөвшөөрөл эсвэл профайл дутуу).", "error")
        return redirect(url_for("conv_home"))
    open_n = db.execute("SELECT COUNT(*) FROM conv_invites WHERE creator_speaker_id=? AND status='pending' AND expires_at > datetime('now')", (me,)).fetchone()[0]
    if open_n >= CONV_MAX_OPEN_INVITES:
        db.close(); flash(f"Хүлээгдэж буй урилга хэт олон байна ({open_n}). Хариу ирсний дараа дахин урина уу.", "error")
        return redirect(url_for("conv_home"))
    if db.execute("SELECT 1 FROM conv_invites WHERE status='pending' AND expires_at > datetime('now') AND "
                  "((creator_speaker_id=? AND invitee_speaker_id=?) OR (creator_speaker_id=? AND invitee_speaker_id=?))",
                  (me, target, target, me)).fetchone():
        db.close(); flash(f"{target}-тэй хүлээгдэж буй урилга аль хэдийн байна.", "info")
        return redirect(url_for("conv_home"))
    ok, violations = _conv_pairing_check(db, me, target)
    if not ok:
        db.close(); flash("Энэ хослолоор одоогоор бичлэг хийх боломжгүй: " + " ".join(violations), "error")
        return redirect(url_for("conv_home"))
    topic_sel = (request.form.get("topic_id") or "auto").strip()
    topic_id = None
    if topic_sel != "auto":
        try:
            topic_id = int(topic_sel)
        except ValueError:
            topic_id = None
        if topic_id and not db.execute("SELECT 1 FROM conv_topics WHERE id=? AND status='active'", (topic_id,)).fetchone():
            topic_id = None
    for _ in range(10):
        code = _conv_invite_code()
        if not db.execute("SELECT 1 FROM conv_invites WHERE code=?", (code,)).fetchone():
            break
    db.execute("INSERT INTO conv_invites (code, creator_speaker_id, invitee_speaker_id, status, topic_id, expires_at) "
               "VALUES (?,?,?,'pending',?,datetime('now', ?))", (code, me, target, topic_id, f"+{CONV_INVITE_DAYS} days"))
    db.commit(); db.close()
    flash(f"{target}-д урилга илгээлээ. Хүлээн авмагц яриа үүснэ.", "success")
    return redirect(url_for("conv_home"))

@app.route("/conv/invite/<int:inv_id>/respond", methods=["POST"])
@login_required
@conv_gate_required
def conv_invite_respond(inv_id):
    """Invitee accepts (session created here, pairing rules re-checked) or declines."""
    if session.get("role") != "contributor":
        abort(403)
    me = session["speaker_id"]
    action = (request.form.get("action") or "").strip()
    db = get_db()
    inv = db.execute("SELECT * FROM conv_invites WHERE id=? AND invitee_speaker_id=? AND status='pending'", (inv_id, me)).fetchone()
    if not inv:
        db.close(); flash("Урилга олдсонгүй эсвэл хариу өгөгдсөн байна.", "error")
        return redirect(url_for("conv_home"))
    if action == "decline":
        db.execute("UPDATE conv_invites SET status='declined', responded_at=datetime('now') WHERE id=?", (inv_id,))
        db.commit(); db.close(); flash("Урилгаас татгалзлаа.", "info")
        return redirect(url_for("conv_home"))
    if action != "accept":
        db.close(); abort(400)
    if db.execute("SELECT expires_at <= datetime('now') FROM conv_invites WHERE id=?", (inv_id,)).fetchone()[0]:
        db.execute("UPDATE conv_invites SET status='expired' WHERE id=?", (inv_id,)); db.commit(); db.close()
        flash("Урилгын хугацаа дууссан байна.", "error"); return redirect(url_for("conv_home"))
    try:
        sess_id, sc = _conv_accept_invite(db, inv, me)
    except _ConvCallError as e:
        db.close(); flash(str(e), "error")
        return redirect(url_for("conv_home"))
    db.close()
    flash(f"Яриа {sc} үүслээ. Хамтрагчтайгаа хамт энэ хуудсанд орж «Холбогдох» дарна уу.", "success")
    return redirect(url_for("conv_session_page", sess_id=sess_id))

class _ConvCallError(Exception):
    """A speaker-facing reason why a call/invite cannot become a session."""

def _conv_accept_invite(db, inv, me):
    """Turn a pending invite/call into a session (caller = A, answerer = B). Re-checks
    eligibility and pairing rules at this moment. Commits. Returns (sess_id, session_code)."""
    creator = inv["creator_speaker_id"]
    if not (_conv_is_eligible(db, me) and _conv_is_eligible(db, creator)):
        raise _ConvCallError("Хоёр оролцогч хоёулаа бүртгэлтэй, зөвшөөрлөө баталгаажуулсан байх шаардлагатай.")
    circle = _conv_circle_of(db, creator)
    if circle != _conv_circle_of(db, me):
        raise _ConvCallError("[ДРАФТ] Зөвхөн нэг редакторын багийн оролцогчид хоорондоо ярилцах боломжтой.")
    ok, violations = _conv_pairing_check(db, creator, me)
    if not ok:
        raise _ConvCallError("Энэ хослолоор бичлэг үүсгэх боломжгүй: " + " ".join(violations))
    topic_id = None
    if inv["topic_id"] and db.execute("SELECT 1 FROM conv_topics WHERE id=? AND status='active'", (inv["topic_id"],)).fetchone():
        topic_id = inv["topic_id"]
    if not topic_id:
        topic_id = _conv_pick_topic(db, creator, me)
    if not topic_id:
        raise _ConvCallError("Одоогоор идэвхтэй сэдэв байхгүй байна. Дараа дахин оролдоно уу.")
    sc = _conv_next_session_code(db)
    cur = db.execute(
        "INSERT INTO conv_sessions (session_code, topic_id, status, speaker_a_id, speaker_b_id, invite_code, created_by, editor_id) "
        "VALUES (?,?,'created',?,?,?,?,?)", (sc, topic_id, creator, me, inv["code"], me, (circle or None)))
    sess_id = cur.lastrowid
    db.execute("UPDATE conv_invites SET status='accepted', responded_at=datetime('now'), used_session_id=? WHERE id=?", (sess_id, inv["id"]))
    db.execute("UPDATE conv_topics SET use_count=use_count+1 WHERE id=?", (topic_id,))
    db.commit()
    return sess_id, sc

# ── Speaker: one-tap calling (dial → ring → answer) ───────────────────────────
# Replaces the invite → accept → open-session → connect steps for speakers. A call is a
# conv_invites row with kind='call' that rings for CONV_CALL_RING_SEC seconds; the
# session is created only when the partner answers (nobody is ever put into a session
# they did not agree to). Unanswered/declined calls leave no session. A new attempt
# always starts fresh: dialing or answering first aborts the speaker's own leftover
# created/recording sessions, so failed attempts never count or clutter the list.
CONV_CALL_RING_SEC = 90

def _conv_abort_my_stale(db, me):
    """Abort my sessions still in created/recording whose own track has not passed checks.
    _conv_on_abort does its B2 calls before any write, so the lock is never held on network."""
    rows = db.execute(
        "SELECT s.id FROM conv_sessions s WHERE s.status IN ('created','recording') AND (s.speaker_a_id=? OR s.speaker_b_id=?) "
        "AND s.created_by IN (s.speaker_a_id, s.speaker_b_id) "   # speaker-made only; admin-paired sessions are left alone
        "AND NOT EXISTS (SELECT 1 FROM conv_tracks t WHERE t.session_id=s.id AND t.speaker_id=? AND t.check_status='ok')",
        (me, me, me)).fetchall()
    for r in rows:
        try:
            _conv_on_abort(db, r["id"], f"superseded by a new call attempt from {me}")
        except ValueError:
            pass
    return len(rows)

def _conv_session_brief(db, sess_id, me):
    """What the call page needs to run a session it just entered."""
    s, ch = _conv_participant(db, sess_id, me)
    if not s:
        return None
    t = db.execute("SELECT title_mn, prompt_mn FROM conv_topics WHERE id=?", (s["topic_id"],)).fetchone() if s["topic_id"] else None
    affirmed = bool(db.execute("SELECT 1 FROM conv_affirmations WHERE session_id=? AND speaker_id=?", (sess_id, me)).fetchone())
    return {"id": s["id"], "code": s["session_code"], "status": s["status"], "channel": ch,
            "partner": s["speaker_b_id"] if ch == "A" else s["speaker_a_id"],
            "topic_title": (t["title_mn"] if t else None), "topic_prompt": (t["prompt_mn"] if t else None),
            "affirmed": affirmed}

def _conv_call_fail(db, msg, code=400):
    db.close(); return jsonify({"ok": False, "error": msg}), code

@app.route("/conv/call", methods=["POST"])
@login_required
@conv_gate_required
def conv_call_dial():
    """Dial a partner by speaker ID (JSON). Rings for CONV_CALL_RING_SEC s."""
    if session.get("role") != "contributor":
        return jsonify({"ok": False, "error": "forbidden"}), 403
    me = session["speaker_id"]
    data = request.get_json(silent=True) or {}
    target = (data.get("invitee") or "").strip().upper().replace(" ", "")
    db = get_db()
    if not _conv_is_eligible(db, me):
        return _conv_call_fail(db, "Та одоогоор ярианы бичлэгт оролцох боломжгүй байна (профайл болон зөвшөөрлөө шалгана уу).")
    if not target or target == me:
        return _conv_call_fail(db, "Хамтрагчийн ID-г зөв оруулна уу (өөрийгөө залгах боломжгүй).")
    if not db.execute("SELECT 1 FROM users WHERE speaker_id=? AND role='contributor'", (target,)).fetchone():
        return _conv_call_fail(db, f"{target} гэсэн оролцогч олдсонгүй.")
    if not _conv_is_eligible(db, target):
        return _conv_call_fail(db, f"{target} одоогоор ярианы бичлэгт оролцох боломжгүй байна (зөвшөөрөл эсвэл профайл дутуу).")
    if _conv_circle_of(db, me) != _conv_circle_of(db, target):
        return _conv_call_fail(db, f"[ДРАФТ] {target} таны багийн оролцогч биш байна. Зөвхөн нэг редакторын багийн оролцогчид хоорондоо залгаж болно.", 409)
    if db.execute("SELECT 1 FROM conv_invites WHERE kind='call' AND status='pending' AND expires_at > datetime('now') "
                  "AND creator_speaker_id=? AND invitee_speaker_id=?", (target, me)).fetchone():
        return _conv_call_fail(db, f"{target} таныг яг одоо залгаж байна — «Хариулах» дарна уу.", 409)
    ok, violations = _conv_pairing_check(db, me, target)
    if not ok:
        return _conv_call_fail(db, "Энэ хослолоор одоогоор бичлэг хийх боломжгүй: " + " ".join(violations), 409)
    topic_sel = str(data.get("topic_id") or "auto").strip()
    topic_id = None
    if topic_sel != "auto":
        try:
            topic_id = int(topic_sel)
        except ValueError:
            topic_id = None
        if topic_id and not db.execute("SELECT 1 FROM conv_topics WHERE id=? AND status='active'", (topic_id,)).fetchone():
            topic_id = None
        allowed = _conv_group_topic_ids(db, _conv_circle_of(db, me))
        if topic_id and allowed and topic_id not in allowed:
            topic_id = None          # not in this circle's batch → auto-pick within the batch
    # fresh start: drop my earlier rings and any half-finished session of mine
    _conv_abort_my_stale(db, me)
    db.execute("UPDATE conv_invites SET status='cancelled', responded_at=datetime('now') "
               "WHERE creator_speaker_id=? AND kind='call' AND status='pending'", (me,))
    for _ in range(10):
        code = _conv_invite_code()
        if not db.execute("SELECT 1 FROM conv_invites WHERE code=?", (code,)).fetchone():
            break
    cur = db.execute("INSERT INTO conv_invites (code, creator_speaker_id, invitee_speaker_id, status, topic_id, expires_at, kind) "
                     "VALUES (?,?,?,'pending',?,datetime('now', ?),'call')",
                     (code, me, target, topic_id, f"+{CONV_CALL_RING_SEC} seconds"))
    inv_id = cur.lastrowid
    db.commit(); db.close()
    return jsonify({"ok": True, "invite_id": inv_id, "ring_sec": CONV_CALL_RING_SEC, "partner": target})

@app.route("/conv/call/poll")
@login_required
@conv_gate_required
def conv_call_poll():
    """Read-only poll (no writes — polled every few seconds by every open page).
    incoming: the newest live call ringing me. outgoing: the state of my call (?invite=id)."""
    if session.get("role") != "contributor":
        return jsonify({"ok": False, "error": "forbidden"}), 403
    me = session["speaker_id"]
    db = get_db()
    inc = db.execute(
        "SELECT i.id, i.creator_speaker_id, t.title_mn AS topic_title, "
        "(julianday(i.expires_at) - julianday('now')) * 86400.0 AS left_sec "
        "FROM conv_invites i LEFT JOIN conv_topics t ON t.id=i.topic_id "
        "WHERE i.invitee_speaker_id=? AND i.kind='call' AND i.status='pending' AND i.expires_at > datetime('now') "
        "ORDER BY i.id DESC LIMIT 1", (me,)).fetchone()
    incoming = ([{"id": inc["id"], "from": inc["creator_speaker_id"], "topic_title": inc["topic_title"],
                  "left_sec": int(inc["left_sec"] or 0)}] if inc else [])
    outgoing = None
    try:
        inv_id = int(request.args.get("invite") or 0)
    except ValueError:
        inv_id = 0
    if inv_id:
        o = db.execute("SELECT id, status, used_session_id, invitee_speaker_id, (expires_at > datetime('now')) AS live "
                       "FROM conv_invites WHERE id=? AND creator_speaker_id=? AND kind='call'", (inv_id, me)).fetchone()
        if o:
            st = o["status"]
            if st == "pending" and not o["live"]:
                st = "expired"
            outgoing = {"id": o["id"], "status": st, "partner": o["invitee_speaker_id"]}
            if st == "accepted" and o["used_session_id"]:
                outgoing["session"] = _conv_session_brief(db, o["used_session_id"], me)
    db.close()
    return jsonify({"ok": True, "incoming": incoming, "outgoing": outgoing})

@app.route("/conv/call/<int:inv_id>/answer", methods=["POST"])
@login_required
@conv_gate_required
def conv_call_answer(inv_id):
    """Answer (creates the session; answerer = B) or decline a ringing call. JSON."""
    if session.get("role") != "contributor":
        return jsonify({"ok": False, "error": "forbidden"}), 403
    me = session["speaker_id"]
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "accept").strip()
    db = get_db()
    inv = db.execute("SELECT * FROM conv_invites WHERE id=? AND invitee_speaker_id=? AND status='pending' AND kind='call'",
                     (inv_id, me)).fetchone()
    if not inv:
        return _conv_call_fail(db, "Дуудлага олдсонгүй эсвэл аль хэдийн дууссан байна.", 404)
    if action == "decline":
        db.execute("UPDATE conv_invites SET status='declined', responded_at=datetime('now') WHERE id=?", (inv_id,))
        db.commit(); db.close()
        return jsonify({"ok": True, "declined": True})
    if db.execute("SELECT expires_at <= datetime('now') FROM conv_invites WHERE id=?", (inv_id,)).fetchone()[0]:
        db.execute("UPDATE conv_invites SET status='expired' WHERE id=?", (inv_id,)); db.commit()
        return _conv_call_fail(db, "Дуудлагын хугацаа дууссан байна — хамтрагчаа дахин залгуулна уу.", 410)
    # fresh start for me too: my own rings and half-finished sessions go away first
    _conv_abort_my_stale(db, me)
    db.execute("UPDATE conv_invites SET status='cancelled', responded_at=datetime('now') "
               "WHERE creator_speaker_id=? AND kind='call' AND status='pending'", (me,))
    try:
        sess_id, sc = _conv_accept_invite(db, inv, me)
    except _ConvCallError as e:
        db.commit()
        return _conv_call_fail(db, str(e), 409)
    brief = _conv_session_brief(db, sess_id, me)
    db.close()
    return jsonify({"ok": True, "session": brief})

@app.route("/conv/call/<int:inv_id>/cancel", methods=["POST"])
@login_required
@conv_gate_required
def conv_call_cancel(inv_id):
    """Caller hangs up while ringing. JSON."""
    me = session.get("speaker_id")
    db = get_db()
    db.execute("UPDATE conv_invites SET status='cancelled', responded_at=datetime('now') "
               "WHERE id=? AND creator_speaker_id=? AND status='pending' AND kind='call'", (inv_id, me))
    db.commit(); db.close()
    return jsonify({"ok": True})

@app.route("/conv/invite/<int:inv_id>/cancel", methods=["POST"])
@login_required
@conv_gate_required
def conv_invite_cancel(inv_id):
    me = session.get("speaker_id")
    db = get_db()
    db.execute("UPDATE conv_invites SET status='cancelled', responded_at=datetime('now') WHERE id=? AND creator_speaker_id=? AND status='pending'", (inv_id, me))
    db.commit(); db.close()
    flash("Урилгыг цуцаллаа.", "info")
    return redirect(url_for("conv_home"))

# ── Speaker: session page ─────────────────────────────────────────────────────
@app.route("/conv/session/<int:sess_id>")
@login_required
@conv_gate_required
def conv_session_page(sess_id):
    if session.get("role") != "contributor":
        abort(403)
    me = session["speaker_id"]
    db = get_db()
    _conv_sweep_incomplete(db)
    s, ch = _conv_participant(db, sess_id, me)
    if not s:
        db.close(); abort(404)
    partner = s["speaker_b_id"] if ch == "A" else s["speaker_a_id"]
    my_track = db.execute("SELECT * FROM conv_tracks WHERE session_id=? AND channel=?", (sess_id, ch)).fetchone()
    other_track = db.execute("SELECT * FROM conv_tracks WHERE session_id=? AND channel=?",
                             (sess_id, "B" if ch == "A" else "A")).fetchone()
    affirmed = bool(db.execute("SELECT 1 FROM conv_affirmations WHERE session_id=? AND speaker_id=?",
                               (sess_id, me)).fetchone())
    db.close()
    my_ok = bool(my_track and my_track["check_status"] == "ok")
    other_rejected = bool(other_track and other_track["check_status"] == "rejected")
    # A fresh take needs BOTH sides: allow recording unless my track passed and the
    # partner's is still in progress or also passed.
    can_record = s["status"] in ("created", "recording", "incomplete") and (not my_ok or other_rejected)
    active_topics = []
    if ch == "A" and s["status"] == "created":
        _d = get_db(); active_topics = _d.execute("SELECT id, title_mn FROM conv_topics WHERE status='active' ORDER BY title_mn").fetchall(); _d.close()
    return render_template("conv_session.html", s=s, channel=ch, partner=partner, my_sid=me,
                           my_track=my_track, other_track=other_track, affirmed=affirmed,
                           can_record=can_record, active_topics=active_topics,
                           warning_mn=CONV_PRIVACY_WARNING_MN, start_instruction_mn=CONV_START_INSTRUCTION_MN,
                           min_min=int(CONV_MIN_MINUTES), max_min=int(CONV_MAX_MINUTES),
                           sample_rate=CONV_SAMPLE_RATE, part_bytes=CONV_PART_BYTES)

@app.route("/conv/session/<int:sess_id>/topic", methods=["POST"])
@login_required
@conv_gate_required
def conv_session_topic(sess_id):
    """Initiator (Speaker A) changes the topic before recording starts."""
    me = session.get("speaker_id")
    db = get_db()
    s, ch = _conv_participant(db, sess_id, me)
    if not s or ch != "A" or s["status"] != "created":
        db.close(); flash("Сэдвийг зөвхөн урьсан хүн, бичлэг эхлэхээс өмнө солих боломжтой.", "error")
        return redirect(url_for("conv_session_page", sess_id=sess_id))
    try:
        new_id = int(request.form.get("topic_id") or 0)
    except ValueError:
        new_id = 0
    if not new_id or not db.execute("SELECT 1 FROM conv_topics WHERE id=? AND status='active'", (new_id,)).fetchone():
        db.close(); flash("Сэдэв олдсонгүй.", "error")
        return redirect(url_for("conv_session_page", sess_id=sess_id))
    if new_id != s["topic_id"]:
        if s["topic_id"]:
            db.execute("UPDATE conv_topics SET use_count=MAX(0, use_count-1) WHERE id=?", (s["topic_id"],))
        db.execute("UPDATE conv_topics SET use_count=use_count+1 WHERE id=?", (new_id,))
        db.execute("UPDATE conv_sessions SET topic_id=? WHERE id=?", (new_id, sess_id))
        db.commit()
        flash("Сэдэв солигдлоо.", "success")
    db.close()
    return redirect(url_for("conv_session_page", sess_id=sess_id))

@app.route("/conv/session/<int:sess_id>/affirm", methods=["POST"])
@login_required
@conv_gate_required
def conv_session_affirm(sess_id):
    me = session.get("speaker_id")
    db = get_db()
    s, ch = _conv_participant(db, sess_id, me)
    if not s:
        db.close(); return jsonify({"ok": False, "error": "not_participant"}), 403
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr) or "").split(",")[0].strip()
    db.execute("INSERT INTO conv_affirmations (session_id, speaker_id, ip) VALUES (?,?,?)", (sess_id, me, ip))
    db.commit(); db.close()
    return jsonify({"ok": True})

@app.route("/conv/session/<int:sess_id>/upload/init", methods=["POST"])
@login_required
@conv_gate_required
def conv_upload_init(sess_id):
    """Create the B2 multipart upload for this speaker's track. Requires the per-session
    affirmation. Moves the session created -> recording on first start."""
    me = session.get("speaker_id")
    db = get_db()
    s, ch = _conv_participant(db, sess_id, me)
    if not s:
        db.close(); return jsonify({"ok": False, "error": "not_participant"}), 403
    if s["status"] not in ("created", "recording", "incomplete"):
        db.close(); return jsonify({"ok": False, "error": f"session is {s['status']}"}), 409
    if not db.execute("SELECT 1 FROM conv_affirmations WHERE session_id=? AND speaker_id=?", (sess_id, me)).fetchone():
        db.close(); return jsonify({"ok": False, "error": "affirmation_required"}), 403
    data = request.get_json(silent=True) or {}
    try:
        client_start_ms = int(data.get("client_start_ms") or 0)
    except (TypeError, ValueError):
        client_start_ms = 0
    try:
        clock_skew_ms = int(data.get("clock_skew_ms")) if data.get("clock_skew_ms") is not None else None
    except (TypeError, ValueError):
        clock_skew_ms = None
    key = _conv_track_key(s["session_code"], me)
    b2 = get_b2()
    prev = db.execute("SELECT upload_id FROM conv_tracks WHERE session_id=? AND channel=?", (sess_id, ch)).fetchone()
    if prev and prev["upload_id"]:
        try: b2.abort_multipart_upload(Bucket=B2_BUCKET_NAME, Key=key, UploadId=prev["upload_id"])
        except Exception: pass
    mp = b2.create_multipart_upload(Bucket=B2_BUCKET_NAME, Key=key, ContentType="audio/wav")
    upload_id = mp["UploadId"]
    db.execute(
        "INSERT INTO conv_tracks (session_id, speaker_id, channel, b2_key, client_start_ms, clock_skew_ms, check_status, upload_id) "
        "VALUES (?,?,?,?,?,?,'uploading',?) "
        "ON CONFLICT(session_id, channel) DO UPDATE SET b2_key=excluded.b2_key, client_start_ms=excluded.client_start_ms, "
        "  clock_skew_ms=excluded.clock_skew_ms, client_stop_ms=NULL, duration_sec=NULL, size_bytes=NULL, rms=NULL, "
        "  check_status='uploading', check_error=NULL, uploaded_at=NULL, upload_id=excluded.upload_id",
        (sess_id, me, ch, key, client_start_ms, clock_skew_ms, upload_id))
    # Both phones call this within milliseconds of each other (shared Start). Re-read the
    # status now and treat "already recording" as success rather than an illegal move.
    cur = db.execute("SELECT status FROM conv_sessions WHERE id=?", (sess_id,)).fetchone()["status"]
    if cur == "created":
        try:
            _conv_transition(db, sess_id, "recording",
                             recording_started_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        except ValueError:
            pass   # the partner's request got there first
    db.commit(); db.close()
    return jsonify({"ok": True, "upload_id": upload_id, "key": key, "part_bytes": CONV_PART_BYTES})

@app.route("/conv/session/<int:sess_id>/upload/sign", methods=["POST"])
@login_required
@conv_gate_required
def conv_upload_sign(sess_id):
    """Presigned PUT URL for one part; the browser uploads the bytes straight to B2."""
    me = session.get("speaker_id")
    data = request.get_json(silent=True) or {}
    db = get_db()
    s, ch = _conv_participant(db, sess_id, me)
    t = db.execute("SELECT * FROM conv_tracks WHERE session_id=? AND channel=?", (sess_id, ch)).fetchone() if s else None
    db.close()
    if not s or not t or not t["upload_id"]:
        return jsonify({"ok": False, "error": "no_upload"}), 409
    try:
        pn = int(data.get("part_number"))
        assert 1 <= pn <= 10000
    except Exception:
        return jsonify({"ok": False, "error": "bad_part"}), 400
    url = get_b2().generate_presigned_url(
        "upload_part",
        Params={"Bucket": B2_BUCKET_NAME, "Key": t["b2_key"], "UploadId": t["upload_id"], "PartNumber": pn},
        ExpiresIn=3600, HttpMethod="PUT")
    return jsonify({"ok": True, "url": url})

@app.route("/conv/session/<int:sess_id>/upload/complete", methods=["POST"])
@login_required
@conv_gate_required
def conv_upload_complete(sess_id):
    """Assemble the multipart object, record client timing, and start the background check."""
    me = session.get("speaker_id")
    data = request.get_json(silent=True) or {}
    db = get_db()
    s, ch = _conv_participant(db, sess_id, me)
    t = db.execute("SELECT * FROM conv_tracks WHERE session_id=? AND channel=?", (sess_id, ch)).fetchone() if s else None
    if not s or not t or not t["upload_id"]:
        db.close(); return jsonify({"ok": False, "error": "no_upload"}), 409
    parts = data.get("parts") or []
    try:
        parts = sorted(({"PartNumber": int(p["PartNumber"]), "ETag": str(p["ETag"])} for p in parts),
                       key=lambda p: p["PartNumber"])
        assert parts
    except Exception:
        db.close(); return jsonify({"ok": False, "error": "bad_parts"}), 400
    try:
        get_b2().complete_multipart_upload(Bucket=B2_BUCKET_NAME, Key=t["b2_key"], UploadId=t["upload_id"],
                                           MultipartUpload={"Parts": parts})
    except Exception as e:
        db.close(); return jsonify({"ok": False, "error": f"complete failed: {e}"}), 502
    try: stop_ms = int(data.get("client_stop_ms") or 0)
    except (TypeError, ValueError): stop_ms = 0
    try: gain_db = float(data.get("gain_db")) if data.get("gain_db") is not None else None
    except (TypeError, ValueError): gain_db = None
    db.execute("UPDATE conv_tracks SET client_stop_ms=?, gain_db=?, uploaded_at=datetime('now'), check_status='pending', upload_id=NULL WHERE id=?",
               (stop_ms, gain_db, t["id"]))
    db.commit(); db.close()
    threading.Thread(target=_conv_check_track, args=(t["id"],), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/conv/session/<int:sess_id>/abort", methods=["POST"])
@login_required
@conv_gate_required
def conv_session_speaker_abort(sess_id):
    """Speaker stopped before the minimum (§4.4) or gave up: abort any open upload and
    mark the session aborted (does not count; pair can retry as a new session)."""
    me = session.get("speaker_id")
    db = get_db()
    s, ch = _conv_participant(db, sess_id, me)
    if not s:
        db.close(); return jsonify({"ok": False, "error": "not_participant"}), 403
    try:
        _conv_on_abort(db, sess_id, "speaker abort")
        db.commit(); db.close()
        return jsonify({"ok": True})
    except ValueError as e:
        db.commit(); db.close()
        return jsonify({"ok": False, "error": str(e)}), 409

@app.route("/conv/session/<int:sess_id>/status")
@login_required
@conv_gate_required
def conv_session_status(sess_id):
    me = session.get("speaker_id")
    db = get_db()
    _conv_sweep_incomplete(db)
    s, ch = _conv_participant(db, sess_id, me)
    if not s:
        db.close(); return jsonify({"ok": False}), 403
    def trk(c):
        t = db.execute("SELECT check_status, check_error, duration_sec, uploaded_at FROM conv_tracks WHERE session_id=? AND channel=?", (sess_id, c)).fetchone()
        return dict(t) if t else None
    out = {"ok": True, "status": s["status"], "me": trk(ch), "partner": trk("B" if ch == "A" else "A")}
    db.close()
    return jsonify(out)

# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL — Stage B (drafting, speaker self-correction, review)
# ══════════════════════════════════════════════════════════════════════════════

def _conv_import_utterances(db, sess_id, utts, model_version, offsets_ms, strict=False):
    """Replace this session's DRAFT utterances. utts: dicts {channel,start_sec,end_sec,text}
    on the session timeline. Per channel they must be sorted and non-overlapping:
    strict=True rejects (ValueError with index, spec §7.2); strict=False clamps.
    Idempotent (draft rows are replaced wholesale)."""
    s = db.execute("SELECT * FROM conv_sessions WHERE id=?", (sess_id,)).fetchone()
    spk = {"A": s["speaker_a_id"], "B": s["speaker_b_id"]}
    by_ch = {"A": [], "B": []}
    for i, u in enumerate(utts):
        ch = (u.get("channel") or "").upper()
        if ch not in by_ch:
            raise ValueError(f"utterance {i}: bad channel")
        try:
            st, en = float(u.get("start_sec")), float(u.get("end_sec"))
        except (TypeError, ValueError):
            raise ValueError(f"utterance {i}: bad times")
        if en <= st or st < 0:
            raise ValueError(f"utterance {i}: end must be after start")
        by_ch[ch].append([st, en, (u.get("text") or "").strip(), i])
    for ch, rows in by_ch.items():
        for k in range(1, len(rows)):
            if rows[k][0] < rows[k-1][1]:
                if strict:
                    raise ValueError(f"utterance {rows[k][3]}: overlaps previous in channel {ch}")
                rows[k][0] = rows[k-1][1]
                if rows[k][1] <= rows[k][0]:
                    rows[k][1] = rows[k][0] + 0.05
        rows.sort(key=lambda r: r[0])
    db.execute("DELETE FROM conv_utterances WHERE session_id=? AND source='draft'", (sess_id,))
    db.execute("DELETE FROM conv_utterances WHERE session_id=?", (sess_id,))   # drafts replace everything pre-edit
    for ch, rows in by_ch.items():
        for seq, (st, en, text, _) in enumerate(rows, start=1):
            db.execute(
                "INSERT INTO conv_utterances (session_id, channel, speaker_id, seq, start_sec, end_sec, draft_text, text, source) "
                "VALUES (?,?,?,?,?,?,?,?,'draft')", (sess_id, ch, spk[ch], seq, round(st, 3), round(en, 3), text, text))
    db.execute("UPDATE conv_sessions SET model_version=?, alignment_offset_json=? WHERE id=?",
               (model_version, json.dumps(offsets_ms or {}), sess_id))
    return sum(len(r) for r in by_ch.values())

def _conv_wav16k_chunks(b2, key, size_bytes, chunk_sec):
    """Stream a 48 kHz mono 16-bit WAV from B2 in ranged reads, decimate 3:1 (with a
    3-tap average) to 16 kHz, and yield (offset_sec, wav_bytes) chunks of chunk_sec.
    Memory stays bounded to one chunk (~19 MB for 10 min)."""
    import array, struct
    def hdr(n):
        return (b'RIFF'+struct.pack('<I',36+n)+b'WAVE'+b'fmt '+struct.pack('<IHHIIHH',16,1,1,16000,32000,2,16)+b'data'+struct.pack('<I',n))
    per_chunk = 16000 * chunk_sec               # 16 k samples per chunk
    out = array.array('h'); off = 44; read = 2 * 1024 * 1024; t0 = 0.0; carry = array.array('h')
    while off < size_bytes:
        end = min(size_bytes - 1, off + read - 1)
        raw = b2.get_object(Bucket=B2_BUCKET_NAME, Key=key, Range=f"bytes={off}-{end}")["Body"].read()
        off = end + 1
        if len(raw) % 2: raw = raw[:-1]
        a = array.array('h'); a.frombytes(raw)
        if carry: a = carry + a; carry = array.array('h')
        n3 = (len(a) // 3) * 3
        for i in range(0, n3, 3):
            out.append(int((a[i] + a[i+1] + a[i+2]) / 3))
        if n3 < len(a): carry = a[n3:]
        while len(out) >= per_chunk:
            piece = out[:per_chunk]; del out[:per_chunk]
            pb = piece.tobytes(); yield t0, hdr(len(pb)) + pb
            t0 += chunk_sec
    if len(out):
        pb = out.tobytes(); yield t0, hdr(len(pb)) + pb

def _conv_draft_session(sess_id):
    """Built-in drafter: OpenAI transcription (language=mn) per track, chunked, on a
    background thread. On success -> 'drafted' with draft utterances on the session
    timeline (channel B shifted by the client-start hint). On failure -> back to
    'uploaded' with draft_error, re-runnable from the admin page."""
    db = get_db()
    try:
        s = db.execute("SELECT * FROM conv_sessions WHERE id=?", (sess_id,)).fetchone()
        if not s or s["status"] != "uploaded":
            return
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI is not configured (OPENAI_API_KEY missing)")
        _conv_transition(db, sess_id, "drafting", drafting_started_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), draft_error=None)
        db.commit()
        tracks = {t["channel"]: t for t in db.execute("SELECT * FROM conv_tracks WHERE session_id=?", (sess_id,)).fetchall()}
        b2 = get_b2()
        # alignment hint: B relative to A from client start clocks (hint only; clamp to ±60 s)
        off = {"A": 0, "B": 0}
        if tracks.get("A") and tracks.get("B") and tracks["A"]["client_start_ms"] and tracks["B"]["client_start_ms"]:
            # B's clock converted to A's clock via the ping-measured skew (A_clock - B_clock).
            skew = int(tracks["B"]["clock_skew_ms"] or 0)
            d = int((tracks["B"]["client_start_ms"] + skew) - tracks["A"]["client_start_ms"])
            off["B"] = max(-60000, min(60000, d))
        utts = []
        # OpenAI's transcription API rejects `language="mn"` ("Language 'mn' is not supported",
        # code unsupported_language) — Mongolian is one of Whisper's 99 languages but is not on
        # the API's accepted-code list, for whisper-1 and the gpt-4o-transcribe models alike.
        # So: try with the language hint (harmless if it is ever accepted); on that specific
        # rejection, drop the hint and steer the model with a Mongolian prompt instead.
        model = CONV_DRAFT_MODEL
        lang_kw = {"language": "mn"}
        MN_PROMPT = "Энэ бол хоёр хүний монгол хэл дээрх чөлөөт яриа."   # steers language detection to Mongolian
        def _call(f, extra):
            f.seek(0)
            try:
                r = client.audio.transcriptions.create(model=model, file=f, response_format="verbose_json",
                                                       timestamp_granularities=["segment"], **extra)
                segs = getattr(r, "segments", None)
                if segs is None and isinstance(r, dict): segs = r.get("segments")
                return r, segs
            except Exception as e1:
                if "unsupported_language" in str(e1) or "not supported" in str(e1): raise
                f.seek(0)
                return client.audio.transcriptions.create(model=model, file=f, **extra), None   # models without timestamps
        def _transcribe(f):
            nonlocal lang_kw
            try:
                return _call(f, dict(lang_kw, prompt=MN_PROMPT))
            except Exception as e:
                if lang_kw and ("unsupported_language" in str(e) or "not supported" in str(e)):
                    print(f"[CONV DRAFT] {model} rejected language 'mn' — continuing without the language hint for {s['session_code']}")
                    lang_kw = {}
                    return _call(f, {"prompt": MN_PROMPT})
                raise
        for ch in ("A", "B"):
            t = tracks.get(ch)
            if not t or not t["b2_key"]:
                raise RuntimeError(f"track {ch} missing")
            size = int(b2.head_object(Bucket=B2_BUCKET_NAME, Key=t["b2_key"])["ContentLength"])
            for t0, wav in _conv_wav16k_chunks(b2, t["b2_key"], size, CONV_DRAFT_CHUNK_SEC):
                f = io.BytesIO(wav); f.name = f"{s['session_code']}_{ch}_{int(t0)}.wav"
                r, segs = _transcribe(f)
                base = t0 + off[ch] / 1000.0
                if segs:
                    for sg in segs:
                        g = sg if isinstance(sg, dict) else {"start": getattr(sg, "start", 0), "end": getattr(sg, "end", 0), "text": getattr(sg, "text", "")}
                        st, en = float(g.get("start", 0)), float(g.get("end", 0))
                        if en <= st: continue
                        utts.append({"channel": ch, "start_sec": max(0.0, base + st), "end_sec": max(0.0, base + en), "text": (g.get("text") or "").strip()})
                else:
                    text = getattr(r, "text", None) or (r.get("text") if isinstance(r, dict) else "") or ""
                    dur = (len(wav) - 44) / 32000.0
                    utts.append({"channel": ch, "start_sec": max(0.0, base), "end_sec": max(0.0, base + dur), "text": text.strip()})
        if not utts:
            raise RuntimeError("transcription returned no utterances (silent audio or ASR failure)")
        n = _conv_import_utterances(db, sess_id, utts, f"openai:{model}", off, strict=False)
        _conv_transition(db, sess_id, "drafted", drafted_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        db.commit()
        print(f"[CONV DRAFT] session {sess_id}: {n} utterances drafted")
    except Exception as e:
        import traceback; print(f"[CONV DRAFT] session {sess_id} failed:\n" + traceback.format_exc())
        try:
            cur = db.execute("SELECT status FROM conv_sessions WHERE id=?", (sess_id,)).fetchone()
            if cur and cur["status"] == "drafting":
                _conv_transition(db, sess_id, "uploaded", draft_error=str(e)[:500])
            else:
                db.execute("UPDATE conv_sessions SET draft_error=? WHERE id=?", (str(e)[:500], sess_id))
            db.commit()
        except Exception:
            pass
    finally:
        db.close()

def _conv_maybe_autodraft(sess_id):
    if CONV_AUTO_DRAFT and get_openai_client() is not None:
        threading.Thread(target=_conv_draft_session, args=(sess_id,), daemon=True).start()
        return
    if not CONV_AUTO_DRAFT:
        return   # intended: the GPU worker (conv_worker/) drafts via /api/conv/pending; session waits in 'uploaded'
    # OpenAI drafting is on but cannot run: say so on the session instead of failing silently.
    why = "OpenAI is not configured (OPENAI_API_KEY missing on Railway)"
    db = get_db()
    try:
        db.execute("UPDATE conv_sessions SET draft_error=? WHERE id=? AND status='uploaded' AND draft_error IS NULL", (why, sess_id))
        db.commit()
    finally:
        db.close()

def _conv_job_auth():
    if not CONV_JOB_API_KEY:
        abort(503)
    if request.headers.get("X-API-Key") != CONV_JOB_API_KEY:
        abort(401)

@app.route("/api/conv/pending")
def conv_api_pending():
    """Spec §7.1: hand 'uploaded' sessions to an external draft job (atomically -> drafting)."""
    _conv_job_auth()
    limit = max(1, min(50, int(request.args.get("limit", "50") or 50)))
    db = get_db()
    # lazy revert: stuck in drafting > 24 h -> uploaded
    for r in db.execute("SELECT id FROM conv_sessions WHERE status='drafting' AND drafting_started_at <= datetime('now','-24 hours')").fetchall():
        try: _conv_transition(db, r["id"], "uploaded", draft_error="drafting timed out (24 h)")
        except ValueError: pass
    rows = db.execute("SELECT * FROM conv_sessions WHERE status='uploaded' ORDER BY id ASC LIMIT ?", (limit,)).fetchall()
    out = []
    for s in rows:
        _conv_transition(db, s["id"], "drafting", drafting_started_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), draft_error=None)
        tr = db.execute("SELECT speaker_id, channel, b2_key, client_start_ms, clock_skew_ms, duration_sec FROM conv_tracks WHERE session_id=? ORDER BY channel", (s["id"],)).fetchall()
        out.append({"session_id": s["session_code"], "topic_id": s["topic_id"], "tracks": [dict(t) for t in tr]})
    db.commit(); db.close()
    return jsonify({"sessions": out})

@app.route("/api/conv/import", methods=["POST"])
def conv_api_import():
    """Spec §7.2: import aligned draft utterances for a 'drafting' session."""
    _conv_job_auth()
    data = request.get_json(silent=True) or {}
    db = get_db()
    s = db.execute("SELECT * FROM conv_sessions WHERE session_code=?", (data.get("session_id"),)).fetchone()
    if not s:
        db.close(); return jsonify({"ok": False, "error": "unknown session"}), 404
    if s["status"] != "drafting":
        db.close(); return jsonify({"ok": False, "error": f"session is {s['status']}, not drafting"}), 409
    try:
        n = _conv_import_utterances(db, s["id"], data.get("utterances") or [], data.get("model_version") or "external",
                                    data.get("alignment_offset_ms") or {}, strict=True)
    except ValueError as e:
        db.rollback(); db.close()
        return jsonify({"ok": False, "error": str(e)}), 400
    _conv_transition(db, s["id"], "drafted", drafted_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    db.commit(); db.close()
    return jsonify({"ok": True, "utterances": n})

@app.route("/api/conv/session/<code>")
def conv_api_session(code):
    """Spec §7.3: read-only session + tracks + utterances."""
    _conv_job_auth()
    db = get_db()
    s = db.execute("SELECT * FROM conv_sessions WHERE session_code=?", (code,)).fetchone()
    if not s:
        db.close(); return jsonify({"ok": False}), 404
    tr = [dict(t) for t in db.execute("SELECT * FROM conv_tracks WHERE session_id=?", (s["id"],)).fetchall()]
    ut = [dict(u) for u in db.execute("SELECT * FROM conv_utterances WHERE session_id=? ORDER BY start_sec, channel", (s["id"],)).fetchall()]
    db.close()
    return jsonify({"session": dict(s), "tracks": tr, "utterances": ut})

@app.route("/admin/conv/sessions/<int:sess_id>/draft", methods=["POST"])
@login_required
@admin_required
def conv_session_draft_now(sess_id):
    db = get_db(); s = db.execute("SELECT status FROM conv_sessions WHERE id=?", (sess_id,)).fetchone(); db.close()
    if not s or s["status"] != "uploaded":
        flash("Drafting can only start from status 'uploaded'.", "error")
        return redirect(url_for("conv_session_detail", sess_id=sess_id))
    if get_openai_client() is None:
        flash("OpenAI is not configured — set OPENAI_API_KEY (or use the external job API).", "error")
        return redirect(url_for("conv_session_detail", sess_id=sess_id))
    threading.Thread(target=_conv_draft_session, args=(sess_id,), daemon=True).start()
    flash("Drafting started in the background. Refresh in a few minutes.", "info")
    return redirect(url_for("conv_session_detail", sess_id=sess_id))

# ── Speaker self-correction ───────────────────────────────────────────────────
CONV_EDITABLE_STATUSES = ("drafted", "in_edit", "rejected")

def _conv_presign(key, secs=3600):
    return get_b2().generate_presigned_url("get_object", Params={"Bucket": B2_BUCKET_NAME, "Key": key}, ExpiresIn=secs)

def _conv_track_progress(db, sess_id, ch):
    tot = db.execute("SELECT COUNT(*) FROM conv_utterances WHERE session_id=? AND channel=?", (sess_id, ch)).fetchone()[0]
    done = db.execute("SELECT COUNT(*) FROM conv_utterances WHERE session_id=? AND channel=? AND (accepted=1 OR source='edited')", (sess_id, ch)).fetchone()[0]
    return tot, done

def _conv_session_progress(db, sess_id):
    """Editing progress over BOTH channels — the initiator (Speaker A) edits the whole
    conversation; the invitee only records. Pay-release rule (Stage C): B's recording pay
    is earned at 'uploaded' (uploaded_at); A's recording + editing pay at 'approved'."""
    tot = db.execute("SELECT COUNT(*) FROM conv_utterances WHERE session_id=?", (sess_id,)).fetchone()[0]
    done = db.execute("SELECT COUNT(*) FROM conv_utterances WHERE session_id=? AND (accepted=1 OR source='edited')", (sess_id,)).fetchone()[0]
    return tot, done

def conv_editor_required(f):
    """Editors and admin: the people who correct and approve conversations."""
    @wraps(f)
    def d(*a, **k):
        if session.get("role") not in ("admin", "editor"): abort(403)
        return f(*a, **k)
    return d

def _conv_owns_session(s):
    """Admin may edit anything. An editor may edit sessions of her own circle."""
    if session.get("role") == "admin":
        return True
    return s["editor_id"] is not None and int(s["editor_id"]) == int(session.get("user_id") or -1)

def _conv_queue_rows(db, me_role, me_id):
    """Sessions in an editing state for this person's circle (admin: unassigned circle)."""
    where = "s.editor_id IS NULL" if me_role == "admin" else "s.editor_id=?"
    params = () if me_role == "admin" else (me_id,)
    return db.execute(
        "SELECT s.*, t.title_mn AS topic_title, "
        "  (SELECT COUNT(*) FROM conv_utterances u WHERE u.session_id=s.id) AS n_utt, "
        "  (SELECT COUNT(*) FROM conv_utterances u WHERE u.session_id=s.id AND u.accepted=1) AS n_ok, "
        f"  (s.status IN ('in_edit','rejected','drafted') AND COALESCE(s.drafted_at, s.uploaded_at) <= datetime('now','-{CONV_EDIT_DEADLINE_DAYS} days')) AS stale "
        f"FROM conv_sessions s LEFT JOIN conv_topics t ON t.id=s.topic_id "
        f"WHERE {where} AND s.status IN ('rejected','drafted','in_edit','in_review','approved','uploaded','drafting') "
        "ORDER BY CASE s.status WHEN 'rejected' THEN 0 WHEN 'drafted' THEN 1 WHEN 'in_edit' THEN 2 WHEN 'uploaded' THEN 3 WHEN 'drafting' THEN 3 WHEN 'in_review' THEN 4 ELSE 5 END, s.id DESC LIMIT 200",
        params).fetchall()

@app.route("/conv/queue")
@login_required
@conv_editor_required
def conv_queue():
    """Editor's (or admin's own circle's) conversation work list."""
    _conv_housekeeping()
    db = get_db()
    rows = _conv_queue_rows(db, session.get("role"), session.get("user_id"))
    db.close()
    return render_template("conv_queue.html", rows=rows)

@app.route("/conv/session/<int:sess_id>/edit")
@login_required
@conv_editor_required
def conv_edit_page(sess_id):
    db = get_db()
    s = _conv_session_row(db, sess_id)
    if not s:
        db.close(); abort(404)
    if not _conv_owns_session(s):
        db.close(); abort(403)
    if s["status"] not in CONV_EDITABLE_STATUSES + ("in_review", "approved"):
        db.close(); flash("Энэ ярианы текст хараахан бэлэн болоогүй байна.", "info")
        return redirect(url_for("conv_queue"))
    if s["status"] in ("drafted", "rejected"):
        _conv_transition(db, sess_id, "in_edit"); db.commit()
        s = _conv_session_row(db, sess_id)
    tracks = {t["channel"]: t for t in db.execute("SELECT * FROM conv_tracks WHERE session_id=?", (sess_id,)).fetchall()}
    utts = db.execute("SELECT * FROM conv_utterances WHERE session_id=? ORDER BY start_sec, channel", (sess_id,)).fetchall()
    tot, done = _conv_session_progress(db, sess_id)
    fillers = [r["token"] for r in db.execute("SELECT token FROM conv_filler_tokens WHERE active=1 ORDER BY id").fetchall()]
    db.close()
    urls = {c: (_conv_presign(t["b2_key"]) if t and t["b2_key"] else None) for c, t in tracks.items()}
    editable = s["status"] == "in_edit"
    return render_template("conv_edit.html", s=s, utts=utts, urls=urls, total=tot, done=done, editable=editable,
                           rules=CONV_SPEAKER_RULES_MN, fillers=fillers,
                           reject_reason=(s["reject_reason"] if s["status"] == "in_edit" and s["reject_reason"] else None))

def _apply_utt_update(db, u, data, who):
    text = (data.get("text") if data.get("text") is not None else u["text"]) or ""
    text = text.strip()
    flags = {k: (1 if data.get(k) else 0) if k in data else u[k] for k in ("pii", "unclear", "nonspeech_only", "bad_audio")}
    source = "edited" if text != (u["draft_text"] or "").strip() else u["source"]
    db.execute("UPDATE conv_utterances SET text=?, source=?, pii=?, unclear=?, nonspeech_only=?, bad_audio=?, accepted=1, "
               "edited_by=?, edited_at=datetime('now') WHERE id=?",
               (text, source, flags["pii"], flags["unclear"], flags["nonspeech_only"], flags["bad_audio"], who, u["id"]))

@app.route("/conv/session/<int:sess_id>/utt/<int:utt_id>", methods=["POST"])
@login_required
@conv_editor_required
def conv_edit_utt(sess_id, utt_id):
    data = request.get_json(silent=True) or {}
    db = get_db()
    s = _conv_session_row(db, sess_id)
    if not s or s["status"] != "in_edit" or not _conv_owns_session(s):
        db.close(); return jsonify({"ok": False, "error": "not_editable"}), 409
    u = db.execute("SELECT * FROM conv_utterances WHERE id=? AND session_id=?", (utt_id, sess_id)).fetchone()
    if not u:
        db.close(); return jsonify({"ok": False, "error": "not_found"}), 404
    _apply_utt_update(db, u, data, f"editor:{session.get('username')}")
    db.commit()
    tot, done = _conv_session_progress(db, sess_id)
    db.close()
    return jsonify({"ok": True, "total": tot, "done": done})

@app.route("/conv/session/<int:sess_id>/edit/done", methods=["POST"])
@login_required
@conv_editor_required
def conv_edit_done(sess_id):
    db = get_db()
    s = _conv_session_row(db, sess_id)
    if not s or s["status"] != "in_edit" or not _conv_owns_session(s):
        db.close(); flash("Одоогоор дуусгах боломжгүй.", "error"); return redirect(url_for("conv_queue"))
    tot, done = _conv_session_progress(db, sess_id)
    if done < tot:
        db.close(); flash(f"Бүх мөрийг шалгана уу ({done}/{tot} шалгагдсан).", "error")
        return redirect(url_for("conv_edit_page", sess_id=sess_id))
    db.execute("UPDATE conv_tracks SET edit_status='done', edit_done_at=datetime('now') WHERE session_id=?", (sess_id,))
    _conv_transition(db, sess_id, "in_review")
    db.commit(); db.close()
    flash("Яриа админд хянуулахаар илгээгдлээ.", "success")
    return redirect(url_for("conv_queue"))

# ── Admin approval (final step, like read-speech final review) ───────────────
def reviewer_required(f):
    @wraps(f)
    def d(*a, **k):
        if session.get("role") != "admin": abort(403)
        return f(*a, **k)
    return d

@app.route("/conv/review")
@login_required
@reviewer_required
def conv_review_list():
    _conv_housekeeping()
    db = get_db()
    q = ("SELECT s.*, t.title_mn AS topic_title, "
         "  (SELECT username FROM users WHERE id=s.editor_id) AS editor_name, "
         "  (SELECT COUNT(*) FROM conv_utterances u WHERE u.session_id=s.id) AS n_utt, "
         "  (SELECT COUNT(*) FROM conv_utterances u WHERE u.session_id=s.id AND u.accepted=1) AS n_ok, "
         f"  (s.status IN ('in_edit','rejected','drafted') AND COALESCE(s.drafted_at, s.uploaded_at) <= datetime('now','-{CONV_EDIT_DEADLINE_DAYS} days')) AS stale "
         "FROM conv_sessions s LEFT JOIN conv_topics t ON t.id=s.topic_id "
         "WHERE s.status IN ('in_review','in_edit','drafted','approved','rejected') "
         "ORDER BY CASE s.status WHEN 'in_review' THEN 0 WHEN 'in_edit' THEN 1 WHEN 'drafted' THEN 2 WHEN 'rejected' THEN 3 ELSE 4 END, s.id DESC LIMIT 200")
    rows = db.execute(q).fetchall(); db.close()
    return render_template("conv_review_list.html", rows=rows)

@app.route("/conv/review/<int:sess_id>")
@login_required
@reviewer_required
def conv_review_page(sess_id):
    db = get_db()
    s = _conv_session_row(db, sess_id)
    if not s:
        db.close(); abort(404)
    tracks = {t["channel"]: t for t in db.execute("SELECT * FROM conv_tracks WHERE session_id=?", (sess_id,)).fetchall()}
    utts = db.execute("SELECT * FROM conv_utterances WHERE session_id=? ORDER BY start_sec, channel", (sess_id,)).fetchall()
    ed = db.execute("SELECT username FROM users WHERE id=?", (s["editor_id"],)).fetchone() if s["editor_id"] else None
    db.close()
    urls = {ch: (_conv_presign(t["b2_key"]) if t and t["b2_key"] else None) for ch, t in tracks.items()}
    try:
        off_b = (json.loads(s["alignment_offset_json"] or "{}").get("B") or 0) / 1000.0
    except Exception:
        off_b = 0.0
    return render_template("conv_review.html", s=s, utts=utts, urls=urls, tracks=tracks, off_b=off_b,
                           editor_name=(ed["username"] if ed else "Admin (unassigned circle)"),
                           can_decide=(s["status"] == "in_review"),
                           n_pii=sum(1 for u in utts if u["pii"]), n_edit=sum(1 for u in utts if u["source"] == "edited"))

@app.route("/conv/review/<int:sess_id>/utt/<int:utt_id>", methods=["POST"])
@login_required
@reviewer_required
def conv_review_utt(sess_id, utt_id):
    data = request.get_json(silent=True) or {}
    db = get_db()
    s = db.execute("SELECT status FROM conv_sessions WHERE id=?", (sess_id,)).fetchone()
    if not s or s["status"] != "in_review":
        db.close(); return jsonify({"ok": False, "error": "not_in_review"}), 409
    u = db.execute("SELECT * FROM conv_utterances WHERE id=? AND session_id=?", (utt_id, sess_id)).fetchone()
    if not u:
        db.close(); return jsonify({"ok": False}), 404
    _apply_utt_update(db, u, data, f"admin:{session.get('username')}")
    db.commit(); db.close()
    return jsonify({"ok": True})

@app.route("/conv/review/<int:sess_id>/approve", methods=["POST"])
@login_required
@reviewer_required
def conv_review_approve(sess_id):
    """Final approval. Freezes the pay rates (both speakers' hourly rates, the circle
    editor's hourly rate) on the session, exactly as clip approval freezes them in
    read speech. Pay itself is settled later with the dashboard 'Mark paid' buttons."""
    db = get_db()
    s = _conv_session_row(db, sess_id)
    if not s:
        db.close(); abort(404)
    def _rate(sid):
        r = db.execute("SELECT hourly_rate FROM users WHERE speaker_id=?", (sid,)).fetchone()
        return int(r["hourly_rate"]) if r and r["hourly_rate"] else RATE_PER_HOUR
    ed_rate = None
    if s["editor_id"]:
        r = db.execute("SELECT hourly_rate FROM users WHERE id=? AND role='editor'", (s["editor_id"],)).fetchone()
        ed_rate = int(r["hourly_rate"]) if r and r["hourly_rate"] else 70000
    try:
        _conv_transition(db, sess_id, "approved", approved_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                         reviewer_id=session.get("user_id"), reject_reason=None)
        db.execute("UPDATE conv_sessions SET rate_a_at_approval=?, rate_b_at_approval=?, editor_rate_at_approval=? WHERE id=?",
                   (_rate(s["speaker_a_id"]), _rate(s["speaker_b_id"]), ed_rate, sess_id))
        db.commit(); flash("Session approved — speaker and editor pay are now pending settlement.", "success")
    except ValueError as e:
        flash(f"Cannot approve: {e}", "error")
    db.close()
    return redirect(url_for("conv_review_list"))

@app.route("/conv/review/<int:sess_id>/reject", methods=["POST"])
@login_required
@reviewer_required
def conv_review_reject(sess_id):
    reason = (request.form.get("reason") or "").strip()
    if len(reason) < 5:
        flash("A reason is required (shown to the editor).", "error")
        return redirect(url_for("conv_review_page", sess_id=sess_id))
    db = get_db()
    try:
        _conv_transition(db, sess_id, "rejected", reject_reason=reason, reviewer_id=session.get("user_id"))
        # back to the editor; accepted flags are kept so she only revisits what she chooses
        db.execute("UPDATE conv_tracks SET edit_status='pending', edit_done_at=NULL WHERE session_id=?", (sess_id,))
        db.commit(); flash("Session rejected — it is back in the editor's queue with your reason.", "info")
    except ValueError as e:
        flash(f"Cannot reject: {e}", "error")
    db.close()
    return redirect(url_for("conv_review_list"))

# ── Pay: unpaid approved conversations (mirrors the clips bookkeeping) ────────
def _conv_pay_rows_speaker(db, speaker_id):
    """Unpaid approved sessions for a speaker: rows with duration_seconds + frozen rate."""
    return [{"duration_seconds": r["duration_sec"] or 0, "rate": r["rate"], "id": r["id"]} for r in db.execute(
        "SELECT id, duration_sec, CASE WHEN speaker_a_id=? THEN rate_a_at_approval ELSE rate_b_at_approval END AS rate "
        "FROM conv_sessions WHERE status='approved' AND ("
        "  (speaker_a_id=? AND COALESCE(paid_a,0)=0) OR (speaker_b_id=? AND COALESCE(paid_b,0)=0))",
        (speaker_id, speaker_id, speaker_id)).fetchall()]

def _conv_pay_rows_editor(db, editor_id):
    return [{"duration_seconds": r["duration_sec"] or 0, "rate": r["editor_rate_at_approval"], "id": r["id"]} for r in db.execute(
        "SELECT id, duration_sec, editor_rate_at_approval FROM conv_sessions "
        "WHERE status='approved' AND editor_id=? AND COALESCE(editor_paid,0)=0", (editor_id,)).fetchall()]

def _conv_pay_amount(rows, fallback_rate):
    return calc_comp_grouped(rows, "rate", fallback_rate)

def _conv_settle_speaker(db, speaker_id, method):
    """Record one payment transaction for the speaker's unpaid conversations and flip the
    paid flags. Returns the amount (0 = nothing to settle)."""
    user = db.execute("SELECT id, hourly_rate FROM users WHERE speaker_id=? AND role='contributor'", (speaker_id,)).fetchone()
    rows = _conv_pay_rows_speaker(db, speaker_id)
    if not user or not rows:
        return 0
    amount = _conv_pay_amount(rows, user["hourly_rate"] or RATE_PER_HOUR)
    if amount > 0:
        db.execute("INSERT INTO payment_transactions (user_id, user_role, amount, clip_count, duration_seconds, paid_by_username, method) "
                   "VALUES (?, 'contributor', ?, ?, ?, ?, ?)",
                   (user["id"], amount, len(rows), sum(r["duration_seconds"] for r in rows), session.get("username"), method))
    db.execute("UPDATE conv_sessions SET paid_a=1 WHERE status='approved' AND speaker_a_id=? AND COALESCE(paid_a,0)=0", (speaker_id,))
    db.execute("UPDATE conv_sessions SET paid_b=1 WHERE status='approved' AND speaker_b_id=? AND COALESCE(paid_b,0)=0", (speaker_id,))
    return amount

def _conv_settle_editor(db, editor_id, method):
    ed = db.execute("SELECT id, hourly_rate, COALESCE(editor_penalty_pct,0) AS pct FROM users WHERE id=? AND role='editor'", (editor_id,)).fetchone()
    rows = _conv_pay_rows_editor(db, editor_id)
    if not ed or not rows:
        return 0
    amount = apply_editor_penalty(_conv_pay_amount(rows, ed["hourly_rate"] or 70000), ed["pct"])
    if amount > 0:
        db.execute("INSERT INTO payment_transactions (user_id, user_role, amount, clip_count, duration_seconds, paid_by_username, method) "
                   "VALUES (?, 'editor', ?, ?, ?, ?, ?)",
                   (editor_id, amount, len(rows), sum(r["duration_seconds"] for r in rows), session.get("username"), method))
    db.execute("UPDATE conv_sessions SET editor_paid=1 WHERE status='approved' AND editor_id=? AND COALESCE(editor_paid,0)=0", (editor_id,))
    return amount

# ── In-page calling: ICE servers + signaling mailbox ─────────────────────────
def _conv_ice_servers():
    """iceServers list for RTCPeerConnection. Cloudflare TURN creds are generated
    server-side with a 12 h TTL and cached; static TURN passes through; else STUN only."""
    import time as _time
    now = _time.time()
    if _ICE_CACHE["servers"] and _ICE_CACHE["exp"] > now:
        return _ICE_CACHE["servers"]
    servers = [{"urls": STUN_URLS}]
    ttl = 60 * 60 * 12
    if CF_TURN_KEY_ID and CF_TURN_API_TOKEN:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"https://rtc.live.cloudflare.com/v1/turn/keys/{CF_TURN_KEY_ID}/credentials/generate",
                data=json.dumps({"ttl": ttl}).encode(),
                headers={"Authorization": f"Bearer {CF_TURN_API_TOKEN}", "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode())
            ice = data.get("iceServers") or data
            if isinstance(ice, dict) and ice.get("urls"):
                servers.append({"urls": ice["urls"], "username": ice.get("username"), "credential": ice.get("credential")})
            elif isinstance(ice, list):
                servers.extend(ice)
        except Exception as e:
            print(f"[CONV ICE] Cloudflare TURN credential fetch failed: {e}")
    elif TURN_URLS:
        servers.append({"urls": TURN_URLS, "username": TURN_USERNAME, "credential": TURN_CREDENTIAL})
    _ICE_CACHE["servers"] = servers
    _ICE_CACHE["exp"] = now + (ttl - 600 if (CF_TURN_KEY_ID and CF_TURN_API_TOKEN) else 3600)
    return servers

@app.route("/conv/session/<int:sess_id>/ice")
@login_required
@conv_gate_required
def conv_ice(sess_id):
    me = session.get("speaker_id")
    db = get_db(); s, ch = _conv_participant(db, sess_id, me); db.close()
    if not s:
        return jsonify({"ok": False}), 403
    servers = _conv_ice_servers()
    return jsonify({"ok": True, "iceServers": servers,
                    "hasTurn": any("turn:" in u or "turns:" in u for sv in servers for u in (sv.get("urls") if isinstance(sv.get("urls"), list) else [sv.get("urls")]) if u)})

@app.route("/conv/session/<int:sess_id>/signal", methods=["POST"])
@login_required
@conv_gate_required
def conv_signal_post(sess_id):
    me = session.get("speaker_id")
    data = request.get_json(silent=True) or {}
    kind = (data.get("kind") or "").strip()
    if kind not in ("hello", "ready", "offer", "answer", "ice", "restart", "start", "stop", "bye"):
        return jsonify({"ok": False, "error": "bad_kind"}), 400
    db = get_db(); s, ch = _conv_participant(db, sess_id, me)
    if not s:
        db.close(); return jsonify({"ok": False}), 403
    payload = json.dumps(data.get("payload") if data.get("payload") is not None else {})
    if len(payload) > 200000:
        db.close(); return jsonify({"ok": False, "error": "too_big"}), 413
    db.execute("INSERT INTO conv_signals (session_id, from_channel, kind, payload) VALUES (?,?,?,?)", (sess_id, ch, kind, payload))
    db.commit(); db.close()
    return jsonify({"ok": True})

@app.route("/conv/session/<int:sess_id>/signal")
@login_required
@conv_gate_required
def conv_signal_get(sess_id):
    """Messages from the partner after ?after=<id>, plus partner presence (seconds since
    their last 'hello'). Polled ~1 s during connection setup, slower afterwards."""
    me = session.get("speaker_id")
    try: after = int(request.args.get("after", "0") or 0)
    except ValueError: after = 0
    db = get_db(); s, ch = _conv_participant(db, sess_id, me)
    if not s:
        db.close(); return jsonify({"ok": False}), 403
    other = "B" if ch == "A" else "A"
    rows = db.execute("SELECT id, kind, payload, created_at FROM conv_signals WHERE session_id=? AND from_channel=? AND id>? "
                      "AND kind<>'hello' ORDER BY id ASC LIMIT 100", (sess_id, other, after)).fetchall()
    last_hello = db.execute("SELECT MAX(id) AS mid, MAX(created_at) AS at FROM conv_signals WHERE session_id=? AND from_channel=? AND kind='hello'",
                            (sess_id, other)).fetchone()
    seen = None
    if last_hello and last_hello["at"]:
        seen = db.execute("SELECT (julianday('now') - julianday(?)) * 86400.0", (last_hello["at"],)).fetchone()[0]
    # the highest id of ANY partner row, so the client can advance its cursor past hellos too
    hi = db.execute("SELECT COALESCE(MAX(id),0) FROM conv_signals WHERE session_id=? AND from_channel=?", (sess_id, other)).fetchone()[0]
    db.close()
    return jsonify({"ok": True, "msgs": [{"id": r["id"], "kind": r["kind"], "payload": json.loads(r["payload"] or "{}")} for r in rows],
                    "cursor": max(hi, after), "partner_seen": (round(seen, 1) if seen is not None else None), "status": s["status"]})

# ── Abort helper + housekeeping sweep ─────────────────────────────────────────
def _conv_on_abort(db, sess_id, reason=""):
    """Everything an abort must do, from ANY path (speaker, admin, sweep): cancel open
    multipart uploads, give the topic its rotation slot back, stamp aborted_at.
    Network calls happen BEFORE any write so the database write lock is never held
    while waiting on B2."""
    tracks = db.execute("SELECT * FROM conv_tracks WHERE session_id=? AND upload_id IS NOT NULL", (sess_id,)).fetchall()
    if tracks:
        try:
            b2 = get_b2()
            for t in tracks:
                try:
                    b2.abort_multipart_upload(Bucket=B2_BUCKET_NAME, Key=t["b2_key"], UploadId=t["upload_id"])
                except Exception:
                    pass
        except Exception:
            pass
    db.execute("UPDATE conv_tracks SET upload_id=NULL, check_status=CASE WHEN check_status IN ('uploading','pending') THEN 'aborted' ELSE check_status END WHERE session_id=?", (sess_id,))
    s = db.execute("SELECT topic_id, status FROM conv_sessions WHERE id=?", (sess_id,)).fetchone()
    if s and s["topic_id"]:
        db.execute("UPDATE conv_topics SET use_count=MAX(0, use_count-1) WHERE id=?", (s["topic_id"],))
    _conv_transition(db, sess_id, "aborted", aborted_at=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    db.commit()
    if reason:
        print(f"[CONV SWEEP] aborted session {sess_id}: {reason}")

def _conv_purge_session_audio(db, sess_id):
    """Delete an aborted session's audio from storage (version-aware where supported).
    Rows stay; tracks are marked purged. Each track: network first, then a short write."""
    b2 = get_b2()
    for t in db.execute("SELECT id, b2_key FROM conv_tracks WHERE session_id=? AND b2_key IS NOT NULL AND check_status<>'purged'", (sess_id,)).fetchall():
        try:
            try:
                vs = b2.list_object_versions(Bucket=B2_BUCKET_NAME, Prefix=t["b2_key"])
                for v in (vs.get("Versions") or []) + (vs.get("DeleteMarkers") or []):
                    if v.get("Key") == t["b2_key"] and v.get("VersionId"):
                        b2.delete_object(Bucket=B2_BUCKET_NAME, Key=t["b2_key"], VersionId=v["VersionId"])
            except Exception:
                b2.delete_object(Bucket=B2_BUCKET_NAME, Key=t["b2_key"])
            db.execute("UPDATE conv_tracks SET check_status='purged', b2_key=NULL WHERE id=?", (t["id"],))
            db.commit()
        except Exception as e:
            print(f"[CONV SWEEP] purge failed for track {t['id']}: {e}")

_SWEEP_LOCK = threading.Lock()

def _conv_housekeeping():
    """Lifecycle sweep, throttled to once per hour and run on a BACKGROUND thread so no
    page request ever waits on it. Every step commits before the next network call."""
    db = get_db()
    try:
        recent = db.execute("SELECT 1 FROM _one_time_flags WHERE key='conv_sweep_at' AND applied_at >= datetime('now','-1 hour')").fetchone()
        if recent:
            return
        db.execute("REPLACE INTO _one_time_flags (key, applied_at) VALUES ('conv_sweep_at', datetime('now'))"); db.commit()
    finally:
        db.close()
    threading.Thread(target=_conv_housekeeping_run, daemon=True).start()

def _conv_housekeeping_run():
    if not _SWEEP_LOCK.acquire(blocking=False):
        return
    stuck, recheck = [], []
    db = get_db()
    try:
        for r in db.execute("SELECT id FROM conv_sessions WHERE status='created' AND created_at <= datetime('now','-14 days')").fetchall():
            try: _conv_on_abort(db, r["id"], "created 14+ days, never started")
            except ValueError: pass
        for r in db.execute(
            "SELECT s.id FROM conv_sessions s WHERE s.status='recording' AND s.recording_started_at <= datetime('now','-2 hours') "
            "AND NOT EXISTS (SELECT 1 FROM conv_tracks t WHERE t.session_id=s.id AND t.uploaded_at >= datetime('now','-2 hours'))").fetchall():
            try: _conv_on_abort(db, r["id"], "recording idle 2 h")
            except ValueError: pass
        for r in db.execute("SELECT id FROM conv_sessions WHERE status='incomplete' AND incomplete_at <= datetime('now','-24 hours')").fetchall():
            try: _conv_on_abort(db, r["id"], "incomplete 24 h")
            except ValueError: pass
        stuck = [r["id"] for r in db.execute("SELECT id FROM conv_sessions WHERE status='drafting' AND drafting_started_at <= datetime('now','-2 hours')").fetchall()]
        for sid in stuck:
            try: _conv_transition(db, sid, "uploaded", draft_error="drafting interrupted (app restart?) — re-run automatically")
            except ValueError: pass
        # uploaded but never drafted (the drafting thread died with a redeploy before it could
        # even start, or auto-draft was unavailable at the time): try again
        stuck += [r["id"] for r in db.execute(
            "SELECT id FROM conv_sessions WHERE status='uploaded' AND draft_error IS NULL "
            "AND uploaded_at <= datetime('now','-10 minutes')").fetchall()]
        db.commit()
        recheck = [r["id"] for r in db.execute("SELECT id FROM conv_tracks WHERE check_status='pending' AND uploaded_at <= datetime('now','-30 minutes')").fetchall()]
        db.execute("UPDATE conv_invites SET status='expired' WHERE status='pending' AND expires_at <= datetime('now')")
        db.execute("DELETE FROM conv_signals WHERE created_at < datetime('now','-1 day')")
        db.commit()
        purge = [r["id"] for r in db.execute(
            "SELECT DISTINCT s.id FROM conv_sessions s JOIN conv_tracks t ON t.session_id=s.id "
            "WHERE s.status='aborted' AND s.aborted_at <= datetime('now','-24 hours') AND t.b2_key IS NOT NULL AND t.check_status<>'purged'").fetchall()]
        for sid in purge:
            _conv_purge_session_audio(db, sid)
    except Exception:
        import traceback; print("[CONV SWEEP] failed:\n" + traceback.format_exc())
    finally:
        db.close()
        _SWEEP_LOCK.release()
    for sid in stuck:
        _conv_maybe_autodraft(sid)
    for tid in recheck:
        threading.Thread(target=_conv_check_track, args=(tid,), daemon=True).start()

if __name__=="__main__":
    init_db(); app.run(debug=True,port=5000)