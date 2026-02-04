import os
import time
import secrets
from datetime import datetime
from fastapi import FastAPI, Form, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
import requests

from passlib.context import CryptContext
import jwt

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

app = FastAPI()

# =====================================================
# Helpers (Response)
# =====================================================

def ok(data=None):
    return {"ok": True, "data": data or {}}

def err(code, message=None):
    r = {"ok": False, "error_code": code}
    if message:
        r["message"] = message
    return r

# =====================================================
# ENV
# =====================================================

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALG = "HS256"

APP_BASE_URL = os.getenv("APP_BASE_URL", "")

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY") or os.getenv("SendGridAPIKey")
SENDGRID_FROM = os.getenv("SENDGRID_FROM") or os.getenv("SendGripFrom")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./auth.db"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =====================================================
# DB Init
# =====================================================

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email_verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS email_verification_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at INTEGER NOT NULL
        );
        """))

@app.on_event("startup")
def startup():
    init_db()

# =====================================================
# Auth Utils
# =====================================================

def hash_pw(pw):
    return pwd_context.hash(pw)

def verify_pw(pw, pw_hash):
    return pwd_context.verify(pw, pw_hash)

def make_jwt(user_id, email, email_verified, remember):
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET missing")
    ttl = 60 * 60 * 24 * (30 if remember else 1)
    payload = {
        "sub": user_id,
        "email": email,
        "email_verified": bool(email_verified),
        "exp": int(time.time()) + ttl
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        return None

def send_verify_mail(email, link):
    if not SENDGRID_API_KEY or not SENDGRID_FROM:
        return False
    payload = {
        "personalizations": [{"to": [{"email": email}]}],
        "from": {"email": SENDGRID_FROM},
        "subject": "GrowDoctor – E-Mail bestätigen",
        "content": [{"type": "text/plain", "value": f"Bitte bestätigen:\n{link}"}]
    }
    r = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=10
    )
    return 200 <= r.status_code < 300

# =====================================================
# Auth Dependencies
# =====================================================

def get_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    return decode_jwt(token)

def require_user(user=Depends(get_user)):
    if not user:
        return err("AUTH_REQUIRED")
    return user

def require_verified(user=Depends(require_user)):
    if isinstance(user, dict) and not user.get("email_verified"):
        return err("EMAIL_NOT_VERIFIED")
    return user

# =====================================================
# Auth Routes
# =====================================================

@app.post("/auth/register")
def register(email: str = Form(...), password: str = Form(...)):
    email = email.lower().strip()
    if "@" not in email or len(password) < 8:
        return err("INVALID_INPUT")

    uid = secrets.token_hex(16)
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO users (id, email, password_hash, email_verified, created_at)
                VALUES (:id, :email, :pw, 0, :ts)
            """), {"id": uid, "email": email, "pw": hash_pw(password), "ts": datetime.utcnow().isoformat()})
    except IntegrityError:
        return err("EMAIL_EXISTS")

    token = secrets.token_urlsafe(32)
    expires = int(time.time()) + 86400

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO email_verification_tokens (token, user_id, expires_at)
            VALUES (:t, :u, :e)
        """), {"t": token, "u": uid, "e": expires})

    if not APP_BASE_URL:
        return err("APP_BASE_URL_MISSING")

    link = f"{APP_BASE_URL}/auth/verify-email?token={token}"
    if not send_verify_mail(email, link):
        return err("MAIL_FAILED")

    return ok({"message": "Bitte E-Mail bestätigen"})

@app.get("/auth/verify-email")
def verify_email(token: str):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT user_id, expires_at FROM email_verification_tokens WHERE token = :t
        """), {"t": token}).mappings().first()

    if not row:
        return err("INVALID_TOKEN")
    if row["expires_at"] < int(time.time()):
        return err("TOKEN_EXPIRED")

    with engine.begin() as conn:
        conn.execute(text("UPDATE users SET email_verified = 1 WHERE id = :u"), {"u": row["user_id"]})
        conn.execute(text("DELETE FROM email_verification_tokens WHERE token = :t"), {"t": token})

    return ok({"message": "E-Mail bestätigt"})

@app.post("/auth/login")
def login(email: str = Form(...), password: str = Form(...), remember_me: bool = Form(False)):
    email = email.lower().strip()
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id, email, password_hash, email_verified FROM users WHERE email = :e
        """), {"e": email}).mappings().first()

    if not row or not verify_pw(password, row["password_hash"]):
        return err("INVALID_CREDENTIALS")
    if not row["email_verified"]:
        return err("EMAIL_NOT_VERIFIED")

    token = make_jwt(row["id"], row["email"], row["email_verified"], remember_me)
    return ok({"access_token": token, "token_type": "bearer"})

@app.get("/auth/me")
def me(user=Depends(require_user)):
    if isinstance(user, dict) and not user.get("ok", True):
        return user
    return ok({"email": user["email"], "email_verified": user["email_verified"]})

# =====================================================
# Health
# =====================================================

@app.get("/health")
def health():
    return {"ok": True}
