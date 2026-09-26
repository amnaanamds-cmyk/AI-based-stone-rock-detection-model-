"""Authentication, role-based access control, CSRF protection, API tokens and audit logging.

Roles (each includes the rights of the ones below it):

* ``admin``   - manage users, delete anything, view the audit log
* ``analyst`` - upload data, create regions, draw training areas, run processing jobs
* ``viewer``  - view maps, statistics and reports, download products
"""
from __future__ import annotations

import functools
import hashlib
import hmac
import secrets
import time
from collections import defaultdict
from typing import Optional

from flask import abort, current_app, flash, g, jsonify, make_response, redirect, request, session, url_for
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash

from .db import Database, now

ROLES = ("viewer", "analyst", "admin")
ROLE_LEVEL = {r: i for i, r in enumerate(ROLES)}
MAX_FAILURES, LOCK_SECONDS = 5, 300


def _failures() -> dict[str, list[float]]:
    return current_app.extensions.setdefault("rockmap_login_failures", defaultdict(list))


def create_user(db: Database, username: str, password: str, role: str = "analyst",
                must_change: bool = False) -> int:
    username = username.strip()
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if not username or len(username) > 64:
        raise ValueError("invalid username")
    if len(password) < 8:
        raise ValueError("password must have at least 8 characters")
    if db.one("users", "username = ?", (username,)):
        raise ValueError(f"user '{username}' already exists")
    return db.insert("users", username=username, password_hash=generate_password_hash(password), role=role,
                     must_change=int(must_change))


def set_password(db: Database, user_id: int, password: str, must_change: bool = False) -> None:
    if len(password) < 8:
        raise ValueError("password must have at least 8 characters")
    if password.lower() in WEAK_PASSWORDS:
        raise ValueError("this password is too common - choose another one")
    db.update("users", user_id, password_hash=generate_password_hash(password), must_change=int(must_change))


WEAK_PASSWORDS = {"rockmap-demo", "password", "password1", "12345678", "123456789", "admin123", "qwerty123",
                  "rockmap123", "11111111", "iloveyou"}


def require_password_change() -> Optional["Response"]:
    """Users with a temporary password may only change it (or sign out) until they do."""
    user = g.get("user")
    if not user or g.get("api") or not user.get("must_change"):
        return None
    if request.endpoint in ("profile", "logout", "static", "login", "healthz") or request.path.startswith("/share/"):
        return None
    if request.path.startswith("/api/") or request.path.startswith("/tiles/"):
        return make_response(jsonify(error="password change required"), 403)
    flash("Please choose a new password before continuing.", "error")
    return redirect(url_for("profile"))


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_api_token(db: Database, user_id: int) -> str:
    token = "rmk_" + secrets.token_urlsafe(32)
    db.update("users", user_id, api_token_hash=_hash_token(token))
    return token


def audit(action: str, detail: str = "") -> None:
    db: Database = current_app.extensions["rockmap_db"]
    user = getattr(g, "user", None)
    db.insert("audit", username=user["username"] if user else None, action=action, detail=detail[:500],
              ip=request.remote_addr if request else None)


def _locked(key: str) -> bool:
    recent = [t for t in _failures()[key] if time.time() - t < LOCK_SECONDS]
    _failures()[key] = recent
    return len(recent) >= MAX_FAILURES


def authenticate(db: Database, username: str, password: str) -> tuple[Optional[dict], str]:
    key = f"{username.lower()}|{request.remote_addr}"
    if _locked(key):
        return None, "Too many failed attempts - try again in a few minutes."
    user = db.one("users", "username = ?", (username.strip(),))
    if not user or not user["active"] or not check_password_hash(user["password_hash"], password):
        _failures()[key].append(time.time())
        return None, "Invalid username or password."
    _failures().pop(key, None)
    db.update("users", user["id"], last_login=now())
    return user, ""


def load_user() -> None:
    """Attach the current user (session cookie or ``Authorization: Bearer`` API token) to ``g``."""
    db: Database = current_app.extensions["rockmap_db"]
    g.user, g.api = None, False
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        token = header[7:].strip()
        user = db.one("users", "api_token_hash = ? AND active = 1", (_hash_token(token),)) if token else None
        if user:
            g.user, g.api = user, True
        return
    uid = session.get("uid")
    if uid:
        user = db.get("users", uid)
        if user and user["active"] and session.get("pw") == user["password_hash"][-16:]:
            g.user = user
        else:
            session.clear()


def login_session(user: dict) -> None:
    session.clear()
    session.permanent = True
    session["uid"] = user["id"]
    session["pw"] = user["password_hash"][-16:]   # invalidates sessions after a password change
    session["csrf"] = secrets.token_urlsafe(24)


def has_role(role: str) -> bool:
    user = getattr(g, "user", None)
    return bool(user) and ROLE_LEVEL[user["role"]] >= ROLE_LEVEL[role]


def requires(role: str = "viewer"):
    """Route decorator: login required, minimum role."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            if not g.get("user"):
                if request.path.startswith("/api/") or request.path.startswith("/tiles/") or g.get("api"):
                    return jsonify(error="authentication required"), 401
                return redirect(url_for("login", next=request.full_path))
            if not has_role(role):
                if request.path.startswith("/api/"):
                    return jsonify(error=f"'{role}' role required"), 403
                abort(403)
            return fn(*a, **kw)
        return wrapper
    return deco


# ---------------------------------------------------------------------------- CSRF
def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(24)
    return session["csrf"]


def csrf_field() -> Markup:
    return Markup(f'<input type="hidden" name="csrf_token" value="{csrf_token()}">')


def check_csrf() -> None:
    if not current_app.config.get("CSRF_ENABLED", True) or request.method in ("GET", "HEAD", "OPTIONS"):
        return
    if g.get("api"):   # token-authenticated API calls do not use cookies
        return
    sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
    if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
        if request.path.startswith("/api/"):
            abort(make_response(jsonify(error="CSRF token missing or invalid"), 400))
        flash("Your session expired - please try again.", "error")
        abort(400)
