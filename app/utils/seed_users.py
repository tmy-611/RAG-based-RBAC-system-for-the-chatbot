"""Seed the Supabase users table with initial RBAC users (one-off script)."""
import hashlib
import os

from app.utils.auth import get_supabase_client

USERS = [
    ("Tony", "password123", "engineering"),
    ("Bruce", "securepass", "marketing"),
    ("Sam", "financepass", "finance"),
    ("Peter", "pete123", "engineering"),
    ("Sid", "sidpass123", "marketing"),
    ("Natasha", "hrpass123", "hr")
]

ITERATIONS = 100_000


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2 + random salt (stdlib only, no extra dep)."""
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), ITERATIONS
        ).hex()
    return f"pbkdf2${ITERATIONS}${salt}${digest}"


def main() -> None:
    """Upsert every user so re-runs never create duplicates."""
    client = get_supabase_client()
    for username, password, role in USERS:
        client.table("users").upsert(
            {
                "username": username,
                "password_hash": hash_password(password),
                "role": role,
            },
            on_conflict="username",
        ).execute()
    print(f"Seeded {len(USERS)} users")

if __name__ == "__main__":
    main()