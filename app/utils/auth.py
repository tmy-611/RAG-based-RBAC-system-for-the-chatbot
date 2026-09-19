""" Supabase Auth helper: client singleton + user lookup for RBAC"""

import hashlib
import hmac
import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()  # Load environment variables from .env file

USER_TABLE = "users"


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Build the Supabase client once from environment variables."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)

def get_user(username: str) -> dict | None:
    """Fetch one user row by username. Returns None when not found."""
    response = (
        get_supabase_client()
        .table(USER_TABLE)
        .select("*")
        .eq("username", username)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def verify_password(password:str, stored_hash: str) -> bool:
    """Check a password against a stored PBKDF2 hash. 
    
    Returns False (never raises) on wrong password or bad hash format."""

    try:
        algo, iters, salt, digest = stored_hash.split("$")
        if algo != "pbkdf2":
            return False
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters)).hex()
        return hmac.compare_digest(check, digest)
    except (ValueError, AttributeError):
        return False