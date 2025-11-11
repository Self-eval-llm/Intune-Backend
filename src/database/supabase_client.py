"""
Supabase database client and utility functions.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def get_supabase_client() -> Client:
    """
    Create and return Supabase client.
    
    Returns:
        Client: Initialized Supabase client
        
    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY are not set
    """
    # Prefer server-side service role key if available (allows bypassing RLS for trusted backend)
    key_to_use = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY

    if not SUPABASE_URL or not key_to_use:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set in .env file")

    # If using the anon/public key and your table has Row-Level Security (RLS) enabled,
    # attempts to insert/update rows may fail with a RLS policy error. For server-side
    # inserts from a trusted backend, provide the service role key in SUPABASE_SERVICE_ROLE_KEY.
    return create_client(SUPABASE_URL, key_to_use)


def int8_to_decimal(value):
    """
    Convert INT8 metric value to decimal.
    
    Supabase stores metrics as INT8 (scaled by 10000) to avoid precision issues.
    This function converts them back to float.
    
    Args:
        value: INT8 value or None
        
    Returns:
        float: Decimal value between 0 and 1, or 0.0 if value is None
    """
    if value is None:
        return 0.0
    return round(value / 10000, 4)


def decimal_to_int8(value: float) -> int:
    """
    Convert decimal metric value to INT8 for storage.
    
    Args:
        value: Float value between 0 and 1
        
    Returns:
        int: INT8 value scaled by 10000
    """
    if value is None:
        return 0
    return int(round(value * 10000))
