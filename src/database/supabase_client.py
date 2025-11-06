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


def get_supabase_client() -> Client:
    """
    Create and return Supabase client.
    
    Returns:
        Client: Initialized Supabase client
        
    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY are not set
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


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
