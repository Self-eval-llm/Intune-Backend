#!/usr/bin/env python3
"""
Quick Realtime Test - Verify Supabase Realtime is working
"""
import os
import asyncio
import logging
from dotenv import load_dotenv
from supabase import acreate_client

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_realtime():
    """Test if Supabase Realtime detects changes on intune_db"""

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Missing SUPABASE_URL or SUPABASE_KEY")
        return

    logger.info("🔍 Testing Realtime connection to intune_db...")

    try:
        # Create async Supabase client
        supabase = await acreate_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client created")

        # Define change handler
        def handle_change(payload):
            logger.info(f"🎯 REALTIME CHANGE DETECTED!")
            logger.info(f"   Table: {payload.get('table')}")
            logger.info(f"   Event: {payload.get('eventType')}")
            logger.info(f"   Record: {payload.get('new', {}).get('id', 'unknown')}")
            logger.info(f"   Status: {payload.get('new', {}).get('status_eval_first')}")

        # Subscribe to intune_db changes
        channel = supabase.channel('test-channel')
        channel.on_postgres_changes(handle_change, event='*', schema='public', table='intune_db')
        await channel.subscribe()
        logger.info("📡 Subscribed to intune_db changes")

        # Keep running for 30 seconds
        logger.info("⏳ Listening for changes (30 seconds)...")
        logger.info("💡 Now run: python3 manual_test_data_generator.py and trigger some data!")

        await asyncio.sleep(30)

        # Cleanup
        await channel.unsubscribe()
        logger.info("👋 Test complete")

    except Exception as e:
        logger.error(f"❌ Realtime test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_realtime())