"""Test configuration and fixtures."""
import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
# Don't set MONGO_DB here - let tests control it