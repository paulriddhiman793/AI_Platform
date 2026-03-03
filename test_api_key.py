"""
test_groq_key.py — Quick test to verify your Groq API key works.

Usage:
    pip install groq python-dotenv
    python test_groq_key.py
"""

import os
import sys

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded")
except ImportError:
    print("⚠  python-dotenv not installed — reading key from system environment only")

# ── Check key exists ──────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("\n❌ GROQ_API_KEY not found.")
    print("   Add it to your .env file:")
    print("   GROQ_API_KEY=gsk_...")
    sys.exit(1)

print(f"✅ API key found: gsk_...{api_key[-6:]}")

# ── Try importing groq ────────────────────────────────────────────────────────
try:
    from groq import Groq
    print("✅ groq package imported")
except ImportError:
    print("\n❌ groq package not installed.")
    print("   Run: pip install groq")
    sys.exit(1)

# ── Make a real API call ──────────────────────────────────────────────────────
print("\nSending test message to Groq (llama-3.3-70b-versatile)...")
print("-" * 50)

try:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # fast + free tier available
        max_tokens=50,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly this sentence: API key is working correctly."
            }
        ]
    )

    reply = response.choices[0].message.content
    print(f"Groq replied: {reply}")
    print("-" * 50)
    print("\n✅ SUCCESS — Your Groq API key is valid and working!")
    print(f"   Model:         {response.model}")
    print(f"   Input tokens:  {response.usage.prompt_tokens}")
    print(f"   Output tokens: {response.usage.completion_tokens}")
    print(f"   Total tokens:  {response.usage.total_tokens}")

    # Groq returns inference speed — show it if available
    if hasattr(response, "usage") and hasattr(response.usage, "completion_time"):
        speed = round(response.usage.completion_tokens / response.usage.completion_time)
        print(f"   Speed:         ~{speed} tokens/sec  ⚡")

except Exception as e:
    err = type(e).__name__
    msg = str(e).lower()

    if "401" in msg or "invalid api key" in msg or "authentication" in msg:
        print("\n❌ AUTHENTICATION FAILED — Your Groq API key is invalid or expired.")
        print("   Get a fresh key at console.groq.com → API Keys")

    elif "429" in msg or "rate limit" in msg:
        print("\n⚠  RATE LIMIT — Key is valid but you've hit Groq's free tier limit.")
        print("   Wait a minute and try again, or check console.groq.com → Usage")

    elif "connection" in msg or "network" in msg:
        print("\n❌ CONNECTION ERROR — Could not reach Groq servers.")
        print("   Check your internet connection.")

    elif "model" in msg or "404" in msg:
        print(f"\n⚠  MODEL NOT FOUND — trying fallback model mixtral-8x7b-32768")
        try:
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                max_tokens=50,
                messages=[{"role": "user", "content": "Say: API key is working correctly."}]
            )
            print(f"Groq replied: {response.choices[0].message.content}")
            print("\n✅ SUCCESS with fallback model mixtral-8x7b-32768")
        except Exception as e2:
            print(f"\n❌ Fallback also failed: {e2}")
    else:
        print(f"\n❌ Unexpected error: {err}: {e}")