import os
import sys


def load_env_file(env_path: str = ".env"):
    """Load environment variables from a .env file."""
    if not os.path.exists(env_path):
        return False
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)
    return True


def check_gemini_key(api_key: str) -> bool:
    """Check if the provided Gemini API key is valid and can generate responses."""
    try:
        from google import genai
    except ImportError:
        print("Installing google-genai...")
        os.system(f"{sys.executable} -m pip install google-genai -q")
        from google import genai

    # Step 1: Validate key by listing models (no quota used)
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        print("✅ Gemini API key is VALID!")
        print(f"   Authenticated successfully. {len(models)} model(s) accessible.")
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "API_KEY_INVALID" in error_str or "UNAUTHENTICATED" in error_str:
            print("❌ Gemini API key is INVALID.")
        else:
            print("⚠️  Could not verify key — unexpected error.")
        print(f"   Error: {e}")
        return False

    # Step 2: Test actual generation
    print("\n🔄 Testing generation (sending a message)...")
    try:
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents="Reply with exactly: 'Hello! Gemini is working.'"
        )
        print("✅ Generation test PASSED!")
        print(f"   Model reply: {response.text.strip()}")
        return True
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("⚠️  Generation test SKIPPED — free-tier quota exhausted.")
            print("   Your key is valid but you've hit the rate limit.")
            print("   💡 Check usage/upgrade at: https://ai.dev/rate-limit")
            return True
        else:
            print(f"❌ Generation test FAILED.")
            print(f"   Error: {e}")
            return False


if __name__ == "__main__":
    # Load .env file first
    if load_env_file(".env"):
        print("📄 Loaded variables from .env file.")
    else:
        print("⚠️  No .env file found in current directory.")

    # Try to get API key from argument, then environment variable
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print("Using API key from command-line argument.")
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ["GEMINI_API_KEY"]
        print("Using API key from GEMINI_API_KEY environment variable.")
    else:
        api_key = input("Enter your Gemini API key: ").strip()

    if not api_key:
        print("❌ No API key provided.")
        sys.exit(1)

    success = check_gemini_key(api_key)
    sys.exit(0 if success else 1)