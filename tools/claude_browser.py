import time
from pathlib import Path


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_claude_browser_automation(
    output_text: str,
    code_out_path: Path,
    user_data_dir: Path,
    timeout_s: int = 420,
) -> dict:
    """
    Best-effort Claude web automation using Playwright.
    Requires:
      - `playwright` Python package
      - `playwright install` executed at least once
      - A valid Claude session (login) in the launched browser
    """
    log = []

    def _log(msg: str) -> None:
        log.append(f"[{_now()}] {msg}")

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except Exception as exc:
        return {
            "success": False,
            "error": f"playwright not available: {exc}",
            "log": "\n".join(log),
        }

    if not output_text.strip():
        return {
            "success": False,
            "error": "output_text empty",
            "log": "\n".join(log),
        }

    code_out_path.parent.mkdir(parents=True, exist_ok=True)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        "Use the following output to produce a complete training script with feature engineering.\n"
        "Return ONLY Python code. Ensure no syntax errors.\n\n"
        f"OUTPUT:\n{output_text}\n"
    )

    selectors_input = [
        "textarea",
        "div[contenteditable='true']",
        "div[role='textbox']",
    ]
    selectors_send = [
        "button[aria-label='Send']",
        "button:has-text('Send')",
        "button[type='submit']",
    ]

    with sync_playwright() as p:
        _log("Launching Chromium (persistent context).")
        ctx = p.chromium.launch_persistent_context(str(user_data_dir), headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://claude.ai", wait_until="domcontentloaded")
        _log("Navigated to Claude.")

        input_sel = None
        start = time.time()
        while time.time() - start < timeout_s:
            for sel in selectors_input:
                try:
                    page.wait_for_selector(sel, timeout=1500)
                    input_sel = sel
                    break
                except PWTimeout:
                    continue
            if input_sel:
                break
            time.sleep(1.0)

        if not input_sel:
            ctx.close()
            return {
                "success": False,
                "error": "input box not found (login required or UI changed)",
                "log": "\n".join(log),
            }

        _log(f"Found input box: {input_sel}")
        try:
            if input_sel == "textarea":
                page.fill(input_sel, prompt)
            else:
                page.click(input_sel)
                page.keyboard.insert_text(prompt)
        except Exception as exc:
            ctx.close()
            return {
                "success": False,
                "error": f"failed to fill input: {exc}",
                "log": "\n".join(log),
            }

        sent = False
        for sel in selectors_send:
            try:
                page.click(sel, timeout=2000)
                sent = True
                _log(f"Clicked send: {sel}")
                break
            except Exception:
                continue
        if not sent:
            page.keyboard.press("Enter")
            _log("Send button not found; pressed Enter.")

        _log("Waiting for code response.")
        code_text = ""
        start = time.time()
        last_count = 0
        while time.time() - start < timeout_s:
            try:
                blocks = page.locator("pre code")
                count = blocks.count()
                if count > 0 and count != last_count:
                    code_text = blocks.nth(count - 1).inner_text().strip()
                    last_count = count
                if code_text:
                    break
            except Exception:
                pass
            time.sleep(2.0)

        if not code_text:
            ctx.close()
            return {
                "success": False,
                "error": "no code block found in response",
                "log": "\n".join(log),
            }

        code_out_path.write_text(code_text, encoding="utf-8")
        _log(f"Saved code to: {code_out_path}")
        ctx.close()
        return {"success": True, "error": "", "log": "\n".join(log)}

