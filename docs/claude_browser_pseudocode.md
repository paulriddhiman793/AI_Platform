# Claude Browser Automation (Pseudocode Only)

This file is **pseudocode** for a future browser-automation flow. It is not wired to the current runtime and will not execute as-is.

Goal: when the user types "train a model" in the GUI, the ML Engineer waits for `shared/output.txt`, opens Claude in a browser, pastes the output, requests a training script, downloads it, writes `ml_engineer/model_training.py`, then runs it in a Jupyter kernel. During this flow, the GUI should show the ML Engineer as working.

## Pseudocode (Python-like)

```python
# Trigger point: GUI "train a model" -> orchestrator -> ML Engineer

async def ml_engineer_on_train_request(task_id: str):
    # 1) Mark agent working in GUI
    publish_status("ml_engineer", "working")

    # 2) Wait for output.txt to be available
    output_path = wait_for_file("shared/output.txt", timeout_s=300)
    if not output_path:
        publish_status("ml_engineer", "error")
        report("output.txt missing; cannot proceed.")
        return

    # 3) Read output.txt contents
    output_text = read_text(output_path)

    # 4) Launch browser (Playwright/Selenium placeholder)
    browser = launch_browser(headless=False)
    page = browser.new_page()
    page.goto("https://claude.ai")

    # 5) Authenticate (assumes user session or stored cookies)
    # TODO: Implement login or cookie restore.
    ensure_logged_in(page)

    # 6) Paste prompt + output.txt
    prompt = (
        "Use the following output to produce a complete training script with feature engineering.\n"
        "Return ONLY code. Ensure no syntax errors.\n\n"
        f"OUTPUT:\n{output_text}\n"
    )
    page.fill("textarea.prompt", prompt)
    page.click("button.send")

    # 7) Wait for Claude response
    response_code = wait_for_code_block(page, timeout_s=180)
    if not response_code:
        publish_status("ml_engineer", "error")
        report("Claude did not return code.")
        return

    # 8) Write response to repo as model_training.py
    write_text("ml_engineer/model_training.py", response_code)

    # 9) Run in Jupyter kernel
    exec_result = run_code_in_jupyter_kernel(response_code, timeout_s=600)
    write_text("ml_engineer/jupyter_output.txt", exec_result.output)

    # 10) Report success/failure
    if exec_result.success:
        report("ML Engineer completed training pipeline (Claude).")
        publish_status("ml_engineer", "idle")
    else:
        report("ML Engineer failed to execute training pipeline (Claude).")
        publish_status("ml_engineer", "error")

    # 11) Cleanup
    browser.close()
```

## Integration Notes (Required for real implementation)

- Browser automation stack: Playwright or Selenium.
- A persistent user session for Claude (cookies or login flow).
- UI selectors for Claude’s input and response elements.
- Permission to launch a browser and interact with the GUI from the agent process.
- Error handling for timeouts, rate limits, and empty responses.

