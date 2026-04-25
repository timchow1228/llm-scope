"""
Phase 1 验收测试脚本。
用法:
  1. 设置环境变量: $env:DEEPSEEK_API_KEY = "sk-your-key"
  2. 确保 uvicorn 在 7070 运行: python -m uvicorn llm_scope.proxy:app --port 7070
  3. 运行: python tests/test_phase1.py
"""

import os
import sys
import httpx
import json


def test_health():
    """测试 /health 端点"""
    print("=" * 60)
    print("Test 1: /health endpoint")
    print("=" * 60)
    resp = httpx.get("http://localhost:7070/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    print("✓ Health check passed\n")


def test_streaming():
    """测试流式转发"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("=" * 60)
        print("Test 2: Streaming (SKIPPED - no DEEPSEEK_API_KEY)")
        print("=" * 60)
        print("⚠ Set $env:DEEPSEEK_API_KEY to run this test\n")
        return

    print("=" * 60)
    print("Test 2: Streaming via proxy")
    print("=" * 60)

    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "stream": True,
    }

    collected_text = ""
    chunk_count = 0

    with httpx.Client(timeout=30.0) as client:
        with client.stream(
            "POST",
            "http://localhost:7070/v1/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ) as resp:
            print(f"  Status: {resp.status_code}")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

            for line in resp.iter_lines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                collected_text += content
                                chunk_count += 1
                                sys.stdout.write(content)
                                sys.stdout.flush()
                        usage = data.get("usage")
                        if usage:
                            print(f"\n  Usage: {json.dumps(usage)}")
                    except json.JSONDecodeError:
                        pass

    print(f"\n  Collected text: {collected_text}")
    print(f"  Chunks with content: {chunk_count}")
    assert len(collected_text) > 0, "No content received"
    print("✓ Streaming test passed\n")
    print(">>> Check the server terminal for the TTFT line <<<")


if __name__ == "__main__":
    test_health()
    test_streaming()
    print("\nAll tests passed! ✓")
