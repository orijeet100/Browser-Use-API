import os
import json
import uuid
from typing import Any, Dict, List

from openai import OpenAI

# ==============================
# CONFIG
# ==============================

SUBCONSCIOUS_API_KEY = os.environ.get("SUBCONSCIOUS_API_KEY")
BROWSER_MCP_URL = "https://exhibits-rebound-intelligent-requirements.trycloudflare.com/mcp"

if not SUBCONSCIOUS_API_KEY:
    raise RuntimeError("SUBCONSCIOUS_API_KEY env var is required")

FORM_URL = "https://form-testing-playground-subconcious.vercel.app/1"
FULL_NAME = "Arijit Mukherjee"
MODEL_NAME = "tim-large"

# Generate unique session ID for this run
SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"


# ==============================
# HELPER: Extract tool calls from response
# ==============================

def collect_tool_calls(tasks: List[Dict[str, Any]], out: List[Dict[str, Any]]) -> None:
    """Walk the TIM reasoning tree and extract tooluse blocks."""
    for t in tasks:
        tooluse = t.get("tooluse")
        if tooluse:
            out.append({
                "tool_name": tooluse.get("tool_name"),
                "parameters": tooluse.get("parameters"),
                "tool_result": tooluse.get("tool_result"),
            })
        subtasks = t.get("subtasks") or []
        if isinstance(subtasks, list) and subtasks:
            collect_tool_calls(subtasks, out)


# ==============================
# MAIN
# ==============================

def main() -> None:
    print("=" * 60)
    print("🤖 OpenAI-Compatible Client - Saved Tools")
    print("=" * 60)
    print(f"📝 Session ID: {SESSION_ID}")
    print(f"🌐 MCP Endpoint: {BROWSER_MCP_URL}")
    print(f"🎯 Task: Fill form at {FORM_URL}")
    print(f"📊 Model: {MODEL_NAME}")
    print("=" * 60)

    # Initialize OpenAI-compatible client pointing to Subconscious
    client = OpenAI(
        base_url="https://api.subconscious.dev/v1",
        api_key=SUBCONSCIOUS_API_KEY or "",
    )

    # Reference saved tool from Subconscious dashboard
    # This assumes you have configured "browser_mcp" tool in the dashboard
    tools = [
        {
            "type": "browser_mcp"  # Just reference the saved tool name
        }
    ]

    # Prepare messages
    system_msg = {
        "role": "system",
        "content": (
            "# Required first step:\n"
            "0. Call get_tool_schemas and follow the returned schemas exactly.\n"
            "- Use index for element indices (never element_index).\n"
            "- Use tab_index only for tab operations.\n"
            "# Required final step:\n"
            "After successful submission, call close_browser_session with the same session_id."
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Fill out this form using browser_mcp tools with my name: {FULL_NAME}. "
            f"URL: {FORM_URL}"
        )
    }

    messages = [system_msg, user_msg]

    print("\n🚀 Calling TIM with OpenAI-compatible client...")
    print(f"🔧 Using saved tool: browser_mcp")
    print("-" * 60)

    # Call TIM with streaming (required by Subconscious API)
    raw_json_str = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # type: ignore
            tools=tools,  # type: ignore
            stream=True,  # REQUIRED by Subconscious API
        )
        
        # Collect streamed content
        print("\n=== STREAMING FROM TIM ===")
        
        try:
            for chunk in stream:
                try:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    
                    # Extract content from delta
                    content = getattr(delta, "content", None)
                    if content:
                        if isinstance(content, str):
                            raw_json_str += content
                            print(".", end="", flush=True)  # Progress indicator
                        elif isinstance(content, (list, tuple)):
                            # Handle content parts if needed
                            for part in content:
                                text = getattr(part, "text", None)
                                if text:
                                    raw_json_str += text
                                    print(".", end="", flush=True)
                except StopIteration:
                    break
                except Exception as chunk_error:
                    print(f"\n⚠️ Error processing chunk: {chunk_error}")
                    print(f"Chunk data: {chunk}")
                    continue
            
            print("\n✓ Streaming complete")
        
        except Exception as stream_error:
            print(f"\n❌ Error during streaming iteration: {stream_error}")
            print(f"Error type: {type(stream_error).__name__}")
            import traceback
            traceback.print_exc()
            
            # If we got some content before the error, try to use it
            if raw_json_str:
                print(f"\n⚠️ Partial content received ({len(raw_json_str)} chars), attempting to parse...")
            else:
                print("\n❌ No content received before error")
                return
        
    except Exception as e:
        print(f"\n❌ Error creating TIM API stream: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):  # type: ignore
            print(f"Response status: {e.response.status_code}")  # type: ignore
        import traceback
        traceback.print_exc()
        return

    print("\n\n=== RAW JSON FROM TIM ===")
    print(raw_json_str[:500] + "..." if len(raw_json_str) > 500 else raw_json_str)
    print(f"\n📊 Received {len(raw_json_str)} characters")

    # Parse the JSON object that TIM returned
    obj = None
    try:
        obj = json.loads(raw_json_str)
    except json.JSONDecodeError as e:
        print("\n" + "=" * 60)
        print("❌ JSON PARSE ERROR")
        print("=" * 60)
        print("Could not parse TIM output as JSON.")
        print("Error:", e)
        print("\n⚠️ Attempting to extract partial information...")
        
        # Try to extract what we can from partial JSON
        try:
            # Look for reasoning array in the partial JSON
            if '"reasoning"' in raw_json_str and '"tool_name"' in raw_json_str:
                print("\n🔍 DETECTED TOOL CALLS IN PARTIAL JSON:")
                tool_call_count = raw_json_str.count('"tooluse"')
                print(f"   Found {tool_call_count} tool call(s) being prepared")
                
                # Try to extract tool names
                import re
                tool_names = re.findall(r'"tool_name":\s*"([^"]+)"', raw_json_str)
                if tool_names:
                    print(f"\n   Tool calls attempted:")
                    for i, name in enumerate(tool_names, 1):
                        print(f"   {i}. {name}")
                
                # Try to extract parameters
                if '"parameters"' in raw_json_str:
                    print(f"\n   ✅ Parameters were being formatted")
                    session_ids = re.findall(r'"session_id":\s*"([^"]+)"', raw_json_str)
                    if session_ids:
                        print(f"   📝 Session IDs found: {list(set(session_ids))}")
        except Exception as extract_error:
            print(f"   Could not extract partial info: {extract_error}")
        
        print("\n💡 This suggests the tool call was being executed when streaming broke.")
        print("   Check uvicorn logs to see if the request reached your server.")
        return

    # Only process if we successfully parsed
    if obj:
        # Extract the final answer
        answer = obj.get("answer")
        reasoning = obj.get("reasoning", [])

        print("\n" + "=" * 60)
        print("✅ FINAL ANSWER FROM TIM")
        print("=" * 60)
        if isinstance(answer, dict):
            print(json.dumps(answer, indent=2))
        else:
            print(answer)

        # Extract tool calls from reasoning
        tool_calls: List[Dict[str, Any]] = []
        if isinstance(reasoning, list):
            collect_tool_calls(reasoning, tool_calls)

        print("\n" + "=" * 60)
        print(f"🛠 TOOL CALLS USED BY TIM ({len(tool_calls)} total)")
        print("=" * 60)
        
        if not tool_calls:
            print("No tool calls found in reasoning (TIM may have only reasoned textually).")
        else:
            for i, tc in enumerate(tool_calls, start=1):
                print(f"\n--- Tool call #{i} ---")
                print("🔍 tool_name:", tc.get("tool_name"))
                print("🔍 parameters:", json.dumps(tc.get("parameters"), indent=2))
                
                tool_result = tc.get("tool_result")
                if tool_result:
                    if isinstance(tool_result, dict):
                        if "error" in tool_result or "detail" in tool_result:
                            print("❌ tool_result:", json.dumps(tool_result, indent=2))
                        else:
                            print("✅ tool_result:", json.dumps(tool_result, indent=2))
                    else:
                        print("📥 tool_result:", json.dumps(tool_result, indent=2))
                else:
                    print("📥 tool_result: None")
        
        print("\n" + "=" * 60)
        print("🎉 TASK COMPLETED SUCCESSFULLY!" if answer else "⚠️ TASK MAY BE INCOMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()

