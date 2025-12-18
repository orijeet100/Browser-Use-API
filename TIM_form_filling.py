import os
import json
import uuid
from typing import Any, Dict, List

from subconscious import Client

# ==============================
# CONFIG
# ==============================

SUBCONSCIOUS_API_KEY = os.environ.get("SUBCONSCIOUS_API_KEY")
BROWSER_MCP_URL = "https://days-components-founder-tires.trycloudflare.com/mcp"

if not SUBCONSCIOUS_API_KEY:
    raise RuntimeError("SUBCONSCIOUS_API_KEY env var is required")

if not BROWSER_MCP_URL:
    raise RuntimeError("BROWSER_MCP_URL env var is required")

FORM_URL = "https://form-testing-playground-subconcious.vercel.app/1"
FULL_NAME = "Robert Martinez"
MODEL_NAME = "tim-large"  # Using smaller model for better reliability
AGENT_NAME = "browser_automation_agent"

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
    print("🤖 SUBCONSCIOUS SDK - Browser Automation Agent")
    print("=" * 60)
    print(f"📝 Session ID: {SESSION_ID}")
    print(f"🌐 MCP Endpoint: {BROWSER_MCP_URL}")
    print(f"🎯 Task: Fill form at {FORM_URL}")
    print("=" * 60)

    # Initialize Subconscious client
    client = Client(
        base_url="https://api.subconscious.dev/v1",
        api_key=SUBCONSCIOUS_API_KEY or "",  # Already validated above
    )

    # Define browser_mcp tool with proper schema (matching Subconscious spec)
    tools = [
        {
            "type": "function",
            "name": "browser_mcp",
            "description": "Browser automation tool for navigating, clicking, typing, and extracting data from web pages",
            "url": BROWSER_MCP_URL,
            "method": "POST",
            "timeout": 5000,  # seconds
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "MCP tool to execute (e.g., create_browser_session, navigate, browser_get_state, type, click)"
                    },
                    "parameters": {
                        "type": "string",
                        "description": "JSON string of parameters for the MCP tool. Must include session_id for session-scoped tools."
                    }
                },
                "required": ["tool_name", "parameters"],
                "additionalProperties": False
            }
        }
    ]

    # Build the toolkit (register tools with agent)
    print("\n🔧 Building toolkit...")
    client.build_toolkit(tools, agent_name=AGENT_NAME)
    print("✅ Toolkit built successfully!")

    # Prepare messages with system and user prompts
    # messages = [
    # {
    #             "role": "system",
    #             "content": f"""You are a browser automation agent. You MUST follow these steps in order.

    #     # STEP 1: Get Tool Schemas (MANDATORY FIRST STEP)
    #     Before doing ANYTHING else, call the browser_mcp tool to get available tools:
    #     - tool_name: "get_tool_schemas"
    #     - parameters: "{{}}"

    #     This returns a JSON object with:
    #     - "rules": Array of rules you must follow
    #     - "tools": Object mapping tool names to their parameter schemas and examples

    #     Study the returned schemas carefully and use them exactly as specified.

    #     # STEP 2: Create Browser Session
    #     After getting schemas, create a session:
    #     - tool_name: "create_browser_session"  
    #     - parameters: '{{"session_id": "{SESSION_ID}"}}'

    #     If you get "Session already exists" error, skip creation and continue with the existing session.

    #     # STEP 3: Execute Browser Actions
    #     Use tools from the schema. Common workflow:
    #     1. navigate - Go to URL
    #     2. browser_get_state - Get page elements with their indices
    #     3. type - Enter text into input fields (use "index" from state)
    #     4. click - Click buttons/links (use "index" from state)
        
    #     # STEP 4: Close Browser and Verify Completion (MANDATORY FINAL STEPS)
    #     After submitting the form:
    #     1. close_browser_session - Close the browser with your session_id
    #     2. check_form_submission_status - Verify form was submitted
    #        - If it returns form_submitted: true → TASK COMPLETE, STOP EXECUTION
    #        - If it returns form_submitted: false → Something went wrong

    #     # CRITICAL RULES
    #     - Session ID: Always use "{SESSION_ID}" for all operations
    #     - Element indices: Use "index" (never "element_index")
    #     - Fresh state: Call browser_get_state before click/type to get current indices
    #     - Parameters format: Pass parameters as a JSON string

    #     # ERROR RECOVERY
    #     - "Session already exists": Continue using existing session, don't recreate
    #     - "Element not found": Call browser_get_state to refresh indices, then retry
    #     - Click timeout: Scroll the page, get fresh state, retry with updated index"""
    #         },
    #         {
    #             "role": "user",
    #             "content": f"""Task: Fill out the form at {FORM_URL}

    #     Steps to complete:
    #     1. First call get_tool_schemas to see available tools
    #     2. Create browser session with session_id: {SESSION_ID}
    #     3. Navigate to {FORM_URL}
    #     4. Get browser state to find the name input field
    #     5. Type "{FULL_NAME}" into the name field
    #     6. Get browser state again to find the submit button
    #     7. Click the submit button
    #     8. Close browser session with close_browser_session
    #     9. Check status with check_form_submission_status
    #     10. If form_submitted is True, STOP - task is complete!

    #     Use the browser_mcp tool for all operations. Format:
    #     - tool_name: the MCP tool name (e.g., "get_tool_schemas", "navigate", "click")
    #     - parameters: JSON string with the tool's parameters"""
    #         },
    #     {
    #         "role": "system",
    #         "content": (
    #         f"You must use create session_id: {SESSION_ID} and use itfor all browser operations.\n"
    #         f"Don't recreate the exact session_id: {SESSION_ID}\n\n and rather just use it again if needed for operations"
    #         "# Required first step:\n"
    #         "0. Call get_tool_schemas and follow the returned schemas exactly.\n"
    #         "- Use index for element indices (never element_index).\n"
    #         "- Use tab_index only for tab operations.\n\n"
    #         "# Error handling:\n"
    #         "- If 'Session already exists' error: DO NOT recreate. Just continue using the existing session.\n"
    #         "- If click fails: Try scrolling first, or get fresh state and retry with updated indices.\n"
    #         "- If element not found: Call browser_get_state to refresh element indices.\n\n"
    #         "# CRITICAL - Task Completion:\n"
    #         "After form submission:\n"
    #         "1. Call close_browser_session to close the browser\n"
    #         "2. Call check_form_submission_status to verify completion\n"
    #         "3. If the response shows form_submitted: true, STOP ALL EXECUTION\n"
    #         "4. Return your final summary and exit"
    #     )
    #     },
    #     {
    #         "role": "user",
    #         "content": (
    #             f"Fill out the form at {FORM_URL}\n"
    #             f"Enter name: {FULL_NAME}\n"
    #             f"Then submit the form.\n\n"
    #             f"Use browser_mcp tool with tool_name"
    #         )
    #     }
    # ]


    messages = [
    {
        "role": "system", 
        "content": f"""You are a browser automation agent using hierarchical task decomposition.

# Task Decomposition Rules
- Break the main task into INDEPENDENT subtasks at the SAME LEVEL
- Subtasks should NOT be nested inside each other unless there's a true parent-child relationship
- Sequential steps are SIBLINGS, not nested children
- Only nest when a subtask genuinely requires sub-decomposition

# Correct Structure Example:
Task: "Fill form"
├── Subtask: "Initialize" (get_tool_schemas, create_browser_session)
├── Subtask: "Navigate to form" (navigate)
├── Subtask: "Fill fields" (browser_get_state, type)
├── Subtask: "Submit" (browser_get_state, click)
└── Subtask: "Verify" (check result)

# WRONG Structure (avoid this):
Task: "Get schemas"
└── Subtask: "Create session"  ← Wrong! These are siblings, not parent-child
    └── Subtask: "Navigate"    ← Wrong! 
        └── Subtask: "Type"    ← Wrong!

# Session: {SESSION_ID}
# Target: {FORM_URL}
# Data: Name = "{FULL_NAME}"

# Tool format:
tool_name: "browser_mcp"
parameters: {{"tool_name": "<tool>", "parameters": "<JSON string>"}}

# Available tools (call get_tool_schemas first to confirm):
- get_tool_schemas: Get all available tools
- create_browser_session: Start browser with session_id
- navigate: Go to URL
- browser_get_state: Get page elements with indices
- type: Enter text at element index
- click: Click element at index
- close_browser_session: End session"""
    },
    {
        "role": "user",
        "content": f"""Fill out the form at {FORM_URL} with name "{FULL_NAME}" and submit it.

Decompose this into independent subtasks at the same level:
1. Initialize browser tools and session
2. Navigate to the form
3. Find and fill the name field  
4. Find and click submit button
5. Verify submission succeeded

Use session_id: {SESSION_ID} for all browser operations."""
    }
]
    # Run the agent
    print("\n🚀 Running agent...")
    print(f"📊 Model: {MODEL_NAME}")
    print("-" * 60)
    
    try:
        # Try to pass model if SDK supports it, otherwise use default
        try:
            response = client.agent.run(messages, agent_name=AGENT_NAME, model=MODEL_NAME)
        except TypeError:
            # SDK might not support model parameter, use default
            print("⚠️ Note: Using default model (SDK doesn't support model parameter)")
            response = client.agent.run(messages, agent_name=AGENT_NAME)
        
        print("\n" + "=" * 60)
        print("📥 RAW AGENT RESPONSE")
        print("=" * 60)
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        # Parse response like the SDK example shows
        ans_obj = None
        if hasattr(response, 'content'):
            # SDK returns object with .content attribute
            print(f"\n✅ Response has .content attribute")
            try:
                content = response.content
                if isinstance(content, str):
                    ans_obj = json.loads(content)
                    print("✅ Successfully parsed response.content as JSON")
                elif isinstance(content, dict):
                    ans_obj = content
                    print("✅ response.content is already a dict")
                else:
                    print(f"⚠️ Unexpected content type: {type(content)}")
                    print(f"Raw content: {content}")
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                print(f"⚠️ Could not parse response.content: {e}")
                print(f"Raw content: {response.content}")
        elif isinstance(response, dict):
            # Already a dict
            ans_obj = response
            print("✅ Response is already a dict")
        elif isinstance(response, str):
            # String response, try to parse
            try:
                ans_obj = json.loads(response)
                print("✅ Successfully parsed response string as JSON")
            except json.JSONDecodeError:
                print("⚠️ Response is string but not valid JSON")
                print(response)
        
        if ans_obj:
            print("\n" + "=" * 60)
            print("🔍 ALL AVAILABLE KEYS IN RESPONSE")
            print("=" * 60)
            print(f"Keys: {list(ans_obj.keys())}")
            
            print("\n" + "=" * 60)
            print("📊 FULL REASONING")
            print("=" * 60)
            reasoning = ans_obj.get("reasoning", [])
            print(json.dumps(reasoning, indent=2))
            
            print("\n" + "=" * 60)
            print("✅ FINAL ANSWER")
            print("=" * 60)
            answer = ans_obj.get("answer")
            if isinstance(answer, dict):
                print(json.dumps(answer, indent=2))
            else:
                print(answer)
            
            # Extract tool calls from reasoning
            if reasoning and isinstance(reasoning, list):
                tool_calls: List[Dict[str, Any]] = []
                collect_tool_calls(reasoning, tool_calls)
                
                if tool_calls:
                    print("\n" + "=" * 60)
                    print(f"🛠 TOOL CALLS ({len(tool_calls)} total)")
                    print("=" * 60)
                    
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
        else:
            print("\n⚠️ Could not extract structured data from response")
            print("Raw response:")
            print(response)
        
        print("\n" + "=" * 60)
        print("🎉 AGENT EXECUTION COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
