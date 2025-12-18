"""Helper to get auto-generated tool documentation for LLM system prompts.

Usage:
    # Option 1: Import directly from the server module
    from browser_use.api.standalone_server import generate_tool_documentation
    docs = generate_tool_documentation()
    
    # Option 2: Fetch from running server
    import requests
    response = requests.get("http://localhost:8000/mcp/documentation")
    docs = response.json()["documentation"]
"""

from browser_use.api.standalone_server import generate_tool_documentation

if __name__ == "__main__":
    print("=" * 80)
    print("AUTO-GENERATED TOOL DOCUMENTATION")
    print("=" * 80)
    print()
    print(generate_tool_documentation())
    print()
    print("=" * 80)
    print("Copy the above into your TIM system prompt!")
    print("=" * 80)
