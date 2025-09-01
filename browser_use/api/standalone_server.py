"""Standalone FastAPI Server for browser-use - completely isolated version.

This server provides browser automation capabilities via REST API endpoints
without importing the main browser_use package to avoid dependency issues.

Usage:
    uvicorn browser_use.api.standalone_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
import sys
import time
import secrets
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add browser-use to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import specific browser_use modules to avoid config system
try:
    from browser_use.browser import BrowserProfile, BrowserSession
    from browser_use.controller.service import Controller
    from browser_use.filesystem.file_system import FileSystem
    from browser_use.llm.openai.chat import ChatOpenAI
    from browser_use.agent.service import Agent
    from browser_use import ActionModel
except ImportError as e:
    logger.error(f"Failed to import browser_use modules: {e}")
    logger.error("Make sure you have browser-use installed with: pip install browser-use")
    sys.exit(1)

# Pydantic models for API requests and responses
class BrowserNavigateRequest(BaseModel):
    url: str = Field(..., description="The URL to navigate to")
    new_tab: bool = Field(False, description="Whether to open in a new tab")
    session_id: str = Field("default", description="Browser session ID")

class BrowserClickRequest(BaseModel):
    index: int = Field(..., description="The index of the element to click")
    new_tab: bool = Field(False, description="Whether to open in a new tab")
    session_id: str = Field("default", description="Browser session ID")

class BrowserTypeRequest(BaseModel):
    index: int = Field(..., description="The index of the input element")
    text: str = Field(..., description="The text to type")
    session_id: str = Field("default", description="Browser session ID")

class BrowserStateRequest(BaseModel):
    include_screenshot: bool = Field(False, description="Whether to include a screenshot")
    session_id: str = Field("default", description="Browser session ID")

class BrowserKeyRequest(BaseModel):
    key: str = Field(..., description="The key to press (e.g., 'Enter', 'Escape', 'Tab', 'Space')")
    session_id: str = Field("default", description="Browser session ID")

class BrowserScrollRequest(BaseModel):
    direction: str = Field("down", description="Direction to scroll ('up' or 'down')")
    session_id: str = Field("default", description="Browser session ID")

class BrowserExtractContentRequest(BaseModel):
    query: str = Field(..., description="What information to extract from the page")
    extract_links: bool = Field(False, description="Whether to include links in the extraction")
    session_id: str = Field("default", description="Browser session ID")

class BrowserGoBackRequest(BaseModel):
    session_id: str = Field("default", description="Browser session ID")

class BrowserListTabsRequest(BaseModel):
    session_id: str = Field("default", description="Browser session ID")

class BrowserSwitchTabRequest(BaseModel):
    tab_index: int = Field(..., description="Index of the tab to switch to")
    session_id: str = Field("default", description="Browser session ID")

class BrowserCloseTabRequest(BaseModel):
    tab_index: int = Field(..., description="Index of the tab to close")
    session_id: str = Field("default", description="Browser session ID")

class RetryWithAgentRequest(BaseModel):
    task: str = Field(..., description="High-level goal and detailed task description")
    max_steps: int = Field(100, description="Maximum number of steps the agent can take")
    model: str = Field("gpt-4o", description="LLM model to use")
    allowed_domains: List[str] = Field([], description="List of domains the agent is allowed to visit")
    use_vision: bool = Field(True, description="Whether to use vision capabilities")
    session_id: str = Field("default", description="Browser session ID")

class CloseSessionRequest(BaseModel):
    session_id: str = Field(..., description="Browser session ID to close")

class MCPRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the tool")

class PasswordGenerationRequest(BaseModel):
    website: str = Field(..., description="Website/domain for which to generate credentials")
    email: str = Field(..., description="Email address to use for the account")
    session_id: str = Field("default", description="Browser session ID")

class AccountRetrievalRequest(BaseModel):
    website: str = Field(..., description="Website/domain for which to retrieve credentials")
    email: str = Field(..., description="Email address to retrieve credentials for")
    session_id: str = Field("default", description="Browser session ID")

class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Custom session ID")
    headless: bool = Field(True, description="Whether to run browser in headless mode")
    allowed_domains: List[str] = Field([], description="List of allowed domains")
    wait_between_actions: float = Field(0.5, description="Wait time between actions")

class AgentTaskRequest(BaseModel):
    task: str = Field(..., description="The task description for the agent")
    max_steps: int = Field(100, description="Maximum number of steps")
    model: str = Field("gpt-4o", description="LLM model to use")
    session_id: str = Field("default", description="Browser session ID")

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str

class SelectDropdownRequest(BaseModel):
    session_id: str
    index: int
    text: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    session_id: str

class BrowserStateResponse(BaseModel):
    url: str
    title: str
    tabs: List[Dict[str, str]]
    interactive_elements: List[Dict[str, Any]]
    screenshot: Optional[str] = None

# Global state management
class ServerState:
    def __init__(self):
        self.browser_sessions: Dict[str, BrowserSession] = {}
        self.agents: Dict[str, Agent] = {}
        self.controllers: Dict[str, Controller] = {}
        self.file_systems: Dict[str, FileSystem] = {}
        # Account storage: {session_id: {website: {email: {"email": str, "password": str}}}}
        self.account_storage: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}

    async def cleanup(self):
        """Clean up all active sessions"""
        logger.info("Cleaning up server state...")
        
        # Close all agents
        for agent_id, agent in self.agents.items():
            try:
                await agent.close()
                logger.debug(f"Closed agent {agent_id}")
            except Exception as e:
                logger.error(f"Error closing agent {agent_id}: {e}")
        
        # Close all browser sessions
        for session_id, session in self.browser_sessions.items():
            try:
                await session.stop()
                logger.debug(f"Closed browser session {session_id}")
            except Exception as e:
                logger.error(f"Error closing browser session {session_id}: {e}")

    def generate_password(self, length: int = 16) -> str:
        """Generate a secure random password"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for i in range(length))
        return password

    def store_account(self, session_id: str, website: str, email: str, password: str):
        """Store account credentials for a session and website"""
        if session_id not in self.account_storage:
            self.account_storage[session_id] = {}
        
        if website not in self.account_storage[session_id]:
            self.account_storage[session_id][website] = {}
        
        self.account_storage[session_id][website][email] = {
            "email": email,
            "password": password
        }
        
        # Also save to file for persistence
        self._save_accounts_to_file()

    def get_account(self, session_id: str, website: str, email: str) -> Optional[Dict[str, str]]:
        """Retrieve account credentials for a session, website, and email"""
        return self.account_storage.get(session_id, {}).get(website, {}).get(email)

    def _save_accounts_to_file(self):
        """Save accounts to a JSON file for persistence"""
        try:
            accounts_file = Path.home() / '.browser-use-api' / 'accounts.json'
            accounts_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(accounts_file, 'w') as f:
                json.dump(self.account_storage, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving accounts to file: {e}")

    def _load_accounts_from_file(self):
        """Load accounts from JSON file"""
        try:
            accounts_file = Path.home() / '.browser-use-api' / 'accounts.json'
            if accounts_file.exists():
                with open(accounts_file, 'r') as f:
                    self.account_storage = json.load(f)
                logger.info("Loaded existing account storage from file")
        except Exception as e:
            logger.error(f"Error loading accounts from file: {e}")
            self.account_storage = {}

# Global server state
server_state = ServerState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle server startup and shutdown"""
    # Startup
    logger.info("Starting standalone browser-use API server")
    server_state._load_accounts_from_file()
    yield
    # Shutdown
    await server_state.cleanup()

# Create FastAPI app
def create_app() -> FastAPI:
    app = FastAPI(
        title="Browser-Use Standalone API",
        description="Standalone REST API for browser automation capabilities",
        version="0.1.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "message": "Browser-Use Standalone API Server",
            "version": "0.1.0",
            "docs": "/docs",
            "mcp_endpoint": "/mcp",
            "note": "Set OPENAI_API_KEY environment variable for agent functionality",
            "capabilities": [
                "Browser navigation and control",
                "Element clicking and typing",
                "Keyboard input and scrolling", 
                "Content extraction with LLM",
                "Tab management",
                "Browser history navigation",
                "Autonomous agent tasks"
            ],
            "endpoints": {
                "unified_mcp": "POST /mcp - Single endpoint for all tools",
                "individual": "Original REST endpoints still available",
                "docs": "GET /docs - Interactive API documentation"
            }
        }

    @app.get("/health")
    async def health_check():
        api_key_status = "set" if os.getenv('OPENAI_API_KEY') else "not set"
        return {
            "status": "healthy",
            "active_sessions": len(server_state.browser_sessions),
            "active_agents": len(server_state.agents),
            "openai_api_key": api_key_status
        }

    # Helper function to get or create session
    async def get_session(session_id: str) -> BrowserSession:
        if session_id not in server_state.browser_sessions:
            if session_id == "default":
                # Create default session with sensible defaults
                default_request = CreateSessionRequest(
                    session_id="default",
                    headless=True,
                    allowed_domains=[],
                    wait_between_actions=0.5
                )
                await create_session(default_request)
            else:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return server_state.browser_sessions[session_id]

    # Session management endpoints
    @app.post("/sessions", response_model=SessionResponse)
    async def create_session(request: CreateSessionRequest):
        """Create a new browser session"""
        try:
            session_id = request.session_id or f"session_{int(time.time() * 1000)}"
            
            if session_id in server_state.browser_sessions:
                raise HTTPException(status_code=400, detail=f"Session {session_id} already exists")

            # Create browser profile with basic settings
            profile = BrowserProfile(
                downloads_path=str(Path.home() / 'Downloads' / 'browser-use-api'),
                wait_between_actions=request.wait_between_actions,
                keep_alive=True,
                user_data_dir='~/.config/browseruse/profiles/api',
                headless=request.headless,
                allowed_domains=request.allowed_domains,
            )
            
            session = BrowserSession(browser_profile=profile)
            await session.start()

            # Store session and create controller
            server_state.browser_sessions[session_id] = session
            server_state.controllers[session_id] = Controller()

            # Initialize FileSystem
            file_system_path = Path.home() / '.browser-use-api'
            server_state.file_systems[session_id] = FileSystem(base_dir=file_system_path)

            logger.info(f"Created browser session {session_id}")
            
            return SessionResponse(
                session_id=session_id,
                status="created",
                message=f"Browser session {session_id} created successfully"
            )

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions/close")
    async def close_session(request: CloseSessionRequest):
        """Close a browser session"""
        try:
            session_id = request.session_id
            if session_id not in server_state.browser_sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

            # Close agent if exists
            if session_id in server_state.agents:
                await server_state.agents[session_id].close()
                del server_state.agents[session_id]

            # Close browser session
            session = server_state.browser_sessions[session_id]
            await session.stop()
            del server_state.browser_sessions[session_id]

            # Clean up other resources
            if session_id in server_state.controllers:
                del server_state.controllers[session_id]
            if session_id in server_state.file_systems:
                del server_state.file_systems[session_id]

            logger.info(f"Closed browser session {session_id}")
            
            return {"message": f"Session {session_id} closed successfully"}

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions")
    async def list_sessions():
        """List all active browser sessions"""
        sessions = []
        for session_id, session in server_state.browser_sessions.items():
            try:
                current_page = await session.get_current_page()
                sessions.append({
                    "session_id": session_id,
                    "url": current_page.url if current_page else None,
                    "title": await current_page.title() if current_page and not current_page.is_closed() else None,
                    "has_agent": session_id in server_state.agents
                })
            except Exception as e:
                sessions.append({
                    "session_id": session_id,
                    "error": str(e),
                    "has_agent": session_id in server_state.agents
                })
        
        return {"sessions": sessions}

    # Unified MCP endpoint
    @app.post("/mcp")
    async def mcp_endpoint(request: MCPRequest, background_tasks: BackgroundTasks):
        """Unified MCP endpoint that routes to appropriate tools based on tool_name"""
        try:
            tool_name = request.tool_name
            parameters = request.parameters
            
            # Session management tools
            if tool_name == "create_browser_session":
                create_req = CreateSessionRequest(**parameters)
                return await create_session(create_req)
            
            elif tool_name == "list_browser_sessions":
                return await list_sessions()
            
            elif tool_name == "close_browser_session":
                close_req = CloseSessionRequest(**parameters)
                return await close_session(close_req)
            
            # Browser navigation tools
            elif tool_name == "browser_navigate":
                nav_req = BrowserNavigateRequest(**parameters)
                return await browser_navigate(nav_req)
            
            elif tool_name == "browser_click":
                click_req = BrowserClickRequest(**parameters)
                return await browser_click(click_req)
            
            elif tool_name == "browser_type":
                type_req = BrowserTypeRequest(**parameters)
                return await browser_type(type_req)
            
            elif tool_name == "browser_key":
                key_req = BrowserKeyRequest(**parameters)
                return await browser_key(key_req)
            
            elif tool_name == "browser_scroll":
                scroll_req = BrowserScrollRequest(**parameters)
                return await browser_scroll(scroll_req)
            
            elif tool_name == "browser_get_state":
                state_req = BrowserStateRequest(**parameters)
                return await get_browser_state(state_req)
            
            # Content extraction tool
            elif tool_name == "browser_extract_content":
                extract_req = BrowserExtractContentRequest(**parameters)
                return await browser_extract_content(extract_req)
            
            # Browser navigation tools
            elif tool_name == "browser_go_back":
                back_req = BrowserGoBackRequest(**parameters)
                return await browser_go_back(back_req)
            
            # Tab management tools
            elif tool_name == "browser_list_tabs":
                list_tabs_req = BrowserListTabsRequest(**parameters)
                return await browser_list_tabs(list_tabs_req)
            
            elif tool_name == "browser_switch_tab":
                switch_req = BrowserSwitchTabRequest(**parameters)
                return await browser_switch_tab(switch_req)
            
            elif tool_name == "browser_close_tab":
                close_tab_req = BrowserCloseTabRequest(**parameters)
                return await browser_close_tab(close_tab_req)
            
            # Agent tools
            elif tool_name == "browse_agent":
                agent_req = AgentTaskRequest(**parameters)
                return await run_agent_task(agent_req, background_tasks)
            
            elif tool_name == "retry_browse_agent":
                retry_req = RetryWithAgentRequest(**parameters)
                return await retry_with_browser_use_agent(retry_req, background_tasks)
            
            elif tool_name == "get_agent_task_status":
                task_id = parameters.get("task_id")
                if not task_id:
                    raise HTTPException(status_code=400, detail="task_id parameter required")
                return await get_agent_task_status(task_id)
            
            elif tool_name == "select_dropdown_option":
                select_req = SelectDropdownRequest(**parameters)
                return await select_dropdown_option(select_req)

            # Account management tools
            elif tool_name == "generate_account_credentials":
                cred_req = PasswordGenerationRequest(**parameters)
                return await generate_account_credentials(cred_req)
            
            elif tool_name == "retrieve_account_credentials":
                retrieve_req = AccountRetrievalRequest(**parameters)
                return await retrieve_account_credentials(retrieve_req)
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error in MCP endpoint for tool {request.tool_name}: {e}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/mcp")
    async def mcp_get_endpoint():
        """GET endpoint for MCP - returns available tools"""
        return {
            "message": "MCP Unified Endpoint",
            "method": "POST",
            "endpoint": "/mcp",
            "description": "Send tool requests with tool_name and parameters",
            "available_tools": [
                # Session management
                "create_browser_session",
                "list_browser_sessions", 
                "close_browser_session",
                # Browser control
                "browser_navigate",
                "browser_click",
                "browser_type",
                "browser_key",
                "browser_scroll",
                "browser_get_state",
                # Content extraction
                "browser_extract_content",
                # Navigation
                "browser_go_back",
                # Tab management
                "browser_list_tabs",
                "browser_switch_tab",
                "browser_close_tab",
                # Agent tasks
                "run_agent_task",
                "retry_with_browser_use_agent",
                "get_agent_task_status",
                # Account management
                "generate_account_credentials",
                "retrieve_account_credentials"
            ],
            "schema": {
                "tool_name": "string (required)",
                "parameters": "object (required) - parameters specific to each tool"
            }
        }

    # Browser control endpoints
    @app.post("/browser/navigate")
    async def browser_navigate(request: BrowserNavigateRequest):
        """Navigate to a URL"""
        try:
            session = await get_session(request.session_id)
            
            if request.new_tab:
                page = await session.create_new_tab(request.url)
                tab_idx = session.tabs.index(page)
                message = f'Opened new tab #{tab_idx} with URL: {request.url}'
            else:
                await session.navigate_to(request.url)
                message = f'Navigated to: {request.url}'

            return {"message": message}

        except Exception as e:
            logger.error(f"Error navigating: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/click")
    async def browser_click(request: BrowserClickRequest):
        """Click an element by index"""
        try:
            session = await get_session(request.session_id)
            
            element = await session.get_dom_element_by_index(request.index)
            if not element:
                raise HTTPException(status_code=404, detail=f'Element with index {request.index} not found')

            await session._click_element_node(element)
            return {"message": f'Clicked element {request.index}'}

        except Exception as e:
            logger.error(f"Error clicking element: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/type")
    async def browser_type(request: BrowserTypeRequest):
        """Type text into an element"""
        try:
            session = await get_session(request.session_id)
            
            element = await session.get_dom_element_by_index(request.index)
            if not element:
                raise HTTPException(status_code=404, detail=f'Element with index {request.index} not found')

            await session._input_text_element_node(element, request.text)
            return {"message": f"Typed '{request.text}' into element {request.index}"}

        except Exception as e:
            logger.error(f"Error typing text: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/key")
    async def browser_key(request: BrowserKeyRequest):
        """Press a keyboard key"""
        try:
            session = await get_session(request.session_id)
            page = await session.get_current_page()
            
            # Press the specified key
            await page.keyboard.press(request.key)
            
            return {"message": f"Pressed key: {request.key}"}

        except Exception as e:
            logger.error(f"Error pressing key: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/browser/select")
    async def select_dropdown_option(
        request: SelectDropdownRequest
        # index: int,
        # text: str
    ):
        browser_session = await get_session(request.session_id)
        index = request.index
        text = request.text

        """Select dropdown option or ARIA menu item by the text of the option you want to select"""
        page = await browser_session.get_current_page()
        dom_element = await browser_session.get_dom_element_by_index(index)
        if dom_element is None:
            raise Exception(f'Element index {index} does not exist - retry or use alternative actions')

        logger.debug(f"Attempting to select '{text}' using xpath: {dom_element.xpath}")
        logger.debug(f'Element attributes: {dom_element.attributes}')
        logger.debug(f'Element tag: {dom_element.tag_name}')

        xpath = '//' + dom_element.xpath

        try:
            frame_index = 0
            for frame in page.frames:
                try:
                    logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

                    # First check what type of element we're dealing with
                    element_info_js = """
                        (xpath) => {
                            try {
                                const element = document.evaluate(xpath, document, null,
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!element) return null;
                                
                                const tagName = element.tagName.toLowerCase();
                                const role = element.getAttribute('role');
                                
                                // Check if it's a native select
                                if (tagName === 'select') {
                                    return {
                                        type: 'select',
                                        found: true,
                                        id: element.id,
                                        name: element.name,
                                        tagName: element.tagName,
                                        optionCount: element.options.length,
                                        currentValue: element.value,
                                        availableOptions: Array.from(element.options).map(o => o.text.trim())
                                    };
                                }
                                
                                // Check if it's an ARIA menu or similar
                                if (role === 'menu' || role === 'listbox' || role === 'combobox') {
                                    const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
                                    return {
                                        type: 'aria',
                                        found: true,
                                        id: element.id || '',
                                        role: role,
                                        tagName: element.tagName,
                                        itemCount: menuItems.length,
                                        availableOptions: Array.from(menuItems).map(item => item.textContent.trim())
                                    };
                                }
                                
                                return {
                                    error: `Element is neither a select nor an ARIA menu (tag: ${tagName}, role: ${role})`,
                                    found: false
                                };
                            } catch (e) {
                                return {error: e.toString(), found: false};
                            }
                        }
                    """

                    element_info = await frame.evaluate(element_info_js, dom_element.xpath)

                    if element_info and element_info.get('found'):
                        logger.debug(f'Found {element_info.get("type")} element in frame {frame_index}: {element_info}')

                        if element_info.get('type') == 'select':
                            # Handle native select element
                            # "label" because we are selecting by text
                            # nth(0) to disable error thrown by strict mode
                            # timeout=1000 because we are already waiting for all network events
                            selected_option_values = (
                                await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=1000)
                            )

                            msg = f'selected option {text} with value {selected_option_values}'
                            logger.info(msg + f' in frame {frame_index}')

                            return {'message': msg}

                        elif element_info.get('type') == 'aria':
                            # Handle ARIA menu
                            click_aria_item_js = """
                                (params) => {
                                    const { xpath, targetText } = params;
                                    try {
                                        const element = document.evaluate(xpath, document, null,
                                            XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                        if (!element) return {success: false, error: 'Element not found'};
                                        
                                        // Find all menu items
                                        const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
                                        
                                        for (const item of menuItems) {
                                            const itemText = item.textContent.trim();
                                            if (itemText === targetText) {
                                                // Simulate click on the menu item
                                                item.click();
                                                
                                                // Also try dispatching a click event in case the click handler needs it
                                                const clickEvent = new MouseEvent('click', {
                                                    view: window,
                                                    bubbles: true,
                                                    cancelable: true
                                                });
                                                item.dispatchEvent(clickEvent);
                                                
                                                return {
                                                    success: true,
                                                    message: `Clicked menu item: ${targetText}`
                                                };
                                            }
                                        }
                                        
                                        return {
                                            success: false,
                                            error: `Menu item with text '${targetText}' not found`
                                        };
                                    } catch (e) {
                                        return {success: false, error: e.toString()};
                                    }
                                }
                            """

                            result = await frame.evaluate(
                                click_aria_item_js, {'xpath': dom_element.xpath, 'targetText': text}
                            )

                            if result.get('success'):
                                msg = result.get('message', f'Selected ARIA menu item: {text}')
                                logger.info(msg + f' in frame {frame_index}')
                                return {'message': msg}
                            else:
                                logger.error(f'Failed to select ARIA menu item: {result.get("error")}')
                                continue

                    elif element_info:
                        logger.error(f'Frame {frame_index} error: {element_info.get("error")}')
                        continue

                except Exception as frame_e:
                    logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
                    logger.error(f'Frame type: {type(frame)}')
                    logger.error(f'Frame URL: {frame.url}')

                frame_index += 1

            msg = f"Could not select option '{text}' in any frame"
            logger.info(msg)
            return {'message': msg}

        except Exception as e:
            msg = f'Selection failed: {str(e)}'
            logger.error(msg)
            return {'message': msg}

    @app.post("/browser/scroll")
    async def browser_scroll(request: BrowserScrollRequest):
        """Scroll the page"""
        try:
            session = await get_session(request.session_id)
            page = await session.get_current_page()
            
            # Validate direction
            if request.direction not in ["up", "down"]:
                raise HTTPException(status_code=400, detail="Direction must be 'up' or 'down'")
            
            # Get viewport height for scrolling
            viewport_height = await page.evaluate('() => window.innerHeight')
            
            # Calculate scroll distance (positive for down, negative for up)
            scroll_distance = viewport_height if request.direction == "down" else -viewport_height
            
            # Perform the scroll
            await page.evaluate('(distance) => window.scrollBy(0, distance)', scroll_distance)
            
            return {"message": f"Scrolled {request.direction} by {abs(scroll_distance)} pixels"}

        except Exception as e:
            logger.error(f"Error scrolling: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/extract_content")
    async def browser_extract_content(request: BrowserExtractContentRequest):
        """Extract structured content from the current page"""
        try:
            session = await get_session(request.session_id)
            
            # Get API key for LLM
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise HTTPException(status_code=400, detail='OPENAI_API_KEY environment variable not set for content extraction')

            # Get controller and file system
            controller = server_state.controllers.get(request.session_id)
            file_system = server_state.file_systems.get(request.session_id)
            
            if not controller or not file_system:
                raise HTTPException(status_code=500, detail='Controller or FileSystem not initialized for session')

            # Create LLM for extraction
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Using mini for content extraction to reduce costs
                api_key=api_key,
                temperature=0.7,
            )

            # Use the extract_structured_data action
            from pydantic import create_model
            from browser_use import ActionModel

            # Create action model dynamically
            ExtractAction = create_model(
                'ExtractAction',
                __base__=ActionModel,
                extract_structured_data=(dict[str, Any], {'query': request.query, 'extract_links': request.extract_links}),
            )

            action = ExtractAction()
            action_result = await controller.act(
                action=action,
                browser_session=session,
                page_extraction_llm=llm,
                file_system=file_system,
            )

            extracted_content = action_result.extracted_content or 'No content extracted'
            
            return {"extracted_content": extracted_content}

        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/go_back")
    async def browser_go_back(request: BrowserGoBackRequest):
        """Go back to the previous page"""
        try:
            session = await get_session(request.session_id)
            await session.go_back()
            return {"message": "Navigated back"}

        except Exception as e:
            logger.error(f"Error going back: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/list_tabs")
    async def browser_list_tabs(request: BrowserListTabsRequest):
        """List all open tabs"""
        try:
            session = await get_session(request.session_id)
            
            tabs = []
            for i, tab in enumerate(session.tabs):
                try:
                    tab_info = {
                        'index': i, 
                        'url': tab.url, 
                        'title': await tab.title() if not tab.is_closed() else 'Closed'
                    }
                except Exception:
                    tab_info = {
                        'index': i,
                        'url': 'Unknown',
                        'title': 'Error getting tab info'
                    }
                tabs.append(tab_info)
            
            return {"tabs": tabs}

        except Exception as e:
            logger.error(f"Error listing tabs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/switch_tab")
    async def browser_switch_tab(request: BrowserSwitchTabRequest):
        """Switch to a different tab"""
        try:
            session = await get_session(request.session_id)
            
            if request.tab_index < 0 or request.tab_index >= len(session.tabs):
                raise HTTPException(status_code=400, detail=f"Invalid tab index: {request.tab_index}")
            
            await session.switch_to_tab(request.tab_index)
            page = await session.get_current_page()
            
            return {"message": f"Switched to tab {request.tab_index}: {page.url}"}

        except Exception as e:
            logger.error(f"Error switching tab: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/close_tab")
    async def browser_close_tab(request: BrowserCloseTabRequest):
        """Close a specific tab"""
        try:
            session = await get_session(request.session_id)
            
            if request.tab_index < 0 or request.tab_index >= len(session.tabs):
                raise HTTPException(status_code=400, detail=f"Invalid tab index: {request.tab_index}")
            
            tab = session.tabs[request.tab_index]
            url = tab.url
            await tab.close()
            
            return {"message": f"Closed tab {request.tab_index}: {url}"}

        except Exception as e:
            logger.error(f"Error closing tab: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/retry_with_agent")
    async def retry_with_browser_use_agent(request: RetryWithAgentRequest, background_tasks: BackgroundTasks):
        """Retry a task using the browser-use agent as fallback"""
        try:
            # Get API key for LLM
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise HTTPException(status_code=400, detail='OPENAI_API_KEY environment variable not set')

            # Create LLM
            llm = ChatOpenAI(
                model=request.model,
                api_key=api_key,
                temperature=0.7,
            )

            # Create browser profile with allowed domains
            profile = BrowserProfile(
                downloads_path=str(Path.home() / 'Downloads' / 'browser-use-api'),
                wait_between_actions=0.5,
                keep_alive=True,
                user_data_dir='~/.config/browseruse/profiles/api-agent',
                headless=True,
                allowed_domains=request.allowed_domains,
            )

            # Create and run agent
            agent = Agent(
                task=request.task,
                llm=llm,
                browser_profile=profile,
                use_vision=request.use_vision,
            )

            # Generate unique task ID
            task_id = f"retry_task_{int(time.time() * 1000)}"
            
            # Store agent (will replace if session_id exists)
            server_state.agents[task_id] = agent

            # Run agent in background
            async def run_retry_agent():
                try:
                    history = await agent.run(max_steps=request.max_steps)
                    logger.info(f"Retry agent task {task_id} completed. Success: {history.is_successful()}")
                except Exception as e:
                    logger.error(f"Retry agent task {task_id} failed: {e}")
                finally:
                    # Clean up agent
                    try:
                        await agent.close()
                    except Exception as e:
                        logger.error(f"Error closing retry agent {task_id}: {e}")

            background_tasks.add_task(run_retry_agent)

            return {
                "task_id": task_id,
                "status": "started",
                "message": f"Retry agent task {task_id} started with {request.max_steps} max steps",
                "session_id": request.session_id
            }

        except Exception as e:
            logger.error(f"Error running retry agent: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/browser/state", response_model=BrowserStateResponse)
    async def get_browser_state(request: BrowserStateRequest):
        """Get current browser state"""
        try:
            session = await get_session(request.session_id)
            
            state = await session.get_browser_state_with_recovery(cache_clickable_elements_hashes=False)

            interactive_elements = []
            for index, element in state.selector_map.items():
                elem_info = {
                    'index': index,
                    'tag': element.tag_name,
                    'text': element.get_all_text_till_next_clickable_element(max_depth=2)[:100],
                }
                if element.attributes.get('placeholder'):
                    elem_info['placeholder'] = element.attributes['placeholder']
                if element.attributes.get('href'):
                    elem_info['href'] = element.attributes['href']
                interactive_elements.append(elem_info)

            response_data = {
                'url': state.url,
                'title': state.title,
                'tabs': [{'url': tab.url, 'title': tab.title} for tab in state.tabs],
                'interactive_elements': interactive_elements,
            }

            if request.include_screenshot and state.screenshot:
                response_data['screenshot'] = state.screenshot

            return BrowserStateResponse(**response_data)

        except Exception as e:
            logger.error(f"Error getting browser state: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Agent endpoints
    @app.post("/agent/task", response_model=TaskResponse)
    async def run_agent_task(request: AgentTaskRequest, background_tasks: BackgroundTasks):
        """Run an autonomous agent task"""
        try:
            session = await get_session(request.session_id)
            
            # Get API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise HTTPException(status_code=400, detail='OPENAI_API_KEY environment variable not set')

            llm = ChatOpenAI(
                model=request.model,
                api_key=api_key,
                temperature=0.7,
            )

            # Create agent
            agent = Agent(
                task=request.task,
                llm=llm,
                browser_session=session,
            )

            # Store agent
            task_id = f"task_{int(time.time() * 1000)}"
            server_state.agents[task_id] = agent

            # Run agent in background
            async def run_agent():
                try:
                    await agent.run(max_steps=request.max_steps)
                    logger.info(f"Agent task {task_id} completed successfully")
                except Exception as e:
                    logger.error(f"Agent task {task_id} failed: {e}")

            background_tasks.add_task(run_agent)

            return TaskResponse(
                task_id=task_id,
                status="started",
                message=f"Agent task {task_id} started",
                session_id=request.session_id
            )

        except Exception as e:
            logger.error(f"Error running agent task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agent/task/{task_id}")
    async def get_agent_task_status(task_id: str):
        """Get the status of an agent task"""
        if task_id not in server_state.agents:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        agent = server_state.agents[task_id]
        
        # Check if agent is still running
        is_running = hasattr(agent, 'running') and getattr(agent, 'running', False)
        
        # Get history if available
        history_info = {}
        if hasattr(agent, 'state') and hasattr(agent.state, 'history'):
            history = agent.state.history
            history_info = {
                "steps_completed": len(history.history),
                "is_successful": history.is_successful(),
                "total_duration": history.total_duration_seconds(),
                "urls_visited": [str(url) for url in history.urls() if url is not None],
            }
            
            # Get final result if available
            final_result = history.final_result()
            if final_result:
                history_info["final_result"] = final_result
            
            # Get errors if any
            errors = history.errors()
            if errors:
                history_info["errors"] = errors

        return {
            "task_id": task_id,
            "status": "running" if is_running else "completed",
            **history_info
        }

    # Account management endpoints
    @app.post("/account/generate")
    async def generate_account_credentials(request: PasswordGenerationRequest):
        """Generate account credentials for a website"""
        try:
            # Normalize website URL
            website = request.website.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
            
            # Use provided email and generate password
            email = request.email
            password = server_state.generate_password()
            
            # Store credentials
            server_state.store_account(request.session_id, website, email, password)
            
            logger.info(f"Generated credentials for {website} in session {request.session_id}")
            
            return {
                "website": website,
                "email": email,
                "password": password,
                "session_id": request.session_id,
                "message": f"Generated and stored credentials for {website}"
            }

        except Exception as e:
            logger.error(f"Error generating credentials: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/account/retrieve")
    async def retrieve_account_credentials(request: AccountRetrievalRequest):
        """Retrieve stored account credentials for a website"""
        try:
            # Normalize website URL
            website = request.website.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
            
            # Retrieve credentials
            credentials = server_state.get_account(request.session_id, website, request.email)
            
            if not credentials:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No credentials found for {request.email} on {website} in session {request.session_id}"
                )
            
            logger.info(f"Retrieved credentials for {request.email} on {website} in session {request.session_id}")
            
            ret_info = {
                "website": website,
                "email": credentials["email"],
                "password": credentials["password"],
                "session_id": request.session_id,
                "message": f"Retrieved credentials for {request.email} on {website}"
            }
            print(f"Retrieved credentials: {ret_info}")
            return ret_info

        except Exception as e:
            logger.error(f"Error retrieving credentials: {e}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))

    return app

# Create app instance
app = create_app()

def main():
    """Main entry point for running the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Browser-Use Standalone FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting Browser-Use Standalone API server on {args.host}:{args.port}")
    print(f"API docs will be available at: http://{args.host}:{args.port}/docs")
    print(f"Set OPENAI_API_KEY environment variable for agent functionality")
    
    uvicorn.run(
        "browser_use.api.standalone_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
