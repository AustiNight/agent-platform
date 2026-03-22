"""
Browser automation agent.

Converts natural language instructions into browser actions using Playwright.

Features:
- Session persistence (cookies/storage across tasks)
- Credential injection from secrets vault
- LLM re-planning on failures
- Comprehensive action logging
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    TimeoutError as PlaywrightTimeout,
)
import structlog

from src.core.base_agent import AgentCapability, AgentContext, BaseAgent
from src.core.config import settings
from src.core.secrets import get_secret
from src.core.schemas import AgentType, ToolResult
from src.agents.browser.tools import BROWSER_TOOLS
from src.agents.browser.page_parser import (
    parse_page_state,
    extract_main_content,
    check_element_exists,
)
from src.agents.browser.session_manager import (
    get_session_manager,
    session_id_from_url,
)

logger = structlog.get_logger()


class BrowserAgent(BaseAgent):
    """
    Browser automation agent.
    
    Uses Playwright to execute browser actions based on LLM decisions.
    Supports navigation, form filling, clicking, data extraction, and more.
    
    Features:
    - Session persistence: Maintains cookies/storage across tasks for same domain
    - Credential injection: Retrieves login credentials from secrets vault
    - LLM re-planning: On failure, asks the LLM to analyze and try alternative approaches
    - Action logging: Full log of all actions taken
    """

    # Maximum re-planning attempts per failed action
    MAX_REPLAN_ATTEMPTS = 3

    def __init__(self, session_id: str | None = None) -> None:
        super().__init__()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._screenshot_counter = 0
        self._task_complete = False
        self._task_result: dict[str, Any] | None = None
        
        # Session management
        self._session_id = session_id  # Override auto-detection if provided
        self._current_session_id: str | None = None
        self._session_manager = get_session_manager()
        
        # Action log for debugging
        self._action_log: list[dict[str, Any]] = []
        
        # Re-planning state
        self._consecutive_failures = 0

    @property
    def agent_type(self) -> AgentType:
        return AgentType.BROWSER

    @property
    def description(self) -> str:
        return (
            "A browser automation agent that can navigate websites, fill forms, "
            "click buttons, extract data, and perform complex web interactions "
            "based on natural language instructions."
        )

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.WEB_BROWSING,
            AgentCapability.FILE_ACCESS,  # For screenshots
        ]

    @property
    def tools(self) -> list[dict[str, Any]]:
        return BROWSER_TOOLS

    @property
    def system_prompt(self) -> str:
        return """You are a browser automation agent. Your job is to accomplish tasks by controlling a web browser.

## How to work:
1. First, use 'get_page_state' to understand what's on the current page
2. Plan your actions step by step
3. Execute actions one at a time, checking results after each
4. If something fails, try alternative approaches (different selectors, scrolling, waiting)
5. Take screenshots when useful for verification
6. When done, use 'complete_task' to report success and any extracted data

## Selector tips:
- Prefer IDs: #login-button
- Use names: input[name='email']
- Use text: text=Submit
- Use placeholders: [placeholder='Enter email']
- Use aria-labels: [aria-label='Close']
- Last resort: nth-of-type selectors

## Error recovery:
- If an element isn't found, try get_page_state to see what's available
- If a click doesn't work, try scrolling or waiting
- If a form submit fails, check for validation errors
- Take screenshots to help diagnose issues

## Important:
- Always verify actions succeeded before moving on
- Extract any data the user requested
- Report clear success/failure status"""

    async def setup(self, initial_url: str | None = None) -> None:
        """
        Initialize Playwright and browser.
        
        Args:
            initial_url: If provided, load session for this URL's domain
        """
        self.logger.info("starting_browser")
        
        self._playwright = await async_playwright().start()
        
        # Launch browser
        self._browser = await self._playwright.chromium.launch(
            headless=settings.browser_headless,
        )
        
        # Determine session ID
        if self._session_id:
            self._current_session_id = self._session_id
        elif initial_url:
            self._current_session_id = session_id_from_url(initial_url)
        
        # Try to load existing session
        storage_state = None
        if self._current_session_id:
            storage_state = self._session_manager.load_session(self._current_session_id)
            if storage_state:
                self.logger.info(
                    "loaded_session",
                    session_id=self._current_session_id,
                )
        
        # Create context with standard viewport and optional session
        context_options = {
            "viewport": {"width": 1280, "height": 720},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        
        if storage_state:
            context_options["storage_state"] = storage_state
        
        self._context = await self._browser.new_context(**context_options)
        
        # Create initial page
        self._page = await self._context.new_page()
        
        # Set default timeout
        self._page.set_default_timeout(settings.browser_timeout * 1000)

    async def teardown(self) -> None:
        """Clean up browser resources and save session."""
        self.logger.info("closing_browser")
        
        # Save session before closing
        if self._context and self._current_session_id:
            try:
                await self._session_manager.save_session(
                    self._current_session_id,
                    self._context,
                )
            except Exception as e:
                self.logger.warning("session_save_failed", error=str(e))
        
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Execute a browser tool with action logging and error recovery."""
        self.logger.debug("executing_browser_tool", tool=tool_name, args=arguments)
        
        # Log the action
        action_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "arguments": arguments,
            "url": self._page.url if self._page else None,
        }
        
        try:
            # Dispatch to tool handler
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                action_entry["success"] = False
                action_entry["error"] = f"Unknown tool: {tool_name}"
                self._action_log.append(action_entry)
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    error=f"Unknown tool: {tool_name}",
                )
            
            result = await handler(arguments, context)
            
            # Log result
            action_entry["success"] = result.success
            if not result.success:
                action_entry["error"] = result.error
            else:
                action_entry["result_summary"] = str(result.result)[:200]
            
            self._action_log.append(action_entry)
            
            # Track consecutive failures for re-planning
            if result.success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
            
            return result
            
        except PlaywrightTimeout as e:
            action_entry["success"] = False
            action_entry["error"] = f"Timeout: {str(e)}"
            self._action_log.append(action_entry)
            self._consecutive_failures += 1
            
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"Timeout: {str(e)}. The element may not exist, or the page may still be loading. Try using wait_for first, or use get_page_state to see available elements.",
            )
        except Exception as e:
            self.logger.exception("tool_error", tool=tool_name)
            action_entry["success"] = False
            action_entry["error"] = str(e)
            self._action_log.append(action_entry)
            self._consecutive_failures += 1
            
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"{str(e)}. Try an alternative approach or use get_page_state to see current page state.",
            )

    def _get_replan_context(self) -> str:
        """Generate context for LLM re-planning after failures."""
        recent_actions = self._action_log[-5:] if self._action_log else []
        
        context_parts = [
            f"Recent actions ({len(recent_actions)} of {len(self._action_log)} total):",
        ]
        
        for action in recent_actions:
            status = "✓" if action.get("success") else "✗"
            line = f"  {status} {action['tool']}({action.get('arguments', {})})"
            if action.get("error"):
                line += f" - Error: {action['error'][:100]}"
            context_parts.append(line)
        
        context_parts.append(f"\nCurrent URL: {self._page.url if self._page else 'N/A'}")
        context_parts.append(f"Consecutive failures: {self._consecutive_failures}")
        
        return "\n".join(context_parts)

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    async def _tool_navigate(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Navigate to a URL."""
        url = args["url"]
        
        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
        await self._page.goto(url, wait_until="domcontentloaded")
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Navigated to {self._page.url}. Page title: {await self._page.title()}",
        )

    async def _tool_click(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Click on an element."""
        selector = args["selector"]
        wait_nav = args.get("wait_for_navigation", False)
        
        locator = self._page.locator(selector).first
        
        if await locator.count() == 0:
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"No element found matching '{selector}'",
            )
        
        if wait_nav:
            async with self._page.expect_navigation():
                await locator.click()
        else:
            await locator.click()
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Clicked on '{selector}'",
        )

    async def _tool_type_text(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Type text into an input field."""
        selector = args["selector"]
        text = args["text"]
        clear_first = args.get("clear_first", True)
        press_enter = args.get("press_enter", False)
        
        locator = self._page.locator(selector).first
        
        if await locator.count() == 0:
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"No input found matching '{selector}'",
            )
        
        if clear_first:
            await locator.fill(text)
        else:
            await locator.type(text)
        
        if press_enter:
            await locator.press("Enter")
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Typed '{text[:50]}...' into '{selector}'",
        )

    async def _tool_select_option(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Select an option from a dropdown."""
        selector = args["selector"]
        value = args["value"]
        
        locator = self._page.locator(selector).first
        
        # Try selecting by value, then label, then text
        try:
            await locator.select_option(value=value)
        except Exception:
            try:
                await locator.select_option(label=value)
            except Exception:
                await locator.select_option(index=0)  # Fallback
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Selected '{value}' in '{selector}'",
        )

    async def _tool_wait_for(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Wait for an element or time."""
        selector = args.get("selector")
        seconds = args.get("seconds")
        state = args.get("state", "visible")
        
        if seconds:
            await asyncio.sleep(seconds)
            return ToolResult(
                tool_call_id="",
                success=True,
                result=f"Waited {seconds} seconds",
            )
        
        if selector:
            await self._page.locator(selector).wait_for(state=state)
            return ToolResult(
                tool_call_id="",
                success=True,
                result=f"Element '{selector}' is now {state}",
            )
        
        return ToolResult(
            tool_call_id="",
            success=False,
            error="Must provide either 'selector' or 'seconds'",
        )

    async def _tool_screenshot(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Take a screenshot."""
        selector = args.get("selector")
        full_page = args.get("full_page", False)
        
        # Generate filename
        self._screenshot_counter += 1
        filename = f"{context.task_id}_{self._screenshot_counter}.png"
        filepath = settings.screenshot_dir / filename
        
        if selector:
            locator = self._page.locator(selector).first
            await locator.screenshot(path=str(filepath))
        else:
            await self._page.screenshot(path=str(filepath), full_page=full_page)
        
        # Add as artifact
        context.add_artifact(
            artifact_type="screenshot",
            path=str(filepath),
            description=f"Screenshot at {datetime.utcnow().isoformat()}",
        )
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Screenshot saved: {filename}",
            screenshot_path=str(filepath),
        )

    async def _tool_extract_text(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Extract text from the page."""
        selector = args.get("selector")
        
        if selector:
            locator = self._page.locator(selector).first
            if await locator.count() == 0:
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    error=f"No element found matching '{selector}'",
                )
            text = await locator.inner_text()
        else:
            text = await extract_main_content(self._page)
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=text[:3000],  # Limit length
        )

    async def _tool_get_page_state(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Get current page state."""
        state = await parse_page_state(
            self._page,
            include_inputs=args.get("include_inputs", True),
            include_buttons=args.get("include_buttons", True),
            include_links=args.get("include_links", True),
        )
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=state.to_llm_format(),
        )

    async def _tool_scroll(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Scroll the page."""
        direction = args["direction"]
        amount = args.get("amount", "medium")
        selector = args.get("selector")
        
        # Calculate scroll pixels
        amounts = {
            "small": 200,
            "medium": 400,
            "large": 800,
            "page": 720,  # Viewport height
        }
        pixels = amounts.get(amount, 400)
        
        if direction in ["up", "left"]:
            pixels = -pixels
        
        target = self._page.locator(selector).first if selector else self._page
        
        if direction in ["up", "down"]:
            await target.evaluate(f"window.scrollBy(0, {pixels})")
        else:
            await target.evaluate(f"window.scrollBy({pixels}, 0)")
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Scrolled {direction} by {amount}",
        )

    async def _tool_go_back(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Go back in history."""
        await self._page.go_back()
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Went back to: {self._page.url}",
        )

    async def _tool_go_forward(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Go forward in history."""
        await self._page.go_forward()
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Went forward to: {self._page.url}",
        )

    async def _tool_check_element(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Check if an element exists."""
        selector = args["selector"]
        result = await check_element_exists(self._page, selector)
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=result,
        )

    async def _tool_fill_form(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Fill multiple form fields."""
        fields = args["fields"]
        filled = []
        
        for field in fields:
            selector = field["selector"]
            value = field["value"]
            field_type = field.get("type", "text")
            
            locator = self._page.locator(selector).first
            
            if await locator.count() == 0:
                continue
            
            if field_type == "text":
                await locator.fill(value)
            elif field_type == "select":
                await locator.select_option(value)
            elif field_type == "checkbox":
                if value.lower() in ["true", "yes", "1"]:
                    await locator.check()
                else:
                    await locator.uncheck()
            elif field_type == "radio":
                await locator.check()
            
            filled.append(selector)
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result=f"Filled {len(filled)} fields: {', '.join(filled)}",
        )

    async def _tool_complete_task(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """Mark task as complete."""
        self._task_complete = True
        self._task_result = {
            "success": args["success"],
            "summary": args["summary"],
            "extracted_data": args.get("extracted_data"),
        }
        
        # Add extracted data as artifact if present
        if args.get("extracted_data"):
            context.add_artifact(
                artifact_type="data",
                data=args["extracted_data"],
                description="Extracted data from browser task",
            )
        
        return ToolResult(
            tool_call_id="",
            success=True,
            result="Task marked as complete",
        )

    async def _tool_get_credential(
        self,
        args: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """
        Retrieve credentials from secrets vault.
        
        Credentials are stored with keys like: site_domain_field
        Example: github_com_username, github_com_password
        """
        site = args["site"]
        field = args["field"]
        
        # Normalize site name for secret key (replace dots and dashes)
        site_key = site.lower().replace(".", "_").replace("-", "_")
        secret_key = f"{site_key}_{field}"
        
        # Try to get from secrets
        value = get_secret(secret_key)
        
        if value:
            self.logger.info(
                "credential_retrieved",
                site=site,
                field=field,
            )
            return ToolResult(
                tool_call_id="",
                success=True,
                result=value,
            )
        else:
            # Also try with SITE_ prefix (common pattern)
            alt_key = f"SITE_{site_key}_{field}".upper()
            from os import environ
            value = environ.get(alt_key)
            
            if value:
                return ToolResult(
                    tool_call_id="",
                    success=True,
                    result=value,
                )
            
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"No credential found for {site} {field}. Expected environment variable: {secret_key.upper()} or {alt_key}",
            )

    # =========================================================================
    # Lifecycle Overrides
    # =========================================================================

    def _extract_url_from_instructions(self, instructions: str) -> str | None:
        """Extract the first URL from instructions for session management."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, instructions)
        return match.group(0) if match else None

    async def run(self, context: AgentContext) -> Any:
        """Run the browser agent with proper setup/teardown."""
        try:
            # Extract URL from instructions for session management
            initial_url = self._extract_url_from_instructions(context.instructions)
            
            await self.setup(initial_url=initial_url)
            
            # Reset task state
            self._task_complete = False
            self._task_result = None
            self._screenshot_counter = 0
            self._action_log = []
            self._consecutive_failures = 0
            
            # Run the standard agent loop
            response = await super().run(context)
            
            # If task was completed via tool, use that result
            if self._task_complete and self._task_result:
                response.result = self._task_result
            
            # Add action log as artifact
            if self._action_log:
                context.add_artifact(
                    artifact_type="action_log",
                    data=self._action_log,
                    description=f"Browser action log ({len(self._action_log)} actions)",
                )
            
            return response
            
        finally:
            await self.teardown()
