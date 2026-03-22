"""
Page parsing utilities for the browser agent.

Extracts structured information from web pages in a format the LLM can understand.
"""

from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page


@dataclass
class PageElement:
    """Represents an interactive element on the page."""
    
    tag: str
    selector: str
    text: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    value: str | None = None
    element_type: str | None = None  # input type, button, link, etc.


@dataclass 
class PageState:
    """Represents the current state of a web page."""
    
    url: str
    title: str
    inputs: list[PageElement] = field(default_factory=list)
    buttons: list[PageElement] = field(default_factory=list)
    links: list[PageElement] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    main_text: str = ""
    
    def to_llm_format(self) -> str:
        """Format page state for the LLM."""
        parts = [
            f"Current URL: {self.url}",
            f"Page Title: {self.title}",
        ]
        
        if self.headings:
            parts.append(f"\nMain Headings: {', '.join(self.headings[:5])}")
        
        if self.inputs:
            parts.append("\n--- Form Inputs ---")
            for inp in self.inputs[:20]:  # Limit to prevent context overflow
                input_desc = f"  [{inp.element_type or 'input'}] {inp.selector}"
                if inp.attributes.get("placeholder"):
                    input_desc += f" (placeholder: {inp.attributes['placeholder']})"
                if inp.attributes.get("name"):
                    input_desc += f" (name: {inp.attributes['name']})"
                if inp.value:
                    input_desc += f" = '{inp.value[:50]}'"
                parts.append(input_desc)
        
        if self.buttons:
            parts.append("\n--- Buttons ---")
            for btn in self.buttons[:15]:
                btn_text = btn.text or btn.attributes.get("aria-label", "unnamed")
                parts.append(f"  [{btn.selector}] {btn_text}")
        
        if self.links:
            parts.append("\n--- Links ---")
            for link in self.links[:20]:
                link_text = link.text or link.attributes.get("aria-label", "unnamed")
                href = link.attributes.get("href", "")
                if href and not href.startswith("javascript:"):
                    parts.append(f"  [{link.selector}] {link_text[:50]} -> {href[:80]}")
        
        return "\n".join(parts)


async def parse_page_state(
    page: Page,
    include_inputs: bool = True,
    include_buttons: bool = True,
    include_links: bool = True,
) -> PageState:
    """
    Parse the current page state into a structured format.
    
    Args:
        page: Playwright page object
        include_inputs: Whether to include form inputs
        include_buttons: Whether to include buttons
        include_links: Whether to include links
        
    Returns:
        PageState object with parsed elements
    """
    state = PageState(
        url=page.url,
        title=await page.title(),
    )
    
    # Extract headings
    headings = await page.eval_on_selector_all(
        "h1, h2, h3",
        "elements => elements.map(e => e.textContent.trim()).filter(t => t)"
    )
    state.headings = headings[:5]
    
    # Extract inputs
    if include_inputs:
        inputs = await page.eval_on_selector_all(
            "input:visible, textarea:visible, select:visible",
            """elements => elements.map((e, i) => ({
                tag: e.tagName.toLowerCase(),
                type: e.type || null,
                name: e.name || null,
                id: e.id || null,
                placeholder: e.placeholder || null,
                value: e.value || null,
                ariaLabel: e.getAttribute('aria-label') || null,
                className: e.className || null,
            }))"""
        )
        
        for i, inp in enumerate(inputs):
            selector = _build_selector(inp, i, "input")
            state.inputs.append(PageElement(
                tag=inp["tag"],
                selector=selector,
                element_type=inp.get("type"),
                value=inp.get("value"),
                attributes={
                    k: v for k, v in inp.items() 
                    if v and k not in ["tag", "value"]
                },
            ))
    
    # Extract buttons
    if include_buttons:
        buttons = await page.eval_on_selector_all(
            "button:visible, [role='button']:visible, input[type='submit']:visible",
            """elements => elements.map((e, i) => ({
                tag: e.tagName.toLowerCase(),
                type: e.type || null,
                text: e.textContent?.trim() || null,
                id: e.id || null,
                name: e.name || null,
                ariaLabel: e.getAttribute('aria-label') || null,
                className: e.className || null,
            }))"""
        )
        
        for i, btn in enumerate(buttons):
            selector = _build_selector(btn, i, "button")
            state.buttons.append(PageElement(
                tag=btn["tag"],
                selector=selector,
                text=btn.get("text"),
                attributes={
                    k: v for k, v in btn.items() 
                    if v and k not in ["tag", "text"]
                },
            ))
    
    # Extract links
    if include_links:
        links = await page.eval_on_selector_all(
            "a[href]:visible",
            """elements => elements.slice(0, 30).map((e, i) => ({
                tag: 'a',
                text: e.textContent?.trim().substring(0, 100) || null,
                href: e.href || null,
                id: e.id || null,
                ariaLabel: e.getAttribute('aria-label') || null,
            }))"""
        )
        
        for i, link in enumerate(links):
            selector = _build_selector(link, i, "link")
            state.links.append(PageElement(
                tag="a",
                selector=selector,
                text=link.get("text"),
                attributes={
                    k: v for k, v in link.items() 
                    if v and k not in ["tag", "text"]
                },
            ))
    
    return state


def _build_selector(element: dict[str, Any], index: int, element_type: str) -> str:
    """Build the most reliable selector for an element."""
    # Prefer ID
    if element.get("id"):
        return f"#{element['id']}"
    
    # Then name attribute
    if element.get("name"):
        tag = element.get("tag", element_type)
        return f"{tag}[name='{element['name']}']"
    
    # Then aria-label
    if element.get("ariaLabel"):
        return f"[aria-label='{element['ariaLabel']}']"
    
    # Then text content for buttons/links
    if element.get("text") and element_type in ["button", "link"]:
        text = element["text"][:30]
        return f"text={text}"
    
    # Then placeholder for inputs
    if element.get("placeholder"):
        return f"[placeholder='{element['placeholder']}']"
    
    # Fallback to nth-child (less reliable)
    tag = element.get("tag", element_type)
    return f"{tag}:nth-of-type({index + 1})"


async def extract_main_content(page: Page, max_length: int = 2000) -> str:
    """
    Extract the main text content from the page.
    
    Args:
        page: Playwright page object
        max_length: Maximum characters to return
        
    Returns:
        Main text content of the page
    """
    # Try common main content selectors
    main_selectors = [
        "main",
        "article",
        "[role='main']",
        "#content",
        ".content",
        "#main",
        ".main",
    ]
    
    for selector in main_selectors:
        try:
            element = page.locator(selector).first
            if await element.count() > 0:
                text = await element.inner_text()
                if text and len(text.strip()) > 100:
                    return text.strip()[:max_length]
        except Exception:
            continue
    
    # Fallback to body
    try:
        text = await page.locator("body").inner_text()
        return text.strip()[:max_length]
    except Exception:
        return ""


async def check_element_exists(page: Page, selector: str) -> dict[str, Any]:
    """
    Check if an element exists and get information about it.
    
    Args:
        page: Playwright page object
        selector: CSS selector
        
    Returns:
        Dictionary with element info or error
    """
    try:
        locator = page.locator(selector).first
        
        if await locator.count() == 0:
            return {
                "exists": False,
                "error": f"No element found matching '{selector}'",
            }
        
        is_visible = await locator.is_visible()
        is_enabled = await locator.is_enabled()
        
        # Get element details
        tag_name = await locator.evaluate("e => e.tagName.toLowerCase()")
        text = await locator.inner_text() if is_visible else None
        
        return {
            "exists": True,
            "visible": is_visible,
            "enabled": is_enabled,
            "tag": tag_name,
            "text": text[:100] if text else None,
        }
        
    except Exception as e:
        return {
            "exists": False,
            "error": str(e),
        }
