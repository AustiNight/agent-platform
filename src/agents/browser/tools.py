"""
Browser automation tools.

Defines the tool vocabulary available to the browser agent.
"""

# Tool definitions in the format expected by the LLM
BROWSER_TOOLS = [
    {
        "name": "navigate",
        "description": "Navigate to a URL. Use this to go to a new page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to navigate to (e.g., 'https://example.com')",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "click",
        "description": "Click on an element. Use CSS selectors, text content, or accessibility labels to identify elements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector, text content (e.g., 'text=Submit'), or test ID (e.g., '[data-testid=login-btn]')",
                },
                "wait_for_navigation": {
                    "type": "boolean",
                    "description": "Whether to wait for page navigation after clicking. Default: false",
                    "default": False,
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input field. First focuses the element, then types.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the input field",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the field",
                },
                "clear_first": {
                    "type": "boolean",
                    "description": "Whether to clear existing content before typing. Default: true",
                    "default": True,
                },
                "press_enter": {
                    "type": "boolean",
                    "description": "Whether to press Enter after typing. Default: false",
                    "default": False,
                },
            },
            "required": ["selector", "text"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option from a dropdown/select element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the select element",
                },
                "value": {
                    "type": "string",
                    "description": "Option value, label, or text to select",
                },
            },
            "required": ["selector", "value"],
        },
    },
    {
        "name": "wait_for",
        "description": "Wait for an element to appear or for a specified time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector to wait for (optional if using seconds)",
                },
                "seconds": {
                    "type": "number",
                    "description": "Number of seconds to wait (optional if using selector)",
                },
                "state": {
                    "type": "string",
                    "enum": ["visible", "hidden", "attached", "detached"],
                    "description": "Element state to wait for. Default: visible",
                    "default": "visible",
                },
            },
        },
    },
    {
        "name": "screenshot",
        "description": "Take a screenshot of the current page or a specific element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to screenshot a specific element",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Whether to capture the full scrollable page. Default: false",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "extract_text",
        "description": "Extract text content from the page or a specific element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector. If omitted, extracts main content.",
                },
            },
        },
    },
    {
        "name": "get_page_state",
        "description": "Get the current state of the page including URL, title, and interactive elements. Use this to understand what's on the page before taking actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_inputs": {
                    "type": "boolean",
                    "description": "Include form inputs and their current values. Default: true",
                    "default": True,
                },
                "include_links": {
                    "type": "boolean",
                    "description": "Include links on the page. Default: true",
                    "default": True,
                },
                "include_buttons": {
                    "type": "boolean",
                    "description": "Include buttons on the page. Default: true",
                    "default": True,
                },
            },
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page or an element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction to scroll",
                },
                "amount": {
                    "type": "string",
                    "enum": ["small", "medium", "large", "page"],
                    "description": "How much to scroll. Default: medium",
                    "default": "medium",
                },
                "selector": {
                    "type": "string",
                    "description": "Optional: scroll within a specific element",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "go_back",
        "description": "Navigate back in browser history.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "go_forward",
        "description": "Navigate forward in browser history.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "check_element",
        "description": "Check if an element exists and get information about it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element to check",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "fill_form",
        "description": "Fill multiple form fields at once. More efficient than individual type_text calls.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "description": "Array of field definitions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "value": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["text", "select", "checkbox", "radio"],
                                "default": "text",
                            },
                        },
                        "required": ["selector", "value"],
                    },
                },
            },
            "required": ["fields"],
        },
    },
    {
        "name": "complete_task",
        "description": "Mark the task as complete and provide a summary. Use this when you've finished the requested task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the task was completed successfully",
                },
                "summary": {
                    "type": "string",
                    "description": "A summary of what was accomplished",
                },
                "extracted_data": {
                    "type": "object",
                    "description": "Any data extracted during the task",
                },
            },
            "required": ["success", "summary"],
        },
    },
    {
        "name": "get_credential",
        "description": "Retrieve a stored credential (username, password, API key, etc.) from the secrets vault. Use this when you need login credentials for a website. Credentials are stored by site domain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "description": "The site domain (e.g., 'github.com', 'example.com')",
                },
                "field": {
                    "type": "string",
                    "description": "The credential field to retrieve",
                    "enum": ["username", "password", "email", "api_key", "token"],
                },
            },
            "required": ["site", "field"],
        },
    },
]


def get_tool_names() -> list[str]:
    """Get list of all tool names."""
    return [tool["name"] for tool in BROWSER_TOOLS]


def get_tool_by_name(name: str) -> dict | None:
    """Get tool definition by name."""
    for tool in BROWSER_TOOLS:
        if tool["name"] == name:
            return tool
    return None
