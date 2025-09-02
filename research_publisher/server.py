import sys
import subprocess
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Try to import optional runner from research_publisher. If not present, we'll fallback.
try:
    from research_publisher import run_research_publisher as _run_research_publisher
    _HAS_RUNNER = True
    _IMPORT_ERR = None
except Exception as _e:  # noqa: BLE001 - show any import error text in tool response
    _run_research_publisher = None
    _HAS_RUNNER = False
    _IMPORT_ERR = str(_e)

# Ensure UTF-8 stdio for non-ASCII outputs
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Create FastMCP instance
mcp = FastMCP("research_publisher")


@mcp.tool()
def research_publisher(query: str) -> str:
    """
    MCP Tool: research_publisher

    Takes an input query string from the user and returns the response
    from the multi-agent pipeline using run_research_publisher(). If the
    function is not available (because the script currently only contains
    raw notebook code), we fallback to executing the script directly and
    returning its stdout.
    """
    if _HAS_RUNNER and _run_research_publisher is not None:
        try:
            return _run_research_publisher(query)
        except Exception as exc:  # noqa: BLE001
            return f"Error while running research_publisher: {exc}"

    # Fallback: run the notebook script as a subprocess and return stdout
    try:
        script_path = Path(__file__).with_name("research_publisher.py")
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            return (
                "research_publisher.py exited with code "
                f"{completed.returncode}.\n"
                f"stderr:\n{completed.stderr}\n"
                f"import_error: {_IMPORT_ERR}"
            )
        return completed.stdout
    except Exception as exc:  # noqa: BLE001
        return f"Error executing research_publisher.py: {exc}\nimport_error: {_IMPORT_ERR}"


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
