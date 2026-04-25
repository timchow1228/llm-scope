"""
llm-scope CLI — Phase 6: click + rich command-line interface.

Commands:
  llm-scope start   [--port] [--host] [--no-browser] [--strict-port]
  llm-scope status
  llm-scope config
  llm-scope clear   [--payloads]
"""

import os
import socket
import sys
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from llm_scope import __version__

console = Console()


# ---------------------------------------------------------------------------
# Port helpers
# ---------------------------------------------------------------------------

def _is_port_in_use(port: int) -> bool:
    """Return True if the given TCP port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _find_available_port(start: int = 7070, max_attempts: int = 10) -> int:
    """Return the first free port in [start, start+max_attempts)."""
    for port in range(start, start + max_attempts):
        if not _is_port_in_use(port):
            return port
    raise RuntimeError(f"No available ports in range {start}–{start + max_attempts - 1}")


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner(actual_port: int) -> None:
    """Print the startup banner using rich."""
    url_v1 = f"http://localhost:{actual_port}/v1"
    url_dashboard = f"http://localhost:{actual_port}"

    lines = [
        Text.assemble(("🔭 llm-scope ", "bold cyan"), (f"v{__version__} is running", "green bold")),
        Text(""),
        Text.assemble(("  Proxy:     ", "dim"), (url_v1, "cyan underline")),
        Text.assemble(("  Dashboard: ", "dim"), (url_dashboard, "cyan underline")),
        Text.assemble(("  Routing:   ", "dim"), ("auto (model name → provider)", "white")),
        Text(""),
        Text.assemble(("  Quick config:", "dim")),
        Text.assemble(("  export OPENAI_BASE_URL=", "yellow"), (url_v1, "yellow bold")),
        Text(""),
        Text.assemble(
            ("  🎯 Cursor users: ", "white"),
            ('set "OpenAI Base URL" in Cursor settings', "dim"),
        ),
        Text.assemble(("     to ", "dim"), (url_v1, "cyan")),
        Text(""),
        Text.assemble(
            ("  ℹ️  Anthropic native SDK not yet supported.\n", "dim"),
            ("     Bridge: pip install litellm && litellm --model\n", "dim"),
            ("     claude-sonnet-4-20250514 --port 4000\n", "dim"),
            ("     Then point DevScope at http://localhost:4000", "dim"),
        ),
        Text(""),
        Text.assemble(("  🐳 Docker/WSL? Use ", "dim"), (f"http://<host-ip>:{actual_port}/v1", "cyan")),
        Text(""),
        Text.assemble(("  Press ", "dim"), ("Ctrl+C", "bold red"), (" to stop", "dim")),
    ]

    content = Text("\n").join(lines)

    panel = Panel(
        content,
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
def main():
    """🔭 llm-scope — Local LLM proxy with waterfall breakdown."""
    pass


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

@main.command()
@click.option("--port", default=7070, show_default=True,
              help="Proxy listen port (auto-advances if busy).")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind host. Use 0.0.0.0 for LAN / WSL access.")
@click.option("--no-browser", is_flag=True,
              help="Don't open the dashboard in a browser automatically.")
@click.option("--strict-port", is_flag=True,
              help="Exit with error if requested port is busy (no auto-advance).")
def start(port: int, host: str, no_browser: bool, strict_port: bool):
    """Start the llm-scope proxy and (optionally) open the dashboard."""
    # ── Resolve actual port ──
    if _is_port_in_use(port):
        if strict_port:
            console.print(
                f"\n[bold red][FATAL][/bold red] Port {port} is in use! "
                f"Stop the existing process or run:\n"
                f"       devscope start --port {port + 1}\n"
            )
            sys.exit(1)
        else:
            try:
                actual_port = _find_available_port(port + 1)
            except RuntimeError as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                sys.exit(1)
            console.print(
                f"\n[bold yellow]⚠️  Port {port} is in use.[/bold yellow] "
                f"llm-scope starting on port [bold cyan]{actual_port}[/bold cyan].\n"
                f"   Update your base_url to: [cyan]http://localhost:{actual_port}/v1[/cyan]\n"
            )
    else:
        actual_port = port

    # ── Banner ──
    _print_banner(actual_port)

    # ── Open browser ──
    if not no_browser and not os.environ.get("DEVSCOPE_NO_BROWSER"):
        dashboard_url = f"http://localhost:{actual_port}/dashboard"
        try:
            webbrowser.open(dashboard_url)
        except Exception:
            pass  # Non-fatal — browser open failure must never crash the proxy

    # ── Launch uvicorn ──
    import uvicorn
    uvicorn.run(
        "llm_scope.proxy:app",
        host=host,
        port=actual_port,
        workers=1,          # Single-process: keeps in-memory state consistent
        log_level="warning",  # Suppress uvicorn's verbose access logs (we have our own)
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command()
@click.option("--port", default=7070, show_default=True,
              help="Port to probe.")
def status(port: int):
    """Check whether llm-scope is currently running."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                console.print(
                    f"[bold green]✓[/bold green] llm-scope is running on port [cyan]{port}[/cyan]  "
                    f"→ Dashboard: [cyan underline]http://localhost:{port}/dashboard[/cyan underline]"
                )
            else:
                console.print(
                    f"[yellow]⚠[/yellow]  Port {port} responded with HTTP {resp.status}"
                )
    except (urllib.error.URLError, OSError):
        if _is_port_in_use(port):
            console.print(
                f"[yellow]⚠[/yellow]  Port {port} is in use, but /health did not respond. "
                f"Another process may be occupying it."
            )
        else:
            console.print(
                f"[red]✗[/red]  llm-scope is [bold]not running[/bold] on port {port}."
            )


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@main.command("config")
def show_config():
    """Display the current llm-scope configuration."""
    from llm_scope.storage import _get_db_path  # type: ignore[attr-defined]

    db_path = _get_db_path()

    rows = [
        ("Port",         os.environ.get("DEVSCOPE_PORT", "7070 (default)")),
        ("Host",         os.environ.get("DEVSCOPE_HOST", "127.0.0.1 (default)")),
        ("Provider",     os.environ.get("DEVSCOPE_PROVIDER", "deepseek (default)")),
        ("Base URL",     os.environ.get("DEVSCOPE_BASE_URL", "(auto from model name)")),
        ("DB Path",      str(db_path)),
        ("Body Limit",   os.environ.get("DEVSCOPE_BODY_LIMIT", "4000 (default)")),
        ("Ghost Mode",   "ON" if os.environ.get("DEVSCOPE_GHOST_MODE") else "OFF"),
        ("Strict Port",  "ON" if os.environ.get("DEVSCOPE_STRICT_PORT") else "OFF"),
        ("Daily Limit",  os.environ.get("DEVSCOPE_DAILY_LIMIT", "(no limit)")),
    ]

    console.print(f"\n[bold cyan]🔭 llm-scope v{__version__} — current config[/bold cyan]\n")
    for key, val in rows:
        console.print(f"  [dim]{key:<14}[/dim]  {val}")
    console.print()


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

@main.command()
@click.option("--payloads", is_flag=True,
              help="Only delete payload JSON files; keep the SQLite call records.")
@click.confirmation_option(prompt="This will delete local data. Are you sure?")
def clear(payloads: bool):
    """Clear local call history (and optionally payload files)."""
    import asyncio
    import aiosqlite

    db_path = Path.home() / ".local" / "share" / "llm-scope" / "calls.db"
    custom = os.environ.get("DEVSCOPE_DB_PATH")
    if custom:
        db_path = Path(custom)

    payloads_dir = db_path.parent / "payloads"

    if payloads:
        # ── Only delete payload JSON files ──
        if payloads_dir.exists():
            files = list(payloads_dir.glob("*.json"))
            for f in files:
                try:
                    f.unlink()
                except OSError:
                    pass
            console.print(
                f"[green]✓[/green] Deleted {len(files)} payload file(s) from {payloads_dir}"
            )
        else:
            console.print("[dim]No payload directory found — nothing to delete.[/dim]")
    else:
        # ── Delete everything: DB rows + payload files ──
        deleted_rows = 0

        async def _clear_db():
            nonlocal deleted_rows
            if not db_path.exists():
                return
            async with aiosqlite.connect(str(db_path)) as db:
                cur = await db.execute("SELECT COUNT(*) FROM calls")
                row = await cur.fetchone()
                deleted_rows = row[0] if row else 0
                await db.execute("DELETE FROM calls")
                await db.commit()

        asyncio.run(_clear_db())

        # Delete payload files too
        payload_count = 0
        if payloads_dir.exists():
            files = list(payloads_dir.glob("*.json"))
            payload_count = len(files)
            for f in files:
                try:
                    f.unlink()
                except OSError:
                    pass

        console.print(
            f"[green]✓[/green] Cleared {deleted_rows} call record(s) from SQLite"
            + (f" and {payload_count} payload file(s)." if payload_count else ".")
        )
