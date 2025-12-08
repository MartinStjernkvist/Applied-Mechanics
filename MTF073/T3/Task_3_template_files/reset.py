# --- universal reset that works in Spyder, VS Code, Jupyter, and plain Python ---
import inspect
import os
import sys

def universal_reset(protect=None, restart=False, verbose=False):
    """
    Reset the current working environment in a cross-tool way.

    Behavior:
      - If running under IPython (Spyder, VS Code Interactive/Notebook, Jupyter):
            runs the IPython magic: %reset -sf
      - Else (plain Python script):
            deletes most globals in the *caller* module to simulate a reset
      - Optionally, restart the full Python process (cleanest) when restart=True

    Parameters
    ----------
    protect : set[str] | None
        Additional global names to protect from deletion (in non-IPython mode).
        Example: {"universal_reset"} to keep the function available.
    restart : bool
        If True and not in IPython, performs a full process restart via os.execv.
        Ignored when in IPython (use kernel restart via UI if needed).
    verbose : bool
        If True, prints diagnostic messages about the chosen reset method.
    """
    # Names that must remain
    protected = {
        '__name__', '__file__', '__package__', '__spec__',
        '__annotations__', '__builtins__'
    }
    if protect:
        protected |= set(protect)

    # Detect IPython kernel (Spyder, VS Code Interactive/Notebook, Jupyter)
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        ip = None

    if ip is not None:
        if verbose:
            print("[universal_reset] IPython detected -> running %reset -sf")
        ip.run_line_magic('reset', '-sf')
        return

    # Plain Python path
    if restart:
        if verbose:
            print("[universal_reset] Plain Python detected -> full process restart")
        os.execv(sys.executable, [sys.executable] + sys.argv)
        return

    # Clear the caller's globals (simulate %reset for scripts)
    if verbose:
        print("[universal_reset] Plain Python detected -> clearing caller globals")

    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    caller_globals = caller.f_globals if caller is not None else None

    if not caller_globals:
        if verbose:
            print("[universal_reset] No caller globals found; nothing to clear.")
        return

    to_delete = [
        name for name in list(caller_globals)
        if not name.startswith('_') and name not in protected
    ]
    for name in to_delete:
        try:
            del caller_globals[name]
        except Exception:
            pass

    if verbose:
        print(f"[universal_reset] Deleted {len(to_delete)} names from caller globals.")
