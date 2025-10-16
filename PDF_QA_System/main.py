import sys
from ui_web import run_web_mode
from ui_cli import run_cli_mode


# ---------------------------
# Main Execution (Mode Switch)
# ---------------------------
if __name__ == "__main__":
    # Check command-line argument to determine mode
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['cli', 'commandline']:
        run_cli_mode()
    else:
        # Default to web mode if no argument or invalid argument
        run_web_mode()
