"""
Demo Visuals: ASCII Art, Animations, and Colorized Output
for the Interactive Cybersecurity Demo.
"""

import time
import random
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)

# --- Color Shortcuts ---
OK = Fore.GREEN
WARN = Fore.YELLOW
CRIT = Fore.RED
INFO = Fore.CYAN
BOLD = Style.BRIGHT
RESET = Style.RESET_ALL


def print_banner(title: str):
    """Print a styled header banner."""
    width = 60
    print()
    print(f"{BOLD}{INFO}{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}{RESET}")
    print()


def print_status(level: str, msg: str):
    """Print a status message with colored level indicator."""
    colors = {
        "ok": OK,
        "warn": WARN,
        "critical": CRIT,
        "info": INFO,
    }
    color = colors.get(level.lower(), RESET)
    print(f"  {color}[{level.upper()}]{RESET} {msg}")


def print_ciphertext_wall(data: str, lines: int = 5, delay: float = 0.03):
    """Print a 'Matrix-style' wall of hex ciphertext."""
    hex_chars = "0123456789ABCDEF"
    width = 64
    
    print(f"\n{BOLD}{Fore.GREEN}--- ENCRYPTED PAYLOAD (Server View) ---{RESET}")
    for _ in range(lines):
        line = "".join(random.choice(hex_chars) for _ in range(width))
        # Insert spaces for readability
        formatted = " ".join(line[i:i+4] for i in range(0, len(line), 4))
        print(f"  {Fore.GREEN}{formatted}{RESET}")
        time.sleep(delay)
    print(f"{BOLD}{Fore.GREEN}--- END ENCRYPTED ---{RESET}\n")


def print_dp_blur(epsilon: float):
    """Print an ASCII histogram showing noise distribution at given epsilon."""
    # Higher epsilon = less noise = sharper distribution
    # Lower epsilon = more noise = wider distribution
    print(f"\n{INFO}Privacy Blur Visualization (epsilon = {epsilon}){RESET}")
    
    # Simulate a simple ASCII Gaussian-like distribution
    if epsilon >= 5.0:
        bars = ["       #       ", "      ###      ", "     #####     ", "    #######    ", "   #########   "]
        label = "LOW NOISE (Less Private)"
    elif epsilon >= 1.0:
        bars = ["  #         #  ", " ###       ### ", "#####     #####", "####### #######", "###############"]
        label = "MEDIUM NOISE (Balanced)"
    else:
        bars = ["# # # # # # # #", "# # # # # # # #", "#################", "#################", "#################"]
        label = "HIGH NOISE (Very Private)"
    
    for bar in bars:
        print(f"  {Fore.MAGENTA}{bar}{RESET}")
    print(f"  {WARN}=> {label}{RESET}\n")


def animate_quantum_attack(duration: float = 2.0):
    """Display an animated Quantum Attack wave."""
    print(f"\n{CRIT}{BOLD}[ALERT] Quantum Attack Detected!{RESET}")
    
    wave_chars = "_.-~^~-._"
    width = 40
    frames = int(duration / 0.1)
    
    for i in range(frames):
        offset = i % len(wave_chars)
        wave = "".join(wave_chars[(j + offset) % len(wave_chars)] for j in range(width))
        print(f"\r  {Fore.CYAN}{wave}{RESET}", end="", flush=True)
        time.sleep(0.1)
    
    print()  # Newline after animation
    print_status("info", "Pattern matching Shor's Algorithm signatures...")
    time.sleep(0.5)
    print_status("warn", "Initiating Kyber-768 key rotation...")
    time.sleep(0.5)
    print_status("ok", "Channel re-secured. Quantum breach averted.")
    print()


def animate_mitm_intercept():
    """Display a Man-in-the-Middle interception animation."""
    print(f"\n{WARN}[INTERCEPT] Capturing network traffic...{RESET}")
    
    for i in range(3):
        print(f"\r  {'.' * (i+1)}", end="", flush=True)
        time.sleep(0.3)
    print()
    
    print_status("warn", "Packet captured! Attempting to decode...")
    time.sleep(0.5)


def print_comparison(client_view: str, server_view: str):
    """Print a side-by-side comparison of client vs server view."""
    print(f"\n{BOLD}Client View (Plaintext):{RESET}")
    print(f"  {OK}{client_view}{RESET}")
    print(f"\n{BOLD}Server View (Ciphertext):{RESET}")
    print(f"  {Fore.GREEN}{server_view[:60]}...{RESET}")
    print()
