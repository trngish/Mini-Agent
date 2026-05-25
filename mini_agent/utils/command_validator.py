"""Command injection prevention utilities.

Provides input validation and sanitization for shell command execution.
"""

import re
from enum import Enum
from typing import Optional


class DangerLevel(Enum):
    """Danger level for command patterns."""
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


# Patterns that should be blocked entirely
BLOCKED_PATTERNS = [
    re.compile(r'\brm\s+(-rf?|--force)\s+(/\s*$|/\*|~)', re.IGNORECASE),  # rm -rf / or rm -rf ~
    re.compile(r'\bdel\s+/[fqs]', re.IGNORECASE),  # Windows del /f
    re.compile(r'\bformat\s+', re.IGNORECASE),  # format command
    re.compile(r'\bmkfs\.', re.IGNORECASE),  # filesystem formatting
    re.compile(r'\bdd\s+if=/dev/zero', re.IGNORECASE),  # disk zeroing
    re.compile(r'>\s*/dev/sd', re.IGNORECASE),  # disk device writing
    re.compile(r'\bshutdown\s+(-h|-r)', re.IGNORECASE),  # shutdown/reboot
    re.compile(r'\bsystemctl\s+(reboot|poweroff|halt)', re.IGNORECASE),  # system control
    re.compile(r':\(\)\{\s*:\|:\s*&\s*\}\s*;', re.IGNORECASE),  # fork bomb
]

# Patterns that require caution
CAUTION_PATTERNS = [
    re.compile(r'\brm\s+(-r|-R)'),  # recursive deletion
    re.compile(r'\bsudo\s+', re.IGNORECASE),  # sudo usage
    re.compile(r'\bchmod\s+[0-7]{3,4}\s+', re.IGNORECASE),  # permission changes
    re.compile(r'\bchown\s+', re.IGNORECASE),  # ownership changes
    re.compile(r'\bnetsh\s+', re.IGNORECASE),  # Windows network config
    re.compile(r'\breg\s+(add|delete|set)', re.IGNORECASE),  # Windows registry
    re.compile(r'\bcurl\s+.*\|\s*(bash|sh)', re.IGNORECASE),  # pipe curl to shell
    re.compile(r'\bwget\s+.*\|\s*(bash|sh)', re.IGNORECASE),  # pipe wget to shell
]


# Platform-specific command patterns
# These commands exist on one platform but NOT on the other (without WSL/compat layer)
LINUX_SPECIFIC_PATTERNS = [
    re.compile(r'\bgrep\s+'),           # no grep on Windows (no PS alias)
    re.compile(r'\bchmod\s+'),          # no chmod on Windows
    re.compile(r'\bchown\s+'),          # no chown on Windows
    re.compile(r'\bsed\s+'),            # no sed on Windows
    re.compile(r'\bawk\s+'),            # no awk on Windows
    re.compile(r'\bwhich\s+'),          # no which on Windows (use where)
    re.compile(r'\bexport\s+'),         # no export in PowerShell
    re.compile(r'\bsource\s+'),         # no source in PowerShell
    re.compile(r'\bwc\s+'),             # no wc on Windows
    re.compile(r'\bdiff\s+'),           # no diff on Windows
    re.compile(r'\blocate\b'),          # no locate on Windows
    re.compile(r'\bnohup\b'),           # no nohup on Windows
    re.compile(r'\buname\b'),           # no uname on Windows
    re.compile(r'\bwhoami\b'),          # exists on both, but POSIX version
    re.compile(r'\bkill\s+-\d+'),       # kill -9 style (PowerShell uses Stop-Process)
    re.compile(r'\bps\s+(aux|ef)\b'),   # ps aux/ef format (PowerShell uses Get-Process)
    re.compile(r'\b(?:apt|apt-get|yum|dnf|pacman|emerge|zypper)\b'),  # Linux pkg managers
    re.compile(r'\bbrew\b'),            # macOS Homebrew (not on Windows)
    re.compile(r'^\./'),                # Unix ./script execution
    re.compile(r'\b\./configure\b'),    # Unix build step
]

WINDOWS_SPECIFIC_PATTERNS = [
    re.compile(r'\bipconfig\b'),        # no ipconfig on Linux (use ip addr)
    re.compile(r'\btasklist\b'),        # no tasklist on Linux (use ps)
    re.compile(r'\btaskkill\b'),        # no taskkill on Linux (use kill)
    re.compile(r'\bfindstr\b'),         # no findstr on Linux (use grep)
    re.compile(r'\bxcopy\b'),           # no xcopy on Linux
    re.compile(r'\brobocopy\b'),        # no robocopy on Linux
    re.compile(r'\battrib\b'),          # no attrib on Linux
    re.compile(r'\bchkdsk\b'),          # no chkdsk on Linux
    re.compile(r'\bdriverquery\b'),     # no driverquery on Linux
    re.compile(r'\bmstsc\b'),           # no mstsc on Linux
    re.compile(r'\bcls\b'),             # no cls on Linux (use clear)
    re.compile(r'\breg\s+(query|add|delete|copy|save)\b'),  # Windows registry
    re.compile(r'\bnet\s+(use|view|user|share|localgroup|statistics)\b'),  # Windows net
    re.compile(r'\bsc\s+(query|config|start|stop|create|delete)\b'),  # Windows service
    re.compile(r'\bwmic\b'),            # no wmic on Linux
    re.compile(r'\bdism\b'),            # no dism on Linux
    re.compile(r'\bsfc\b'),             # no sfc on Linux
    re.compile(r'[A-Za-z]:\\'),         # C:\ style Windows paths
    re.compile(r'\bwhere\s+\S+'),       # where (file locate, not SQL)
]


def detect_platform_mismatch(command: str, is_windows: bool) -> str | None:
    """Detect if a command is intended for the wrong platform.

    Args:
        command: The shell command to check
        is_windows: True if running on Windows, False if running on Linux/macOS

    Returns:
        A warning string if mismatch detected, None if compatible
    """
    if is_windows:
        for pattern in LINUX_SPECIFIC_PATTERNS:
            if pattern.search(command):
                return (
                    "⚠️  Detected Linux/macOS command syntax. "
                    "Current shell is PowerShell on Windows. "
                    "This command may fail without WSL/Git-Bash compatibility layer."
                )
    else:
        for pattern in WINDOWS_SPECIFIC_PATTERNS:
            if pattern.search(command):
                return (
                    "⚠️  Detected Windows command syntax. "
                    "Current shell is bash on Linux/macOS. "
                    "This command may fail."
                )
    return None


def assess_command_danger(command: str) -> tuple[DangerLevel, Optional[str]]:
    """Assess the danger level of a command.
    
    Args:
        command: The command to assess
        
    Returns:
        Tuple of (danger_level, reason_if_blocked)
    """
    # Check blocked patterns first
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return DangerLevel.BLOCKED, "Command contains dangerous pattern that could harm the system"
    
    # Check caution patterns
    for pattern in CAUTION_PATTERNS:
        if pattern.search(command):
            return DangerLevel.CAUTION, "Command contains potentially risky operations"
    
    # Check for common safe patterns
    safe_prefixes = [
        'ls', 'dir', 'cat', 'type', 'echo', 'pwd', 'cd',
        'git', 'npm', 'pip', 'python', 'node', 'cargo',
        'grep', 'find', 'wc', 'head', 'tail', 'less', 'more',
        'mkdir', 'touch', 'cp', 'copy', 'mv', 'move',
        'pytest', 'make', 'cmake', 'docker', 'kubectl',
    ]
    
    cmd_lower = command.lower().strip()
    for prefix in safe_prefixes:
        if cmd_lower.startswith(prefix):
            return DangerLevel.SAFE, None
    
    # Default to caution for unknown commands
    return DangerLevel.CAUTION, "Unknown command - review recommended"


def is_command_safe(command: str) -> bool:
    """Quick check if command is safe to execute.
    
    Args:
        command: The command to check
        
    Returns:
        True if command is safe, False otherwise
    """
    level, _ = assess_command_danger(command)
    return level in (DangerLevel.SAFE, DangerLevel.CAUTION)


def sanitize_file_path(path: str) -> str:
    """Sanitize a file path to prevent directory traversal.
    
    Args:
        path: Raw file path
        
    Returns:
        Sanitized file path
    """
    # Remove null bytes
    path = path.replace('\x00', '')
    
    # Remove or escape shell metacharacters
    dangerous_chars = [';', '|', '&', '$', '`', '(', ')', '{', '}', '[', ']', '<', '>', '!', '\\']
    for char in dangerous_chars:
        path = path.replace(char, '')
    
    # Normalize path separators
    path = path.replace('\\', '/')
    
    # Remove directory traversal attempts
    while '../' in path or '..\\' in path:
        path = path.replace('../', '').replace('..\\', '')
    
    return path.strip()




def validate_command_safety(command: str, is_windows: bool = True) -> tuple[bool, str, DangerLevel]:
    """Comprehensive command safety validation.
    
    Combines danger assessment, platform check, and input validation.
    
    Args:
        command: The command to validate
        is_windows: True if running on Windows
        
    Returns:
        Tuple of (is_safe, warning_message, danger_level)
    """
    # Step 1: Basic input validation
    if not command or not command.strip():
        return False, "Empty command", DangerLevel.BLOCKED
    
    if len(command) > 10000:
        return False, f"Command too long ({len(command)} > 10000 chars)", DangerLevel.BLOCKED
    
    # Step 2: Danger level check
    danger_level, reason = assess_command_danger(command)
    if danger_level == DangerLevel.BLOCKED:
        return False, reason or "Blocked command", danger_level
    
    # Step 3: Platform mismatch check
    platform_warning = detect_platform_mismatch(command, is_windows)
    
    # Step 4: Check for suspicious patterns in arguments
    suspicious_args = check_suspicious_arguments(command)
    
    if suspicious_args:
        warning = f"Suspicious arguments detected: {suspicious_args}"
        if platform_warning:
            warning = f"{platform_warning}\n{warning}"
        return False, warning, DangerLevel.CAUTION
    
    if platform_warning:
        return True, platform_warning, danger_level
    
    return True, "", danger_level


def check_suspicious_arguments(command: str) -> list[str]:
    """Check for suspicious patterns in command arguments.
    
    Args:
        command: The command to check
        
    Returns:
        List of suspicious patterns found
    """
    suspicious = []
    lower_cmd = command.lower()
    
    # Check for potential command injection via semicolons
    if ';rm' in lower_cmd or '; del' in lower_cmd:
        suspicious.append("Potential command injection (semicolon)")
    
    # Check for pipe to shell (common attack pattern)
    if '| sh' in lower_cmd or '| bash' in lower_cmd or '&& sh' in lower_cmd:
        suspicious.append("Pipe to shell detected")
    
    # Check for download and execute patterns
    if ('curl' in lower_cmd or 'wget' in lower_cmd) and ('| sh' in lower_cmd or '| python' in lower_cmd):
        suspicious.append("Download and execute pattern")
    
    # Check for environment variable manipulation
    if 'export ' in lower_cmd and '=' in command:
        suspicious.append("Environment variable assignment")
    
    return suspicious


def sanitize_command_output(output: str, max_length: int = 50000) -> str:
    """Sanitize command output to prevent terminal escape codes.
    
    Args:
        output: Raw command output
        max_length: Maximum output length
        
    Returns:
        Sanitized output
    """
    if not output:
        return ""
    
    # Remove ANSI escape codes
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    output = ansi_pattern.sub('', output)
    
    # Truncate if too long
    if len(output) > max_length:
        output = output[:max_length] + f"... [truncated {len(output) - max_length} chars]"
    
    return output


def validate_command_input(command: str, max_length: int = 10000) -> tuple[bool, str]:
    """Validate command input for safety.
    
    Args:
        command: The command to validate
        max_length: Maximum allowed command length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check length
    if len(command) > max_length:
        return False, f"Command too long ({len(command)} > {max_length} chars)"
    
    # Check for empty command
    if not command.strip():
        return False, "Command is empty"
    
    # Check danger level
    level, reason = assess_command_danger(command)
    if level == DangerLevel.BLOCKED:
        return False, reason or "Command blocked for safety"
    
    return True, ""
