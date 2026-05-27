"""Command injection prevention utilities.

Provides input validation and sanitization for shell command execution.
"""

import re
from enum import Enum
from pathlib import Path


class DangerLevel(Enum):
    """Danger level for command patterns."""

    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


# Patterns that should be blocked entirely
BLOCKED_PATTERNS = [
    re.compile(r"\brm\s+(-rf?|--force)\s+(/\s*$|/\*|~)", re.IGNORECASE),  # rm -rf / or rm -rf ~
    re.compile(r"\bdel\s+/[fqs]", re.IGNORECASE),  # Windows del /f
    re.compile(r"\bformat\s+", re.IGNORECASE),  # format command
    re.compile(r"\bmkfs\.", re.IGNORECASE),  # filesystem formatting
    re.compile(r"\bdd\s+if=/dev/zero", re.IGNORECASE),  # disk zeroing
    re.compile(r">\s*/dev/sd", re.IGNORECASE),  # disk device writing
    re.compile(r"\bshutdown\s+(-h|-r)", re.IGNORECASE),  # shutdown/reboot
    re.compile(r"\bsystemctl\s+(reboot|poweroff|halt)", re.IGNORECASE),  # system control
    re.compile(r":\(\)\{\s*:\|:\s*&\s*\}\s*;", re.IGNORECASE),  # fork bomb
]

# Patterns that require caution
CAUTION_PATTERNS = [
    re.compile(r"\brm\s+(-r|-R)"),  # recursive deletion
    re.compile(r"\bsudo\s+", re.IGNORECASE),  # sudo usage
    re.compile(r"\bchmod\s+[0-7]{3,4}\s+", re.IGNORECASE),  # permission changes
    re.compile(r"\bchown\s+", re.IGNORECASE),  # ownership changes
    re.compile(r"\bnetsh\s+", re.IGNORECASE),  # Windows network config
    re.compile(r"\breg\s+(add|delete|set)", re.IGNORECASE),  # Windows registry
    re.compile(r"\bcurl\s+.*\|\s*(bash|sh)", re.IGNORECASE),  # pipe curl to shell
    re.compile(r"\bwget\s+.*\|\s*(bash|sh)", re.IGNORECASE),  # pipe wget to shell
]


# Platform-specific command patterns
# These commands exist on one platform but NOT on the other (without WSL/compat layer)
LINUX_SPECIFIC_PATTERNS = [
    re.compile(r"\bgrep\s+"),
    re.compile(r"\bchmod\s+"),
    re.compile(r"\bchown\s+"),
    re.compile(r"\bsed\s+"),
    re.compile(r"\bawk\s+"),
    re.compile(r"\bwhich\s+"),
    re.compile(r"\bexport\s+"),
    re.compile(r"\bsource\s+"),
    re.compile(r"\bwc\s+"),
    re.compile(r"\bdiff\s+"),
    re.compile(r"\blocate\b"),
    re.compile(r"\bnohup\b"),
    re.compile(r"\buname\b"),
    re.compile(r"\bwhoami\b"),
    re.compile(r"\bkill\s+-\d+"),
    re.compile(r"\bps\s+(aux|ef)\b"),
    re.compile(r"\b(?:apt|apt-get|yum|dnf|pacman|emerge|zypper)\b"),
    re.compile(r"\bbrew\b"),
    re.compile(r"^\./"),
    re.compile(r"\b\./configure\b"),
    re.compile(r"\bhead\s+"),
    re.compile(r"\btail\s+"),
    re.compile(r"\btouch\s+"),
    re.compile(r"\bcat\s+"),
    re.compile(r"\bls\s+-"),
    re.compile(r"\bfind\s+"),
    re.compile(r"\bxargs\s+"),
    re.compile(r"\btee\s+"),
    re.compile(r"\bsort\s+"),
    re.compile(r"\buniq\s+"),
    re.compile(r"\bcut\s+"),
    re.compile(r"\btr\s+"),
    re.compile(r"\bpaste\s+"),
    re.compile(r"\bcolumn\s+"),
    re.compile(r"\bshuf\b"),
    re.compile(r"\bfold\b"),
    re.compile(r"\bfmt\b"),
    re.compile(r"\brev\b"),
    re.compile(r"\bjoin\b"),
    re.compile(r"\bsplit\b"),
    re.compile(r"\bdate\s+\+"),
    re.compile(r"\bsleep\s+"),
    re.compile(r"\bwatch\s+"),
    re.compile(r"\btime\s+"),
    re.compile(r"\btimeout\s+"),
    re.compile(r"\benv\s+"),
    re.compile(r"\bprintenv\b"),
    re.compile(r"\bset\s+-\w"),
    re.compile(r"\bunset\s+"),
    re.compile(r"\bshift\b"),
    re.compile(r"\btest\s+"),
    re.compile(r"\bexpr\s+"),
    re.compile(r"\blet\s+"),
    re.compile(r"\bseq\s+"),
    re.compile(r"\bread\s+"),
    re.compile(r"\btrap\s+"),
    re.compile(r"\bwait\b"),
    re.compile(r"\bexec\s+"),
    re.compile(r"\beval\s+"),
    re.compile(r"\breadonly\s+"),
    re.compile(r"\bdeclare\s+"),
    re.compile(r"\btypeset\s+"),
    re.compile(r"\blocal\s+"),
    re.compile(r"\breturn\s+"),
    re.compile(r"\bbreak\b"),
    re.compile(r"\bcontinue\b"),
    re.compile(r"2>/dev/null"),
    re.compile(r"2>&1"),
    re.compile(r"&\s*$"),
    re.compile(r"\bdhclient\b"),
    re.compile(r"\bifconfig\b"),
    re.compile(r"\bip\s+(addr|link|route)\b"),
    re.compile(r"\bnetstat\b"),
    re.compile(r"\bss\s+"),
    re.compile(r"\bping\s+-[cn]\s"),
    re.compile(r"\btraceroute\b"),
    re.compile(r"\bdig\s+"),
    re.compile(r"\bnslookup\b"),
    re.compile(r"\barp\b"),
    re.compile(r"\bmount\s+"),
    re.compile(r"\bumount\b"),
    re.compile(r"\bdf\s+-"),
    re.compile(r"\bdu\s+-"),
    re.compile(r"\bln\s+-s"),
    re.compile(r"\breadlink\b"),
    re.compile(r"\bstat\s+"),
    re.compile(r"\bfile\s+"),
    re.compile(r"\bmd5sum\b"),
    re.compile(r"\bsha256sum\b"),
    re.compile(r"\bbase64\s+"),
    re.compile(r"\btar\s+"),
    re.compile(r"\bgzip\b"),
    re.compile(r"\bbzip2\b"),
    re.compile(r"\bxz\b"),
    re.compile(r"\bzip\b"),
    re.compile(r"\bunzip\b"),
    re.compile(r"\bcron\b"),
    re.compile(r"\bcrontab\b"),
    re.compile(r"\bat\b\s+"),
    re.compile(r"\bsystemctl\s+"),
    re.compile(r"\bservice\s+"),
    re.compile(r"\bjournalctl\b"),
    re.compile(r"\blogrotate\b"),
]

WINDOWS_SPECIFIC_PATTERNS = [
    re.compile(r"\bipconfig\b"),
    re.compile(r"\btasklist\b"),
    re.compile(r"\btaskkill\b"),
    re.compile(r"\bfindstr\b"),
    re.compile(r"\bxcopy\b"),
    re.compile(r"\brobocopy\b"),
    re.compile(r"\battrib\b"),
    re.compile(r"\bchkdsk\b"),
    re.compile(r"\bdriverquery\b"),
    re.compile(r"\bmstsc\b"),
    re.compile(r"\bcls\b"),
    re.compile(r"\breg\s+(query|add|delete|copy|save)\b"),
    re.compile(r"\bnet\s+(use|view|user|share|localgroup|statistics)\b"),
    re.compile(r"\bsc\s+(query|config|start|stop|create|delete)\b"),
    re.compile(r"\bwmic\b"),
    re.compile(r"\bdism\b"),
    re.compile(r"\bsfc\b"),
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"\bwhere\s+\S+"),
    re.compile(r"\bdir\s+/[sb]\b"),
    re.compile(r"\btype\s+\S+\s*>\s*"),
    re.compile(r"\bcopy\s+/[by]\b"),
    re.compile(r"\bdel\s+/[fqsv]\b"),
    re.compile(r"\bmove\s+/[y]\b"),
    re.compile(r"\brmdir\s+/[sq]\b"),
    re.compile(r"\bmkdir\s+\S+\\"),
    re.compile(r"\b2>nul\b"),
    re.compile(r"\b1>con\b"),
    re.compile(r"\bmore\s+<\s*"),
    re.compile(r"\bschtasks\b"),
    re.compile(r"\bpowershell\b"),
    re.compile(r"\bpwsh\b"),
    re.compile(r"\bcmd\s+/c\b"),
    re.compile(r"\bcmd\s+/k\b"),
    re.compile(r"\bStart-Process\b"),
    re.compile(r"\bGet-ChildItem\b"),
    re.compile(r"\bGet-Content\b"),
    re.compile(r"\bSelect-String\b"),
    re.compile(r"\bSelect-Object\b"),
    re.compile(r"\bSet-Location\b"),
    re.compile(r"\bTest-Path\b"),
    re.compile(r"\bNew-Item\b"),
    re.compile(r"\bRemove-Item\b"),
    re.compile(r"\bCopy-Item\b"),
    re.compile(r"\bMove-Item\b"),
    re.compile(r"\bGet-Process\b"),
    re.compile(r"\bStop-Process\b"),
    re.compile(r"\bWrite-Output\b"),
    re.compile(r"\bWrite-Host\b"),
    re.compile(r"\bInvoke-WebRequest\b"),
    re.compile(r"\bInvoke-Expression\b"),
    re.compile(r"\bGet-Command\b"),
    re.compile(r"\bGet-Help\b"),
    re.compile(r"\bFormat-Table\b"),
    re.compile(r"\bFormat-List\b"),
    re.compile(r"\bOut-File\b"),
    re.compile(r"\bTee-Object\b"),
    re.compile(r"\bMeasure-Object\b"),
    re.compile(r"\bSort-Object\b"),
    re.compile(r"\bWhere-Object\b"),
    re.compile(r"\bForEach-Object\b"),
    re.compile(r"\bGroup-Object\b"),
    re.compile(r"\bCompare-Object\b"),
    re.compile(r"\bGet-Date\b"),
    re.compile(r"\bSet-Date\b"),
    re.compile(r"\bicacls\b"),
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
                suggestion = get_translation_suggestion(command, is_windows)
                msg = (
                    "Detected Linux/macOS command syntax. "
                    "Current shell is PowerShell on Windows. "
                    "This command may fail without WSL/Git-Bash compatibility layer."
                )
                if suggestion:
                    msg += f"\n{suggestion}"
                return msg
    else:
        for pattern in WINDOWS_SPECIFIC_PATTERNS:
            if pattern.search(command):
                suggestion = get_translation_suggestion(command, is_windows)
                msg = "Detected Windows command syntax. Current shell is bash on Linux/macOS. This command may fail."
                if suggestion:
                    msg += f"\n{suggestion}"
                return msg
    return None


def assess_command_danger(command: str) -> tuple[DangerLevel, str | None]:
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
        "ls",
        "dir",
        "cat",
        "type",
        "echo",
        "pwd",
        "cd",
        "git",
        "npm",
        "pip",
        "python",
        "node",
        "cargo",
        "grep",
        "find",
        "wc",
        "head",
        "tail",
        "less",
        "more",
        "mkdir",
        "touch",
        "cp",
        "copy",
        "mv",
        "move",
        "pytest",
        "make",
        "cmake",
        "docker",
        "kubectl",
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
    path = path.replace("\x00", "")

    dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "{", "}", "[", "]", "<", ">", "!", "\\"]
    for char in dangerous_chars:
        path = path.replace(char, "")

    path = path.replace("\\", "/")

    if not path.strip():
        return ""

    path = path.strip()
    resolved = str(Path(path).resolve())
    cwd = str(Path.cwd())
    if resolved.startswith(cwd):
        resolved = resolved[len(cwd) :].lstrip("/\\")

    return resolved


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


UNIX_TO_POWERSHELL_MAP: dict[str, str] = {
    "head": "Select-Object -First",
    "tail": "Select-Object -Last",
    "cat": "Get-Content",
    "ls": "Get-ChildItem",
    "ls -la": "Get-ChildItem | Format-Table",
    "grep": "Select-String",
    "find": "Get-ChildItem -Recurse",
    "which": "Get-Command",
    "touch": "New-Item -ItemType File -Path",
    "wc -l": "(... | Measure-Object -Line).Lines",
    "wc -c": "(... | Measure-Object -Character).Characters",
    "sort": "Sort-Object",
    "uniq": "Select-Object -Unique",
    "cut": "ForEach-Object { $_.split()[] }",
    "tr": "ForEach-Object { $_ -replace }",
    "tee": "Tee-Object",
    "diff": "Compare-Object",
    "pwd": "Get-Location",  # nosec B105
    "env": "Get-ChildItem env:",
    "printenv": "Get-ChildItem env:",
    "whoami": "[System.Security.Principal.WindowsIdentity]::GetCurrent().Name",
    "hostname": "$env:COMPUTERNAME",
    "uname -a": "[System.Environment]::OSVersion",
    "df -h": "Get-PSDrive -PSProvider FileSystem",
    "du -sh": "(Get-ChildItem -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB",
    "mkdir -p": "New-Item -ItemType Directory -Force -Path",
    "rm -rf": "Remove-Item -Recurse -Force",
    "cp -r": "Copy-Item -Recurse",
    "mv": "Move-Item",
    "chmod": "icacls",
    "chown": "icacls",
    "ps aux": "Get-Process | Format-Table",
    "kill -9": "Stop-Process -Force",
    "export": "$env:",
    "source": ". (dot-source)",
    "date": "Get-Date",
    "sleep": "Start-Sleep",
    "watch": "while($true) { Clear-Host; ...; Start-Sleep }",
    "xargs": "ForEach-Object",
    "base64": "[Convert]::ToBase64String / [Convert]::FromBase64String",
    "md5sum": "Get-FileHash -Algorithm MD5",
    "sha256sum": "Get-FileHash -Algorithm SHA256",
    "stat": "Get-Item | Format-List",
    "file": "Get-Item | Select-Object Name,Length,Extension",
    "ln -s": "New-Item -ItemType SymbolicLink",
    "readlink": "(Get-Item).Target",
}

POWERSHELL_TO_UNIX_MAP: dict[str, str] = {
    "Get-ChildItem": "ls",
    "Get-Content": "cat",
    "Select-String": "grep",
    "Select-Object": "head/tail (use -First/-Last)",
    "Get-Location": "pwd",
    "Get-Command": "which",
    "New-Item": "touch/mkdir",
    "Remove-Item": "rm",
    "Copy-Item": "cp",
    "Move-Item": "mv",
    "Get-Process": "ps",
    "Stop-Process": "kill",
    "Write-Output": "echo",
    "Invoke-WebRequest": "curl",
    "Get-Date": "date",
    "Start-Sleep": "sleep",
    "Tee-Object": "tee",
    "Sort-Object": "sort",
    "Where-Object": "grep (filtering)",
    "ForEach-Object": "xargs",
    "Measure-Object": "wc",
    "Compare-Object": "diff",
    "Format-Table": "column -t",
    "Test-Path": "test -e",
    "Set-Location": "cd",
    "Out-File": "> (redirect)",
    "icacls": "chmod/chown",
    "dir /s /b": "find . -type f",
    "findstr": "grep",
    "tasklist": "ps aux",
    "taskkill": "kill",
    "ipconfig": "ip addr",
    "type": "cat",
    "where": "which",
    "2>nul": "2>/dev/null",
}


def translate_command_for_platform(command: str, is_windows: bool) -> tuple[str, list[str]]:
    """Translate a command to be compatible with the target platform.

    Attempts to translate Unix commands to PowerShell equivalents when running
    on Windows, and PowerShell/CMD commands to Unix equivalents when running
    on Linux/macOS.

    Args:
        command: The shell command to translate
        is_windows: True if running on Windows, False if running on Linux/macOS

    Returns:
        Tuple of (translated_command, list_of_translations_made)
        Each translation entry is a string like "head -> Select-Object -First"
    """
    translations: list[str] = []
    result = command

    if is_windows:
        result, translations = _translate_unix_to_powershell(command)
    else:
        result, translations = _translate_powershell_to_unix(command)

    return result, translations


def _translate_unix_to_powershell(command: str) -> tuple[str, list[str]]:
    """Translate Unix-style commands to PowerShell equivalents.

    Args:
        command: Unix-style command string

    Returns:
        Tuple of (translated_command, list_of_translations_made)
    """
    translations: list[str] = []
    result = command

    result = re.sub(r"2>/dev/null", r"2>`$null", result)
    if "2>`$null" in result and "2>/dev/null" in command:
        translations.append("2>/dev/null -> 2>$null")

    result = re.sub(r"2>&1", r"2>&1", result)

    head_match = re.search(r"\bhead\s+-(\d+)\b", result)
    if head_match:
        n = head_match.group(1)
        result = re.sub(r"\bhead\s+-\d+\b", f"Select-Object -First {n}", result)
        translations.append(f"head -{n} -> Select-Object -First {n}")

    head_default = re.search(r"\bhead\b(?!\s+-)", result)
    if head_default and "Select-Object" not in result:
        result = re.sub(r"\bhead\b(?!\s+-)", "Select-Object -First 10", result)
        translations.append("head -> Select-Object -First 10")

    tail_match = re.search(r"\btail\s+-(\d+)\b", result)
    if tail_match:
        n = tail_match.group(1)
        result = re.sub(r"\btail\s+-\d+\b", f"Select-Object -Last {n}", result)
        translations.append(f"tail -{n} -> Select-Object -Last {n}")

    tail_default = re.search(r"\btail\b(?!\s+-)", result)
    if tail_default and "Select-Object" not in result:
        result = re.sub(r"\btail\b(?!\s+-)", "Select-Object -Last 10", result)
        translations.append("tail -> Select-Object -Last 10")

    if re.search(r"\bcat\s+", result) and "Get-Content" not in result:
        result = re.sub(r"\bcat\s+", "Get-Content ", result)
        translations.append("cat -> Get-Content")

    if re.search(r"\bls\s+-la\b", result):
        result = re.sub(r"\bls\s+-la\b", "Get-ChildItem | Format-Table", result)
        translations.append("ls -la -> Get-ChildItem | Format-Table")
    elif re.search(r"\bls\s+-l\b", result):
        result = re.sub(r"\bls\s+-l\b", "Get-ChildItem", result)
        translations.append("ls -l -> Get-ChildItem")
    elif re.search(r"\bls\b", result) and "Get-ChildItem" not in result:
        result = re.sub(r"\bls\b", "Get-ChildItem", result)
        translations.append("ls -> Get-ChildItem")

    if re.search(r"\bgrep\s+", result) and "Select-String" not in result:
        result = re.sub(r"\bgrep\s+", "Select-String -Pattern ", result)
        translations.append("grep -> Select-String -Pattern")

    if re.search(r"\bfind\s+\.", result) and "Get-ChildItem" not in result:
        name_match = re.search(r'\bfind\s+\.\s+-name\s+"([^"]+)"', result)
        if name_match:
            pattern = name_match.group(1)
            result = re.sub(
                r'\bfind\s+\.\s+-name\s+"[^"]+"',
                f'Get-ChildItem -Recurse -Filter "{pattern}"',
                result,
            )
            translations.append(f'find . -name "{pattern}" -> Get-ChildItem -Recurse -Filter "{pattern}"')
        else:
            result = re.sub(r"\bfind\s+\.", "Get-ChildItem -Recurse", result)
            translations.append("find . -> Get-ChildItem -Recurse")

    if re.search(r"\bwhich\s+", result) and "Get-Command" not in result:
        result = re.sub(r"\bwhich\s+", "Get-Command ", result)
        translations.append("which -> Get-Command")

    if re.search(r"\btouch\s+", result) and "New-Item" not in result:
        result = re.sub(r"\btouch\s+", "New-Item -ItemType File -Path ", result)
        translations.append("touch -> New-Item -ItemType File -Path")

    if re.search(r"\bpwd\b", result) and "Get-Location" not in result:
        result = re.sub(r"\bpwd\b", "Get-Location", result)
        translations.append("pwd -> Get-Location")

    if re.search(r"\bwhoami\b", result) and "WindowsIdentity" not in result:
        result = re.sub(r"\bwhoami\b", "[System.Security.Principal.WindowsIdentity]::GetCurrent().Name", result)
        translations.append("whoami -> [System.Security.Principal.WindowsIdentity]::GetCurrent().Name")

    if re.search(r"\bps\s+aux\b", result) and "Get-Process" not in result:
        result = re.sub(r"\bps\s+aux\b", "Get-Process | Format-Table", result)
        translations.append("ps aux -> Get-Process | Format-Table")

    if re.search(r"\bkill\s+-9\s+", result) and "Stop-Process" not in result:
        result = re.sub(r"\bkill\s+-9\s+", "Stop-Process -Force -Id ", result)
        translations.append("kill -9 -> Stop-Process -Force -Id")

    if re.search(r"\bexport\s+(\w+)=", result):
        result = re.sub(r"\bexport\s+(\w+)=", r"$env:\1=", result)
        translations.append("export VAR= -> $env:VAR=")

    if re.search(r"\bwc\s+-l\b", result):
        result = re.sub(r"\bwc\s+-l\b", "| Measure-Object -Line | Select-Object -ExpandProperty Lines", result)
        translations.append("wc -l -> | Measure-Object -Line | Select-Object -ExpandProperty Lines")

    if re.search(r"\bsort\b", result) and "Sort-Object" not in result:
        result = re.sub(r"\bsort\b", "Sort-Object", result)
        translations.append("sort -> Sort-Object")

    if re.search(r"\buniq\b", result) and "Select-Object" not in result:
        result = re.sub(r"\buniq\b", "Select-Object -Unique", result)
        translations.append("uniq -> Select-Object -Unique")

    if re.search(r"\btee\s+-a\b", result) and "Tee-Object" not in result:
        result = re.sub(r"\btee\s+-a\b", "Tee-Object -Append", result)
        translations.append("tee -a -> Tee-Object -Append")
    elif re.search(r"\btee\s+", result) and "Tee-Object" not in result:
        result = re.sub(r"\btee\s+", "Tee-Object -FilePath ", result)
        translations.append("tee -> Tee-Object -FilePath")

    if re.search(r"\bdiff\b", result) and "Compare-Object" not in result:
        result = re.sub(r"\bdiff\b", "Compare-Object", result)
        translations.append("diff -> Compare-Object")

    if re.search(r"\bsleep\s+(\d+)", result) and "Start-Sleep" not in result:
        sleep_match = re.search(r"\bsleep\s+(\d+)", result)
        secs = sleep_match.group(1) if sleep_match else "1"
        result = re.sub(r"\bsleep\s+\d+", f"Start-Sleep -Seconds {secs}", result)
        translations.append(f"sleep {secs} -> Start-Sleep -Seconds {secs}")

    if re.search(r"\bdate\s+\+", result) and "Get-Date" not in result:
        result = re.sub(r"\bdate\s+\+", "Get-Date -Format ", result)
        translations.append("date +format -> Get-Date -Format")

    if re.search(r"\brm\s+-rf\s+", result) and "Remove-Item" not in result:
        result = re.sub(r"\brm\s+-rf\s+", "Remove-Item -Recurse -Force ", result)
        translations.append("rm -rf -> Remove-Item -Recurse -Force")

    if re.search(r"\bcp\s+-r\s+", result) and "Copy-Item" not in result:
        result = re.sub(r"\bcp\s+-r\s+", "Copy-Item -Recurse ", result)
        translations.append("cp -r -> Copy-Item -Recurse")

    if re.search(r"\bmkdir\s+-p\s+", result) and "New-Item" not in result:
        result = re.sub(r"\bmkdir\s+-p\s+", "New-Item -ItemType Directory -Force -Path ", result)
        translations.append("mkdir -p -> New-Item -ItemType Directory -Force -Path")

    if re.search(r"\bxargs\s+", result) and "ForEach-Object" not in result:
        result = re.sub(r"\bxargs\s+", "| ForEach-Object { ", result)
        translations.append("xargs -> | ForEach-Object { }")

    if re.search(r"\bmd5sum\b", result) and "Get-FileHash" not in result:
        result = re.sub(r"\bmd5sum\b", "Get-FileHash -Algorithm MD5", result)
        translations.append("md5sum -> Get-FileHash -Algorithm MD5")

    if re.search(r"\bsha256sum\b", result) and "Get-FileHash" not in result:
        result = re.sub(r"\bsha256sum\b", "Get-FileHash -Algorithm SHA256", result)
        translations.append("sha256sum -> Get-FileHash -Algorithm SHA256")

    if re.search(r"\bstat\s+", result) and "Get-Item" not in result:
        result = re.sub(r"\bstat\s+", "Get-Item ", result)
        translations.append("stat -> Get-Item")

    return result, translations


def _translate_powershell_to_unix(command: str) -> tuple[str, list[str]]:
    """Translate PowerShell/CMD commands to Unix equivalents.

    Args:
        command: PowerShell/CMD-style command string

    Returns:
        Tuple of (translated_command, list_of_translations_made)
    """
    translations: list[str] = []
    result = command

    if re.search(r"\bdir\s+/s\s+/b\b", result):
        result = re.sub(r"\bdir\s+/s\s+/b\b", "find . -type f", result)
        translations.append("dir /s /b -> find . -type f")

    if re.search(r"\bdir\s+/s\b", result) and "find" not in result:
        result = re.sub(r"\bdir\s+/s\b", "find . -type f", result)
        translations.append("dir /s -> find . -type f")

    if re.search(r"\bfindstr\b", result) and "grep" not in result:
        result = re.sub(r"\bfindstr\b", "grep", result)
        translations.append("findstr -> grep")

    if re.search(r"\b2>nul\b", result):
        result = re.sub(r"\b2>nul\b", "2>/dev/null", result)
        translations.append("2>nul -> 2>/dev/null")

    if re.search(r"\btype\s+", result) and "cat" not in result:
        result = re.sub(r"\btype\s+", "cat ", result)
        translations.append("type -> cat")

    if re.search(r"\bwhere\s+", result) and "which" not in result:
        result = re.sub(r"\bwhere\s+", "which ", result)
        translations.append("where -> which")

    if re.search(r"\btasklist\b", result) and "ps" not in result:
        result = re.sub(r"\btasklist\b", "ps aux", result)
        translations.append("tasklist -> ps aux")

    if re.search(r"\btaskkill\s+/PID\s+(\d+)", result):
        pid_match = re.search(r"\btaskkill\s+/PID\s+(\d+)", result)
        pid = pid_match.group(1) if pid_match else "0"
        result = re.sub(r"\btaskkill\s+/PID\s+\d+", f"kill {pid}", result)
        translations.append(f"taskkill /PID {pid} -> kill {pid}")

    if re.search(r"\btaskkill\s+/F\s+/PID\s+(\d+)", result):
        pid_force_match = re.search(r"\btaskkill\s+/F\s+/PID\s+(\d+)", result)
        pid = pid_force_match.group(1) if pid_force_match else "0"
        result = re.sub(r"\btaskkill\s+/F\s+/PID\s+\d+", f"kill -9 {pid}", result)
        translations.append(f"taskkill /F /PID {pid} -> kill -9 {pid}")

    if re.search(r"\bipconfig\b", result) and "ip addr" not in result:
        result = re.sub(r"\bipconfig\b", "ip addr", result)
        translations.append("ipconfig -> ip addr")

    if re.search(r"\bGet-ChildItem\b", result) and "ls" not in result:
        result = re.sub(r"\bGet-ChildItem\b", "ls", result)
        translations.append("Get-ChildItem -> ls")

    if re.search(r"\bGet-Content\b", result) and "cat" not in result:
        result = re.sub(r"\bGet-Content\b", "cat", result)
        translations.append("Get-Content -> cat")

    if re.search(r"\bSelect-String\b", result) and "grep" not in result:
        result = re.sub(r"\bSelect-String\b", "grep", result)
        translations.append("Select-String -> grep")

    if re.search(r"\bSelect-Object\s+-First\s+(\d+)", result):
        first_match = re.search(r"\bSelect-Object\s+-First\s+(\d+)", result)
        n = first_match.group(1) if first_match else "10"
        result = re.sub(r"\bSelect-Object\s+-First\s+\d+", f"head -{n}", result)
        translations.append(f"Select-Object -First {n} -> head -{n}")

    if re.search(r"\bSelect-Object\s+-Last\s+(\d+)", result):
        last_match = re.search(r"\bSelect-Object\s+-Last\s+(\d+)", result)
        n = last_match.group(1) if last_match else "10"
        result = re.sub(r"\bSelect-Object\s+-Last\s+\d+", f"tail -{n}", result)
        translations.append(f"Select-Object -Last {n} -> tail -{n}")

    if re.search(r"\bGet-Location\b", result) and "pwd" not in result:
        result = re.sub(r"\bGet-Location\b", "pwd", result)
        translations.append("Get-Location -> pwd")

    if re.search(r"\bGet-Command\b", result) and "which" not in result:
        result = re.sub(r"\bGet-Command\b", "which", result)
        translations.append("Get-Command -> which")

    if re.search(r"\bNew-Item\s+-ItemType\s+File\b", result):
        result = re.sub(r"\bNew-Item\s+-ItemType\s+File\s+-Path\s+", "touch ", result)
        translations.append("New-Item -ItemType File -Path -> touch")

    if re.search(r"\bRemove-Item\s+-Recurse\s+-Force\b", result):
        result = re.sub(r"\bRemove-Item\s+-Recurse\s+-Force\s+", "rm -rf ", result)
        translations.append("Remove-Item -Recurse -Force -> rm -rf")

    if re.search(r"\bCopy-Item\s+-Recurse\b", result):
        result = re.sub(r"\bCopy-Item\s+-Recurse\s+", "cp -r ", result)
        translations.append("Copy-Item -Recurse -> cp -r")

    if re.search(r"\bMove-Item\b", result) and "mv" not in result:
        result = re.sub(r"\bMove-Item\b", "mv", result)
        translations.append("Move-Item -> mv")

    if re.search(r"\bGet-Process\b", result) and "ps" not in result:
        result = re.sub(r"\bGet-Process\b", "ps aux", result)
        translations.append("Get-Process -> ps aux")

    if re.search(r"\bStop-Process\s+-Force\s+-Id\s+", result):
        result = re.sub(r"\bStop-Process\s+-Force\s+-Id\s+", "kill -9 ", result)
        translations.append("Stop-Process -Force -Id -> kill -9")

    if re.search(r"\bStop-Process\b", result) and "kill" not in result:
        result = re.sub(r"\bStop-Process\b", "kill", result)
        translations.append("Stop-Process -> kill")

    if re.search(r"\bWrite-Output\b", result) and "echo" not in result:
        result = re.sub(r"\bWrite-Output\b", "echo", result)
        translations.append("Write-Output -> echo")

    if re.search(r"\bInvoke-WebRequest\b", result) and "curl" not in result:
        result = re.sub(r"\bInvoke-WebRequest\b", "curl", result)
        translations.append("Invoke-WebRequest -> curl")

    if re.search(r"\bGet-Date\b", result) and "date" not in result:
        result = re.sub(r"\bGet-Date\b", "date", result)
        translations.append("Get-Date -> date")

    if re.search(r"\bStart-Sleep\s+-Seconds\s+(\d+)", result):
        sleep_ps_match = re.search(r"\bStart-Sleep\s+-Seconds\s+(\d+)", result)
        secs = sleep_ps_match.group(1) if sleep_ps_match else "1"
        result = re.sub(r"\bStart-Sleep\s+-Seconds\s+\d+", f"sleep {secs}", result)
        translations.append(f"Start-Sleep -Seconds {secs} -> sleep {secs}")

    if re.search(r"\bTee-Object\b", result) and "tee" not in result:
        result = re.sub(r"\bTee-Object\b", "tee", result)
        translations.append("Tee-Object -> tee")

    if re.search(r"\bSort-Object\b", result) and "sort" not in result:
        result = re.sub(r"\bSort-Object\b", "sort", result)
        translations.append("Sort-Object -> sort")

    if re.search(r"\bMeasure-Object\b", result) and "wc" not in result:
        result = re.sub(r"\bMeasure-Object\b", "wc", result)
        translations.append("Measure-Object -> wc")

    if re.search(r"\bCompare-Object\b", result) and "diff" not in result:
        result = re.sub(r"\bCompare-Object\b", "diff", result)
        translations.append("Compare-Object -> diff")

    if re.search(r"\bTest-Path\b", result) and "test" not in result:
        result = re.sub(r"\bTest-Path\b", "test -e", result)
        translations.append("Test-Path -> test -e")

    if re.search(r"\bicacls\b", result) and "chmod" not in result:
        result = re.sub(r"\bicacls\b", "chmod", result)
        translations.append("icacls -> chmod")

    return result, translations


def get_translation_suggestion(command: str, is_windows: bool) -> str | None:
    """Get a human-readable translation suggestion for a platform-mismatched command.

    Args:
        command: The shell command
        is_windows: True if running on Windows

    Returns:
        Translation suggestion string, or None if no translation available
    """
    _, translations = translate_command_for_platform(command, is_windows)

    if not translations:
        if is_windows:
            for unix_cmd, ps_cmd in UNIX_TO_POWERSHELL_MAP.items():
                if re.search(rf"\b{re.escape(unix_cmd.split()[0])}\b", command):
                    return f"Try: {ps_cmd} instead of {unix_cmd}"
        else:
            for ps_cmd, unix_cmd in POWERSHELL_TO_UNIX_MAP.items():
                if re.search(rf"\b{re.escape(ps_cmd.split()[0])}\b", command):
                    return f"Try: {unix_cmd} instead of {ps_cmd}"
        return None

    return "Translation suggestions:\n  " + "\n  ".join(translations)


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
    if ";rm" in lower_cmd or "; del" in lower_cmd:
        suspicious.append("Potential command injection (semicolon)")

    # Check for pipe to shell (common attack pattern)
    if "| sh" in lower_cmd or "| bash" in lower_cmd or "&& sh" in lower_cmd:
        suspicious.append("Pipe to shell detected")

    # Check for download and execute patterns
    if ("curl" in lower_cmd or "wget" in lower_cmd) and ("| sh" in lower_cmd or "| python" in lower_cmd):
        suspicious.append("Download and execute pattern")

    # Check for environment variable manipulation
    if "export " in lower_cmd and "=" in command:
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
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    output = ansi_pattern.sub("", output)

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
