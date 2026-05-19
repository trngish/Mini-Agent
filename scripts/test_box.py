import os
os.chdir('D:/Mini-Agent')
import re
ansi = re.compile(r'\x1b\[[0-9;]*m')

from mini_agent.utils.display import Colors
from mini_agent.utils.terminal_utils import calculate_display_width

# Simulate exactly what the code does
step = 0
max_steps = 50
step_text = f'{Colors.BOLD}{Colors.BRIGHT_CYAN}  Step {step + 1}/{max_steps}{Colors.RESET}'
step_width = calculate_display_width(step_text)

print(f'step_text raw: {repr(step_text)}')
print(f'step_text clean: {repr(ansi.sub("", step_text))}')
print(f'step_text display width: {step_width}')

# Line 379, 380, 381
line1 = f"\n  {Colors.DIM}╭{'─' * 44}╮{Colors.RESET}"
line2 = f"  {Colors.DIM}│{Colors.RESET} {step_text}{' ' * (44 - step_width)}{Colors.DIM}│{Colors.RESET}"
line3 = f"  {Colors.DIM}╰{'─' * 44}╯{Colors.RESET}"

print()
print('Line 1 raw:', repr(line1))
print('Line 2 raw:', repr(line2))
print('Line 3 raw:', repr(line3))

l1_clean = ansi.sub('', line1)
l2_clean = ansi.sub('', line2)
l3_clean = ansi.sub('', line3)

print()
print(f'Line 1 clean length: {len(l1_clean)}')
print(f'Line 2 clean length: {len(l2_clean)}')
print(f'Line 3 clean length: {len(l3_clean)}')
print()
print('Line 1:', l1_clean)
print('Line 2:', l2_clean)
print('Line 3:', l3_clean)

# What should they be?
# Line 1: \n + 2 spaces + ╭ + 44 ─ + ╮ = 1 + 2 + 1 + 44 + 1 = 49
# Line 2: 2 spaces + │ + 1 space + step_text(11) + padding(33) + │ = 2 + 1 + 1 + 11 + 33 + 1 = 49
# Line 3: 2 spaces + ╰ + 44 ─ + ╯ = 2 + 1 + 44 + 1 = 48

# The issue: line 2 has extra content because step_text includes "  " (leading spaces)
# But the ╭ in line 1 is at position 3, not position 2

print()
print('Comparison of positions:')
print('Line 1: border at position 3 (╭) and 48 (╮)')
print('Line 2: border at position 2 (│) and 48 (│)')
print('Line 3: border at position 2 (╰) and 47 (╯)')
print()
print('Line 1 has 2 spaces before ╭, Line 2 and 3 have 2 spaces before their borders')
print('Line 2 has extra 1 space between left border and content')
print('So Line 2 content area is 1 space + step_text + padding = 1 + 11 + 33 = 45')
print('But Line 1 content area is 44 ─')
print()
print('This causes the right border to be misaligned!')

# The fix: adjust the content area width
# Line 2 needs: left border + content + right border to align with line 1
# If line 1 ╭ is at position 3 and ╮ at 48, content area is 44 (positions 4-47)
# Line 2 ╭ is at position 2, so to align right border at 48, content area should be 45
# But the step_text is 11, so padding = 45 - 11 = 34

print()
print('Possible fix: change 44 to 45 in line 380 calculation')
print('New line 2 would have 1 + 11 + 34 = 46 content width')
print('And line 1/3 would have 45 ─ instead of 44')
print()
print('Let me try: line1 = f"  {Colors.DIM}╭{\'─\' * 45}╮{Colors.RESET}"')
print('And line2 padding = 45 - step_width')
print()

# Test this theory
line1_fixed = f"  {Colors.DIM}╭{'─' * 45}╮{Colors.RESET}"
line2_fixed = f"  {Colors.DIM}│{Colors.RESET} {step_text}{' ' * (45 - step_width)}{Colors.DIM}│{Colors.RESET}"
line3_fixed = f"  {Colors.DIM}╰{'─' * 45}╯{Colors.RESET}"

l1f = ansi.sub('', line1_fixed)
l2f = ansi.sub('', line2_fixed)
l3f = ansi.sub('', line3_fixed)

print(f'Fixed Line 1 clean length: {len(l1f)}')
print(f'Fixed Line 2 clean length: {len(l2f)}')
print(f'Fixed Line 3 clean length: {len(l3f)}')
print()
print('Fixed Line 1:', l1f)
print('Fixed Line 2:', l2f)
print('Fixed Line 3:', l3f)