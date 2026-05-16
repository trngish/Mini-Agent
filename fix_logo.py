import re

with open('mini_agent/ui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the logo block using regex
pattern = r'    logo = \[.*?\n    \]'
match = re.search(pattern, content, re.DOTALL)

if match:
    old_block = match.group(0)
    print("Found old logo block:")
    print(repr(old_block[:200]))
    
    new_block = '''    logo = [
        f"{Colors.CYAN}███╗   ███╗███╗   ███╗██╗  ██╗",
        f"{Colors.BRIGHT_CYAN}████╗ ████║████╗ ████║╚██╗██╔╝",
        f"{Colors.BRIGHT_CYAN}██╔████╔██║██╔████╔██║ ╚███╔╝",
        f"{Colors.CYAN}██║╚██╔╝██║██║╚██╔╝██║ ██╔██╗",
        f"{Colors.CYAN}██║ ╚═╝ ██║██║ ╚═╝ ██║██╔╝ ██╗",
        f"{Colors.CYAN}╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝",
    ]'''
    
    content = content.replace(old_block, new_block)
    with open('mini_agent/ui.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('\nReplaced successfully')
else:
    print("Logo block not found")