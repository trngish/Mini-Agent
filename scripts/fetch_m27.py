"""Fetch and analyze M2.7 specifications from MiniMax website"""

import requests
import urllib3
import re
import json

urllib3.disable_warnings()

def fetch_m27_specs():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
    }
    
    print('=== Fetching MiniMax M2.7 Model Page ===')
    r = requests.get('https://www.minimaxi.com/models/text/m27', timeout=15, verify=False, headers=headers)
    r.encoding = 'utf-8'
    print(f'Status: {r.status_code}')
    print(f'Content length: {len(r.text)}')
    
    # Remove scripts and styles
    html_clean = re.sub(r'<script[^>]*>.*?</script>', '', r.text, flags=re.DOTALL|re.IGNORECASE)
    html_clean = re.sub(r'<style[^>]*>.*?</style>', '', html_clean, flags=re.DOTALL|re.IGNORECASE)
    
    # Extract visible text
    text = re.sub(r'<[^>]+>', ' ', html_clean)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Look for __NEXT_DATA__
    next_data_match = re.search(r'id="__NEXT_DATA__"[^>]*>([^<]+)<', r.text, re.DOTALL)
    if next_data_match:
        print()
        print('=== Found __NEXT_DATA__ ===')
        try:
            data = json.loads(next_data_match.group(1))
            data_str = json.dumps(data)
            print(f'Data length: {len(data_str)} chars')
            
            # Search for spec keywords
            keywords = ['context', 'token', 'thinking', '1m', '1M', '32k', '32K', 'output', 'budget']
            for kw in keywords:
                if kw.lower() in data_str.lower():
                    print(f'Found keyword: {kw}')
                    # Find the context around it
                    idx = data_str.lower().find(kw.lower())
                    print(f'  Context: ...{data_str[max(0,idx-50):idx+100]}...')
        except Exception as e:
            print(f'Error parsing __NEXT_DATA__: {e}')
    
    # Look for table data or spec lists
    print()
    print('=== Looking for Tables/Specs ===')
    tables = re.findall(r'<table[^>]*>(.*?)</table>', r.text, re.DOTALL|re.IGNORECASE)
    print(f'Found {len(tables)} tables')
    
    for i, table in enumerate(tables[:3]):
        # Clean table
        table_text = re.sub(r'<[^>]+>', ' ', table)
        table_text = re.sub(r'\s+', ' ', table_text).strip()
        if len(table_text) > 20:
            print(f'Table {i}: {table_text[:300]}')
    
    # Look for definition lists or spec sections
    print()
    print('=== Looking for Spec Sections ===')
    dl_matches = re.findall(r'<dl[^>]*>(.*?)</dl>', r.text, re.DOTALL|re.IGNORECASE)
    print(f'Found {len(dl_matches)} definition lists')
    
    # Search for specific patterns in visible text
    print()
    print('=== Key Spec Patterns ===')
    
    # Look for numbers followed by units
    patterns = [
        (r'(\d+(?:,\d{3})*)\s*(?:K|k)\s*(?:token|tokens|Tokens)', 'K tokens'),
        (r'(\d+(?:,\d{3})*)\s*(?:M|m)\s*(?:token|tokens|Tokens)', 'M tokens'),
        (r'(\d+(?:,\d{3})*)\s*(?:K|k)\s*(?:context|Context)', 'K context'),
        (r'(\d+(?:,\d{3})*)\s*(?:M|m)\s*(?:context|Context)', 'M context'),
        (r'(\d+(?:,\d{3})*)\s*(?:K|k)\s*(?:output|Output)', 'K output'),
        (r'(\d+(?:,\d{3})*)\s*(?:M|m)\s*(?:output|Output)', 'M output'),
        (r'(\d+(?:,\d{3})*)\s*(?:K|k)\s*(?:thinking|Thinking|budget)', 'K thinking/budget'),
        (r'(\d+(?:,\d{3})*)\s*(?:M|m)\s*(?:thinking|Thinking|budget)', 'M thinking/budget'),
    ]
    
    for pattern, label in patterns:
        matches = re.findall(pattern, text)
        if matches:
            print(f'{label}: {matches[:10]}')
    
    # Final output
    print()
    print('=== Summary ===')
    print(f'Total text length: {len(text)} chars')
    
    # Try to find M2.7 specific info
    m27_sections = []
    lines = text.split('.')
    for line in lines:
        if 'M2.7' in line or 'm2.7' in line:
            cleaned = line.strip()
            if len(cleaned) > 20:
                m27_sections.append(cleaned[:200])
    
    if m27_sections:
        print()
        print('M2.7 mentions:')
        for section in m27_sections[:5]:
            print(f'  - {section}')

if __name__ == '__main__':
    fetch_m27_specs()