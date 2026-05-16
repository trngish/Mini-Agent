"""Fetch M2.7 info from MiniMax news page"""

import requests
import urllib3
import re
import json

urllib3.disable_warnings()

def fetch_news_page():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/json',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
    }
    
    print('=== Fetching MiniMax M2.7 News Page ===')
    r = requests.get('https://www.minimaxi.com/news/minimax-m27-zh', timeout=15, verify=False, headers=headers)
    r.encoding = 'utf-8'
    print(f'Status: {r.status_code}')
    print(f'Content length: {len(r.text)}')
    
    # Check for __NEXT_DATA__
    next_data = re.search(r'id="__NEXT_DATA__"[^>]*>([^<]+)<', r.text, re.DOTALL)
    if next_data:
        print()
        print('=== Found __NEXT_DATA__ ===')
        try:
            data = json.loads(next_data.group(1))
            # Look for props.pageProps content
            if 'props' in data and 'pageProps' in data['props']:
                page_data = data['props']['pageProps']
                data_str = json.dumps(page_data)
                print(f'pageProps length: {len(data_str)} chars')
                
                # Search for keywords
                keywords = ['context', 'token', 'thinking', '1m', '1M', '32k', '32K', '100万', '100k', '10k']
                for kw in keywords:
                    if kw.lower() in data_str.lower():
                        idx = data_str.lower().find(kw.lower())
                        print(f'Found "{kw}" at index {idx}: ...{data_str[max(0,idx-50):idx+150]}...')
            else:
                # Check full data structure
                print('Keys in data:', list(data.keys()))
                data_str = json.dumps(data)
                print(f'Full data length: {len(data_str)} chars')
                
                # Search for keywords
                for kw in ['context', 'token', 'thinking', '1m', '32k']:
                    if kw.lower() in data_str.lower():
                        print(f'Found keyword: {kw}')
        except Exception as e:
            print(f'Error: {e}')
    else:
        print('No __NEXT_DATA__ found')
    
    # Extract visible text
    html_clean = re.sub(r'<script[^>]*>.*?</script>', '', r.text, flags=re.DOTALL|re.IGNORECASE)
    html_clean = re.sub(r'<style[^>]*>.*?</style>', '', html_clean, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html_clean)
    text = re.sub(r'\s+', ' ', text).strip()
    
    print()
    print('=== Page Text (first 5000 chars) ===')
    print(text[:5000])

if __name__ == '__main__':
    fetch_news_page()