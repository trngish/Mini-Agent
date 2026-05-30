from playwright.sync_api import sync_playwright

# 示例：发现页面上的按钮和其他元素

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 导航到页面并等待其完全加载
    page.goto("http://localhost:5173")
    page.wait_for_load_state("networkidle")

    # 发现页面上的所有按钮
    buttons = page.locator("button").all()
    print(f"Found {len(buttons)} buttons:")
    for i, button in enumerate(buttons):
        text = button.inner_text() if button.is_visible() else "[hidden]"
        print(f"  [{i}] {text}")

    # 发现链接
    links = page.locator("a[href]").all()
    print(f"\nFound {len(links)} links:")
    for link in links[:5]:  # 显示前 5 个
        text = link.inner_text().strip()
        href = link.get_attribute("href")
        print(f"  - {text} -> {href}")

    # 发现输入字段
    inputs = page.locator("input, textarea, select").all()
    print(f"\nFound {len(inputs)} input fields:")
    for input_elem in inputs:
        name = input_elem.get_attribute("name") or input_elem.get_attribute("id") or "[unnamed]"
        input_type = input_elem.get_attribute("type") or "text"
        print(f"  - {name} ({input_type})")

    # 截图作为视觉参考
    page.screenshot(path="/tmp/page_discovery.png", full_page=True)
    print("\nScreenshot saved to /tmp/page_discovery.png")

    browser.close()
