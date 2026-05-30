import os

from playwright.sync_api import sync_playwright

# 示例：使用 file:// URL 自动化静态 HTML 文件的交互操作

html_file_path = os.path.abspath("path/to/your/file.html")
file_url = f"file://{html_file_path}"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    # 导航到本地 HTML 文件
    page.goto(file_url)

    # 截图
    page.screenshot(path="/mnt/user-data/outputs/static_page.png", full_page=True)

    # 与元素进行交互
    page.click("text=Click Me")
    page.fill("#name", "John Doe")
    page.fill("#email", "john@example.com")

    # 提交表单
    page.click('button[type="submit"]')
    page.wait_for_timeout(500)

    # 拍摄最终截图
    page.screenshot(path="/mnt/user-data/outputs/after_submit.png", full_page=True)

    browser.close()

print("Static HTML automation completed!")
