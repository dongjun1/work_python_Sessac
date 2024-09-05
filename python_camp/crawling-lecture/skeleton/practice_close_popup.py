from playwright.async_api import async_playwright
from time import time, sleep

# 예시 URL
TARGET_URL = "https://www.naver.com"

async def close_unwanted_tabs(context, target_url):
    # 현재 열린 모든 페이지를 확인합니다.
    for page in context.pages:
        # 페이지의 URL이 대상 URL과 일치하지 않는 경우 닫습니다.
        if page.url != target_url:
            await page.close()

async def main():
    # Playwright를 비동기 방식으로 실행합니다.
    async with async_playwright() as p:
        # 브라우저를 실행합니다 (여기서는 Chrome 기반 브라우저 사용).
        browser = await p.chromium.launch(headless=False)
        # 브라우저 컨텍스트 생성.
        context = await browser.new_context()

        # 테스트를 위해 여러 페이지를 엽니다.
        page1 = await context.new_page()
        await page1.goto("https://naver.com")  # 유지할 페이지
        page2 = await context.new_page()
        await page2.goto("https://google.com")   # 닫을 페이지
        page3 = await context.new_page()
        await page3.goto("https://bing.com")     # 닫을 페이지

        # 지정된 URL을 제외한 모든 탭 닫기.
        await close_unwanted_tabs(context, TARGET_URL)
        sleep(3)

        # 브라우저를 닫기 전에 일정 시간 대기.
        input("브라우저를 종료하려면 엔터 키를 누르세요...")

        # 브라우저 종료.
        await browser.close()

# 비동기 main 함수 실행.
import asyncio
asyncio.run(main())