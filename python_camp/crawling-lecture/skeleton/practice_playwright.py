from playwright.async_api import async_playwright
from time import time, sleep
import asyncio

async def naver_search(page, search_keyword = '새싹'):
    search = await page.wait_for_selector('#query') # query 속성을 가진 tag가 일정 시간 이상 응답이 없을 경우 TimeOut Error
    await search.type(search_keyword)
    excute_search = await page.wait_for_selector('.btn_search')
    await excute_search.click()
    await page.wait_for_selector('.sc_page_inner')
    await page.evaluate('() => {window.scrollTo(0, 10000);};')

    sleep(10)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless = False)
        context = await browser.new_context()
        page = await browser.new_page()
        await page.goto('https://www.naver.com')
        await naver_search(page, search_keyword='새싹')

if __name__ == '__main__':
    asyncio.run(main())