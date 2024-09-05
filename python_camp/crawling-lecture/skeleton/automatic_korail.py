from playwright.async_api import async_playwright, Page
from time import time, sleep
import asyncio


# async def assert_value(page, query, desired_value):
#     res = await page.evaluate('() => {return document.querySelector(' + query + ').value; }')
#     while res != desired_value:
#         await page.evaluate('''() => {
#                                 document.querySelector(''' + query + ''').value = ''' + desired_value + ''';
#                             }''')
#         res = await page.evaluate('() => {return document.querySelector(' + query + ').value; }')

async def type_to_selector(page, txt, selector, retry = 3):
    select_component = await page.query_selector(selector)
    await select_component.type(txt)
    

    current_typed_content = await page.evaluate('''() => {
                            return document.querySelector('%s').value; 
                         }'''%selector)
    print(current_typed_content)
    try_n = 0
    while current_typed_content != txt:
        select_component = await page.query_selector(selector)
        await select_component.type(txt)
        current_typed_content = await page.evaluate('''() => {
                            return document.querySelector('%s').value; 
                         }'''%selector)
        try_n += 1 
        print(current_typed_content)

        if try_n == retry:
            raise ValueError(f'tried to write {txt} to {selector} on {page.url}, faield with {retry} tries...')


async def login_korail(context, page, my_id = '', my_pwd = ''):
    
    await page.evaluate('''(() => {
        img = document.querySelector('img[src="/images/gnb_login.gif"]');
        img.click();
    })();''')

    # login = await page.wait_for_selector('.gnb')
    # await login.click()
    
    Id = await page.wait_for_selector('#txtMember.txt')
    await Id.type(my_id[:4])
    print(Id)
    assert Id is not None 
    # sleep(1)
    
    res = await page.evaluate('''() => {
                            id_input = document.querySelector('input[id="txtMember"]'); 
                            return id_input.value; 
                         }''')
    while not res == my_id:
        # sleep(1)
        await page.evaluate('''() => {
                            id_input = document.querySelector('input[id="txtMember"]'); 
                            return id_input.value = %s; 
                         }''' % my_id)
        # sleep(1)
        res = await page.evaluate('''() => {
                            id_input = document.querySelector('input[id="txtMember"]'); 
                            return id_input.value; 
                         }''')
    
    await type_to_selector(page, my_id, 'input[id="txtMember"]')
    await type_to_selector(page, my_pwd, 'input[id="txtPwd"]')
    
    pwd = await page.wait_for_selector('#txtPwd.txt')
    await pwd.fill(my_pwd)

    btn = await page.wait_for_selector('.btn_login')
    await btn.click()

    # target_url = 'https://www.letskorail.com/index.jsp'
    # await close_unwanted_tabs(context, target_url)
    

    # sleep(5)

async def select_place(page, start, end, date, time):
    start_place = await page.wait_for_selector('#txtGoStart.txt120')
    await start_place.type(start)

    end_place = await page.wait_for_selector('#txtGoEnd.txt120')
    await end_place.type(end)

    go_date = await page.wait_for_selector('#selGoStartDay.txt120')
    await go_date.type(date)

    go_time = await page.wait_for_selector('#time.select')
    await go_time.click()

    # sleep(5)

async def close_unwanted_tabs(context, target_url):
    for page in context.pages:
        print(page)
        if page.url != target_url:
            await page.close()

async def handle_popup(popup: Page):
    print('함수실행')
    print(popup.url)
    await popup.close()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless = False)
        context = await browser.new_context()
        page = await browser.new_page()
        
        await page.goto('https://www.letskorail.com/')
        await login_korail(context, page, my_id = '1663468006', my_pwd = '@heart8556')
        page.on('popup', handle_popup)
        res = await page.wait_for_selector('.btn_res')
        
        cur_context = page.context
        # from code import interact
        # interact(local = locals())
        # sleep(10)
        # await cur_context.wait_for_event('popup')

        

        # for context in browser.contexts:
        #     print(context)
        #     for page in context.pages:
        #         if page.url != 'https://www.letskorail.com/index.jsp':
        #             await page.close()
        
        # await close_unwanted_tabs(context, 'https://www.letskorail.com/index.jsp')
        
        await browser.close()
        
        
        
        
        
        
        # await close_unwanted_tabs(context, target_url)
        

        # await select_place(page, '서울', '대전', '2024.10.1', '08')

if __name__ == '__main__':
    begin = time()
    asyncio.run(main())
    end = time()

    print(end - begin)