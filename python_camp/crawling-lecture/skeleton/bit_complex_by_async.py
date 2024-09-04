import asyncio
import aiohttp
import aiohttp.client
from time import time
import requests 
from bs4 import BeautifulSoup


def crawl_press_names_and_codes():
    """Make the dict that have press code as key, and press name as value. Crawl from https://media.naver.com/channel/settings. 
    """
    begin = time()
    url = 'https://media.naver.com/channel/settings'
    code2name = {}
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        div = soup.find('div', {'class' : 'channel_add'})
        for li in div.find_all('li'):
            code2name[li['data-office']] = li.find('div', {'class' : 'ca_name'}).text
    
    end = time()
    print(end - begin)

    return code2name 


async def afetch(url, session):
    response = await session.get(url)

    if response.status == 200:
        soup = BeautifulSoup((await response.text()), 'html.parser')
        journalist_list = []
        for li in soup.find_all('li', {'class':'journalist_list_content_item'}):
            info = li.find('div', {'class':'journalist_list_content_title'})
            a = info.find('a')
            journalist_name = a.text.strip(' 기자')
            journalist_link = a['href']
            journalist_list.append((journalist_name, journalist_link))

    return journalist_list

async def main():
    code2name = crawl_press_names_and_codes()
    urls = [f'https://media.naver.com/journalists/whole?officeId={press_code}' for press_code in code2name]

    session = aiohttp.ClientSession()

    tasks = [afetch(url, session) for url in urls]

    # for task in tasks:
    #     print(task)

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
