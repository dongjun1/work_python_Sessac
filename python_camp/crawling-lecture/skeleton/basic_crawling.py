import requests 
from bs4 import BeautifulSoup

def crawl_breaking_news_list():
    news_url = 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=001&sid2=140&oid=001&isYeonhapFlash=Y&aid=0014907888'

    response = requests.get(news_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        td = soup.find('td', {'class' : 'content'})
        
        for li in td.find_all('li'):
            try:
                if li['data-comment'] is not None:
                    a = li.find('a')
                    link = a['href']
                    text = a.text
                    print(link, text)
            except KeyError:
                pass


def crawl_ranking_news():
    ranking_url = 'https://news.naver.com/main/ranking/popularDay.naver'

    response = requests.get(ranking_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # strong = soup.find_all('strong', {'class' : 'rankingnews_name'})
        
        div = soup.find_all('div', {'class' : 'rankingnews_box'})
        for d in div:
            strong = d.find('strong', {'class' : 'rankingnews_name'})
            print(strong.text)
            ul = d.find('ul')
            for di in ul.find_all('div', {'class' : 'list_content'}):
                a = di.find('a')
                link = a['href']
                text = a.text
                print(link, text)            
        
        # for s in strong:
        #     name = s.text
        #     print(name)
        #     for d in div:
        #         ul = d.find('ul')
        #         for di in ul.find_all('div', {'class' : 'list_content'}):
        #             a = di.find('a')
        #             link = a['href']
        #             text = a.text
        #             print(link, text)            
        

if __name__ == '__main__':
    #crawl_breaking_news_list()
    crawl_ranking_news()