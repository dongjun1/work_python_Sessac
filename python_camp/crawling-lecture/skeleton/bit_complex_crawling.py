import requests 
from bs4 import BeautifulSoup
from time import time
import pickle
import os
import json

# json.loads() : python dict로 만들어주는 함수
# 언론사별 코드를 언론사와 매칭시키는 dict를 반환하는 함수.
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

# 위의 함수에서 반환받은 dict를 input으로 받아 기자들의 이름과 정보페이지를 매칭시킨 dict를 반환.
def crawl_journalist_info_pages(code2name):
    """Accepts press code - press name dict, and return dict having press code as key, and 2-tuple of (press name, listof 2-tuple containing journalist name and their link) as value. 

    For now, you DO NOT have to crawl all journalists; for now, it's impossible. 
    Crawl from https://media.naver.com/journalists/. 
    """
    
    res = {}
    # begin = time()
    for press_code, press_name in code2name.items():
        url = f'https://media.naver.com/journalists/whole?officeId={press_code}'
        
        response = requests.get(url)

        

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            journalist_list = []
            for li in soup.find_all('li', {'class':'journalist_list_content_item'}):
                info = li.find('div', {'class':'journalist_list_content_title'})
                a = info.find('a')
                journalist_name = a.text.strip(' 기자')
                journalist_link = a['href']
                journalist_list.append((journalist_name, journalist_link))
        
        res[press_code] = (press_name, journalist_list)
    # end = time()
    # print(end - begin)
    return res 

class Journalist:
    def __init__(self, name, press_code, 
                career_list, 
                graduated_from, 
                no_of_subscribers, 
                subscriber_age_statistics, 
                subscriber_gender_statistics, 
                article_list):
        self.name = name 
        self.press_code = press_code 
        self.career_list = career_list
        self.graduated_from = graduated_from
        self.no_of_subscribers = no_of_subscribers
        self.subscriber_age_statistics = subscriber_age_statistics
        self.subscriber_gender_statistics = subscriber_gender_statistics
        self.article_list = article_list 



def crawl_journalist_info(link):
    """Make a Journalist class instance using the information in the given link. 
    """
    response = requests.get(link)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        profile_head = soup.find('div', {'class' : 'media_reporter_basic_text'})

        press_code = profile_head.find('a')['href'].split('/')[-1]
        journalist_name = profile_head.find('h2', {'class' : 'media_reporter_basic_name'}).text
        # print(press_code, journalist_name)
        award_div = soup.find('div', {'class':'media_reporter_profile_award'})
        award_dict = {}
        school_list = []
        for award in award_div.find_all('div', {'class':'media_reporter_award_div'}):
            award_category = award.find('h4', {'class':'media_reporter_award_category'}).text
            award_list = award.find('ul', {'class':'media_reporter_award_list'})
            # print(award_category)
            if award_category != '학력':
                career_list = []
                for award_item in award_list.find_all('li', {'class':'media_reporter_award_item'}):
                    award_year = award_item.find('em', {'class':'media_reporter_award_year'}).text.strip()
                    award_name = award_item.find('ul', {'class':'media_reporter_award_name'}).text.strip()
                    career_list.append((award_year, award_name))
                    print('      ', award_year, award_name)
                award_dict[award_category] = career_list
            else:
                for school_item in award_list.find_all('li', {'class':'media_reporter_award_item'}):
                    graduation_year = school_item.find('em', {'class':'media_reporter_award_year'}).text.strip()
                    school_name = school_item.find('ul', {'class':'media_reporter_award_name'}).text.strip()
                    
                    school_list.append((graduation_year, school_name))

        no_of_subscribers = requests.get(f'https://media.naver.com/myfeed/getFeedCount?channelKeys=JOURNALIST_{link.split("/")[-1]}')

        if no_of_subscribers.status_code == 200:
            no_of_subscribers = json.loads(no_of_subscribers.text)['result'][0]['count']
        else:
            no_of_subscribers = 0
        
        print(no_of_subscribers)
        

        subscriber_age_statistics = {}
        age_statistics_chart = soup.find('dl', {'class':'group_barchart'})

        for age_category in age_statistics_chart.find_all('div', {'class':'group'}):
            age = age_category.find('dt').text.strip()
            value = age_category.find('span', {'class':'percent'}).text.strip('%')
            subscriber_age_statistics[age] = value

        for k, v in subscriber_age_statistics.items():
            print(k, v)


if __name__ == '__main__':
    code2info_pickle = 'code2info.pickle'

    # if code2info_pickle in os.listdir():
    #     begin = time()
    #     code2info = pickle.load(open(code2info_pickle, 'rb'))
    #     end = time()
    #     print(f'{end - begin} sec passed for unpickling')
    # else:
    #     begin = time()
    #     code2name = crawl_press_names_and_codes()
    #     code2info = crawl_journalist_info_pages(code2name)
    #     pickle.dump(code2info, open(code2info_pickle, 'wb+'))
    #     end = time()
    #     print(f'{end - begin} sec passed for execution and pickling')
    code2name = crawl_press_names_and_codes()
    begin = time()
    code2info = crawl_journalist_info_pages(code2name)
    end = time()
    print(end - begin)
    # crawl_journalist_info('https://media.naver.com/journalist/006/31564')
    # crawl_journalist_info('https://media.naver.com/journalist/586/75734')
        # for code, (press_name, journalist_list) in code2info.items():
        #     for journalist_name, link in journalist_list:
        #         j = crawl_journalist_info(link)
        #         assert j.name == journalist_name

    
    