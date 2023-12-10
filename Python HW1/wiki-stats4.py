#!/bin/python

import argparse
import requests
import bs4
from bs4 import BeautifulSoup
import re
from typing import Union


def counter_func(page: str, region: str)-> Union[str,bool]:
    
    """
    Manual:

    page: name directory after /wiki/
    region: country name

    """
    your_link = f"https://{region}.wikipedia.org/wiki/{page}"
    response = requests.get(your_link)
    if response.status_code != requests.codes.ok:
        return False

    soup = BeautifulSoup(response.content,'html.parser')
    total_lks = soup.find_all('a')
    count_need = 0
    count_total =len( total_lks)
    disambiguation = []
    for link in total_lks:
        href = str(link.get('href'))
        if href.startswith('/wiki/') and not href.startswith('/wiki/Template') and not href.startswith('#') and not href.startswith('/wiki/Help') and not href.startswith('/wiki/Special'):
            count_need += 1
            if href.startswith(your_link):
                disambiguation.append(href)

    print('Disambiguation links',disambiguation)
    infratio = count_total/count_need
    if infratio >=1:
        return f'{infratio}',True
    else:
        return f'{infratio}',False




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('page')
    parser.add_argument('--lang',default = 'en')
    args = parser.parse_args()
    your_lang = args.lang
    your_page = args.page
    check_req = counter_func(page = your_page, region = your_lang)
    return check_req

if __name__== '__main__':
	print(main())

