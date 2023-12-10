#! bin/python3

import requests
from bs4 import BeautifulSoup
import re
from typing import Union
import argparse

def count_wiki_links(page:str , region:str)-> Union[str,bool]:

    your_link = f"https://{region}.wikipedia.org/wiki/{page}"
    response = requests.get(your_link)
    if response.status_code != requests.codes.ok:
            return False

    soup = BeautifulSoup(response.content, 'html.parser')
    wiki_links = soup.find_all('a', href=re.compile(r'^/wiki/[^:]+$'))
    print(wiki_links)
    summary = len(wiki_links)
    wiki_links_count = 0
    disambiguation_links = []

    for link in wiki_links: 
        href = link.get('href')
        if href.startswith('/wiki/') and not href.startswith('/wiki/File:') and not href.startswith('/wiki/Template:') and not href.startswith('/wiki/Wikipedia:'):
            wiki_links_count += 1
            if href.startswith(your_link):
                disambiguation_links.append(href)
    
    infratio = wiki_links_count/summary
    if infratio > 0.01:
        return f'{infratio} True'
    else:
        if len(disambiguation_links) != 0:
            return disambiguation_links
        else:
            return f'{infratio} < 0.01 - False'

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('page')
        parser.add_argument('--lang',default = 'en')
        args = parser.parse_args()
        your_lang = args.lang
        your_page = args.page
        check_req = count_wiki_links(page = your_page, region = your_lang)
        return check_req

if __name__== '__main__':
    print(main())
