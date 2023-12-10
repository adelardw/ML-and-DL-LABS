#! /bin/python3


import requests
import argparse
import re

def go_to_the_site(page: str ,region: str)-> str:

    """
    Manual:
    page: name of directory after /wiki/
    region: country name
    return str type message
    """
    try:
        your_link = f"https://{region}.wikipedia.org/wiki/{page}"
        rq = requests.get(your_link)
        if rq.status_code != requests.codes.ok:
            return "Oups, your page has not defined.Check your page's name"
        if rq.status_code  == requests.codes.ok:
            return "Good!"
    except:
        your_link = "https://en.wikipedia.org/wiki/"
        request = requests.get(your_link)
        return "You returned on basic Wiki page.Please,check you region"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('page')
    parser.add_argument('--lang',default = 'en')
    args = parser.parse_args()
    your_lang = args.lang
    your_page = args.page
    check_req = go_to_the_site(page = your_page, region = your_lang)
    return check_req

if __name__== '__main__':
	print(main())
