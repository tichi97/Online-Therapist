from bs4 import BeautifulSoup
import requests


def anx():
    source = requests.get(
        'https://www.healthline.com/health/how-to-calm-anxiety#1').text

    soup = BeautifulSoup(source, 'lxml')
    article = soup.find('article')
    h = []
    par = []
    for a in article.find_all('h2'):
        x = a.text
        x = x[2:]
        h.append(x)
    for a in article.find_all('div'):
        if a.p:
            par.append(a.p.text)

    source = requests.get(
        'https://www.webmd.com/mental-health/features/ways-to-reduce-anxiety').text

    soup2 = BeautifulSoup(source, 'lxml')
    article = soup2.find('div', class_="article-body")
    h2 = []
    par2 = []
    for s in article.find_all('section'):
        if s.h2 and s.p:
            x = s.h2.text
            x = x[2:]
            h2.append(x)
            par2.append(s.p.text)

    anxiety = h + h2
    return anxiety


def sad():
    source = requests.get(
        'https://www.gundersenhealth.org/health-wellness/live-happy/healthy-ways-to-deal-with-sadness/').text

    soup = BeautifulSoup(source, 'lxml')
    article = soup.find('div', class_='cmsPageContent')

    par = []
    for p in article.find_all('li'):
        par.append(p.text)

    return par


def angry():
    source = requests.get(
        'https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/anger-management/art-20045434').text

    soup = BeautifulSoup(source, 'lxml')
    article = soup.find('div', id='main-content')

    h = []
    for hd in article.find_all('h3'):
        x = hd.text
        x = x[2:]
        h.append(x)

    return h


print(angry())
# print(h2)
# print(par2)
# print(len(par))
# print(len(h))
# print()
# print(len(par2))
# print(len(h2))
# print(anxiety)
