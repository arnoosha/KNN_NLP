import requests
from bs4 import BeautifulSoup
import csv


def crawl_and_save_data(url, target_count):
    data_count = 0
    page_number = 1

    with open('c.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['index', 'category', 'content', 'url']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while data_count < target_count:
            page_url = f"{url}{page_number}"
            response = requests.get(page_url)
            print(page_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                news_elements = soup.find_all('li', class_='media py-3 border-bottom align-items-start')

                for element in news_elements:
                    title_element = element.find('a', class_='col-5 px-0 mb-2 mb-sm-0')
                    if title_element:
                        title = title_element.text.strip()
                        link = title_element['href']

                        news_response = requests.get(f"https://www.farsnews.ir{link}")
                        if news_response.status_code == 200:
                            news_soup = BeautifulSoup(news_response.text, 'html.parser')
                            content_element = news_soup.find('div', id='nt-body-ck', itemprop='articleBody')
                            if content_element:
                                content_paragraphs = content_element.find_all('p')
                                content = " ".join([p.text.strip() for p in content_paragraphs])

                                data_count += 1
                                writer.writerow({'index': data_count, 'category': 'سیاسی', 'content': content,
                                                 'url': f"https://www.farsnews.ir{link}"})
                                print(f"News {data_count} has been crawled.")

                page_number += 1
            else:
                print(f"Failed to retrieve the page. Status code: {response.status_code}")


target_data_count = 3000

crawl_url = "https://www.farsnews.ir/politics?p="
crawl_and_save_data(crawl_url, target_data_count)
