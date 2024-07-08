import csv
import requests
from bs4 import BeautifulSoup


def get_news_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        item_text = soup.find('div', itemprop='articleBody')

        if item_text:
            p_tags = item_text.find_all('p')
            content = ""

            for p_tag in p_tags:
                content += p_tag.text + '\n'

            return content.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching content for {url}: {e}")

    return None


def crawl_and_save_to_csv(base_url, sitemap_index_url, csv_filename, target_count=2500):
    response = requests.get(sitemap_index_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'xml')
        sitemap_tags = soup.find_all('loc')
        data = []
        count = 0

        for index, sitemap_tag in enumerate(sitemap_tags, start=1):
            if count >= target_count:
                break

            sitemap_url = sitemap_tag.text
            response = requests.get(sitemap_url)

            if response.status_code == 200:
                sitemap_soup = BeautifulSoup(response.text, 'xml')
                url_tags = sitemap_soup.find_all('url')

                for url_tag in url_tags:
                    if count >= target_count:
                        break

                    loc = url_tag.find('loc').text
                    title_tag = url_tag.find('title')

                    if title_tag:
                        title = title_tag.text
                    else:
                        title = f'Untitled - {loc}'

                    content = get_news_content(loc)

                    if content:
                        data.append({
                            'Index': index,
                            'Content': content,
                            'Category': 'ورزشی',
                            'URL': loc
                        })
                        count += 1

        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Index', 'Content', 'Category', 'URL']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f'{count} data points have been successfully saved to {csv_filename}')


base_url = 'https://www.khabarvarzeshi.com/'
sitemap_index_url = 'https://www.khabarvarzeshi.com/sitemap/all/sitemap.xml'

csv_filename_3000 = 'news_data_3000.csv'

crawl_and_save_to_csv(base_url, sitemap_index_url, csv_filename_3000, target_count=3000)

