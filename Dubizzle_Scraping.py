# dubizzle_full_crawler_playwright.py
# -----------------------------------------------------------------------------
# Scrapes all car ads from Dubizzle Egypt using Playwright (headless browser).
# Accurate for JS-rendered pages.
#
# ▸ python Dubizzle_Scraping.py --pages all
#
# Requirements:
#     pip install playwright beautifulsoup4 tqdm
#     playwright install chromium

import re, csv, time, argparse, random, sys, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests

BASE = "https://www.dubizzle.com.eg"
LIST_URL = BASE + "/en/vehicles/cars-for-sale/?page={}"
OUT_FILE = "dubizzle_full_dataset.csv"
BACKUP_FILE = "backup_dubizzle.csv"
LIST_WORKERS = 5
AD_WORKERS = 5
FIELDS = [
    "source_url", "title", "price", "location", "date_posted",
    "fuel_type", "kilometers", "year", "body_type", "engine_capacity_cc",
    "transmission", "color", "doors", "seats", "payment_option", "brand",
    "model", "power_hp", "air_conditioning", "interior", "owners",
    "consumption", "extra_features"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# --------------------------- Helper Functions --------------------------- #

def fetch(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.text
    except:
        return None

def parse_list_page(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select('div[class*="_637fa00f"] a[href]')
    links = []
    for a in cards:
        href = a['href']
        if not href.startswith('http'):
            href = BASE + href
        links.append(href)
    return links

def gather_links(max_pages):
    links = []
    if max_pages == "all":
        page = 1
        while True:
            html = fetch(LIST_URL.format(page))
            new_links = parse_list_page(html) if html else []
            if not new_links:
                break
            links.extend(new_links)
            page += 1
    else:
        with ThreadPoolExecutor(max_workers=LIST_WORKERS) as pool:
            futures = {pool.submit(fetch, LIST_URL.format(p)): p for p in range(1, int(max_pages)+1)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Collecting Links"):
                html = fut.result()
                if html:
                    links.extend(parse_list_page(html))
    return list(set(links))

def scrape_ad_playwright(url):
    row = {k: None for k in FIELDS}
    row["source_url"] = url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=60000)
            page.wait_for_selector('h1', timeout=20000)
            html = page.content()
        except:
            browser.close()
            return row
        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    def grab(label):
        tag = soup.find('span', attrs={'aria-label': label})
        return tag.text.strip() if tag else None

    row['title'] = soup.select_one('h1').get_text(strip=True) if soup.select_one('h1') else None
    row['price'] = grab('Price')
    row['location'] = grab('Location')
    row['date_posted'] = grab('Creation date')

    highlights = soup.find('div', attrs={'aria-label': 'Highlighted Details'})
    if highlights:
        for box in highlights.select('div'):
            spans = box.find_all("span")
            if len(spans) == 2:
                label, val = spans[0].text.strip(), spans[1].text.strip()
                key_map = {
                    'Fuel Type': 'fuel_type', 'Kilometers': 'kilometers',
                    'Year': 'year', 'Body Type': 'body_type',
                    'Engine Capacity': 'engine_capacity_cc',
                    'Transmission Type': 'transmission', 'Color': 'color',
                    'Number of doors': 'doors', 'Number of seats': 'seats'
                }
                if label in key_map:
                    row[key_map[label]] = val

    details = soup.find('div', attrs={'aria-label': 'Details'})
    if details:
        spans = details.find_all('span')
        for i in range(0, len(spans)-1, 2):
            label, val = spans[i].text.strip(), spans[i+1].text.strip()
            key_map = {
                'Payment Options': 'payment_option', 'Brand': 'brand',
                'Model': 'model', 'Power (hp)': 'power_hp',
                'Air Conditioning': 'air_conditioning',
                'Interior': 'interior', 'Number of owners': 'owners',
                'Fuel Consumption': 'consumption',
                'Body Type': 'body_type', 'Color': 'color',
                'Number of doors': 'doors', 'Number of seats': 'seats',
                'Engine Capacity': 'engine_capacity_cc'
            }
            if label in key_map:
                row[key_map[label]] = val

    chips_parent = soup.find("div", class_=lambda c: c and "chip" in c)
    if chips_parent:
        chips = [c.get_text(" ", strip=True) for c in chips_parent.find_all("span")]
        if chips:
            row["extra_features"] = ", ".join(chips)

    return row

def write_csv(path, rows, header=False):
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if header:
            writer.writeheader()
        writer.writerows(rows)

def backup_csv(source, backup):
    try:
        shutil.copyfile(source, backup)
    except Exception as e:
        print(f"[!] Backup failed: {e}", file=sys.stderr)

# --------------------------- Main --------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="3", help="Number of pages to crawl or 'all'")
    args = parser.parse_args()

    print("Gathering links...")
    links = gather_links(args.pages)
    print(f"{len(links)} links collected.")

    random.shuffle(links)
    csv_path = Path(OUT_FILE)
    if csv_path.exists():
        csv_path.unlink()

    write_csv(csv_path, [], header=True)
    rows = []

    with ThreadPoolExecutor(max_workers=AD_WORKERS) as pool:
        futures = {pool.submit(scrape_ad_playwright, url): url for url in links}
        for idx, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Scraping Ads")):
            try:
                row = fut.result()
                rows.append(row)
                if len(rows) % 20 == 0:
                    write_csv(csv_path, rows)
                    backup_csv(csv_path, BACKUP_FILE)
                    rows.clear()
            except Exception as e:
                print(f"[!] Error: {e}", file=sys.stderr)

    if rows:
        write_csv(csv_path, rows)
        backup_csv(csv_path, BACKUP_FILE)

    print(f"Done. Dataset saved to {OUT_FILE}")

if __name__ == "__main__":
    main()