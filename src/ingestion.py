from playwright.sync_api import sync_playwright
import pandas as pd
import os
from urllib.parse import urljoin

CATALOG_URL = "https://www.shl.com/products/product-catalog/"
OUTPUT_PATH = "data/raw/shl_catalog_raw.csv"
FILTERED_OUTPUT_PATH = "data/raw/shl_catalog_filtered_out.csv"

DEBUG_HTML = "data/raw/debug_page.html"
DEBUG_PNG = "data/raw/debug_page.png"
PROFILE_DIR = "data/playwright_profile"

KEY_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(PROFILE_DIR, exist_ok=True)


def dump_debug(page, note: str):
    ensure_dirs()
    try:
        page.screenshot(path=DEBUG_PNG, full_page=True)
    except Exception:
        pass
    try:
        with open(DEBUG_HTML, "w", encoding="utf-8") as f:
            f.write(page.content())
    except Exception:
        pass

    try:
        title = page.title()
    except Exception:
        title = "N/A"

    print(f"[DEBUG] {note}")
    print(f"[DEBUG] url={page.url}")
    print(f"[DEBUG] title={title}")
    print(f"[DEBUG] saved {DEBUG_PNG} and {DEBUG_HTML}")


def try_accept_cookies(page) -> bool:
    for sel in (
        "button:has-text('Accept all')",
        "button:has-text('Accept All')",
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
        "button:has-text('OK')",
        "button:has-text('Got it')",
    ):
        try:
            btn = page.locator(sel).first
            if btn.count() > 0:
                btn.click(timeout=2000)
                page.wait_for_timeout(500)
                return True
        except Exception:
            pass
    return False


def pick_best_frame(page):
    best = page.main_frame
    best_count = -1
    for fr in page.frames:
        try:
            c = fr.locator("a").count()
        except Exception:
            c = 0
        if c > best_count:
            best = fr
            best_count = c
    return best, best_count


def yes_no(td) -> str:
    return "Yes" if td.locator("span.catalogue__circle.-yes").count() > 0 else "No"


def scrape_one_type(page, type_id: int, row_selector: str, id_attr: str, kind_label: str):
    records = []
    current_url = f"{CATALOG_URL}?type={type_id}"
    seen_urls = set()

    while current_url and current_url not in seen_urls:
        seen_urls.add(current_url)

        resp = page.goto(current_url, wait_until="domcontentloaded", timeout=60000)
        status = resp.status if resp else None
        print(f"[DEBUG] goto_status={status} type={type_id} url={current_url}")

        try:
            page.wait_for_selector("div.product-catalogue__list", timeout=30000)
        except Exception:
            dump_debug(page, "product-catalogue__list not found (page structure changed?)")
            break

        try_accept_cookies(page)

        # Prefer main page; fallback to the frame with most anchors if needed
        root = page
        if root.locator(row_selector).count() == 0:
            fr, _ = pick_best_frame(page)
            if fr.locator(row_selector).count() > 0:
                root = fr

        rows = root.locator(row_selector)
        row_count = rows.count()
        print(f"[DEBUG] type={type_id} rows_on_page={row_count}")

        for i in range(row_count):
            row = rows.nth(i)
            rid = row.get_attribute(id_attr) or ""

            tds = row.locator("td")
            if tds.count() < 4:
                continue

            a = tds.nth(0).locator("a").first
            try:
                name = (a.inner_text(timeout=2000) or "").strip()
            except Exception:
                name = ""

            href = a.get_attribute("href") or ""
            full_url = urljoin("https://www.shl.com", href)

            remote = yes_no(tds.nth(1))
            adaptive = yes_no(tds.nth(2))

            keys_loc = tds.nth(3).locator("span.product-catalogue__key")
            keys = []
            for k in range(keys_loc.count()):
                keys.append((keys_loc.nth(k).inner_text() or "").strip())

            key_names = [KEY_MAP.get(k, k) for k in keys if k]

            if not name and not href:
                continue

            records.append(
                {
                    "id": rid,
                    "catalog_type": kind_label,  # type_1_individual / type_2_job_solution
                    "assessment_name": name,
                    "url": full_url,
                    "remote_testing_support": remote,
                    "adaptive_irt": adaptive,
                    "test_type_keys": ", ".join([k for k in keys if k]),
                    "test_type": ", ".join(key_names),
                }
            )

        # Next page (for THIS type)
        next_link = root.locator(
            f"li.pagination__item.-arrow.-next a.pagination__arrow[href*='type={type_id}']"
        ).first
        if next_link.count() == 0:
            current_url = None
        else:
            current_url = urljoin("https://www.shl.com", next_link.get_attribute("href"))

    return records


def scrape_shl_catalog():
    ensure_dirs()

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            headless=False,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")

        # type=1: Individual Test Solutions (assessments)
        assessments = scrape_one_type(
            page,
            type_id=1,
            row_selector="tr[data-entity-id]",
            id_attr="data-entity-id",
            kind_label="type_1_individual",
        )

        # type=2: Pre-packaged Job Solutions (we will “filter out”)
        job_solutions = scrape_one_type(
            page,
            type_id=2,
            row_selector="tr[data-course-id]",
            id_attr="data-course-id",
            kind_label="type_2_job_solution",
        )

        df_assessments = pd.DataFrame(assessments).drop_duplicates(subset=["url"])
        df_filtered = pd.DataFrame(job_solutions).drop_duplicates(subset=["url"])

        df_assessments.to_csv(OUTPUT_PATH, index=False)
        print(f"Found {len(df_assessments) + len(df_filtered)} product cards")
        print(f"Saved {len(df_assessments)} assessments to {OUTPUT_PATH}")

        if len(df_filtered) > 0:
            df_filtered.to_csv(FILTERED_OUTPUT_PATH, index=False)
            print(f"Saved {len(df_filtered)} filtered-out items to {FILTERED_OUTPUT_PATH}")

        context.close()


if __name__ == "__main__":
    scrape_shl_catalog()