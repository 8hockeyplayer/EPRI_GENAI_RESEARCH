# Import standard libraries for file handling, timing, and database interaction
import os
import time
import sqlite3

# Import tools for automating a web browser (Selenium)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Import data processing and parsing libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import fitz  # Library used to read and extract text from PDFs


# ======================
# Configuration
# ======================

# Define where downloaded PDFs will be stored locally
DOWNLOAD_DIR = os.path.abspath("ferc_pdfs")

# Create the folder if it does not already exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Maximum number of pages to scrape from the website
MAX_PAGES = 500


# ======================
# Helper functions
# ======================

# Get a list of all files currently in the download folder
def get_current_files():
    return set(os.listdir(DOWNLOAD_DIR))


# Wait for a new PDF file to appear in the download folder after clicking download
def wait_for_new_pdf(files_before, timeout=20):
    end_time = time.time() + timeout

    # Keep checking until timeout is reached
    while time.time() < end_time:
        current_files = set(os.listdir(DOWNLOAD_DIR))
        new_files = current_files - files_before

        # Only keep files that are PDFs
        pdf_files = {f for f in new_files if f.lower().endswith(".pdf")}
        if pdf_files:
            return pdf_files

        time.sleep(0.5)

    # If nothing shows up in time, return empty
    return set()


# Extract a unique identifier ("accession number") from a table row on the webpage
def get_accession_from_row(row):
    try:
        cells = row.find_elements(By.TAG_NAME, "td")
        accession = cells[1].text.strip()
        return accession.replace(" ", "_")
    except Exception:
        return None


# Extract structured metadata (important fields) from a table row
def extract_row_metadata(row):
    try:
        cells = row.find_elements(By.TAG_NAME, "td")

        # Helper function to safely grab a column value
        def safe(idx):
            return cells[idx].text.strip() if idx < len(cells) else None

        return {
            "doc_type": safe(0),
            "accession": safe(1),
            "filed_date": safe(2),
            "issued_date": safe(3),
            "description": safe(5),
            "doc_category": safe(6),
            "access_level": safe(7),
        }

    except Exception as e:
        print(f"❌ Row extraction failed: {e}")
        return None


# Check if a PDF has already been downloaded (to avoid duplicates)
def pdf_already_downloaded(accession):
    if not accession:
        return False

    for fname in os.listdir(DOWNLOAD_DIR):
        if fname.startswith(accession) and fname.lower().endswith(".pdf"):
            return True

    return False


# Read a PDF file and extract all text page-by-page
def ocr_pdf_tables(pdf_path):
    rows = []

    # Open the PDF document
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    # Loop through each page of the PDF
    for page_num in range(total_pages):
        page = doc.load_page(page_num)

        # Extract text from the page
        text = page.get_text("text") or ""

        # Store extracted text along with page number
        rows.append({
            "page": page_num + 1,
            "block_type": "pdf_text",
            "content": text
        })

        # Print progress every 10 pages
        if (page_num + 1) % 10 == 0 or (page_num + 1) == total_pages:
            print(f"PDF text extraction progress: {page_num + 1}/{total_pages}")

    doc.close()

    # Convert results into a structured table (DataFrame)
    return pd.DataFrame(rows)


# ======================
# SQLite setup
# ======================

# Create (or connect to) a local database file
conn = sqlite3.connect("ocr_results.db")
cursor = conn.cursor()

# Create a table to store metadata about each PDF
cursor.execute("""
CREATE TABLE IF NOT EXISTS pdfs (
    pdf_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_file TEXT UNIQUE,
    accession TEXT,
    doc_type TEXT,
    filed_date TEXT,
    issued_date TEXT,
    description TEXT,
    doc_category TEXT,
    access_level TEXT
)
""")

# Create a table to store extracted text from PDFs
cursor.execute("""
CREATE TABLE IF NOT EXISTS ocr_data (
    pdf_id INTEGER,
    page INTEGER,
    block_type TEXT,
    content TEXT,
    FOREIGN KEY (pdf_id) REFERENCES pdfs(pdf_id)
)
""")

conn.commit()


# Process a single PDF: store metadata + extracted text into the database
def process_pdf_and_store(pdf_path, metadata, conn, cursor):
    pdf_file = os.path.basename(pdf_path)

    # Insert metadata into database (ignore if already exists)
    cursor.execute("""
    INSERT OR IGNORE INTO pdfs (
        pdf_file, accession, doc_type, filed_date,
        issued_date, description, doc_category, access_level
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pdf_file,
        metadata.get("accession"),
        metadata.get("doc_type"),
        metadata.get("filed_date"),
        metadata.get("issued_date"),
        metadata.get("description"),
        metadata.get("doc_category"),
        metadata.get("access_level")
    ))
    conn.commit()

    # Retrieve the unique ID assigned to this PDF
    cursor.execute(
        "SELECT pdf_id FROM pdfs WHERE pdf_file = ?",
        (pdf_file,)
    )
    pdf_id = cursor.fetchone()[0]

    # Extract text from the PDF
    df = ocr_pdf_tables(pdf_path)
    df["pdf_id"] = pdf_id

    # Store extracted text into database
    df.to_sql("ocr_data", conn, if_exists="append", index=False)

    # Delete the PDF after processing to save space
    os.remove(pdf_path)
    print(f"🧹 Deleted {pdf_file}")


# ======================
# Pagination helpers
# ======================

# XPath selectors used to navigate pages on the website
NEXT_BTN_XPATH = "//button[@aria-label='Next page']"
RANGE_LABEL_XPATH = "//div[contains(@class,'mat-paginator-range-label')]"


# Check if a "next page" button is available and clickable
def has_next_page(driver, timeout=5):
    try:
        btn = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, NEXT_BTN_XPATH))
        )
        return btn.is_enabled()
    except TimeoutException:
        return False


# Click the "next page" button and wait for the new page to load
def go_to_next_page(driver, timeout=15):
    wait = WebDriverWait(driver, timeout)

    wait.until(
        lambda d: d.find_element(By.XPATH, RANGE_LABEL_XPATH).text.strip() != ""
    )

    range_label = wait.until(
        EC.presence_of_element_located((By.XPATH, RANGE_LABEL_XPATH))
    )
    previous_text = range_label.text.strip()

    next_btn = wait.until(
        EC.presence_of_element_located((By.XPATH, NEXT_BTN_XPATH))
    )

    # Scroll to the button and click it
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'});", next_btn
    )

    wait.until(lambda d: next_btn.is_enabled())
    driver.execute_script("arguments[0].click();", next_btn)

    print(previous_text)

    # Wait until the page content changes
    wait.until(
        lambda d: (
            d.find_element(By.XPATH, RANGE_LABEL_XPATH).text.strip() != ""
            and d.find_element(By.XPATH, RANGE_LABEL_XPATH).text.strip() != previous_text
        )
    )

    new_range_label = wait.until(
        EC.presence_of_element_located((By.XPATH, RANGE_LABEL_XPATH))
    )
    print(new_range_label.text.strip())

    time.sleep(20)


# ======================
# Selenium setup
# ======================

# Configure Chrome browser options (runs in background/headless mode)
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

# Set preferences for automatic downloading of PDFs
prefs = {
    "download.default_directory": DOWNLOAD_DIR,
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True,
    "profile.default_content_setting_values.automatic_downloads": 1,
}
chrome_options.add_experimental_option("prefs", prefs)

# Launch Chrome browser using Selenium
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)

# Open the FERC search page
wait = WebDriverWait(driver, 30)
driver.get("https://elibrary.ferc.gov/eLibrary/search")

# Click the search button to load results
search_button = wait.until(
    EC.element_to_be_clickable((By.ID, "submit"))
)
search_button.click()

# Wait until the results table appears
wait.until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
)


# ======================
# Main scraping loop
# ======================

# Loop through rows on the current page and download PDFs
def download_all_pdfs_skip_existing():
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

    # NOTE: Currently hardcoded to only process 15 rows
    total = 15
    print(f"✅ Rows found: {total}")

    downloaded = []

    for i in range(total):
        print(f"\n➡️ Processing row {i + 1} of {total}")

        try:
            rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            row = rows[i]

            # Extract metadata for the row
            metadata = extract_row_metadata(row)
            accession = metadata["accession"] if metadata else None

            # Skip if already downloaded
            if accession and pdf_already_downloaded(accession):
                print(f"⏭️ Already downloaded ({accession}). Skipping.")
                continue

            # Scroll to the row
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});",
                row
            )
            time.sleep(0.4)

            # Find and click the "Generate PDF" button
            pdf_button = row.find_element(
                By.XPATH, ".//button[contains(., 'Generate PDF')]"
            )

            # Track files before download
            files_before = set(os.listdir(DOWNLOAD_DIR))
            driver.execute_script("arguments[0].click();", pdf_button)

            # Wait for new PDF to appear
            new_files = wait_for_new_pdf(files_before, timeout=20)

            if not new_files:
                print("❌ No new PDF detected. Skipping.")
                continue

            # Store downloaded file names + metadata
            for pdf_file in new_files:
                downloaded.append((pdf_file, metadata))

        except Exception as e:
            print(f"❌ Failed on row {i + 1}: {e}")

    # Process all downloaded PDFs
    for pdf_file, metadata in downloaded:
        pdf_path = os.path.join(DOWNLOAD_DIR, pdf_file)
        process_pdf_and_store(pdf_path, metadata, conn, cursor)

    print(len(downloaded))


# Track how many pages have been processed
pages = 0

# Loop through multiple pages of results
while pages < MAX_PAGES:
    download_all_pdfs_skip_existing()

    print("🧹 Cleaning up Angular overlays...")

    # Remove UI elements that may block clicks
    driver.execute_script("""
        document.querySelectorAll('.cdk-overlay-container').forEach(e => e.remove());
        document.querySelectorAll('.cdk-overlay-pane').forEach(e => e.remove());
        document.querySelectorAll('.cdk-overlay-backdrop').forEach(e => e.remove());
        document.body.classList.remove('cdk-global-scrollblock');
        document.body.style.overflow = 'auto';
    """)
    time.sleep(1.5)

    # Stop if no more pages
    if not has_next_page(driver):
        print("✅ No more pages.")
        break

    # Move to next page
    print("➡️ Moving to next page...")
    go_to_next_page(driver)
    pages += 1


# Close database and browser when done
conn.close()
driver.quit()

print("🏁 Done.")