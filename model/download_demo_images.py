from pathlib import Path
from io import BytesIO
import time

import requests
from PIL import Image
from ddgs import DDGS


ROOT = Path("data/demo_finetune")

# Calm is already downloaded, so continue only with these.
QUERIES = {
    "moderate": [
        "moderate ocean waves open sea whitecaps",
        "moderate sea swell ocean waves",
    ],

    "rough": [
        "rough sea large waves open ocean whitecaps",
        "rough ocean strong waves",
    ],

    "very_rough": [
        "very rough sea huge storm waves",
        "violent storm ocean massive waves",
    ],
}

IMAGES_PER_CLASS = 24

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}


def download_class(class_name, queries):
    folder = ROOT / class_name
    folder.mkdir(parents=True, exist_ok=True)

    existing_images = list(folder.glob("*.jpg"))
    downloaded = len(existing_images)

    seen_urls = set()

    print(f"\n========== {class_name.upper()} ==========")
    print(f"Existing images: {downloaded}")

    if downloaded >= IMAGES_PER_CLASS:
        print("Already complete.")
        return

    for query in queries:
        if downloaded >= IMAGES_PER_CLASS:
            break

        print("Searching:", query)

        try:
            results = DDGS(timeout=8).images(
                query,
                safesearch="moderate",
                max_results=35,
            )

        except Exception as e:
            print("Search skipped:", e)
            continue

        for result in results:
            if downloaded >= IMAGES_PER_CLASS:
                break

            url = result.get("image")

            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)

            try:
                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=8,
                )

                response.raise_for_status()

                image = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                # Skip tiny thumbnails
                if image.width < 400:
                    continue

                if image.height < 250:
                    continue

                ratio = image.width / image.height

                # Skip extremely tall/wide images
                if ratio < 0.8 or ratio > 3.0:
                    continue

                downloaded += 1

                output_path = folder / (
                    f"{class_name}_{downloaded:02d}.jpg"
                )

                image.save(
                    output_path,
                    "JPEG",
                    quality=90,
                )

                print(
                    f"[{downloaded}/{IMAGES_PER_CLASS}] "
                    f"{output_path}"
                )

                time.sleep(0.15)

            except Exception:
                continue

    print(
        f"{class_name}: {downloaded} images available"
    )


for class_name, queries in QUERIES.items():
    try:
        download_class(
            class_name,
            queries,
        )

    except KeyboardInterrupt:
        print(
            f"\nSearch interrupted while processing "
            f"{class_name}. Continuing..."
        )
        continue

    except Exception as e:
        print(
            f"\nUnexpected error for {class_name}:",
            e,
        )
        continue


print("\n======================================")
print("DOWNLOAD FINISHED")
print("======================================")

all_classes = [
    "calm",
    "moderate",
    "rough",
    "very_rough",
]

for class_name in all_classes:
    folder = ROOT / class_name

    count = len(
        list(folder.glob("*.jpg"))
    )

    print(
        f"{class_name}: {count} images"
    )