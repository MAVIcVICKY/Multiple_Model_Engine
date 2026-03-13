"""
Django management command to ingest Flickr8k images and captions into Pinecone.

Usage:
    python manage.py ingest_data

This command:
1. Reads captions.csv and groups captions by image (takes the first caption per image).
2. For each image in data/images/:
   a) Embeds the image using Gemini and stores the vector in Pinecone.
   b) Embeds the first caption text and stores that vector in Pinecone.
3. Prints progress every 50 images and a final summary.
"""

import time
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings

from core.embed import embed_image, embed_text
from core.pinecone_db import store_vector


class Command(BaseCommand):
    help = "Ingest Flickr8k images and captions into Pinecone"

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Limit the number of images to ingest (0 = all)",
        )
        parser.add_argument(
            "--skip",
            type=int,
            default=0,
            help="Number of images to skip from the beginning",
        )

    def handle(self, *args, **options):
        limit = options["limit"]
        skip = options["skip"]

        data_dir = settings.BASE_DIR / "data"
        images_dir = data_dir / "images"
        captions_file = data_dir / "captions.csv"

        # ── Step 1: Load captions ──────────────────────────────────────
        self.stdout.write(self.style.NOTICE("📂 Loading captions.csv..."))

        if not captions_file.exists():
            self.stderr.write(
                self.style.ERROR(f"❌ captions.csv not found at {captions_file}")
            )
            return

        df = pd.read_csv(captions_file)
        self.stdout.write(f"   Found {len(df)} caption rows")

        # Group by image filename and take the first caption
        captions_map = df.groupby("image")["caption"].first().to_dict()
        self.stdout.write(f"   Unique images with captions: {len(captions_map)}")

        # ── Step 2: Get list of image files ────────────────────────────
        if not images_dir.exists():
            self.stderr.write(
                self.style.ERROR(f"❌ Images directory not found at {images_dir}")
            )
            return

        image_files = sorted(images_dir.glob("*.jpg"))
        total_images = len(image_files)
        self.stdout.write(f"   Found {total_images} jpg files in {images_dir}")

        # Apply skip and limit
        if skip > 0:
            image_files = image_files[skip:]
            self.stdout.write(f"   Skipping first {skip} images")

        if limit > 0:
            image_files = image_files[:limit]
            self.stdout.write(f"   Limiting to {limit} images")

        target_count = len(image_files)
        self.stdout.write(
            self.style.SUCCESS(f"\n🚀 Starting ingestion of {target_count} images...\n")
        )

        # ── Step 3: Ingest each image ─────────────────────────────────
        ingested = 0
        errors = 0

        for i, image_path in enumerate(image_files, start=1):
            filename = image_path.name

            try:
                # Get caption for this image
                caption_text = captions_map.get(filename, "No caption available")

                # a) Embed the image directly
                img_vector = embed_image(str(image_path))
                store_vector(
                    id=f"img_{filename}",
                    vector=img_vector,
                    metadata={
                        "type": "image",
                        "source": "image",
                        "path": str(image_path),
                        "filename": filename,
                        "caption": caption_text,
                    },
                )

                # b) Embed the caption text
                txt_vector = embed_text(caption_text)
                store_vector(
                    id=f"txt_{filename}",
                    vector=txt_vector,
                    metadata={
                        "type": "image",
                        "source": "caption",
                        "path": str(image_path),
                        "filename": filename,
                        "caption": caption_text,
                    },
                )

                ingested += 1

                # Progress reporting every 50 images
                if i % 50 == 0 or i == target_count:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✓ {i}/{target_count} ingested..."
                        )
                    )

            except Exception as e:
                errors += 1
                self.stderr.write(
                    self.style.WARNING(f"⚠ Error processing {filename}: {e}")
                )

            # Rate limiting: sleep between each image
            time.sleep(0.5)

        # ── Step 4: Final summary ──────────────────────────────────────
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write(
            self.style.SUCCESS(
                f"✅ Ingestion complete. {ingested} images ingested."
            )
        )
        if errors:
            self.stdout.write(
                self.style.WARNING(f"⚠  {errors} errors encountered.")
            )
        self.stdout.write("=" * 50)
