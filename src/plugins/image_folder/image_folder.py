from plugins.base_plugin.base_plugin import BasePlugin
from PIL import Image, ImageOps, ImageColor, ImageDraw, ImageFont
from PIL.ExifTags import Base as ExifBase, IFD as ExifIFD
from datetime import datetime
import logging
import os
import random

from utils.app_utils import get_font
from utils.image_utils import pad_image_blur

logger = logging.getLogger(__name__)

def list_files_in_folder(folder_path):
    """Return a list of image file paths in the given folder, excluding hidden files."""
    image_extensions = ('.avif', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heif', '.heic')
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(image_extensions) and not f.startswith('.'):
                image_files.append(os.path.join(root, f))

    return image_files

DEFAULT_DATE_FORMAT = "%b %d, %Y"

# Supported strftime patterns for the date-taken overlay.
SUPPORTED_DATE_FORMATS = (
    "%b %d, %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
)

def _parse_exif_date(value, date_format=DEFAULT_DATE_FORMAT):
    """Parse an EXIF date string ('YYYY:MM:DD HH:MM:SS') into a formatted display string."""
    if not value:
        return None
    try:
        parsed = datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
        return parsed.strftime(date_format)
    except (ValueError, TypeError):
        logger.debug(f"Unparseable EXIF date: {value!r}")
        return None

def get_date_taken(image_path, date_format=DEFAULT_DATE_FORMAT):
    """Return the photo's capture date as a formatted string, or None if unavailable."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
    except Exception as e:
        logger.debug(f"Could not read EXIF from {image_path}: {e}")
        return None

    if not exif:
        return None

    # DateTimeOriginal / DateTimeDigitized live in the Exif sub-IFD; DateTime is in the base IFD.
    try:
        exif_ifd = exif.get_ifd(ExifIFD.Exif)
    except Exception:
        exif_ifd = {}

    candidates = (
        exif_ifd.get(ExifBase.DateTimeOriginal.value),
        exif_ifd.get(ExifBase.DateTimeDigitized.value),
        exif.get(ExifBase.DateTime.value),
    )
    for value in candidates:
        formatted = _parse_exif_date(value, date_format)
        if formatted:
            return formatted

    return None

def overlay_date_taken(img, date_text):
    """Draw the date text inside a black frame in the bottom-right corner of the image."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    font_size = max(14, height // 24)
    font = get_font("Jost", font_size, "bold") or ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), date_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    pad = max(6, font_size // 3)
    margin = max(8, font_size // 2)

    box_width = text_width + 2 * pad
    box_height = text_height + 2 * pad
    box_x1 = width - margin
    box_y1 = height - margin
    box_x0 = box_x1 - box_width
    box_y0 = box_y1 - box_height

    radius = max(4, pad // 2)
    draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=radius, fill="black")

    # Center the text within the black frame, compensating for the font bbox offset.
    text_x = box_x0 + pad - text_bbox[0]
    text_y = box_y0 + pad - text_bbox[1]
    draw.text((text_x, text_y), date_text, font=font, fill="white")

    return img

class ImageFolder(BasePlugin):
    def _render_image(self, image_url, dimensions, settings):
        """Load a single image and fit/pad it to the given dimensions."""
        # Use adaptive loader for memory-efficient processing
        # Load without auto-resize first to handle padding options
        # Note: Loader automatically handles EXIF orientation correction
        img = self.image_loader.from_file(image_url, dimensions, resize=False)

        if not img:
            raise RuntimeError("Failed to load image from file")

        use_padding = settings.get('padImage') == "true"
        background_option = settings.get('backgroundOption', 'blur')

        if use_padding:
            logger.debug(f"Applying padding with {background_option} background")
            if background_option == "blur":
                img = pad_image_blur(img, dimensions)
            else:
                background_color = ImageColor.getcolor(settings.get('backgroundColor') or "white", img.mode)
                img = ImageOps.pad(img, dimensions, color=background_color, method=Image.Resampling.LANCZOS)
        else:
            # No padding requested, scale to fit dimensions (crop to preserve aspect ratio)
            logger.debug(f"Scaling to fit dimensions: {dimensions[0]}x{dimensions[1]}")
            img = ImageOps.fit(img, dimensions, method=Image.LANCZOS)

        return img

    def generate_image(self, settings, device_config):
        logger.info("=== Image Folder Plugin: Starting image generation ===")

        folder_path = settings.get('folder_path')
        if not folder_path:
            logger.error("No folder path provided in settings")
            raise RuntimeError("Folder path is required.")

        if not os.path.exists(folder_path):
            logger.error(f"Folder does not exist: {folder_path}")
            raise RuntimeError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            logger.error(f"Path is not a directory: {folder_path}")
            raise RuntimeError(f"Path is not a directory: {folder_path}")

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]
            logger.debug(f"Vertical orientation detected, dimensions: {dimensions[0]}x{dimensions[1]}")

        logger.info(f"Scanning folder: {folder_path}")
        image_files = list_files_in_folder(folder_path)

        if not image_files:
            logger.warning(f"No image files found in folder: {folder_path}")
            raise RuntimeError(f"No image files found in folder: {folder_path}")

        logger.debug(f"Found {len(image_files)} image file(s) in folder")

        grid_mode = settings.get('gridMode') == "true"
        use_padding = settings.get('padImage') == "true"
        background_option = settings.get('backgroundOption', 'blur')
        show_date_taken = settings.get('showDateTaken') == "true"
        date_format = settings.get('dateFormat') or DEFAULT_DATE_FORMAT
        if date_format not in SUPPORTED_DATE_FORMATS:
            logger.warning(f"Unsupported date format {date_format!r}, falling back to default")
            date_format = DEFAULT_DATE_FORMAT
        logger.debug(f"Settings: grid_mode={grid_mode}, pad_image={use_padding}, background_option={background_option}, show_date_taken={show_date_taken}, date_format={date_format!r}")

        try:
            if grid_mode:
                return self._generate_grid(image_files, dimensions, settings)

            image_url = random.choice(image_files)
            logger.info(f"Selected random image: {os.path.basename(image_url)}")
            logger.debug(f"Full path: {image_url}")

            img = self._render_image(image_url, dimensions, settings)

            if show_date_taken:
                date_text = get_date_taken(image_url, date_format)
                if date_text:
                    logger.info(f"Overlaying date taken: {date_text}")
                    img = overlay_date_taken(img, date_text)
                else:
                    logger.info("Date taken overlay enabled but no capture date found in image")

            logger.info("=== Image Folder Plugin: Image generation complete ===")
            return img
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise RuntimeError("Failed to load image, please check logs.")

    def _generate_grid(self, image_files, dimensions, settings):
        """Compose 4 random images into a 2x2 grid filling the display."""
        width, height = dimensions
        cell_dimensions = (width // 2, height // 2)

        # Pick 4 images without repeats when possible, otherwise allow repeats.
        if len(image_files) >= 4:
            selected = random.sample(image_files, 4)
        else:
            selected = random.choices(image_files, k=4)
        logger.info(f"Grid mode: selected {[os.path.basename(p) for p in selected]}")

        canvas = Image.new("RGB", (width, height), "white")
        # Top-left, top-right, bottom-left, bottom-right
        positions = [
            (0, 0),
            (width - cell_dimensions[0], 0),
            (0, height - cell_dimensions[1]),
            (width - cell_dimensions[0], height - cell_dimensions[1]),
        ]

        for image_url, position in zip(selected, positions):
            cell = self._render_image(image_url, cell_dimensions, settings)
            canvas.paste(cell.convert("RGB"), position)

        logger.info("=== Image Folder Plugin: Grid image generation complete ===")
        return canvas
