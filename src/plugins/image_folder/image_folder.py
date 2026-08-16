from plugins.base_plugin.base_plugin import BasePlugin
from PIL import Image, ImageOps, ImageColor, ImageDraw, ImageFont, ImageFilter
from PIL.ExifTags import Base as ExifBase, IFD as ExifIFD
from datetime import datetime
import logging
import os
import random

from utils.app_utils import get_font
from utils.image_utils import pad_image_blur

logger = logging.getLogger(__name__)

# Font size as a fraction of the image height, keyed by the "date size" setting.
DATE_SIZE_DIVISORS = {
    "small": 24,
    "medium": 17,
    "large": 12,
}
DEFAULT_DATE_SIZE = "medium"

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

def _region_busyness(img, box):
    """Return an edge-energy score for a region; lower means flatter/less busy."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    region = img.crop((x0, y0, x1, y1)).convert("L")
    edges = region.filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    # Weighted sum of edge intensities, normalized by area so region sizes compare fairly.
    energy = sum(intensity * count for intensity, count in enumerate(histogram))
    area = max(1, (x1 - x0) * (y1 - y0))
    return energy / area

def _choose_least_busy_corner(img, box_width, box_height, margin):
    """Pick the corner whose badge footprint covers the least busy area of the image."""
    width, height = img.size
    corners = {
        "top-left": (margin, margin),
        "top-right": (width - margin - box_width, margin),
        "bottom-left": (margin, height - margin - box_height),
        "bottom-right": (width - margin - box_width, height - margin - box_height),
    }
    best_corner = None
    best_score = None
    for name, (x0, y0) in corners.items():
        score = _region_busyness(img, (x0, y0, x0 + box_width, y0 + box_height))
        logger.debug(f"Corner {name} busyness={score:.2f}")
        if best_score is None or score < best_score:
            best_score, best_corner = score, (x0, y0)
    return best_corner

def overlay_date_taken(img, date_text, size=DEFAULT_DATE_SIZE):
    """Draw the date text inside a black frame in the least busy corner of the image."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    divisor = DATE_SIZE_DIVISORS.get(size, DATE_SIZE_DIVISORS[DEFAULT_DATE_SIZE])
    font_size = max(14, height // divisor)
    font = get_font("Jost", font_size, "bold") or ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), date_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    pad = max(6, font_size // 3)
    margin = max(8, font_size // 2)

    box_width = text_width + 2 * pad
    box_height = text_height + 2 * pad

    box_x0, box_y0 = _choose_least_busy_corner(img, box_width, box_height, margin)
    box_x1 = box_x0 + box_width
    box_y1 = box_y0 + box_height

    radius = max(4, pad // 2)
    draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=radius, fill="black")

    # Center the text within the black frame, compensating for the font bbox offset.
    text_x = box_x0 + pad - text_bbox[0]
    text_y = box_y0 + pad - text_bbox[1]
    draw.text((text_x, text_y), date_text, font=font, fill="white")

    return img

class ImageFolder(BasePlugin):
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
        image_url = random.choice(image_files)
        logger.info(f"Selected random image: {os.path.basename(image_url)}")
        logger.debug(f"Full path: {image_url}")

        # Check padding options
        use_padding = settings.get('padImage') == "true"
        background_option = settings.get('backgroundOption', 'blur')
        show_date_taken = settings.get('showDateTaken') == "true"
        date_format = settings.get('dateFormat') or DEFAULT_DATE_FORMAT
        if date_format not in SUPPORTED_DATE_FORMATS:
            logger.warning(f"Unsupported date format {date_format!r}, falling back to default")
            date_format = DEFAULT_DATE_FORMAT
        date_size = settings.get('dateSize') or DEFAULT_DATE_SIZE
        if date_size not in DATE_SIZE_DIVISORS:
            logger.warning(f"Unsupported date size {date_size!r}, falling back to default")
            date_size = DEFAULT_DATE_SIZE
        logger.debug(f"Settings: pad_image={use_padding}, background_option={background_option}, show_date_taken={show_date_taken}, date_format={date_format!r}, date_size={date_size!r}")

        try:
            # Use adaptive loader for memory-efficient processing
            # Load without auto-resize first to handle padding options
            # Note: Loader automatically handles EXIF orientation correction
            img = self.image_loader.from_file(image_url, dimensions, resize=False)

            if not img:
                raise RuntimeError("Failed to load image from file")

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

            if show_date_taken:
                date_text = get_date_taken(image_url, date_format)
                if date_text:
                    logger.info(f"Overlaying date taken: {date_text}")
                    img = overlay_date_taken(img, date_text, date_size)
                else:
                    logger.info("Date taken overlay enabled but no capture date found in image")

            return img
        except Exception as e:
            logger.error(f"Error loading image from {image_url}: {e}")
            raise RuntimeError("Failed to load image, please check logs.")

        logger.info("=== Image Folder Plugin: Image generation complete ===")
        return img
