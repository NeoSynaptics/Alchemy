"""EXIF parser — extracts date_taken, GPS, camera info from photos.

Uses Pillow (already a project dependency) for EXIF reading.
Handles iPhone-specific EXIF tags including HEIC via fallback.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# EXIF tag IDs
_TAG_DATETIME_ORIGINAL = 36867    # DateTimeOriginal (when photo was taken)
_TAG_DATETIME_DIGITIZED = 36868   # DateTimeDigitized
_TAG_DATETIME = 306               # DateTime (file modified)
_TAG_GPS_INFO = 34853             # GPSInfo
_TAG_MAKE = 271                   # Camera make
_TAG_MODEL = 272                  # Camera model
_TAG_ORIENTATION = 274            # Image orientation


@dataclass
class PhotoMetadata:
    """Extracted metadata from a photo file."""
    date_taken: float | None = None      # Unix timestamp
    latitude: float | None = None
    longitude: float | None = None
    camera_make: str = ""
    camera_model: str = ""
    orientation: int = 1
    width: int = 0
    height: int = 0

    @property
    def has_location(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def to_meta_dict(self) -> dict:
        """Convert to dict suitable for timeline event meta_json."""
        d: dict = {}
        if self.date_taken:
            d["date_taken"] = self.date_taken
        if self.has_location:
            d["gps"] = {"lat": self.latitude, "lon": self.longitude}
        if self.camera_make:
            d["camera"] = f"{self.camera_make} {self.camera_model}".strip()
        if self.width and self.height:
            d["dimensions"] = f"{self.width}x{self.height}"
        d["orientation"] = self.orientation
        return d


def _parse_datetime(dt_str: str) -> float | None:
    """Parse EXIF datetime string like '2025:12:25 14:30:00' to unix ts."""
    if not dt_str or not isinstance(dt_str, str):
        return None
    dt_str = dt_str.strip().strip("\x00")
    if not dt_str:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _gps_to_decimal(gps_data: dict) -> tuple[float | None, float | None]:
    """Convert EXIF GPS IFD to decimal lat/lon."""
    try:
        def _dms_to_decimal(dms, ref: str) -> float:
            # dms is typically ((d, 1), (m, 1), (s, 100)) or IFDRational objects
            if hasattr(dms[0], 'numerator'):
                # IFDRational
                d = float(dms[0])
                m = float(dms[1])
                s = float(dms[2])
            elif isinstance(dms[0], tuple):
                d = dms[0][0] / dms[0][1]
                m = dms[1][0] / dms[1][1]
                s = dms[2][0] / dms[2][1]
            else:
                d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
            decimal = d + m / 60.0 + s / 3600.0
            if ref in ("S", "W"):
                decimal = -decimal
            return decimal

        lat_ref = gps_data.get(1, "N")  # GPSLatitudeRef
        lat = gps_data.get(2)           # GPSLatitude
        lon_ref = gps_data.get(3, "E")  # GPSLongitudeRef
        lon = gps_data.get(4)           # GPSLongitude

        if lat and lon:
            return _dms_to_decimal(lat, lat_ref), _dms_to_decimal(lon, lon_ref)
    except Exception:
        logger.debug("GPS parsing failed", exc_info=True)
    return None, None


def extract_metadata(photo_path: Path) -> PhotoMetadata:
    """Extract EXIF metadata from a photo file.

    Handles JPEG and PNG via Pillow. HEIC files need the file's
    modification time as fallback since Pillow can't read HEIC EXIF
    without pillow-heif (not a dependency).
    """
    meta = PhotoMetadata()

    try:
        from PIL import Image

        if photo_path.suffix.lower() in (".heic", ".heif"):
            # Try pillow-heif if available, otherwise use file mtime
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
                img = Image.open(photo_path)
            except ImportError:
                # No HEIC support — use file modification time
                stat = photo_path.stat()
                meta.date_taken = stat.st_mtime
                logger.debug("HEIC without pillow-heif, using mtime for %s", photo_path.name)
                return meta
        else:
            img = Image.open(photo_path)

        meta.width, meta.height = img.size

        exif = img.getexif()
        if not exif:
            # No EXIF — use file mtime
            meta.date_taken = photo_path.stat().st_mtime
            return meta

        # Date taken (prefer DateTimeOriginal > DateTimeDigitized > DateTime)
        for tag in (_TAG_DATETIME_ORIGINAL, _TAG_DATETIME_DIGITIZED, _TAG_DATETIME):
            val = exif.get(tag)
            if val:
                ts = _parse_datetime(str(val))
                if ts:
                    meta.date_taken = ts
                    break

        # Camera info
        meta.camera_make = str(exif.get(_TAG_MAKE, "")).strip("\x00").strip()
        meta.camera_model = str(exif.get(_TAG_MODEL, "")).strip("\x00").strip()
        meta.orientation = exif.get(_TAG_ORIENTATION, 1)

        # GPS
        gps_ifd = exif.get_ifd(_TAG_GPS_INFO) if hasattr(exif, 'get_ifd') else exif.get(_TAG_GPS_INFO)
        if gps_ifd and isinstance(gps_ifd, dict):
            meta.latitude, meta.longitude = _gps_to_decimal(gps_ifd)

        img.close()

    except Exception:
        logger.warning("EXIF extraction failed for %s", photo_path.name, exc_info=True)
        # Fallback to file modification time
        try:
            meta.date_taken = photo_path.stat().st_mtime
        except Exception:
            pass

    # Final fallback: if still no date, use file mtime
    if meta.date_taken is None:
        try:
            meta.date_taken = photo_path.stat().st_mtime
        except Exception:
            pass

    return meta
