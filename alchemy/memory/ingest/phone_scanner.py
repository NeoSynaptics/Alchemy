"""PhoneScanner — detects connected phones via Windows Image Acquisition (WIA) API.

Uses win32com.client to access WIA.DeviceManager — the only reliable way to
enumerate and transfer photos from iPhones on Windows. Shell.Application COM
cannot browse iPhone internal storage (NameSpace returns null).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".mov", ".mp4"}


@dataclass
class PhoneDevice:
    """A detected phone/camera device."""
    name: str
    device_id: str
    device_type: int  # WIA type: 1=scanner, 2=camera, 3=video
    total_folders: int = 0
    total_photos: int = 0


@dataclass
class PhotoFile:
    """A photo file found on the device via WIA."""
    filename: str
    folder_name: str       # e.g. "800AAAAA"
    folder_index: int      # 1-based WIA folder index
    item_index: int        # 1-based WIA item index within folder (0 = flat)
    extension: str = ""    # e.g. ".jpg"
    date_taken: float = 0.0  # Unix timestamp from WIA Item Time Stamp


def _heic_to_jpeg(wia_image, dest: Path) -> Path:
    """Save a WIA ImageFile as JPEG, converting from HEIC if needed.

    Saves the raw HEIC first, converts via Pillow + pillow-heif, then
    deletes the temp HEIC. Returns the final .jpg path.
    """
    temp_heic = dest.with_suffix(".heic")
    jpg_dest = dest.with_suffix(".jpg")
    wia_image.SaveFile(str(temp_heic))
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        from PIL import Image as PILImage
        img = PILImage.open(str(temp_heic))
        img.save(str(jpg_dest), "JPEG", quality=92)
        temp_heic.unlink()
        return jpg_dest
    except Exception:
        logger.warning("HEIC conversion failed for %s, keeping raw", dest.name)
        # Keep the raw HEIC as fallback
        if not dest.exists() and temp_heic.exists():
            temp_heic.rename(dest)
        return dest


class PhoneScanner:
    """Scans for connected phones and lists/transfers photos via WIA.

    WIA (Windows Image Acquisition) is the correct API for iPhones on Windows.
    Requires pywin32 (win32com.client).
    """

    def __init__(self, staging_dir: Path) -> None:
        self._staging_dir = staging_dir
        self._staging_dir.mkdir(parents=True, exist_ok=True)

    def _get_wia_manager(self):
        """Get WIA DeviceManager COM object."""
        import win32com.client
        return win32com.client.Dispatch("WIA.DeviceManager")

    def detect_devices(self) -> list[PhoneDevice]:
        """Detect connected phones/cameras via WIA."""
        devices: list[PhoneDevice] = []
        try:
            wia = self._get_wia_manager()
            count = wia.DeviceInfos.Count
            logger.info("WIA found %d device(s)", count)

            for i in range(1, count + 1):
                info = wia.DeviceInfos.Item(i)
                name = info.Properties("Name").Value
                dev_id = info.DeviceID
                dev_type = info.Type

                # Connect to count folders/photos
                total_folders = 0
                total_photos = 0
                try:
                    device = info.Connect()
                    top_count = device.Items.Count
                    # Check if flat (items are photos) or nested (items are folders)
                    if top_count > 0:
                        first = device.Items.Item(1)
                        if first.Items.Count == 0:
                            # Flat: all top items are photos
                            total_photos = top_count
                        else:
                            # Nested: count children
                            total_folders = top_count
                            for fi in range(1, top_count + 1):
                                total_photos += device.Items.Item(fi).Items.Count
                except Exception:
                    logger.warning("Could not connect to %s for counting", name)

                devices.append(PhoneDevice(
                    name=name,
                    device_id=dev_id,
                    device_type=dev_type,
                    total_folders=total_folders,
                    total_photos=total_photos,
                ))
                logger.info(
                    "Device: %s (type=%d, folders=%d, photos=%d)",
                    name, dev_type, total_folders, total_photos,
                )

        except Exception:
            logger.warning("WIA device scan failed", exc_info=True)

        return devices

    def list_photos(self, device_index: int = 1) -> list[PhotoFile]:
        """List all photos on a WIA device.

        Handles two WIA layouts:
        - Flat: top-level items ARE photos (iPhone typical)
        - Nested: top-level items are folders containing photos

        Args:
            device_index: 1-based WIA device index (default 1 = first device).
        """
        photos: list[PhotoFile] = []
        try:
            wia = self._get_wia_manager()
            info = wia.DeviceInfos.Item(device_index)
            device = info.Connect()

            for fi in range(1, device.Items.Count + 1):
                item = device.Items.Item(fi)
                children = item.Items.Count

                if children > 0:
                    # Nested: this is a folder containing photos
                    folder_name = item.Properties("Item Name").Value
                    for ii in range(1, children + 1):
                        sub = item.Items.Item(ii)
                        photo = self._make_photo(sub, folder_name, fi, ii)
                        if photo:
                            photos.append(photo)
                else:
                    # Flat: this item IS a photo
                    photo = self._make_photo(item, "root", fi, 0)
                    if photo:
                        photos.append(photo)

            logger.info("Found %d photos via WIA on device %d", len(photos), device_index)

        except Exception:
            logger.error("WIA photo listing failed", exc_info=True)

        return photos

    def _make_photo(self, item, folder_name: str, folder_index: int, item_index: int) -> PhotoFile | None:
        """Build a PhotoFile from a WIA item, or None if not a photo."""
        name = item.Properties("Item Name").Value
        ext = Path(name).suffix.lower()

        if not ext:
            try:
                ext = "." + item.Properties("Filename extension").Value.lower()
                name = name + ext
            except Exception:
                ext = ".jpg"
                name = name + ext

        if ext not in IMAGE_EXTENSIONS:
            return None

        # Extract original capture timestamp from WIA
        date_taken = 0.0
        try:
            ts_vec = item.Properties("Item Time Stamp").Value
            # WIA Vector: [year, month, day_of_week, day, hour, min, sec, ms]
            from datetime import datetime
            parts = [ts_vec.Item(i) for i in range(1, 9)]
            dt = datetime(parts[0], parts[1], parts[3], parts[4], parts[5], parts[6])
            date_taken = dt.timestamp()
        except Exception:
            pass

        return PhotoFile(
            filename=name,
            folder_name=folder_name,
            folder_index=folder_index,
            item_index=item_index,
            extension=ext,
            date_taken=date_taken,
        )

    def transfer_photo(
        self,
        photo: PhotoFile,
        dest_dir: Path,
        device_index: int = 1,
    ) -> Path:
        """Transfer a single photo from device to local filesystem via WIA.

        Uses WIA item.Transfer() which returns an ImageFile object,
        then saves to disk with ImageFile.SaveFile().
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / photo.filename

        if dest.exists():
            return dest  # Already transferred

        wia = self._get_wia_manager()
        info = wia.DeviceInfos.Item(device_index)
        device = info.Connect()

        item = self._resolve_item(device, photo)
        image = item.Transfer()
        image.SaveFile(str(dest))

        return dest

    @staticmethod
    def _resolve_item(device, photo: PhotoFile):
        """Get the WIA item for a PhotoFile (flat or nested)."""
        if photo.item_index == 0:
            # Flat layout: item is directly at folder_index
            return device.Items.Item(photo.folder_index)
        else:
            # Nested: folder then item
            folder = device.Items.Item(photo.folder_index)
            return folder.Items.Item(photo.item_index)

    def bulk_transfer(
        self,
        photos: list[PhotoFile],
        dest_dir: Path,
        device_index: int = 1,
        on_progress: callable | None = None,
    ) -> list[Path]:
        """Transfer multiple photos from device via WIA.

        Connects once per batch. Reports progress every 10 photos.
        HEIC/HEIF files are automatically converted to JPEG.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []

        try:
            wia = self._get_wia_manager()
            info = wia.DeviceInfos.Item(device_index)
            device = info.Connect()

            for i, photo in enumerate(photos):
                dest = dest_dir / photo.filename
                # Check for both original and converted filenames
                jpg_dest = dest.with_suffix('.jpg')
                if dest.exists() or (photo.extension.lower() in ('.heic', '.heif') and jpg_dest.exists()):
                    copied.append(jpg_dest if jpg_dest.exists() else dest)
                    if on_progress and (i + 1) % 10 == 0:
                        on_progress(i + 1, len(photos))
                    continue

                try:
                    item = self._resolve_item(device, photo)
                    image = item.Transfer()
                    # Convert HEIC/HEIF to JPEG (browsers cannot display HEIC)
                    if photo.extension.lower() in ('.heic', '.heif'):
                        dest = _heic_to_jpeg(image, dest)
                    else:
                        image.SaveFile(str(dest))
                    copied.append(dest)
                except Exception:
                    logger.warning("Failed to transfer %s", photo.filename)

                if on_progress and (i + 1) % 10 == 0:
                    on_progress(i + 1, len(photos))

        except Exception:
            logger.error("WIA bulk transfer failed after %d files", len(copied), exc_info=True)

        logger.info("Transferred %d/%d photos to %s", len(copied), len(photos), dest_dir)
        return copied
