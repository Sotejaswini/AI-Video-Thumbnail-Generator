from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define directories
thumbnail_dir = "thumbnails"
original_dir = "original_frames"
comparison_dir = "comparisons"
os.makedirs(comparison_dir, exist_ok=True)

def find_matching_thumbnail(video_name, thumbnail_dir):
    """Find a matching 16:9 thumbnail for the given video name."""
    for thumb in os.listdir(thumbnail_dir):
        # Ensure the file is a jpg and contains '16x9'
        if thumb.endswith(".jpg") and "16x9" in thumb:
            # Check if the thumbnail starts with the video name (before the aspect ratio and timestamp)
            thumb_prefix = thumb.split("_16x9_")[0]
            if thumb_prefix == video_name:
                return thumb
    return None

# Iterate over original frames
for video in os.listdir(original_dir):
    if video.endswith(".jpg"):
        try:
            # Extract the video name by removing '_original.jpg' and '.mp4'
            video_name = video.replace("_original.jpg", "").replace(".mp4", "")
            logger.info(f"Processing original frame: {video} (video name: {video_name})")

            # Find a matching 16:9 thumbnail
            matching_thumb = find_matching_thumbnail(video_name, thumbnail_dir)
            if not matching_thumb:
                logger.warning(f"No matching 16:9 thumbnail found for {video_name}. Skipping.")
                continue

            logger.info(f"Found matching thumbnail: {matching_thumb}")

            # Open the original frame and thumbnail
            orig_img = Image.open(os.path.join(original_dir, video))
            thumb_img = Image.open(os.path.join(thumbnail_dir, matching_thumb))

            # Convert images to RGB if they are in RGBA mode (to avoid issues with pasting)
            if orig_img.mode == "RGBA":
                orig_img = orig_img.convert("RGB")
            if thumb_img.mode == "RGBA":
                thumb_img = thumb_img.convert("RGB")

            # Resize both images to 640x360 for consistency
            orig_img = orig_img.resize((640, 360), Image.Resampling.LANCZOS)
            thumb_img = thumb_img.resize((640, 360), Image.Resampling.LANCZOS)

            # Create a new image for side-by-side comparison (1280x360)
            combined = Image.new("RGB", (1280, 360))
            combined.paste(orig_img, (0, 0))
            combined.paste(thumb_img, (640, 0))

            # Save the comparison image
            comparison_path = os.path.join(comparison_dir, f"{video_name}_comparison.jpg")
            combined.save(comparison_path, quality=95)
            logger.info(f"Saved comparison image: {comparison_path}")

        except Exception as e:
            logger.error(f"Error processing {video}: {e}")
            continue

# Check if any comparisons were generated
if not os.listdir(comparison_dir):
    logger.warning("No comparison images were generated. Check if original frames and thumbnails exist and match.")
else:
    logger.info(f"Generated {len(os.listdir(comparison_dir))} comparison images in {comparison_dir}.")
