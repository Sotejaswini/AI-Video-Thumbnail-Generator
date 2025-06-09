import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import os
import argparse
import glob
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThumbnailGenerator:
    def __init__(self, model_path='yolov8n.pt', font_path='ariblk.ttf'):
        """Initialize the thumbnail generator with YOLO model and font."""
        logger.info("Initializing ThumbnailGenerator...")
        self.model = YOLO(model_path)
        self.font_path = font_path
        self.aspect_ratios = [(16, 9), (4, 3), (1, 1)]  # Common thumbnail ratios
        self.output_dir = "thumbnails"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("ThumbnailGenerator initialized successfully.")

    def extract_key_frames(self, video_path, max_frames=100):
        """Extract key frames from a video based on motion detection with a fallback."""
        logger.info(f"Attempting to open video: {video_path}")
        # Ensure the video path is absolute to avoid WSL issues
        video_path = os.path.abspath(video_path)
        logger.info(f"Absolute video path: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Cannot open video file: {video_path}")

        frames = []
        prev_frame = None
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in video: {total_frames}")

        if total_frames == 0:
            logger.error("Video has 0 frames. It might be corrupted or inaccessible.")
            cap.release()
            raise ValueError("Video has no frames.")

        step = max(1, total_frames // max_frames)
        logger.info(f"Step size: {step}")

        first_frame = None
        while cap.isOpened() and frame_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * step)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame at position {frame_count * step}")
                break

            logger.info(f"Read frame {frame_count} at position {frame_count * step}")
            if first_frame is None:
                first_frame = frame  # Store the first frame as a fallback

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                # Calculate motion score using frame difference
                diff = cv2.absdiff(gray, prev_frame)
                motion_score = np.mean(diff)
                logger.info(f"Motion score for frame {frame_count}: {motion_score}")
                # Lowered threshold for motion detection
                if motion_score > 5:  # Reduced from 10 to 5 to be more lenient
                    logger.info(f"Adding frame {frame_count} with motion score {motion_score}")
                    frames.append((frame, motion_score))
            else:
                logger.info("No previous frame to compare (first frame)")
            prev_frame = gray
            frame_count += 1

        cap.release()

        # Fallback: If no frames were added due to low motion, use the first frame
        if not frames and first_frame is not None:
            logger.warning("No frames met the motion threshold. Using the first frame as a fallback.")
            frames.append((first_frame, 0))

        logger.info(f"Total key frames extracted: {len(frames)}")
        if not frames:
            logger.error("No frames extracted, even after fallback.")
            raise ValueError("No key frames extracted, even after fallback.")
        return frames

    def detect_objects_and_faces(self, frame):
        """Detect objects and faces using YOLO."""
        logger.info("Detecting objects and faces...")
        results = self.model(frame)
        boxes = []
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                cls = int(box.cls.item())
                label = self.model.names[cls]
                if label == 'person' and conf > 0.5:
                    # Approximate face detection within person bounding box
                    face_width = (x2 - x1) // 3
                    face_height = (y2 - y1) // 3
                    face_x = x1 + (x2 - x1) // 3
                    face_y = y1
                    faces.append((face_x, face_y, face_x + face_width, face_y + face_height))
                boxes.append((x1, y1, x2, y2, conf, label))
        logger.info(f"Detected {len(faces)} faces and {len(boxes)} objects.")
        return boxes, faces

    def score_frame(self, frame, motion_score, boxes, faces):
        """Score frame based on motion, objects, and faces."""
        logger.info("Scoring frame...")
        score = motion_score
        score += len(faces) * 50  # Prioritize frames with faces
        score += len(boxes) * 20  # Add points for detected objects
        # Bonus for vibrant colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        score += saturation * 0.1
        logger.info(f"Frame score: {score}")
        return score

    def auto_crop(self, frame, faces, boxes, aspect_ratio):
        """Crop frame to focus on faces/objects with specified aspect ratio."""
        logger.info("Auto-cropping frame...")
        h, w = frame.shape[:2]
        target_w, target_h = aspect_ratio
        target_ratio = target_w / target_h
        frame_ratio = w / h

        # Calculate crop area centered on faces or objects
        if faces:
            x1, y1, x2, y2 = faces[0]  # Focus on first detected face
        elif boxes:
            x1, y1, x2, y2 = boxes[0][:4]
        else:
            x1, y1, x2, y2 = 0, 0, w, h

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        crop_w = int(min(w, h * target_ratio))
        crop_h = int(min(h, w / target_ratio))

        crop_x = max(0, center_x - crop_w // 2)
        crop_y = max(0, center_y - crop_h // 2)
        crop_x = min(crop_x, w - crop_w)
        crop_y = min(crop_y, h - crop_h)

        cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        resized = cv2.resize(cropped, (1280, int(1280 / target_ratio)))
        logger.info("Frame cropped and resized successfully.")
        return resized

    def add_text_overlay(self, image, text):
        """Add text overlay to the image using PIL."""
        logger.info(f"Adding text overlay: '{text}'")
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(self.font_path, 60)
            logger.info("Loaded font successfully.")
        except Exception as e:
            logger.warning(f"Failed to load font: {e}. Using default font.")
            font = ImageFont.load_default()

        # Calculate text position (top-left with outline)
        # Use textbbox instead of textsize (textsize is deprecated)
        bbox = draw.textbbox((0, 0), text, font=font)  # Returns (left, top, right, bottom)
        text_w = bbox[2] - bbox[0]  # Width: right - left
        text_h = bbox[3] - bbox[1]  # Height: bottom - top
        x, y = 50, 50
        outline_color = (0, 0, 0)
        fill_color = (255, 255, 0)

        # Draw text with outline
        for offset_x, offset_y in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((x + offset_x, y + offset_y), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=fill_color)

        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        logger.info("Text overlay added successfully.")
        return result

    def apply_filters(self, image):
        """Apply brightness and contrast filters."""
        logger.info("Applying filters to image...")
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        logger.info("Filters applied successfully.")
        return image

    def generate_thumbnail(self, video_path, text="Click Me!", output_prefix="thumbnail"):
        """Generate thumbnails for a video."""
        logger.info(f"Generating thumbnail for video: {video_path}")
        frames = self.extract_key_frames(video_path)
        if not frames:
            logger.error("No key frames extracted after all attempts.")
            raise ValueError("No key frames extracted after all attempts.")

        # Score and select best frame
        logger.info("Scoring extracted frames...")
        best_frame = None
        best_score = -1
        for frame, motion_score in frames:
            boxes, faces = self.detect_objects_and_faces(frame)
            score = self.score_frame(frame, motion_score, boxes, faces)
            if score > best_score:
                best_score = score
                best_frame = (frame, boxes, faces)

        if best_frame is None:
            logger.error("No suitable frame found for thumbnail generation.")
            raise ValueError("No suitable frame found")

        frame, boxes, faces = best_frame
        outputs = []
        for aspect_ratio in self.aspect_ratios:
            # Crop and process frame
            logger.info(f"Processing frame for aspect ratio {aspect_ratio[0]}:{aspect_ratio[1]}")
            cropped = self.auto_crop(frame, faces, boxes, aspect_ratio)
            filtered = self.apply_filters(cropped)
            thumb = self.add_text_overlay(filtered, text)

            # Save thumbnail
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"{output_prefix}_{aspect_ratio[0]}x{aspect_ratio[1]}_{timestamp}.jpg")
            cv2.imwrite(output_path, thumb)
            outputs.append((output_path, frame))  # Return original frame for comparison
            logger.info(f"Saved thumbnail: {output_path}")

        return outputs

    def batch_process(self, video_dir, text="Click Me!"):
        """Process multiple videos in a directory."""
        logger.info(f"Batch processing videos in directory: {video_dir}")
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        all_outputs = []
        for video_path in video_files:
            try:
                # Use the video's base name as the output prefix
                video_name = os.path.basename(video_path).split('.')[0]
                outputs = self.generate_thumbnail(video_path, text, output_prefix=video_name)
                all_outputs.extend(outputs)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
        logger.info(f"Batch processing completed. Processed {len(all_outputs)} thumbnails.")
        return all_outputs

def main():
    parser = argparse.ArgumentParser(description="AI Video Thumbnail Generator")
    parser.add_argument("--video", help="Path to input video file or directory")
    parser.add_argument("--text", default="Click Me!", help="Text for thumbnail overlay")
    parser.add_argument("--batch", action="store_true", help="Process all videos in directory")
    args = parser.parse_args()

    generator = ThumbnailGenerator()
    if args.batch:
        outputs = generator.batch_process(args.video, args.text)
    else:
        # Use the video's base name as the output prefix
        video_name = os.path.basename(args.video).split('.')[0]
        outputs = generator.generate_thumbnail(args.video, args.text, output_prefix=video_name)

    for output_path, original_frame in outputs:
        print(f"Generated thumbnail: {output_path}")

if __name__ == "__main__":
    main()
