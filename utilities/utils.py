import cv2
from cv2.typing import MatLike

class Utils:
    @staticmethod
    def crop_image(image: MatLike, bbox: list) -> MatLike:
        """
        Crop the OpenCV image based on bounding box.
        
        Args:
            image: Provide the OpenCV image
            bbox (list[x1, y1, x2, y2]): Coordinates of the bounding box
        Return:
            Cropped OpenCV image
        """
        height, width = image.shape[:2]

        # YOLO-style boxes are often float; convert and clamp to valid image bounds.
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Fallback to the full frame if the crop is invalid after clamping.
        if x2 <= x1 or y2 <= y1:
            return image

        return image[y1:y2, x1:x2]
    
    @staticmethod
    def draw_bounding_box(image: MatLike, name: str, bbox: list, color: tuple[int, int, int]) -> MatLike:
        # Draw a bounding box for the prediction
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color=color,
            thickness=2
        )
        
        # Constant
        FONT_FACE = cv2.FONT_HERSHEY_PLAIN
        FONT_SCALE = 1.0
        THICKNESS = 2
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text=name,
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            thickness=THICKNESS
        )
        # Calculate the box position based on text size
        text_box_min = int(bbox[0]), int(bbox[1])
        text_box_max = int(bbox[0] + text_width + baseline), int(bbox[1] + text_height + baseline)
        # Draw a rectangle according to the text size
        cv2.rectangle(
            image,
            text_box_min,
            text_box_max,
            color=color,
            thickness=-1
        )
        # Put text into the box
        text_x = int(text_box_min[0] + baseline // 2)
        text_y = int(text_box_max[1] - baseline // 2)
        cv2.putText(
            image,
            name,
            (text_x, text_y),
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            thickness=THICKNESS,
            color=(255, 255, 255)
        )

        return image