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
        [x1, y1, x2, y2] = bbox
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def draw_bounding_box(image: MatLike, name: str, bbox: list, color: tuple[int, int, int]) -> MatLike:
        (text_width, text_height), baseline = cv2.getTextSize(
            text=name,
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.0,
            thickness=2
        )
        text_box_min = int(bbox[0]), int(bbox[1])
        text_box_max = int(bbox[0] + text_width), int(bbox[1] + text_height - baseline)

        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color=color,
            thickness=3
        )

        return image