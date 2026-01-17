class Utils:
    @staticmethod
    def crop_image(image, bbox):
        """
        Crop the OpenCV image based on bounding box.
        
        Args:
            image: Provide the OpenCV image
            bbox (list[x1, y1, x2, y2]): Coordinates of the bounding box
        Return:
            Croped OpenCV image
        """
        [x1, y1, x2, y2] = bbox
        return image[y1:y2, x1:x2]