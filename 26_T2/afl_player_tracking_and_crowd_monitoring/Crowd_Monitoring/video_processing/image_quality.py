import cv2
import numpy as np


class ImageQualityAnalyzer:
    """
    Analyze basic image quality metrics for extracted video frames.
    """

    def analyze(self, image_path: str) -> dict:
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Brightness
        brightness = float(np.mean(gray))

        # Contrast
        contrast = float(np.std(gray))

        # Blur (variance of Laplacian)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Simple sharpness estimate
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = float(np.mean(np.sqrt(sobelx ** 2 + sobely ** 2)))

        return {
            "image_path": image_path,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "blur": round(blur, 2),
            "sharpness": round(sharpness, 2),
        }


def recommend_enhancement(metrics: dict) -> dict:
    recommendations = []

    if metrics["contrast"] < 250:
        recommendations.append("Apply CLAHE")

    if metrics["brightness"] < 250:
        recommendations.append("Increase brightness")

    if metrics["blur"] < 250:
        recommendations.append("Frame may be blurred")

    if metrics["sharpness"] < 250:
        recommendations.append("Low sharpness detected")

    return {
        "recommendations": recommendations or ["No enhancement required"]
    }


if __name__ == "__main__":
    sample_image = r"video_processing/data/extracted_frames/frame_0001.jpg"

    analyzer = ImageQualityAnalyzer()
    metrics = analyzer.analyze(sample_image)

    print("\n========== IMAGE QUALITY METRICS ==========\n")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    result = recommend_enhancement(metrics)

    print("\n========== ENHANCEMENT RECOMMENDATIONS ==========\n")
    for item in result["recommendations"]:
        print(f"- {item}")