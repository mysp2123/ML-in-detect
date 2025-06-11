import cv2
import numpy as np
from skimage import feature
from skimage.filters import threshold_otsu
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

@dataclass
class DefectInfo:
    defect_type: str
    confidence: float
    location: Tuple[int, int]  # (x, y) coordinates
    size: float  # area in pixels
    severity: str  # 'Low', 'Medium', 'High'

class DefectAnalyzer:
    def __init__(self):
        self.severity_thresholds = {
            'size': {
                'low': 100,    # pixels
                'medium': 500   # pixels above this is high
            },
            'contrast': {
                'low': 30,     # pixel intensity difference
                'medium': 60    # above this is high
            }
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh để tăng cường khả năng phát hiện khuyết tật."""
        # Chuyển sang ảnh xám
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Giảm nhiễu
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Cân bằng histogram
        equalized = cv2.equalizeHist(denoised)
        
        return equalized
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Phát hiện cạnh sử dụng Canny edge detection."""
        # Tự động tìm ngưỡng dựa trên histogram
        thresh = threshold_otsu(image)
        
        # Áp dụng Canny edge detection
        edges = feature.canny(
            image,
            sigma=3.0,
            low_threshold=0.55 * thresh,
            high_threshold=thresh
        )
        
        return edges.astype(np.uint8) * 255
    
    def find_contours(self, edge_image: np.ndarray) -> List[np.ndarray]:
        """Tìm các contour từ ảnh edge."""
        contours, _ = cv2.findContours(
            edge_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def analyze_defect(self, contour: np.ndarray, original_image: np.ndarray) -> DefectInfo:
        """Phân tích một khuyết tật cụ thể."""
        # Tính toán các đặc trưng của khuyết tật
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Tính độ tương phản trong vùng khuyết tật
        mask = np.zeros_like(original_image)
        cv2.drawContours(mask, [contour], -1, (255), -1)
        mean_intensity = cv2.mean(original_image, mask=mask)[0]
        surrounding_mask = cv2.dilate(mask, None) - mask
        surrounding_intensity = cv2.mean(original_image, mask=surrounding_mask)[0]
        contrast = abs(mean_intensity - surrounding_intensity)
        
        # Xác định mức độ nghiêm trọng
        if area < self.severity_thresholds['size']['low']:
            severity = 'Low'
        elif area < self.severity_thresholds['size']['medium']:
            severity = 'Medium'
        else:
            severity = 'High'
            
        # Xác định loại khuyết tật dựa trên hình dạng và kích thước
        aspect_ratio = float(w) / h if h != 0 else 0
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            defect_type = 'Crack'
        elif cv2.isContourConvex(contour):
            defect_type = 'Pit'
        else:
            defect_type = 'Surface Irregularity'
        
        # Tính độ tin cậy dựa trên contrast và kích thước
        confidence = min((contrast / 100.0 + area / 1000.0) / 2.0, 1.0)
        
        return DefectInfo(
            defect_type=defect_type,
            confidence=confidence,
            location=(x + w//2, y + h//2),
            size=area,
            severity=severity
        )
    
    def analyze_image(self, image_path: str) -> Tuple[List[DefectInfo], np.ndarray]:
        """Phân tích toàn bộ ảnh và trả về danh sách các khuyết tật."""
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        # Tiền xử lý
        preprocessed = self.preprocess_image(image)
        
        # Phát hiện cạnh
        edges = self.detect_edges(preprocessed)
        
        # Tìm contours
        contours = self.find_contours(edges)
        
        # Phân tích từng khuyết tật
        defects = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Lọc nhiễu
                defect = self.analyze_defect(contour, preprocessed)
                defects.append(defect)
        
        # Vẽ kết quả lên ảnh
        result_image = image.copy()
        for defect in defects:
            x, y = defect.location
            # Vẽ điểm trung tâm
            cv2.circle(result_image, (x, y), 5, (0, 0, 255), -1)
            # Vẽ thông tin
            text = f"{defect.defect_type} ({defect.severity})"
            cv2.putText(result_image, text, (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return defects, result_image
    
    def visualize_results(self, image_path: str, save_path: str = None):
        """Hiển thị và lưu kết quả phân tích."""
        defects, result_image = self.analyze_image(image_path)
        
        # In thông tin chi tiết
        print("\nKết quả phân tích khuyết tật:")
        print("-" * 50)
        for i, defect in enumerate(defects, 1):
            print(f"\nKhuyết tật #{i}:")
            print(f"Loại: {defect.defect_type}")
            print(f"Độ nghiêm trọng: {defect.severity}")
            print(f"Kích thước: {defect.size:.2f} pixels")
            print(f"Độ tin cậy: {defect.confidence:.2%}")
            print(f"Vị trí: {defect.location}")
        
        # Hiển thị ảnh kết quả
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Kết quả phân tích khuyết tật")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            print(f"\nĐã lưu kết quả phân tích tại: {save_path}")
        
        plt.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Sử dụng: python defect_analyzer.py <đường_dẫn_ảnh>")
        sys.exit(1)
    
    analyzer = DefectAnalyzer()
    analyzer.visualize_results(
        sys.argv[1],
        save_path="defect_analysis_result.png"
    ) 