import os
import pandas as pd
from defect_analyzer import DefectAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class BatchAnalyzer:
    def __init__(self):
        self.analyzer = DefectAnalyzer()
        self.results_dir = "analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def analyze_directory(self, directory):
        """Phân tích tất cả các ảnh trong một thư mục."""
        results = []
        for filename in os.listdir(directory):
            if filename.endswith('.jpeg'):
                image_path = os.path.join(directory, filename)
                try:
                    # Phân tích ảnh
                    defects, result_image = self.analyzer.analyze_image(image_path)
                    
                    # Lưu ảnh kết quả
                    result_filename = os.path.join(
                        self.results_dir,
                        f"result_{os.path.splitext(filename)[0]}.png"
                    )
                    plt.imsave(result_filename, result_image)
                    
                    # Lưu thông tin về các khuyết tật
                    for defect in defects:
                        results.append({
                            'image_name': filename,
                            'defect_type': defect.defect_type,
                            'confidence': defect.confidence,
                            'size': defect.size,
                            'severity': defect.severity,
                            'is_defective': 'def_front' in directory
                        })
                        
                except Exception as e:
                    print(f"Lỗi khi xử lý {filename}: {str(e)}")
        
        return results
    
    def analyze_all(self, base_dir="casting_512x512"):
        """Phân tích tất cả các ảnh trong cả hai thư mục."""
        all_results = []
        
        # Phân tích thư mục def_front
        def_dir = os.path.join(base_dir, "def_front")
        if os.path.exists(def_dir):
            print(f"\nĐang phân tích thư mục {def_dir}...")
            all_results.extend(self.analyze_directory(def_dir))
        
        # Phân tích thư mục ok_front
        ok_dir = os.path.join(base_dir, "ok_front")
        if os.path.exists(ok_dir):
            print(f"\nĐang phân tích thư mục {ok_dir}...")
            all_results.extend(self.analyze_directory(ok_dir))
        
        # Chuyển kết quả thành DataFrame
        df = pd.DataFrame(all_results)
        
        # Lưu kết quả vào CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.results_dir, f"analysis_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Tạo báo cáo thống kê
        self.generate_report(df)
        
        return df
    
    def generate_report(self, df):
        """Tạo báo cáo thống kê với các biểu đồ."""
        # Set style
        sns.set_style("whitegrid")
        
        # Tạo figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Phân bố loại khuyết tật
        plt.subplot(2, 2, 1)
        sns.countplot(data=df, x='defect_type')
        plt.title('Phân bố loại khuyết tật')
        plt.xticks(rotation=45)
        
        # 2. Phân bố mức độ nghiêm trọng
        plt.subplot(2, 2, 2)
        sns.countplot(data=df, x='severity')
        plt.title('Phân bố mức độ nghiêm trọng')
        
        # 3. Box plot kích thước theo loại khuyết tật
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x='defect_type', y='size')
        plt.title('Phân bố kích thước theo loại khuyết tật')
        plt.xticks(rotation=45)
        
        # 4. Histogram độ tin cậy
        plt.subplot(2, 2, 4)
        sns.histplot(data=df, x='confidence', bins=20)
        plt.title('Phân bố độ tin cậy')
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.results_dir, f"statistics_{timestamp}.png"))
        plt.close()
        
        # Tạo báo cáo tổng hợp
        report = f"""
BÁO CÁO PHÂN TÍCH KHUYẾT TẬT
===========================
Thời gian: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

1. Tổng quan
-----------
Tổng số ảnh đã phân tích: {len(df['image_name'].unique())}
Tổng số khuyết tật phát hiện được: {len(df)}

2. Thống kê theo loại khuyết tật
------------------------------
{df['defect_type'].value_counts().to_string()}

3. Thống kê theo mức độ nghiêm trọng
---------------------------------
{df['severity'].value_counts().to_string()}

4. Thống kê kích thước
--------------------
Kích thước trung bình: {df['size'].mean():.2f} pixels
Kích thước lớn nhất: {df['size'].max():.2f} pixels
Kích thước nhỏ nhất: {df['size'].min():.2f} pixels

5. Thống kê độ tin cậy
--------------------
Độ tin cậy trung bình: {df['confidence'].mean():.2%}
Độ tin cậy cao nhất: {df['confidence'].max():.2%}
Độ tin cậy thấp nhất: {df['confidence'].min():.2%}
"""
        
        # Lưu báo cáo
        with open(os.path.join(self.results_dir, f"report_{timestamp}.txt"), 'w') as f:
            f.write(report)
        
        print("\nĐã hoàn thành phân tích!")
        print(f"Kết quả được lưu trong thư mục: {self.results_dir}")

if __name__ == "__main__":
    analyzer = BatchAnalyzer()
    analyzer.analyze_all() 