import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

def load_model():
    # Tạo mô hình
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    
    # Load trọng số đã huấn luyện
    model.load_state_dict(torch.load('casting_model.pth'))
    model.eval()
    return model

def predict_image(image_path):
    # Load và chuẩn bị ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # Load model và dự đoán
    model = load_model()
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.item()
    
    # In kết quả
    if prediction > 0.5:
        print(f"Kết quả: SẢN PHẨM LỖI (Độ tin cậy: {prediction:.2%})")
    else:
        print(f"Kết quả: SẢN PHẨM TỐT (Độ tin cậy: {(1-prediction):.2%})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Sử dụng: python predict.py <đường_dẫn_ảnh>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_image(image_path) 