import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 1. 載入 ResNet-18 模型 (預訓練)
model = models.resnet18(pretrained=True)
model.eval()  # 設定為評估模式

# 2. 定義圖片預處理 (與模型訓練過程一致)
transform = transforms.Compose([
    transforms.Resize(256),               # 調整圖片大小
    transforms.CenterCrop(224),           # 中心裁剪至 224x224
    transforms.ToTensor(),                # 轉換為 Tensor
    transforms.Normalize(                  # 正規化
        mean=[0.485, 0.456, 0.406],       # ImageNet 訓練集的均值
        std=[0.229, 0.224, 0.225]         # ImageNet 訓練集的標準差
    )
])

# 3. 載入圖片並應用預處理
image_path = '/workspace/work/Fast-BEV/data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'  # 替換為你的圖片路徑
image = Image.open(image_path).convert('RGB')  # 確保圖片為 RGB 模式
input_tensor = transform(image).unsqueeze(0)  # 增加一個 batch 維度

# 4. 使用模型進行推論
with torch.no_grad():
    outputs = model(input_tensor)

# 5. 取得分類結果
_, predicted = outputs.max(1)  # 獲取預測的類別索引
print(f"Predicted class index: {predicted.item()}")