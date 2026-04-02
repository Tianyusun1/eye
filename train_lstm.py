import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 自动检测并使用 GPU (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Module 1: 提取器
# ==========================================
class YoloEyeExtractor:
    def __init__(self, model_path):
        print(f"Loading YOLOv8-pose ONNX model: {model_path}")
        self.model = YOLO(model_path, task='pose') 

    def extract_from_frame(self, frame_or_path):
        results = self.model(frame_or_path, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

        keypoints = results[0].keypoints.xy[0].cpu().numpy() 
        if len(keypoints) < 3:
             return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

        return keypoints[1], keypoints[2]

# ==========================================
# Module 2: 时序序列工程 (专为 LSTM 设计)
# 不再提取统计特征，而是输出 (30, 4) 的归一化原始轨迹
# ==========================================
class SequenceEngineer:
    @staticmethod
    def extract_sequence(left_seq, right_seq):
        """
        输入: left_seq (30, 2), right_seq (30, 2)
        输出: seq_features (30, 4)
        """
        # 1. 双眼间距归一化 (抗距离干扰)
        mean_l = np.mean(left_seq, axis=0)
        mean_r = np.mean(right_seq, axis=0)
        eye_dist = np.linalg.norm(mean_l - mean_r)
        if eye_dist < 1e-5: 
            eye_dist = 1.0 

        l_scaled = left_seq / eye_dist
        r_scaled = right_seq / eye_dist

        # 2. 轨迹中心化 (去除绝对位置影响，只保留相对运动变化)
        l_centered = l_scaled - np.mean(l_scaled, axis=0)
        r_centered = r_scaled - np.mean(r_scaled, axis=0)

        # 3. 拼接为单帧 4 维特征: [左眼X, 左眼Y, 右眼X, 右眼Y]
        seq_features = np.hstack((l_centered, r_centered)) # 形状: (30, 4)
        return seq_features

# ==========================================
# Module 3 & 4: 数据处理
# ==========================================
def process_healthy_images_strict(image_paths_list, extractor, seq_len=30, max_seqs=150, desc=""):
    X_healthy, y_healthy = [], []
    seq_count = 0
    print(f"\n[Healthy Group - {desc}] Extracting temporal sequences...")
    for i in range(0, len(image_paths_list) - seq_len, seq_len):
        if seq_count >= max_seqs: break
        chunk_paths = image_paths_list[i:i+seq_len]
        l_seq, r_seq = [], []
        for img_path in chunk_paths:
            l_eye, r_eye = extractor.extract_from_frame(img_path)
            l_seq.append(l_eye)
            r_seq.append(r_eye)
        # 获取 (30, 4) 的时序序列
        seq_data = SequenceEngineer.extract_sequence(np.array(l_seq), np.array(r_seq))
        X_healthy.append(seq_data)
        y_healthy.append(0) 
        seq_count += 1
    return np.array(X_healthy), np.array(y_healthy)

def process_patient_videos(video_paths, extractor, seq_len=30):
    X_ill, y_ill = [], []
    for vid_path in video_paths:
        cap = cv2.VideoCapture(vid_path)
        l_seq, r_seq = [], []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break 
            l_eye, r_eye = extractor.extract_from_frame(frame)
            l_seq.append(l_eye)
            r_seq.append(r_eye)
            frame_count += 1
            if len(l_seq) == seq_len:
                seq_data = SequenceEngineer.extract_sequence(np.array(l_seq), np.array(r_seq))
                X_ill.append(seq_data)
                y_ill.append(1)
                l_seq, r_seq = [], []
        cap.release()
        print(f"  -> Video {os.path.basename(vid_path)} processed, extracted {frame_count} frames.")
    return np.array(X_ill), np.array(y_ill)

# ==========================================
# Module 5: 定义 PyTorch LSTM 模型
# ==========================================
class EyeMotionLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_layers=2, num_classes=2):
        super(EyeMotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层 (加入 Dropout 防止小数据集过拟合)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 全连接分类层
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x 形状: (batch_size, seq_len=30, input_size=4)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 解码最后一个时间步的隐藏状态
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ==========================================
# Module 6: LSTM 主训练流程
# ==========================================
if __name__ == "__main__":
    print(f"=== ECI Binocular Screening System (Deep Learning: LSTM on {device}) ===")
    
    onnx_path = "/home/610-sty/money/eye/model/onnx/model.onnx"
    healthy_path = "/home/610-sty/money/eye/MPIIFaceGaze_dataset/MPIIFaceGaze"
    ill_dir = "/home/610-sty/money/eye/MPIIFaceGaze_dataset/ILL"
    
    extractor = YoloEyeExtractor(model_path=onnx_path)
    
    all_healthy_images = glob.glob(os.path.join(healthy_path, "**", "*.jpg"), recursive=True)
    all_healthy_images = sorted(all_healthy_images) 
    split_index = int(len(all_healthy_images) * 0.8)
    healthy_train_images = all_healthy_images[:split_index]
    healthy_test_images = all_healthy_images[split_index:]
    
    print("\n[Strategy] Healthy Data: Strict physical chronological split (80/20).")
    X_h_train, y_h_train = process_healthy_images_strict(healthy_train_images, extractor, seq_len=30, max_seqs=120, desc="Train")
    X_h_test, y_h_test = process_healthy_images_strict(healthy_test_images, extractor, seq_len=30, max_seqs=30, desc="Test")
    
    all_ill_videos = glob.glob(os.path.join(ill_dir, "*.mp4")) + glob.glob(os.path.join(ill_dir, "*.avi"))
    test_videos = [v for v in all_ill_videos if "test1" in os.path.basename(v)]
    train_videos = [v for v in all_ill_videos if "test1" not in os.path.basename(v)]
    
    print(f"\n[Strategy] Patient Data: Isolating test set.")
    X_ill_train, y_ill_train = process_patient_videos(train_videos, extractor, seq_len=30)
    X_ill_test, y_ill_test = process_patient_videos(test_videos, extractor, seq_len=30)
    
    # 汇总数据，形状为 (样本数, 30, 4)
    X_train = np.vstack((X_h_train, X_ill_train)).astype(np.float32)
    y_train = np.hstack((y_h_train, y_ill_train)).astype(np.int64)
    X_test = np.vstack((X_h_test, X_ill_test)).astype(np.float32)
    y_test = np.hstack((y_h_test, y_ill_test)).astype(np.int64)
    
    print(f"\n[Data Aggregation Complete]")
    print(f"Training set: {len(X_train)} sequences of shape {X_train.shape[1:]}")
    print(f"Test set: {len(X_test)} sequences of shape {X_test.shape[1:]}")

    # ==========================================
    # PyTorch 数据封装与加载
    # ==========================================
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 初始化模型
    model = EyeMotionLSTM(input_size=4, hidden_size=16, num_layers=2, num_classes=2).to(device)
    
    # 计算类别权重以应对不平衡数据
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3) # weight_decay 即 L2 正则化
    
    num_epochs = 100
    train_losses = []
    
    print("\n[Model Training] Starting LSTM Training on", device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"  -> Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
    # ==========================================
    # 测试与评估
    # ==========================================
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    
    print("\n==================================================")
    print("        System Evaluation Report (LSTM Model)")
    print("==================================================")
    print(f"Test Set Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Metrics:")
    print(classification_report(all_targets, all_preds, target_names=["Low Risk (Healthy)", "High Risk (ECI)"]))

    # ==========================================
    # 绘制训练 Loss 曲线 (替代特征重要性)
    # ==========================================
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), train_losses, color='#C44E52', linewidth=2)
        plt.title('LSTM Training Loss Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Cross Entropy Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("lstm_training_loss.png", dpi=300)
        print("\n[Plot Generated] LSTM loss curve saved as 'lstm_training_loss.png'.")
    except Exception as e:
        print(f"\n[Plotting Failed] Error: {e}")

    # ==========================================
    # 保存模型 (PyTorch 格式)
    # ==========================================
    model_save_path = "eci_model_lstm.pth"
    print("\n[Model Saving] Saving the trained LSTM model...")
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"  -> Model successfully saved to: {model_save_path}")
    except Exception as e:
        print(f"  -> Failed to save model. Error: {e}")

    print("==================================================")