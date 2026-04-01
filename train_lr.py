import os
import cv2
import glob
import numpy as np
import joblib  # 新增：用于保存和加载模型与scaler
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Module 1 & 2: 提取器与特征工程 (保持一致)
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

class FeatureEngineer:
    @staticmethod
    def moving_average(data, window_size=3):
        pad_size = window_size // 2
        padded_data = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            smoothed[i] = np.mean(padded_data[i:i+window_size], axis=0)
        return smoothed

    @staticmethod
    def normalize_trajectory(seq):
        std = np.std(seq, axis=0)
        std[std == 0] = 1e-6 
        return (seq - np.mean(seq, axis=0)) / std

    @staticmethod
    def compute_velocity(trajectory):
        if len(trajectory) < 2:
            return np.zeros_like(trajectory)
        vel = np.diff(trajectory, axis=0)
        return np.vstack((np.zeros(2), vel))

    @staticmethod
    def extract_features(left_seq, right_seq):
        features = []
        l_seq_smooth = FeatureEngineer.moving_average(left_seq)
        r_seq_smooth = FeatureEngineer.moving_average(right_seq)
        l_seq_norm = FeatureEngineer.normalize_trajectory(l_seq_smooth)
        r_seq_norm = FeatureEngineer.normalize_trajectory(r_seq_smooth)

        l_vel = FeatureEngineer.compute_velocity(l_seq_norm)
        r_vel = FeatureEngineer.compute_velocity(r_seq_norm)
        
        for i in range(2): 
            if np.std(l_vel[:, i]) == 0 or np.std(r_vel[:, i]) == 0:
                corr = 0.0 
            else:
                corr = np.corrcoef(l_vel[:, i], r_vel[:, i])[0, 1]
            features.append(corr)

        l_amp = np.linalg.norm(l_vel, axis=1)
        r_amp = np.linalg.norm(r_vel, axis=1)
        amp_diff = np.abs(l_amp - r_amp)
        features.append(np.mean(amp_diff))
        features.append(np.var(amp_diff))

        l_std_x, l_std_y = np.std(l_seq_smooth[:, 0]), np.std(l_seq_smooth[:, 1])
        r_std_x, r_std_y = np.std(r_seq_smooth[:, 0]), np.std(r_seq_smooth[:, 1])
        
        features.append(np.abs(l_std_x - r_std_x)) 
        features.append(np.abs(l_std_y - r_std_y)) 
        
        features.append(0.0)
        features.append(0.0)
        return np.array(features)

# ==========================================
# Module 3: 严格物理隔离的健康数据处理
# ==========================================
def process_healthy_images_strict(image_paths_list, extractor, seq_len=30, max_seqs=150, desc=""):
    X_healthy, y_healthy = [], []
    seq_count = 0
    print(f"\n[Healthy Group - {desc}] Extracting features...")
    for i in range(0, len(image_paths_list) - seq_len, seq_len):
        if seq_count >= max_seqs: break
        chunk_paths = image_paths_list[i:i+seq_len]
        l_seq, r_seq = [], []
        for img_path in chunk_paths:
            l_eye, r_eye = extractor.extract_from_frame(img_path)
            l_seq.append(l_eye)
            r_seq.append(r_eye)
        features = FeatureEngineer.extract_features(np.array(l_seq), np.array(r_seq))
        X_healthy.append(features)
        y_healthy.append(0) 
        seq_count += 1
    return np.array(X_healthy), np.array(y_healthy)

# ==========================================
# Module 4: 患者视频处理
# ==========================================
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
                features = FeatureEngineer.extract_features(np.array(l_seq), np.array(r_seq))
                X_ill.append(features)
                y_ill.append(1)
                l_seq, r_seq = [], []
        cap.release()
        print(f"  -> Video {os.path.basename(vid_path)} processed, extracted {frame_count} frames.")
    return np.array(X_ill), np.array(y_ill)

# ==========================================
# Module 5: 逻辑回归 对比实验主流程 (绝对双向物理隔离)
# ==========================================
if __name__ == "__main__":
    print("=== ECI Binocular Screening System (LR Strict Baseline) ===")
    
    onnx_path = "/home/610-sty/money/eye/model/onnx/model.onnx"
    healthy_path = "/home/610-sty/money/eye/MPIIFaceGaze_dataset/MPIIFaceGaze"
    ill_dir = "/home/610-sty/money/eye/MPIIFaceGaze_dataset/ILL"
    
    extractor = YoloEyeExtractor(model_path=onnx_path)
    
    # --------------------------------------------------
    # 健康组数据的严格物理切分
    # --------------------------------------------------
    all_healthy_images = glob.glob(os.path.join(healthy_path, "**", "*.jpg"), recursive=True)
    all_healthy_images = sorted(all_healthy_images) 
    
    split_index = int(len(all_healthy_images) * 0.8)
    healthy_train_images = all_healthy_images[:split_index]
    healthy_test_images = all_healthy_images[split_index:]
    
    print("\n[Strategy] Healthy Data: Strict physical chronological split (80/20).")
    X_h_train, y_h_train = process_healthy_images_strict(healthy_train_images, extractor, seq_len=30, max_seqs=120, desc="Train")
    X_h_test, y_h_test = process_healthy_images_strict(healthy_test_images, extractor, seq_len=30, max_seqs=30, desc="Test")
    
    # --------------------------------------------------
    # 患者组数据的留一法物理隔离
    # --------------------------------------------------
    all_ill_videos = glob.glob(os.path.join(ill_dir, "*.mp4")) + glob.glob(os.path.join(ill_dir, "*.avi"))
    test_videos = [v for v in all_ill_videos if "test1" in os.path.basename(v)]
    train_videos = [v for v in all_ill_videos if "test1" not in os.path.basename(v)]
    
    print(f"\n[Strategy] Patient Data: Isolating {os.path.basename(test_videos[0])} as test set.")
    X_ill_train, y_ill_train = process_patient_videos(train_videos, extractor, seq_len=30)
    X_ill_test, y_ill_test = process_patient_videos(test_videos, extractor, seq_len=30)
    
    # --------------------------------------------------
    # 汇聚最终训练和测试集
    # --------------------------------------------------
    X_train = np.vstack((X_h_train, X_ill_train))
    y_train = np.hstack((y_h_train, y_ill_train))
    X_test = np.vstack((X_h_test, X_ill_test))
    y_test = np.hstack((y_h_test, y_ill_test))
    
    print(f"\n[Data Aggregation Complete]")
    print(f"Training set: {len(X_train)} (Healthy: {sum(y_train==0)}, Patient: {sum(y_train==1)})")
    print(f"Test set: {len(X_test)} (Healthy: {sum(y_test==0)}, Patient: {sum(y_test==1)})")
    
    # ⚠️ 数据标准化 (Z-score) 是逻辑回归收敛和解析权重的前提
    print("\n[Data Preprocessing] Scaling features for Logistic Regression (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # 测试集严禁fit
    
    print("\n[Model Training] Training Logistic Regression (Medical Gold Standard)...")
    clf_lr = LogisticRegression(class_weight='balanced', random_state=42)
    clf_lr.fit(X_train_scaled, y_train)
    
    y_pred = clf_lr.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n==================================================")
    print("        System Evaluation Report (LR Baseline)")
    print("==================================================")
    print(f"Test Set Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Metrics:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk (Healthy)", "High Risk (ECI)"]))
    
    # ==================================================
    # Module 6: LR 模型可解释性分析与模型保存
    # ==================================================
    print("\n==================================================")
    print("       LR Interpretability Analysis (Coefficients)")
    print("==================================================")
    
    feature_names = [
        "Horizontal Sync (X-Corr)", 
        "Vertical Sync (Y-Corr)", 
        "Mean Norm Amp Diff", 
        "Var Norm Amp Diff",
        "Diff L/R X Std", 
        "Diff L/R Y Std", 
        "N/A 1", 
        "N/A 2"
    ]
    
    # 获取逻辑回归的系数（取绝对值评估影响力大小）
    importances = np.abs(clf_lr.coef_[0])
    importances = importances / np.sum(importances) 
    indices = np.argsort(importances)[::-1]
    
    print("[Feature Weight Ranking]:")
    for f in range(6):
        idx = indices[f]
        print(f"  TOP {f + 1}. {feature_names[idx]:<25} Absolute Weight: {importances[idx] * 100:.2f}%")
        
    try:
        plt.figure(figsize=(12, 6))
        plt.title("Logistic Regression Baseline: Feature Weight Analysis", fontsize=14, fontweight='bold')
        
        # 逻辑回归使用蓝灰色系柱状图区分
        bars = plt.bar(range(6), importances[indices][:6], align="center", color='#607C8E') 
        plt.xticks(range(6), [feature_names[i] for i in indices][:6], rotation=45, ha='right', fontsize=11)
        plt.xlim([-1, 6])
        plt.ylabel("Normalized Absolute Coefficient", fontsize=12)
        plt.tight_layout()
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=10)
            
        plt.savefig("feature_importance_lr.png", dpi=300)
        print("\n[Plot Generated] LR feature weight chart saved as 'feature_importance_lr.png'.")
    except Exception as e:
        print(f"\n[Plotting Failed] Error: {e}")
        
    # --------------------------------------------------
    # 新增核心：保存训练好的 LR 模型和 StandardScaler
    # --------------------------------------------------
    model_save_path = "eci_model_lr.pkl"
    scaler_save_path = "eci_scaler_lr.pkl"
    print("\n[Model Saving] Saving the trained LR model and Scaler...")
    try:
        joblib.dump(clf_lr, model_save_path)
        joblib.dump(scaler, scaler_save_path)
        print(f"  -> Model successfully saved to: {model_save_path}")
        print(f"  -> Scaler successfully saved to: {scaler_save_path}")
        print("     (Both files are required for the Demo application.)")
    except Exception as e:
        print(f"  -> Failed to save model or scaler. Error: {e}")

    print("==================================================")