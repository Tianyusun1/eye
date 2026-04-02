import os
import cv2
import glob
import numpy as np
import joblib  
from ultralytics import YOLO
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Module 1 & 2: 提取器与【全新尺度不变】特征工程
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
        
        # ---------------------------------------------------------
        # 【核心修复】: 双眼间距归一化 (彻底消除距离和分辨率影响)
        # ---------------------------------------------------------
        mean_l = np.mean(left_seq, axis=0)
        mean_r = np.mean(right_seq, axis=0)
        eye_dist = np.linalg.norm(mean_l - mean_r)
        if eye_dist < 1e-5: 
            eye_dist = 1.0 # 防止除零

        # 转化为相对于眼距的比例坐标
        l_seq_scaled = left_seq / eye_dist
        r_seq_scaled = right_seq / eye_dist

        # 平滑处理
        l_seq_smooth = FeatureEngineer.moving_average(l_seq_scaled)
        r_seq_smooth = FeatureEngineer.moving_average(r_seq_scaled)
        
        l_seq_norm = FeatureEngineer.normalize_trajectory(l_seq_smooth)
        r_seq_norm = FeatureEngineer.normalize_trajectory(r_seq_smooth)

        l_vel = FeatureEngineer.compute_velocity(l_seq_smooth)
        r_vel = FeatureEngineer.compute_velocity(r_seq_smooth)
        
        # 1-2. X/Y 轴相关性
        for i in range(2): 
            if np.std(l_vel[:, i]) == 0 or np.std(r_vel[:, i]) == 0:
                corr = 0.0 
            else:
                corr = np.corrcoef(l_vel[:, i], r_vel[:, i])[0, 1]
            features.append(corr)

        # 3-4. 振幅差异均值与方差
        l_amp = np.linalg.norm(l_vel, axis=1)
        r_amp = np.linalg.norm(r_vel, axis=1)
        amp_diff = np.abs(l_amp - r_amp)
        features.append(np.mean(amp_diff))
        features.append(np.var(amp_diff))

        # 5-6. X/Y轴标准差(晃动幅度)的绝对差异
        l_std_x, l_std_y = np.std(l_seq_smooth[:, 0]), np.std(l_seq_smooth[:, 1])
        r_std_x, r_std_y = np.std(r_seq_smooth[:, 0]), np.std(r_seq_smooth[:, 1])
        
        features.append(np.abs(l_std_x - r_std_x)) 
        features.append(np.abs(l_std_y - r_std_y)) 
        
        # =========================================================
        # 【新增核心】: 凝视点分散度特征 (Gaze Dispersion)
        # 估算凝视点：取归一化后的左右眼坐标的中心点
        # =========================================================
        gaze_points = (l_seq_scaled + r_seq_scaled) / 2.0  
        
        gaze_dispersion_x = np.std(gaze_points[:, 0])
        gaze_dispersion_y = np.std(gaze_points[:, 1])
        
        features.append(gaze_dispersion_x) 
        features.append(gaze_dispersion_y) 
        # =========================================================
        
        return np.array(features)

# ==========================================
# Module 3 & 4: 数据处理
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
# Module 5: SVM 对比实验主流程
# ==========================================
if __name__ == "__main__":
    print("=== ECI Binocular Screening System (Robust SVM Baseline) ===")
    
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
    
    X_train = np.vstack((X_h_train, X_ill_train))
    y_train = np.hstack((y_h_train, y_ill_train))
    X_test = np.vstack((X_h_test, X_ill_test))
    y_test = np.hstack((y_h_test, y_ill_test))
    
    # ⚠️ SVM 必须做数据标准化 (Z-score)，而且只用训练集拟合 StandardScaler
    print("\n[Data Preprocessing] Scaling features for SVM (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # 【核心修复】: 强抗过拟合的 SVM 配置
    # 加入 probability=True 允许输出概率
    # 调低 C=0.5 (默认是1.0)，增加正则化惩罚，强制平滑分类边界，防止死记硬背
    # ---------------------------------------------------------
    print("\n[Model Training] Training Robust Support Vector Machine (Linear Kernel)...")
    clf_svm = SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=42)
    clf_svm.fit(X_train_scaled, y_train)
    
    y_pred = clf_svm.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n==================================================")
    print("        System Evaluation Report (Robust SVM)")
    print("==================================================")
    print(f"Test Set Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Metrics:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk (Healthy)", "High Risk (ECI)"]))
    
    # ==================================================
    # Module 6: SVM 模型可解释性分析与模型保存
    # ==================================================
    print("\n==================================================")
    print("       SVM Interpretability Analysis (Linear Weights)")
    print("==================================================")
    
    # =========================================================
    # 【修改名称】: 同步修改为对应的特征名称
    # =========================================================
    feature_names = [
        "Horizontal Sync (X-Corr)", 
        "Vertical Sync (Y-Corr)", 
        "Mean Norm Amp Diff", 
        "Var Norm Amp Diff",
        "Diff L/R X Std", 
        "Diff L/R Y Std", 
        "Gaze Dispersion X", 
        "Gaze Dispersion Y"
    ]
    
    importances = np.abs(clf_svm.coef_[0])
    importances = importances / np.sum(importances) 
    indices = np.argsort(importances)[::-1]
    
    print("[Feature Weight Ranking]:")
    for f in range(6):
        idx = indices[f]
        print(f"  TOP {f + 1}. {feature_names[idx]:<25} Absolute Weight: {importances[idx] * 100:.2f}%")
        
    try:
        plt.figure(figsize=(12, 6))
        plt.title("Robust SVM: Feature Weight Analysis", fontsize=14, fontweight='bold')
        bars = plt.bar(range(6), importances[indices][:6], align="center", color='#DD8452') 
        plt.xticks(range(6), [feature_names[i] for i in indices][:6], rotation=45, ha='right', fontsize=11)
        plt.xlim([-1, 6])
        plt.ylabel("Normalized Absolute Weight", fontsize=12)
        plt.tight_layout()
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=10)
            
        plt.savefig("feature_importance_svm_robust.png", dpi=300)
    except Exception as e:
        print(f"\n[Plotting Failed] Error: {e}")
        
    model_save_path = "eci_model_svm.pkl"
    scaler_save_path = "eci_scaler_svm.pkl"
    print("\n[Model Saving] Saving the trained SVM model and Scaler...")
    joblib.dump(clf_svm, model_save_path)
    joblib.dump(scaler, scaler_save_path) # 对于 SVM 必须保存 scaler
    print(f"  -> Model successfully saved to: {model_save_path}")
    print(f"  -> Scaler successfully saved to: {scaler_save_path}")
    print("==================================================")