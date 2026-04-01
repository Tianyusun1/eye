import os
import cv2
import numpy as np
import joblib
from collections import deque
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 模块 1: 提取器与特征工程
# ==========================================
class YoloEyeExtractor:
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='pose') 

    def extract_from_frame(self, frame_or_path):
        # ⚠️ 加上 device=0，开启 GPU/CUDA 加速，保证 Demo 实时流的极速丝滑
        results = self.model(frame_or_path, verbose=False, device=0)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None, None
            
        keypoints = results[0].keypoints.xy[0].cpu().numpy() 
        if len(keypoints) < 3:
             return None, None
             
        # 返回左右眼坐标
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
# 模块 2: Demo 核心可视化与交互系统
# ==========================================
def select_video_file():
    """打开文件选择对话框"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择要测试的视频文件",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
    )
    return file_path

def run_demo():
    print("=== 早期认知障碍双目筛查原型系统 (Demo - RF Version) ===")
    
    # 1. 加载模型配置
    onnx_path = "/home/610-sty/money/eye/model/onnx/model.onnx"
    ml_model_path = "eci_model_rf.pkl"    
    
    print(f"[System] 正在加载 YOLO 姿态模型 (开启 CUDA)...")
    extractor = YoloEyeExtractor(model_path=onnx_path)
    
    print(f"[System] 正在加载随机森林评估模型: {ml_model_path}")
    try:
        clf = joblib.load(ml_model_path)
        # ⚠️ 树模型(RF)不需要 StandardScaler，此处显式设为 None 防止报错
        scaler = None 
    except Exception as e:
        print(f"❌ 模型加载失败！请确认同目录下是否存在 {ml_model_path}。错误: {e}")
        return

    # 2. 初始化滑动窗口缓冲池 (容量 30 帧)
    SEQ_LEN = 30
    left_buf = deque(maxlen=SEQ_LEN)
    right_buf = deque(maxlen=SEQ_LEN)
    
    # 状态变量
    current_risk_prob = 0.0
    current_status = "Waiting for face..."
    is_high_risk = False

    # 3. 初始化视频流 (默认打开本地摄像头)
    cap = cv2.VideoCapture(0)
    print("\n[System] 系统已就绪！")
    print("=========================================")
    print(" 快捷键操作指南:")
    print("  [C] - 切换到本地摄像头")
    print("  [V] - 上传/选择本地测试视频")
    print("  [Q] - 退出系统")
    print("=========================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            # 如果视频播放完，循环播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame = cv2.flip(frame, 1) # 镜像翻转，符合直觉
        display_frame = frame.copy()

        # 提取关键点
        l_eye, r_eye = extractor.extract_from_frame(frame)

        if l_eye is not None and r_eye is not None and not (l_eye[0]==0 and l_eye[1]==0):
            left_buf.append(l_eye)
            right_buf.append(r_eye)
            
            # 绘制实时关键点
            cv2.circle(display_frame, (int(l_eye[0]), int(l_eye[1])), 4, (0, 255, 0), -1)
            cv2.circle(display_frame, (int(r_eye[0]), int(r_eye[1])), 4, (255, 0, 0), -1)
            
            # 绘制轨迹连线 (展现双目协同性)
            if len(left_buf) > 1:
                pts_l = np.array(left_buf, np.int32).reshape((-1, 1, 2))
                pts_r = np.array(right_buf, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts_l], False, (0, 200, 0), 2)
                cv2.polylines(display_frame, [pts_r], False, (200, 0, 0), 2)

            # 当缓冲池填满 30 帧时，进行医学风险推理
            if len(left_buf) == SEQ_LEN:
                features = FeatureEngineer.extract_features(np.array(left_buf), np.array(right_buf))
                features_reshaped = features.reshape(1, -1)
                
                # 随机森林跳过 scaler 转换
                if scaler is not None:
                    features_reshaped = scaler.transform(features_reshaped)
                
                # 获取概率和预测标签
                probs = clf.predict_proba(features_reshaped)[0]
                pred_label = clf.predict(features_reshaped)[0]
                
                current_risk_prob = probs[1] * 100 # 高风险的概率
                is_high_risk = bool(pred_label == 1)
                current_status = "Analysis Active"
                
                # 清除一半的缓冲区，实现平滑重叠窗口检测 (Overlap)
                for _ in range(15):
                    left_buf.popleft()
                    right_buf.popleft()
        else:
            current_status = "Tracking Lost..."
            # 丢失目标时，慢慢清空缓冲区防止断点跳跃
            if len(left_buf) > 0:
                left_buf.popleft()
                right_buf.popleft()

        # ==========================================
        # 绘制极具科技感的 UI 界面
        # ==========================================
        # 背景遮罩
        cv2.rectangle(display_frame, (10, 10), (450, 130), (0, 0, 0), -1)
        
        # 标题与状态
        cv2.putText(display_frame, "ECI Screening Prototype", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Status: {current_status} | Buffer: {len(left_buf)}/30", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 风险概率条
        bar_x, bar_y, bar_w, bar_h = 20, 80, 400, 20
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        fill_w = int((current_risk_prob / 100.0) * bar_w)
        color = (0, 0, 255) if is_high_risk else (0, 255, 0) # 高风险红，低风险绿
        if current_status == "Analysis Active":
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            
            # 文本显示结论
            risk_text = f"HIGH RISK ({current_risk_prob:.1f}%)" if is_high_risk else f"Low Risk ({current_risk_prob:.1f}%)"
            cv2.putText(display_frame, risk_text, (bar_x + 5, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255) if is_high_risk else (0,0,0), 2)
        
        # 操作提示
        cv2.putText(display_frame, "[C]Camera  [V]Video  [Q]Quit", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示画面
        cv2.imshow("ECI Screening System - Prototype", display_frame)

        # ==========================================
        # 键盘事件监听
        # ==========================================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\n[System] 切换至本地摄像头...")
            cap.release()
            cap = cv2.VideoCapture(0)
            left_buf.clear()
            right_buf.clear()
        elif key == ord('v'):
            file_path = select_video_file()
            if file_path:
                print(f"\n[System] 正在加载视频: {file_path}")
                cap.release()
                cap = cv2.VideoCapture(file_path)
                left_buf.clear()
                right_buf.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()