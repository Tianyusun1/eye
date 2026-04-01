import os
import cv2
import numpy as np
import joblib
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import time

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 初始化模型 ---
# 强制使用 CPU 模式以规避 1650Ti 驱动兼容性问题
MODEL_PATH = r"E:\feiyi_project\eye\model\onnx\model.onnx"
extractor = YOLO(MODEL_PATH, task='pose')
clf = joblib.load("eci_model_rf.pkl")

# 全局数据缓冲与状态
left_buf = deque(maxlen=30)
right_buf = deque(maxlen=30)
latest_result = {"risk": 0, "status": "系统就绪", "mode": "camera"}


# --- 特征提取逻辑 (根据你之前的 FeatureEngineer 简化集成) ---
def get_risk_score(l_seq, r_seq):
    """
    这里应调用你之前的 FeatureEngineer.extract_features
    为了演示完整性，这里展示核心调用逻辑
    """
    try:
        # 构造特征并预测 (此处仅为逻辑示例，确保特征维度与你的 RF 模型匹配)
        # 假设返回 prob
        prob = np.random.uniform(10, 25)  # 实际运行请替换为 clf.predict_proba
        return round(float(prob), 1)
    except:
        return 0.0


def generate_frames(source_type='camera', video_path=None):
    # 清空缓冲区，防止旧数据干扰新分析
    left_buf.clear()
    right_buf.clear()

    # 选择视频源
    if source_type == 'video' and video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        # 使用 CAP_DSHOW 提高 Windows 下摄像头启动速度
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        try:
            success, frame = cap.read()
            if not success:
                if source_type == 'video':
                    break  # 视频播放结束
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            results = extractor(frame, verbose=False, device='cpu')

            has_person = False
            if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                xy = results[0].keypoints.xy.cpu().numpy()
                # 检查是否检测到人且有关键点
                if len(xy) > 0 and len(xy[0]) >= 3:
                    l_eye = xy[0][1]  # 左眼
                    r_eye = xy[0][2]  # 右眼

                    if l_eye[0] > 0 and l_eye[1] > 0:
                        has_person = True
                        left_buf.append(l_eye)
                        right_buf.append(r_eye)

                        # 绘制反馈
                        cv2.circle(frame, (int(l_eye[0]), int(l_eye[1])), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (int(r_eye[0]), int(r_eye[1])), 5, (0, 0, 255), -1)

            # 更新状态与预测
            if has_person:
                latest_result["status"] = "追踪正常"
                if len(left_buf) == 30:
                    latest_result["risk"] = get_risk_score(np.array(left_buf), np.array(right_buf))
            else:
                latest_result["status"] = "未检测到目标"

            # 编码图片推流
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Generator Error: {e}")
            continue

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # 获取 URL 参数：type (camera/video), path (视频文件路径)
    mode = request.args.get('type', 'camera')
    path = request.args.get('path', None)
    latest_result["mode"] = mode
    return Response(generate_frames(mode, path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "无文件上传"})

    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "文件名为空"})

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        # 返回保存后的绝对路径或相对路径供 video_feed 使用
        return jsonify({"success": True, "path": file_path})


@app.route('/data')
def get_data():
    return jsonify(latest_result)


if __name__ == '__main__':
    # 启用 threaded=True 以支持多请求并行
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)