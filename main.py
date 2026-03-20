#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3588 YOLOv11 全流程并行版（支持类别差异化置信度阈值）
"""
import os
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from rknnlite.api import RKNNLite

# ----------------------------
# 核心配置
# ----------------------------
# 路径与模型配置
current_path = os.path.dirname(os.path.abspath(__file__))
RKNN_MODEL_PATH = "/home/firefly/New folder1/rknn3588-yolov11-python/best-relu-960-s.rknn"
CAMERA_DEVICE = 11  # 摄像头设备号

# 模型参数
MODEL_SIZE = (960, 960)
CLASSES = ["pipe", "butt", "plastic_bottle", "aluminum_can", 
           "leaves", "plastic_wrap", "waste_paper", "carton",
           "glass", "HDPEM", "cig_case", "cup", "battery", "orange"]
# ========== 关键修改：类别差异化置信度阈值 ==========
# 默认阈值（未指定的类别用这个）
DEFAULT_OBJ_THRESH = 0.1
# 自定义类别阈值（格式：{"类别名": 阈值}，按需修改）
CLASS_OBJ_THRESH = {
    "glass": 0.8,    
    "aluminum_can": 0.5,     
    "plastic_bottle": 0.6,     
    "carton": 0.5,

}
# ==================================================
NMS_THRESH = 0.2       # 非极大值抑制阈值
color_palette = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 摄像头/显示配置
PREVIEW_WIDTH = 1280  # 预览窗口宽度
PREVIEW_HEIGHT = 720  # 预览窗口高度
WINDOW_NAME = "YOLOv11 RK3588 Full Parallel (960×960 | All Stages)"
CAMERA_FPS = 30         # 摄像头帧率上限
CAMERA_BUFFER = 1       # 最小化摄像头缓冲区

# 多线程配置
NPU_CORES = [0, 1, 2]                  # NPU核数
PREPROCESS_THREADS = 3                 # 预处理线程数
POSTPROCESS_THREADS = 3                # 后处理线程数
FRAME_QUEUE_SIZE = 1                  # 原始帧队列大小
PREPROCESS_QUEUE_SIZE = 1              # 预处理后帧队列大小
INFERENCE_QUEUE_SIZE = 1               # 推理结果队列大小
POSTPROCESS_QUEUE_SIZE = 1             # 后处理结果队列大小
RESULT_QUEUE_SIZE = 1                  # 最终结果队列大小
EXIT_FLAG = False
QUEUE_TIMEOUT = 0.001                   # 队列超时时间（缩短以提升响应）


# ----------------------------
# 全局资源（多队列解耦）
# ----------------------------
# 原始帧队列（采集→预处理）
raw_frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
# 预处理后队列（预处理→推理）
preprocessed_queue = Queue(maxsize=PREPROCESS_QUEUE_SIZE)
# 推理结果队列（推理→后处理）
inference_result_queue = Queue(maxsize=INFERENCE_QUEUE_SIZE)
# 后处理结果队列（后处理→显示）
postprocessed_queue = Queue(maxsize=POSTPROCESS_QUEUE_SIZE)
# 最终显示队列
result_queue = Queue(maxsize=RESULT_QUEUE_SIZE)

rknn_instances = []  # 存储3个RKNN实例


# 耗时统计
time_stats = {
    "capture": [],
    "preprocess": [],
    "inference": [],
    "postprocess": [],
    "total": []
}
stats_lock = threading.Lock()

# ----------------------------
# 图像预处理工具
# ----------------------------
class ImagePreprocessor:
    def __init__(self):
        self.pad_info = None

    def letter_box(self, im, new_shape=MODEL_SIZE, pad_color=(0, 0, 0)):
        """保持比例缩放+填充，返回处理后图像和填充信息"""
        if im is None or len(im.shape) != 3:
            return None, None
        
        h0, w0 = im.shape[:2]
        target_h, target_w = new_shape

        # 计算缩放比例
        scale = min(target_w / w0, target_h / h0)
        w1 = int(round(w0 * scale))
        h1 = int(round(h0 * scale))

        # 硬件加速resize
        if (w0, h0) != (w1, h1):
            im = cv2.resize(im, (w1, h1), interpolation=cv2.INTER_LINEAR)

        # 计算填充量
        dw = target_w - w1
        dh = target_h - h1
        dw_left = dw // 2
        dw_right = dw - dw_left
        dh_top = dh // 2
        dh_bottom = dh - dh_top

        # 硬件加速填充
        im = cv2.copyMakeBorder(
            im, dh_top, dh_bottom, dw_left, dw_right,
            cv2.BORDER_CONSTANT, value=pad_color
        )

        # 记录填充信息
        pad_info = (w0, h0, scale, dw_left, dh_top)
        return im, pad_info

    @staticmethod
    def get_real_box(boxes, pad_info):
        """映射检测框到原始图像坐标（静态方法，支持多线程调用）"""
        if pad_info is None or boxes is None or len(boxes) == 0:
            return boxes

        w0, h0, scale, dw, dh = pad_info
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale

        # 裁剪到图像边界
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w0)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h0)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w0)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h0)
        return boxes

# ----------------------------
# 后处理核心函数（支持并行）
# ----------------------------
def sigmoid_fast(x):
    """快速sigmoid"""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def filter_boxes(boxes, box_confidences, box_class_probs):
    """轻量化过滤框（支持类别差异化阈值）"""
    try:
        if boxes is None or box_confidences is None or box_class_probs is None:
            return None, None, None
        
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        # ========== 关键修改：按类别应用不同阈值 ==========
        # 初始化掩码为全False
        mask = np.zeros_like(class_max_score, dtype=bool)
        # 遍历每个类别，应用对应阈值
        for cls_idx in np.unique(classes):
            if cls_idx < 0 or cls_idx >= len(CLASSES):
                continue
            # 获取该类别的阈值（自定义或默认）
            cls_name = CLASSES[cls_idx]
            cls_thresh = CLASS_OBJ_THRESH.get(cls_name, DEFAULT_OBJ_THRESH)
            # 找到该类别的所有检测框，应用阈值
            cls_mask = (classes == cls_idx) & (class_max_score * box_confidences >= cls_thresh)
            mask = mask | cls_mask
        # ==================================================

        scores = (class_max_score * box_confidences)[mask]
        boxes = boxes[mask]
        classes = classes[mask]
        return boxes, classes, scores
    except Exception as e:
        print(f"过滤框失败: {e}")
        return None, None, None

def nms_boxes_fast(boxes, scores):
    """快速NMS"""
    if len(boxes) == 0:
        return []
    try:
        indices = cv2.dnn.NMSBoxes(
            boxes[:, :4].tolist(), scores.tolist(),
            DEFAULT_OBJ_THRESH,  # NMS仍用默认阈值，或按需修改
            NMS_THRESH
        )
        if isinstance(indices, (np.ndarray, list)):
            indices = np.array(indices).flatten()
        return indices
    except Exception as e:
        print(f"NMS失败: {e}")
        return []

def dfl_fast(position):
    """快速DFL"""
    try:
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        
        y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
        y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
        
        acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        y = (y_softmax * acc_metrix).sum(2)
        return y
    except Exception as e:
        print(f"DFL失败: {e}")
        return None

def box_process_fast(position):
    """快速边界框计算"""
    if position is None:
        return None
    try:
        grid_h, grid_w = position.shape[2:4]
        col = np.arange(grid_w).repeat(grid_h).reshape(grid_w, grid_h).T
        row = np.arange(grid_h).repeat(grid_w).reshape(grid_h, grid_w)
        grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w)
        
        stride_w = MODEL_SIZE[0] // grid_w
        stride_h = MODEL_SIZE[1] // grid_h
        stride = np.array([stride_w, stride_h], dtype=np.float32).reshape(1, 2, 1, 1)

        position = dfl_fast(position)
        if position is None:
            return None
        
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    except Exception as e:
        print(f"边界框计算失败: {e}")
        return None

def post_process_single(input_data, pad_info):
    """单帧后处理（支持并行调用）"""
    start_time = time.time()
    try:
        if input_data is None or len(input_data) == 0:
            return None, None, None, pad_info, 0

        boxes, scores, classes_conf = [], [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch

        for i in range(defualt_branch):
            idx = pair_per_branch * i
            if idx >= len(input_data):
                continue
            box = box_process_fast(input_data[idx])
            if box is None:
                continue
            boxes.append(box)
            if idx + 1 < len(input_data):
                classes_conf.append(input_data[idx + 1])
                scores.append(np.ones_like(input_data[idx + 1][:, :1, :, :], dtype=np.float32))

        # 展平
        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

        boxes = [sp_flatten(v) for v in boxes if v is not None]
        if not boxes:
            return None, None, None, pad_info, time.time() - start_time
        boxes = np.concatenate(boxes)
        if not classes_conf:
            return None, None, None, pad_info, time.time() - start_time
        classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
        scores = np.concatenate([sp_flatten(v) for v in scores])

        # 过滤+NMS（已修改为支持类别差异化阈值）
        boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
        if boxes is None or len(boxes) == 0:
            return None, None, None, pad_info, time.time() - start_time

        # 按类别NMS
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b, c_cls, s = boxes[inds], classes[inds], scores[inds]
            keep = nms_boxes_fast(b, s)
            if len(keep) > 0:
                nboxes.append(b[keep])
                nclasses.append(c_cls[keep])
                nscores.append(s[keep])

        if not nboxes:
            return None, None, None, pad_info, time.time() - start_time
        
        # 映射到原始图像坐标
        final_boxes = ImagePreprocessor.get_real_box(np.concatenate(nboxes), pad_info)
        final_classes = np.concatenate(nclasses)
        final_scores = np.concatenate(nscores)
        
        return final_boxes, final_classes, final_scores, pad_info, time.time() - start_time
    except Exception as e:
        print(f"后处理失败: {e}")
        return None, None, None, pad_info, time.time() - start_time

def draw_detections(image, boxes, scores, classes, fps, time_info):
    """绘制检测结果（显示各环节耗时）"""
    img_copy = image.copy()
    
    # 显示FPS和各环节耗时
    time_text = f"Cap: {time_info['capture']:.2f}ms | Pre: {time_info['preprocess']:.2f}ms | Inf: {time_info['inference']:.2f}ms | Post: {time_info['postprocess']:.2f}ms"
    cv2.putText(img_copy, f"FPS: {fps:.1f} | {time_text}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img_copy, f"RK3588 Full Parallel | 960×960 | Class-Specific Thresholds", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    if boxes is None or len(boxes) == 0:
        return img_copy
    
    try:
        for box, score, cl in zip(boxes, scores, classes):
            # ========== 新增：显示该类别的阈值 ==========
            cls_name = CLASSES[cl] if cl < len(CLASSES) else "Unknown"
            cls_thresh = CLASS_OBJ_THRESH.get(cls_name, DEFAULT_OBJ_THRESH)
            # ==========================================
            
            x1, y1, x2, y2 = [int(round(b)) for b in box]
            x1 = max(0, min(x1, image.shape[1]-1))
            y1 = max(0, min(y1, image.shape[0]-1))
            x2 = max(0, min(x2, image.shape[1]-1))
            y2 = max(0, min(y2, image.shape[0]-1))
            
            color = color_palette[cl].tolist()
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            # ========== 修改：标签显示类别阈值 ==========
            label = f"{cls_name} {score:.2f} (thresh:{cls_thresh})"
            # ==========================================
            text_pos = (x1, y1 - 10) if y1 > 10 else (x1, y1 + 20)
            cv2.putText(img_copy, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    except Exception as e:
        print(f"绘制失败: {e}")
    
    return img_copy

# ----------------------------
# RKNN初始化
# ----------------------------
def init_rknn_core(core_id):
    """创建并初始化单个RKNN实例，绑定到指定NPU核"""
    rknn = RKNNLite()
    ret = rknn.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print(f"[ERROR] Core{core_id} 加载RKNN模型失败！")
        return None
    
    core_mask = 1 << core_id
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        print(f"[ERROR] Core{core_id} 初始化RKNN失败！")
        return None
    print(f"[INFO] Core{core_id} RKNN初始化成功（mask={core_mask}）")
    return rknn

def init_all_rknn_cores():
    global rknn_instances
    for core_id in NPU_CORES:
        rknn = init_rknn_core(core_id)
        if rknn:
            rknn_instances.append(rknn)
    if len(rknn_instances) != len(NPU_CORES):
        print(f"[ERROR] 仅成功初始化 {len(rknn_instances)} 个RKNN实例（预期{len(NPU_CORES)}）")
        exit(1)
    print("[INFO] 所有NPU核RKNN实例初始化完成")

# ----------------------------
# 1. 采集线程（仅负责帧采集）
# ----------------------------
# ----------------------------
# 1. 采集线程（仅负责帧采集）
# ----------------------------
def capture_thread():
    global EXIT_FLAG  # 仅保留EXIT_FLAG
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] 摄像头{CAMERA_DEVICE}打开失败")
        EXIT_FLAG = True
        return
    
    # ========== 关键优化：摄像头参数 ==========
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 强制MJPG硬件解码
    
    # 清空摄像头缓冲区（关键！）
    for _ in range(5):
        cap.read()
    
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 摄像头已打开 - 分辨率: {cam_width}×{cam_height} (MJPG硬件解码)")
    print("[INFO] 按ESC退出检测，按s保存当前帧")

    while not EXIT_FLAG:
        capture_start = time.time()
        ret, frame = cap.read()
        capture_time = (time.time() - capture_start) * 1000
        
        if not ret:
            time.sleep(0.001)
            continue
        
        # ========== 优化：只保留最新帧（丢弃旧帧） ==========
        with stats_lock:
            time_stats["capture"].append(capture_time)
            if len(time_stats["capture"]) > 10:  # 缩短统计窗口，更实时
                time_stats["capture"].pop(0)
        
        # 关键：队列满时清空旧帧，只留最新的
        try:
            while True:
                raw_frame_queue.get_nowait()
        except Empty:
            pass
        
        raw_frame_queue.put({
            "frame": frame,
            "timestamp": time.time(),
            "capture_time": capture_time
        }, timeout=QUEUE_TIMEOUT)
    
    cap.release()
    print("[INFO] 采集线程已退出")

# ----------------------------
# 2. 预处理线程池（并行处理）
# ----------------------------
def preprocess_worker(frame_data):
    """预处理工作函数（单个帧）"""
    pre_start = time.time()
    preprocessor = ImagePreprocessor()
    frame = frame_data["frame"]
    
    # 执行预处理
    img_processed, pad_info = preprocessor.letter_box(frame.copy())
    if img_processed is None:
        return None
    
    # 转换为RGB并添加batch维度
    img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
    
    preprocess_time = (time.time() - pre_start) * 1000
    
    # 记录预处理耗时
    with stats_lock:
        time_stats["preprocess"].append(preprocess_time)
        if len(time_stats["preprocess"]) > 30:
            time_stats["preprocess"].pop(0)
    
    return {
        "raw_frame": frame,
        "input_data": input_data,
        "pad_info": pad_info,
        "timestamp": frame_data["timestamp"],
        "capture_time": frame_data["capture_time"],
        "preprocess_time": preprocess_time
    }

def preprocess_thread_pool():
    """预处理线程池管理器"""
    global EXIT_FLAG
    executor = ThreadPoolExecutor(max_workers=PREPROCESS_THREADS)
    print(f"[INFO] 预处理线程池启动 - 线程数: {PREPROCESS_THREADS}")
    
    futures = set()
    while not EXIT_FLAG:
        # 从原始帧队列获取数据
        try:
            frame_data = raw_frame_queue.get(timeout=QUEUE_TIMEOUT)
            # 提交到线程池
            future = executor.submit(preprocess_worker, frame_data)
            futures.add(future)  # 只存future，不存字典
        except Empty:
            pass
        
        # 处理完成的预处理任务
        completed_futures = []
        for future in futures:
            if future.done():
                result = future.result()
                if result is not None:
                    # 放入预处理后队列
                    if preprocessed_queue.full():
                        try: preprocessed_queue.get_nowait()
                        except Empty: pass
                    preprocessed_queue.put(result, timeout=QUEUE_TIMEOUT)
                completed_futures.append(future)
        
        # 清理已完成的future
        for f in completed_futures:
            futures.remove(f)
        
        # 避免CPU空转
        time.sleep(0.001)
    
    # 关闭线程池
    executor.shutdown(wait=True)
    print("[INFO] 预处理线程池已关闭")

# ----------------------------
# 3. 推理线程（3个，绑定不同NPU核）
# ----------------------------
def inference_thread(thread_id, core_id, rknn_instance):
    """单个推理线程"""
    global EXIT_FLAG
    print(f"[INFO] 推理线程{thread_id}启动，绑定NPU Core{core_id}")
    
    while not EXIT_FLAG:
        try:
            # 从预处理队列获取数据
            pre_data = preprocessed_queue.get(timeout=QUEUE_TIMEOUT)
            
            # 执行推理
            infer_start = time.time()
            outputs = rknn_instance.inference(inputs=[pre_data["input_data"]])
            infer_time = (time.time() - infer_start) * 1000
            
            # 记录推理耗时
            with stats_lock:
                time_stats["inference"].append(infer_time)
                if len(time_stats["inference"]) > 30:
                    time_stats["inference"].pop(0)
            
            # 放入推理结果队列
            if inference_result_queue.full():
                try: inference_result_queue.get_nowait()
                except Empty: pass
            
            inference_result_queue.put({
                "raw_frame": pre_data["raw_frame"],
                "outputs": outputs,
                "pad_info": pre_data["pad_info"],
                "timestamp": pre_data["timestamp"],
                "capture_time": pre_data["capture_time"],
                "preprocess_time": pre_data["preprocess_time"],
                "inference_time": infer_time,
                "core_id": core_id
            }, timeout=QUEUE_TIMEOUT)
            
        except Empty:
            time.sleep(0.001)
            continue
        except Exception as e:
            print(f"[ERROR] 推理线程{thread_id}(Core{core_id})出错：{e}")
            continue
    
    print(f"[INFO] 推理线程{thread_id}(Core{core_id})已退出")

# ----------------------------
# 4. 后处理线程池（并行处理）- 修复字典不可哈希错误
# ----------------------------
def postprocess_thread_pool():
    """后处理线程池管理器"""
    global EXIT_FLAG
    executor = ThreadPoolExecutor(max_workers=POSTPROCESS_THREADS)
    print(f"[INFO] 后处理线程池启动 - 线程数: {POSTPROCESS_THREADS}")
    
    # 改用列表存储(future, infer_data)，避免set的哈希问题
    futures_list = []  
    while not EXIT_FLAG:
        # 从推理结果队列获取数据
        try:
            infer_data = inference_result_queue.get(timeout=QUEUE_TIMEOUT)
            # 提交到后处理线程池
            future = executor.submit(
                post_process_single,
                infer_data["outputs"],
                infer_data["pad_info"]
            )
            # 存入列表（列表支持不可哈希元素）
            futures_list.append((future, infer_data))
        except Empty:
            pass
        
        # 处理完成的后处理任务
        completed_indices = []
        for idx, (future, infer_data) in enumerate(futures_list):
            if future.done():
                boxes, classes, scores, _, postprocess_time = future.result()
                postprocess_time *= 1000  # 转毫秒
                
                # 记录后处理耗时
                with stats_lock:
                    time_stats["postprocess"].append(postprocess_time)
                    if len(time_stats["postprocess"]) > 30:
                        time_stats["postprocess"].pop(0)
                
                # 计算总耗时
                total_time = (time.time() - infer_data["timestamp"]) * 1000
                with stats_lock:
                    time_stats["total"].append(total_time)
                    if len(time_stats["total"]) > 30:
                        time_stats["total"].pop(0)
                
                # 放入后处理结果队列
                if postprocessed_queue.full():
                    try: postprocessed_queue.get_nowait()
                    except Empty: pass
                
                postprocessed_queue.put({
                    "raw_frame": infer_data["raw_frame"],
                    "boxes": boxes,
                    "classes": classes,
                    "scores": scores,
                    "timestamp": infer_data["timestamp"],
                    "time_info": {
                        "capture": infer_data["capture_time"],
                        "preprocess": infer_data["preprocess_time"],
                        "inference": infer_data["inference_time"],
                        "postprocess": postprocess_time,
                        "total": total_time
                    }
                }, timeout=QUEUE_TIMEOUT)
                
                completed_indices.append(idx)
        
        # 逆序删除已完成的元素（避免索引错乱）
        for idx in sorted(completed_indices, reverse=True):
            del futures_list[idx]
        
        time.sleep(0.001)
    
    executor.shutdown(wait=True)
    print("[INFO] 后处理线程池已关闭")

# ----------------------------
# 5. 显示和保存线程
# ----------------------------
def display_and_save_thread():
    global EXIT_FLAG
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    
    frame_count, start_time, current_fps = 0, time.time(), 0
    avg_time_info = {
        "capture": 0,
        "preprocess": 0,
        "inference": 0,
        "postprocess": 0
    }

    while not EXIT_FLAG:
        try:
            # 获取后处理结果
            post_data = postprocessed_queue.get(timeout=QUEUE_TIMEOUT)
            frame = post_data["raw_frame"]
            boxes = post_data["boxes"]
            classes = post_data["classes"]
            scores = post_data["scores"]
            time_info = post_data["time_info"]
            
            # 计算平均耗时
            with stats_lock:
                if time_stats["capture"]:
                    avg_time_info["capture"] = np.mean(time_stats["capture"])
                if time_stats["preprocess"]:
                    avg_time_info["preprocess"] = np.mean(time_stats["preprocess"])
                if time_stats["inference"]:
                    avg_time_info["inference"] = np.mean(time_stats["inference"])
                if time_stats["postprocess"]:
                    avg_time_info["postprocess"] = np.mean(time_stats["postprocess"])
            
            # 计算FPS
            frame_count += 1
            if time.time() - start_time >= 1:
                current_fps = frame_count / (time.time() - start_time)
                frame_count, start_time = 0, time.time()
                # 打印平均耗时
                print(f"\n[性能统计] 平均耗时(ms) - 采集: {avg_time_info['capture']:.2f} | 预处理: {avg_time_info['preprocess']:.2f} | 推理: {avg_time_info['inference']:.2f} | 后处理: {avg_time_info['postprocess']:.2f} | FPS: {current_fps:.1f}")
            
            # 绘制检测结果（已修改为显示类别阈值）
            frame_draw = draw_detections(frame.copy(), boxes, scores, classes, current_fps, avg_time_info)
            
            # 显示画面
            cv2.imshow(WINDOW_NAME, frame_draw)
            
            # 键盘交互
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                EXIT_FLAG = True
                print("\n[INFO] 按下ESC键退出")
                break
            elif key == ord('s'):  # 保存帧
                save_path = f"camera_capture_{int(time.time())}.jpg"
                cv2.imwrite(save_path, frame_draw)
                print(f"[INFO] 当前帧已保存至: {save_path}")
                
        except Empty:
            continue
        except Exception as e:
            print(f"[ERROR] 显示线程出错：{e}")
            continue
    
    cv2.destroyAllWindows()
    print("[INFO] 显示窗口已关闭，处理线程已退出")

# ----------------------------
# 主函数
# ----------------------------
def main():
    global EXIT_FLAG, start_running_time, rknn_instances
    
    # 设置OpenCV显示环境
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    os.environ['DISPLAY'] = ':0'
    
    
    EXIT_FLAG = False

    # 1. 初始化3个RKNN实例
    init_all_rknn_cores()
    if not rknn_instances:
        print("[ERROR] RKNN实例初始化失败")
        return

    # 2. 启动各线程/线程池
    threads = []
    
    # 采集线程
    capture_t = threading.Thread(target=capture_thread, daemon=True)
    threads.append(capture_t)
    capture_t.start()
    
    # 预处理线程池
    preprocess_t = threading.Thread(target=preprocess_thread_pool, daemon=True)
    threads.append(preprocess_t)
    preprocess_t.start()
    
    # 后处理线程池
    postprocess_t = threading.Thread(target=postprocess_thread_pool, daemon=True)
    threads.append(postprocess_t)
    postprocess_t.start()
    
    # 推理线程（3个）
    inference_threads = []
    for i, (core_id, rknn_inst) in enumerate(zip(NPU_CORES, rknn_instances)):
        t = threading.Thread(target=inference_thread, args=(i, core_id, rknn_inst), daemon=True)
        inference_threads.append(t)
        threads.append(t)
        t.start()
    
    # 3. 启动显示线程（主线程执行）
    try:
        display_and_save_thread()
    except KeyboardInterrupt:
        EXIT_FLAG = True
        print("\n[INFO] Ctrl+C退出")

    # 4. 等待所有线程结束
    for t in threads:
        t.join(timeout=3.0)
    
    # 5. 释放RKNN资源
    for rknn in rknn_instances:
        rknn.release()
    
    # 打印最终性能统计
    print("\n==================== 最终性能统计 ====================")
    with stats_lock:
        if time_stats["capture"]:
            print(f"采集平均耗时: {np.mean(time_stats['capture']):.2f}ms")
        if time_stats["preprocess"]:
            print(f"预处理平均耗时: {np.mean(time_stats['preprocess']):.2f}ms")
        if time_stats["inference"]:
            print(f"推理平均耗时: {np.mean(time_stats['inference']):.2f}ms")
        if time_stats["postprocess"]:
            print(f"后处理平均耗时: {np.mean(time_stats['postprocess']):.2f}ms")
        if time_stats["total"]:
            print(f"端到端平均耗时: {np.mean(time_stats['total']):.2f}ms")
    
    print("[INFO] 所有资源已释放，程序结束")

if __name__ == "__main__":
    main()