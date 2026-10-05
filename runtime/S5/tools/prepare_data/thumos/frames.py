import os
import cv2
from decord import VideoReader, cpu

# ===== 路径配置 =====
video_root = "/nvme3/stu04/OpenTAD-main/data/thumos-14/raw_data/video"
save_root = "/nvme3/stu04/qwen/data/thumos"

os.makedirs(save_root, exist_ok=True)

# 支持的常见视频格式
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm")

def extract_video(video_path, save_dir):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)

    os.makedirs(save_dir, exist_ok=True)

    for idx in range(num_frames):
        frame = vr[idx].asnumpy()          # (H, W, 3), RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(save_dir, f"{idx}.jpg")
        cv2.imwrite(save_path, frame)

    print(f"[OK] {os.path.basename(video_path)} -> {num_frames} frames")

def main():
    videos = sorted(os.listdir(video_root))

    for name in videos:
        if not name.lower().endswith(VIDEO_EXTS):
            continue

        video_path = os.path.join(video_root, name)
        video_name = os.path.splitext(name)[0]
        save_dir = os.path.join(save_root, video_name)

        # 避免重复处理
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
            print(f"[SKIP] {video_name}")
            continue

        extract_video(video_path, save_dir)

if __name__ == "__main__":
    main()
