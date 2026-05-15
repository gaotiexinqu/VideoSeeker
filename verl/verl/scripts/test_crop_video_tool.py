"""
Test script for CropVideoTool variants.
Compares three modes:
  1. Current: fetch_video(..., image_patch_size=16) + process_image(img)
  2. No patch size: fetch_video(...) (no image_patch_size) + process_image(img)
  3. process_video path: directly use process_video() and return video= instead of image=

Video: /mnt/tidal-alsh01/dataset/zeus/zhaoy/DATA/-YwrMtiqHKg.mp4
"""

import asyncio
import os
import sys
import time

import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Ensure the verl package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.dataset.vision_utils import process_image, process_video
from qwen_vl_utils import fetch_video


VIDEO_PATH = "/mnt/tidal-alsh01/dataset/zeus/zhaoy/DATA/-YwrMtiqHKg.mp4"
START_TIME = 0.0
END_TIME = 5.0


def print_sep(title: str):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def inspect_image(label: str, img, indent: int = 2):
    prefix = " " * indent
    if isinstance(img, Image.Image):
        print(f"{prefix}[{label}] type=PIL.Image | size={img.size} | mode={img.mode}")
    elif isinstance(img, torch.Tensor):
        print(f"{prefix}[{label}] type=torch.Tensor | shape={img.shape} | dtype={img.dtype}")
    else:
        print(f"{prefix}[{label}] type={type(img).__name__} | value={img}")


async def test_mode_1_with_patch_size():
    """Mode 1: current implementation — fetch_video(..., image_patch_size=16) + process_image"""
    print_sep("Mode 1: fetch_video(image_patch_size=16) + process_image")

    video_ele = {
        "type": "video",
        "video": VIDEO_PATH,
        "fps": 1,
        "min_frames": 4,
        "max_frames": 8,
        "min_pixels": 12544,
        "max_pixels": 50176,
        "video_start": START_TIME,
        "video_end": END_TIME,
    }

    t0 = time.perf_counter()
    video_frames = fetch_video(video_ele, image_patch_size=16)
    elapsed = time.perf_counter() - t0

    print(f"  fetch_video elapsed: {elapsed:.3f}s")
    inspect_image("video_frames", video_frames)

    pil_images = [to_pil_image(frame).convert("RGB") for frame in video_frames]
    print(f"  Number of frames: {len(pil_images)}")
    for i, img in enumerate(pil_images):
        inspect_image(f"pil_images[{i}]", img)

    processed_images = [process_image(img) for img in pil_images]
    for i, img in enumerate(processed_images):
        inspect_image(f"processed_images[{i}]", img)

    # Check if sizes are multiples of 28
    for i, img in enumerate(processed_images):
        w, h = img.size
        print(f"  processed_images[{i}]: {w}x{h}, w%28={w%28}, h%28={h%28}")

    return processed_images


async def test_mode_2_no_patch_size():
    """Mode 2: fetch_video without image_patch_size + process_image"""
    print_sep("Mode 2: fetch_video(no image_patch_size) + process_image")

    video_ele = {
        "type": "video",
        "video": VIDEO_PATH,
        "fps": 1,
        "min_frames": 4,
        "max_frames": 8,
        "min_pixels": 12544,
        "max_pixels": 50176,
        "video_start": START_TIME,
        "video_end": END_TIME,
    }

    t0 = time.perf_counter()
    # NOTE: no image_patch_size argument — uses qwen_vl_utils default
    video_frames = fetch_video(video_ele)
    elapsed = time.perf_counter() - t0

    print(f"  fetch_video elapsed: {elapsed:.3f}s")
    inspect_image("video_frames", video_frames)

    pil_images = [to_pil_image(frame).convert("RGB") for frame in video_frames]
    print(f"  Number of frames: {len(pil_images)}")
    for i, img in enumerate(pil_images):
        inspect_image(f"pil_images[{i}]", img)

    processed_images = [process_image(img) for img in pil_images]
    for i, img in enumerate(processed_images):
        inspect_image(f"processed_images[{i}]", img)

    # Check if sizes are multiples of 14 (default patch_size in process_image)
    for i, img in enumerate(processed_images):
        w, h = img.size
        print(f"  processed_images[{i}]: {w}x{h}, w%14={w%14}, h%14={h%14}")

    return processed_images


async def test_mode_3_process_video_direct():
    """Mode 3: use process_video directly and return video= instead of image="""
    print_sep("Mode 3: process_video() + return video= (no image=)")

    video_dict = {
        "type": "video",
        "video": VIDEO_PATH,
        "fps": 1,
        "min_frames": 4,
        "max_frames": 8,
        "min_pixels": 12544,
        "max_pixels": 50176,
        "video_start": START_TIME,
        "video_end": END_TIME,
    }

    t0 = time.perf_counter()
    # process_video internally calls fetch_video with image_patch_size=14 by default
    video_tensor = process_video(video_dict, image_patch_size=14)
    elapsed = time.perf_counter() - t0

    print(f"  process_video elapsed: {elapsed:.3f}s")
    inspect_image("video_tensor", video_tensor)

    # The tensor shape: [n_frames, 3, H, W]
    n_frames = video_tensor.shape[0]
    print(f"  Number of frames: {n_frames}")
    for i in range(n_frames):
        frame = video_tensor[i]  # [3, H, W]
        h, w = frame.shape[1], frame.shape[2]
        print(f"  frame[{i}]: {w}x{h}, w%14={w%14}, h%14={h%14}")

    return video_tensor


def compare_results(mode1_imgs, mode2_imgs, mode3_tensor):
    print_sep("Comparison Summary")
    print(f"{'Mode':<8} {'Output type':<25} {'Frame count':<12} {'Frame size':<20} {'Size mod 14':<12} {'Size mod 28':<12}")
    print("-" * 90)

    def size_str(imgs):
        if not imgs:
            return "N/A"
        return f"{imgs[0].size[0]}x{imgs[0].size[1]}"

    def mod_str(imgs, divisor):
        if not imgs:
            return "N/A"
        w, h = imgs[0].size
        return f"w={w%divisor}, h={h%divisor}"

    print(f"{'Mode 1':<8} {'list[PIL.Image]':<25} {len(mode1_imgs):<12} {size_str(mode1_imgs):<20} {mod_str(mode1_imgs, 14):<12} {mod_str(mode1_imgs, 28):<12}")
    print(f"{'Mode 2':<8} {'list[PIL.Image]':<25} {len(mode2_imgs):<12} {size_str(mode2_imgs):<20} {mod_str(mode2_imgs, 14):<12} {mod_str(mode2_imgs, 28):<12}")
    m3_w, m3_h = mode3_tensor.shape[3], mode3_tensor.shape[2]
    print(f"{'Mode 3':<8} {'torch.Tensor':<25} {mode3_tensor.shape[0]:<12} {m3_w}x{m3_h:<20} "
          f"w={m3_w%14}, h={m3_h%14}    w={m3_w%28}, h={m3_h%28}")

    print()
    print("Key differences:")
    print("  - Mode 1 (image_patch_size=16): uses 16px patch, sizes are multiples of 28 (LCM of 14 and 28)")
    print("  - Mode 2 (no patch_size): qwen_vl_utils default is 14px, process_image uses 14px → multiples of 14")
    print("  - Mode 3 (process_video): returns torch.Tensor [N,3,H,W], H/W multiples of 14 (default)")
    print("  - Mode 1 vs Mode 2: the image_patch_size in fetch_video controls resizing inside qwen_vl_utils")
    print("  - Mode 3 vs Mode 1/2: returns raw tensor in CHW format instead of processed PIL images")


async def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        sys.exit(1)

    # Quick video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    print(f"Video: {VIDEO_PATH}")
    print(f"  Duration: {duration:.2f}s | FPS: {fps:.2f} | Frames: {frame_count}")
    print(f"  Test range: [{START_TIME}s, {END_TIME}s]")

    mode1_imgs = await test_mode_1_with_patch_size()
    mode2_imgs = await test_mode_2_no_patch_size()
    mode3_tensor = await test_mode_3_process_video_direct()

    compare_results(mode1_imgs, mode2_imgs, mode3_tensor)

    print_sep("All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
