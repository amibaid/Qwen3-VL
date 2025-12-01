
import os
import math
import hashlib
import requests
import sys
import json
import csv
import re
import torch
import time
from typing import Optional, List, Dict, Any, Set

sys.path.append('/content/drive/MyDrive/381V-final-project/Qwen3-VL/qwen-vl-utils/src/')

from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


model_path = "Qwen/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)

model, output_loading_info = AutoModelForVision2Seq.from_pretrained(model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    output_loading_info=True,
    attn_implementation="sdpa",
    device_map={"": "cuda"})
print("output_loading_info", output_loading_info)

print(model.hf_device_map)
for name, p in model.named_parameters():
    if p.device.type != "cuda":
        print("ON CPU:", name, p.device)
model.eval()
torch.set_grad_enabled(False)

print("CUDA available:", torch.cuda.is_available())
print("First param device:", next(model.parameters()).device)

def load_video_id_filter(csv_path: Optional[str]) -> Optional[Set[str]]:
    """
    Load a set of video IDs (e.g. P01-20240203-123350) from a CSV file.
    Assumes either:
      - A single column with or without header, or
      - A column named 'video_id'.
    """
    if csv_path is None:
        return None

    video_ids = set()
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Empty file
    if not rows:
        return set()

    # Try to detect header
    header = rows[0]
    if len(header) == 1 and header[0].lower() in {"video_id", "id"}:
        # Skip header row
        for row in rows[1:]:
            if row and row[0].strip():
                video_ids.add(row[0].strip())
    else:
        # Treat all non-empty entries as IDs
        for row in rows:
            if not row:
                continue
            if row[0].strip():
                video_ids.add(row[0].strip())

    return video_ids


def build_mcq_prompt(question: str, choices: List[str]) -> str:
    """
    Build a multiple-choice prompt. The model is asked to output ONLY
    the index of the correct choice (0, 1, 2, ...).
    """
    choices_text = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
    prompt = f"""You are given a video segment and a multiple-choice question about it.

    Question:
    {question}

    Choices:
    {choices_text}

    Respond with the index of the correct choice (0-{len(choices) - 1}) ONLY.
    Do not output any words or explanation, just a single integer."""
    return prompt

def parse_predicted_index(model_output: str, num_choices: int) -> Optional[int]:
    """
    Parse the model's output as an integer index in [0, num_choices-1].
    Returns None if parsing fails.
    """
    text = model_output.strip()
    # Grab first integer that appears
    m = re.search(r"\d+", text)
    if not m:
        return None
    idx = int(m.group(0))
    if 0 <= idx < num_choices:
        return idx
    return None


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)

    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image

def inference(video, prompt, max_new_tokens=2048, total_pixels=20480 * 32 * 32, min_pixels=64 * 32 * 32, max_frames= 2048, sample_fps = 2):
    """
    Perform multimodal inference on input video and text prompt to generate model response.

    Args:
        video (str or list/tuple): Video input, supports two formats:
            - str: Path or URL to a video file. The function will automatically read and sample frames.
            - list/tuple: Pre-sampled list of video frames (PIL.Image or url).
              In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
        prompt (str): User text prompt to guide the model's generation.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
        total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
        min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
        sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
            Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
            Used for temporal alignment or normalization in the model. Default is 2.

    Returns:
        str: Generated text response from the model.

    Notes:
        - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
        - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
    """
    print(time.time(), "starting inference")
    messages = [
        {"role": "user", "content": [
                {"video": video,
                "total_pixels": total_pixels,
                "min_pixels": min_pixels,
                "max_frames": max_frames,
                'sample_fps':sample_fps},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True,
                                                                   image_patch_size= 16,
                                                                   return_video_metadata=True)
    print(time.time(), "processed vision info")
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to('cuda')
    print(time.time(), "put inputs on device")
    print("input_ids shape:", inputs["input_ids"].shape)
    print("total tokens:", inputs["input_ids"].numel())
    print("max_new_tokens", max_new_tokens)

    with torch.inference_mode():
        # output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_ids = model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,  # actually 16 🙂
          do_sample=False,
          num_beams=1,
          use_cache=True,
        )
    print(time.time(), "did generation")

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(time.time(), "finished function")
    return output_text[0]


def evaluate_from_json(
    json_path: str,
    video_dir: str,
    csv_filter_path: Optional[str] = None,
    num_frames: int = 12,
    sample_fps: float = 0.25,
    total_pixels: int = 24 * 1024 * 32 * 32,
    max_new_tokens: int = 256,
  ) -> Dict[str, Any]:
    """
    Evaluate Qwen3-VL on a JSON dataset of video multiple-choice questions.

    Args:
        json_path: Path to the JSON file with structure like in the example.
        video_dir: Directory that contains per-question video clips, named like
                  <q_key>.mp4 (e.g. fine_grained_action_localization_0.mp4).
        csv_filter_path: Optional path to CSV with allowed video IDs. If given,
                        only questions whose q_data.inputs["video 1"]["id"]
                        is in this CSV will be evaluated.
        num_frames: Number of frames to sample per video.
        sample_fps: Sampling FPS passed to the model when using frame lists.
        total_pixels: total_pixels argument forwarded to inference().
        max_new_tokens: max_new_tokens argument forwarded to inference().

    Returns:
        A dict with overall accuracy and per-question details.
    """
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    video_id_filter = load_video_id_filter(csv_filter_path)

    results = []
    n_total = 0
    n_correct = 0

    for q in questions:
        t_q_start = time.perf_counter()

        q_key = q["q_key"]                      # e.g. 'fine_grained_action_localization_0'
        q_data = q["q_data"]
        q_inputs = q_data["inputs"]["video 1"]
        video_id = q_inputs["id"]              # e.g. 'P01-20240203-123350'
        question_text = q_data["question"]
        choices = q_data["choices"]
        correct_idx = q_data["correct_idx"]

        # Filter by CSV video IDs if provided
        if video_id_filter is not None and video_id not in video_id_filter:
            continue

        # Build path to video file: <video_dir>/<q_key>.mp4
        video_filename = f"{q_key}.mp4"
        video_path = os.path.join(video_dir, video_filename)

        if not os.path.exists(video_path):
            print(f"[WARN] Video file not found for {q_key}: {video_path}, skipping.")
            continue

        # Sample frames from the video
        t_frames_start = time.perf_counter()
        try:
            _, frames_np, timestamps = get_video_frames(
                video_path,
                num_frames=num_frames,
                cache_dir=".cache",
            )
        except Exception as e:
            print(f"[ERROR] Failed to read video {video_path}: {e}")
            continue
        t_frames_end = time.perf_counter()

        frames_pil = [Image.fromarray(arr) for arr in frames_np]

        # Build MCQ prompt
        prompt = build_mcq_prompt(question_text, choices)

        torch.cuda.synchronize()
        t_infer_start = time.perf_counter()

        # Call your existing inference() function
        model_output = inference(
            frames_pil,
            prompt,
            sample_fps=sample_fps,
            total_pixels=total_pixels,
            max_new_tokens=max_new_tokens,
        )

        torch.cuda.synchronize()
        t_infer_end = time.perf_counter()
        
        t_q_end = time.perf_counter()

        print(
            f"Q: {q_key} | "
            f"frames={t_frames_end - t_frames_start:.3f}s | "
            f"infer={t_infer_end - t_infer_start:.3f}s | "
            f"total={t_q_end - t_q_start:.3f}s"
        )

        pred_idx = parse_predicted_index(model_output, len(choices))

        is_correct = (pred_idx == correct_idx)
        if pred_idx is not None:
            n_total += 1
            if is_correct:
                n_correct += 1

        results.append({
            "q_key": q_key,
            "video_id": video_id,
            "video_path": video_path,
            "question": question_text,
            "choices": choices,
            "correct_idx": correct_idx,
            "model_output_raw": model_output,
            "pred_idx": pred_idx,
            "is_correct": is_correct,
        })

        print(
            f"Q: {q_key}  |  video_id={video_id}  |  "
            f"pred={pred_idx}  |  gold={correct_idx}  |  correct={is_correct}"
        )

    accuracy = (n_correct / n_total) if n_total > 0 else 0.0
    print("\n=== Evaluation summary ===")
    print(f"Evaluated examples: {n_total}")
    print(f"Correct: {n_correct}")
    print(f"Accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "n_total": n_total,
        "n_correct": n_correct,
        "results": results,
    }

def main():
    JSON_PATH = "/content/drive/MyDrive/381V-final-project/HD-EPIC/test_vqa.json"
    VIDEO_DIR = "/content/drive/MyDrive/381V final project/eval data/trimmed_clips"

    # If you don't want filtering, set CSV_PATH = None
    CSV_PATH = None

    eval_stats = evaluate_from_json(
        json_path=JSON_PATH,
        video_dir=VIDEO_DIR,
        csv_filter_path=CSV_PATH,
        num_frames=8,
        sample_fps=0.25,
        total_pixels=3 * 384 * 384,
        max_new_tokens=16,
    )

if __name__ == "__main__":
    main()


