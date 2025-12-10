from typing import Tuple, List, Optional
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import time
import sys
import argparse
import gzip
import base64
import numpy as np
import mss          # -> screenshot capture module

from skimage.metrics import structural_similarity as ssim

import torch
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor, AutoModelForImageTextToText, AutoProcessor

torch_dtype = torch.float32
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore")

import ctypes
import ctypes.wintypes as wintypes
import os

user32 = ctypes.windll.user32
psapi = ctypes.windll.psapi
kernel32 = ctypes.windll.kernel32

EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
GetWindowThreadProcessId = user32.GetWindowThreadProcessId
OpenProcess = kernel32.OpenProcess
GetModuleFileNameExW = psapi.GetModuleFileNameExW
CloseHandle = kernel32.CloseHandle

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

# Win32 ShowWindow constants
SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
SW_RESTORE = 9
SW_MINIMIZE = 6

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser()
    #parser.add_argument('--exe_name', default='BlackDesert64.exe', help='Name of the EXE which spawns the process (default: BlackDesert64.exe)')
    parser.add_argument('--exe_name', default='notepad.exe', help='Name of the EXE which spawns the process (default: BlackDesert64.exe)')

    parser.add_argument('x1', type=int, help='The X coordinate of the top corner of the guild-only-chat-window')
    parser.add_argument('y1', type=int, help='The Y coordinate of the top corner of the guild-only-chat-window')

    parser.add_argument('x2', type=int, help='The X coordinate of the bottom corner of the guild-only-chat-window')
    parser.add_argument('y2', type=int, help='The Y coordinate of the bottom corner of the guild-only-chat-window')

    return parser


def restore_window(hwnd: int):
    """Restore a minimized window to normal view."""
    try:
        ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
    except Exception:
        pass


def minimize_window(hwnd: int):
    """Minimize a window."""
    try:
        #win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
    except Exception:
        pass


def screenshot_and_crop(crop_box: Tuple[int, int, int, int]) -> Image:
    """
    crop_box: (left, top, right, bottom) relative to the primary monitor
    """
    with mss.mss() as sct:
        #print(f"Capturing region: {str(crop_box)}")
        shot = sct.grab(crop_box)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    return img


def format_guild_lines(items: List[str]) -> str:
    """
    Group items into lines starting with 'Guild'
    """

    lines = []
    current = []
    for t in items:
        if t == "Guild" or t == "Gulld":
            # start a new line
            if current:
                lines.append(" ".join(current))
            current = ["Guild"]
        else:
            current.append(t)
    if current:
        lines.append(" ".join(current))

    # return one single string
    return "\n".join(lines)


def format_guild_lines2(text: Optional[str]) -> list[str]:
    """
    Group items into lines starting with 'Guild'
    """

    if text is None:
        return ""

    lines = text.split("\n")
    lines = [x for x in lines if x != "Guild" or ""]

    return lines

def get_last_line(lines: list[str]) -> str:
    last_item = lines[-1]
    return last_item


def compress_and_encode(text: str) -> str:
    """
    Compress a string with gzip and encode it as base64.
    Returns the base64-encoded string.
    """

    # Convert each string to bytes
    text_bytes = text.encode("utf-8")

    # Compress each string with gzip
    compressed = gzip.compress(text_bytes)

    # Encode with base64
    b64encoded = base64.b64encode(compressed).decode("utf-8")

    return b64encoded


def write_to_file(encoded_result: str, filename: str = "output.txt"):
    """
    Write the base64-encoded gzip string to a file.
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(encoded_result + "\n")


def get_first_hwnd_by_exe(exe_name: str) -> int:
    result_hwnd = None

    def callback(hwnd, lParam):
        nonlocal result_hwnd
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        h_process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid.value)
        if h_process:
            exe_path = (ctypes.c_wchar * 260)()
            if GetModuleFileNameExW(h_process, None, exe_path, 260) > 0:
                if os.path.basename(exe_path.value).lower() == exe_name.lower():
                    result_hwnd = hwnd
                    CloseHandle(h_process)
                    return False  # stop enumeration once found
            CloseHandle(h_process)
        return True  # continue enumeration

    EnumWindows(EnumWindowsProc(callback), 0)
    return result_hwnd


def is_window_minimized(hwnd: int) -> bool:
    # IsIconic returns nonzero if the window is minimized (iconic)
    return bool(user32.IsIconic(hwnd))


def read_text_easyocr(img: Image) -> List[str]:
    import easyocr
    reader = easyocr.Reader(['en'])
    img_np = np.array(img)
    results = reader.readtext(img_np)

    # Extract only the text
    texts = [text for (_, text, _) in results]

    return texts


def load_ocr_model() -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load Caption model and processor with proper data type handling."""
    try:
        print("Loading ocr model Qwen/Qwen3-VL-2B-Instruct...")

        repo_id = "Qwen/Qwen3-VL-2B-Instruct"
        model = (AutoModelForImageTextToText.from_pretrained(repo_id, device_map="auto", dtype="auto"))
        processor = AutoProcessor.from_pretrained(repo_id, dtype="auto")

        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")

        return model, processor
    except Exception as e:
        print(f"Error loading ocr model: {e}")
        sys.exit(1)


def generate_ocr(model: AutoModelForImageTextToText,
                     processor: AutoProcessor,
                     img: Image) -> Optional[str]:
    """Extract text from an image"""
    prompt = f"Extract the text from the image, only answer with the extracted text. Be very precise when extracting the images text. If no text is visible, just return 'no text can be extracted'"

    try:
        image = img.convert('RGB')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Produce text with image tokens
        text_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )

        # Build multimodal inputs (text + image)
        inputs = processor(
            images=image,
            text=text_inputs,
            return_tensors="pt",
        )

        # Move inputs to same device/dtype as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
            )

        # Decode the generated text
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        # Remove everything upto (including) the word "assistant"

        marker = "assistant"

        # Use find() to get the index of the first occurrence (of the marker)
        start_index = generated_text.find(marker) + len(marker)

        # Slice the string from that index to the end
        generated_text = generated_text[start_index:]

        return generated_text
    except Exception:
        print(f"Error extracting text from image")
        import traceback
        traceback.print_exc()
        return None

def load_upscaler_model() -> Optional[Tuple[Swin2SRImageProcessor, Swin2SRForImageSuperResolution]]:
    """Load the DiffusionPipeline for upscaling."""
    print("Loading Upscaler-Ultra model...")

    try:
        upscale_processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64",
                                                                  trust_remote_code=True)
        upscale_model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64",
            trust_remote_code=True,
            dtype=torch_dtype
        )
        print(f"Model loaded on cpu with dtype {torch_dtype}")
        return upscale_processor, upscale_model
    except Exception as e:
        print(f"Warning: Could not load upscaler model. Skipping upscaling. Error: {e}")
        # Return a mock object or None if upscaler is critical
        return None


def sharpen_image(img: Image, upscale_processor: Swin2SRImageProcessor, upscale_model: Swin2SRForImageSuperResolution) -> Image:
    """Sharpen image using Upscaler-Ultra before further processing."""
    if upscale_processor or upscale_model is None:
        return img.convert("RGB")

    input_image = img.convert("RGB")

    inputs = upscale_processor(images=input_image, return_tensors="pt")
    pixel_values = inputs['pixel_values']

    # 3. Run Model
    with torch.no_grad():
        outputs = upscale_model(pixel_values=pixel_values)

    # Correct Post-processing
    # The upscaled tensor is accessed via the 'reconstruction' attribute
    upscaled_tensor = outputs.reconstruction

    # Move to CPU, remove batch dim (squeeze), clip, and convert to NumPy
    # The tensor is in (1, C, H, W) format, normalized to [0, 1].
    upscaled_np = upscaled_tensor.squeeze(0).cpu().clamp(0, 1).numpy()

    # Scale to [0, 255] and change data type to 8-bit integer
    upscaled_np = (upscaled_np * 255.0).astype(np.uint8)

    # Permute dimensions from (C, H, W) to (H, W, C) for PIL
    upscaled_np = np.transpose(upscaled_np, (1, 2, 0))

    # Convert NumPy array to PIL Image object
    upscaled_image = Image.fromarray(upscaled_np)

    return upscaled_image

def preprocess_image(img: Image, upscale_processor: Swin2SRImageProcessor, upscale_model: Swin2SRForImageSuperResolution) -> Image:
    """Apply preprocessing to improve OCR accuracy"""

    image = sharpen_image(img, upscale_processor, upscale_model)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    # Sharpen slightly
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)

    # Remove noise
    image = image.filter(ImageFilter.MedianFilter(size=1))

    return img

def get_similarity_score(img1: Image, img2: Image) -> int:
    # Convert to same mode and size
    img1 = img1.convert("RGB").resize((256, 256))
    img2 = img2.convert("RGB").resize((256, 256))

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Mean squared error
    mse = np.mean((arr1 - arr2) ** 2)

    # Structural similarity (requires skimage)
    similarity_score, _ = ssim(arr1, arr2, channel_axis=2, full=True)

    return similarity_score


if __name__ == "__main__":
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    exe_name = args.exe_name

    x1 = args.x1
    y1 = args.y1

    x2 = args.x2
    y2 = args.y2

    print("Hooking Process...")

    hwnd = None
    first_run = True
    previous_image = None

    while True:
        hwnd = get_first_hwnd_by_exe(exe_name)
        if hwnd:
            if first_run:
                print(f"Found HWND: {hwnd}")
                # load required ai models
                text_extract_model, text_extract_processor = load_ocr_model()
                upscale_processor, upscale_model = load_upscaler_model()

            # tuple of four integers tuple[left, top, right, bottom] or tuple[x1,y1,x2,y2]
            crop_box = (x1, y1, x2, y2)

            # small_chat crop_box = (303, 595, 620, 745) # small chat
            # big_chat crop_box = (303, 595, 827, 861) # big chat

            img = screenshot_and_crop(crop_box)
            img = preprocess_image(img, upscale_processor, upscale_model)

            similarity_score_threshold = 0.6

            previous_img_save_filename = "extracted.png"

            if not Path(previous_img_save_filename).exists():
                similarity_score = similarity_score_threshold + 0.01
            else:
                previous_image = Image.open(previous_img_save_filename)
                similarity_score = get_similarity_score(img, previous_image)
                #print("SSIM/similarity score:", similarity_score)

            was_window_minimized = False
            try:
                # check if the previouses image similaity is less than the specified threshold (equal=1.0)
                if similarity_score < similarity_score_threshold:

                    if is_window_minimized(hwnd):
                        restore_window(hwnd)
                        was_window_minimized = True

                    # save the current image
                    img.save(previous_img_save_filename)

                    full_text = generate_ocr(
                        model=text_extract_model,
                        processor=text_extract_processor,
                        img=img)

                    if full_text is not None or "no text can be extracted" in full_text:
                        #print(f"\nExtracted text:\n{full_text}")
                        pass
                    else:
                        print("\nNo extracted text")
                        continue

                    output = format_guild_lines2(full_text)

                    #print("\nformat text:" + str(output))

                    ll = get_last_line(output)

                    print("\nlast line:" + str(ll))

                    #encoded_result = compress_and_encode(ll)

                    #print(f"\nencoded: {encoded_result}")

                    #write_to_file(encoded_result)
                    write_to_file(ll)
                else:
                    print(f"Images are too similar. Captured image similarity {similarity_score}, expected <{similarity_score_threshold}")
                    time.sleep(2)
                    continue
            finally:
                # minimize the window again, if it was minimized
                if was_window_minimized:
                    minimize_window(hwnd)
                first_run = False
        else:
            print("Window not found")

        seconds_to_wait = 10
        print(f"Waiting {seconds_to_wait}s ...")
        time.sleep(seconds_to_wait)
