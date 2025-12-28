import os
import io
import sys
import time
import gzip
import base64
import asyncio
import signal
import ctypes
import psutil
import argparse
import ctypes.wintypes as wintypes
from typing import Tuple, List, Optional
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from functools import wraps

from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor, AutoModelForImageTextToText, AutoProcessor
import torch

from skimage.metrics import structural_similarity as ssim
import numpy as np
import mss          # -> screenshot capture module

def handler(sig, frame):
    print("Ctrl+C captured! Cleaning up...")
    restore_window(hwnd)
    sys.exit(0)  # exit gracefully

# Register the handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handler)

torch_dtype = torch.float32
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

user32 = ctypes.windll.user32

EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
GetWindowThreadProcessId = user32.GetWindowThreadProcessId

IsWindowVisible = user32.IsWindowVisible
IsWindowVisible.argtypes = [wintypes.HWND]
IsWindowVisible.restype = wintypes.BOOL

# Win32 ShowWindow constants
SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
SW_RESTORE = 9
SW_MINIMIZE = 6

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser()
    #parser.add_argument('--exe_name', default='BlackDesert64.exe', help='Name of the EXE which spawns the process (default: BlackDesert64.exe)')
    parser.add_argument('--exe_name', type=str, default='notepad.exe', help='Name of the EXE which spawns the process (default: BlackDesert64.exe)')

    parser.add_argument('x1', type=int, help='The X coordinate of the top corner of the guild-only-chat-window')
    parser.add_argument('y1', type=int, help='The Y coordinate of the top corner of the guild-only-chat-window')

    parser.add_argument('x2', type=int, help='The X coordinate of the bottom corner of the guild-only-chat-window')
    parser.add_argument('y2', type=int, help='The Y coordinate of the bottom corner of the guild-only-chat-window')

    parser.add_argument('--bot_token', type=str, help='The bot token')
    parser.add_argument('--channelid', type=int, help='The channel id, the bot should write to')

    parser.add_argument('--easyocr', action='store_true', default=False, help='Uses easocr to extract text (fast but not acurate, old USE AT OWN RISK)')

    parser.add_argument('--ocrwsusr', default="", help="The username to use OCRWebService.com to extract text")
    parser.add_argument("--ocrwslicense", default="", help="The license to use OCRWebService.com to extract text")

    return parser

def is_window_minimized(hwnd: int) -> bool:
    return IsWindowVisible(hwnd)

def get_hwnds_for_pid(pid):
    hwnds = []

    # Define the callback inside so it can access 'hwnds' list directly
    def callback(hwnd, lParam):
        # Get the PID of the window
        lpdw_pid = ctypes.c_ulong()
        GetWindowThreadProcessId(hwnd, ctypes.byref(lpdw_pid))

        if lpdw_pid.value == pid:
            hwnds.append(hwnd)
        return True

    # Call the API
    EnumWindows(EnumWindowsProc(callback), 0)
    return hwnds

def manage_window(exe_name, action="hide"):
    target_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] and proc.info['name'].lower() == exe_name.lower():
            target_pid = proc.info['pid']
            break

    if not target_pid:
        print(f"Process '{exe_name}' not found.")
        return

    hwnds = get_hwnds_for_pid(target_pid)

    if not hwnds:
        print(f"No windows found for {exe_name}.")
        return

    cmd = SW_HIDE if action == "hide" else SW_RESTORE

    for hwnd in hwnds:
        ctypes.windll.user32.ShowWindow(hwnd, cmd)
        if action == "show":
            ctypes.windll.user32.SetForegroundWindow(hwnd)

    print(f"Action '{action}' applied to {len(hwnds)} window(s) of {exe_name}.")

def restore_window(exe_name: str, hwnd: int):
    """Restore a minimized window to normal view."""
    if is_window_minimized(hwnd):
        manage_window(exe_name, action="show")

def minimize_window(exe_name: str, hwnd: int):
    """Minimize a window."""
    if not is_window_minimized(hwnd):
        manage_window(exe_name, action="hide")

def screenshot_and_crop(crop_box: Tuple[int, int, int, int]) -> Image:
    """
    crop_box: (left, top, right, bottom) relative to the primary monitor
    """
    with mss.mss() as sct:
        #print(f"Capturing region: {str(crop_box)}")
        shot = sct.grab(crop_box)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
    return img

def format_guild_lines(text: Optional[str]) -> list[str] | None:
    """
    Group items into lines
    """

    if text is None:
        return None

    lines = text.split("\n")
    lines = [line for line in lines if line != "Guild" or ""]

    return lines


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
        repo_id = "Qwen/Qwen3-VL-2B-Instruct"
        print(f"Loading model {repo_id}...")

        model = (AutoModelForImageTextToText.from_pretrained(repo_id, device_map="auto", dtype="auto"))
        processor = AutoProcessor.from_pretrained(repo_id, dtype="auto")

        print(f"Model loaded on {torch_device} with dtype {torch_dtype}")

        return model, processor
    except Exception as e:
        print(f"Error loading ocr model: {e}")
        sys.exit(1)

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

async def generate_ocr(model: AutoModelForImageTextToText,
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

        @timer
        def extract_text():
            with torch.no_grad():
                global text_extract_generated_ids
                text_extract_generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_beams=1,
                    do_sample=False,
                )

        await asyncio.get_running_loop().run_in_executor(None, extract_text)

        # Decode the generated text
        generated_text = processor.batch_decode(
            text_extract_generated_ids,
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

def ocrwebservice(img: Image, ocrwsusr: str, ocrwslicense: str) -> Image:
    from zeep import Client

    # Provide your username and license code
    LOGIN = ocrwsusr
    LICENSE = ocrwslicense

    # Service URL
    client = Client('http://www.ocrwebservice.com/services/OCRWebService.asmx?WSDL')

    # FilePath = os.path.abspath(img.filename)
    # with open(FilePath, 'rb') as image_file:
    #     image_data = image_file.read()
    #
    # InputImage = {
    #     'fileName': FilePath,
    #     'fileData': image_data,
    # }

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_data = buf.getvalue()

    InputImage = {
        "fileName": "in_memory.png",
        "fileData": image_data,
    }

    # Specify coordinates for zones (only from these zone text will be extracted)
    # To extract text from all page you do not need setup any zones
    ocrZones = {'OCRWSZone': [{'Top': 0, 'Left': 0, 'Height': 600, 'Width': 400, 'ZoneType': 0},
                              {'Top': 500, 'Left': 1000, 'Height': 150, 'Width': 400, 'ZoneType': 0}]}

    OCRSettings = {
        'ocrLanguages': 'ENGLISH',
        'outputDocumentFormat': 'TXT',
        'convertToBW': 'false',
        'getOCRText': 'true',
        'createOutputDocument': 'false',
        'multiPageDoc': 'false',
        'pageNumbers': 'allpages',
        'ocrZones': ocrZones,
        'ocrWords': 'false',
        'Reserved': '',
    }

    # Perform OCR recognition
    result = client.service.OCRWebServiceRecognize(user_name=LOGIN,
                                                   license_code=LICENSE,
                                                   OCRWSInputImage=InputImage,
                                                   OCRWSSetting=OCRSettings)

    extracted_text = result.ocrText.ArrayOfString[0].string[0]
    return extracted_text

async def sharpen_image(img: Image, upscale_processor: Swin2SRImageProcessor, upscale_model: Swin2SRForImageSuperResolution) -> Image:
    """Sharpen image using Upscaler-Ultra before further processing."""
    if upscale_processor or upscale_model is None:
        return img.convert("RGB")

    input_image = img.convert("RGB")

    inputs = upscale_processor(images=input_image, return_tensors="pt")
    pixel_values = inputs['pixel_values']

    def upscale():
        with torch.no_grad():
            global upscale_outputs
            upscale_outputs = upscale_model(pixel_values=pixel_values)

    await asyncio.get_running_loop().run_in_executor(None, upscale)

    # Correct Post-processing
    # The upscaled tensor is accessed via the 'reconstruction' attribute
    upscaled_tensor = upscale_outputs.reconstruction

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

async def preprocess_image(img: Image, upscale_processor: Swin2SRImageProcessor, upscale_model: Swin2SRForImageSuperResolution) -> Image:
    """Apply preprocessing to improve OCR accuracy"""

    image = await sharpen_image(img, upscale_processor, upscale_model)

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

    # Structural similarity (requires skimage)
    similarity_score, _ = ssim(arr1, arr2, channel_axis=2, full=True)

    return similarity_score

async def get_last_line(
        exe_name: str,
        hwnd: int,
        upscale_model: Swin2SRForImageSuperResolution,
        upscale_processor: Swin2SRImageProcessor,
        crop_box: Tuple[int, int, int, int],
        use_easyocr: bool = False,
        text_extract_model: AutoModelForImageTextToText = None,
        text_extract_processor: AutoProcessor = None,
        ocrwsusr: str = None,
        ocrwslicense: str = None,
        ) -> str | None:
    img = screenshot_and_crop(crop_box)
    img = await preprocess_image(img, upscale_processor, upscale_model)

    similarity_score_threshold = 0.6

    previous_img_save_filename = "extracted.png"

    if not Path(previous_img_save_filename).exists():
        similarity_score = similarity_score_threshold - 0.01
    else:
        global previous_image
        previous_image = Image.open(previous_img_save_filename)
        similarity_score = get_similarity_score(img, previous_image)
        #print("SSIM/similarity score:", similarity_score)

    try:
        # check if the previouses image similaity is less than the specified threshold (equal=1.0)
        if similarity_score < similarity_score_threshold:
            restore_window(exe_name, hwnd)

            # save the current image, as the previous image, for the next iteration
            img.save(previous_img_save_filename)

            print("Extracting text...")

            if ocrwsusr and ocrwslicense:
                full_text = ocrwebservice(img, ocrwsusr, ocrwslicense)
                print(f"text: {full_text}\nthis version is unfinished and will not be printed to discord!!!!!")
            elif use_easyocr:
                full_text = read_text_easyocr(img)
                #output = format_guild_lines(full_text)
                return full_text[-1]
            else:
                full_text = await generate_ocr(
                    model=text_extract_model,
                    processor=text_extract_processor,
                    img=img)

                if full_text is not None or "no text can be extracted" in full_text:
                    # print(f"\nExtracted text:\n{full_text}")

                    output = format_guild_lines(full_text)

                    # print("\nformat text:" + str(output))

                    # get the lastline from the extracted text
                    last_line = output[-1]

                    return last_line
                else:
                    print("\nNo extracted text")
                    return None
        else:
            print(
                f"Images are too similar. Captured image similarity {similarity_score}, expected <{similarity_score_threshold}")
            time.sleep(2)
    finally:
        minimize_window(exe_name, hwnd)

    return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    exe_name = args.exe_name
    ocrwsusr = args.ocrwsusr
    ocrwslicense = args.ocrwslicense

    x1 = args.x1
    y1 = args.y1

    x2 = args.x2
    y2 = args.y2

    bot_token: str = args.bot_token
    channelid: int = args.channelid
    use_easyocr: bool = args.easyocr

    use_ocrws: bool = ocrwsusr and ocrwslicense

    start_discord_bot = bot_token and channelid

    crop_box = (x1, y1, x2, y2)

    # upscale_processor, upscale_model = load_upscaler_model()
    # async def get_ll():
    #     last_line = await get_last_line(None,
    #                               crop_box=crop_box,
    #                               text_extract_model=None,
    #                               text_extract_processor=None,
    #                               upscale_processor=upscale_processor,
    #                               upscale_model=upscale_model,
    #                                     use_easyocr=True)
    #     print(last_line)
    # asyncio.run(get_ll())

    hwnd = None
    first_run = True

    global previous_image
    global channel

    hwnd = get_first_hwnd_by_exe(exe_name)
    print("Hooking Process...")
    print(f"Found HWND: {hwnd}")

    # tuple of four integers tuple[left, top, right, bottom] or tuple[x1,y1,x2,y2]
    crop_box = (x1, y1, x2, y2)

    # small_chat crop_box = (303, 595, 620, 745) # small chat
    # big_chat crop_box = (303, 595, 827, 861) # big chat

    seconds_to_wait: int = 10

    text_extract_method = "ocr"
    if use_ocrws:
        text_extract_method = "ocrwebservice"
    if use_easyocr:
        text_extract_method = "easyocr"

    print(f"text extraction method: {text_extract_method}")

    if text_extract_method == "ocr":
        text_extract_model, text_extract_processor = load_ocr_model()
    elif text_extract_method == "easyocr":
        text_extract_model, text_extract_processor = None, None
    elif text_extract_method == "ocrwebservice":
        text_extract_model, text_extract_processor = None, None

    upscale_processor, upscale_model = load_upscaler_model()

    try:
        if start_discord_bot:
            print("Using Discord Bot...")

            import discord
            from discord.ext import commands, tasks

            intents = discord.Intents.default()  # Creates a default set of intents. Intents tell Discord what events your bot wants to receive (e.g. messages, guild info, reactions)
            intents.guilds = True # Explicitly enables the guilds intent, so your bot can receive events related to servers (guilds) it’s in — such as being notified when it connects to a server
            intents.messages = True  # optional, if you want to handle messages

            bot = commands.Bot(command_prefix="!", intents=intents)

            @bot.event
            async def on_ready():
                print(f"Logged in as {bot.user}")
                global channel
                channel = bot.get_channel(channelid)
                if channel is None:
                    print("Channel not found. Check channel permissions")
                else:
                    loop.start()

            @tasks.loop(seconds=seconds_to_wait)
            async def loop():
                if channel is not None:
                    last_line = await get_last_line(hwnd,
                                                    crop_box=crop_box,
                                                    text_extract_model=text_extract_model,
                                                    text_extract_processor=text_extract_processor,
                                                    upscale_processor=upscale_processor,
                                                    upscale_model=upscale_model,
                                                    use_easyocr=use_easyocr,
                                                    ocrwsusr=ocrwsusr,
                                                    ocrwslicense=ocrwslicense)
                    if last_line:
                        if "Guild" in last_line:
                            last_line = last_line.replace("Guild", "")
                        print("\nlast line:" + str(last_line))
                        await channel.send(last_line)

            bot.run(bot_token)
        else:
            print("Not using Discord Bot... (insufficient data provided)")

            # just press Ctrl+Z to end the loop
            while True:
                async def get_ll():
                    last_line = await get_last_line(hwnd,
                                                    crop_box=crop_box,
                                                    text_extract_model=None,
                                                    text_extract_processor=None,
                                                    upscale_processor=upscale_processor,
                                                    upscale_model=upscale_model,
                                                    use_easyocr=use_easyocr,
                                                    ocrwsusr=ocrwsusr,
                                                    ocrwslicense=ocrwslicense)
                    print(last_line)
                asyncio.run(get_ll())

                if last_line:
                    if "Guild" in last_line:
                        last_line = last_line.replace("Guild", "")
                    print("\nlast line:" + str(last_line))

                    # encoded_result = compress_and_encode(last_line)
                    # print(f"\nencoded: {encoded_result}")
                    # write_to_file(encoded_result)
                    write_to_file(last_line)

                    print(f"Waiting {seconds_to_wait}s ...")
                    time.sleep(seconds_to_wait)
    except Exception as e:
        print(e)