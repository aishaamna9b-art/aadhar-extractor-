import cv2
import numpy as np
import pytesseract
import easyocr
import re
import io
from PIL import Image
import fitz  # PyMuPDF
import base64
import logging

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader (this takes a moment on first load)
# Using Hindi and English
reader = easyocr.Reader(['hi', 'en'], gpu=False)

STOPLIST = [
    "government of india", "भारत सरकार", "aadhar", "आधार",
    "unique identification authority of india",
    "भारतीय विशिष्ट पहचान प्राधिकरण",
    "enrollment no", "नामांकन क्रम संख्या"
]

def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def load_image_from_bytes(file_bytes, content_type):
    if content_type == 'application/pdf':
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("Empty PDF")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deskew first
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    deskewed = deskew_image(thresh)
    
    # Apply deskew rotation to original gray image
    if gray.shape != deskewed.shape:
         gray = deskew_image(gray) # Approximation for simplicity, better to use the same angle
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return denoised

def split_front_back(gray_img):
    h, w = gray_img.shape
    
    # Detect vertical dashed line
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    
    # Sum edge intensity per column
    col_sums = np.sum(edges, axis=0)
    
    # Look for peak in middle third
    third = w // 3
    mid_sums = col_sums[third: 2*third]
    
    if np.max(mid_sums) > 255 * h * 0.1: # Threshold for a valid line
        split_x = third + np.argmax(mid_sums)
    else:
        split_x = w // 2
        
    front = gray_img[:, :split_x]
    back = gray_img[:, split_x:]
    return front, back

def crop_to_content(img):
    # Find bounding box of ALL content to remove white margins
    # instead of just the largest contour which could be a QR code.
    _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add a small padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2*pad)
    h = min(img.shape[0] - y, h + 2*pad)
    
    return img[y:y+h, x:x+w]

def encode_image_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

def run_ocr(img, engine='tesseract', is_front=False):
    if engine == 'tesseract':
        if is_front:
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='eng+hin')
        
        words = []
        confidences = []
        
        current_line = []
        current_line_conf = []
        
        lines = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:
                words.append(data['text'][i])
                confidences.append(int(data['conf'][i]))
                
        # Group into lines by line_num
        line_dict = {}
        for i in range(n_boxes):
            if int(data['conf'][i]) > -1: # including spaces/empty to keep structure, but filter text
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                line_num = data['line_num'][i]
                block_num = data['block_num'][i]
                par_num = data['par_num'][i]
                key = (block_num, par_num, line_num)
                
                if key not in line_dict:
                    line_dict[key] = {'text': [], 'conf': []}
                
                if text:
                    line_dict[key]['text'].append(text)
                    line_dict[key]['conf'].append(conf)
                    
        for key in sorted(line_dict.keys()):
            if line_dict[key]['text']:
                lines.append({
                    'text': ' '.join(line_dict[key]['text']),
                    'conf': sum(line_dict[key]['conf']) / len(line_dict[key]['conf']) if line_dict[key]['conf'] else 0
                })
                
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        return lines, avg_conf
        
    elif engine == 'easyocr':
        if is_front:
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
        result = reader.readtext(img)
        lines = []
        confidences = []
        for bbox, text, conf in result:
            lines.append({
                'text': text,
                'conf': conf * 100
            })
            confidences.append(conf * 100)
            
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        return lines, avg_conf

def get_best_ocr(img, is_front=False):
    lines, avg_conf = run_ocr(img, engine='tesseract', is_front=is_front)
    
    if avg_conf < 60:
        logger.info(f"Tesseract conf {avg_conf} < 60. Re-running with EasyOCR.")
        lines_easy, avg_conf_easy = run_ocr(img, engine='easyocr', is_front=is_front)
        if avg_conf_easy > avg_conf:
            logger.info(f"EasyOCR provided better confidence: {avg_conf_easy}")
            return lines_easy
    return lines

def clean_text_list(lines):
    return [line['text'] for line in lines]

def extract_name(lines):
    dob_patterns = ["dob", "date of birth", "जन्म"]
    
    dob_idx = -1
    for i, line in enumerate(lines):
        text_lower = line.lower()
        if any(p in text_lower for p in dob_patterns):
            dob_idx = i
            break
            
    if dob_idx == -1:
        return None
        
    for i in range(dob_idx - 1, -1, -1):
        text = lines[i].strip()
        if not text:
            continue
            
        text_lower = text.lower()
        is_stop = any(stop in text_lower for stop in STOPLIST)
        if is_stop:
            continue
            
        # Check if contains only letters, spaces, and dots
        # Allow unicode letters (Hindi) by checking if all chars are alpha, space, or dot
        if all(c.isalpha() or c.isspace() or c == '.' for c in text):
            return text
            
    return None

def extract_address(lines):
    address_patterns = ["address:", "पता:", "address", "ddress", "s/o", "d/o", "c/o", "w/o"]
    
    start_idx = -1
    start_offset = -1
    for i, line in enumerate(lines):
        text_lower = line.lower()
        for p in address_patterns:
            idx = text_lower.find(p)
            if idx != -1:
                start_idx = i
                # If we matched a relationship marker like s/o, the whole line is part of the address
                if p in ["s/o", "d/o", "c/o", "w/o", "s/o:", "d/o:", "c/o:", "w/o:"]:
                    start_offset = idx
                else:
                    start_offset = idx + len(p)
                break
        if start_idx != -1:
            break
            
    if start_idx == -1:
        return None
        
    address_parts = []
    
    # First line might have text after "Address:"
    first_line_remainder = lines[start_idx][start_offset:].strip()
    if first_line_remainder:
        address_parts.append(first_line_remainder.rstrip(','))
        
    for i in range(start_idx + 1, len(lines)):
        text = lines[i].strip()
        if not text:
            continue
            
        # Stop condition
        aadhaar_pattern = r'\d{4}\s?\d{4}\s?\d{4}'
        if re.search(aadhaar_pattern, text) or "VID" in text.upper():
            break
            
        # Fallback to remove any lingering vertical text that got merged
        text = re.sub(r'(?i)details as on.*', '', text).strip()
        if not text:
            continue
            
        address_parts.append(text.rstrip(','))
        
    if not address_parts:
        return None
        
    joined = ', '.join(address_parts)
    # Strip trailing punctuation and whitespace
    joined = re.sub(r'[,.\s]+$', '', joined)
    return joined

def crop_bottom_card_part(gray_img):
    h, w = gray_img.shape
    
    # Detect horizontal lines
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    
    # Sum edge intensity per row
    row_sums = np.sum(edges, axis=1)
    
    # The cut line is usually in the middle to lower half of the page
    # Let's search from h//3 to 4*h//5
    start_y = h // 3
    end_y = 4 * h // 5
    
    if start_y < end_y:
        search_area = row_sums[start_y:end_y]
        if len(search_area) > 0:
            max_val = np.max(search_area)
            if max_val > 255 * w * 0.1: # Threshold for a valid horizontal line
                split_y = start_y + np.argmax(search_area)
                # Crop slightly below the line to remove the line itself
                return gray_img[split_y+10:, :]
                
    return gray_img

def process_file(file_bytes, content_type):
    img = load_image_from_bytes(file_bytes, content_type)
    
    gray = preprocess_image(img)
    
    # 1. Split vertically first (Front and Back halves)
    front_full, back_full = split_front_back(gray)
    
    # Check if original image is likely a full A4 page (portrait mode)
    # A full A4 page will have height > width. 
    # If the user uploads just the card (front + back side by side), it will typically be landscape (width > height).
    h, w = gray.shape
    if h > w * 1.1:
        # 2. Horizontal cut on both halves individually
        front_bottom = crop_bottom_card_part(front_full)
        back_bottom = crop_bottom_card_part(back_full)
    else:
        front_bottom = front_full
        back_bottom = back_full
    
    # 3. Crop to content (remove extra padding/borders)
    front_cropped = crop_to_content(front_bottom)
    back_cropped = crop_to_content(back_bottom)
    
    # (Removed 4% left crop as it was cutting into valid address text on normal photos)
    
    front_lines_data = get_best_ocr(front_cropped, is_front=True)
    back_lines_data = get_best_ocr(back_cropped, is_front=False)
    
    front_text = clean_text_list(front_lines_data)
    back_text = clean_text_list(back_lines_data)
    
    name = extract_name(front_text)
    address = extract_address(back_text)
    
    result = {
        "name": name,
        "address": address,
        "confidence": {
            "name": "high" if name else "low",
            "address": "high" if address else "low"
        },
        "previews": {
            "front": encode_image_base64(front_cropped),
            "back": encode_image_base64(back_cropped)
        }
    }
    
    return result

