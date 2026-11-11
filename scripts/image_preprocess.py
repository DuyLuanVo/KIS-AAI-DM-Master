"""
Tiền xử lý & tăng cường ảnh dùng chung cho dedup_images.py và finetune_clip.py.

Hai phần tách bạch:
1. Tiền xử lý XÁC ĐỊNH (mọi tập train/val/test và backend đều như nhau):
   open_rgb: xoay đúng chiều EXIF, ghép nền trắng cho ảnh trong suốt, chuyển RGB;
   sau đó là preprocess gốc của CLIP (resize cạnh ngắn 224 bicubic, cắt giữa 224, chuẩn hóa theo mean/std CỐ ĐỊNH
   của CLIP, không ước lượng từ dữ liệu -> không rò rỉ thống kê của val/test vào train).
2. Tăng cường NGẪU NHIÊN (chỉ áp cho ảnh thuộc tập train):
   cắt ngẫu nhiên, lật ngang, xoay/phối cảnh nhẹ, sắc độ (độ sáng, tương phản, bão hòa, hue), xám hóa, làm mờ/nét,
   xóa vùng nhỏ. Chính sách "hiểu caption": phép biến đổi nào làm caption sai thì tắt cho ảnh đó
     - caption nói trái/phải, có chữ/biển hiệu      -> không lật ngang (lật làm ngược chữ, đổi trái-phải)
     - caption nhắc màu sắc ("red roof", "yellow walls") -> không đổi hue, không xám hóa, giảm bão hòa
     - caption nhắc trời/ánh sáng ("night", "sunset")   -> giảm biên độ độ sáng
"""
import re

from PIL import Image, ImageOps

POLICY_VERSION = "v2"  # v2: từ khóa khớp cả dạng số nhiều ("inscriptions", "signboards"...)

_FLIP_BLOCK = re.compile(
    r"\b(?:left|right|leftmost|rightmost|sign|signboard|text|reading|reads|written|letter|lettering|inscription|"
    r"inscribed|banner|logo|number|word|calligraphy|character|plaque|placard|poster|printed|label|map|diagram|"
    r"chart|document|page)(?:s|es)?\b", re.I)
_COLOR_WORDS = re.compile(
    r"\b(red|reddish|orange|yellow|yellowish|green|greenish|blue|bluish|purple|violet|pink|brown|brownish|beige|tan|"
    r"cream|white|whitewashed|black|gray|grey|golden|gold|silver|turquoise|teal|maroon|ochre|colorful|colourful|"
    r"multicolored|pastel|terracotta)\b", re.I)
_LIGHT_WORDS = re.compile(r"\b(?:night|dusk|dawn|sunset|sunrise|dark|illuminated|lit|light|lantern|"
                          r"bright|sunlight|sunny|shadow|overcast|twilight)(?:s|es)?\b", re.I)


def open_rgb(path) -> Image.Image:
    """Mở ảnh theo đúng chiều EXIF, ghép nền trắng cho ảnh trong suốt, trả về RGB."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def caption_flags(caption: str) -> dict:
    return {"no_flip": bool(_FLIP_BLOCK.search(caption)),
            "has_color": bool(_COLOR_WORDS.search(caption)),
            "has_light": bool(_LIGHT_WORDS.search(caption))}


def clip_normalize(preprocess):
    from torchvision import transforms as T

    return [t for t in preprocess.transforms if isinstance(t, T.Normalize)][0]


POLICIES = ("default", "blip_geo")
_FILL = (124, 116, 104)  # màu lấp vùng trống khi biến đổi hình học (gần trung bình ảnh)


class RandGeoAugment:
    """Thí nghiệm B: RandAugment(N=2, M=5) kiểu BLIP nhưng chỉ gồm phép HÌNH HỌC (không đổi màu, vì caption hay nhắc
    màu). Mỗi ảnh chọn ngẫu nhiên N phép trong {ShearX, ShearY, TranslateX, TranslateY, Rotate}, mỗi phép áp với xác
    suất 0,5 như RandomAugment của BLIP. Cường độ M/10 = 0,5: shear hệ số 0,15 (~8,5°), dịch 10% cạnh ảnh, xoay 15°."""

    def __init__(self, n: int = 2, m: int = 5, p: float = 0.5):
        self.n, self.p, f = n, p, m / 10
        self.shear, self.translate, self.rotate = 0.3 * f, 0.2 * f, 30 * f

    def __call__(self, img):
        import math
        import random

        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        w, h = img.size
        for op in random.sample(["shear_x", "shear_y", "translate_x", "translate_y", "rotate"], self.n):
            if random.random() > self.p:
                continue
            sign = random.choice((-1, 1))
            angle, tx, ty, sx, sy = 0.0, 0, 0, 0.0, 0.0
            if op == "shear_x":
                sx = sign * math.degrees(math.atan(self.shear))
            elif op == "shear_y":
                sy = sign * math.degrees(math.atan(self.shear))
            elif op == "translate_x":
                tx = int(sign * self.translate * w)
            elif op == "translate_y":
                ty = int(sign * self.translate * h)
            else:
                angle = sign * self.rotate
            img = TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=[sx, sy],
                            interpolation=InterpolationMode.BILINEAR, fill=list(_FILL))
        return img


def train_transform(preprocess, flags: dict, size: int = 224, policy: str = "default"):
    """Chuỗi tăng cường cho MỘT ảnh train, điều chỉnh theo cờ của caption.
    policy = "default": cấu hình chính; "blip_geo": thêm RandGeoAugment sau bước lật (thí nghiệm B)."""
    from torchvision import transforms as T

    assert policy in POLICIES, policy
    bicubic = T.InterpolationMode.BICUBIC
    color = flags.get("has_color", False)
    light = flags.get("has_light", False)
    ops = [
        # hình học: kiến trúc nhiều đường thẳng đứng nên xoay/phối cảnh chỉ ở mức nhẹ
        T.RandomApply([T.RandomChoice([
            T.RandomRotation(8, interpolation=T.InterpolationMode.BILINEAR, fill=_FILL),
            T.RandomPerspective(distortion_scale=0.15, p=1.0, interpolation=T.InterpolationMode.BILINEAR,
                                fill=_FILL),
        ])], p=0.3),
        # cắt: giữ >= 50% diện tích để công trình chính vẫn còn trong khung
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3), interpolation=bicubic),
    ]
    if not flags.get("no_flip", False):
        ops.append(T.RandomHorizontalFlip(0.5))
    if policy == "blip_geo":
        ops.append(RandGeoAugment(n=2, m=5))
    # sắc độ
    ops.append(T.RandomApply([T.ColorJitter(brightness=0.15 if light else 0.3, contrast=0.3,
                                            saturation=0.1 if color else 0.25, hue=0.0 if color else 0.03)], p=0.8))
    if not color:
        ops.append(T.RandomGrayscale(p=0.05))
    ops += [
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
        T.RandomAdjustSharpness(2.0, p=0.15),
        T.RandomAutocontrast(p=0.15),
        T.ToTensor(),
        clip_normalize(preprocess),
        T.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0),
    ]
    return T.Compose(ops)


def describe_policy(policy: str = "default") -> str:
    base = ("RandomRotation(±8°)|RandomPerspective(0.15) p=0.3; RandomResizedCrop(224, scale 0.5-1, ratio 3/4-4/3); "
            "HorizontalFlip p=0.5 (tắt khi caption có trái/phải/chữ); ")
    extra = ("RandGeoAugment(N=2, M=5: ShearX/Y 0.15, TranslateX/Y 10%, Rotate 15°, mỗi phép p=0.5); "
             if policy == "blip_geo" else "")
    return (base + extra + "ColorJitter(b 0.3, c 0.3, s 0.25, h 0.03) p=0.8 "
            "(caption có màu: h 0, s 0.1; có ánh sáng: b 0.15); Grayscale p=0.05 (tắt khi có màu); "
            "GaussianBlur p=0.15; Sharpness p=0.15; Autocontrast p=0.15; RandomErasing(2-8%) p=0.1")
