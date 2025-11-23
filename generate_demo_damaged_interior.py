from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import random


def make_variant(idx: int, W=1280, H=720, seed=42):
    random.seed(seed + idx)
    img = Image.new('RGB', (W, H), (200, 200, 200))
    d = ImageDraw.Draw(img)

    # background gradient (top darker)
    for y in range(H):
        t = y / H
        r = int(170 + (30 * t) + random.uniform(-6, 6))
        g = int(170 + (30 * t) + random.uniform(-6, 6))
        b = int(175 + (20 * t) + random.uniform(-6, 6))
        d.line([(0, y), (W, y)], fill=(r, g, b))

    # draw dashboard
    dash_h = int(H * 0.25)
    dash_y = int(H * 0.25)
    d.rectangle([(0, dash_y), (W, dash_y + dash_h)], fill=(35, 35, 40))

    # center console
    cx1, cy1 = int(W * 0.4), int(dash_y + dash_h * 0.2)
    cx2, cy2 = int(W * 0.6), int(dash_y + dash_h * 0.8)
    d.rectangle([(cx1, cy1), (cx2, cy2)], fill=(60, 60, 65))

    # seats (two)
    seat_w, seat_h = int(W * 0.22), int(H * 0.35)
    left_seat_box = [(int(W * 0.15), int(H * 0.35)), (int(W * 0.15) + seat_w, int(H * 0.35) + seat_h)]
    right_seat_box = [(int(W * 0.63), int(H * 0.35)), (int(W * 0.63) + seat_w, int(H * 0.35) + seat_h)]
    d.ellipse([left_seat_box[0][0], left_seat_box[0][1] - 20, left_seat_box[1][0], left_seat_box[1][1] + 20], fill=(90, 90, 100))
    d.ellipse([right_seat_box[0][0], right_seat_box[0][1] - 20, right_seat_box[1][0], right_seat_box[1][1] + 20], fill=(90, 90, 100))

    # steering wheel
    sw_x, sw_y = int(W * 0.25), int(dash_y + dash_h * 0.5)
    d.ellipse([(sw_x - 60, sw_y - 60), (sw_x + 60, sw_y + 60)], outline=(30, 30, 30), width=8)

    # center screen
    d.rectangle([(int(W * 0.45), int(H * 0.30)), (int(W * 0.55), int(H * 0.42))], fill=(10, 10, 12))

    # place damages with some randomness
    # damage on left seat
    for i in range(12 + idx % 5):
        x = int(W * 0.15 + 20 + random.random() * seat_w * 0.8)
        y = int(H * 0.45 + random.random() * seat_h * 0.5)
        ex = x + int(random.uniform(10, 80))
        ey = y + int(random.uniform(-6, 6))
        d.line([(x, y), (ex, ey)], fill=(180, 20, 20), width=3)

    # scuff on dashboard
    for i in range(8 + (idx % 3)):
        x1 = int(W * 0.5 + random.uniform(-150, 150))
        y1 = int(dash_y + random.uniform(10, dash_h - 10))
        x2 = x1 + int(random.uniform(-40, 40))
        y2 = y1 + int(random.uniform(-6, 6))
        d.line([(x1, y1), (x2, y2)], fill=(120, 10, 10), width=4)

    # puncture-like marks on center console
    for i in range(4 + (idx % 4)):
        cx = int(W * 0.5 + random.uniform(-20, 20))
        cy = int(H * 0.35 + random.uniform(10, 140))
        d.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=(200, 40, 40))

    # dark smudge
    for i in range(200 + idx * 20):
        rx = int(W * 0.3 + random.random() * W * 0.4)
        ry = int(H * 0.4 + random.random() * H * 0.2)
        d.point((rx, ry), fill=(30, 30, 30))

    # add film grain
    grain = Image.new('RGB', (W, H))
    gd = ImageDraw.Draw(grain)
    for _ in range(int(W * H * 0.002)):
        x = random.randrange(0, W)
        y = random.randrange(0, H)
        shade = random.randint(0, 30)
        gd.point((x, y), fill=(shade, shade, shade))
    img = Image.blend(img, grain, alpha=0.08)

    # slight blur to simulate camera
    if idx % 2 == 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # contrast and color jitter
    enh = ImageEnhance.Contrast(img)
    img = enh.enhance(1.02 + (idx % 3) * 0.02)
    enhc = ImageEnhance.Color(img)
    img = enhc.enhance(0.95 + (idx % 4) * 0.03)

    # vignetting
    vign = Image.new('L', (W, H), 0)
    vd = ImageDraw.Draw(vign)
    for r in range(max(W, H)//2):
        alpha = int(180 * (r / (max(W, H)//2))**2)
        vd.ellipse([W//2 - r, H//2 - r, W//2 + r, H//2 + r], outline=alpha)
    vign = vign.filter(ImageFilter.GaussianBlur(radius=30))
    img = Image.composite(Image.new('RGB', (W, H), (20, 20, 20)), img, ImageOps.invert(vign))

    # slight perspective shear to feel more photographic
    shear = (random.uniform(-0.03, 0.03), random.uniform(-0.02, 0.02))
    coeffs = (1, shear[0], -W * shear[0] * 0.3, shear[1], 1, -H * shear[1] * 0.2)
    img = img.transform((W, H), Image.AFFINE, coeffs, resample=Image.BICUBIC)

    return img


def main():
    # create multiple variants
    out_paths = []
    for i in range(4):
        img = make_variant(i)
        path = f'demo_damaged_interior_variant{i+1}.png'
        img.save(path)
        out_paths.append(path)
    print('Saved', ', '.join(out_paths))


if __name__ == '__main__':
    main()
