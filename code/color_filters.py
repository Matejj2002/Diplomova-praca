import cv2
import numpy as np
import matplotlib.pyplot as plt


class MoldClassificationExtractor:
    def __init__(self, patch_size=224, a_min=2000, a_max=500000, 
                 hole_threshold=0.1, line_threshold=1.3, kernel_size=11):
        self.patch_size = patch_size
        self.patch_size = patch_size
        self.a_min = a_min
        self.a_max = a_max
        self.hole_threshold = hole_threshold
        self.line_threshold = line_threshold
        self.kernel_size = kernel_size
        

    def vymaz_text(self, image,
                               min_area=10, max_area=3000,
                               min_aspect=0.1, max_aspect=10.0,
                               min_solidity=0.4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=10
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        text_mask = np.zeros(gray.shape, dtype=np.uint8)

        for i in range(1, num_labels): 
            x    = stats[i, cv2.CC_STAT_LEFT]
            y    = stats[i, cv2.CC_STAT_TOP]
            w    = stats[i, cv2.CC_STAT_WIDTH]
            h    = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if not (min_area < area < max_area):
                continue

            aspect = float(w) / h if h > 0 else 0
            if not (min_aspect < aspect < max_aspect):
                continue

            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
            solidity  = area / hull_area if hull_area > 0 else 0
            if solidity < min_solidity:
                continue

            text_mask[labels == i] = 255

        kernel    = np.ones((3, 3), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        inpainted = cv2.inpaint(image, text_mask, inpaintRadius=5,
                                flags=cv2.INPAINT_TELEA)

        return inpainted, text_mask
    
    def _dark_mask(self, image: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def _color_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ranges = [
            (np.array([5,  60,  60]),  np.array([25, 255, 180])), 
            (np.array([0,  70,  40]),  np.array([15, 255,  90])),
            (np.array([35, 30,  60]),  np.array([85, 255, 200])),
            (np.array([20, 50,  80]),  np.array([35, 255, 220])), 
        ]
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
        combined = cv2.GaussianBlur(combined, (7, 7), 0)
        _, combined = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)
        return combined

    @staticmethod
    def _circularity(cnt) -> float:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        return (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0.0
    
    
    def is_valid_size(self, a_k):
        """Skontroluje, či plocha spadá do povoleného rozsahu."""
        return self.a_min < a_k < self.a_max

    def is_valid_hole(self, a_k, a_filled):
        """Skontroluje pomer 'dier' v objekte. 
        a_filled je plocha objektu po vyplnení (napr. cez Convex Hull)."""
        if a_k == 0: return False
        ratio = (a_filled - a_k) / a_k
        return ratio <= self.hole_threshold

    def is_valid_line(self, a_k, perimeter):
        """Skontroluje, či objekt nie je príliš tenký (čiara).
        Pomer plocha/obvod je pri čiarach veľmi nízky."""
        if perimeter == 0: 
            return False
        return (a_k / perimeter) > self.line_threshold
    

    def get_patches(self, image, remove_text=True):        
        if remove_text:
            clean_image, text_mask = self.vymaz_text(image)
        else:
            clean_image = image.copy()
            text_mask = None

        dark     = self._dark_mask(clean_image)
        color    = self._color_mask(clean_image)
        combined = cv2.bitwise_or(dark, color)

        k_big   = np.ones((25, 25), np.uint8)
        k_small = np.ones((7,  7),  np.uint8)
        mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_big)
        mask = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  k_small)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        patches     = []
        display_img = image.copy()

        for cnt in contours:
            area         = cv2.contourArea(cnt)
            x, y, w, h  = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            circularity  = self._circularity(cnt)
            perimeter = cv2.arcLength(cnt, True)

            reject_reason = None
            if area < 2000:
                reject_reason = f"area={int(area)}"
            elif aspect_ratio > 5.0 or aspect_ratio < 0.2:
                reject_reason = f"aspect={aspect_ratio:.1f}"
            elif circularity < 0.08:
                reject_reason = f"circ={circularity:.2f}"
            elif not self.is_valid_line(area, perimeter):
                # Tento filter zachytí tenké čiary, ktoré predtým robili problém
                reject_reason = f"line={area/perimeter:.1f}"

            if reject_reason:
                cv2.drawContours(display_img, [cnt], -1, (0, 0, 255), 2)
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            p2     = self.patch_size // 2
            padded = cv2.copyMakeBorder(image, p2, p2, p2, p2,
                                        cv2.BORDER_CONSTANT,
                                        value=[255, 255, 255])
            patch  = padded[cy : cy + self.patch_size,
                            cx : cx + self.patch_size]

            if patch.shape[0] == self.patch_size and \
               patch.shape[1] == self.patch_size:
                patches.append(patch)
                cv2.drawContours(display_img, [cnt], -1, (0, 255, 0), 4)
                cv2.putText(
                    display_img,
                    f"Plesn {len(patches)}  circ={circularity:.2f}",
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

        return patches, mask, display_img, clean_image, text_mask


if __name__ == "__main__":
    # images/I1.png
    # images/I5.jpg
    # images/IMG_0273.jpg - T
    # images/IMG_0278.jpg
    # images/IC1.jpg - IC4.jpg
    img = cv2.imread('images/I1.png')
    if img is None:
        raise FileNotFoundError("Obrázok nenájdený – uprav cestu.")

    ext = MoldClassificationExtractor(patch_size=224)
    patches, final_mask, result_view, clean_img, text_mask = ext.get_patches(img, remove_text=False)

    print(f"\nNájdené oblasti plesne: {len(patches)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 11))
    fig.suptitle(f"Detektor plesne  –  nájdené: {len(patches)}", fontsize=15)

    panels = [
        ("Vstup (originál)", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None),
        ("Finálna maska plesne", final_mask, 'gray'),
        ("Výsledok detekcie", cv2.cvtColor(result_view, cv2.COLOR_BGR2RGB), None),
    ]

    for ax, (title, data, cmap) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
#white balance - pozriet - https://stackoverflow.com/questions/46441690/white-balance-correction-in-opencv-python