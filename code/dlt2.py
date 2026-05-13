import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class DetekciaPlesniDLT:
    def __init__(self, a_min=20, a_max=500000, hole_threshold=0.1, line_threshold=1.3, kernel_size=5):
        '''
        Morfologické parametre na filtrovanie:
            a_min = Minimálna plocha v pixeloch
            a_max = Maximálna plocha v pixeloch
            hole_threshold = Pomer vnútornej celistvosti
            line_threshold = Pomer plochy k obvodu (eliminácia vlasov/čiar)
        '''
        self.a_min = a_min
        self.a_max = a_max
        self.hole_threshold = hole_threshold
        self.line_threshold = line_threshold
        self.kernel_size = kernel_size

    def objektove_pixely_percento(self, binary_img):
        """Počíta percento bielych pixelov (objektov) v binárnom obraze."""
        total_pixels = binary_img.shape[0] * binary_img.shape[1]
        white_pixels = cv2.countNonZero(binary_img)
        return white_pixels / total_pixels
    
    def vymaz_text_pred_dlt(self, image):
        """Pomocná funkcia na elimináciu textu pomocou inpaintingu."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=10
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mapa_textu = np.zeros(gray.shape, dtype=np.uint8)

        for i in range(1, num_labels):
            w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
            if 10 < area < 3000 and 0.1 < (float(w)/h) < 10.0:
                mapa_textu[labels == i] = 255

        kernel = np.ones((3, 3), np.uint8)
        mapa_textu = cv2.dilate(mapa_textu, kernel, iterations=1)
        bez_textu = cv2.inpaint(image, mapa_textu, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return bez_textu, mapa_textu
    
    def preprocess_dlt(self, image, layer_count=40):
        """
        KROK 1: Dynamické hľadanie optimálneho prahu (podľa článku).
        Hľadáme stabilitu v raste plochy objektov.
        """
        cisty_obr, _ = self.vymaz_text_pred_dlt(image)
        gray = cv2.cvtColor(cisty_obr, cv2.COLOR_BGR2GRAY)
        
        binary_layers = []
        y_area = []

        for j in range(1, layer_count + 1):
            t_j = int((j / layer_count) * 255)
            _, bin_img = cv2.threshold(gray, t_j, 255, cv2.THRESH_BINARY_INV)
            binary_layers.append(bin_img)
            y_area.append(self.objektove_pixely_percento(bin_img))

        dy_dx = np.gradient(y_area)
        peaks, _ = find_peaks(dy_dx, distance=5)
        
        if len(peaks) >= 2:
            m_s = (peaks[0] + peaks[1]) // 2
        elif len(peaks) == 1:
            m_s = max(0, peaks[0] - 5)
        else:
            m_s = layer_count // 4

        m_s = min(max(m_s, 0), len(binary_layers) - 1)
        optimum_image = binary_layers[m_s]

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        processed_mask = cv2.morphologyEx(optimum_image, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)

        return processed_mask, optimum_image, y_area

    def detect_and_filter(self, image, saturation_threshold=20):
        """
        KROK 2: Detekcia kontúr a aplikácia morfologických a farebných filtrov.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_channel = hsv_image[:, :, 1]

        dlt_mask, raw_optimum, y_curve = self.preprocess_dlt(image)
        kontury, _ = cv2.findContours(dlt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_defects = []
        finalna_maska = np.zeros_like(dlt_mask)
        display_img = image.copy()

        for contour in kontury:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0: continue

            filled_mask = np.zeros_like(dlt_mask)
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
            area_filled = cv2.countNonZero(filled_mask)

            size_ok = self.a_min < area < self.a_max
            hole_ok = ((area_filled - area) / (area if area > 0 else 1)) <= self.hole_threshold
            line_ok = (area / perimeter) > self.line_threshold
            
            mean_s = cv2.mean(s_channel, mask=filled_mask)[0]
            color_ok = mean_s > saturation_threshold

            if size_ok and hole_ok and line_ok and color_ok:
                valid_defects.append(contour)
                cv2.drawContours(finalna_maska, [contour], -1, 255, thickness=cv2.FILLED)
                cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)
            else:
                if area > self.a_min:
                    cv2.drawContours(display_img, [contour], -1, (0, 0, 255), 1)

        return valid_defects, finalna_maska, display_img, dlt_mask

    def vizualizuj(self, image, dlt_mask, display_img, final_mask):
        plt.figure(figsize=(16, 10))
        
        tituly = ["Originál", "DLT Maska (Krok 1)", "Detekcia (Zelená=Pleseň)", "Finálna maska"]
        obrazy = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dlt_mask, 
                  cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), final_mask]
        
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(obrazy[i], cmap='gray' if i in [1, 3] else None)
            plt.title(tituly[i])
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    detector = DetekciaPlesniDLT()
    
    img_path = 'images/IC1.jpg'
    image = cv2.imread(img_path)

    if image is not None:
        kontury, maska, detekcia_viz, dlt_raw = detector.detect_and_filter(image)
        detector.vizualizuj(image, dlt_raw, detekcia_viz, maska)
    else:
        print("Obrázok sa nepodarilo načítať.")