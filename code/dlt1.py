import cv2
import numpy as np
from scipy.signal import find_peaks
import itertools
import matplotlib.pyplot as plt

class DetekciaPlesniDLT:
    def __init__(self, a_min=20, a_max=500000, hole_threshold=0.1, line_threshold=1.3, kernel_size=5):
        '''
        Inicializácia triedy
        Morfologické parametre na filtrovanie:
            a_min = Dolná hranica plochy defektu
            a_max = Horná hranica plochy defektu
            hole_threshold = Pomer medzi plochou kontúry a plochou kontúry po vyplnení dier
            line_threshold = Pomer plochy kontúry k jej obvodu
            
        kernel_size = Rozmer matice, pomocou ktorej sa uzatvárajú blízke body do jedného celku a na zahladenie zubatých okrajov
        '''
        
        self.a_min = a_min
        self.a_max = a_max
        self.hole_threshold = hole_threshold
        self.line_threshold = line_threshold
        self.kernel_size = kernel_size

    def cierne_pixely_percento(self, binary_img):
        pocet_ciernych_pixelov = binary_img.shape[0] * binary_img.shape[1]
        cierne_pixely = pocet_ciernych_pixelov - cv2.countNonZero(binary_img)
        return cierne_pixely / pocet_ciernych_pixelov
    
    def vymaz_text_pred_dlt(self, image, 
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

        pocet_labels, labels, statistiky, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        mapa_textu = np.zeros(gray.shape, dtype=np.uint8)

        for i in range(1, pocet_labels):
            x, y, w, h, area = statistiky[i, cv2.CC_STAT_LEFT], \
                                statistiky[i, cv2.CC_STAT_TOP], \
                                statistiky[i, cv2.CC_STAT_WIDTH], \
                                statistiky[i, cv2.CC_STAT_HEIGHT], \
                                statistiky[i, cv2.CC_STAT_AREA]

            if not (min_area < area < max_area):
                continue

            aspect = float(w) / h if h > 0 else 0
            if not (min_aspect < aspect < max_aspect):
                continue

            component_mask = (labels == i).astype(np.uint8) * 255
            kontury, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not kontury:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(kontury[0]))
            solidity = area / hull_area if hull_area > 0 else 0

            if solidity < min_solidity:
                continue

            mapa_textu[labels == i] = 255

        kernel = np.ones((3, 3), np.uint8)
        mapa_textu = cv2.dilate(mapa_textu, kernel, iterations=2)

        bez_textu = cv2.inpaint(image, mapa_textu, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        return bez_textu, mapa_textu
    
    def preprocess_dlt(self, image, layer_count=22):
        """
        KROK 1.
        """
        
        cisty_obr, _ = self.vymaz_text_pred_dlt(image)
    
        gray = cv2.cvtColor(cisty_obr, cv2.COLOR_BGR2GRAY)
        
        binary_images = []
        y = []

        for j in range(1, layer_count+1):
            t_j = (j / layer_count) * 255
            _, bin_img = cv2.threshold(gray, t_j, 255, cv2.THRESH_BINARY)
            binary_images.append(bin_img)
            y.append(self.cierne_pixely_percento(bin_img))

        dy_dx = np.gradient(y)
        peaks, _ = find_peaks(dy_dx)
        
        if len(peaks) < 2:
            peaks = np.argsort(dy_dx)[-2:]
            
        peaks = sorted(peaks)
        j_peak1 = peaks[-2]
        j_peak2 = peaks[-1]
        
        if j_peak1 > j_peak2:
            j_peak1, j_peak2 = j_peak2, j_peak1

        r = j_peak2 - j_peak1
        if r <= 4:
            j_peak1_star = max(0, j_peak1 - 2) 
            j_peak2_star = min(layer_count-1, j_peak2 + 2)  
        else:
            j_peak1_star = j_peak1
            j_peak2_star = j_peak2

        subtracted_images = []
        y2 = []
        
        indices = list(range(j_peak1_star, j_peak2_star + 1))
        combinations = list(itertools.combinations(indices, 2))
        
        for idx1, idx2 in combinations:
            sub_img = cv2.absdiff(binary_images[idx2], binary_images[idx1])
            subtracted_images.append(sub_img)
            y2.append(self.cierne_pixely_percento(sub_img))

        q_peaks, _ = find_peaks(y2)
        v_valleys, _ = find_peaks([-val for val in y2]) 
        
        if len(q_peaks) > 0 and len(v_valleys) > 0:
            q_ave = np.mean([y2[p] for p in q_peaks])     
            v_ave = np.mean([y2[v] for v in v_valleys])   
            
            valid_valleys = [v for v in v_valleys if y2[v] > v_ave]
            m_LL = valid_valleys[0] if valid_valleys else 0
            
            valid_peaks = [p for p in q_peaks if y2[p] < q_ave]
            m_UL = valid_peaks[-1] if valid_peaks else len(y2) - 1
            
            m_s = int(round((m_UL + m_LL) / 2.0))           
            m_s = min(max(m_s, 0), len(subtracted_images) - 1)
        else:
            m_s = len(subtracted_images) // 2
            
        optimum_image = subtracted_images[m_s]          

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        processed_mask = cv2.morphologyEx(optimum_image, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)

        return processed_mask, optimum_image, y

    def is_valid_size(self, a_k):
        return self.a_min < a_k < self.a_max

    def is_valid_hole(self, a_k, a_filled):
        if a_k == 0: 
            return False
        
        ratio = (a_filled - a_k) / a_k
        return ratio <= self.hole_threshold

    def is_valid_line(self, a_k, perimeter):
        if perimeter == 0: 
            return False
        
        return (a_k / perimeter) > self.line_threshold

    def detect_and_filter(self, image, saturation_threshold=60):
        """
        KROK 2.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_channel = hsv_image[:, :, 1]

        dlt_mask, optimum_img, y_curve = self.preprocess_dlt(image)
        kontury, _ = cv2.findContours(dlt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_defects = []
        result_binary_mf = np.zeros_like(dlt_mask)
        display_img = image.copy()

        for contour in kontury:
            a_k = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            filled_mask = np.zeros_like(dlt_mask)
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
            a_filled = cv2.countNonZero(filled_mask)

            size_ok = self.is_valid_size(a_k)
            hole_ok = self.is_valid_hole(a_k, a_filled)
            line_ok = self.is_valid_line(a_k, perimeter)
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(a_k) / hull_area if hull_area > 0 else 0
            is_clutter = aspect_ratio > 6.0 or solidity < 0.3

            mean_val = cv2.mean(s_channel, mask=filled_mask)[0]
            color_ok = mean_val > saturation_threshold

            if size_ok and hole_ok and line_ok and not is_clutter and color_ok:
                valid_defects.append(contour)
                cv2.drawContours(result_binary_mf, [contour], -1, 255, thickness=cv2.FILLED)
                color = (0, 255, 0)
            else:
                color = (0, 0, 255) 

            cv2.drawContours(display_img, [contour], -1, color, 2)

        return valid_defects, result_binary_mf, display_img, dlt_mask, optimum_img, y_curve

    def vizualizuj_vysledky(self, image, display_img, result_binary, dlt_mask):
        '''
        Vizualizácia výsledkov 
        1.Riadok: Originálny obrázok a DLT maska
        2.Riadok: Detekcia a finálna binárna maska defektov
        '''
        
        plt.figure(figsize=(16, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Originálny obrázok")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(dlt_mask, cmap='gray')
        plt.title("DLT Maska")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title("Detekcia")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(result_binary, cmap='gray')
        plt.title("Finálna maska")
        plt.axis('off')

        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    detector = DetekciaPlesniDLT()
    # images/I1.png
    # images/I2.png
    image = cv2.imread('images/I1.png')
    validne_kontury, maska_vysledku, obrazok_s_konturami, dlt_maska, opt_img, y_krivka = detector.detect_and_filter(image)
    detector.vizualizuj_vysledky(image, obrazok_s_konturami, maska_vysledku, dlt_maska)