import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from DLT import DLTMoldDetector

class DetectionFilter:
    def __init__(self, a_min=20, a_max=500000, hole_threshold=0.1, line_threshold=1.3, kernel_size=11):
        self.a_min = a_min
        self.a_max = a_max
        self.hole_threshold = hole_threshold
        self.line_threshold = line_threshold
        self.kernel_size=kernel_size
        
        self.dlt_detector = DLTMoldDetector()

    def is_valid_size(self, a_k):
        return self.a_min < a_k < self.a_max

    def is_valid_hole(self, a_k, a_filled):
        if a_k == 0: return False
        ratio = (a_filled - a_k) / a_k
        return ratio <= self.hole_threshold

    def is_valid_line(self, a_k, perimeter):
        if perimeter == 0: 
            return False
        
        return (a_k / perimeter) > self.line_threshold

    def detect_mold(self, image, show_defects=False, save_defects=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        text_free_gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((self.kernel_size, self.kernel_size), np.uint8))

        _, binary_otsu = cv2.threshold(text_free_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_valid_defect = False
        result_binary_mf = np.zeros_like(gray)
        display_img = image.copy()

        for contour in contours:
            a_k = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            filled_mask = np.zeros_like(gray)
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

            if size_ok and hole_ok and line_ok and not is_clutter:
                found_valid_defect = True
                cv2.drawContours(result_binary_mf, [contour], -1, 255, thickness=cv2.FILLED)
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            
            cv2.drawContours(display_img, [contour], -1, color, 2)

        
        text_free_viz = cv2.cvtColor(text_free_gray, cv2.COLOR_GRAY2BGR)
        
        final_mask_viz = cv2.cvtColor(result_binary_mf, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, '1.', (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(text_free_viz, '2.', (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, '3.', (10, 30), font, 0.7, (0, 255, 0), 2)

        combined_view = np.hstack((image, text_free_viz, final_mask_viz))

        screen_res = 1500, 800
        scale = min(screen_res[0] / combined_view.shape[1], screen_res[1] / combined_view.shape[0])
        
        if scale < 1.0:
            combined_view = cv2.resize(combined_view, (0,0), fx=scale, fy=scale)

        if show_defects:
            cv2.imshow("Detection Workflow", combined_view)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_defects is not None:
            cv2.imwrite(save_defects, combined_view)

        return found_valid_defect, result_binary_mf

    def detect_mold_visual(self, img, method):
        def on_trackbar(val, name):
            if name == "kernel_size":
                self.kernel_size = max(1, val)
            if name == "a_min":
                self.a_min = max(1, val)
            if name == "a_max":
                self.a_max = max(1, val)
            if name == "hole_threshold":
                if val > 1:
                    val /= 100 
                self.hole_threshold = val
            if name == "line_threshold":
                if val >1:
                    val /=10
                self.line_threshold = val
            
            if method == "filter.detect_mold":  
                _, mask = self.detect_mold(img)

                mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((img, mask_color))
                screen_res = 1500, 800
                scale = min(screen_res[0] / combined.shape[1], screen_res[1] / combined.shape[0])
                
                if scale < 1.0:
                    combined = cv2.resize(combined, (0,0), fx=scale, fy=scale)
                cv2.imshow("Mold Detection", combined)
                
            elif method == "dlt.detect_mold":
                self.dlt_detector.kernel_size = self.kernel_size
                self.dlt_detector.a_min = self.a_min
                self.dlt_detector.a_max = self.a_max
                self.dlt_detector.hole_threshold = self.hole_threshold
                self.dlt_detector.line_threshold = self.line_threshold
                
                kontury,maska,y = self.detect_mold_dlt(img)
                output_img = img.copy()
                combined = np.hstack((img, cv2.drawContours(output_img, kontury, -1, (0, 255, 0), 2)))
                
                screen_res = 1500, 800
                scale = min(screen_res[0] / combined.shape[1], screen_res[1] / combined.shape[0])
                
                if scale < 1.0:
                    combined = cv2.resize(combined, (0,0), fx=scale, fy=scale)
                
                cv2.imshow("Mold Detection", combined)
        
        cv2.namedWindow("Mold Detection")
        cv2.createTrackbar("KS", "Mold Detection", self.kernel_size, 30, lambda x: on_trackbar(x,name="kernel_size"))
        cv2.createTrackbar("AMi", "Mold Detection", self.a_min, 30000, lambda x: on_trackbar(x,name="a_min"))
        cv2.createTrackbar("AMx", "Mold Detection", self.a_max, 500000, lambda x: on_trackbar(x,name="a_max"))
        cv2.createTrackbar("HT", "Mold Detection", int(self.hole_threshold * 100), 100, lambda x: on_trackbar(x,name="hole_threshold"))
        cv2.createTrackbar("LT", "Mold Detection", int(self.hole_threshold * 10), 50, lambda x: on_trackbar(x,name="line_threshold"))

        on_trackbar(self.kernel_size, name="kernel_size")
        on_trackbar(self.a_min, name="a_min")
        on_trackbar(self.a_max, name="a_max")
        on_trackbar(self.hole_threshold, name="hole_threshold")
        on_trackbar(self.line_threshold, name="line_threshold")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_mold_dlt(self, image):
        kontury, maska, y_data = self.dlt_detector.detect_dlt(image)
        return kontury, maska, y_data   

    def show_hsv_lab_channels(self, img):
        if img is None:
            raise ValueError("Obrázok sa nepodarilo načítať")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # prevody
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 4, 1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis("off")

        plt.subplot(2, 4, 2)
        plt.title("H channel")
        plt.imshow(h, cmap='gray')
        plt.axis("off")

        plt.subplot(2, 4, 3)
        plt.title("S channel")
        plt.imshow(s, cmap='gray')
        plt.axis("off")

        plt.subplot(2, 4, 4)
        plt.title("V channel")
        plt.imshow(v, cmap='gray')
        plt.axis("off")

        plt.subplot(2, 4, 6)
        plt.title("L channel")
        plt.imshow(l, cmap='gray')
        plt.axis("off")

        plt.subplot(2, 4, 7)
        plt.title("A channel")
        plt.imshow(a, cmap='gray')
        plt.axis("off")

        plt.subplot(2, 4, 8)
        plt.title("B channel")
        plt.imshow(b, cmap='gray')
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def analyze_defects_hsv(self, image, show_defects=False, save_defects=None):
        # 1. Konverzia do HSV farebného priestoru
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. Vytiahnutie kanálu sýtosti (Saturation)
        # S-kanál je index 1 (H=0, S=1, V=2)
        s_channel = hsv[:, :, 1]

        # 3. Prahovanie S-kanálu
        # Pixely s vyššou sýtosťou (nad hodnotu 40) budú biele, zvyšok čierny.
        # Hodnotu '40' možno budeš musieť mierne doladiť (napr. 30-50) podľa toho, 
        # ako veľmi sú tvoje ďalšie obrázky vyblednuté.
        _, binary_mask = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY)

        # 4. Morfologické vyčistenie
        # Použijeme tvoj kernel na odstránenie malého šumu a spojenie rozbitých fľakov
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Najprv OPENING na odstránenie drobných bodiek (šumu z papiera)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        # Potom CLOSING na zliatie väčších fľakov plesne do ucelených kontúr
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 5. Hľadanie kontúr na novej maske
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_valid_defect = False
        result_binary_mf = np.zeros_like(s_channel) # Zmenené na s_channel pre správny rozmer
        display_img = image.copy()

        for contour in contours:
            a_k = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            filled_mask = np.zeros_like(s_channel)
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
            a_filled = cv2.countNonZero(filled_mask)

            size_ok = self.is_valid_size(a_k)
            hole_ok = self.is_valid_hole(a_k, a_filled)
            line_ok = self.is_valid_line(a_k, perimeter)
            
            # Tvoju clutter logiku som nechal zakomentovanú tak, ako si ju mal
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(a_k) / hull_area if hull_area > 0 else 0
            is_clutter = aspect_ratio > 6.0 or solidity < 0.3 

            if size_ok and hole_ok and line_ok and not is_clutter:
                found_valid_defect = True
                cv2.drawContours(result_binary_mf, [contour], -1, 255, thickness=cv2.FILLED)
                color = (0, 255, 0) # Zelená pre validnú pleseň/fľak
            else:
                color = (0, 0, 255) # Červená pre odfiltrované objekty

            cv2.drawContours(display_img, [contour], -1, color, 2)

        # Príprava na zobrazenie výsledkov
        # Zobrazíme: Originál -> Sýtosť (ako sivý obrázok) -> Výsledná maska s kontúrami
        s_channel_viz = cv2.cvtColor(s_channel, cv2.COLOR_GRAY2BGR)
        final_mask_viz = cv2.cvtColor(result_binary_mf, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_img, '1.', (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(s_channel_viz, '2.', (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(final_mask_viz, '3.', (10, 30), font, 0.7, (0, 255, 0), 2)

        combined_view = np.hstack((display_img, s_channel_viz, final_mask_viz))

        screen_res = 1500, 800
        scale = min(screen_res[0] / combined_view.shape[1], screen_res[1] / combined_view.shape[0])
        
        if scale < 1.0:
            combined_view = cv2.resize(combined_view, (0,0), fx=scale, fy=scale)

        if show_defects:
            cv2.imshow("Detection Workflow", combined_view)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_defects is not None:
            cv2.imwrite(save_defects, combined_view)

        return found_valid_defect, result_binary_mf

    #Foxing hladanie
    def detect_foxing(self, image_path, s=20, alpha=0.5):
        img = cv2.imread(image_path)
        if img is None:
            return None

        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        _, cr, _ = cv2.split(ycrcb_img)

        max_cr = np.max(cr)
        
        threshold_value = max_cr - s
        _, fox_map = cv2.threshold(cr, threshold_value, 255, cv2.THRESH_BINARY)
        
        fox_map_bin = fox_map // 255
        overlay = img.copy()
        overlay[fox_map_bin == 1] = [0, 0, 255]
        
        fox_overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        return fox_map, cr, fox_overlay

    def preprocess_all_images(self,hsv=False, dir = "images", out_dir = "out"):
        if not os.path.exists(dir):
            return
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(dir, ext)))

        
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Preskakujem (chyba načítania): {path}")
                continue
            
            print(path)
            if hsv:
                self.analyze_defects_hsv(img, show_defects=False, save_defects=out_dir+"\\proc_"+path.split("\\")[1])
            else:
                self.analyze_defects(img, show_defects=False, save_defects=out_dir+"\\proc_"+path.split("\\")[1])
        
        return
            

        
        
