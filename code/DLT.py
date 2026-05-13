import cv2
import numpy as np
from scipy.signal import find_peaks
import itertools
import matplotlib.pyplot as plt

class DLTMoldDetector:
    def __init__(self, a_min=20, a_max=5000000, hole_threshold=0.5, line_threshold=0.6, kernel_size=5):
        self.a_min = a_min           
        self.a_max = a_max           
        self.hole_threshold = hole_threshold 
        self.line_threshold = line_threshold
        self.kernel_size = kernel_size

    def _get_black_pixel_percentage(self, binary_img):
        total_pixels = binary_img.shape[0] * binary_img.shape[1]
        black_pixels = total_pixels - cv2.countNonZero(binary_img)
        return black_pixels / total_pixels

    def detect_dlt(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        binary_images = []
        y = []
        for j in range(1, 21):
            t_j = (j / 20.0) * 255
            _, bin_img = cv2.threshold(gray, t_j, 255, cv2.THRESH_BINARY)
            binary_images.append(bin_img)
            y.append(self._get_black_pixel_percentage(bin_img))

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
            j_peak2_star = min(19, j_peak2 + 2)  
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
            y2.append(self._get_black_pixel_percentage(sub_img))

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

        # optimum_image = cv2.bitwise_not(optimum_image) # invertovanie obrazka ak potrebne
        
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        processed_mask = cv2.morphologyEx(optimum_image, cv2.MORPH_CLOSE, kernel)
        
        processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(processed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_defects = []
        
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    continue
                    
                a_filled = cv2.contourArea(contour)       
                perimeter = cv2.arcLength(contour, True)
                
                hole_area = 0
                child_idx = hierarchy[0][i][2]
                while child_idx != -1:
                    hole_area += cv2.contourArea(contours[child_idx])
                    child_idx = hierarchy[0][child_idx][0]
                
                a_k = a_filled - hole_area               

                if not (self.a_min < a_k < self.a_max):   
                    continue
                    
                if a_k == 0: continue
                hole_ratio = (a_filled - a_k) / a_k        
                if hole_ratio > self.hole_threshold:      
                    continue
                    
                if perimeter == 0: continue
                if (a_k / perimeter) <= self.line_threshold: #
                    continue
                    
                valid_defects.append(contour)

        return valid_defects, optimum_image, y
    
    
    def visualize_results(self,image, valid_contours, optimum_mask, y_curve):
        output_img = image.copy()
        cv2.drawContours(output_img, valid_contours, -1, (0, 255, 0), 2)
        
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detekcia (Počet: {len(valid_contours)})")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(optimum_mask, cmap='gray')
        plt.title("Optimálna maska (m_s)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.plot(range(1, 21), y_curve, marker='o')
        plt.title("Krivka podielu čiernych pixelov (y)")
        plt.xlabel("Úroveň prahu (j)")
        plt.ylabel("Podiel pixelov")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# detector = DLTMoldDetector()
# image = cv2.imread('images/I1.png')
# kontury, maska, y_data = detector.detect_dlt(image) 
# detector.visualize_results(image, kontury, maska, y_data)