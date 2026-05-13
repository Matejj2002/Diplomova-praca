import cv2 as cv
import numpy as np
from morphology_filters import DetectionFilter


if __name__ == "__main__":
    print("Initialized...")
    filter = DetectionFilter()
    # filter.detect_mold(cv.imread("images/I5.png"), show_defects=True)
    #filter.preprocess_all_images()
    
    img = cv.imread("images/IMG_0278.jpg")
    #filter.detect_mold
    
    # images/I1.png
    # images/I2.png
    # images/I5.jpg
    # images/IMG_0278.jpg
    filter.detect_mold_visual(img, "filter.detect_mold")
    
    #filter.show_hsv_lab_channels(img)
    
    # mapa, cr_kanal = filter.detect_mold(img, show_defects=True)
    
    # mapa, _, overlay_img = filter.detect_foxing("images/I1.png", s=15)
    # cv.imshow('Foxing Overlay', overlay_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()