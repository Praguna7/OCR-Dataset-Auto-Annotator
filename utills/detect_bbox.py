import numpy as np
import sys
# sys.path.append('./UVDoc')
sys.path.append('./utills')
# from unwarp import unwarp_img
from docuwarp.unwarp import Unwarp
import cv2
from PIL import Image as PilImage
from utills.utills import utills,Image
import matplotlib.pyplot as plt
import copy
unwarp = Unwarp()

def draw_bbox(image,line_bboxes):
    # Loop through the bounding boxes and draw them on the image
    for bbox in line_bboxes:
        x_min, y_min, width, height = bbox
        top_left = (x_min, y_min)
        bottom_right = (x_min + width, y_min + height)
        
        # Draw the rectangle (bounding box) on the image
        cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

    # Display the image with bounding boxes using matplotlib
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.axis('off')  # Hide the axes
    plt.show()

def extract_bbox_with_text(page, photo_image):
    pixmap = page.get_pixmap()

    # Convert Pixmap to NumPy array
    img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)

    # Convert grayscale or RGB to BGR (for OpenCV compatibility)
    if pixmap.n == 4:  # RGBA
        pdf_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif pixmap.n == 3:  # RGB
        pdf_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # photo_img_unwarp = unwarp_img(photo_image)
    photo_img_unwarp = Image(photo_image).unwarp()
    photo_img_gray = cv2.cvtColor(photo_img_unwarp, cv2.COLOR_RGB2GRAY)
    photo_img_deskewed =Image(photo_img_gray).deskew()

    aligned_pdg_img, homography_matrix = Image(pdf_img).allign_pdf_img_to_photo_img(photo_img_deskewed,0)

    line_bboxes_aligned_pdf = utills.findLines(aligned_pdg_img)


    line_bboxes_map_from_aligned_to_original_pdf = utills.invert_aligned_pdf_img_bboxes_to_original_pdf_img_bboxes(homography_matrix,line_bboxes_aligned_pdf)
    # draw_bbox(copy.deepcopy(aligned_pdg_img),line_bboxes_aligned_pdf)

    bbox_to_unicode_dict = utills.extract_text_bbox_text_from_pdf(page=page,original_pdf_image_gray=pdf_img,line_bboxes_map_from_aligned_to_original_pdf=line_bboxes_map_from_aligned_to_original_pdf,homography_matrix=homography_matrix)

    return bbox_to_unicode_dict, photo_img_deskewed
   

