import pygame
import numpy as np
import cv2
from PIL import Image
from skimage import color

# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

def rgb_to_lab(img):
    rgb = np.array(img)
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    return color.rgb2lab(rgb)

def adjust_lab_values(lab_image, deficiency):
    L, A, B = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]
    L_range = np.max(L) - np.min(L)

    L_adjusted = L.copy()
    A_adjusted = A.copy()
    B_adjusted = B.copy()

    if deficiency == 'protanopia':
        max_A, min_A = np.max(A), np.min(A)
        A_adjusted = np.where(A > 0, A + (max_A - A) * 0.9, A - (A - min_A) * 0.9)
        max_B, min_B = np.max(B), np.min(B)
        B_adjusted = np.where(B > 0, B + (max_B - B) * -1, B - (B - min_B) * -1)
    elif deficiency == 'deuteranopia':
        max_B, min_B = np.max(B), np.min(B)
        B_adjusted = np.where(B > 0, B + (max_B - B) * -.7, B - (B - min_B) * -.7)
        max_A, min_A = np.max(A), np.min(A)
        A_adjusted = np.where(A > 0, A + (max_A - A) * 0.8, A - (A - min_A) * 0.8)
    elif deficiency == 'tritanopia':
        max_L, min_L = np.max(L), np.min(L)
        L_adjusted = np.where(L > 0, L + (max_L - L) * 0.2, L - (L - min_L) * 0.2)
        max_B, min_B = np.max(B), np.min(B)
        B_adjusted = np.where(B > 0, B + (max_B - B) * .7, B - (B - min_B) * .7)
        max_A, min_A = np.max(A), np.min(A)
        A_adjusted = np.where(A > 0, A + (max_A - A) * .9, A - (A - min_A) * .9)

    lab_image[:, :, 0] = np.clip(L_adjusted, np.min(L), np.max(L))
    lab_image[:, :, 1] = np.clip(A_adjusted, np.min(A), np.max(A))
    lab_image[:, :, 2] = np.clip(B_adjusted, np.min(B), np.max(B))
    return lab_image

def gamma_correction(rgb, gamma):
    return np.clip(np.power(rgb, 1/gamma), 0, 1)

def remove_undefined_pixels(image, background_color=[0, 0, 0]):
    img_array = np.array(image)
    undefined_pixels = np.all(img_array[..., :3] == 0, axis=-1)
    img_array[undefined_pixels] = background_color
    return Image.fromarray(img_array)

def correct_color_vision(frame, deficiency):
    lab_image = rgb_to_lab(frame)
    adjusted_lab_image = adjust_lab_values(lab_image, deficiency)
    adjusted_rgb_image = color.lab2rgb(adjusted_lab_image)
    gamma_value = 2.2
    gamma_corrected_rgb_image = gamma_correction(adjusted_rgb_image, gamma_value)
    return (gamma_corrected_rgb_image * 255).astype(np.uint8)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AR Color Vision Corrector")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    clock = pygame.time.Clock()
    running = True

    image_path = "it.jpeg"  # Placeholder image path
    deficiency = 'deuteranopia'  # Placeholder deficiency type

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect

        # Apply color vision correction
        corrected_frame = correct_color_vision(frame, deficiency)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Fill screen with white
        pygame.surfarray.blit_array(screen, corrected_frame)  # Blit corrected frame onto Pygame screen
        pygame.display.flip()
        clock.tick(60)

    # Release the webcam and quit Pygame
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
