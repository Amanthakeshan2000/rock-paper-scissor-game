import cv2
import numpy as np

def overlay_image(background, foreground, position, radius=None):
    """Overlay the foreground image on the background at the specified position with optional circular mask"""
    if foreground is None:
        return background
        
    # Resize the foreground image
    foreground = cv2.resize(foreground, (150, 150))
    
    # Get dimensions
    h, w = foreground.shape[:2]
    x, y = position
    
    # Check if position is within frame boundaries
    bg_h, bg_w = background.shape[:2]
    if x + w > bg_w or y + h > bg_h or x < 0 or y < 0:
        return background
        
    # Create circular mask if radius is specified
    if radius is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), radius, 255, -1)
        
        # Apply mask to foreground
        if foreground.shape[2] == 4:  # If image has alpha channel
            alpha = foreground[:, :, 3] / 255.0
            alpha = alpha * (mask / 255.0)  # Combine with circular mask
            for c in range(3):
                background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - alpha) + foreground[:, :, c] * alpha
        else:
            for c in range(3):
                background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - mask/255.0) + foreground[:, :, c] * (mask/255.0)
    else:
        # Regular overlay without mask
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255.0
            for c in range(3):
                background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - alpha) + foreground[:, :, c] * alpha
        else:
            background[y:y+h, x:x+w] = foreground
        
    return background 