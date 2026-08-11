import numpy as np
from PIL import Image, ImageDraw

# The provided binary string
binary_str = "01000100001100011101100000000111101010000101000010001001000011010000100001001000"

# Reshape into 8x10 grid
grid = np.array(list(binary_str)).reshape(8, 10)

# Settings for the image
cell_size = 40
padding = 20
dot_radius = 10
img_width = 10 * cell_size + 2 * padding
img_height = 8 * cell_size + 2 * padding

# Create black background image
img = Image.new("RGB", (img_width, img_height), "black")
draw = ImageDraw.Draw(img)

# Draw the dots
for r in range(8):
    for c in range(10):
        if grid[r, c] == '1':
            x = padding + c * cell_size + cell_size // 2
            y = padding + r * cell_size + cell_size // 2
            draw.ellipse(
                [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                fill="white"
            )

# Save or display (in a real scenario, I'd return the image)
img.save("data_grid.png")