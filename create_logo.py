from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create a new image with a white background
    width = 500
    height = 500
    background_color = (255, 255, 255)
    image = Image.new('RGBA', (width, height), background_color)
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    
    # Draw a circle
    circle_color = (70, 130, 180)  # Steel blue
    circle_center = (width // 2, height // 2)
    circle_radius = 200
    draw.ellipse(
        [
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius
        ],
        fill=circle_color
    )
    
    # Draw a wand symbol
    wand_color = (255, 255, 255)  # White
    wand_start = (width // 2 - 100, height // 2 + 100)
    wand_end = (width // 2 + 100, height // 2 - 100)
    draw.line([wand_start, wand_end], fill=wand_color, width=20)
    
    # Draw a star at the wand tip
    star_points = [
        (wand_end[0], wand_end[1] - 30),
        (wand_end[0] + 10, wand_end[1] - 10),
        (wand_end[0] + 30, wand_end[1] - 10),
        (wand_end[0] + 15, wand_end[1] + 5),
        (wand_end[0] + 20, wand_end[1] + 25),
        (wand_end[0], wand_end[1] + 15),
        (wand_end[0] - 20, wand_end[1] + 25),
        (wand_end[0] - 15, wand_end[1] + 5),
        (wand_end[0] - 30, wand_end[1] - 10),
        (wand_end[0] - 10, wand_end[1] - 10),
    ]
    draw.polygon(star_points, fill=wand_color)
    
    # Save the image
    image.save('logo.png', 'PNG')

if __name__ == '__main__':
    create_logo() 