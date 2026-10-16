import cv2
import numpy as np
import turtle
import time
import sys


IMAGE_PATH = "image.png"

# High resolution. Increase to 1400 or 1600 if your PC is fast.
TARGET_WIDTH = 750

# Animation controls
OUTLINE_DELAY = 0.0015
FILL_DELAY = 0.001
POINTS_PER_UPDATE = 25

# Keep tiny contours for ornaments and facial details.
MIN_CONTOUR_AREA = 0.5

# Lower values preserve more contour details.
SIMPLIFY_EPSILON = 0.45

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"ERROR: Could not load '{IMAGE_PATH}'")
    print("Place image.png in the same folder as this Python file.")
    sys.exit(1)

original_height, original_width = image.shape

scale = TARGET_WIDTH / original_width
target_height = int(original_height * scale)

image = cv2.resize(
    image,
    (TARGET_WIDTH, target_height),
    interpolation=cv2.INTER_CUBIC
)

print(f"Processing resolution: {TARGET_WIDTH} x {target_height}")

# ============================================================
# 2. PREPROCESS IMAGE WITHOUT LOSING FINE LINES
# ============================================================

# Light sharpening preserves jewelry and facial linework.
blur = cv2.GaussianBlur(image, (0, 0), 0.8)

sharpened = cv2.addWeighted(
    image, 1.25,
    blur, -0.25,
    0
)

# Automatic threshold selection.
_, binary = cv2.threshold(
    sharpened,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Find all contours and their parent-child relationships.
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_NONE
)

if hierarchy is None or not contours:
    print("ERROR: No contours were found.")
    sys.exit(1)

hierarchy = hierarchy[0]


# ============================================================
# 3. BUILD CONTOUR INFORMATION
# ============================================================

def contour_depth(index):
    depth = 0
    parent = hierarchy[index][3]

    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]

    return depth


shapes = []

for index, contour in enumerate(contours):
    area = abs(cv2.contourArea(contour))

    if area < MIN_CONTOUR_AREA:
        continue

    # Tiny contours are retained because they may be real details.
    polygon = cv2.approxPolyDP(
        contour,
        SIMPLIFY_EPSILON,
        True
    ).reshape(-1, 2)

    if len(polygon) < 3:
        continue

    shapes.append({
        "index": index,
        "depth": contour_depth(index),
        "points": polygon,
        "area": area
    })

# Parents are filled before children.
shapes.sort(key=lambda shape: shape["depth"])

print(f"Contours detected: {len(contours)}")
print(f"Contours retained: {len(shapes)}")


# ============================================================
# 4. SET UP TURTLE
# ============================================================

screen = turtle.Screen()

screen.setup(
    width=TARGET_WIDTH + 80,
    height=target_height + 100
)

screen.title("Maa Durga — Outline, Fill and Detailing")
screen.bgcolor("white")

# Manual refresh lets us control animation speed.
screen.tracer(0, 0)

pen = turtle.Turtle()
pen.shape("classic")
pen.color("black")
pen.pensize(1)
pen.speed(1)
pen.penup()
pen.showturtle()

half_width = TARGET_WIDTH / 2
half_height = target_height / 2


def convert_point(point):
    x, y = point
    return x - half_width, half_height - y


# ============================================================
# 5. ANIMATED OUTLINE DRAWING
# ============================================================

def draw_outline(points, delay=OUTLINE_DELAY):
    if len(points) < 3:
        return

    x, y = convert_point(points[0])

    pen.penup()
    pen.goto(x, y)
    pen.pendown()

    for i, point in enumerate(points[1:], start=1):
        x, y = convert_point(point)
        pen.goto(x, y)

        # Refresh regularly so the cursor movement is visible.
        if i % POINTS_PER_UPDATE == 0:
            screen.update()
            time.sleep(delay)

    # Close the outline.
    x, y = convert_point(points[0])
    pen.goto(x, y)

    screen.update()
    time.sleep(delay)

    pen.penup()


print("\nSTAGE 1/3: Drawing black outlines...")
print("Watch the cursor trace the artwork.")

for number, shape in enumerate(shapes, start=1):
    draw_outline(shape["points"])

    if number % 20 == 0:
        print(f"Outline progress: {number}/{len(shapes)}")

pen.hideturtle()
screen.update()

print("Outline stage completed!")

# Pause so you can see the completed sketch before filling.
time.sleep(0.5)


# ============================================================
# 6. FILL SHAPES IN BLACK AND WHITE
# ============================================================

def fill_shape(points, fill_color):
    if len(points) < 3:
        return

    x, y = convert_point(points[0])

    pen.penup()
    pen.goto(x, y)

    # Fill without drawing a thick additional border.
    pen.pencolor(fill_color)
    pen.fillcolor(fill_color)
    pen.pendown()

    pen.begin_fill()

    for i, point in enumerate(points[1:], start=1):
        x, y = convert_point(point)
        pen.goto(x, y)

        if i % 100 == 0:
            screen.update()
            time.sleep(FILL_DELAY)

    x, y = convert_point(points[0])
    pen.goto(x, y)

    pen.end_fill()
    pen.penup()


print("\nSTAGE 2/3: Filling black regions and white cutouts...")

pen.hideturtle()

for number, shape in enumerate(shapes, start=1):
    # Even hierarchy depth = black.
    # Odd hierarchy depth = white.
    #
    # The alternating fills preserve holes in the artwork.
    fill_color = (
        "black" if shape["depth"] % 2 == 0
        else "white"
    )

    fill_shape(shape["points"], fill_color)

    if number % 20 == 0:
        screen.update()
        print(f"Fill progress: {number}/{len(shapes)}")

screen.update()
print("Filling stage completed!")

time.sleep(1.0)


# ============================================================
# 7. RESTORE ALL BLACK DETAIL OUTLINES
# ============================================================

print("\nSTAGE 3/3: Restoring fine black details...")

pen.showturtle()
pen.fillcolor("#000000")
pen.pencolor("#000000")
pen.pensize(1)

for number, shape in enumerate(shapes, start=1):
    draw_outline(
        shape["points"],
        delay=OUTLINE_DELAY * 0.5
    )

    if number % 20 == 0:
        print(f"Detail progress: {number}/{len(shapes)}")

pen.hideturtle()
screen.update()

print("\nMaa Durga drawing completed!")
print("All three stages are finished.")

screen.mainloop()