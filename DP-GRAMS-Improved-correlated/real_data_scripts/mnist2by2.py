from PIL import Image

# Load your existing figure
im = Image.open("results/mnist_centroid_comparison/mnist_clustering_comparison.png")

# Optionally split into four sub-images (requires knowing coordinates of each subplot)
width, height = im.size
w2, h2 = width//4, height  # original was 1x4
subplots = [im.crop((i*w2, 0, (i+1)*w2, height)) for i in range(4)]

# Create new 2x2 canvas
new_im = Image.new("RGB", (w2*2, h2*2))

# Paste subplots in 2x2
positions = [(0,0), (w2,0), (0,h2), (w2,h2)]
for pos, sub in zip(positions, subplots):
    new_im.paste(sub, pos)

# Save
new_im.save("results/mnist_centroid_comparison/mnist_clustering_comparison_2x2.png")
new_im.show()
