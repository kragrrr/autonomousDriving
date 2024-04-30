from utilities import *

image_path = '/Users/krish/Development/SDC/data/IMG/center_2024_02_06_00_23_54_627.jpg'

# perform preprocessing and show img output

img = mpimg.imread(image_path)

img = preProcessing(img)
# save image
plt.imshow(img)
plt.axis('off')
plt.show()