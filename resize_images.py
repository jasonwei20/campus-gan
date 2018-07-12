from PIL import Image
import os, sys
from os.path import join

input_dir = 'eleven_campuses'
output_dir = 'eleven_campuses_64'

images = os.listdir( input_dir )
if '.DS_Store' in images:
	images.remove('.DS_Store')
print(len(images), 'images found.')

for i in range(len(images)):
	try:
		image = images[i]
		im = Image.open(join(input_dir, image))
		imResize = im.resize((64, 64), Image.ANTIALIAS)
		imResize.save(join(output_dir, str(i)+'.jpg'))
		print(image, "converted.")
	except:
		print("failed", image)

print("finished")