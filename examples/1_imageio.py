from context import image_processing as ipn

img = ipn.read_image("inputs/peppers3.tif", ipn.READ_COLOR)
ipn.show(img)
ipn.write_image(img, "outputs/peppers_opened.png")

img1 = ipn.read_image("inputs/peppers3.tif", ipn.READ_GRAY)
ipn.show(img1)
ipn.write_image(img1, "outputs/peppers_gray_opened.png")

