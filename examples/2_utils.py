"""
MIT License

Copyright (c) 2021 booleangabs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# local
from context import image_processing as ipn


img = ipn.read_image("inputs/peppers2.tif", ipn.READ_GRAY)
img_color = ipn.read_image("inputs/peppers3.tif", ipn.READ_COLOR)

mn, mx = img.min(), img.max()

img_norm = ipn.normalize(img.astype("float"))
print(img_norm.min() == 0 and img_norm.max() == 1)

img_remap = ipn.map_to_range(img.astype("float"), mn, mx)
print(img_remap.min() == mn and img_remap.max() == mx)

r, g, b = ipn.split(img_color)
img_merged = ipn.merge([r, g, b])
ipn.show(img_merged) 

