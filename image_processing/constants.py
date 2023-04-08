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

# Image reading (READ)
READ_COLOR = 0
READ_GRAY = 1

# Color conversion (COLOR)
COLOR_RGB2BGR = 0
COLOR_BGR2RGB = 1
COLOR_RGB2GRAY = 2
COLOR_GRAY2RGB = 3
COLOR_RGB2RGBA = 4
COLOR_RGBA2RGB = 5
COLOR_RGB2HSV = 6
COLOR_HSV2RGB = 7
COLOR_RGB2HLS = 8
COLOR_HLS2RGB = 9
COLOR_RGB2LAB = 10
COLOR_LAB2RGB = 11

# Thresholding (THRESH)
THRESH_BINARY = 0
THRESH_INVERSE = 1
THRESH_TOZERO = 2
THRESH_TOMAX = 3
THRESH_OTSU = 4

# Image morphology (MORPH)
MORPH_SQUARE = 0
MORPH_CIRCLE = 1
MORPH_CROSS = 2