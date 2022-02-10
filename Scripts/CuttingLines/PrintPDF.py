import os

MainDirectory = '/home/mathieu/Documents/PhD/06_Histology/Cutting Lines/'

FileName = 'C0002074_Proximal.png'

Factor = 0.27312

transform('scale(0.27312, 0.27312)')

image(os.path.join(MainDirectory,FileName), (40/Factor*mm, 40/Factor*mm), embed=True)

FileName = 'C0002074_Distal.png'
image(os.path.join(MainDirectory,FileName), (120/Factor*mm, 40/Factor*mm), embed=True)


# ########################################################################
# # Demonstrate the use of explicitly specified units in Simple Inkscape #
# # Scripting.                                                           #
# ########################################################################
#
# # Define some program parameters.
# tx, ty = 18*pt, 36*pt
# y_inc = 3*cm
# lx, ly = 8*cm, ty - 4*pt
# style(font_size=18*pt, font_family='Norasi')
#
#
# def show_units(name, factor, row):
#     'Present a single unit.'
#     text('10\u00d71 %s:' % name, (tx, ty + row*y_inc))
#     line((lx, ly + row*y_inc), (lx + 10*factor, ly + row*y_inc),
#          stroke_width=1*factor, stroke='blue')
#
#
# # Compare various units.
# show_units('user units',  1,    0)
# show_units('pixels',      px,   1)
# show_units('points',      pt,   2)
# show_units('millimeters', mm,   3)
# show_units('centimeters', cm,   4)
# show_units('inches',      inch, 5)