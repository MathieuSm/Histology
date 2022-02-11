import os

MainDirectory = r'C:\Users\mathi\OneDrive\Documents\PhD\06_Histology\Cutting Lines'


# Unit conversion
# mm = 1
# pt = 2.835
# pc = 0.236
# px = 3.78

Font = ['Norasi', 'Normal', 18*pt, 'center', 'middle', '#000000']

Text = ['C0002337','C0002337','Proximal Slice','Distal Slice']
XPos = [62*mm, 142*mm, 62*mm, 142*mm]
YPos = [35*mm, 35*mm, 87*mm, 87*mm]

for i in range(4):
     text(Text[i], (XPos[i], YPos[i] + Font[2] / 2),
          font_family=Font[0],
          font_weight=Font[1],
          font_size=Font[2],
          text_align=Font[3],
          text_anchor=Font[4],
          fill=Font[5])


Text = ['Lateral','Medial','Lateral','Medial']
XPos = [35*mm, 89*mm, 115*mm, 168*mm]
YPos = [25*mm, 25*mm, 25*mm, 25*mm]

for i in range(4):
     Transform = 'rotate(90,(' + str(XPos[i]) + ',' + str(YPos[i]) + '))'
     transform(Transform)

     text(Text[i], (XPos[i], YPos[i] + Font[2] / 2),
          font_family=Font[0],
          font_weight=Font[1],
          font_size=Font[2],
          text_align=Font[3],
          text_anchor=Font[4],
          fill=Font[5])

Factor = 0.27312
transform('scale(0.27312, 0.27312)')
FileName = 'C0002337_Proximal.png'
image(os.path.join(MainDirectory,FileName), (40/Factor*mm, 40/Factor*mm), embed=True)

FileName = 'C0002337_Distal.png'
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