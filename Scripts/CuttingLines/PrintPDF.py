import os
import numpy as np

# MainDirectory = r'C:\Users\mathi\OneDrive\Documents\PhD\06_Histology\Cutting Lines'
MainDirectory = '/home/mathieu/Documents/PhD/06_Histology/Cutting Lines/'


# Unit conversion
# mm = 1
# pt = 2.835
# pc = 0.236
# px = 3.78

Font = ['Norasi', 'Normal', 18*pt, 'center', 'middle', '#000000']

Scans = ['C0002074','C0002076','C0002078']
Scans = ['C0002337','C0002338']

Factor = 0.27312

Delta = 80

ScanNumber = 0

for Scan in Scans:

     Text = [Scan, Scan, 'Proximal Slice', 'Distal Slice']

     XPos = np.array([62, 142, 62, 142])
     YPos = np.array([35, 35, 87, 87])

     for i in range(4):

          X = (XPos[i] + 0 * Delta) * mm
          Y = (YPos[i] + ScanNumber * Delta) * mm

          text(Text[i], (X, Y + Font[2] / 2),
               font_family=Font[0],
               font_weight=Font[1],
               font_size=Font[2],
               text_align=Font[3],
               text_anchor=Font[4],
               fill=Font[5])


     Text = ['Lateral','Medial','Lateral','Medial']

     XPos = [35, 88, 115, 168]
     YPos = [62, 62, 62, 62]

     for i in range(4):

          X = (XPos[i] + 0 * Delta) * mm
          Y = (YPos[i] + ScanNumber * Delta) * mm

          Transform = 'rotate(-90,(' + str(X) + ',' + str(Y) + '))'
          transform(Transform)

          text(Text[i], (X, Y + Font[2] / 2),
               font_family=Font[0],
               font_weight=Font[1],
               font_size=Font[2],
               text_align=Font[3],
               text_anchor=Font[4],
               fill=Font[5])



     Transform = 'scale(' + str(Factor) + ',' + str(Factor) + ')'
     transform(Transform)
     FileName = Scan + '_Proximal.png'
     X = (40 + 0 * Delta) / Factor * mm
     Y = (40 + ScanNumber * Delta) / Factor * mm
     image(os.path.join(MainDirectory,FileName), (X, Y), embed=True)

     FileName = Scan + '_Distal.png'
     X = (120 + 0 * Delta) / Factor * mm
     Y = (40 + ScanNumber * Delta) / Factor * mm
     image(os.path.join(MainDirectory,FileName), (X, Y), embed=True)
     transform('scale(1, 1)')

     ScanNumber += 1



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