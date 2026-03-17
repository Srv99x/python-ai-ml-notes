# Revision Notes:
# Topic: Matplotlib basics and plotting/drawing shapes
# - Read this file top-to-bottom and trace the output mentally before running.
# - Change one input/value at a time to understand behavior deeply.
# - Keep this file as a quick recap for this specific concept.

import matplotlib.pyplot as plt 

class Circle(object):
    
    def __init__(self, radius, color):
        self.radius = radius
        self.color = color

    def add_radius(self, r):
        self.radius = self.radius+r
        return self.radius

    def drawCircle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show()

RedCircle = Circle(6, "red")
RedCircle.drawCircle()
