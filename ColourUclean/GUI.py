import pygame
from matplotlib.pyplot import imread
from tkinter import *
from tkinter.colorchooser import  *

"""pygame.init()
Screen = pygame.display.set_mode((800, 450))

pressed = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                pressed = False
        elif event.type == pygame.MOUSEMOTION and pressed:
            pygame.draw.circle(Screen, (100, 100, 100), event.pos, 5)


    pygame.time.wait(15)
    pygame.display.flip()"""

def getColour():
    colour = askcolor()
    print(colour)

Button(text='Select Colour', command=getColour).pack()
mainloop()