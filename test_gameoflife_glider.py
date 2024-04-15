# -*- coding: utf-8 -*-
"""
Game of life script with animated evolution

Created on Tue Jan 15 12:37:52 2019

@author: shakes
"""
import conway

N = 64
padding = 0

with open("herschel_loop1_plaintext.txt", "r") as file:
    pattern = file.read()

# create the game of life object
life = conway.GameOfLife(N, fastMode=True)
life.insertBlinker((0, 0))
# life.insertGlider((10, 10))
# life.insertGliderGun((10, 10))
# life.insertFromPlainText(pattern, padding)
cells = life.getStates()  # initial state

# -------------------------------
# plot cells
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

plt.gray()

img = plt.imshow(cells, animated=True)


def animate(i):
    """perform animation step"""
    global life

    life.evolve()
    cellsUpdated = life.getStates()

    img.set_array(cellsUpdated)

    return (img,)


interval = 200  # ms

# animate 24 frames with interval between them calling animate function at each frame
ani = animation.FuncAnimation(fig, animate, frames=24, interval=interval, blit=True)

plt.show()
