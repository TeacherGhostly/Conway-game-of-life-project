# -*- coding: utf-8 -*-
"""
The Game of Life (GoL) module named in honour of John Conway

This module defines the classes required for the GoL simulation.

Created on Tue Jan 15 12:21:17 2019

@author: shakes
"""
import numpy as np
from scipy import signal
import rle


class GameOfLife:
    """
    Object for computing Conway's Game of Life (GoL) cellular machine/automata
    """

    def __init__(self, N=256, finite=False, fastMode=False):
        self.grid = np.zeros((N, N), np.int64)
        self.neighborhood = np.ones((3, 3), np.int64)  # 8 connected kernel
        self.neighborhood[1, 1] = 0  # do not count centre pixel
        self.finite = finite
        self.fastMode = fastMode
        self.aliveValue = 1
        self.deadValue = 0
        self.N = N

    def getStates(self):
        """
        Returns the current states of the cells
        """
        return self.grid

    def getGrid(self):
        """
        Same as getStates()
        """
        return self.getStates()

    def evolve(self):
        """
        Given the current states of the cells, apply the GoL rules:
        - Any live cell with fewer than two live neighbors dies, as if by underpopulation.
        - Any live cell with two or three live neighbors lives on to the next generation.
        - Any live cell with more than three live neighbors dies, as if by overpopulation.
        - Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction
        """

        # use convolve2d function to calculate sum of cell's neighbors
        num_neighbors = signal.convolve2d(
            self.grid, self.neighborhood, mode="same", boundary="wrap"
        )

        # birth requires the number of neighbors to be 3 around a dead cell
        birth = (num_neighbors == 3) & (self.grid == 0)

        # surviving requires an alive cell to have 2-3 neighbours
        survive = ((num_neighbors == 2) | (num_neighbors == 3)) & (self.grid == 1)

        # update the grid
        self.grid = np.where(birth | survive, self.aliveValue, self.deadValue)

    def insertBlinker(self, index=(0, 0)):
        """
        Insert a blinker oscillator construct at the index position
        """
        self.grid[index[0], index[1] + 1] = self.aliveValue
        self.grid[index[0] + 1, index[1] + 1] = self.aliveValue
        self.grid[index[0] + 2, index[1] + 1] = self.aliveValue

    def insertGlider(self, index=(0, 0)):
        """
        Insert a glider construct at the index position
        """
        self.grid[index[0], index[1] + 1] = self.aliveValue
        self.grid[index[0] + 1, index[1] + 2] = self.aliveValue
        self.grid[index[0] + 2, index[1]] = self.aliveValue
        self.grid[index[0] + 2, index[1] + 1] = self.aliveValue
        self.grid[index[0] + 2, index[1] + 2] = self.aliveValue

    def insertGliderGun(self, index=(0, 0)):
        """
        Insert a glider construct at the index position
        """
        self.grid[index[0] + 1, index[1] + 25] = self.aliveValue

        self.grid[index[0] + 2, index[1] + 23] = self.aliveValue
        self.grid[index[0] + 2, index[1] + 25] = self.aliveValue

        self.grid[index[0] + 3, index[1] + 13] = self.aliveValue
        self.grid[index[0] + 3, index[1] + 14] = self.aliveValue
        self.grid[index[0] + 3, index[1] + 21] = self.aliveValue
        self.grid[index[0] + 3, index[1] + 22] = self.aliveValue
        self.grid[index[0] + 3, index[1] + 35] = self.aliveValue
        self.grid[index[0] + 3, index[1] + 36] = self.aliveValue

        self.grid[index[0] + 4, index[1] + 12] = self.aliveValue
        self.grid[index[0] + 4, index[1] + 16] = self.aliveValue
        self.grid[index[0] + 4, index[1] + 21] = self.aliveValue
        self.grid[index[0] + 4, index[1] + 22] = self.aliveValue
        self.grid[index[0] + 4, index[1] + 35] = self.aliveValue
        self.grid[index[0] + 4, index[1] + 36] = self.aliveValue

        self.grid[index[0] + 5, index[1] + 1] = self.aliveValue
        self.grid[index[0] + 5, index[1] + 2] = self.aliveValue
        self.grid[index[0] + 5, index[1] + 11] = self.aliveValue
        self.grid[index[0] + 5, index[1] + 17] = self.aliveValue
        self.grid[index[0] + 5, index[1] + 21] = self.aliveValue
        self.grid[index[0] + 5, index[1] + 22] = self.aliveValue

        self.grid[index[0] + 6, index[1] + 1] = self.aliveValue
        self.grid[index[0] + 6, index[1] + 2] = self.aliveValue
        self.grid[index[0] + 6, index[1] + 11] = self.aliveValue
        self.grid[index[0] + 6, index[1] + 15] = self.aliveValue
        self.grid[index[0] + 6, index[1] + 17] = self.aliveValue
        # one of these is wrong
        self.grid[index[0] + 6, index[1] + 18] = self.aliveValue
        self.grid[index[0] + 6, index[1] + 23] = self.aliveValue
        self.grid[index[0] + 6, index[1] + 25] = self.aliveValue

        self.grid[index[0] + 7, index[1] + 11] = self.aliveValue
        self.grid[index[0] + 7, index[1] + 17] = self.aliveValue
        self.grid[index[0] + 7, index[1] + 25] = self.aliveValue

        self.grid[index[0] + 8, index[1] + 12] = self.aliveValue
        self.grid[index[0] + 8, index[1] + 16] = self.aliveValue

        self.grid[index[0] + 9, index[1] + 13] = self.aliveValue
        self.grid[index[0] + 9, index[1] + 14] = self.aliveValue

    def insertFromPlainText(self, txtString, pad=0):
        """
        Assumes txtString contains the entire pattern as a human readable pattern without comments
        """
        lines = txtString.split("\n")
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == "O":
                    self.grid[x + pad, y + pad] = self.aliveValue

    def insertFromRLE(self, rleString, pad=0):
        """
        Given string loaded from RLE file, populate the game grid
        """
        rle_parser = rle.RunLengthEncodedParser(rleString)

        pattern_2d_array = rle_parser.pattern_2d_array

        for y, row in enumerate(pattern_2d_array):
            for x, char in enumerate(row):
                if char == "o":
                    self.grid[x + pad, y + pad] = self.aliveValue
