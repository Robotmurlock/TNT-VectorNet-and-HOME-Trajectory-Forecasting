"""
TODO: docstring
"""
from dataclasses import dataclass


@dataclass
class RectangleBox:
    up: int
    left: int
    bottom: int
    right: int

    @property
    def height(self):
        return self.bottom - self.up

    @property
    def width(self):
        return self.right - self.left

    def move(self, x: int, y: int):
        return RectangleBox(self.up+x, self.left+y, self.bottom+x, self.right+y)

    def contains(self, x: int, y: int) -> bool:
        """
        Checks if box contains given point

        Args:
            x: X (first) coordinate
            y: Y (second) coordinate

        Returns: True if box contains point else False
        """
        return (self.up <= x <= self.bottom) and (self.left <= y <= self.right)

    def __repr__(self):
        return f'RectangleBox([{self.up}:{self.bottom}, {self.left}:{self.right}])'
