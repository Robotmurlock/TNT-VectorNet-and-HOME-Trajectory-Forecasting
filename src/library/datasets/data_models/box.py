"""
BBox
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
        """
        Returns: Box height
        """
        return self.bottom - self.up

    @property
    def width(self):
        """
        Returns: Box width
        """
        return self.right - self.left

    def move(self, dx: int, dy: int):
        """
        Moves box by (dx, dy)

        Args:
            dx: x-axis movement
            dy: y-axis movement

        Returns: New object with new location
        """
        return RectangleBox(self.up+dx, self.left+dy, self.bottom+dx, self.right+dy)

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
