import enum
import numpy as np


# noinspection PyArgumentList
class ObjectType(enum.Enum):
    AGENT = enum.auto()
    NEIGHBOUR = enum.auto()
    CENTERLINE = enum.auto()
    CANDIDATE_CENTERLINE = enum.auto()

    @property
    def one_hot(self) -> np.ndarray:
        vector = np.zeros(shape=(len(ObjectType)), dtype=np.float32)
        vector[self.value-1] = 1.0
        return vector

    @classmethod
    def from_one_hot(cls, encoded):
        return ObjectType(np.argmax(encoded)+1)

    @property
    def color(self):
        if self.value == ObjectType.AGENT.value:
            return 'blue'
        elif self.value == ObjectType.AGENT.NEIGHBOUR.value:
            return 'cyan'
        elif self.value == ObjectType.CENTERLINE.value:
            return 'gray'
        elif self.value == ObjectType.CANDIDATE_CENTERLINE.value:
            return 'red'
        else:
            assert 'Invalid Program State!'

    @property
    def label(self):
        if self.value == ObjectType.AGENT.value:
            return 'agent'
        elif self.value == ObjectType.AGENT.NEIGHBOUR.value:
            return 'neighbor'
        elif self.value == ObjectType.CENTERLINE.value:
            return 'centerline'
        elif self.value == ObjectType.CANDIDATE_CENTERLINE.value:
            return 'candidate_centerline'
        else:
            assert 'Invalid Program State!'
