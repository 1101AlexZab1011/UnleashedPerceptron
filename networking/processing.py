import collections
from abc import ABC, abstractmethod

from networking.units import *


class SequentialComputer(object):
    def __init__(self, network: SequentialNetwork):
        self.forward = network
        self.backward = network

    @property
    def forward(self):
        return self._forward

    @forward.setter
    def forward(self, network: SequentialNetwork):
        layers = list()
        for layer in network:
            elements = np.array([
                unit.function.forward if not isinstance(unit, Input) else 'Input'
                for unit in layer
            ])
            uniq = list(collections.Counter(elements).keys())
            layers.append(
                dict(
                    zip(
                        range(len(uniq)), uniq)
                )
            )
        self._forward = layers

    @property
    def backward(self):
        return self._backward

    @backward.setter
    def backward(self, network: SequentialNetwork):
        layers = list()
        for layer in network:
            elements = np.array([
                unit.function.backward if not isinstance(unit, Input) else 'Input'
                for unit in layer
            ])
            uniq = list(collections.Counter(elements).keys())
            layers.append(
                dict(
                    zip(
                        range(len(uniq)), uniq)
                )
            )
        self._backward = layers


class SequentialStorage(object):
    def __init__(self):
        pass