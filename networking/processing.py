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
    def __init__(self, network: SequentialNetwork):
        self.map = network

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, network: SequentialNetwork):
        layers = list()
        for layer in network:
            elements = np.array([
                unit.function.equation if not isinstance(unit, Input) else 'Input'
                for unit in layer
            ])
            uniq_equations = list(collections.Counter(elements).keys())
            uniq_equations = dict(
                zip(
                    uniq_equations, range(len(uniq_equations))
                )
            )
            mapper = list()
            for uniq_equation in uniq_equations:
                n_elements = 0
                for elem in layer:
                    if isinstance(elem, Input) and uniq_equation == 'Input':
                        n_elements += 1
                    elif uniq_equation == elem.function.equation:
                       n_elements += 1
                mapper.append(n_elements)
            layers.append(mapper)
        self._map = layers