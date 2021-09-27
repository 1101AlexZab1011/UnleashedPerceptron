from collections import UserList
from dataclasses import dataclass
import time
from typing import Optional, Union, Iterable, Generator
from abc import ABC

from networking.activator import *


class Unit(ABC):
    def __init__(self, id: Optional[str] = None):
        if not id:
            self._id = str(time.time_ns())

    def __str__(self):
        return 'Neural network unit'

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.id = self.id + '_copy'
        return obj

    def copy(self):
        return self.__copy__()


class ActiveUnit(Unit, ABC):
    def __init__(self, function: t.Union[Activator, ManualFunction], id: Optional[str] = None):
        super().__init__(id)
        self._function = function

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function: t.Union[Activator, ManualFunction]):
        self._function = function

    def __str__(self):
        pass


class Neuron(ActiveUnit):
    def __init__(self, function: t.Union[Activator, ManualFunction], id: Optional[str] = None):
        super().__init__(function, id)

    def __str__(self):
        return f'Neural network neuron, activated by {self.function.equation}'


class Input(Unit):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)

    def __str__(self):
        return f'Neural network input'


class Output(ActiveUnit):
    def __init__(self, function: t.Union[Activator, ManualFunction], id: Optional[str] = None):
        super().__init__(function, id)

    def __str__(self):
        return f'Neural network output, loss: {self.function.equation}'


class UnitsContainer(Unit, ABC, UserList):

    def __init__(self, *args: Unit, id: Optional[str] = None):
        super().__init__(id)
        if len(list([*args])) == 1 and isinstance(*args, Generator):
            self.data = list(*args)
        else:
            data = list()
            for arg in args:
                if isinstance(arg, Iterable) and not isinstance(arg, UnitsContainer):
                    data += list(arg)
                else:
                    data.append(arg)
            self.data = data

    def __add__(self, other: Unit):
        pass

    def __str__(self):
        return 'Container of neural network units'

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, elements: Iterable[Unit]):
        self._data = elements


class Layer(UnitsContainer):
    def __init__(self,
                 *args: t.Union[
                     Neuron, Input, Output,
                     Generator[t.Union[Neuron, Input, Output], None, None],
                     list[
                         t.Union[Neuron, Input, Output],
                         Neuron, Input, Output
                     ]
                 ],
                 id: Optional[str] = None):
        super().__init__(*args, id=id)

    def __str__(self):
        neurons, inputs, outputs = 0, 0, 0

        for elem in self.data:
            if isinstance(elem, Neuron):
                neurons += 1
            elif isinstance(elem, Input):
                inputs += 1
            elif isinstance(elem, Output):
                outputs += 1

        out = 'Neural network layer: '
        for elem, name in zip([neurons, inputs, outputs], ['neuron', 'input', 'output']):
            if elem != 0:
                if elem > 1:
                    out += f'{elem} {name}s '
                else:
                    out += f'{elem} {name} '
        return out

    def __add__(self, other: Unit):
        if isinstance(other, type(self)):
            self.data = self.data + other.data
        elif isinstance(other, Neuron) or isinstance(other, Input) or isinstance(other, Output):
            self.data += other
        else:
            raise ValueError(f'Impossible to concatenate Layer and {type(other)}')
        return self


class SequentialNetwork(UnitsContainer):
    def __init__(self, *args: t.Union[Layer, Generator[Layer, None, None], list[Layer]], id: Optional[str] = None):
        super().__init__(*args, id=id)

    def __str__(self):
        structure = list()
        for layer in self.data:
            if not isinstance(layer, Layer):
                raise ValueError(f'{type(layer)} object among neural network layers')
            structure.append(str(len(layer)))
        ', '.join(structure)

        return f'Neural network of {len(self)} layers: {tuple(structure)}'

    def __add__(self, other: Unit):
        if isinstance(other, type(self)):
            self.data = self.data + other.data
        elif isinstance(other, Layer):
            self.data += other
        else:
            raise ValueError(f'Impossible to concatenate SequentialNetwork and {type(other)}')
        return self

    def print(self, kind: Optional[str] = 'forward', detailed: Optional[bool] = False):
        print()
        strings = list()
        for layer in self:
            elems = list()
            for elem in layer:
                if not isinstance(elem, Input):
                    elem_detailed = str(elem.function.equation) if kind == 'forward' else str(elem.function.derivative)
                    if isinstance(elem, Neuron):
                        if not detailed:
                            elems.append(elem.function.name)
                        else:
                            elems.append(elem_detailed)
                    elif isinstance(elem, Output):
                        if not detailed:
                            elems.append('Output')
                        else:
                            elems.append(elem_detailed)
                else:
                    elems.append('Input')

            strings.append(f'[ {" | ".join(elems)} ]')

        space = len(max(strings, key=len))
        for string, i in zip(strings, range(len(strings))):
            print(string.center(space, ' '))
            if i != len(strings) - 1:
                print()
                if kind == 'forward':
                    print('|'.center(space, ' '))
                    print('v'.center(space, ' '))
                elif kind == 'backward':
                    print('^'.center(space, ' '))
                    print('|'.center(space, ' '))
            print()