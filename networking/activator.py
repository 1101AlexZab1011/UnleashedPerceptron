from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
from typing import *
import typing as t
from sympy import *


def standard_activation_function(
        name: str,
        ratio: Optional[t.Union[int, float]] = 1,
        offset_x: Optional[t.Union[int, float]] = 0,
        offset_y: Optional[t.Union[int, float]] = 0
):
    return {
        'identity': f'{ratio}*x - {offset_x}',
        'heaviside': f'{ratio}*Heaviside(x - {offset_x}) + {offset_y}',
        'sigmoid': f'{ratio}*1/(1 + exp(-(x - {offset_x}))) + {offset_y}',
        'tanh': f'{ratio}*tanh(x - {offset_x}) + {offset_y}',
        'relu': f'{ratio}*Max(0, x - {offset_x}) + {offset_y}',
        'gelu': f'{ratio}*0.5 * (x - {offset_x}) * (1 + tanh(sqrt(2/pi) *'
                f' ((x - {offset_x}) + 0.044715*(x - {offset_x})**3))) + {offset_y}',
        'softplus': f'{ratio}*log(1 + exp(x - {offset_x})) + {offset_y}',
        'elu': f'{ratio}*Heaviside((x - {offset_x}))*(x - {offset_x}) +'
               f' Heaviside(-(x - {offset_x}))*(exp(x - {offset_x}) - 1) + {offset_y}',
        'lelu': f'{ratio}*Max(0.01*(x - {offset_x}), (x - {offset_x})) + {offset_y}',
        'silu': f'{ratio}*(x - {offset_x})/(1 + exp(-(x - {offset_x}))) + {offset_y}',
        'mish': f'{ratio}*(x - {offset_x})*tanh(log(1 + exp(x - {offset_x}))) + {offset_y}',
        'gaussian': f'{ratio}*exp(-(x - {offset_x})**2) + {offset_y}'
    }[name]


basic_representations = {
    'Max': lambda y, x: np.fmax(y, x),
    'Heaviside': lambda x: np.heaviside(x, 0.5),
    'DiracDelta': np.vectorize(lambda x: 1 if -.001 < x < .001 else 0),
    'exp': np.exp,
    'tanh': np.tanh,
    'log': np.log,
}


class Function(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @name.setter
    @abstractmethod
    def name(self, val):
        pass

    @property
    @abstractmethod
    def equation(self):
        pass

    @equation.setter
    @abstractmethod
    def equation(self, val):
        pass

    @property
    @abstractmethod
    def derivative(self):
        pass

    @derivative.setter
    @abstractmethod
    def derivative(self, val):
        pass

    @property
    @abstractmethod
    def backward(self):
        pass

    @backward.setter
    @abstractmethod
    def backward(self, val):
        pass

    @property
    @abstractmethod
    def forward(self):
        pass

    @forward.setter
    @abstractmethod
    def forward(self, val):
        pass

    def plot(
            self,
            kind: Optional[str] = 'forward',
            x_scale: Optional[tuple[t.Union[int, float], t.Union[int, float]]] = (-5, 5)
    ):
        x = np.linspace(x_scale[0], x_scale[1], 1000)
        if kind == 'forward':
            if type(self.forward(x)) != np.ndarray:
                plt.plot(x, np.ones(x.shape) * self.forward(x))
            else:
                plt.plot(x, self.forward(x))
            plt.show()

        elif kind == 'backward':
            if type(self.backward(x)) != np.ndarray:
                plt.plot(x, np.ones(x.shape) * self.backward(x))
            else:
                plt.plot(x, self.backward(x))
            plt.show()


class ManualFunction(Function):
    def __init__(
            self,
            name: str,
            forward: Callable,
            backward: Callable,
            equation: Optional[str] = None,
            derivative: Optional[str] = None
    ):
        self.name = name
        self._forward = forward
        self._backward = backward
        self._equation = equation
        self._derivative = derivative

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> NoReturn:
        self._name = name

    @property
    def forward(self) -> Callable:
        return self._forward

    @forward.setter
    def forward(self, val: Any) -> NoReturn:
        raise AttributeError('Impossible to change that function on the fly')

    @property
    def backward(self) -> Callable:
        return self._backward

    @backward.setter
    def backward(self, val: Any) -> NoReturn:
        raise AttributeError('Impossible to change that function on the fly')

    @property
    def equation(self) -> str:
        return self._equation

    @equation.setter
    def equation(self, value: str) -> NoReturn:
        self._equation = value

    @property
    def derivative(self) -> str:
        return self._derivative

    @derivative.setter
    def derivative(self, value: str) -> NoReturn:
        self._derivative = value


class Activator(Function):
    def __init__(self,
                 name: str,
                 equation: str,
                 representations: Optional[list] = [basic_representations]
                 ):
        self._representations = representations
        self.name = name
        self.equation = equation

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def equation(self):
        return self._equation

    @property
    def derivative(self):
        return self._derivative

    @property
    def symbol(self):
        return self._symbol

    @equation.setter
    def equation(self, equation_str: str):
        self._equation = sympify(equation_str)
        if len(self._equation.free_symbols) != 1:
            raise ValueError('The activation function must be a function of one variable')
        self._symbol = list(self._equation.free_symbols)[0]
        self._forward = lambdify(self._symbol, self._equation, modules=self._representations)
        self._derivative = self._equation.diff(self._symbol)
        self._backward = lambdify(self._symbol, self._derivative, modules=self._representations)

    @derivative.setter
    def derivative(self, val):
        raise AttributeError('The derivative can be changed only by a new equation')

    @property
    def forward(self):
        return self._forward

    @property
    def backward(self):
        return self._backward

    @forward.setter
    def forward(self, val):
        raise AttributeError('The forward representation of an equation can be changed only by a new equation')

    @backward.setter
    def backward(self, val):
        raise AttributeError('The backward representation of an equation can be changed only by a new equation')

    @symbol.setter
    def symbol(self, val):
        raise AttributeError('The variable name can be changed only by a new equation')


class Loss(Activator):
    def __init__(self, name: str, equation: str):
        super().__init__(name, equation)

    @property
    def equation(self):
        return self._equation

    @equation.setter
    def equation(self, equation_str: str):
        self._equation = sympify(equation_str)
        if len(self._equation.free_symbols) != 2:
            raise ValueError('The loss function must be a function of two variable '
                             'as only one-output neural networks are supported')
        self._symbol = list(self._equation.free_symbols)
        self._forward = lambdify(self._symbol, self._equation, modules=self._representations)
        self._derivative = self._equation.diff(self._symbol[0])
        self._backward = lambdify(self._symbol, self._derivative, modules=self._representations)


standard_activators = {
    'identity': 'x',
    'heaviside': 'Heaviside(x)',
    'sigmoid': '1/(1 + exp(-x))',
    'tanh': 'tanh(x)',
    'relu': 'Max(0, x)',
    'gelu': '0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x**3)))',
    'softplus': 'log(1 + exp(x))',
    'elu': 'Heaviside(x)*x + Heaviside(-x)*(exp(x) - 1)',
    'lelu': 'Max(0.01*x, x)',
    'silu': 'x/(1 + exp(-x))',
    'mish': 'x*tanh(log(1 + exp(x)))',
    'gaussian': 'exp(-x**2)'
}

standard_loss_functions = {
    'logarithmic_loss': '-(y*log(p) + (1 - y)*log(1 - p))',
    'MSE': '(y - p)**2',
}
