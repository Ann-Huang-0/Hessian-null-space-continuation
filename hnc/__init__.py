"""Hessian Null Space Continuation (HNC): traverse the set of weights that preserve a trained
network's function, optionally steered toward solutions with chosen properties."""
from . import curvature, potentials
from .directions import Flattest, RandomHeading, Steer, make_direction
from .problem import Problem, function_matching, functional
from .walk import restore, walk
