__version__ = '0.3.6'


from .animation import animation
from .protes import protes
from .protes_general import protes_general
import sys
sys.path.append('../demo/')  
from Ackley_function_P01 import *
from Alpine_function_P02 import *
from Griewank_function_P04 import *
from Michalewicz_function_P05 import *
from Rastrigin_function_P08 import *
from Schwefel_function_P10 import *
