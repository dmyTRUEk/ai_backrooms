# pipe extensions

from pipe import Pipe

from extensions import *


int_ = Pipe(int)
list_ = Pipe(list)
sorted_ = Pipe(sorted)

shuffled_ = Pipe(shuffled)
string_multisplit_ = Pipe(string_multisplit)
time_to_my_format_ = Pipe(time_to_my_format)
# to_base_ = Pipe(to_base)
to_base36_ = Pipe(to_base36)

