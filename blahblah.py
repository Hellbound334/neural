import numpy
import matplotlib.pyplot
import doggy
a = numpy.zeros([3, 2])

a[0, 0] = 1
a[0, 1] = 2
a[1, 0] = 9
a[2, 1] = 12

sizzles = doggy.Dog("Sizzles", 37)
sizzles.set_temperature(40)
sizzles.status()