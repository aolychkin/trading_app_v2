import numpy as np


def converter(old):
  old_min, old_max = -3, 3
  new_min, new_max = 0, 2

  old_range = old_max - old_min
  new_range = new_max - new_min
  converted = ((old - old_min) * new_range / old_range) + new_min
  return converted


def converter_multiplication(one, two):
  old_min, old_max = -5, 5
  new_min, new_max = 0, 10

  old_range = old_max - old_min
  new_range = new_max - new_min

  converted_one = ((one - old_min) * new_range / old_range) + new_min
  converted_two = ((two - old_min) * new_range / old_range) + new_min
  print(converted_one)
  print(converted_two)
  return converted_one / converted_two - 1


def converter_two(one, two):
  old_min, old_max = -5, 5
  new_min, new_max = 0, 10

  old_range = old_max - old_min
  new_range = new_max - new_min

  converted_one = ((one - old_min) * new_range / old_range) + new_min
  converted_two = ((two - old_min) * new_range / old_range) + new_min
  print(converted_one)
  print(converted_two)
  return converted_one - converted_two


if __name__ == '__main__':

  print(converter_multiplication(-1.615685, -0.999412))
  print("_____")
  print(converter_multiplication(-3.464635, -3.005556))
  print("_____")
  print(converter_two(4.024445,  -4.045636))
  print("_____")
  print(converter_multiplication(4.024445,  -4.045636))
  # print(converter(-0.01))
  # print(converter(0.01))
  # print(converter(0.01)-converter(-0.01))
  # print(converter(-0.15)-converter(-0.09))
