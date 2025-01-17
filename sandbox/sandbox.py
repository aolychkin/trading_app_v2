
def converter(old):
  old_min, old_max = -1, 1
  new_min, new_max = 0, 1

  old_range = old_max - old_min
  new_range = new_max - new_min
  converted = ((old - old_min) * new_range / old_range) + new_min
  return converted


if __name__ == '__main__':

  print(converter(-0.01))
  print(converter(0.01))
  print(converter(0.01)-converter(-0.01))
  print(converter(-0.02)-converter(-0.01))
