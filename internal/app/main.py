import internal.services.analytics.analytics as analytics
import internal.services.learning.learning as learning
import internal.services.simulation.simulation as simulation


if __name__ == '__main__':
  analytics.start_analytics()
  learning.start_learning()
  simulation.start_simulation()
