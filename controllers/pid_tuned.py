from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self, kp=0.1732, ki=0.15, kd=-0.003):
    self.p = kp
    self.i = ki
    self.d = kd
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff
