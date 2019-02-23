
from scipy.stats import norm
import math
from scipy.stats import t

# part a
x = (2 * math.sqrt(5)) / 3
p = 2*(1 - norm.cdf(x))

# part b
ci_l = 12 - (1.96 * 3 / math.sqrt(5))
ci_h = 12 + (1.96 * 3 / math.sqrt(5))

# part c
tp = t.ppf(.95,4)

print(x)
print(p)

print(ci_l)
print(ci_h)

print(tp)
