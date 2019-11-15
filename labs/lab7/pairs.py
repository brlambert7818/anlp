import itertools
import numpy as np
import matplotlib.pyplot as plt

girls = ['Colleen', 'Pia']
boys = ['Brian', 'Jakub']

pairs = list(itertools.product(girls, boys))
rand=np.zeros(10000)
for i in range(len(rand)):
    rand[i] = np.random.randint(0, 4)
    # pairs[rand]

plt.hist(rand, bins=3)
plt.show()

