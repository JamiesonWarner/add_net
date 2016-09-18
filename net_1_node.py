import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# starting position
state = np.array([1])

# seed random numbers to make calculation
# deterministic (just a good practice)
# np.random.seed(1)

# initialize weights randomly with mean 0
while True:
    syn0 = 2 * np.random.random(1) - 1
    if syn0 < 0:
        break

# called each frame of animation
states = []
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def update(i):
    # forward propagation
    global state
    l0 = state
    l1 = np.dot(l0,syn0)
    print (state, l0, syn0, l1)
    state = l1 + state
    states.append(state)
    ax.clear()
    ax.plot(range(len(states)), states)

a = anim.FuncAnimation(fig, update, frames=500, repeat=False)
plt.show()
