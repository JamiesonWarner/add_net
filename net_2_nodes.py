import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
a2 = fig.add_subplot(2,1,2)
state = np.array([1, 1])
syn0 = 2 * np.random.random((2, 2)) - 1
state1s = []
state2s = []

def reset_simulation():
    global state
    global syn0
    global state1s
    global state2s

    # starting position
    state = np.array([1, 1])

    # seed random numbers to make calculation
    # deterministic (just a good practice)
    # np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = 2 * np.random.random((2, 2)) - 1

    # called each frame of animation
    state1s = []
    state2s = []
def update(i):
    # forward propagation
    global state
    l0 = state
    l1 = np.dot(l0,syn0)
    state = l1 + state
    state1s.append(state[0])
    state2s.append(state[1])
    ax.clear()
    ax.plot(range(len(state1s)), state1s)
    a2.clear()
    a2.plot(range(len(state2s)), state2s)

def onclick(event):
    reset_simulation()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

a = anim.FuncAnimation(fig, update, frames=500, repeat=False)
plt.show()

