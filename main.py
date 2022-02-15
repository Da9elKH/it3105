# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np

def print_board():
    fig = plt.figure(figsize=[7, 7])
    fig.patch.set_facecolor((0.827, 0.827, 0.827))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    return fig, ax

def print_cart(fig, ax, position=(0,0)):

    #position = (0, 0)
    #length = 2
    #height = 1

    # Horizontal lines
    ax.plot([0, 2], [0, 0], 'k')
    ax.plot([0, 2], [1, 1], 'k')

    # Vertical lines
    ax.plot([0, 0], [0, 1], 'k')
    ax.plot([2, 2], [0, 1], 'k')

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    fig, ax = print_board()
    print_cart(fig, ax)

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
