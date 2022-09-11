import matplotlib.pyplot as plt
import numpy as np

# Linear solver
def my_linfit(x, y):
    n = len(x)
    a =  (n*np.sum(np.multiply(x,y)) - np.sum(x)*np.sum(y)) / (n*np.sum(np.multiply(x,x)) - np.sum(x) ** 2)
    b = np.average(y)- a*np.average(x)

    return a, b

def main():

    x = []
    y = []
    plt.axis([0, 10, 0, 6])

    plt.waitforbuttonpress()
    print("Middle click to end putting points")
    pts = []
    pts = np.asarray(plt.ginput(-1, timeout=-1))
    print("Points: ")
    for i in range(len(pts) - 1):
        x.append(pts[i][0])
        y.append(pts[i][1])
        print(pts[i])

    plt.plot(x,y,'kx')
    a,b = my_linfit(x,y)
    xp = np.arange(0,10,0.1)
    plt.plot(xp, a*xp+b, 'r-')
    plt.show()
    

if __name__ == "__main__":
    main()
