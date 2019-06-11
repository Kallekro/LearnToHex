#!/usr/bin/python3

import sys
import math
#### For plotting data
import matplotlib.pyplot as plt

def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print ("usage : title xlabel ylabel file1 file2*")
        exit(-1)

    title = args[0]
    xlabel = args[1]
    ylabel = args[2]

    files = args[3:]

    # list holding data to be plotted
    data = []

    # open, read and save data from files
    for fname in files:
        x, y = [], []
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                dp = line.split(' ')
                x.append( float(dp[0]) )
                y.append( float(dp[1]) )
            fdata =(x, y)
            data.append( fdata )

    fig = plt.figure()
    for x, y in data:
        plt.plot(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()
    return 0


if __name__ == "__main__":
    main()