#!/usr/bin/python3

import sys
import math
#### For plotting data
import matplotlib.pyplot as plt

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print ("usage : title file1 file2*")
        exit(-1)

    title = args[0]
    files = args[1:]

    # list holding data to be plotted
    data = []

    # open, read and save data from files
    for fname in files:
        x, y = [], []
        with open(fname, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip('\n')


            for line in lines[1:]:
                dp = line.split(' ')
                x.append( float(dp[0]) )
                y.append( float(dp[1]) )
            fdata =(header, x, y)
            data.append( fdata )

    header = data[0][0]
    splt   = header.split(',')
    xlabel = splt[0]
    ylabel = splt[1]

    fig = plt.figure()
    for header, x, y in data:
        plt.plot(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()
    return 0


if __name__ == "__main__":
    main()