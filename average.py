

import sys

filename = sys.argv[1]
args = sys.argv[2:]

first = True

time = []
avg = []

for fn in args:
    with open(fn, 'r') as fd:
        i = 0
        for line in fd:
            splt = line.split(' ')
            if first:
                time.append( splt[0] )
                avg.append( float(splt[1]) )
            else:
                avg[i] += float(splt[1])
                i += 1
    first = False

avg = [ x/len(args) for x in avg ]

with open(filename, 'w') as f:
    for i in range(len(avg)):
        f.write( f"{time[i]} {avg[i]}\n")


