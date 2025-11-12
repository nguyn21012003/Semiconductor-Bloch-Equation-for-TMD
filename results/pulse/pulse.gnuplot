set terminal qt size 900,900

set datafile separator ","

set xrange [-10:10]

plot "pulse.dat" using 1:2 with lines lw 5 lc rgb "blue" notitle,\
