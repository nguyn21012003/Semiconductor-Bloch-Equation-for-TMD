set terminal qt size 900,900


plot "kgrid.dat" using 1:2 w p pt 7 ps 0.7 lc rgb "blue" notitle,\
     "kgridcutoff.dat" using 1:2 with points pt 7 ps 0.7 lc rgb "#f123" notitle,\
