
set terminal pdfcairo size 16cm,16cm enhanced
set output "MoS2.pdf"
set bmargin at screen 0.07
set lmargin at screen 0.07
set datafile separator ","
pi = 3.14159265358979323846
set xtics ("-M" -2 * pi / 3.19, "-K" -4 * pi / (3 * 3.19), "Γ" 0, "K" 4 * pi / (3 * 3.19), "M" 2 * pi / 3.19)
set ytics 1

set key top font "CMU Serif,20"
set xtics font "CMU Serif,20" offset 0,-0.5
set ytics font "CMU Serif,20" 
set ylabel "Energy(eV)" font "CMU Serif,20"
set xlabel "" font "CMU Serif,20"
set style line 81 lc rgb "#808080" lw 2.5
set grid xtics ls 81
set key outside top center 
set yrange [*:5.0]

dir = "results/Fri-11-07/TNN"
plot "bandstructure.dat" using 1:2 w l lw 4 lc "black" notitle,    "bandstructure.dat" using 1:3 w l lw 4 lc "black" notitle,    "bandstructure.dat" using 1:4 w l lw 4 lc "black" notitle

        