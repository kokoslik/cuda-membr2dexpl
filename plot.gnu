#!/usr/bin/gnuplot -p
#set term wxt
unset key
#set dgrid3d 101,101
set zrange [-1:1]
do for [i=0:1000:5]{
splot  'datacpu.dat' index(i) matrix w l
}
