#!/bin/tcsh

## start and end are starting and ending config files to run
set start = $1
set end = $2

#num of reps
cd conf.files/
foreach q (`seq $start 1 $end`)
	echo $q	
	/Scr/akvasan2/autodock_vina_1_1_2_linux_x86/bin/vina --config dock.$q.conf --cpu 8
	
end


































