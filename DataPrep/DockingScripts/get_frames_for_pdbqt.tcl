# first run this script in VMD to extract snapshots of protein + heme
# then run set.element.tcl script to add a column with atom names

mol new ../WT_Docking/cyp2d6_si.pdb
mol new last_50ns.js waitfor all
set ref [atomselect 0 "protein and name CA" frame 0]

set numframes [molinfo top get numframes]

for {set f 0} {$f < $numframes} {incr f} {

	puts $f

	set comp [atomselect top "protein and name CA" frame $f]
	set all [atomselect top "protein or resname HEME" frame $f]

	$all move [measure fit $comp $ref]

	$all writepdb ./pdb.receptor/$f.frame.pdb

	$all delete	
	$comp delete

}

exit
