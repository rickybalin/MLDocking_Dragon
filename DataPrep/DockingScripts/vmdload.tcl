#############################################################################
# Script to load all merged files
#
cd merged.files
mol new 0.1.merged.Test.pdb

for {set j 2} {$j <= 20} {incr j} {
	if {[file exists 0.$j.merged.Test.pdb]==1} {
		mol addfile 0.$j.merged.Test.pdb
	} else {
		continue
	}
	
}

for {set i 1} {$i <= 499} {incr i} {
	for {set j 1} {$j <= 20} {incr j} {
		if {[file exists $i.$j.merged.Test.pdb]==1} {
			mol addfile $i.$j.merged.Test.pdb
		} else {
			continue
		}
		
	}
}
cd ..
