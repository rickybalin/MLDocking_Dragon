#load pdb and add the element to each atom name
#this step is required to generate the receptor pdbqt with autodick vina!!!

set pdbs [glob pdb.receptor/*.pdb]
set pdbs [lsort -dictionary $pdbs]

foreach pdbi $pdbs {

	#set names [split [lindex [split $pdbi /] 2] .]	
	set names [file rootname [file tail $pdbi]]
	
	#set str0 [lindex $names 0]
	#set str1 [lindex $names 1]
	#set str2 [lindex $names 2]

	puts $names

	if {1} {

	mol new $pdbi

	set all [atomselect top all]

	set indexes [$all get index]

	foreach indexx $indexes {

		set atom [atomselect top "index $indexx"]
		
		set elementx [lindex [split [$atom get name] {}] 0]

		if {[string compare [$atom get name] "F1"] == 0} {

			set elementx F
			puts $indexx

		} 

		if {[string compare [$atom get name] "FE"] == 0} {

			set elementx Fe
			puts $indexx

		} 

		$atom set element $elementx
		$atom delete


	}

		#$all writepdb ../pdb.receptor/$str0.$str1.pdb
		$all writepdb pdb.receptor/$names.pdb
		mol delete all
	}
}

exit
