#creating movie with all poses and  entire trajectory

#Po-Chao Load each pdb into one molecule

set what "DRUG"
set workdir [exec pwd]	

cd $workdir/merged.files/all_from_docking


#Number of frames and mol ID tracker
set num_frames 0
set mol_track 0


for {set i 0} {$i <= 499} {incr i} { 

#loading receptor
	mol new $workdir/pdb.receptor/$i.frame.pdb type pdb waitfor all 
	

	mol rep licorice
	mol color colorid 1
	mol selection "not protein"
	mol addrep top

	mol rep newcartoon
	mol material AOChalky
	mol color colorid 15
	mol selection "protein"
	mol addrep top
	

	#keeps track of mol ID
	set mol_track [expr {$mol_track+$num_frames}]
	
	#this is takes into account loading two molecules per loop
	set mol1 [expr {$i*2+$mol_track}]
	puts "mol1 is $mol1"
#loading the ligand 

	set g [expr {$i}]
	
	mol new $workdir/pdb.ligand/$g.ligand.pdb waitfor all
	
	#two molecules plus 1
	set mol2 [expr {$i*2+1+$mol_track}]
	puts "mol2 is $mol2"

	set num_frames [molinfo $mol2 get numframes]
	puts "#############################    Ligand $g has $num_frames frames, mol2 is $mol2 , mol1 is $mol1    ######################"

	for {set framex 1} {$framex <= $num_frames} {incr framex} {
		
		set sel1 [atomselect $mol1 "all"]
		set sel2 [atomselect $mol2 "all" frame $framex]
		
		package require topotools
		
		set merge [::TopoTools::selections2mol "$sel1 $sel2"]
		
		animate write pdb $i.$framex.merged.Test.pdb 
		puts "#############################   works Receptor $i and ligand $g and pose $framex   "
	}

	mol delete all
}


	
cd ..

exit
