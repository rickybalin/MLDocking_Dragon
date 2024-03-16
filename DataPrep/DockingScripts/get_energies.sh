#!/bin/bash

# This script is to get docking score per pose

for j in {100..130}
do
	
	grep "REMARK VINA RESULT:" /Scr/akvasan2/Cytochrome_P450/Cyp2D6/CBD_Docking/MUT2_Docking/pdbqt.ligand/$j.ligand.pdbqt >> energy.files/all_energ.affi.log
	## since I am lazy, this will print REMARK VINA RESULT -8.5 where -8.5 is some arbitrary docking score.
	## just read this file into python using np.genfromtxt("") and you should be fine
done
