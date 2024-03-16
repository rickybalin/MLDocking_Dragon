# Docking_Scripts

Contributors: Javier Baylon, Andres Arango, Archit Vasan
## Required softwares

First, download autodock vina, and adtools. 

vina: http://vina.scripps.edu
adtools: http://autodock.scripps.edu/resources/adt

## Steps 

0. - make pdbqt file of ligand, starting from the pdb file (this can be done with adtwools).  
Run make_ligand_pdbqt. This puts pdbqt file into pdbqt.ligand

1. - get snapshots of the protein from simulations trajectory, using get_frames_for_pdbqt.tcl, then fix using 'set.element.tcl'.

2. - make receptor pdbqts using 'make_receptor_pdbqt'.  this step takes a while, you can start creating a small number of frames just to get an idea of how it works.

3. - make conf files for autodock vina, using 'make_confs' scripts. you will need to change the name of ligand to match whatever your ligand pdbqt is called.

4. - run docking using autodick vina - the main line is :

	path_to_autodock_vina/vina --config conf.files/$JOBNAME.conf --cpu $NCPUS
	Just run run_docking.sh and this should be taken care of

5. - get ligand binding poses from the generated log files using 'get_pdb_from_dlg', this will generated pdbs that can be visualized in vmd!

6. - merge protein and ligand together using merging_poses.tcl

7. - load merged files to vmd using vmdload.tcl

8. - optionally, after loading in vmd, create a js file with all docked poses... this should decrease the time it takes to load.

9. - save list of energies using get_energies.sh 
