#!/bin/bash
#SBATCH --job-name=sail
##SBATCH --output=toptt300-N1-P2-Q1e10-G101-m5-i0
#SBATCH --output=SiNt300-N3-P3-Q1e10-G101
#SBATCH -N 1
#SBATCH --ntasks-per-node=20
#SBATCH --partition=normal
#SBATCH --time=30:00:00
#SBATCH --error=out.%j.%N.err

np=20
topt=1
Job=1
inverse=1
material='SiN'  #SiN,silica,silicon,gold
init_type='./DATA/acc_topt1_inv1_N1_ps_sym00_Nx300_Ny300_Pmicron3.0_mload0.1_Nf20_Qf50.0_angle0.0_nG101_bproj0.0_mload0.1_mp1.5_tmin100.0_SiN_tnm300.0_dof394.txt'
Qref=1e10
Mx=300
nG=101
Period=3e-6
tmin=100e-9
thickness=300e-9
Nlayer=1
mload=0.1
mpower=1

bproj=0
polarization='ps'
angle=0

dx=1e-4
ind=10

mpirun -np $np python run_propulsion.py -init_type $init_type -nG $nG -Qref $Qref -bproj $bproj -Period $Period -thickness $thickness -Nlayer $Nlayer -mload $mload -mpower $mpower -polarization $polarization -angle $angle -Job $Job -dx $dx -ind $ind -inverse $inverse -Mx $Mx -material $material -topt $topt -tmin $tmin
