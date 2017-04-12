These source files are part of the set of programs used to derive environmental
models' performance indices using multidimensional kernel-based
probability density functions, as described by:

Multi-objective environmental model evaluation by means of multidimensional 
kernel density estimators: efficient and multi-core implementations, by 
Unai Lopez-Novoa, Jon Sáenz, Alexander Mendiburu, Jose Miguel-Alonso, 
Iñigo Errasti, Ganix Esnaola, Agustín Ezcurra, Gabriel Ibarra-Berastegi, 2014.

--

PROGRAM DESCRIPTIONS
------------------------------------------------------------------------

mpdfestimator: Performs an estimation of the multidimensional probability
  density function in a multidimensional space using a kernel approach.
  The multidimensional Epanechnikov kernel is used.

mpdfestimator-bootstrap: Performs a bootstrap analysis of the behavior of 
  the evaluated PDF around different values of the bandwidth used to 
  estimate the probability density function.

mpdf_score: Computes the PDF-score from two N-dimensional PDFs stored as 
  netCDF files for the same domain by the first program.

COMPILATION
------------------------------------------------------------------------

This program has been tested and developed in a Linux environment.
Instructions in this Readme file are described as if the target environment
was Linux too.

Requirements:
	-C compiler (e.g. gcc or icc)
	-Meschach math library. Available at:
	 http://homepage.math.uiowa.edu/~dstewart/meschach (Visited Feb,2014)
	-Netcdf data handling libray. Available at:
	 http://www.unidata.ucar.edu/software/netcdf (Visited Feb, 2014)
	
Compilation steps:
1) Modify Makefile with the following informations	
	
	1.1) Paths to Meschach and NetCDF libraries. E.g:
		
	#Meschach library
	MESCH_INC=/usr/local/include
	MESCH_LIB=/usr/local/lib

	#NetCDF library
	NETCDF_INC=/usr/local/include
	NETCDF_LIB=/usr/local/lib	

	1.2) C compiler to be used. By default gcc is used:
	
	#### C COMPILER ####
	CC=gcc

	1.3) Decide wether to compile single-threaded or multi-threaded (OpenMP)
	code. Uncomment one of the following lines depending on your choice.
		
	#Flag set for multi-threaded implementation (Default)
	DEBUG=-O2 -ftree-vectorize -msse2 -fopenmp
	#Flag set for serial implementation
	#DEBUG=-O2 -ftree-vectorize -msse2	

	IMPORTANT: If the multi-threaded implementation is selected, ensure
	to set the proper OpenMP flag. In the current Makefile "-fopenmp" is
	used, which is the flag to activate OpenMP compilation with gcc.
	E.g. if using intel compiler (icc) the flag shoudl be "-openmp"
	instead of "-fopenmp".
	
2) Type "make" in the source directory. Programs will be created in a
   folder called "bin".
   
3) You can test the progams by tipying "make check" or "make bscheck"   
   to run a test with mpdfestimator or mpdfestimator-bootstrap, respectively
   
USAGE
------------------------------------------------------------------------

mpdfestimator:
	Usage:

	./mpdfestimator [-h bandwidth] [-o ncfile] -x xmin:xmax:deltax ifile
   
	Arguments:
       - h bandwidth: Bandwidth to be used in the kernel. If not 
		 provided, it is calculated as described in:
		  Silverman, Bernard W. Density estimation for statistics and 
		  data analysis. Vol. 26. CRC press, 1986.
		 		  
	   - o ncfile: Output netCDF file. Default: MPDF.nc		 		  
		 		  
	   - x xmin:xmax:deltax: Interval to compute the density. Each of 
	     the values should be a n-dimensional vector. For instance: 
	     -1/1.5:1/2.5:0.1/.5 can be used for a 2D distribution on 
	     [-1,1]x[1.5,2.5] stepping by 0.1 (1st dimension) and by 0.5 (2nd one).	 	   
   
	   ifile: Data as ASCII rows from ifile (default stdin). Each row should 
	     be a vector or comment and all the rows must be the same size
		 For example:
				0.613 1.550 4.381
				3.915 4.514 4.389
				6.386 7.166 5.636
				2.032 4.576 7.796
   
	Usage Example:
	./mpdfestimator -h 0.6 -x -20/-15/-15:25/30/30:0.2/0.2/0.2 sample-data/sample-data
	
	These programs are provided with a sample data file, contained in the sample-data
	folder. This file has been generated using R and MASS library, with the following
	script:	
		library('MASS')
		mu=c(1,5,10)
		S1=matrix(c(12.6,2.5,2.5,2.5,16.3,2.5,2.5,2.5,16.3),nrow=3)
		data=mvrnorm(3000,mu,S1)
		write.table(data,"sample-data",col.names=FALSE,row.names=FALSE)

mpdfestimator-bootstrap:
	Usage:
	
	./mpdfestimator-bootstrap 
	 [-n maxreal] [-h bandwidth] [-H hmin/hmax/deltah] -x xmin:xmax:deltax ifile
	
	Arguments:
       - h bandwidth: See above
       - x xmin:xmax:deltax: See above
       ifile: See above
	
	   -n maxreal: Maximum number of realizations
	  
	   -H hmin/hmax/deltah: Range of bandwidths to test

	Usage Example:
	./mpdfestimator-bootstrap -h 0.6 -H 0.1/1/0.1 \
		-x -20/-15/-15:25/30/30:0.2/0.2/0.2 -n 5 sample-data/sample-data

mpdf_score:
	Usage:
	
	./mpdf_score ifile1 ifile2
	
	where ifile1 and ifile2 are the path to the two NetCDF files that contain
	the PDFs to be compared.

OUTPUT FILE
------------------------------------------------------------------------

The estimation generated by mpdestimator is contained in a binary NetCDF
file. This special format is able to preseve the precision of a standard
binary file, but provides portability across different architectures.

The content of the file can be dumped as plain text with the "ncdump"
program, contained inside the NetCDF library. A simple usage example:
 :~$ ncdump MPDF.nc > output_plain_text

The content of the NetCDF file is organized in variables, being each
an array of values. Our programs writes in the NetCDF file a variable
per dimension, containing the range of gridpoints to be evaluated, named 
X0, X1, X2,... and a variable called PDF where the estimation is dumped.
For example, with a 3-Dimensional estimation:

data:
 X0 = -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
 X1 = -100, -99, -98, -97, -96, -95, -94, -93, -92, -91, -90;
 X2 = -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10;

 PDF = 0, 1.51004913619138e-08, 1.04130739099683e-07, ...

COPYRIGHT
------------------------------------------------------------------------

Copyright (c) 2014, Unai Lopez-Novoa, Jon Saenz, Alexander Mendiburu 
and Jose Miguel-Alonso  (from Universidad del Pais Vasco/Euskal 
		    Herriko Unibertsitatea)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universidad del Pais Vasco/Euskal 
      Herriko Unibertsitatea  nor the names of its contributors may be 
      used to endorse or promote products derived from this software 
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
