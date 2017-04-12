/**************


computePDF.h

Definition of data structures and function prototypes of the parts of the code
used to actually compute the PDFs for several dimensionalities.


This source file is part of the set of programs used to derive environmental
models' performance indices using multidimensional kernel-based
probability density functions, as described by:

Multi-objective environmental model evaluation by means of multidimensional 
kernel density estimators: efficient and multi-core implementations, by 
Unai Lopez-Novoa, Jon Sáenz, Alexander Mendiburu, Jose Miguel-Alonso, 
Iñigo Errasti, Ganix Esnaola, Agustín Ezcurra, Gabriel Ibarra-Berastegi, 2014.


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

***************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <unistd.h> //Para los recursos

/* Use library meschach */
#include "matrix.h"
#include "matrix2.h"
#include "MPDFEstimator.h"
#include "PDF.h"

//OpenMP Query API
#ifdef _OPENMP
	#include <omp.h>
#endif

#ifndef __computePDF__
#define __computePDF__ 1

//Calculate the PDF of a defined 2D space (box) for a given sample. 
//This function is called either by computePDF2D or computePDFND
void compute2DBox(MPDFEstimatorPtr mpdf, PDFPtr pdf, VEC * XX, VEC * PCdot, 
		  double * PC, double * lower, int * tot_ev_per_dim, double * gridpoint, 
		  size_t * dif_pos, double * x0,double * dx, int dim, double h2, double cd, 
		  MAT * eigenvectors, int * densCounter, double * densValues, int * densPosition);

//Compute the PDF of a one-dimensional grid space
void computePDF1D(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , 
		double detSm1 , double *x0,	double *x1, double *dx, 
		double *bounds, MAT *eigenvectors );

//Compute the PDF of a 2D grid space
void computePDF2D(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , 
		  double detSm1 , double *x0, double *x1, double *dx, 
		  double *bounds, MAT *eigenvectors );

//Compute the PDF of grid spaces of dimension 3 or higher
void computePDFND(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , 
		  double detSm1, double *x0, double *x1, double *dx, 
		  double *bounds, MAT *eigenvectors, int dim);

#endif
