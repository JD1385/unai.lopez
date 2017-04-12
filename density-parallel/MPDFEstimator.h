/**************

MPSFEstimator.h

Function prototypes

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

#ifndef __MPDFEstimator__
#define __MPDFEstimator__ 1

/* Meschach needed */
#include "matrix.h"

typedef struct {
  /* The data vectors */
  double *X;
  /* The data vectors (as PCs) */
  double *P;
  /* A flag telling the user that the PC data is not initialized */
  int PCOK;
  /* The length of each vector */
  int length;
  /* The size of a chunk, but the actual size is CURRENT!!!! */ 
  int npoints;
  /* Counter used to allow fast reallocation and actual number of elements */
  int current;
} MPDFEstimator, *MPDFEstimatorPtr;

/* Create a kernel-based PDF estimator using the Epanechnikov kernel */
MPDFEstimatorPtr NewMPDFEstimator( int npoints , int length );

/** Free the memory **/
void KillMPDFEstimator( MPDFEstimatorPtr pdf );

/* The position of a point in the mpdf record */
double *MPDFPosition( MPDFEstimatorPtr mpdf , int ipos );
double *MPDFPCPosition( MPDFEstimatorPtr mpdf , int ipos );

/* Add a sample to the set of samples */
void MPDFUpdate( MPDFEstimatorPtr pdf , double *Xi );

/* Get principal components from X */
void MPDF_ComputePCData( MPDFEstimatorPtr mpdf , MAT *E );
/* Get X from principal components */
void MPDF_ComputeXData( MPDFEstimatorPtr mpdf , MAT *E );

double MPDFgetoptimumbandwidth( int dim , int samples);

/* 
   This is used to copy the structure and allocate space,
   but data contents (X,P) are not copied because this
   is used inside bootstrap and data will be perturbed */
MPDFEstimatorPtr copyMPDFs(MPDFEstimatorPtr mpdf);

double unit_sphere_volume( int d );

#endif
