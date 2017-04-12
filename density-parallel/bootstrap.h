/**************

bootstrap.h

Header file describing the data structure used for bootstrap and imported by
source files using it


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

#ifndef __bootstrap__
#define __bootstrap__ 1

#include "matrix.h"

#include "PDF.h"
#include "MPDFEstimator.h"

typedef struct {
  double *max;
  double *adave;
  double *sqer;
  int maxreal;
} * BootstrapDiagPtr, BootstrapDiag;

BootstrapDiagPtr NewBootstrapDiagnostics( int maxrealizations );
void KillBootstrapDiagnostics( BootstrapDiagPtr bsptr );

/*
  From a reference fixed mpdf (fmpdf), generate a randomly
  modified rmpdf
*/
void randomMPDF( MPDFEstimatorPtr fmpdf , MPDFEstimatorPtr rmpdf , double h ,
		 MAT *sqrtevals, MAT *lm1E , double *sfac );

void getBootstrapDiagnostics(PDFPtr pdf, PDFPtr pdf2, BootstrapDiagPtr bsdiag, 
			     int irealization );

void dumpBootstrapDiag(double h, double href, BootstrapDiagPtr bsdiag , 
		       double dV);
void init_BS_factors(double *sfacs,double h,double varEPS, int dim);
#endif
