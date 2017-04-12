/**************

bootstrap.c

Parts of the code dealing with the allocation, creation and handling of the
perturbed PDF estimates for the bootstrap analysis of the optimum bandwidth.


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
#include <math.h>
#include <assert.h>

#include "bootstrap.h"
#include "MPDFEstimator.h"
#include "genepsilon.h"
#include "PDF.h"

/*
  Create a new Bootstrap Diagnostics data structure used to 
  store and finally print the diagnostics after all the
  bootstrap estimates have been produced.
*/
BootstrapDiagPtr NewBootstrapDiagnostics( int maxrealizations ){
  BootstrapDiagPtr diag;

  diag=malloc(sizeof(BootstrapDiag));
  assert(diag);
  /* Maximum number of realizations */
  diag->maxreal=maxrealizations;
  /* This is for the maximum */
  diag->max=(double*)calloc(maxrealizations,sizeof(double));
  assert(diag->max);
  /* To be used for the mean */
  diag->adave=(double*)calloc(maxrealizations,sizeof(double));
  assert(diag->adave);
  /* This is for mean squared error */
  diag->sqer=(double*)calloc(maxrealizations,sizeof(double));
  assert(diag->sqer);
  return diag;
}

/* No need to use it anymore, free memory */
void KillBootstrapDiagnostics( BootstrapDiagPtr bsptr )
{
  free((void*)bsptr->max);
  free((void*)bsptr->adave);
  free((void*)bsptr->sqer);
  free((void*)bsptr);
}

/* Produce a random integer for the sampling */
static int get_rand_int(int min,int max)
{
  return (int)rint(min+(max-min)*((double)random())/(double)RAND_MAX);
}

/* 
Create a random PDF into rmpdf bootstraping from the
reference fmpdf using the bandwidth and eigenvalues/E projection matrix
used in the reference one.
*/
void randomMPDF( MPDFEstimatorPtr fmpdf , MPDFEstimatorPtr rmpdf ,
		 double h , MAT *sqrtevals , MAT *E , double *sfac )
{
  int samples=fmpdf->current;
  int dim=fmpdf->length;
  int i,j,ir;
  int samplesm1=samples-1;
  double *origin;
  double *target;
  double eps;

  // Otherwise, PCs are not computed
  rmpdf->current=fmpdf->current;
  rmpdf->length=fmpdf->length;
  /* The first time, this is not initialized */
  if (rmpdf->P==NULL){
    rmpdf->P=(double*)calloc(rmpdf->current*rmpdf->length,sizeof(double));
    assert(rmpdf->P);
  }
  /* 
     Create random values using sphered PCs (so that univariate EPS is 
     correct because space is spherized) 
  */
  for(i=0;i<samples;i++){
    ir=get_rand_int(0,samplesm1);
    origin=MPDFPCPosition(fmpdf,ir);
    target=MPDFPCPosition(rmpdf,i);
    // printf("%d %d %d %d :\n",i,0,ir,samples);
    for(j=0;j<dim;j++){
      eps=epsilon();
      target[j]=(origin[j]+h*eps)/sfac[j];
      // printf("%d %g [%f -> %f] \n",j,eps,origin[j],target[j]);
    }
    // printf("\n");
  }
  MPDF_ComputeXData(rmpdf,E);
}

/* 
   Compute bootstrap diagnostics from the individual ones obtained from 
   bootstrap 
*/
void getBootstrapDiagnostics(PDFPtr pdf, PDFPtr pdf2,BootstrapDiagPtr bsdiag,
			     int ire )
{
  size_t i;
  double dif;
  double adif;

  dif=pdf->PDF[0]-pdf2->PDF[0];
  adif=fabs(dif);
  bsdiag->max[ire]=adif;
  bsdiag->adave[ire]=adif;
  bsdiag->sqer[ire]=adif*adif;
  for(i=1;i<pdf->total_size;i++){
    dif=pdf->PDF[i]-pdf2->PDF[i];
    adif=fabs(dif);
    bsdiag->max[ire]=(bsdiag->max[ire]>adif)?bsdiag->max[ire]:adif;
    bsdiag->adave[ire]+=adif;
    bsdiag->sqer[ire]+=adif*adif;
  }
  bsdiag->adave[ire]=bsdiag->adave[ire]/pdf->total_size;
}

/* Comparison function for sort */
static int cfunc(const void* d1, const void *d2)
{
  int retval;
  double D1,D2;

  D1=*((double*)d1);
  D2=*((double*)d2);
  if (D1<D2)
    retval=-1;
  else if (D1==D2)
    retval=0;
  else
    retval=1;
  return retval;
}

// Dump a line to stdout
static void dumpLine(double h,double href,char *label,double *vec,int items,
		     double s)
{
  int i;
  qsort(vec,items,sizeof(double),cfunc);
  printf("%-10s %10.2e %10.2e %10.2e ",label,href,h,vec[0]*s);
  i=(int)floor(0.025*items);
  printf("%10.2e ",vec[i]*s);
  i=(int)floor(0.5*items);
  printf("%10.2e ",vec[i]*s);
  i=(int)floor(0.975*items);
  printf("%10.2e ",vec[i]*s);
  i=items-1;
  printf("%10.2e\n",vec[i]*s);
}

/* Dump the different bootstrap diagnostics */
void dumpBootstrapDiag(double h, double href, 
		       BootstrapDiagPtr bsdiag, double dV)
{
  double scale=1.0;
  printf("%-10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s\n",
	 "DIAGNOSTIC","href","h","Glo Min","P2.5","P50","P97.5","Glo. Max");
  dumpLine(h,href,"MAX",bsdiag->max,bsdiag->maxreal,scale);
  dumpLine(h,href,"MeanAbsErr",bsdiag->adave,bsdiag->maxreal,scale);
  dumpLine(h,href,"SQ_Err",bsdiag->sqer,bsdiag->maxreal,dV);
}

/* Constant factor used in Bootstrap */
void init_BS_factors(double *sfacs,double h,double varEPS, int dim)
{
  int i;
  for(i=0;i<dim;i++){
    sfacs[i]=sqrt(1.+h*h*varEPS);
  }
}
