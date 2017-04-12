/**************

MPDFEstimator.c

Allocate, free, address for access and so on of data
needed to compute MPDFs.

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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* Meschach needed */
#include "matrix.h"
#include "MPDFEstimator.h"
#include "linalg.h"

double cd;
int mkeinit=0;


double unit_sphere_volume( int d )
{
   double V0=1.;
   double V1=2.;
   double Vn=0.;
   double Vnm1,Vnm2;
   int i;

   if (d==0)
     return V0;

   if (d==1)
     return V1;

   Vnm2=V0;
   Vnm1=V1;
   for(i=2;i<=d;i++){
     Vn=2.*M_PI*Vnm2/((double)i);
     Vnm2=Vnm1;
     Vnm1=Vn;
   }
   return Vn;
}

static void MPDFInit(void)
{  
  mkeinit=1;
}

/* Create a kernel-based PDF estimator using the Epanechnikov kernel */
MPDFEstimatorPtr NewMPDFEstimator( int npoints , int length )
{
  MPDFEstimatorPtr mpdf;

  if(!mkeinit)
    MPDFInit();
  mpdf=(MPDFEstimatorPtr)malloc(sizeof(MPDFEstimator));
  assert(mpdf);
  mpdf->current=0;
  mpdf->length=length;
  mpdf->npoints=npoints;
  mpdf->X=(double*)calloc(npoints*length,sizeof(double));
  mpdf->P=NULL;
  mpdf->PCOK=0;
  assert(mpdf->X);
  return mpdf;
}

/** Free the memory **/
void KillMPDFEstimator( MPDFEstimatorPtr mpdf )
{
  free((void*)mpdf->X);
  if (mpdf->P)
    free((void*)mpdf->P);
  free((void*)mpdf);
}

double *MPDFPosition( MPDFEstimatorPtr mpdf , int ipos )
{
  return (double*)(((char*)mpdf->X)+sizeof(double)*ipos*mpdf->length);
}

double *MPDFPCPosition( MPDFEstimatorPtr mpdf , int ipos )
{
  return (double*)(((char*)mpdf->P)+sizeof(double)*ipos*mpdf->length);
}

/* Add a sample to the set of samples */
void MPDFUpdate( MPDFEstimatorPtr mpdf , double *Xi )
{
  double *temp;
  int npoints;

  if (mpdf->current==mpdf->npoints){
    /* Reallocate a new chunk and update the data structure */
    npoints=mpdf->npoints*2;
    temp=(double*)calloc(npoints*mpdf->length,sizeof(double));
    assert(temp);
    memcpy(temp,mpdf->X,mpdf->length*mpdf->npoints*sizeof(double));
    free((void*)mpdf->X);
    mpdf->X=temp;
    mpdf->npoints=npoints;
  }
  memcpy(MPDFPosition(mpdf,mpdf->current),Xi,mpdf->length*sizeof(double));
  mpdf->current++;
}

double MPDFgetoptimumbandwidth( int dim , int samples)
{
  double Ak=0.0;
  double cd=0.0;
  double hopt;

  if (dim==2)
    Ak=2.40;
  else if (dim==3)
    Ak=2.49;
  else{
    cd=unit_sphere_volume(dim);
    Ak=pow((8.*(dim+4.)*pow(2*sqrt(M_PI),(double)dim)/cd),1./(dim+4.));
  }
  hopt=(Ak/pow(samples,1./(dim+4.)));
  return hopt;
}

#define DEBUG 1
#undef DEBUG

#define TEST_COV_MAT 1
#undef TEST_COV_MAT

/* Get principal components */
void MPDF_ComputePCData( MPDFEstimatorPtr mpdf , MAT *E )
{
  int i;
  double *d;
  VEC *vx;
  VEC *pc;

  /* If this is not set, space for PCs must be allocated */
  if ((!mpdf->PCOK)||(mpdf->P==NULL)){
    mpdf->P=calloc(mpdf->length*mpdf->current,sizeof(double));
    assert(mpdf->P);
  }
  /* In any caase, computed PCs */
  vx=v_get(mpdf->length);
  assert(vx);
  pc=v_get(mpdf->length);
  assert(pc);
  for(i=0;i<mpdf->current;i++){
    d=MPDFPosition(mpdf,i);
    DPx2pc(E,d,vx,pc);
#ifdef DEBUG
    int j;
    double varp=0.0;
    double varx=0.0;
    for(j=0;j<mpdf->length;j++){
      printf(" %f",pc->ve[j]);
      varp+=pc->ve[j]*pc->ve[j];
      varx+=vx->ve[j]*vx->ve[j];
    }
    printf(" = %f %f\n",varp,varx);
#endif
    d=MPDFPCPosition(mpdf,i);
    memcpy(d,pc->ve,sizeof(double)*pc->dim);
  }
  mpdf->PCOK=1;

#ifdef TEST_COV_MAT
  int N=mpdf->current;
  int II;
  int JJ;
  int nn;
  double accum;
  double *pcx,*pcy;

  for(II=0;II<mpdf->length;II++){
    for(JJ=0;JJ<mpdf->length;JJ++){
      accum=0.0;
      for(nn=0;nn<N;nn++){
	pcx=MPDFPCPosition(mpdf,nn)+II;
	pcy=MPDFPCPosition(mpdf,nn)+JJ;
	accum+=(*pcx)*(*pcy);
      }
      printf("II %d JJ %d -> Cov= %f \n",II,JJ,accum/N);
    }
  }
#endif


  V_FREE(vx);
  V_FREE(pc);
}


/* Get X from principal components */
void MPDF_ComputeXData( MPDFEstimatorPtr mpdf , MAT *E )
{
  int i,j,dim;
  double *p;
  VEC *vx;
  VEC *pc;

  dim=mpdf->length;

  vx=v_get(mpdf->length);
  assert(vx);
  pc=v_get(mpdf->length);
  assert(pc);

  for(i=0;i<mpdf->current;i++){
    p=MPDFPCPosition(mpdf,i);
    for(j=0;j<dim;j++)
      pc->ve[j]=p[j];
    pc2x(E,pc,vx);
    p=MPDFPosition(mpdf,i);
    memcpy(p,vx->ve,sizeof(double)*dim);
  }

#ifdef TEST_COV_MAT
  int N=mpdf->current;
  int II;
  int JJ;
  int nn;
  double accum;
  double *pcx,*pcy;

  for(II=0;II<mpdf->length;II++){
    for(JJ=0;JJ<mpdf->length;JJ++){
      accum=0.0;
      for(nn=0;nn<N;nn++){
	pcx=MPDFPCPosition(mpdf,nn)+II;
	pcy=MPDFPCPosition(mpdf,nn)+JJ;
	accum+=(*pcx)*(*pcy);
      }
      printf("II %d JJ %d -> Cov= %f \n",II,JJ,accum/N);
    }
  }
#endif


  V_FREE(vx);
  V_FREE(pc);
}

