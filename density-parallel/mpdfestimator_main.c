/**************

mpdfestimator_main.c

Main program for mpdfestimator

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

/* Use library meschach */
#include "matrix.h"
#include "matrix2.h"
#include "netcdf.h"
#include "MPDFEstimator.h"
#include "mpdfncopers.h"

#include "parseargs.h"
#include "linalg.h"
#include "boundaries.h"
#include "copycenter.h"
#include "PDF.h"
#include "computePDF.h"

#define DEBUG
#undef DEBUG 

#define DEBUGDIST 1
#undef DEBUGDIST

#define kbufsize 100000
#define points 5000

#define kgridsize 250
void redimension(double *x0, double *x1, double *dx , int m )
{
  int i;
  for(i=0;i<m;i++){
    x0[i]=x0[i]-(x1[i]-x0[i])/3.;
    x1[i]=x1[i]+(x1[i]-x0[i])/3.;
    dx[i]=(x1[i]-x0[i])/((double)kgridsize);
  }
}


void computePDF( MPDFEstimatorPtr mpdf, PDFPtr pdf,MAT *Sm1 , double h , 
		 double detS ,double *xmin , double *xmax , double *deltax, 
		 int xset, double * bounds, MAT *eigenvec,int dim)
{
  if(!xset)
    redimension(xmin,xmax,deltax,Sm1->m);
        
  if(dim == 1)
	computePDF1D(mpdf, pdf, Sm1, h, detS, xmin, xmax, deltax, bounds,eigenvec);
  else if (dim == 2)
    computePDF2D(mpdf, pdf, Sm1, h, detS, xmin, xmax, deltax, bounds,eigenvec);
  else if (dim >= 3)
    computePDFND(mpdf, pdf, Sm1, h, detS, xmin, xmax, deltax, bounds,eigenvec, dim);	
}

void updatedomain( double *x , double *xmin , double *xmax , int dim )
{
  int i;
  for(i=0;i<dim;i++){
    xmin[i]=(xmin[i]>x[i])?x[i]:xmin[i];
    xmax[i]=(xmax[i]<x[i])?x[i]:xmax[i];
  }
}

int main( int argc , char ** argv )
{
  char *ifile=NULL;
  FILE *istream=(FILE*) stdin;
  int dim,i,j;
  double bandwidth;
  int xset=0;
  char buffer[kbufsize];
  int validline;
  int firstloop=1;
  int samples=0;
  int bwset=0;
  int status,ncid;
  double dSamples;
  double bwtheo;
  char *oncname="MPDF.nc";
  PDFPtr pdf;

  //TIMING ROUTINES
  struct timeval tv_start,tv_end;
  double t_start, t_end, elapsed;
  gettimeofday(&tv_start, NULL); //Get starting time
  t_start = tv_start.tv_sec + (tv_start.tv_usec/1000000.0);

  MAT *S,*Sm1;
#ifdef DEBUG
  MAT *Itest;
#endif
  VEC *Sx;
  double detS;
  VEC *mean;
  VEC *eigenvals;
  MAT *eigenvectors;
  MAT *eigenvectorsT;
  MAT *sqrtevals;
  MAT *temp_bounds;
  MAT *cdata;
  MAT *lm1E;
  MAT *lm1Einv;

  MPDFEstimatorPtr mpdf;

  memset(buffer,'\0',kbufsize);
  if (argc==1){
    usage(argv[0]);
    exit(1);
  }
    
  /* Infer number of dimensions from program args */
  fetchdims(argc,argv,&dim);
             
  /* Create dimension dependant structures */      
  double * xmin = (double *)malloc(sizeof(double) * dim);
  double * xmax = (double *)malloc(sizeof(double) * dim);
  double * cxmin = (double *)malloc(sizeof(double) * dim);
  double * cxmax = (double *)malloc(sizeof(double) * dim);
  double * deltax = (double *)malloc(sizeof(double) * dim);
  double * x = (double *)malloc(sizeof(double) * dim);
        
  /* Process input arguments */
  parseallargs(argc,argv,&ifile,&bandwidth,xmin,xmax,deltax,buffer,&bwset,&xset,&oncname );

  /* Allocate PDF vector and prepare dimensions from xmin, xmax, deltax */
  pdf=NewPDF(xmin,xmax,deltax,dim);
  assert(pdf);

  /* Create netCDF file */
  status=nc_create(oncname,NC_CLOBBER,&ncid);
  check_nc_error(status);

  /* Add input arguments to netCDF so that it is documented */
  status=nc_put_att_text(ncid,NC_GLOBAL,"Comment1",strlen(buffer),buffer);
  check_nc_error(status);

  /*
    Read input dataset
   */
  if (ifile){
    istream=fopen(ifile,"r");
    assert(istream);
  }
  while(fgets(buffer,kbufsize,istream)){
    validline=parsevector(x,&dim,buffer,firstloop);
    if (!validline)
      continue;
    /* Allocate some matrices after reding the first line (dim is known) */
    if (firstloop){     
      S=m_get(dim,dim);
      assert(S);
      Sx=v_get(dim);
      assert(Sx);
      mean=v_get(dim);
      assert(mean);
      mpdf=NewMPDFEstimator(points,dim);
      assert(mpdf!=NULL);
      firstloop=0;
    }

#ifdef DEBUG
    printf("# Dim:%d",dim);
    for(i=0;i<dim;i++)
      printf(" %g",x[i]);
    printf("\n");
#endif

    /* Construct the sample covariance matrix step by step, as file is read */
    for(i=0;i<dim;i++){
      Sx->ve[i]=Sx->ve[i]+x[i];
      for(j=0;j<dim;j++)
        S->me[i][j]=S->me[i][j]+x[i]*x[j];
    }
#ifdef DEBUG
    m_output(S);
    v_output(Sx);
#endif
    /* Update the multivariate PDF Estimator */
    MPDFUpdate(mpdf,x);
    if(!xset)
      updatedomain(x,xmin,xmax,dim);
    samples++;
  }
  /* Finished with input file */
  if (ifile)
    fclose(istream);

  /* Some info for the user */
  printf("Number of samples: %d\n",samples);
  printf("Dimensions: %d\n",dim);

  /* This is sometimes needed in double precision */
  dSamples=samples*1.;
  printf("Mean vector:");
  /* Compute the mean it is needed to center the dataset */
  for(i=0; i< dim ; i++){
    mean->ve[i]=Sx->ve[i]/dSamples;
    printf(" %g",mean->ve[i]);
  }
  printf("\n");

  /* Prepare mean and std dev vectors to store later */
  double * means = (double *)malloc(sizeof(double) * dim);
  double * stdDevs = (double *)malloc(sizeof(double) * dim);

  for(i=0;i<dim;i++)
	means[i] = mean->ve[i];
	  
  for(i=0;i<dim;i++)
	stdDevs[i] = 0.0f;
	  
  for(i = 0; i < samples; i++)
	for(j = 0; j < dim; j++)
	{
		double temp = MPDFPosition(mpdf,i)[j];
		stdDevs[j] += (temp - means[j]) * (temp - means[j]);						
	}  
	
  for(i = 0; i < dim; i++)
  {
	if(mpdf->current > 1)
		stdDevs[i] = sqrt(stdDevs[i] / (double)(mpdf->current-1.));
	else
		stdDevs[i] = sqrt(stdDevs[i] / (double) mpdf->current);
  }	

  /* Covariance matrix */
  for(i=0;i<dim;i++){
    for(j=0;j<dim;j++){
      S->me[i][j]=S->me[i][j]/samples-Sx->ve[i]*Sx->ve[j]/samples/samples;
    }
  }
  //#ifdef DEBUG
  printf("Covariance matrix:\n");
  m_output(S);
  //#endif
  detS=getdeterminant(S);
  printf("Determinant of covariance matrix: %g\n",detS);

  /* Center the dataset before computing PCs and eigenvectors */
  cdata=m_get(mpdf->current,mpdf->length);
  assert(cdata);
  copyAndCenter(mpdf,mean,xmin,cxmin,xmax,cxmax); // Centered (mean removed data in cdata 
  /* 
     Compute rotation matrix from original 
     data to radially spheric kernel using single h bandwidth .
     Eigenvectors are returned orthonormal and as rows
  */
  computePCsEigenvaluesAndEigenvectors(S,&eigenvectors,&eigenvals,
				       &sqrtevals,&eigenvectorsT,&lm1E, 
				       &lm1Einv);


  /* Compute PCs */
  MPDF_ComputePCData(mpdf,lm1E);

  /* Invert the covariance matrix and get the determinant to be able to apply Fukunaga method */
  Sm1=m_get(S->m,S->n);
  assert(Sm1);
  m_inverse(S,Sm1);
#ifdef DEBUG
  printf("Inverse of covariance matrix:\n");
  m_output(Sm1);
  Itest=m_get(S->m,S->m);
  assert(Itest);
  m_mlt(S,Sm1,Itest);
  printf("Is it the identity matrix?\n");
  m_output(Itest);
  M_FREE(Itest);
#endif
#ifdef DEBUG
  printf("Determinant of the covariance matrix: %g\n",detS);
#endif
  /* Estimate the optimum bandwith */
  bwtheo=MPDFgetoptimumbandwidth(dim,samples);
  if(!bwset)
    bandwidth=bwtheo;

  printf("Bandwidth (gaussian): %g\n",bwtheo);
  printf("Bandwidth (used): %g\n",bandwidth);

  double * bounds = (double *)malloc(sizeof(double) * dim);

  //Calculate bounding box for kernel ellipse
  temp_bounds=m_get(dim,dim);
  assert(temp_bounds);
  calculateBoundaries(bounds,bandwidth,eigenvals,eigenvectors,
		      sqrtevals,temp_bounds);
 
#ifdef DEBUGDIST
  VEC *XG;
  VEC *PG;
  VEC *XO;
  VEC *PO;
  VEC *TMP;
  VEC *TMP2;
  MAT *Lm1;
  MAT *M2;
  MAT *E;
  double xdist;
  double pdist;
  int II;
  int JJ;

  Lm1=m_get(dim,dim);
  for(II=0;II<dim;II++){
    for(JJ=0;JJ<dim;JJ++){
      Lm1->me[II][JJ]=0.0;
    }
    Lm1->me[II][II]=1./eigenvals->ve[II];
  }
  M2=m_get(dim,dim);
  /* These are supposed to be "grid" and observed points in X or PC spaces */
  XG=v_get(dim);
  XG->ve[0]=50;
  XG->ve[1]=-100.;
  XG->ve[2]=30.;
  XO=v_get(dim);
  XO->ve[0]=50.2;
  XO->ve[1]=-100.3;
  XO->ve[2]=-30.2;
  /* we compute PCs */
  PG=v_get(dim);
  PO=v_get(dim);

  E=lm1E;

  x2pc(E,XG,PG);
  x2pc(E,XO,PO);
  /* Now, we compare norms */
  TMP=v_get(dim);
  TMP2=v_get(dim);
  for(II=0;II<dim;II++){
    TMP->ve[II]=XO->ve[II]-XG->ve[II];
  }
  mv_mlt(Sm1,TMP,TMP2);
  xdist=in_prod(TMP,TMP2);
  for(II=0;II<dim;II++){
    TMP->ve[II]=PO->ve[II]-PG->ve[II];
  }
  pdist=in_prod(TMP,TMP);
  printf("Testing distances: X= %f versus PC= %f\n",xdist,pdist);

  printf("Do we recover X from PCs???\n");
  /* Checking that from PCs we recover the original vectors */
  for(II=0;II<dim;II++){
    printf(" %f",XO->ve[II]);
  }
  printf(" - ");
  pc2x(lm1Einv,PO,XO);
  for(II=0;II<dim;II++){
    printf(" %f",XO->ve[II]);
  }
  printf("\n");

  for(II=0;II<dim;II++){
    printf(" %f",XG->ve[II]);
  }
  printf(" - ");
  pc2x(lm1Einv,PG,XG);
  for(II=0;II<dim;II++){
    printf(" %f",XG->ve[II]);
  }
  printf("\n");

  V_FREE(PG);
  V_FREE(XG);
  V_FREE(PO);
  V_FREE(XO);
  V_FREE(TMP);
  V_FREE(TMP2);
  M_FREE(M2);
  M_FREE(Lm1);
#endif

  /********************************************/
  /* Compute the estimation of the PDF */
  /********************************************/   
  computePDF(mpdf,pdf,Sm1,bandwidth,detS,cxmin,cxmax,deltax,xset,bounds,lm1E,dim);

  /* Print to the netCDF file the multivariate PDF */ 
  dumpND(pdf, ncid, bandwidth, dim, xmin, deltax, means, stdDevs);
  
  //Free allocated structures  
  free(means);
  free(stdDevs);    
  free(bounds);
  free(xmin);
  free(xmax);
  free(cxmin);
  free(cxmax);
  free(deltax);
  free(x);  
  
  M_FREE(S);
  V_FREE(Sx);
  M_FREE(Sm1);
  V_FREE(mean);
  V_FREE(eigenvals);
  M_FREE(eigenvectors);
  M_FREE(eigenvectorsT);
  M_FREE(sqrtevals);
  M_FREE(temp_bounds);
  M_FREE(cdata);
  M_FREE(lm1E);
  M_FREE(lm1Einv);
  KillMPDFEstimator(mpdf);
  FreePDF(pdf);
  status=nc_close(ncid);
  check_nc_error(status);

  //DEBUG AND ENDING TIMING ROUTIUNES
  gettimeofday(&tv_end, NULL); //Get starting time
  t_end = tv_end.tv_sec + (tv_end.tv_usec/1000000.0);
  elapsed = t_end - t_start ;

  printf("\nTotal time: %f sec %f min %f hour\n",
	 elapsed,elapsed/60.,elapsed/3600.);

  return 0;
}

