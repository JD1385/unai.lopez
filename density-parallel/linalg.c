/**************

linalg.c

Decomposition of the covariance matrix and computation
of eigenvalues, rotation matrix and so on. Wrapper around meschach routines.


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

#include "linalg.h"


/* Get the eigenvalues and then the determinant (A is symmetric) */
double getdeterminant( MAT *A )
{
  double det=1.0;
  int i;
  VEC *evals;

  evals=symmeig(A,MNULL,VNULL);
  assert(evals);
  for(i=0;i<A->m;i++){
    det=det*evals->ve[i];
#ifdef DEBUG
    printf("#Eigenvalue: %d %g %g\n",i,evals->ve[i],det);
#endif
  }
  V_FREE(evals);
  return det;
}

/* 
Perform the decomposition of S an return to the caller some matrices that
will be used in the forward/backward projection during phase space to PCA 
and PCA to phase space conversions (see below).
*/
void  computePCsEigenvaluesAndEigenvectors(MAT *S, MAT **eigenvectors,
					   VEC **eigenvals, MAT **sqrtevals,
					   MAT **eigenvectorsT , MAT **lm1E ,
					   MAT **lm1Einv )
{
  int i,j,dim;
  double *sqrt_evals;
  double var=1.;

  dim=S->m;
  *eigenvectors=m_get(S->m,S->m);//Allocate space for eigenvectors matrix
  assert(*eigenvectors);	
  *eigenvectorsT=m_get(S->m,S->m);//Allocate space for eigenvectors matrix
  assert(*eigenvectorsT);
  *lm1E=m_get(dim,dim);
  assert(*lm1E);
  *lm1Einv=m_get(dim,dim);
  assert(*lm1Einv);
  // we are assuming here that samples > dimensions,
  // otherwise, the space is too empty and computing the PDF has no
  // sense at all. Additionally, we assume covariance matrix is of 
  // full rank. Otherwise, user is really cheating him/herself and he can 
  // save him/herself one dimension, which is very time-consuming.
  *eigenvals=symmeig(S,*eigenvectors,VNULL); //Get eigen vectors and values	
  assert(*eigenvals);
  //SQRT of eigen values, which gives the length of the ellipses axis
  sqrt_evals = (double *)malloc(sizeof(double)*dim);
  assert(sqrt_evals);
  printf("Length of principal axes:");
  for(i=0; i <dim; i++){
    sqrt_evals[i] = sqrt((*eigenvals)->ve[i]);
    printf(" %g",sqrt_evals[i]);
  }
  printf("\n");
  printf("Variance from eigenvalues:");
  for(i=0; i <dim; i++){
    var=var*(*eigenvals)->ve[i];
  }
  printf("%g\n",var);
  // Principal axes are written as columns now
  // Get principal axes as rows
  m_transp(*eigenvectors, *eigenvectors); //Transpose eigenvectors matrix
  // Principal axes are written as columns now
  printf("Principal axes:\n");
  for(i=0;i<dim;i++){
    printf("u-%4.4d: ",i);
    for(j=0;j<dim;j++){
      printf(" %g",(*eigenvectors)->me[i][j]);
    }
    printf("\n");
  }
  //Make Eigenvalues vector a diagonal matrix	
  *sqrtevals = m_get((*eigenvals)->dim, (*eigenvals)->dim); //Allocate space	
  for(i=0; i < dim; i++)
    for(j=0; j < dim; j++)
      if (i == j)
	(*sqrtevals)->me[i][j] = sqrt_evals[i];
      else
	(*sqrtevals)->me[i][j] = 0.0f;
  m_transp(*eigenvectors,*eigenvectorsT);
  for(i=0;i<dim;i++){
    for(j=0;j<dim;j++){
      (*lm1E)->me[i][j]=(*eigenvectors)->me[i][j]/sqrt((*eigenvals)->ve[i]);
    }
  }
  m_inverse(*lm1E,*lm1Einv);
  free((void*)sqrt_evals);
}

/*

In this implementation, eigenvalues are stored as rows when exiting 
from computePCsEigenvalues... function. Therefore,
we apply here the conversion accordingly

P=Ex

OR

P= (lm1E) x

for standardized principal components

*/



//Original x2pc
void x2pc( MAT *eigenvectors , VEC *x, VEC *pc )
{
  mv_mlt(eigenvectors,x,pc);
}


void pc2x( MAT *eigenvectorsT , VEC *pc, VEC *x )
{
  mv_mlt(eigenvectorsT,pc,x);
}

/* in some cases we have (double *) pointers from reading the ASCII files */
void DPx2pc( MAT *eigenvectors , double *x , VEC *vx, VEC *pc )
{
  memcpy(vx->ve,x,sizeof(double)*vx->dim);
  x2pc(eigenvectors,vx,pc);
}

/* Same back */
void pc2DPx( MAT *eigenvectorsT , VEC *pc, VEC *vx , double *x )
{
  pc2x(eigenvectorsT,pc,vx);
  memcpy(x,vx->ve,sizeof(double)*vx->dim);
}


