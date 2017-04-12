/**************

PDF.c

Functions for creating/deallocating/accessing PDF data structures

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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "PDF.h"

//Constructor (see PDF.h for info about fields of the struct)
PDFPtr NewPDF( double *xmin, double *xmax , double *deltax , int dim )
{
  PDFPtr pdf;
  int i;
	
  pdf=(PDFPtr)malloc(sizeof(PDF));

  pdf->total_size=1;
  pdf->pdfsize = (size_t *)malloc(sizeof(size_t) * dim);
  pdf->pdfcumsize = (size_t *)malloc(sizeof(size_t) * dim);  
  
  assert(pdf);

  //Define length of dimensions. In 2D case, leave an extra evaluation point
  if(dim == 2)
  {
	for(i=0;i<dim;i++)
	{
		pdf->pdfsize[i] = (size_t)((xmax[i]-xmin[i])/deltax[i])+2;
		pdf->total_size = pdf->total_size * pdf->pdfsize[i]; 
	}  
  }
  else
	for(i=0;i<dim;i++)
	{
		pdf->pdfsize[i] = (size_t)((xmax[i]-xmin[i])/deltax[i])+1;
		pdf->total_size = pdf->total_size * pdf->pdfsize[i]; 
	}  

  //Create pdfcumsize structure. It is used to calculate the absolute
  //position in the structure of a gridpoint, given its cartesian coordinates
  pdf->pdfcumsize[dim-1]=1;
  i=dim-2;
  while (i>=0){
    pdf->pdfcumsize[i]=pdf->pdfcumsize[i+1]*pdf->pdfsize[i+1];
    i--;
  }
  
  pdf->PDF=(double*)calloc(pdf->total_size,sizeof(double));
  assert(pdf->PDF);
  memset(pdf->PDF,'\0',pdf->total_size);
  return pdf;
}

//Given the coordinates of a gridpoint (elementpos), returns pointer to position in PDF structure
double *PDFitem( PDFPtr pdf , size_t *elementpos , int maxdim )
{
  double *dpos;
  int i;

  dpos=pdf->PDF;
  for(i=0;i<maxdim;i++){
    dpos+=elementpos[i]*pdf->pdfcumsize[i];
  }
  return dpos;
}

//Given the coordinates of a gridpoint (elementpos), returns position in PDF structure
int PDFposition( PDFPtr pdf , size_t *elementpos , int maxdim )
{
  int i, position;
  position = 0;
  for(i = 0; i<maxdim; i++)
	  position += elementpos[i] * pdf->pdfcumsize[i];

  return position;
}

//Destructor
void FreePDF( PDFPtr pdf )
{
  free(pdf->pdfsize);	
  free(pdf->pdfcumsize);  
  free((void*)pdf->PDF);
  free((void*)pdf);
}


void pcPDF2xPDF( PDFPtr pdf , double sqrtdetJ )
{
  size_t i;
  for(i=0;i<pdf->total_size;i++)
    pdf->PDF[i]=pdf->PDF[i]/sqrtdetJ;
}
