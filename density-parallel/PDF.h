/**************

PDF.h

PDF data structure and function prototypes for PDF.c

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

#ifndef __PDF__
#define __PDF__ 1

typedef struct {
  double *PDF; //Structure to hold the multidimensional PDF
  size_t * pdfsize; //Number of grid points per dimension
  size_t * pdfcumsize; //Accumulated grid points per dimension. 
					//Used to translate gridpoint coordinates to the position in the PDF Structure
  size_t total_size; //Total number of grid points of the PDF
} PDF, *PDFPtr;


PDFPtr NewPDF( double *xmin, double *xmax , double *deltax , int dim ); //Constructor
double *PDFitem( PDFPtr pdf , size_t *elementpos , int maxdim ); //Given the coordinates of a gridpoint (elementpos), returns pointer to position in PDF structure
int PDFposition( PDFPtr pdf , size_t *elementpos , int maxdim ); //Given the coordinates of a gridpoint (elementpos), returns position in PDF structure
void FreePDF( PDFPtr pdf ); //Destructor

void pcPDF2xPDF( PDFPtr pdf , double sqrtdetJ );

#endif
