/**************

genepsilon.c

Generate variates according to the univariate Epanechnikov kernel.

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
#include <time.h>
#include <math.h>



#include "genepsilon.h"
#include "MPDFEstimator.h"

static int initialized=0;
static   char state[256];
static double scale_factor=0.0;

static void initEpsilon( void )
{
  unsigned int seed=(unsigned int)time(NULL);
  size_t n=256;
  initstate(seed,state,n);
  initialized=1;
}

static inline double getV( void )
{
  return (2.*random()-RAND_MAX)/((double)RAND_MAX);
}

double epsilon( void )
{
  double eps;
  double V1;
  double aV1;
  double V2;
  double aV2;
  double V3;
  double aV3;

  if (!initialized){
    initEpsilon();
  }

  V1=getV(); aV1=fabs(V1);
  V2=getV(); aV2=fabs(V2);
  V3=getV(); aV3=fabs(V3);
  // see silverman page 143
  if ((aV3>=aV2) && (aV3>=aV1))
    eps=V2;
  else
    eps=V3;
  return eps*scale_factor;
}

void init_epsilon_data(double *meanEPS,double *varEPS,double *stdEPS,int nruns,
		       int dim )
{
  int i;
  double Sx,Sxx,EPS;
  double cd;

  cd=unit_sphere_volume(dim);
  Sx=0.0;
  Sxx=0.0;
  initEpsilon();
  scale_factor=(dim+2.)*4*sqrt(5.)/6./cd;
  for(i=0;i<nruns;i++){
    EPS=epsilon();
    Sx+=EPS;
    Sxx+=EPS*EPS;
  }
  *meanEPS=Sx/((double)nruns);
  *varEPS=Sxx/(nruns-1.)-Sx*Sx/((nruns-1.)*nruns);
  *stdEPS=sqrt(*varEPS);
}
