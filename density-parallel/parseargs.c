/**************

parseargs.c

Parse input line arguments

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

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "parseargs.h"

void usage(char *pname)
{
  fprintf(stderr,"%s [-h bandwidth] [-o ncfile] -x xmin:xmax:deltax ifile\n",pname);
  fprintf(stderr,"-h bandwidth -> Bandwidth to be used in the kernel\n");
  fprintf(stderr,"-x xmin:xmax:deltax -> Interval to compute the density\n");
  fprintf(stderr,"Each of the values should be a n-dimensional vector.\n");
  fprintf(stderr,"For instance: -1/1.5:1/2.5:0.1/.5\n");
  fprintf(stderr,"can be used for a 2D distribution on [-1,1]x[1.5,2.5],\n");
  fprintf(stderr,"stepping by 0.1 (1st dimension) and by 0.5 (2nd one).\n");
  fprintf(stderr,"-o ncfile Output netCDF file. Default: MPDF.nc\n");
  fprintf(stderr,"ifile Data as ASCII rows from ifile (default stdin)\n");
  fprintf(stderr,"each row should be a vector or comment and all the rows\n");
  fprintf(stderr,"must be the same size\n");
  exit(1);
}

void usageBS(char *pname)
{
  fprintf(stderr,"%s [-n maxreal] [-h bandwidth] [-H hmin/hmax/deltah] -x xmin:xmax:deltax ifile\n",pname);
  fprintf(stderr,"-h bandwidth -> Bandwidth (1D) to be used in the kernel (reference PDF)\n");
  fprintf(stderr,"-x xmin:xmax:deltax -> Interval to compute the density\n");
  fprintf(stderr,"Each of the values should be a n-dimensional vector.\n");
  fprintf(stderr,"For instance: -1/1.5:1/2.5:0.1/.5\n");
  fprintf(stderr,"can be used for a 2D distribution on [-1,1]x[1.5,2.5],\n");
  fprintf(stderr,"stepping by 0.1 (1st dimension) and by 0.5 (2nd one).\n");
  fprintf(stderr,"ifile Data as ASCII rows from ifile (default stdin)\n");
  fprintf(stderr,"each row should be a vector or comment and all the rows\n");
  fprintf(stderr,"must be the same size\n");
  fprintf(stderr,"-n maxreal -> Maximum number of realizations\n");
  fprintf(stderr,"-H hmin/hmax/deltah -> Range of bandwidths to test.\n");
  exit(1);
}

void parseHVector(char *arg , double *h, char *pname , int *dims )
{
  char *tok=strtok(arg,"/");
  char *endptr;
  *dims=0;
  while(tok){
    h[*dims]=strtod(tok,&endptr);
    if (endptr==tok)
      usageBS(pname);
    (*dims)++;
    tok=strtok(NULL,"/");
  }
}


void parseCNVector(char *arg , double *h, char *pname , int *dims )
{
  char *tok=strtok(arg,"/");
  char *endptr;
  *dims=0;
  while(tok){
    h[*dims]=strtod(tok,&endptr);
    if (endptr==tok)
      usage(pname);
    (*dims)++;
    tok=strtok(NULL,"/");
  }
}

void parseX(char *arg, double *xmin, double *xmax, double *deltax,
	    char *pname , int *ndim )
{
  char *tokmin,*tokmax,*tokdelta;

#ifdef DEBUG
  printf("Parsing:%s\n",arg);
#endif
  tokmin=strtok(arg,":");
  tokmax=strtok(NULL,":");
  tokdelta=strtok(NULL,":");
  parseCNVector(tokmin,xmin,pname,ndim);
#ifdef DEBUG
  printf("Parsed: %g %g %g %d\n",xmin[0],xmin[1],xmin[2],*ndim);
#endif
  parseCNVector(tokmax,xmax,pname,ndim);
#ifdef DEBUG
  printf("Parsed: %g %g %g %d\n",xmax[0],xmax[1],xmax[2],*ndim);
#endif
  parseCNVector(tokdelta,deltax,pname,ndim);
#ifdef DEBUG
  printf("Parsed: %g %g %g %d\n",deltax[0],deltax[1],deltax[2],*ndim);
#endif
}

int parsevector(double *x, int *dim , char *buffer , int first )
{
  char *tok=strtok(buffer," \t");
  int id=0;
  char *endptr;

  /* This way, we skip comments */
  if (tok[0]=='#')
    return 0;
  /* This way, it skips blank lines not entering into the cycle ... */
  while(tok){
    x[id]=strtod(tok,&endptr);
    assert(endptr!=tok);
    id++;
    tok=strtok(NULL," \t");
  }
#ifdef DEBUG
  printf("%d,%d\n",*dim,id);
#endif
  if (first)
    *dim=id;
  return (id==*dim);
}

void parseallargs( int argc , char **argv , char **ifile , double *bandwidth ,
		   double *xmin, double *xmax, double *deltax ,
		   char *buffer , int *bwset , int *xset, char **oncname){
  int iarg;
  char *arg;
  char *endptr;
  int dim;
  
  *bwset=0;
  /* Add input arguments to buffer to write them as comment to nc file */
  strcpy(buffer,argv[0]);
  for(iarg=1;iarg<argc;iarg++){
    arg=argv[iarg];
    strcat(buffer," ");
    strcat(buffer,arg);
  }
  for(iarg=1;iarg<argc;iarg++){
    arg=argv[iarg];
    if (arg[0]!='-'){
      /* memory for this is already allocated by the system. This is safe */
      *ifile=arg;
    } else if (strcmp("-h",arg)==0){
      if (iarg+1>=argc)
	usage(argv[0]);
      *bandwidth=strtod(argv[iarg+1],&endptr);
      if(endptr==argv[iarg+1])
	usage(argv[0]);
      iarg++;
      *bwset=1;
    } else if (strcmp("-x",arg)==0) {
      if (iarg+1>=argc)
	usage(argv[0]);
      arg=argv[iarg+1];
      iarg++;
      parseX(arg,xmin,xmax,deltax,argv[0],&dim);
      *xset=1;
    } else if (strcmp("-o",arg)==0) {
      if (iarg+1>=argc)
	usage(argv[0]);
      /* memory for this is already allocated by the system. This is safe */
      *oncname=argv[iarg+1];;
      iarg++;
    } else
      usage(argv[0]);      
  }
}

void parseallargsBS( int argc , char **argv , char **ifile , double *bandwidth ,
		   double *xmin, double *xmax, double *deltax ,
		     int *bwset , int *xset , int *maxreal, double *H ,
		     int *Hset ){
  int iarg;
  char *arg;
  char *endptr;
  int hset=0;
  int Hhset=0;
  int dim;

  *bwset=0;
  for(iarg=1;iarg<argc;iarg++){
    arg=argv[iarg];
    if (arg[0]!='-'){
      /* memory for this is already allocated by the system. This is safe */
      *ifile=arg;
    } else if (strcmp("-h",arg)==0){
      if (iarg+1>=argc)
	usageBS(argv[0]);
      *bandwidth=strtod(argv[iarg+1],&endptr);
      if(endptr==argv[iarg+1])
	usageBS(argv[0]);
      iarg++;
      hset=1;
      *bwset=1;
    } else if (strcmp("-n",arg)==0){
      if (iarg+1>=argc)
	usageBS(argv[0]);
      *maxreal=atoi(argv[iarg+1]);
      iarg++;
    } else if (strcmp("-x",arg)==0) {
      if (iarg+1>=argc)
	usageBS(argv[0]);
      arg=argv[iarg+1];
      iarg++;
      parseX(arg,xmin,xmax,deltax,argv[0],&dim);
      *xset=1;
    } else if (strcmp("-H",arg)==0) {
      if (iarg+1>=argc)
	usageBS(argv[0]);
      arg=argv[iarg+1];
      iarg++;
      parseHVector(arg ,H, argv[0] ,&dim );
      *Hset=1;
    } else
      usageBS(argv[0]);
  }
  Hhset=(*Hset) && (hset);
  if (! (Hhset)){
    fprintf(stderr,"Both -h and -H are absent, using values from Gaussian distribution.\n");
  }
}


void fetchdims(int argc , char ** argv_original, int * dims)
{	
	int i;

	//Perform a local copy of the arguments vector
	char ** argv = (char **)malloc(sizeof(char **) * argc);
	
	for(i = 0; i < argc; i++)
	{
		argv[i] = malloc( strlen( argv_original[i] ) + 1 );
		strcpy( argv[i], argv_original[i] );
	}
		
	int iarg;
	char *arg;	
	
	int dimset = 0;
	
	//Fetch each command line arg until the line that defines the grid space is found
	for(iarg=1; (iarg < argc) && (dimset == 0); iarg++)
	{
		arg=argv[iarg];

		if (strcmp("-x",arg)==0) 
		{
			if (iarg+1>=argc)
				usage(argv[0]);
			arg=argv[iarg+1];
			iarg++;			 
			 
			char *tokmin = strtok(arg,":");			 
			char *tok=strtok(tokmin,"/");
			char *endptr;
			*dims=0;
			  
			//Loop over the defition to count the number of dimensions
			while(tok)
			{
				if (endptr==tok)
					usage("xmin");
				(*dims)++;
				tok=strtok(NULL,"/");
			}			  
			  
			dimset=1;
		}
	}
	
	//Dimension valid check
	if ((dimset==0) || (*dims == 0))
	{
		fprintf(stderr,"Dimension can not be set by input arguments, problem is undefined\n");
		usage(argv[0]);
	}	
	
	//Delete local arv vector
	for(i = 0; i < argc; i++)
		free(argv[i]);
	free(argv);
}
