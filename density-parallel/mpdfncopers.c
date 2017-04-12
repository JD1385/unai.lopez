/**************

mpdfncopers.c

Functions related to writing the output to netCDF files.

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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "netcdf.h"
#include "mpdfncopers.h"
#include "computePDF.h"

void __check_nc_error( int err , char *file , int lineno )
{
  if(err!=NC_NOERR){
    fprintf(stderr,"Error %d writing netCDF file:\nLine:%d\nFile:%s\n",err,
                    lineno,file);
    fprintf(stderr,"Error message:%s\n",nc_strerror(err));
    exit(1);
  }
}

//DUMP PDF to NetCDF
void dumpND(PDFPtr pdf, int ncid, double h, int dim, double * x0, double * dx, double * means, double * stdDevs)
{
  char buffer[256];	
  int status; //Error handling
  int * dims = (int *)malloc(sizeof(int) * dim);
  
  //NetCDF variable handling
  int * var = (int *)malloc(sizeof(int) * dim);
  int varpdf;
  int varMean;
  int varStdDev;
  int i,j; //Loop variable	
	
  /* Save the bandwidth as a comment */
  memset(buffer,'\0',256);
  sprintf(buffer,"%g",h);
  status=nc_put_att_text(ncid,NC_GLOBAL,"Bandwidth",strlen(buffer),buffer);
  
  char label[20];
  
  //Create the dimensions and the variables
  for(i = 0; i < dim; i++)
  {
	  sprintf(label,"dim%d",i); 
	  	  
	  status=nc_def_dim (ncid,label, pdf->pdfsize[i],&(dims[i]));
	  check_nc_error(status);	
	  
	  sprintf(label,"X%d",i); 
	  
	  status=nc_def_var(ncid,label,NC_DOUBLE,1,&dims[i],&var[i]);
	  check_nc_error(status);	    
  }
  
  status=nc_def_var(ncid,"PDF",NC_DOUBLE,dim,dims,&varpdf);
  check_nc_error(status);
  
  //Create the dimension and mean,stddev variables
  int ndimvar;	  
  status=nc_def_dim (ncid,"ndim", dim, &ndimvar);
  check_nc_error(status);	  
  
  status=nc_def_var(ncid,"Average",NC_DOUBLE, 1, &ndimvar,&varMean);
  check_nc_error(status);  
  status=nc_def_var(ncid,"StdDev",NC_DOUBLE, 1, &ndimvar,&varStdDev);
  check_nc_error(status);    
  
  // Finish definitions 
  status=nc_enddef(ncid);
  check_nc_error(status);
    
  //Temp will be used as temporal array for writing values in netCDF file  
  //First find the max required size for the array
  int max_temp_size = 0;  
  
  for(i = 0; i < dim; i++)
	max_temp_size = (pdf->pdfsize[i] > max_temp_size) ? (pdf->pdfsize[i]) : max_temp_size;
          
  double *temp;
  temp = (double*)malloc(max_temp_size * sizeof(double));  
    
  //Write aux variables
  for(i = 0; i < dim; i++)  
  {
	  //Create values vector
	  for(j = 0; j< pdf->pdfsize[i]; j++)
		temp[j]=x0[i]+dx[i]*j;
				
	  //Write to NetCDF	
	  status=nc_put_var_double(ncid,var[i],temp);
	  check_nc_error(status);
  } 
          
  //Write mean and std dev vectors    
  status=nc_put_var_double(ncid,varMean,means);
  check_nc_error(status);        
  status=nc_put_var_double(ncid,varStdDev,stdDevs);
  check_nc_error(status);                
        
  //Write PDF
  status=nc_put_var_double(ncid,varpdf,pdf->PDF);
  check_nc_error(status);		

  //Free variables
  free(temp);  
  free(dims);  
  free(var);
}

