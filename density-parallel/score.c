/**************

score.c

Analysis program that computes the PDF-Score from two N-dimensional PDFs.

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
#include <math.h>
#include "netcdf.h" //NetCDF

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERRCODE 2
#define ERR(e) {if(e>0){printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}}

#define min(a,b) ((a < b) ? a : b)

int main(int argc, char** argv)
{
	//Check argument number
	if(argc != 3)
	{	printf("\nUsage: ./mpdf_score file1 file2\n"); exit(1);	}
	
	int i; //Loop variable
	int retval; //Error handling
	int ncid[2]; // NetCDF ID for the files
	
	// Open file 1 and files 2 for read-only access
	retval = nc_open(argv[1], NC_NOWRITE, &ncid[0]); ERR(retval);	
	retval = nc_open(argv[2], NC_NOWRITE, &ncid[1]); ERR(retval);		
	
	//Check amount of dimensions of two files
	int dims[2];
	retval = nc_inq_ndims(ncid[0], &dims[0]); ERR(retval);
	retval = nc_inq_ndims(ncid[1], &dims[1]); ERR(retval);
	if(dims[0] != dims[1])
	{	printf("\nError: different dimensions in files: 1:%d 2:%d\n",dims[0],dims[1]);	exit(1);	}
	int dim = dims[0]; //Use dim for dimensions	
		
	//Substract 1 as "ndim" is taken into account as dimension	
	dim -= 1;	
		
	//Retrieve identifiers for dimensions
	int * file1_dim_ids = (int *)malloc(sizeof(int) * dim);
	int * file2_dim_ids = (int *)malloc(sizeof(int) * dim);
	char label[20]; //Aux variable to generate per-dim labels in NetCDF file

	//File 1
	for(i = 0; i < dim; i++)
	{
		sprintf(label,"dim%d",i); 	
		retval = nc_inq_dimid(ncid[0], label, &file1_dim_ids[i]); ERR(retval);	
	}
	
	//File 2
	for(i = 0; i < dim; i++)
	{
		sprintf(label,"dim%d",i); 	
		retval = nc_inq_dimid(ncid[1], label, &file2_dim_ids[i]); ERR(retval);	
	}
	
	//Check if length of each dimension is the same
	size_t * file1_dim_length = (size_t *)malloc(sizeof(size_t) * dim);
	size_t * file2_dim_length = (size_t *)malloc(sizeof(size_t) * dim);
	
	for(i = 0; i < dim; i++)
	{
		retval = nc_inq_dimlen(ncid[0], file1_dim_ids[i], &file1_dim_length[i]); ERR(retval);
		retval = nc_inq_dimlen(ncid[1], file2_dim_ids[i], &file2_dim_length[i]); ERR(retval);		
		
		if(file1_dim_length[i] != file2_dim_length[i])
		{	printf("\nError: different length for dimension %d: 1:%zu 2:%zu",i,file1_dim_length[i],file2_dim_length[i]); exit(1); }
	}	
	
	//Retrieve identifiers for variables that hold evaluation points
	int * file1_var_ids = (int *)malloc(sizeof(int) * dim);
	int * file2_var_ids = (int *)malloc(sizeof(int) * dim);
	
	//File 1
	for(i = 0; i < dim; i++)
	{
		sprintf(label,"X%d",i); 	
		retval = nc_inq_varid(ncid[0], label, &file1_var_ids[i]); ERR(retval);	
	}
	
	//File 2
	for(i = 0; i < dim; i++)
	{
		sprintf(label,"X%d",i); 	
		retval = nc_inq_varid(ncid[1], label, &file2_var_ids[i]); ERR(retval);	
	}

	//Retrieve variables that contain evaluation points per dimension
	//Check their equality and calculate delta values
	double ** file1_points = (double **)malloc(sizeof(double*) * dim);
	double ** file2_points = (double **)malloc(sizeof(double*) * dim);
	double * delta = (double *)malloc(sizeof(double) * dim);
	double deltaV;
	
	for(i = 0; i < dim; i++)
	{
		//Retrieve variables
		file1_points[i] = (double *)malloc(sizeof(double) * file1_dim_length[i]);
		file2_points[i] = (double *)malloc(sizeof(double) * file2_dim_length[i]);				
		retval = nc_get_var_double(ncid[0], file1_var_ids[i], file1_points[i]); ERR(retval);	
		retval = nc_get_var_double(ncid[1], file2_var_ids[i], file2_points[i]); ERR(retval);	

		//Check first, second and last value of dimension
		if(file1_points[i][0] != file2_points[i][0])
		{	printf("\nError: different first eval point for dimension %d: 1:%f 2:%f",i,file1_points[i][0],file2_points[i][0]); exit(1); }
		if(file1_points[i][1] != file2_points[i][1])
		{	printf("\nError: different second eval point for dimension %d: 1:%f 2:%f",i,file1_points[i][1],file2_points[i][1]); exit(1); }		
		if(file1_points[i][file1_dim_length[i]-1] != file2_points[i][file2_dim_length[i]-1])
		{	printf("\nError: different last eval point for dimension %d: 1:%f 2:%f",i,file1_points[i][file1_dim_length[i]-1] ,file2_points[i][file1_dim_length[i]-1] ); exit(1); }		

		//Calculate delta
		delta[i] = file1_points[i][1] - file1_points[i][0];
		
		//Free allocated structures in these loops
		free(file1_points[i]);
		free(file2_points[i]);		
	}

	//Calculate delta volume
	deltaV = 1.0f;
	for(i = 0; i < dim; i++)
		deltaV *= delta[i];
				
	//Retrieve means and stdDev vectors for dimensions; calculate potential bias errors		
	int var_mean_ids[2];
	retval = nc_inq_varid(ncid[0], "Average", &var_mean_ids[0]); ERR(retval);	
	retval = nc_inq_varid(ncid[1], "Average", &var_mean_ids[1]); ERR(retval);	
	
	double * file1_means = (double *)malloc(sizeof(double) * dim);
	double * file2_means = (double *)malloc(sizeof(double) * dim);
		
	retval = nc_get_var_double(ncid[0], var_mean_ids[0], file1_means); ERR(retval);	
	retval = nc_get_var_double(ncid[1], var_mean_ids[1], file2_means); ERR(retval);	
				
	int var_stdDev_ids[2];
	retval = nc_inq_varid(ncid[0], "StdDev", &var_stdDev_ids[0]); ERR(retval);	
	retval = nc_inq_varid(ncid[1], "StdDev", &var_stdDev_ids[1]); ERR(retval);	
	
	double * file1_stdDev = (double *)malloc(sizeof(double) * dim);
	double * file2_stdDev = (double *)malloc(sizeof(double) * dim);
		
	retval = nc_get_var_double(ncid[0], var_stdDev_ids[0], file1_stdDev); ERR(retval);	
	retval = nc_get_var_double(ncid[1], var_stdDev_ids[1], file2_stdDev); ERR(retval);			
				
	for(i = 0; i < dim; i++)
	{
		double bias_i = fabs(file1_means[i] - file2_means[i]);
		double stdmin_i = min(file1_stdDev[i], file2_stdDev[i]);
		
		if (bias_i >= 0.05*stdmin_i)		
			fprintf(stderr, "\nWarning: Bias (%g) in dimension %d is greater than 5%% of the minimum std (%g)\n",bias_i,i,stdmin_i);
	}				
	
	//Retrieve PDF variables
	int var_pdf_ids[2];
	retval = nc_inq_varid(ncid[0], "PDF", &var_pdf_ids[0]); ERR(retval);	
	retval = nc_inq_varid(ncid[1], "PDF", &var_pdf_ids[1]); ERR(retval);	

	int pdf_length = 1; //Total number of evaluation points in PDF
	for(i = 0; i < dim; i++)
		pdf_length *= file1_dim_length[i];

	double * file1_pdf = (double *)malloc(sizeof(double) * pdf_length);
	double * file2_pdf = (double *)malloc(sizeof(double) * pdf_length);	

	retval = nc_get_var_double(ncid[0], var_pdf_ids[0], file1_pdf); ERR(retval);	
	retval = nc_get_var_double(ncid[1], var_pdf_ids[1], file2_pdf); ERR(retval);		

	//Calculate Pitman et al score
	double min;
	double score = 0.0;

	for(i = 0; i < pdf_length; i++)
	{
		min = (file1_pdf[i] < file2_pdf[i]) ? file1_pdf[i] : file2_pdf[i];
		score += min *deltaV;		
	}
	printf("Pitman et al score: %.13f",score);
	
	//Free allocated vectors
	free(file1_dim_ids);
	free(file2_dim_ids);	
	free(file1_dim_length);
	free(file2_dim_length);
	free(file1_var_ids);
	free(file2_var_ids);		
	free(file1_points);
	free(file2_points);	
	free(delta);
	free(file1_means);
	free(file1_stdDev);
	free(file2_means);
	free(file2_stdDev);
	free(file1_pdf);
	free(file2_pdf);	
			
	// Close the files, freeing all resources
	retval = nc_close(ncid[0]); ERR(retval);	
	retval = nc_close(ncid[1]); ERR(retval);		
	
	printf("\n");
	return 0;
}
