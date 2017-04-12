/**************

computePDF.c

Functions used to actually compute the PDFs for several dimensionalities.

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

#include "computePDF.h"
#include "linalg.h"

double volumeConstant(int dim)
{
	if(dim == 1)
		return 2.;
	else if(dim == 2)
		return acos(-1.);
	else if (dim == 3)
		return acos(-1.)*4./3.;	
	else
		return unit_sphere_volume(dim);
}
 
//Calculate the PDF of a defined 2D space (box) for a given sample. This function is called either by computePDF2D or computePDFND
//2D box is defined by "lower" and "tot_ev_per_dim". Lower has the coordinates of the lower corner of the box, and tot_ev_per_dim the number of gridpoints per dimension
void compute2DBox(MPDFEstimatorPtr mpdf, PDFPtr pdf, VEC * XX, VEC * PCdot, double * PC, double * lower, int * tot_ev_per_dim, double * gridpoint, 
	size_t * dif_pos, double * x0,double * dx, int dim, double h2, double cd, MAT * eigenvectors, int * densCounter, double * densValues, int * densPosition)
{
	int u,v,l; //Loop variables
	double temp; //Will contain the absolute distance value from gridpoint to sample.

	for(gridpoint[0] = lower[0], u = 0; u <= tot_ev_per_dim[0]; gridpoint[0] += dx[0], u++)
	{   		
		XX->ve[0] = gridpoint[0];
		
		for(gridpoint[1] = lower[1], v = 0; v <= tot_ev_per_dim[1]; gridpoint[1] += dx[1], v++)
		{  	
			XX->ve[1] = gridpoint[1];				
				
			//Conversion to PC space	
			x2pc(eigenvectors,XX,PCdot);	
		
			//Absolute distance calculation
			temp = 0;
			for(l = 0; l < dim; l++)
				temp += ((PC[l] - PCdot->ve[l]) * (PC[l] - PCdot->ve[l]));
			temp /= h2;

			//Check if the gridpoint is inside the influence area of the sample
			if (fabs(temp)<1.) 
			{   	
				dif_pos[0] = (gridpoint[0] - x0[0])/ dx[0];
				dif_pos[1] = (gridpoint[1] - x0[1])/ dx[1];

				//If OpenMP version, store the density value in an auxiliar vector densValues, previous to storing in the final PDF structure
				//Vector densPosition will contain the position of the gridpoint in the final PDF structure
				#ifdef _OPENMP

				densPosition[*densCounter] = PDFposition(pdf,dif_pos,dim);
				densValues[*densCounter] = 0.5/cd*(mpdf->length+2.)*(1.-temp);				
				*densCounter += 1;
				
				//If serial version, store the density value of the sample over the gridpoint in the PDF structure
				#else
				
				*PDFitem(pdf ,dif_pos, dim) += (0.5/cd*(mpdf->length+2.)*(1.-temp));	
				
				#endif
			}		
		}	
	}	
}


//Compute the PDF of a one-dimensional grid space
void computePDF1D(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , double detSm1 , double *x0, 
		double *x1, double *dx, double *bounds, MAT *eigenvectors )
{
  int i,j,u; //Loop variables
  int dim = 1;	//Dimensions of grid space
  double cd = volumeConstant(dim); //Volume constants to calculate kernel values    
  double h2=h*h;  //Squared bandwith value
  double *PC; // Current sample (PC space)
  double theintegral = 0.0;  
  double total_vol = 0.0;  
  double * sample; 
  double k=1./sqrt(detSm1)/mpdf->current/pow(h,mpdf->length);  //Constant to recover the volume in the X space from the volume in the PC space
  VEC *XX;
  VEC *PCdot;
   
 //Variables to calculate coordinates and number of gridpoints of bounding box
  int steps;	  
  double upper, lower, gridpoint;
  int tot_ev;
  size_t dif_pos[2];
  double abs_bound,temp;   
  
  //Auxiliary vectors for OpenMP version
  double * densValues;
  int * densPosition;    
  int densCounter = 0;  
  
  //Calculations required for aux vector in OpenMP version
  #ifdef _OPENMP

  int chunk_size; //Number of samples per data chunk
  int box_size = (ceil(bounds[0] / dx[0]) * 2) + 2; //Number of gridpoints in a bounding box
  long available_memory = (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) - (pdf->total_size * sizeof(double)); //Available memory (bytes) to create the aux vectors         
  int samples_per_thread,initialSample,max_sample_value;         
              
  #endif       
  
  #pragma omp parallel default(none) firstprivate(densCounter) \
  shared(mpdf,pdf,dim,x0,x1,dx,theintegral,total_vol,available_memory,box_size,bounds,eigenvectors,cd,h2,k) \
  private(i,j,u,sample,PC,lower,upper,steps,abs_bound,tot_ev,chunk_size,samples_per_thread,initialSample, \
  max_sample_value,dif_pos,gridpoint,XX,PCdot,densValues,densPosition,temp) 
  {	
 
  //Structures for holding each sample
  XX=v_get(mpdf->length);
  assert(XX);
  PCdot=v_get(mpdf->length);
  assert(PCdot);	

  #ifdef _OPENMP 

  chunk_size = available_memory / (omp_get_num_threads() * box_size * (sizeof(double) + sizeof(int))); //Number of samples in each aux vector

  densValues = (double *)malloc(sizeof(double) * chunk_size * box_size); //Vector to hold density values of each sample-gridpoint combination
  densPosition = (int *)malloc(sizeof(int) * chunk_size * box_size); //Vector to hold the positions of densValues values in the PDF structure
  
  samples_per_thread = mpdf->current / omp_get_num_threads(); //Number of samples to be processed by each OpenMP thread 
  initialSample = (samples_per_thread * omp_get_thread_num()); //First "i" value to be taken by each OpenMP thread
  max_sample_value = initialSample + samples_per_thread - 1; //Last "i" value to be taken by each OpenMP thread

  #endif     
  
  //Initialize PDF structure to 0s
  #pragma omp for
  for(i = 0; i < pdf->total_size; i++)
	pdf->PDF[i] = 0.0f;  
  
  //Main calculation loop. For each sample calculate the PDF of its influence area and store in the PDF structure
  #pragma omp for
  for(i=0;i<mpdf->current;i++) 
  {	
	sample = MPDFPosition(mpdf,i); //Get current sample
	PC = MPDFPCPosition(mpdf,i); //Get current sample (scaled as PC)
	  	  
	//For each sample, calculate its boundaries
	
	//Lower corner
	abs_bound = sample[0] - bounds[0];
	if (x0[0] > abs_bound)
		lower = x0[0];
	else
	{
		steps = floor((abs_bound - x0[0]) / dx[0]);
		lower = x0[0] + (steps * dx[0]);
	}

	//Upper corner
	abs_bound = sample[0] + bounds[0];
	if (x1[0] < abs_bound)
		upper = x1[0];
	else
	{
		steps = ceil((abs_bound - x0[0]) / dx[0]);
		upper = x0[0] + (steps * dx[0]);
	}	
	
	//Calculate number of eval points per dimension	
	tot_ev = rint((upper - lower)/dx[0]) + 1;		   
  
	//Calculate the PDF of the defined 1D space
	for(gridpoint = lower, u = 0; u <= tot_ev; gridpoint += dx[0], u++)
	{  	
		XX->ve[0] = gridpoint;
	
		//Conversion to PC space	
		x2pc(eigenvectors,XX,PCdot);	
	
		//Absolute distance calculation
		temp = ((PC[0] - PCdot->ve[0]) * (PC[0] - PCdot->ve[0])) / h2;
	
		//Check if the gridpoint is inside the influence area of the sample
		if (fabs(temp)<1.) 
		{   	
			dif_pos[0] = (gridpoint - x0[0])/ dx[0];

			//If OpenMP version, store the density value in an auxiliar vector densValues, previous to storing in the final PDF structure
			//Vector densPosition will contain the position of the gridpoint in the final PDF structure
			#ifdef _OPENMP

			densPosition[densCounter] = PDFposition(pdf,dif_pos,dim);
			densValues[densCounter] = 0.5/cd*(mpdf->length+2.)*(1.-temp);				
			densCounter += 1;
			
			//If serial version, store the density value of the sample over the gridpoint in the PDF structure
			#else
			
			*PDFitem(pdf ,dif_pos, dim) += (0.5/cd*(mpdf->length+2.)*(1.-temp));	
			
			#endif
		}	
	
	}	
	
	//Just OpenMP version: copy the aux vectors to the PDF structure,
	//It is done when they are completely filled or the thread reaches the last sample
	#ifdef _OPENMP
	if((((i - initialSample) % chunk_size) == 0) || (i >= (max_sample_value)))
	{
		for(j = 0; j < densCounter; j++)
			#pragma omp atomic
			pdf->PDF[densPosition[j]] += densValues[j];
		densCounter = 0;
	}
	#endif   	  
  }
   
  //Delete memory structures created by threads  
  V_FREE(XX);
  V_FREE(PCdot);     
  
  #ifdef _OPENMP
  free(densValues);
  free(densPosition);      
  #endif

  //Apply k constant to PDF
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
  	  pdf->PDF[i] = pdf->PDF[i] * k;

  //Calculate integral of PDF
  #pragma omp single
  {
  for(i=0; i < pdf->total_size; i++)
      theintegral += pdf->PDF[i];

  theintegral = theintegral * dx[0];
  }        

  //Renormalize PDF using integral
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
     pdf->PDF[i] = pdf->PDF[i]/theintegral;
     
  //Calculate total volume of renormalized PDF   
  #pragma omp for reduction(+:total_vol)  
  for(i=0; i < pdf->total_size; i++)
	total_vol += pdf->PDF[i];
  
  }//End of parallel OpenMP Region    
  
  printf("Total integrated PDF: %g. The integral: %f\n",total_vol*dx[0],theintegral);	   
	  	
} 

//Compute the PDF of a 2D grid space
void computePDF2D(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , double detSm1 , double *x0, 
		double *x1, double *dx, double *bounds, MAT *eigenvectors )
{
  int i,j; //Loop variables
  int dim = 2;	//Dimensions of grid space
  double cd = volumeConstant(dim); //Volume constants to calculate kernel values    
  double h2=h*h;  //Squared bandwith value
  double *PC; // Current sample (PC space)
  double theintegral = 0.0;  
  double total_vol = 0.0;  
  double total_dx = dx[0] * dx[1]; 
  double * sample; 
  double k=1./sqrt(detSm1)/mpdf->current/pow(h,mpdf->length);  //Constant to recover the volume in the X space from the volume in the PC space
  VEC *XX;
  VEC *PCdot;
 
  //Variables to calculate coordinates and number of gridpoints of bounding box
  int steps;	  
  double upper, lower[2], gridpoint[2];
  int tot_ev_per_dim[2];
  size_t dif_pos[2];
  double abs_bound;

  //Auxiliary vectors for OpenMP version
  double * densValues;
  int * densPosition;    
  int densCounter = 0;    

  //Calculations required for aux vector in OpenMP version
  #ifdef _OPENMP

  int chunk_size; //Number of samples per data chunk
  int box_size = ((ceil(bounds[0] / dx[0]) * 2) + 2) * ((ceil(bounds[1] / dx[1]) * 2) + 2); //Number of gridpoints in a bounding box
  long available_memory = (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) - (pdf->total_size * sizeof(double)); //Available memory (bytes) to create the aux vectors         
  int samples_per_thread,initialSample,max_sample_value;         
              
  #endif            
              
  #pragma omp parallel default(none) firstprivate(densCounter) \
  shared(mpdf,pdf,dim,x0,x1,dx,total_dx,theintegral,total_vol,available_memory,box_size,bounds,eigenvectors,cd,h2,k) \
  private(i,j,sample,PC,lower,upper,steps,abs_bound,tot_ev_per_dim,chunk_size,samples_per_thread,initialSample, \
  max_sample_value,dif_pos,gridpoint,XX,PCdot,densValues,densPosition) 
  {	
 
  //Structures for holding each sample
  XX=v_get(mpdf->length);
  assert(XX);
  PCdot=v_get(mpdf->length);
  assert(PCdot);	

  #ifdef _OPENMP 

  chunk_size = available_memory / (omp_get_num_threads() * box_size * (sizeof(double) + sizeof(int))); //Number of samples in each aux vector

  densValues = (double *)malloc(sizeof(double) * chunk_size * box_size); //Vector to hold density values of each sample-gridpoint combination
  densPosition = (int *)malloc(sizeof(int) * chunk_size * box_size); //Vector to hold the positions of densValues values in the PDF structure
  
  samples_per_thread = mpdf->current / omp_get_num_threads(); //Number of samples to be processed by each OpenMP thread 
  initialSample = (samples_per_thread * omp_get_thread_num()); //First "i" value to be taken by each OpenMP thread
  max_sample_value = initialSample + samples_per_thread - 1; //Last "i" value to be taken by each OpenMP thread

  #endif   
 
  //Initialize PDF structure to 0s
  #pragma omp for
  for(i = 0; i < pdf->total_size; i++)
	pdf->PDF[i] = 0.0f;

  //Main calculation loop. For each sample calculate the PDF of its influence area and store in the PDF structure
  #pragma omp for
  for(i=0;i<mpdf->current;i++) 
  {	  
	sample = MPDFPosition(mpdf,i); //Get current sample
	PC = MPDFPCPosition(mpdf,i); //Get current sample (scaled as PC)

	//For each sample, calculate its bounding box, 
	//expressed as coordinates of lower corner and number of gridpoints per dimensions
	for(j = 0; j < 2; j++)
	{	
		//Lower corner
		abs_bound = sample[j] - bounds[j];
		if (x0[j] > abs_bound)
			lower[j] = x0[j];
		else
		{
			steps = floor((abs_bound - x0[j]) / dx[j]);
			lower[j] = x0[j] + (steps * dx[j]);
		}

		//Upper corner
		abs_bound = sample[j] + bounds[j];
		if (x1[j] < abs_bound)
			upper = x1[j];
		else
		{
			steps = ceil((abs_bound - x0[j]) / dx[j]);
			upper = x0[j] + (steps * dx[j]);
		}	
		
		//Calculate number of eval points per dimension	
		tot_ev_per_dim[j] = rint((upper - lower[j])/dx[j]) + 1;			
	}    

	//Calculate the PDF of the defined 2D box
	compute2DBox(mpdf,pdf,XX,PCdot,PC,lower,tot_ev_per_dim,gridpoint,dif_pos,x0,dx,dim,h2,cd,eigenvectors,&densCounter,densValues,densPosition);
	
	//Just OpenMP version: copy the aux vectors to the PDF structure,
	//It is done when they are completely filled or the thread reaches the last sample
	#ifdef _OPENMP
	if((((i - initialSample) % chunk_size) == 0) || (i >= (max_sample_value)))
	{
		for(j = 0; j < densCounter; j++)
			#pragma omp atomic
			pdf->PDF[densPosition[j]] += densValues[j];
		densCounter = 0;
	}	
    #endif
  }
   
 //Delete memory structures created by threads  
  V_FREE(XX);
  V_FREE(PCdot);     
  
  #ifdef _OPENMP
  free(densValues);
  free(densPosition);      
  #endif

  //Apply k constant to PDF
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
  	  pdf->PDF[i] = pdf->PDF[i] * k;

  //Calculate integral of PDF
  #pragma omp single
  {
  for(i=0; i < pdf->total_size; i++)
      theintegral += pdf->PDF[i];

  theintegral = theintegral * total_dx;
  }        

  //Renormalize PDF using integral
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
     pdf->PDF[i] = pdf->PDF[i]/theintegral;
     
  //Calculate total volume of renormalized PDF   
  #pragma omp for reduction(+:total_vol)  
  for(i=0; i < pdf->total_size; i++)
	total_vol += pdf->PDF[i];
  
  }//End of parallel OpenMP Region    
  
  printf("Total integrated PDF: %g. The integral: %f\n",total_vol*dx[0]*dx[1],theintegral);
	
}

#define DEBUG_TEMPS 1
#undef  DEBUG_TEMPS

//Compute the PDF of grid spaces of dimension 3 or higher
void computePDFND(MPDFEstimatorPtr mpdf, PDFPtr pdf, MAT *Sm1 , double h , 
		  double detSm1 , double *x0, double *x1, double *dx, 
		  double *bounds, MAT *eigenvectors, int dim)
{
  int i,j,l,m,u,w; //Loop variables	
  double cd = volumeConstant(dim); //Volume constant
  double k=1./sqrt(detSm1)/mpdf->current/pow(h,mpdf->length); //Constant to recover the volume in the X space from the volume in the PC space  
  double h2=h*h; //Square of bandwith value
  double *PC; // Current sample (PC space)
  double total_vol=0.0;
  double theintegral=0.0;
  double * sample; //Current sample
  VEC *XX;
  VEC *PCdot;
  
  //Variables to calculate the bounding box of a sample
  double * lower;
  double upper;
  double * gridpoint;
  int * tot_ev_per_dim;
  size_t * dif_pos;
  int slices,boxes;	
  int steps;
  double abs_bound; //Absolute bound per sample and dimension, given by ellipsoid shape

  //Calculate acumulated volume for the grid space
  double total_dx = 1.0;
  for (i = 0; i < dim; i++)
	total_dx *= dx[i];   

  //Apply BW effect to inverse covariance matrix
  for(i = 0; i < dim; i++)
	for(j = 0; j < dim; j++)
        	Sm1->me[i][j] /= h2;

  //Variables to perform the calculation of the 2D layering
  double A,B,C,F,Z,theta,cosTheta,sinTheta,X2,Y2,X,Y,XY,termY2,valor,termX2,upy,rightx,upx_rot,upy_rot,rightx_rot,righty_rot; 
  double bound[2],box_center[2],box_min[2],box_max[2],box_steps[2],box_upper[2];
      			
  //Calculate partial equations for the 2D layering    							
  A = Sm1->me[0][0];
  B = 2 * Sm1->me[0][1];
  C = Sm1->me[1][1];
  theta = atan(B/(A-C))/2;		 
  cosTheta = cos(theta);
  sinTheta = sin(theta);					
  X2 =    Sm1->me[0][0]*cosTheta*cosTheta + 2*Sm1->me[0][1]*cosTheta*sinTheta +   Sm1->me[1][1]*sinTheta*sinTheta;
  XY = -2*Sm1->me[0][0]*cosTheta*sinTheta + 2*Sm1->me[0][1]*cosTheta*cosTheta - 2*Sm1->me[0][1]*sinTheta*sinTheta + 2*Sm1->me[1][1]*cosTheta*sinTheta;
  Y2 =    Sm1->me[0][0]*sinTheta*sinTheta - 2*Sm1->me[0][1]*cosTheta*sinTheta +   Sm1->me[1][1]*cosTheta*cosTheta;		
  
  //Aux vector for OpenMP version
  double * densValues; 
  int * densPosition;
  int densCounter = 0;
  int box_size = 1; //Max number of gridpoints in a bounding box
  
  //Only for OpenMP version. Calculate the available memory in the system
  #ifdef _OPENMP
  int chunk_size; //Number of samples per data chunk
  long available_memory = (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) - (pdf->total_size * sizeof(double)); //Available memory (bytes) to create the aux vectors   
  
  for(i = 0; i < dim; i++)
	box_size *= ((ceil(bounds[i] / dx[i]) * 2) + 2);  
  
  #endif    
      
  //Beginning of OpenMP parallel region								
  #pragma omp parallel default(none) firstprivate(densCounter) \
  shared(available_memory,theintegral,total_vol,total_dx,k,mpdf,pdf,cd,dim,bounds,x0,x1,dx,Sm1,cosTheta,sinTheta,eigenvectors,X2,XY,Y2,h2,h,box_size) \
  private(chunk_size,i,j,l,m,u,w,sample,PC,gridpoint,slices,boxes,abs_bound,lower,box_upper,tot_ev_per_dim,box_steps,F,X,Y,Z,termX2,termY2, \
  upy,rightx,upx_rot,upy_rot,valor,rightx_rot,righty_rot,bound,box_center,box_min,box_max,XX,PCdot,dif_pos,steps,upper,densValues,densPosition)
  {			
			
  //Structures for holding each sample			
  XX=v_get(mpdf->length);
  assert(XX);
  PCdot=v_get(mpdf->length);
  assert(PCdot);				
			
  //Allocate variables to calculate the bounding box of a sample			
  lower = (double *)malloc(sizeof(double) * dim);
  gridpoint = (double *)malloc(sizeof(double) * dim);
  tot_ev_per_dim = (int *)malloc(sizeof(int) * dim);
  dif_pos = (size_t *)malloc(sizeof(size_t) * dim);			

  #ifdef _OPENMP
  chunk_size = available_memory / (omp_get_num_threads() * box_size * (sizeof(double) + sizeof(int))); //Number of samples in each aux vector

  densValues = (double *)malloc(sizeof(double) * chunk_size * box_size); //Vector to hold density values of each sample-gridpoint combination
  densPosition = (int *)malloc(sizeof(int) * chunk_size * box_size); //Vector to hold the positions of densValues values in the PDF structure
  
  int samples_per_thread = mpdf->current / omp_get_num_threads(); //Number of samples to be processed by each OpenMP thread 
  int initialSample = (samples_per_thread * omp_get_thread_num()); //First "i" value to be taken by each OpenMP thread
  int max_sample_value = initialSample + samples_per_thread - 1; //Last "i" value to be taken by each OpenMP thread
  #endif

  //Initialize PDF structure to 0s
  #pragma omp for 
  for(i = 0; i < pdf->total_size; i++)
	pdf->PDF[i] = 0.0f;

  //Main calculation loop. For each sample calculate the PDF of its influence area and store in the PDF structure
  #pragma omp for  
  for(i=0;i<mpdf->current;i++) 
  {	  
	sample = MPDFPosition(mpdf,i); //Get current sample
	PC = MPDFPCPosition(mpdf,i); //X is the current sample (scaled as PC)

	//For each sample, calculate its bounding box, 
	//expressed as coordinates of lower corner and number of gridpoints per dimensions
	boxes = 1;			
	for(j = 3; j < dim; j++)
	{	
		//Lower corner
		abs_bound = sample[j] - bounds[j];
		if (x0[j] > abs_bound)
			lower[j] = x0[j];
		else
		{
			steps = ceil((abs_bound - x0[j]) / dx[j]);
			lower[j] = x0[j] + (steps * dx[j]);
		}

		//Upper corner
		abs_bound = sample[j] + bounds[j];
		if (x1[j] < abs_bound)
			upper = x1[j];
		else
		{
			steps = floor((abs_bound - x0[j]) / dx[j]);
			upper = x0[j] + (steps * dx[j]);
		}	
		
		//Calculate number of grid points per dimension	
		tot_ev_per_dim[j] = rint((upper - lower[j])/dx[j]) + 1;	
		boxes *= tot_ev_per_dim[j] ;				
	}		


	//Compute number of slices per box

	//Lower corner
	abs_bound = sample[2] - bounds[2];
	if (x0[2] > abs_bound)
		lower[2] = x0[2];
	else
	{
		steps = floor((abs_bound - x0[2]) / dx[2]);
		lower[2] = x0[2] + (steps * dx[2]);
	}

	//Upper corner
	abs_bound = sample[2] + bounds[2];
	if (x1[2] < abs_bound)
		upper = x1[2];
	else
	{
		steps = ceil((abs_bound - x0[2]) / dx[2]);
		upper = x0[2] + (steps * dx[2]);
	}	
	
	//Calculate number of grid points per dimension	
	tot_ev_per_dim[2] = rint((upper - lower[2])/dx[2]) + 1;	
	slices = tot_ev_per_dim[2] ;				
				
	//Traverse boxes
	for(m = 0; m < boxes; m++)
	{
		int divisor;
		int eval_point = m;
		for(u = 3; u < dim-1; u++)
		{			
			divisor = 1;
			for(w = u+1; w < dim; w++)
				divisor *= tot_ev_per_dim[w];
			
			gridpoint[u] = lower[u] + (dx[u] * (eval_point / divisor));			
			eval_point = eval_point % divisor;			
		}
		gridpoint[dim-1] = lower[dim-1] + (dx[dim-1] * eval_point); //Last case			
																		
		//Copy to structure for conversion to PC Space
		for(l = 3; l < dim; l++)
			XX->ve[l] = gridpoint[l];    
				    
		//Fill structure with gridpoint position                        
		for(l = 3; l < dim; l++)
			dif_pos[l] = (gridpoint[l] - x0[l])/ dx[l];

												
		//For each gridpoint in dimensions 3 to N			
		for(j = 0; j < slices; j++)
		{							
			//Calculate location of grid point
			gridpoint[2] = lower[2] + (dx[2] * j); //Last case			
			XX->ve[2] = gridpoint[2];    
			dif_pos[2] = (gridpoint[2] - x0[2])/ dx[2];

			/* This code calculates, a 2D plane formed by the first two dimensions of the space, the optimal
			 * box inside the initial bounding box */

			Z = gridpoint[2] - sample[2];

			//X,Y, along with X2,XY,Y2 form the equation of the 2D rotated plane

			F = Sm1->me[2][2] * Z * Z - 1;

			X  =  2*Sm1->me[0][2]*Z*cosTheta + 2*Sm1->me[1][2]*Z*sinTheta;
			Y  = -2*Sm1->me[0][2]*Z*sinTheta + 2*Sm1->me[1][2]*Z*cosTheta;

			//Calculate displacements and obtain formula (x-xo)^2 / a^2 + % (y-yo)^2/b^2 = 1
			
			termX2 = (X/X2)/2;
			termY2 = (Y/Y2)/2; 
			valor = -F + termX2*termX2*X2 + termY2*termY2*Y2;

			//Calculate new rotated bounding box. UP and RIGHT are the corners of the new bounding box

			upy = sqrt(1/(Y2/valor));
			rightx = sqrt(1/(X2/valor));
			
			upx_rot    =  0 * cosTheta + upy * sinTheta;
			upy_rot    = -0 * sinTheta + upy * cosTheta;
			rightx_rot =  rightx * cosTheta + 0 * sinTheta;
			righty_rot = -rightx * sinTheta + 0 * cosTheta;
					
			//Calculate original displacement (rotated ellipse)	
					
			box_center[0] = termX2*cosTheta-termY2*sinTheta;
			box_center[1] = termX2*sinTheta+termY2*cosTheta;
			
			bound[0] = sqrt(upx_rot*upx_rot+rightx_rot*rightx_rot);
			bound[1] = sqrt(upy_rot*upy_rot+righty_rot*righty_rot);
				
			//Calculate lower and upper bound of new BoundingBox	
			for(u = 0; u < 2; u++)
			{
				box_min[u] = (sample[u] - box_center[u]) - bound[u]; 
				box_steps[u] = floor((box_min[u] - x0[u]) / dx[u]);
				lower[u] = (x0[u] > box_min[u])?(x0[u]):(x0[u] + (box_steps[u] * dx[u]));

				box_max[u] = (sample[u] - box_center[u]) + bound[u]; 
				box_steps[u] = ceil((box_max[u] - x0[u]) / dx[u]); 
				box_upper[u] = (x1[u] < box_max[u])?(x1[u]):(x0[u] + (box_steps[u] * dx[u])); 

				tot_ev_per_dim[u] = rint((box_upper[u] - lower[u])/dx[u]);			
			}
		
		    	//Calculate the PDF of the defined 2D box
			compute2DBox(mpdf,pdf,XX,PCdot,PC,lower,tot_ev_per_dim,gridpoint,dif_pos,x0,dx,dim,h2,cd,eigenvectors,&densCounter,densValues,densPosition);
			
		}//End of "per slice" for

	}//End of "per box" for

	//Just OpenMP version: copy the aux vectors to the PDF structure,
	//It is done when they are completely filled or the thread reaches the last sample
	#ifdef _OPENMP
	if((((i - initialSample) % chunk_size) == 0) || (i >= (max_sample_value)))
	{
		for(u = 0; u < densCounter; u++)
			#pragma omp atomic
			pdf->PDF[densPosition[u]] += densValues[u];
		densCounter = 0;
	}
	#endif


  } //End of "per sample" for
    
  //Delete memory structures created by threads   
  V_FREE(XX);
  V_FREE(PCdot);  
  free(lower);	
  free(tot_ev_per_dim);				
  free(dif_pos);  
  free(gridpoint);    
  
  #ifdef _OPENMP
  free(densValues);
  free(densPosition);
  #endif

  //Apply k constant to PDF
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
  	  pdf->PDF[i] = pdf->PDF[i] * k;

  //Calculate integral of PDF
  #pragma omp single
  {
  for(i=0; i < pdf->total_size; i++)
      theintegral += pdf->PDF[i];

  theintegral = theintegral * total_dx;
  }
  
  //Renormalize PDF using integral
  #pragma omp for
  for(i=0; i < pdf->total_size; i++)
     pdf->PDF[i] = pdf->PDF[i]/theintegral;
  
  //Calculate total volume of renormalized PDF     
  #pragma omp for reduction(+:total_vol)  
  for(i=0; i < pdf->total_size; i++)
	total_vol += pdf->PDF[i];

  }//End of parallel OpenMP Region   
  
  printf("Total integrated PDF: %g. The integral: %f\n",total_vol*total_dx,theintegral);
}
