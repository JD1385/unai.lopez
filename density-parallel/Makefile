#######################

# Makefile used to build the set of programs used to derive environmental
# models' performance indices using multidimensional kernel-based
# probability density functions, as described by:

# Multi-objective environmental model evaluation by means of multidimensional 
# kernel density estimators: efficient and multi-core implementations, by 
# Unai Lopez-Novoa, Jon Sáenz, Alexander Mendiburu, Jose Miguel-Alonso, 
# Iñigo Errasti, Ganix Esnaola, Agustín Ezcurra, Gabriel Ibarra-Berastegi, 2014.

# Copyright (c) 2014, Unai Lopez-Novoa, Jon Saenz, Alexander Mendiburu 
# and Jose Miguel-Alonso  (from Universidad del Pais Vasco/Euskal 
# 		    Herriko Unibertsitatea)

# Refer to README.txt for more information

#######################

#### C COMPILER ####
CC=gcc

#### LIBRARY PATHS ####

#Output path for program binaries
BINDIR=bin

#Meschach library
MESCH_INC=/home/user/libs/mesch12b
MESCH_LIB=/home/user/libs/mesch12b/meschach.a

#NetCDF library
NETCDF_INC=/usr/local/include
NETCDF_LIB=/usr/local/lib

#######################

#### IMPORTANT ####

#The code has a serial and multi-threaded implementation.
#Choose which to use by uncommenting one of these flag sets.

#Flag set for multi-threaded implementation (Default)
DEBUG=-O2 -ftree-vectorize -msse2 -fopenmp

#Flag set for serial implementation
#DEBUG=-O2 -ftree-vectorize -msse2

####################

MOBJS = $(BINDIR)/mpdfestimator.o $(BINDIR)/MPDFEstimator.o \
	$(BINDIR)/mpdfncopers.o $(BINDIR)/parseargs.o $(BINDIR)/linalg.o \
	$(BINDIR)/boundaries.o $(BINDIR)/copycenter.o $(BINDIR)/PDF.o \
	$(BINDIR)/computePDF.o 

MBSOBJS = $(BINDIR)/mpdfestimator_bootstrap.o $(BINDIR)/MPDFEstimator.o \
	$(BINDIR)/parseargs.o $(BINDIR)/linalg.o $(BINDIR)/bootstrap.o \
	$(BINDIR)/boundaries.o $(BINDIR)/copycenter.o $(BINDIR)/PDF.o \
	$(BINDIR)/computePDF.o 	$(BINDIR)/genepsilon.o

all : $(BINDIR)/mpdfestimator  $(BINDIR)/mpdfestimator-bootstrap \
	$(BINDIR)/mpdf_score


XRANGE=-20/-15/-15:25/30/30:0.2/0.2/0.2
check :
	${BINDIR}/mpdfestimator -h 0.6 -x ${XRANGE} sample-data/sample-data

bscheck :
	${BINDIR}/mpdfestimator-bootstrap -h 0.6 -H 0.1/1/0.1\
		-x ${XRANGE} -n 5 sample-data/sample-data


CCFLAGS= ${DEBUG} -Wall -I${MESCH_INC} -I${NETCDF_INC}

$(BINDIR)/mpdf_score: score.c
	$(CC) -Wall ${DEBUG} -o $@ score.c -lm -L/usr/lib -lnetcdf -I${NETCDF_INC}

$(BINDIR)/mpdfestimator-bootstrap : $(MBSOBJS)
	$(CC) -Wall ${DEBUG} -o $@ $(MBSOBJS) -lm -L/usr/lib ${MESCH_LIB} 

$(BINDIR)/mpdfestimator : $(MOBJS) 
	$(CC) -Wall ${DEBUG} -o $@ $(MOBJS) -lm -L/usr/lib ${MESCH_LIB} -lnetcdf

$(BINDIR)/mpdfestimator.o : mpdfestimator_main.c mpdfncopers.h parseargs.h linalg.h copycenter.h PDF.h computePDF.h MPDFEstimator.h
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} mpdfestimator_main.c

$(BINDIR)/mpdfestimator_bootstrap.o : mpdfestimator_bootstrap.c parseargs.h linalg.h copycenter.h PDF.h computePDF.h genepsilon.h MPDFEstimator.h boundaries.h PDF.h computePDF.h bootstrap.h
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} mpdfestimator_bootstrap.c

$(BINDIR)/PDF.o : PDF.c PDF.h 
	$(CC) $(CCFLAGS) -c -o $@ PDF.c
$(BINDIR)/genepsilon.o : genepsilon.c genepsilon.h MPDFEstimator.h 
	$(CC) $(CCFLAGS) -c -o $@ genepsilon.c
$(BINDIR)/bootstrap.o : bootstrap.c bootstrap.h MPDFEstimator.h genepsilon.h\
			PDF.h
	$(CC) $(CCFLAGS) -c -o $@ bootstrap.c  -I${MESCH_INC}

$(BINDIR)/parseargs.o : parseargs.c parseargs.h 
	$(CC) $(CCFLAGS) -c -o $@ parseargs.c
$(BINDIR)/linalg.o : linalg.c linalg.h MPDFEstimator.h
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} linalg.c
$(BINDIR)/copycenter.o : copycenter.c copycenter.h MPDFEstimator.h
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} copycenter.c
$(BINDIR)/boundaries.o : boundaries.c boundaries.h 
	$(CC) $(CCFLAGS) -c -o $@ -I${MESCH_INC} boundaries.c

$(BINDIR)/MPDFEstimator.o : MPDFEstimator.c MPDFEstimator.h linalg.h
	$(CC) $(CCFLAGS) -c -o $@ MPDFEstimator.c

$(BINDIR)/computePDF.o : computePDF.c computePDF.h linalg.h MPDFEstimator.h PDF.h
	$(CC) $(CCFLAGS) -c -o $@ computePDF.c

$(BINDIR)/mpdfncopers.o : mpdfncopers.c mpdfncopers.h  
	$(CC) $(CCFLAGS) -c -o $@ mpdfncopers.c
clean :
	\rm -f $(BINDIR)/mpdfestimator $(BINDIR)/mpdfestimator-bootstrap $(BINDIR)/mpdf_score ${MOBJS} ${MBSOBJS}


