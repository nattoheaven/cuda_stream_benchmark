/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.9 2009/04/11 16:35:00 mccalpin Exp $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright                                                             */
/* 2012: NISHIMURA Ryohei                                                */
/* 1991-2005: John D. McCalpin                                           */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>

/* INSTRUCTIONS:
 *
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of 
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

#ifndef N
#   define N	500000
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif
#ifndef OFFSET
#   define OFFSET	0
#endif

/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               nvcc -O stream.cu -o stream
 *
 *         This is known to work on NVIDIA GPUs.
 *
 *
 *	4) Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#define CUDA_SAFE_CALL(E) do {                                          \
    cudaError_t e = (E);                                                \
    if (e != cudaSuccess) {                                             \
      printf("line %d: CUDA error: %s\n", __LINE__, cudaGetErrorString(e)); \
      exit(-2);                                                         \
    }                                                                   \
  } while (false)

static __device__ float4	a[N+OFFSET],
				b[N+OFFSET],
				c[N+OFFSET];
static __device__ double	d_sum[3];

static double	avgtime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {
    2 * sizeof(float4) * N,
    2 * sizeof(float4) * N,
    3 * sizeof(float4) * N,
    3 * sizeof(float4) * N
    };

extern double mysecond();
extern void checkSTREAMresults();

static __global__ void STREAM_Init()
{
  int j = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (j < N) {
    a[j] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    b[j] = make_float4(2.0f, 2.0f, 2.0f, 2.0f);
    c[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

static __global__ void STREAM_Test()
{
  int j = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (j < N) {
    a[j] = make_float4(2.0E0f * a[j].x, 2.0E0f * a[j].y, 2.0E0f * a[j].z, 2.0E0f * a[j].y);
  }
}

static __global__ void STREAM_Copy()
{
  int j = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (j < N) {
    c[j] = a[j];
  }
}

static __global__ void STREAM_Scale(float scalar)
{
  int j = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (j < N) {
    b[j] = make_float4(scalar * c[j].x, scalar * c[j].y, scalar * c[j].z, scalar * c[j].w);
  }
}

static __global__ void STREAM_Add()
{
  int j = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (j < N) {
    c[j] = make_float4(a[j].x + b[j].x, a[j].y + b[j].y, a[j].z + b[j].z, a[j].w + b[j].w);
  }
}

static __global__ void STREAM_Triad(float scalar)
{
  int j = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (j < N) {
    a[j] = make_float4(b[j].x + scalar * c[j].x, b[j].y + scalar * c[j].y, b[j].z + scalar * c[j].z, b[j].w + scalar * c[j].w);
  }
}

static __device__ void STREAM_Sum_sub(double *sum, const float4 *a,
				      int j, double *shared)
{
  double x;
  if (j < N) {
    x = a[j].x;
  } else {
    x = 0.0;
  }
  shared[threadIdx.x] = x;
  int w = blockDim.x;
  do {
    int lastw = w;
    w = (w + 1) / 2;
    if (lastw > warpSize) {
      __syncthreads();
    }
    if (threadIdx.x + w < lastw) {
      x += shared[threadIdx.x + w];
      shared[threadIdx.x] = x;
    }
  } while (w != 1);
  if (threadIdx.x == 0) {
    unsigned long long int *address = (unsigned long long int *)sum;
    unsigned long long int old = *address, assumed;
    do {
      assumed = old;
      old = atomicCAS(address, assumed,
		      __double_as_longlong(x + __longlong_as_double(assumed)));
    } while (assumed != old);
  }
}

static __global__ void STREAM_Sum()
{
  int j = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) *
    blockDim.x + threadIdx.x;
  extern __shared__ double shared[];
  STREAM_Sum_sub(&d_sum[0], a, j, shared);
  STREAM_Sum_sub(&d_sum[1], b, j, shared);
  STREAM_Sum_sub(&d_sum[2], c, j, shared);
}

static dim3 grid, block;

int
main()
    {
    int			quantum, checktick();
    int			BytesPerWord;
    register int	j, k;
    float		scalar;
    double		t, times[4][NTIMES];

    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    printf("CUDA_STREAM based on STREAM version $Revision: 5.9 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(float4);
    printf("This system uses %d bytes per SINGLE PRECISION word.\n",
	BytesPerWord);

    printf(HLINE);
#ifdef NO_LONG_LONG
    printf("Array size = %d, Offset = %d\n" , N, OFFSET);
#else
    printf("Array size = %llu, Offset = %d\n", (unsigned long long) N, OFFSET);
#endif

    printf("Total memory required = %.1f MB.\n",
	(3.0 * BytesPerWord) * ( (double) N / 1048576.0));
    printf("Each test is run %d times, but only\n", NTIMES);
    printf("the *best* time for each is used.\n");

#ifdef IGPU
    CUDA_SAFE_CALL(cudaSetDevice(IGPU));
#endif
    {
	int k;
	int x, y, z;
	struct cudaDeviceProp prop;
	CUDA_SAFE_CALL(cudaGetDevice(&k));
	printf ("Ordinal of GPUs requested = %i\n",k);
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, k));
	block = dim3(prop.maxThreadsPerBlock);
	x = MIN((N + prop.maxThreadsPerBlock - 1) / prop.maxThreadsPerBlock,
		prop.maxGridSize[0]);
	y = MIN((N + x * prop.maxThreadsPerBlock - 1) /
		(x * prop.maxThreadsPerBlock),
		prop.maxGridSize[1]);
	z = MIN((N + y * x * prop.maxThreadsPerBlock - 1) /
		(y * x * prop.maxThreadsPerBlock),
		prop.maxGridSize[2]);
	grid = dim3(x, y, z);
    }

    printf(HLINE);

    /* Get initial value for system clock. */
    STREAM_Init<<<grid, block>>>();

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }

    t = mysecond();
    STREAM_Test<<<grid, block>>>();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0f;
    for (k=0; k<NTIMES; k++)
	{
	times[0][k] = mysecond();
        STREAM_Copy<<<grid, block>>>();
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	times[0][k] = mysecond() - times[0][k];
	
	times[1][k] = mysecond();
        STREAM_Scale<<<grid, block>>>(scalar);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	times[1][k] = mysecond() - times[1][k];
	
	times[2][k] = mysecond();
        STREAM_Add<<<grid, block>>>();
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	times[2][k] = mysecond() - times[2][k];
	
	times[3][k] = mysecond();
        STREAM_Triad<<<grid, block>>>(scalar);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	times[3][k] = mysecond() - times[3][k];
	}

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}
    
    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
	avgtime[j] = avgtime[j]/(double)(NTIMES-1);

	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults();
    printf(HLINE);

    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults ()
{
	double aj,bj,cj;
	float scalar;
	double h_sum[3];
	double epsilon;
	int	k;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0f;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }
	aj = aj * (double) (N);
	bj = bj * (double) (N);
	cj = cj * (double) (N);

	h_sum[0] = 0.0;
	h_sum[1] = 0.0;
	h_sum[2] = 0.0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_sum, h_sum, 3 * sizeof(double)));
	STREAM_Sum<<<grid, block, block.x * sizeof(double)>>>();
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_sum, d_sum, 3 * sizeof(double)));
#ifdef VERBOSE
	printf ("Results Comparison: \n");
	printf ("        Expected  : %f %f %f \n",aj,bj,cj);
	printf ("        Observed  : %f %f %f \n",h_sum[0],h_sum[1],h_sum[2]);
#endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
	epsilon = 1.e-8;

	if (abs(aj-h_sum[0])/h_sum[0] > epsilon) {
		printf ("Failed Validation on array a[]\n");
		printf ("        Expected  : %f \n",aj);
		printf ("        Observed  : %f \n",h_sum[0]);
	}
	else if (abs(bj-h_sum[1])/h_sum[1] > epsilon) {
		printf ("Failed Validation on array b[]\n");
		printf ("        Expected  : %f \n",bj);
		printf ("        Observed  : %f \n",h_sum[1]);
	}
	else if (abs(cj-h_sum[2])/h_sum[2] > epsilon) {
		printf ("Failed Validation on array c[]\n");
		printf ("        Expected  : %f \n",cj);
		printf ("        Observed  : %f \n",h_sum[2]);
	}
	else {
		printf ("Solution Validates\n");
	}
}
