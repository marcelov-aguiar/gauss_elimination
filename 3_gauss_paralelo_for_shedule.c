#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int testLinearSystem(float *A, float *b, float *x, int n, int nS) {

	int i, j;
	for (i = 0; i < n; i++) {
		float sum = 0;
		for (j = 0; j < n; j++)
			sum += A[i * n + j] * x[j];
		if (abs(sum - b[i]) >= 0.001) {
			return 1;
		}
	}
	return 0;
}

void generateLinearSystem(int n, float *A, float *b, int nS) {

	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) //A[i * n + j] == A[i][j]
			A[i * n + j] = (1.0 * n + (rand() % n)) / (i + j + 1);
		A[i * n + i] = (10.0 * n) / (i + i + 1);
	}

	for (i = 0; i < n; i++)
		b[i] = 1.;
}

void solveLinearSystem(const float *A, const float *b, float *x, int n) {

	float *Acpy = (float *) malloc(n * n * sizeof(float));
	float *bcpy = (float *) malloc(n * sizeof(float));
	memcpy(Acpy, A, n * n * sizeof(float));
	memcpy(bcpy, b, n * sizeof(float));

	int i, j, count;
	/* Gaussian Elimination */
	for (i = 0; i < (n - 1); i++) { 
		#pragma omp parallel shared(n, i, Acpy, bcpy) private(j, count)
		{
			#pragma omp for schedule(dynamic)
			for (j = (i + 1); j < n; j++) { 
				float ratio = Acpy[j * n + i] / Acpy[i * n + i];
				for (count = i; count < n; count++) { 
					Acpy[j * n + count] -= (ratio * Acpy[i * n + count]); 
				}
				bcpy[j] -= (ratio * bcpy[i]);
			}
		}
	}

	/* Back-substitution */
	x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
	for (i = (n - 2); i >= 0; i--) { 
		float temp = bcpy[i]; 
		for (j = (i + 1); j < n; j++) { 
			temp -= (Acpy[i * n + j] * x[j]);
		}
		x[i] = temp / Acpy[i * n + i];
	}
}

int main(int argc, char **argv) {

	double start; 
    double end; 
	double t_generate;
	double t_solve;
	double t_test;

    struct timeval tstart, tend;

	int n, nS;
	scanf("%d %d", &n, &nS);

	int i, nerros = 0;

	float *A = (float *) malloc(nS * n * n * sizeof(float));
	float *b = (float *) malloc(nS * n * sizeof(float));
	float *x = (float *) malloc(nS * n * sizeof(float));

	gettimeofday(&tstart, NULL);
	for (i = 0; i < nS; i++)
		generateLinearSystem(n, &A[i * n * n], &b[i * n], nS);
	gettimeofday(&tend, NULL);
	t_generate = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
	printf("Generate: %fs\n",t_generate);

	gettimeofday(&tstart, NULL);
	for (i = 0; i < nS; i++)
		solveLinearSystem(&A[i * n * n], &b[i * n], &x[i * n], n);
	gettimeofday(&tend, NULL);

	t_solve = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
	printf("Solve: %fs\n", t_solve);

	gettimeofday(&tstart, NULL);
	for (i = 0; i < nS; i++)
		nerros += testLinearSystem(&A[i * n * n], &b[i * n], &x[i * n], n, nS);
	gettimeofday(&tend, NULL);

	t_test = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
	printf("Test: %fs\n", t_test);

	printf("Errors=%d\n", nerros);

	printf("Total time %fs\n", t_generate+t_solve+t_test);

	return EXIT_SUCCESS;
}

