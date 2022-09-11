#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <sys/time.h>
#include <stdbool.h>
#include <mpi.h>

int testLinearSystem(float *A, float *b, float *x, int n) {

	int i, j;
	float sum;
	for (i = 0; i < n; i++) {
		sum = 0;
		for (j = 0; j < n; j++)
			sum += A[i * n + j] * x[j];
		if (abs(sum - b[i]) >= 0.001) {
			return 1;
		}
	}
	return 0;
}

void generateLinearSystem(int n, float *A, float *b) {

	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) //A[i * n + j] == A[i][j]
			A[i * n + j] = (1.0 * n + (rand() % n)) / (i + j + 1);
		A[i * n + i] = (10.0 * n) / (i + i + 1);
	}

	for (i = 0; i < n; i++)
		b[i] = 1.;
}

int calc_fim(int n_bar, int n, int i, int p, int my_rank){
	int fim;
	if (((((n-1)-i) % p) != 0) && (my_rank == (p-1))){
		fim = n - 1;
	} else{
		fim = (n_bar * (my_rank + 1)) + i;
	}
	return fim;
}

int calc_inicio(int n_bar, int i, int my_rank){
	int inicio;
	inicio = ((n_bar * my_rank) + 1) + i;
	return inicio;
}

int main(int argc, char **argv) {
	int n_bar;
	int p, my_rank;
	MPI_Status status;
	int n;

	double tempo, send_time_total=0.0, t_aux=0.0;
	struct timeval tstart, tend, tsend_end, tsend_start;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

	int i, nerros = 0;
	
	if (my_rank == 0){
		// declaracao da dimensao da matriz (n)
		n = 5000;
	}

	// envia valor de n para todos os processos
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// ponteiros que armazenará a matriz sem a eliminação de Gauss
	float *A = (float *) malloc(n * n * sizeof(float));
	float *b = (float *) malloc(n * sizeof(float));
	float *x = (float *) malloc(n * sizeof(float));
	
	// processo 0 gera a matriz
	if (my_rank == 0){
		generateLinearSystem(n, &A[0], &b[0]);
	}
	
	// processo 0 envia as matrizes para todos os processos
	MPI_Bcast(A, n * n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(b, n, MPI_INT, 0, MPI_COMM_WORLD);

	// array local que armazenará a matriz transformada
	float *Acpy = (float *) malloc(n * n * sizeof(float));
	float *bcpy = (float *) malloc(n * sizeof(float));
	
	// processo 0 copia a matriz para matriz que será transformada
	if (my_rank == 0){
        memcpy(Acpy, A, n * n * sizeof(float));
	    memcpy(bcpy, b, n * sizeof(float));
    }
	int k, j, count;
	int inicio, fim;

	// inicia contagem do tempo
	if (my_rank == 0){
		gettimeofday(&tstart, NULL);
	}

	/* Gaussian Elimination */
	for (i = 0; i < (n - 1); i++) {
        n_bar = round(((n-1)-i) / p);

		/* processo 0 envia para cada processo a linha do pivo e
		as linhas que os processos irão fazer a eliminação de Gauss
		*/
        if (my_rank==0){
			gettimeofday(&tsend_start, NULL);
            for (k = 1; k < p; k++){
		    	inicio = calc_inicio(n_bar, i, k);
		    	fim = calc_fim(n_bar, n, i, p, k);
                MPI_Send(&Acpy[inicio*n], (fim-inicio+1)*n, MPI_FLOAT, k, 0, MPI_COMM_WORLD);
		    	MPI_Send(&bcpy[inicio], fim-inicio+1, MPI_FLOAT, k, 0, MPI_COMM_WORLD);
                // processo 0 envia linha do pivo
				MPI_Send(&Acpy[i*n], n, MPI_FLOAT, k, 0, MPI_COMM_WORLD);
		    	MPI_Send(&bcpy[i], 1, MPI_FLOAT, k, 0, MPI_COMM_WORLD);
    	    }
			gettimeofday(&tsend_end, NULL);
			send_time_total += (tsend_end.tv_sec - tsend_start.tv_sec) + (tsend_end.tv_usec - tsend_start.tv_usec) / 1000000.0;
        }else{
            inicio = calc_inicio(n_bar, i, my_rank);
		    fim = calc_fim(n_bar, n, i, p, my_rank);
            MPI_Recv(&Acpy[inicio*n], (fim-inicio+1)*n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&bcpy[inicio], fim-inicio+1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            // os processos recebem linha do pivo
			MPI_Recv(&Acpy[i*n], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&bcpy[i], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        }

		// recalcula o inicio o fim (número das linhas) de cada processo
		inicio = calc_inicio(n_bar, i, my_rank);
		fim = calc_fim(n_bar, n, i, p, my_rank);

		// inicia contagem do tempo para o calculo em si
		if (my_rank==0){
			gettimeofday(&tsend_start, NULL);
		}
		for (j = inicio; j <= fim; j++) { 
			float ratio = Acpy[j * n + i] / Acpy[i * n + i];
			for (count = i; count < n; count++) { 
				Acpy[j * n + count] -= (ratio * Acpy[i * n + count]); 
			}
			bcpy[j] -= (ratio * bcpy[i]);
		}
		if (my_rank==0){
			gettimeofday(&tsend_end, NULL);
			t_aux += (tsend_end.tv_sec - tsend_start.tv_sec) + (tsend_end.tv_usec - tsend_start.tv_usec) / 1000000.0;
		}
		if (my_rank != 0){
			// envio somente a parte calculada pelo respectivo processo
    	    MPI_Send(&Acpy[inicio*n], (fim-inicio+1)*n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&bcpy[inicio], fim-inicio+1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    	} else{
			// processo 0 recebe os valores calculados
			gettimeofday(&tsend_start, NULL);
    	    for (k = 1; k < p; k++){
				inicio = calc_inicio(n_bar, i, k);
				fim = calc_fim(n_bar, n, i, p, k);
    	        MPI_Recv(&Acpy[inicio*n], (fim-inicio+1)*n, MPI_FLOAT, k, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&bcpy[inicio], fim-inicio+1, MPI_FLOAT, k, 0, MPI_COMM_WORLD, &status);
    	    }
			gettimeofday(&tsend_end, NULL);
			send_time_total += (tsend_end.tv_sec - tsend_start.tv_sec) + (tsend_end.tv_usec - tsend_start.tv_usec) / 1000000.0;
    	}
		// sincronização da coluna i (pivô)
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (my_rank == 0) {
		gettimeofday(&tend, NULL);
	
		tempo = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
		printf("\nTotal Time: %fs\n",tempo);
		printf("Sending time: %fs\n", send_time_total);
		printf("Calculation time: %fs\n", t_aux);

		/* Back-substitution */
		x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
		for (i = (n - 2); i >= 0; i--) { 
			float temp = bcpy[i]; 
			for (j = (i + 1); j < n; j++) { 
				temp -= (Acpy[i * n + j] * x[j]);
			}
			x[i] = temp / Acpy[i * n + i];
		}

		nerros += testLinearSystem(&A[0], &b[0], &x[0], n);

		printf("Errors=%d\n", nerros);
	}
	MPI_Finalize();
	return EXIT_SUCCESS;
}

