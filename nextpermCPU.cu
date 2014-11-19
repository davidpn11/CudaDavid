#include <stddef.h>
#include <stdio.h>
#include "LIS.cu"
#include "LDS.cu"
#include "EnumaratorSequence.cu"
#include <time.h>

//#define NUM_THREADS 1024
#define THREAD_PER_BLOCK 128
#define N 16

__device__
void printVector(char* array, int length){
	for(int k = 0; k < length; k++){
		printf("%d - ",array[k]);	
	}
	printf("\n");
}

__device__
void inversion(char* vet, int length){
	char temp;
	for(int i = 0; i < length/2; i++){
		temp = vet[length-i-1];
		vet[length-i-1] = vet[i];
		vet[i] = temp;
	}	
}

__device__
void rotation(char *array, int length){
  char temp;
  int i;
  temp = array[0];
  for (i = 0; i < length-1; i++)
     array[i] = array[i+1];
  array[i] = temp;
}

unsigned long fatorialHost(unsigned long n){
	int i;
	unsigned long result = 1;
	for(i = n; i > 1; i--){
		result *= i;
	}
	return result;
}

//Calcula o LIS de todo o conjunto R partindo do pivor principal da ordem lexico gráfica
//Caso encontre um valor que é menor do que o máximo local de S, então ele retorna e não faz os outros calculos.
__global__
void decideLS(char* d_lMax_S, int length, unsigned long maxSeq, int numThreads){
	extern __shared__ char s_vet[];
	int tid = threadIdx.x + blockIdx.x*blockDim.x; 	
	int s_index = length*threadIdx.x; //Indice da shared memory
	unsigned long int indexSeq = tid;

	//Esses dois vetores são utilizados no LIS e no LDS, são declarados do lado de fora para
	//gastar menos memória e não ter necessidade de dar malloc.
	char MP[N*(N+1)/2]; //Vetor de most promising
	char last[N]; //Vetor de last de MP
	//Valores com os resultados encontrados no LIS e no LDS
	char lLIS, lLDS;
	char lMin_R;
	bool flagFinalLoop;
	while(indexSeq < maxSeq){
		getSequence(s_vet + s_index, length, indexSeq);
		lMin_R = 20; //Variavel que representa o min encontrado no conjunto R
		flagFinalLoop = true;
		for(int i = 0; i < length; i++){ //Rotação
			lLIS = LIS(s_vet + s_index, last, MP, length);

			//caso seja menor que o minimo do conjunto R, então modificar o valor
			if(lLIS < lMin_R){
				lMin_R = lLIS;	
			}
			//Todo o conjunto pode ser descartado, pois não vai subistituir lMax_S no resultado final
			if(lLIS <= d_lMax_S[tid]){
				flagFinalLoop = false;
				break;				
			}
	
			lLDS = LDS(s_vet + s_index, last, MP, length);

			//caso seja menor que o minimo do conjunto R, então modificar o valor
			if(lLDS < lMin_R){				
				lMin_R = lLDS;
			}
			//Todo o conjunto pode ser descartado, pois não vai subistituir lMax_S no resultado final
			if(lLDS <= d_lMax_S[tid]){
				flagFinalLoop = false;
				break;
			}
			rotation(s_vet + s_index, length);
		}
		//Caso o resultado final encontrado de R chegue ate o final, então significa que ele é maior
		//Que o minimo local encontrado até o momento.
		if(flagFinalLoop){
			d_lMax_S[tid] = lMin_R;
		}
		indexSeq += numThreads;
	}
}

//Com os valores de máximos locais de S, calcular o máximo global.
void calcLMaxGlobalS(char* lMax_globalS, char* lMax_localS, int tamVec){
	//Número de conjuntos
	for(int i = 0; i < tamVec; i++){
		if(*lMax_globalS < lMax_localS[i]){
			*lMax_globalS = lMax_localS[i];
		}
	}
}

//Seja S o conjunto de todas las sequencias dos n primeiros números naturais.
//Defina R(s), com s \in S o conjunto de todas as sequencias que podem
//ser geradas rotacionando S.
//Defina LIS(s) e LDS(s) como você sabe e sejam |LIS(s)| e |LDS(s)| suas
//cardinalidades.
//Determinar Max_{s \in S}(Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)))
int main(int argc, char *argv[]){
	//char* h_sequence;            //Vetor com a sequência pivor do grupo
	//char* h_threadSequences;      //Vetor com as sequências criadas
	char* d_lMax_localS;      //Vetor com os máximos locais de S, cada thread tem um máximo local
	char* h_lMax_localS;      

	int length = atoi(argv[1]);
	int NUM_THREADS = atoi(argv[2]);
	
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	clock_t start,end;

	//Aloca memória dos vetores	
	//h_sequence = (char*) malloc(length);
	//h_threadSequences = (char*) malloc(length*NUM_THREADS);
	h_lMax_localS = (char*) malloc(NUM_THREADS);
	//cudaMalloc(&d_threadSequences, length*NUM_THREADS);
	cudaMalloc(&d_lMax_localS, NUM_THREADS);
	cudaMemset(d_lMax_localS, 0, NUM_THREADS);

	start = clock();
	

	unsigned long numSeq = fatorialHost(length-1)/2;
	
	dim3 num_blocks(ceil(((float) NUM_THREADS)/(float) THREAD_PER_BLOCK));
	int tam_shared = length*THREAD_PER_BLOCK;

	//Cada thread calcula: Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)), e se o resultado for maior que o máximo local,
	//insere na variável
	decideLS<<<num_blocks, THREAD_PER_BLOCK,  tam_shared>>>
		   (d_lMax_localS, length, numSeq, NUM_THREADS);

	cudaMemcpy(h_lMax_localS, d_lMax_localS, NUM_THREADS, cudaMemcpyDeviceToHost);

	char lMax_globalS = 0; //Variável com o máximo global de S
	calcLMaxGlobalS(&lMax_globalS, h_lMax_localS, NUM_THREADS);	

	cudaThreadSynchronize();
	end = clock();

	printf("100%% - Tempo: %f s\n", (float)(end-start)/CLOCKS_PER_SEC);

	printf("Lmax R = %d\n",lMax_globalS);

	free(h_lMax_localS);
	//cudaFree(d_threadSequences);
	cudaFree(d_lMax_localS);
}

