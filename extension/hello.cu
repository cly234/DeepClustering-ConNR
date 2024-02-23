#include<stdio.h>
#define N 10

__global__ void add(int *a,int *b,int *c){
	int t=blockIdx.x;
	if(t<N)
		c[t]=gridDim.x;
}

int main(){
	int a[N],b[N],c[N];
	int *a_cuda,*b_cuda,*c_cuda;
	//赋值
	for(int i=0;i<N;i++){
		a[i]=i-3;
		b[i]=i/2+1;
	}
	cudaMalloc((void**)&a_cuda,N*sizeof(int));
	cudaMalloc((void**)&b_cuda,N*sizeof(int));
	cudaMalloc((void**)&c_cuda,N*sizeof(int));
	cudaMemcpy(a_cuda,a,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(b_cuda,b,N*sizeof(int),cudaMemcpyHostToDevice);
	add<<<N, 1>>>(a_cuda,b_cuda,c_cuda);
	cudaMemcpy(c,c_cuda,N*sizeof(int),cudaMemcpyDeviceToHost);
	printf("a+b=(");
	for(int i=0;i<N;i++)
		printf("%d,",c[i]);
	printf(")\n");
	cudaFree(a_cuda);
	cudaFree(b_cuda);
	cudaFree(c_cuda);
}