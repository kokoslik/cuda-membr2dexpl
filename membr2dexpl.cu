#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
//#define WRITE_TO_FILE
using namespace std;
typedef double(*func2)(double,double);
typedef double(*func3)(double,double,double);
//Обработчик ошибок
static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( error ) (HandleError( error, __FILE__, __LINE__ ))

#define PSI(x,y) 0.0
#define F(x,y,t) 0.0
#define PHI(x,y) __sinf(M_PI*x)*__sinf(M_PI*y)

__global__ void first_layer_kernel(double *U,double *Uprev,double tau,double a, int N1n,int N2n, double h1, double h2/*,func2 phi,func2 psi,func3 f*/)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int i=tid/N2n+1;
    int j=1+tid%(N2n-2);
    if(i*N2n+j<(N1n-1)*N2n-1)
        U[i*N2n+j]=Uprev[i*N2n+j]+tau*PSI(i*h1,j*h2)+
                tau*tau*0.5*F(i*h1,j*h2,0.0)+
                a*a*tau*tau*0.5*((PHI((i+1)*h1,j*h2)-2.0*PHI(i*h1,j*h2)+PHI((i-1)*h1,j*h2))/(h1*h1)+(PHI(i*h1,(j+1)*h2)-2.0*PHI(i*h1,j*h2)+PHI(i*h1,(j-1)*h2))/(h2*h2));

}

__global__ void main_kernel(double *U,double *Uprev,double *Unext,double tau,double a,double t, int N1n,int N2n, double h1, double h2/*,func2 phi,func2 psi,func3 f*/)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    int i=tid/N2n+1;
    int j=1+tid%(N2n-2);
    if(i*N2n+j<(N1n-1)*N2n-1)
        Unext[i*N2n+j]=2.0*U[i*N2n+j]-Uprev[i*N2n+j]+a*a*tau*tau*((U[(i+1)*N2n+j]-2.0*U[i*N2n+j]+U[(i-1)*N2n+j])/(h1*h1)+(U[i*N2n+(j+1)]-2.0*U[i*N2n+j]+U[i*N2n+(j-1)])/(h2*h2))+F(i*h1,j*h2,t);
}

double solveGPU(double a,double L1,double L2,double T,double tau,int N1,int N2,func2 phi,func2 psi,func3 f)
{
    double *Unext,*U,*Uprev,*Uloc;
    double h1=L1/N1,h2=L2/N2;
    int N1n=N1+1,N2n=N2+1;
    double t=tau;
    float gputime=0.0;
    size_t size=N1n*N2n*sizeof(double);
    dim3 threads(1024,1,1),blocks((N1-1)*(N2-1)%1024==0?(N1-1)*(N2-1)/1024:(N1-1)*(N2-1)/1024+1,1,1);
    Uloc=new double[N1n*N2n];
    HANDLE_ERROR( cudaMalloc(&U,size) );
    HANDLE_ERROR( cudaMalloc(&Unext,size) );
    HANDLE_ERROR( cudaMalloc(&Uprev,size) );
#ifdef WRITE_TO_FILE
    ofstream ofile("../membr2dexpl/datagpu.dat");
    ofile.precision(16);
#endif
    //Нулевой временной слой
    for(int i=0;i<N1n;i++)
    {
        for(int j=0;j<N2n;j++)
        {
            Uloc[i*N2n+j]=phi(i*h1,j*h2);
#ifdef WRITE_TO_FILE
            ofile<<Uloc[i*N2n+j]<<' ';
#endif
        }
#ifdef WRITE_TO_FILE
        ofile<<endl;
#endif
    }
#ifdef WRITE_TO_FILE
    ofile<<endl;
    ofile<<endl;
#endif
    HANDLE_ERROR( cudaMemcpy(Uprev,Uloc,size,cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(U,Uprev,size,cudaMemcpyDeviceToDevice) );
    HANDLE_ERROR( cudaMemcpy(Unext,Uprev,size,cudaMemcpyDeviceToDevice) );
    //Первый временной слой
    cudaEvent_t start,stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start) );
    first_layer_kernel<<<blocks,threads>>>(U,Uprev,tau,a,N1n,N2n,h1,h2/*,phi,psi,f*/);
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
#ifdef WRITE_TO_FILE
    HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
    for(int i=0;i<N1n;i++)
    {
        for(int j=0;j<N2n;j++)
            ofile<<Uloc[i*N2n+j]<<' ';
        ofile<<endl;
    }
    ofile<<endl;
    ofile<<endl;
#endif
    //Основной цикл
    while(t<T-0.5*tau)
    {
        main_kernel<<<blocks,threads>>>(U,Uprev,Unext,tau,a,t,N1n,N2n,h1,h2/*,phi,psi,f*/);
        HANDLE_ERROR( cudaGetLastError() );
        HANDLE_ERROR( cudaDeviceSynchronize() );
#ifdef WRITE_TO_FILE
        HANDLE_ERROR( cudaMemcpy(Uloc,Unext,size,cudaMemcpyDeviceToHost) );
        for(int i=0;i<N1n;i++)
        {
            for(int j=0;j<N2n;j++)
                ofile<<Uloc[i*N2n+j]<<' ';
            ofile<<endl;
        }
        ofile<<endl;
        ofile<<endl;
#endif

        t+=tau;
        swap(U,Unext);
        swap(Uprev,Unext);
    }
    HANDLE_ERROR( cudaMemcpy(Uloc,Unext,size,cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaEventRecord(stop) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&gputime,start,stop) );
    HANDLE_ERROR( cudaFree(U) );
    HANDLE_ERROR( cudaFree(Unext) );
    HANDLE_ERROR( cudaFree(Uprev) );
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );
    delete[] Uloc;
#ifdef WRITE_TO_FILE
    ofile.close();
#endif
    return (double)gputime/1000.0;
}

double solveCPU(double a,double L1,double L2,double T,double tau,int N1,int N2,func2 phi,func2 psi,func3 f)
{
    double *Unext,*U,*Uprev;
    double h1=L1/N1,h2=L2/N2;
    int N1n=N1+1,N2n=N2+1;
    double t=tau;
    double cputime=0.0;
    U=new double[N1n*N2n];
    Unext=new double[N1n*N2n];
    Uprev=new double[N1n*N2n];
#ifdef WRITE_TO_FILE
    ofstream ofile("../membr2dexpl/datacpu.dat");
    ofile.precision(16);
#endif
    //Нулевой временной слой
    for(int i=0;i<N1n;i++)
    {
        for(int j=0;j<N2n;j++)
        {
            Uprev[i*N2n+j]=phi(i*h1,j*h2);
#ifdef WRITE_TO_FILE
            ofile<<Uprev[i*N2n+j]<<' ';
#endif
        }
#ifdef WRITE_TO_FILE
        ofile<<endl;
#endif
    }
#ifdef WRITE_TO_FILE
    ofile<<endl;
    ofile<<endl;
#endif
    //Первый временной слой
    cputime-=(double)clock();
    for(int i=0;i<N1n;i++)
    {
        for(int j=0;j<N2n;j++)
        {
            if((i==0)||(j==0)||(i==N1)||(j==N2))
            {
                U[i*N2n+j]=Uprev[i*N2n+j];
                Unext[i*N2n+j]=Uprev[i*N2n+j];
            }
            else
            {
                U[i*N2n+j]=Uprev[i*N2n+j]+tau*psi(i*h1,j*h2)+
                        tau*tau*0.5*f(i*h1,j*h2,0.0)+
                        a*a*tau*tau*0.5*((phi((i+1)*h1,j*h2)-2.0*phi(i*h1,j*h2)+phi((i-1)*h1,j*h2))/(h1*h1)+(phi(i*h1,(j+1)*h2)-2.0*phi(i*h1,j*h2)+phi(i*h1,(j-1)*h2))/(h2*h2));
            }
#ifdef WRITE_TO_FILE
            ofile<<U[i*N2n+j]<<' ';
#endif
        }
#ifdef WRITE_TO_FILE
        ofile<<endl;
#endif
    }
#ifdef WRITE_TO_FILE
    ofile<<endl;
    ofile<<endl;
#endif
    //Основной цикл
    while(t<T-0.5*tau)
    {
        for(int i=1;i<N1n-1;i++)
            for(int j=1;j<N2n-1;j++)
                Unext[i*N2n+j]=2.0*U[i*N2n+j]-Uprev[i*N2n+j]+a*a*tau*tau*((U[(i+1)*N2n+j]-2.0*U[i*N2n+j]+U[(i-1)*N2n+j])/(h1*h1)+(U[i*N2n+(j+1)]-2.0*U[i*N2n+j]+U[i*N2n+(j-1)])/(h2*h2))+f(i*h1,j*h2,t);
#ifdef WRITE_TO_FILE
        for(int i=0;i<N1n;i++)
        {
            for(int j=0;j<N2n;j++)
                ofile<<Unext[i*N2n+j]<<' ';
            ofile<<endl;
        }
        ofile<<endl;
        ofile<<endl;
#endif

        t+=tau;
        swap(U,Unext);
        swap(Uprev,Unext);
    }
    cputime+=(double)clock();
    cputime/=(double)CLOCKS_PER_SEC;
    delete[] U;
    delete[] Unext;
    delete[] Uprev;
#ifdef WRITE_TO_FILE
    ofile.close();
#endif
    return cputime;
}

__host__ __device__ double zero3(double a,double b,double c)
{
    return 0.0;
}
__host__ __device__  double zero2(double a,double b)
{
    return 0.0;
}

__host__ __device__ double init(double x, double y)
{
    return sin(M_PI*x)*sin(M_PI*y);
}
__host__ __device__ double init2(double x, double y)
{
    double sigma=0.1;
    return exp(-((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5))/2.0/sigma/sigma);
}

int main(int argc, char *argv[])
{
    double cputime,gputime;
    cputime=solveCPU(1.0,1.0,1.0,0.1,0.0001,1000,1000,init,zero2,zero3);
    cout<<"CPU time: "<<cputime<<endl;
    gputime=solveGPU(1.0,1.0,1.0,0.1,0.0001,1000,1000,init,zero2,zero3);
    cout<<"GPU time: "<<gputime<<endl;
    cout<<"Ratio: "<<cputime/gputime<<endl;
    return 0;
}
