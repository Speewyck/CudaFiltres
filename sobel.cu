#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) / 1024;
  }
}

__global__ void sobel(unsigned char * g, std::size_t cols, std::size_t rows, unsigned char * sobel){

  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i > 1 && i < cols && j < rows && j > 1){

	// Horizontal
	auto h =     g[((j - 1) * cols + i - 1) ] -     g[((j - 1) * cols + i + 1) ]
	  + 2 * g[( j      * cols + i - 1) ] - 2 * g[( j      * cols + i + 1) ]
	  +     g[((j + 1) * cols + i - 1) ] -     g[((j + 1) * cols + i + 1) ];

	// Vertical
	auto v =     g[((j - 1) * cols + i - 1) ] -     g[((j + 1) * cols + i - 1) ]
	  + 2 * g[((j - 1) * cols + i    ) ] - 2 * g[((j + 1) * cols + i    ) ]
	  +     g[((j - 1) * cols + i + 1) ] -     g[((j + 1) * cols + i + 1) ];

	auto res = h*h + v*v;
	res = res > 65535 ? res = 65535 : res;

	sobel[ j * cols + i ] = sqrtf(res);
 }
}

int main()
{

  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  std::vector< unsigned char > g( rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g.data() );

  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * sobel_d;

  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &sobel_d, rows*cols);
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );

  grayscale<<< b, t >>>( rgb_d, g_d, cols, rows );
  sobel<<< b, t >>>(g_d, cols, rows, sobel_d);

  cudaMemcpy( g.data(), sobel_d, rows * cols, cudaMemcpyDeviceToHost );

  cv::imwrite( "out.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaFree( sobel_d);
  return 0;
}
