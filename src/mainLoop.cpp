// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"

// includes, project
#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
GLuint shader;

int imageW = 1024, imageH = 768;

#define PI 3.1415926535897932384626433832795
#define LENGTH 1024
#define HEIGHT 768
#define WIDTH 1024

int sh_size_v;
int sh_size_c;
int num_of_vectors;
int num_of_clusters;
int drawVectors;
int drawClusters;
int drawNextSet;
float* h_array_v;
float* h_array_c;
float* d_array_v;
float* d_array_c;
float delta;
float treshold;
//////////////////////////////
//camera
/////////////////////////////
// angle of rotation for the camera direction
float angle = 0.0;
// actual vector representing the camera's direction
float lx = 0.0f, lz = -1.0f;
// XZ position of the camera
float x = 0.0f, z = 5.0f;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int g_Kernel = 0;
bool g_FPS = false;
bool g_Diag = false;
StopWatchInterface *timer = NULL;

const int frameN = 24;
int frameCounter = 0;

#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0; // FPS count for averaging
int fpsLimit = 1; // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define REFRESH_DELAY 10 //ms
void displayCUDAInfo() {
	const int kb = 1024;
	const int mb = kb * kb;
	printf("CUDA INFO:\n=========\n\nCUDA version:   v%d\n", CUDART_VERSION);

	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Devices: \n\n");

	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("%d : %s:%d.%d\n", i, props.name, props.major, props.minor);
		printf("  Global memory:   %dmb\n", props.totalGlobalMem / mb);
		printf("  Shared memory:   %dkb\n", props.sharedMemPerBlock / kb);
		printf("  Constant memory: %dkb\n", props.totalConstMem / kb);
		printf("  Block registers: %d\n", props.regsPerBlock);

		printf("  Warp size:         %d\n", props.warpSize);
		printf("  Threads per block: %d\n", props.maxThreadsPerBlock);
		printf("  Max block dimensions: [ %d, %d, %d ]\n",
				props.maxThreadsDim[0], props.maxThreadsDim[1],
				props.maxThreadsDim[2]);
		printf("  Max grid dimensions:  [ %d, %d, %d ]\n\n=========\n\n",
				props.maxGridSize[0], props.maxGridSize[1],
				props.maxGridSize[2]);
	}
}

void computeFPS() {
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		delta = sdkGetAverageTimerValue(&timer) / 1000;
		sprintf(fps, "Sim: %3.1f fps \t delta :%f ", ifps, delta);

		glutSetWindowTitle(fps);
		fpsCount = 0;

//fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

void runSimulation(float threshold) {
	float changesFac = (float) num_of_vectors;
	float *hn_array_v;
	float* hn_array_c;
	hn_array_v = (float*) malloc(sh_size_v);
	hn_array_c = (float*) malloc(sh_size_c);

	switch (g_Kernel) {
	case 0:
		break;
	case 1:
		while (changesFac > threshold) {

			cuda_calculate(d_array_v, d_array_c, num_of_vectors,
					num_of_clusters);
			cuda_calculate_clusters(d_array_v, d_array_c, num_of_vectors,
					num_of_clusters);

			checkCudaErrors(
					cudaMemcpy(hn_array_v, d_array_v,
							num_of_vectors * 4 * sizeof(float),
							cudaMemcpyDeviceToHost));

			checkCudaErrors(
					cudaMemcpy(hn_array_c, d_array_c, sh_size_c,
							cudaMemcpyDeviceToHost));
			int counter = 0;
			for (int i = 0; i < num_of_vectors; i++) {
				int idx = i * 4;
				if (hn_array_v[idx + 3] != h_array_v[idx + 3]) {
					counter++;
					//printf("i = %d \thn_array_v[idx+3] = %f\th_array_v[idx+3] = %f\n", i,
//					/		hn_array_v[idx + 3],h_array_v[idx+3]);
				}
			}
			checkCudaErrors(
					cudaMemcpy(h_array_v, hn_array_v,
							num_of_vectors * 4 * sizeof(float),
							cudaMemcpyHostToHost));

			changesFac = (float) (counter * 1.0f / num_of_vectors * 1.0f);
			printf("counter = %d\t numof vectors  = %d\tchanges factor  = %f\n",
					counter, num_of_vectors, changesFac);

		}
		/*
		 printf("___________________________________after kernel usage\n");
		 for (int i = 0; i < num_of_vectors * 4; i += 4) {
		 printf("i : %d\t x : %f\t y : %f\t z : %f\t cluster  : %f\n", i / 4,
		 hn_array_v[i], hn_array_v[i + 1], hn_array_v[i + 2],
		 hn_array_v[i + 3]);
		 }
		 printf("*************************clusters***********************\n");

		 for (int i = 0; i < num_of_clusters * 4; i += 4) {
		 printf("host i : %d\t x : %f\t y : %f\t z : %f\t4  : %f\n", i / 4,
		 hn_array_c[i], hn_array_c[i + 1], hn_array_c[i + 2],
		 hn_array_c[i + 3]);
		 }
		 */
		break;
	}

	getLastCudaError("Filtering kernel execution failed.\n");
}

void displayFunc(void) {

	sdkStartTimer(&timer);
//TColor *d_dst = NULL;
	size_t num_bytes;

	if (frameCounter++ == 0) {
		sdkResetTimer(&timer);
	}

	runSimulation(treshold);

	{
		glClear(GL_COLOR_BUFFER_BIT);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA,
				GL_UNSIGNED_BYTE, BUFFER_DATA(0));
		glBegin(GL_TRIANGLES);
		glTexCoord2f(0, 0);
		glVertex2f(-1, -1);
		glTexCoord2f(2, 0);
		glVertex2f(+3, -1);
		glTexCoord2f(0, 2);
		glVertex2f(-1, +3);
		glEnd();
		glFinish();
	}

	float static radius = 5;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();

//Tells the camera where to be and where to look
//Format (camera position x,y,z, focal point x,y,z, camera orientation x,y,z)
//Remember that by default the camera points toward negative Z

	gluLookAt(x, 512.0f, z, x + lx, 512.0f, z + lz, 0.0f, 1.0f, 0.0f);

	//gluLookAt(100.0, 512.0, 2500.0, 512.0, 512.0, 0.0, 0.0, 1.0, 0.0);

	glPushMatrix();

//Set Drawing Color - Will Remain this color until otherwise specified
	//glColor3f(0.5, 0.0, 0.5); //Some type of blue

//Draw Circle
	glBegin(GL_POLYGON);
//Change the 6 to 12 to increase the steps (number of drawn points) for a smoother circle
//Note that anything above 24 will have little affect on the circles appearance
//Play with the numbers till you find the result you are looking for
//Value 1.5 - Draws Triangle
//Value 2 - Draws Square
//Value 3 - Draws Hexagon
//Value 4 - Draws Octagon
//Value 5 - Draws Decagon
//Notice the correlation between the value and the number of sides
//The number of sides is always twice the value given this range

//printf("drawing x : %f y : %f r : %f\n",h_array[0],h_array[1],h_array[4]);

	if (drawVectors == 1) {
		for (int i = 0; i < num_of_vectors * 4; i = i + 4) {
			if (drawNextSet != 0) {
				if (drawNextSet == h_array_v[i + 3]) {
					glColor3f(h_array_v[i + 3] / num_of_clusters, 0.0, 0.0);
					glBegin(GL_POLYGON);
					for (double j = 0; j < 2 * PI; j += PI / 12) //<-- Change this Value
						glVertex3f((GLfloat) (h_array_v[i] + cos(j) * radius),
								(GLfloat) (h_array_v[i + 1] + sin(j) * radius),
								h_array_v[i + 2]);
					glEnd();
				}
			} else {
				glColor3f(h_array_v[i + 3] / num_of_clusters, 0.0, 0.0);
				glBegin(GL_POLYGON);
				for (double j = 0; j < 2 * PI; j += PI / 12) //<-- Change this Value
					glVertex3f((GLfloat) (h_array_v[i] + cos(j) * radius),
							(GLfloat) (h_array_v[i + 1] + sin(j) * radius),
							h_array_v[i + 2]);
				glEnd();
			}
		}

	}

	if (drawClusters == 1) {
		if (drawNextSet != 0) {

			for (int i = 0; i < num_of_clusters * 4; i = i + 4) {
				if (drawNextSet == i) {
					glColor3f(0.0, i / num_of_clusters, 0.0);
					glBegin(GL_POLYGON);
					for (double j = 0; j < 2 * PI; j += PI / 12) //<-- Change this Value
						glVertex3f((GLfloat) (h_array_c[i] + cos(j) * radius),
								(GLfloat) (h_array_c[i + 1] + sin(j) * radius),
								h_array_c[i + 2]);
					glEnd();
				}
			}
		} else {
			for (int i = 0; i < num_of_clusters * 4; i = i + 4) {
				glColor3f(0.0, i / num_of_clusters, 0.0);
				glBegin(GL_POLYGON);
				for (double j = 0; j < 2 * PI; j += PI / 12) //<-- Change this Value
					glVertex3f((GLfloat) (h_array_c[i] + cos(j) * radius),
							(GLfloat) (h_array_c[i + 1] + sin(j) * radius),
							h_array_c[i + 2]);
				glEnd();
			}
		}
	}
//Draw Circle

	glPopMatrix();

	glPopMatrix();

	glFlush();
	if (frameCounter == frameN) {
		frameCounter = 0;

		if (g_FPS) {
			printf("FPS: %3.1f\n", frameN / (sdkGetTimerValue(&timer) * 0.001));
			g_FPS = false;
		}
	}

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value) {
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void shutDown(unsigned char k, int /*x*/, int /*y*/) {
	float fraction = 10.0f;
	switch (k) {
	case '\033':
	case 'Q':
		printf("Shutting down...\n");

		sdkStopTimer(&timer);
		sdkDeleteTimer(&timer);

		checkCudaErrors(CUDA_FreeArray());
		free(h_Src);

		exit(EXIT_SUCCESS);
		break;

	case 'w':
		x += lx * fraction;
		z += lz * fraction;
		break;

	case 's':
		x -= lx * fraction;
		z -= lz * fraction;
		break;

	case 'a':
		angle -= 0.01f;
		lx = sin(angle);
		lz = -cos(angle);

		break;

	case 'd':
		angle += 0.01f;
		lx = sin(angle);
		lz = -cos(angle);
		break;
	case 'e':
		if (drawVectors == 1)
			drawVectors = 0;
		else
			drawVectors = 1;

		break;
	case 'r':
		if (drawClusters == 1)
			drawClusters = 0;
		else
			drawClusters = 1;
		break;

	case 't':
	if (drawNextSet != 0) {
		if (drawNextSet + 1 > num_of_clusters)
			drawNextSet = 0;
		else
			drawNextSet++;
	} else
		drawNextSet++;
	break;
}
}

int initGL(int *argc, char **argv) {
	printf("Initializing GLUT...\n");
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
	glutCreateWindow(argv[0]);
	printf("OpenGL window created.\n");

	glewInit();
	printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));

	if (!glewIsSupported(
			"GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, " OpenGL version 1.5\n");
		fprintf(stderr, " GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, " GL_ARB_pixel_buffer_object\n");
		fflush(stderr);
		return false;
	}

	return 0;
}

// shader for displaying floating-point texture
static const char *shader_code = "!!ARBfp1.0\n"
		"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
		"END";

GLuint compileASMShader(GLenum program_type, const char *code) {
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
			(GLsizei) strlen(code), (GLubyte *) code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1) {
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int) error_pos,
				error_string);
		return 0;
	}

	return program_id;
}

void initOpenGLBuffers() {
	printf("Creating GL texture...\n");
//	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, h_Src);
	printf("Texture created.\n");

	printf("Creating PBO...\n");

	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src,
			GL_STREAM_COPY);
//While a PBO is registered to CUDA, it can't be used
//as the destination for OpenGL drawing calls.
//But in our particular case OpenGL is only used
//to display the content of the PBO, specified by CUDA kernels,
//so we need to register/unregister it only once.
// DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(gl_PBO) );
	checkCudaErrors(
			cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
					cudaGraphicsMapFlagsWriteDiscard));
	GLenum gl_error = glGetError();

	if (gl_error != GL_NO_ERROR) {
		fprintf(stderr, "GL Error in file '%s' in line %d :\n", __FILE__,
				__LINE__);
		fprintf(stderr, "%s\n", gluErrorString(gl_error));
		exit(EXIT_FAILURE);
	}

	printf("PBO created.\n");

// load shader program
	shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void cleanup() {
	sdkDeleteTimer(&timer);

	glDeleteProgramsARB(1, &shader);
}

void init() {
//Tells OpenGL to check which objects are in front of other objects
//Otherwise OpenGL would draw the last object in front regardless
//of it's position in Z space
//Note: It is not necessary to enable this for a simple 2D circle
//but is good practice
	glEnable(GL_DEPTH_TEST);

//Tells OpenGL not to draw backfaces
//Backfaces are defined by vertex drawing order
//By default couter-clockwise drawing order specifies front faces
//Note: The circle is drawn counter-clockwise
//Note: It is not necessary to enable this for a simple 2D circle
//but is good practice
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (GLfloat) WIDTH / (GLfloat) HEIGHT, 0.1f, 100000.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

//Assign the clear screen color
//Format (Red, Green, Blue, Alpha)
//Values should remain normalized between 0 and 1
	glClearColor(1.0, 1.0, 1.0, 0.0);
}

void reshape(int w, int h) {
	glViewport(0, 0, (GLsizei) WIDTH, (GLsizei) HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (GLfloat) WIDTH / (GLfloat) HEIGHT, 0.1f, 100000.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

float RandomFloat(float a, float b) {
	float random = ((float) rand()) / (float) RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

int main(int argc, char **argv) {
	x = 100;
	z = 2500;
	drawNextSet = 0;
//float* array;
	drawVectors = 1;
	drawClusters = 1;
	pArgc = &argc;
	pArgv = argv;
	float threshold = 0;
	displayCUDAInfo();
	g_Kernel = 0;
	printf("starting...\n\n");

	printf("choose simulation : \n1 : draw 1 \n2 : none\n3 : none\n");
	int n = 1;
//scanf("%d", &n);
	switch (n) {
	case 1:
		threshold = 0.0f;
		num_of_vectors = 3000;
		num_of_clusters = 32;

		sh_size_v = sizeof(float) * num_of_vectors * 4;
		h_array_v = (float*) malloc(sh_size_v);
		sh_size_c = sizeof(float) * num_of_clusters * 4;
		h_array_c = (float*) malloc(sh_size_c);

		printf("number of vectors = %d \t sh_size_v = %d \n", num_of_vectors,
				sh_size_v);

		for (int i = 0; i < num_of_vectors * 4; i += 4) {
			h_array_v[i] = RandomFloat(0, LENGTH); 		//x
			h_array_v[i + 1] = RandomFloat(0, LENGTH); 		//y
			h_array_v[i + 2] = RandomFloat(0, LENGTH); 		//z
			h_array_v[i + 3] = 0;
			printf("i : %d\t x : %f\t y : %f\t z : %f\tcluster  = %f\n", i / 4,
					h_array_v[i], h_array_v[i + 1], h_array_v[i + 2],
					h_array_v[i + 3]);
		}

		printf("\n\n\nCLUSTERS!!!!!!\n\n\n");

		printf("number of clusters = %d \t sh_size_c = %d \n", num_of_clusters,
				sh_size_c);

		for (int i = 0; i < num_of_clusters * 4; i += 4) {
			if (i / 4 < num_of_vectors) {
				h_array_c[i] = h_array_v[i] + 1; 		//x
				h_array_c[i + 1] = h_array_v[i + 1]; 		//y
				h_array_c[i + 2] = h_array_v[i + 2]; 		//z
			} else {
				h_array_c[i] = RandomFloat(LENGTH, LENGTH); 		//x
				h_array_c[i + 1] = RandomFloat(LENGTH, LENGTH); 		//y
				h_array_c[i + 2] = RandomFloat(LENGTH, LENGTH); 		//z

			}
			h_array_c[i + 3] = 0;
			printf("i : %d\t x : %f\t y : %f\t z : %f\t clusters : %f\n", i / 4,
					h_array_c[i], h_array_c[i + 1], h_array_c[i + 2],
					h_array_c[i + 3]);
		}

		g_Kernel = 1;
		break;
	default:
		g_Kernel = 0;
		break;
	}

	initGL(&argc, argv);
	init();
	glutReshapeFunc(reshape);
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	initOpenGLBuffers();

//cudaGLSetGLDevice (gpuGetMaxGflopsDeviceId());
	/*
	 h_Src = (uchar4*) malloc(imageH * imageW * 4);
	 memset(h_Src, clearColorbit, imageH * imageW * 4);


	 checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));
	 */
//copying to device memory
	checkCudaErrors(cudaMalloc((void** ) &d_array_v, sh_size_v));
	checkCudaErrors(
			cudaMemcpy(d_array_v, h_array_v, sh_size_v,
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void** ) &d_array_c, sh_size_c));
	checkCudaErrors(
			cudaMemcpy(d_array_c, h_array_c, sh_size_c,
					cudaMemcpyHostToDevice));
//runSimulation(threshold);
	printf("Starting GLUT main loop...\n");

	glutDisplayFunc(displayFunc);
	glutKeyboardFunc(shutDown);

	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutMainLoop();

// cudaDeviceReset causes the driver to clean up all state. While
// not mandatory in normal operation, it is good practice.  It is also
// needed to ensure correct operation when the application is being
// profiled. Calling cudaDeviceReset causes all profile data to be
// flushed before the application exits
	cudaDeviceReset();
	return 0;
	exit(EXIT_SUCCESS);
}

