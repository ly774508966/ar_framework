#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

VideoCapture cvCapture;
GLuint cameraImageTextureID;

GLuint initCamera() {
    // initialize 1st camera on the bus
    cvCapture.open(0);

    // initialze OpenGL texture		
    glEnable(GL_TEXTURE_RECTANGLE_ARB);

    glGenTextures(1, &cameraImageTextureID);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraImageTextureID);

    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

void displayFunc() {

    Mat newImage;
	cvCapture >> newImage;

	if(newImage.data) {
	    flip(newImage, newImage, 0);
		// clear the buffers
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glEnable(GL_TEXTURE_RECTANGLE_ARB);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0.0,(GLdouble)newImage.cols,0.0,(GLdouble)newImage.rows);	
		
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, cameraImageTextureID);

		if(newImage.channels() == 3)
			glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, newImage.cols, newImage.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, newImage.data);
		else if(newImage.channels() == 4)
			glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, newImage.cols, newImage.rows, 0, GL_BGRA, GL_UNSIGNED_BYTE, newImage.data);

		glBegin(GL_QUADS);
			glTexCoord2i(0,0);
			glVertex2i(0,0);
			glTexCoord2i(newImage.cols,0);
			glVertex2i(newImage.cols,0);
			glTexCoord2i(newImage.cols,newImage.rows);
			glVertex2i(newImage.cols,newImage.rows);
			glTexCoord2i(0,newImage.rows);
			glVertex2i(0,newImage.rows);
		glEnd();

	}

	glDisable(GL_TEXTURE_RECTANGLE_ARB);
	glutSwapBuffers();
}

void idleFunc(void) {
	glutPostRedisplay();
}

void reshapeFunc(int width, int height) {
	glViewport(0, 0, width, height);
}

static void InitializeGlutCallbacks()
{
    glutDisplayFunc(displayFunc);
    glutReshapeFunc(reshapeFunc);
	glutIdleFunc(idleFunc);
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Camera viewfinder");

    InitializeGlutCallbacks();    
    printf("GL version: %s\n", glGetString(GL_VERSION));
    initCamera();
	glutMainLoop();
	return 0;
}
