#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <glm/gtx/transform.hpp>

#include "common/shader.hpp"
#include "common/texture.hpp"
#include "common/controls.hpp"
#include "common/objloader.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace glm;
using namespace cv;
using namespace std;

GLFWwindow* window;
VideoCapture *cap;
Mat descriptors_object;
std::vector<KeyPoint> keypoints_object;
std::vector<int> object_dims;

int initWindow(int width, int height) {
    // Initialise GLFW
    if( !glfwInit() ) {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return -1;
    }
    
    glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Open a window and create its OpenGL context    
    window = glfwCreateWindow( width, height, "Render object", NULL, NULL);
    if( window == NULL ) {
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    // Initialize GLEW
    glewExperimental=true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }
}

int initCamera(int& width, int& height) {
    // start video capture from camera
    cap = new cv::VideoCapture(0);
    // check that video is opened
    if ( cap == NULL || !cap->isOpened()) {
        fprintf( stderr, "could not start video capture\n" );
        return 1;
    }

    // get width and height
    int w = (int) cap->get( CV_CAP_PROP_FRAME_WIDTH );
    int h = (int) cap->get( CV_CAP_PROP_FRAME_HEIGHT );
    // On Linux, there is currently a bug in OpenCV that returns 
    // zero for both width and height here (at least for video from file)
    // hence the following override to global variable defaults: 
    width = w ? w : width;
    height = h ? h : height;
}

cv::Mat& getTranslationMatrix() {
    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, intrinsics, distortion, rvec, tvec);
    cv::Mat rotation, viewMatrix(4, 4, CV_64F);
    cv::Rodrigues(rvec, rotation);

    for(unsigned int row=0; row<3; ++row)
    {
       for(unsigned int col=0; col<3; ++col)
       {
          viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
       }
       viewMatrix.at<double>(row, 3) = tvec.at<double>(row, 0);
    }
    viewMatrix.at<double>(3, 3) = 1.0f;
    
    cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F); 
    cvToGl.at<double>(0, 0) = 1.0f; 
    cvToGl.at<double>(1, 1) = -1.0f; // Invert the y axis 
    cvToGl.at<double>(2, 2) = -1.0f; // invert the z axis 
    cvToGl.at<double>(3, 3) = 1.0f; 
    viewMatrix = cvToGl * viewMatrix;
    
    cv::Mat glViewMatrix = cv::Mat::zeros(4, 4, CV_64F);
    cv::transpose(viewMatrix , glViewMatrix);
    return glViewMatrix;
}

int initObject(int minHessian) {
    Mat obj_img = imread("../45Hand.jpg", IMREAD_GRAYSCALE);
    if(!obj_img.data) {
        fprintf(stderr, "Unable to open object file\n");
        return 1;
    }
    
    object_dims.push_back(obj_img.cols);
    object_dims.push_back(obj_img.rows);
    
    SurfFeatureDetector detector( minHessian );
    detector.detect( obj_img, keypoints_object );
    
    SurfDescriptorExtractor extractor;
    extractor.compute( obj_img, keypoints_object, descriptors_object );
}

GLuint createTexture(Mat& frame, int width, int height) {
    // Create one OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);
     
    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, textureID);
     
    // Give the image to OpenGL
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
     
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
	glGenerateMipmap(GL_TEXTURE_2D);
	return textureID;
}

bool detectFace(Mat& _img, Rect& r) {
    String haar_face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml";
    CascadeClassifier face_cascade;
    std::vector<Rect> faces;
    Mat img_gray;
    
    if( !face_cascade.load( haar_face_cascade_name ) ) { return false; }
    
    cvtColor( _img, img_gray, CV_BGR2GRAY );
    equalizeHist( img_gray, img_gray );
    face_cascade.detectMultiScale( img_gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(5,5));
    
    int i=faces.size()-1;
    if(faces.size() > 0) {
        r = faces[i];
        return true;
    }
    else {return false;}
}

void detectObject(Mat& image, int minHessian, Rect& r) {

    Mat img_scene;
    cvtColor(image, img_scene, CV_BGR2GRAY);
    equalizeHist(img_scene, img_scene);

    //-- Step 1: Detect the keypoints using SURF Detector
    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_scene;

    detector.detect( img_scene, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_scene;

    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ ) { 
        if( matches[i].distance < 3*min_dist ) {
            good_matches.push_back( matches[i]);
        }
    }

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ ) {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    std::vector<Point2f> scene_corners(4);
    obj_corners[0] = Point(0,0); obj_corners[1] = Point( object_dims[0], 0 );
    obj_corners[2] = Point( object_dims[0], object_dims[1] ); obj_corners[3] = Point( 0, object_dims[1] );
    
    perspectiveTransform( obj_corners, scene_corners, H);
    
    r = Rect(scene_corners[0],scene_corners[3]);
}

void drawCube() {
    std::vector< glm::vec3 > vertices;
    std::vector< glm::vec2 > uvs;
    std::vector< glm::vec3 > normals; // Won't be used at the moment.
    bool res = loadOBJ("../cube.obj", vertices, uvs, normals);

    GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);
	
	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	// 2nd attribute buffer : UVs
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glVertexAttribPointer(
		1,                                // attribute
		2,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	// Draw the triangle !
	glDrawArrays(GL_TRIANGLES, 0, vertices.size() );

	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
}

int main(int argc, char** argv) {
    
    int width = 640;
    int height = 480;
    initCamera(width, height);
    initWindow(width, height);
    //initObject(400);
      
    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    
    // Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);
    
    GLuint programID = LoadShaders( "../vertexShader.txt", "../fragmentShader.txt" );
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");
        
    // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
    
    // Camera matrix
    glm::mat4 View       = glm::lookAt(
        glm::vec3(0,0,5), // Camera is at (4,3,3), in World Space
        glm::vec3(0,0,0), // and looks at the origin
        glm::vec3(0,-1,0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
    // Model matrix : an identity matrix (model will be at the origin)
    glm::mat4 Model      = glm::mat4(1.0f);  // Changes for each model !
    glm::mat4 myScalingMatrix = glm::scale(2.5f, 2.5f ,2.5f);
    // Our ModelViewProjection : multiplication of our 3 matrices
    glm::mat4 MVP        = Projection * View * Model * myScalingMatrix;
    
    GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");
    
    // An array of 3 vectors which represents 3 vertices
    static const GLfloat g_vertex_buffer_data[] = {
         -1,-1,0,
         -1,1,0,
         1,-1,0,
         -1,1,0,
         1,1,0,
         1,-1,0
    };
    
    static const GLfloat g_uv_buffer_data[] = {
        0,0,
        0,1,
        1,0,
        0,1,
        1,1,
        1,0
    };
    
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
    
    GLuint uvbuffer;
    glGenBuffers(1, &uvbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);
    
    do{
        Mat image;
        Rect r;
        cap->read(image);
        //detectObject(image, 400, r);
        if(detectFace(image,r))
            rectangle(image, r, Scalar(0,255,0), 4);
        
        GLuint Texture = createTexture(image, image.cols, image.rows);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(programID);        
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
        
        glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Texture);
		glUniform1i(TextureID, 0);
         
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
           0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
           3,                  // size
           GL_FLOAT,           // type
           GL_FALSE,           // normalized?
           0,                  // stride
           (void*)0            // array buffer offset
        );
        
        // 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			2,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
         
        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, 2*3); // Starting from vertex 0; 3 vertices total -> 1 triangle
        
        Mat glViewMatrix = getTranslationMatrix();
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixd(&glViewMatrix.at<double>(0, 0));
        
        //drawCube();
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
                
        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0 );
    
    // Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteTextures(1, &TextureID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
    return 0;
}
