/**	\file RecogTask.cpp
	\brief ���� � ����������� ������ ������ �������������
*/
#include "RecogTask.h"
#include "ReciverTask.h"


#include <deque>
#include "Poco/Mutex.h"

#include "Poco/File.h"
#include "Poco/Path.h"

#include "Poco/AutoPtr.h"
#include "Poco/Logger.h" 
#include "Poco/FileChannel.h"
#include "Poco/PatternFormatter.h"
#include "Poco/FormattingChannel.h"
#include "Poco/Message.h"
#include "Poco/Format.h" 



#include <io.h>
#include <time.h>

#include <sys\stat.h>
//#include <fstream>
//#include <iostream>


using Poco::Mutex;
using std::deque;



using Poco::AutoPtr;
using Poco::Logger;
using Poco::Channel;
using Poco::FileChannel;
using Poco::FormattingChannel;
using Poco::Formatter;
using Poco::PatternFormatter;

using namespace std;


///�������� ������, ���������� ���� �����������
faces_main facesDB[MAX_IMG_COUNT]; 

extern Mutex mut_new_data;
extern deque<MessageFromServo> deq;

/*****************************************************************************/
//loadBase

int RecogTask::loadBase()
{

    int rc;
    sqlite3 *db;
    sqlite3_stmt *stmt;
    char *sql = "select * from faces_main;";
    const char *tail;
	int i=0;

    rc = sqlite3_open("C:/Robohead/db/faces.db", &db);

    if(rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 0;
    }

    rc = sqlite3_prepare(db, sql, (int)strlen(sql), &stmt, &tail);

    if(rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
    }
    
    rc = sqlite3_step(stmt);

    while(rc == SQLITE_ROW) {
        
		facesDB[i].img_id = sqlite3_column_int(stmt, 0);
		sprintf_s(facesDB[i].name,64, "%s", sqlite3_column_text(stmt, 1));

		printf("%d\t%s\n",facesDB[i].img_id,facesDB[i].name);

        rc = sqlite3_step(stmt);
		i++;
    }

    sqlite3_finalize(stmt);

    sqlite3_close(db);

	return i; //return count

}

/*****************************************************************************/
// addFace
int RecogTask::addFace(char *name)
{
    sqlite3 *db;
    int rc;
	sqlite3_stmt *stmt;
	const char* tail;
    char *zErr;
	char* data;

    rc = sqlite3_open("C:/Robohead/db/faces.db", &db);

    if(rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return -1;
    }
    
    const char* sql = "insert into faces_main (img_id, name) values (NULL, ?100);";

    rc = sqlite3_prepare(db, sql, (int)strlen(sql), &stmt, &tail);

    if(rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare() : Error: %s\n", tail);
        return rc;
    }

    sqlite3_bind_text(stmt, 100, name, (int)strlen(name), SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    sqlite3_close(db);

	facesDB[nTrainFaces].img_id=nTrainFaces+1;
	sprintf_s(facesDB[nTrainFaces].name,64,"%s",name);

	nTrainFaces++;
	printf("Added new face, nTrainFaces=%d\n",nTrainFaces);

	return 0;
}

/*****************************************************************************/
//loadTrainingData

int RecogTask::loadTrainingData()
{
    CvFileStorage * fileStorage;
    int i;

    // create a file-storage interface
    fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
    if( !fileStorage )
    {
        fprintf(stderr, "Can't open facedata.xml\n");
        return 0;
    }

    nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
    nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	trainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
   // *pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
    eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
    projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
    pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
    eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
    for(i=0; i<nEigens; i++)
    {
        char varname[200];
        sprintf( varname, "eigenVect_%d", i );
        eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
    }

    // release the file-storage interface
    cvReleaseFileStorage( &fileStorage );

    return 1;

}

/*****************************************************************************/
//�������� ����������� �� ����

int RecogTask::loadFaceImgArray()
{
    char imgFilename[512];
    int iFace, nFaces=0;

	//load img base
	nFaces=loadBase();

    // allocate the face-image array and person number matrix
    faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
    personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

	char buf[128];

    // store the face images in an array
    for(iFace=0; iFace<nFaces; iFace++)
    {
        // read person number and name of image file
       // fscanf(imgListFile, "%d %s", personNumTruthMat->data.i+iFace, imgFilename);
		
		memcpy(personNumTruthMat->data.i+iFace,&facesDB[iFace].img_id,sizeof(int));
		//mas[iFace]=faces[iFace].img_id;//+++
		//personNumTruthMat->data.i+iFace=&(faces[iFace].img_id);
		//printf("personNumTruthMat=%d\n",personNumTruthMat->data.i[iFace]);

		sprintf_s(imgFilename,"c:/robohead/img/%s%d.bmp",facesDB[iFace].name,facesDB[iFace].img_id);

        // load the face image
		IplImage * img=cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

		faceImgArr[iFace]=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);

		cvEqualizeHist(img,faceImgArr[iFace]);

		cvReleaseImage(&img);

	//	sprintf_s(buf,128,"%d.bmp",iFace);
	//	cvSaveImage(buf,faceImgArr[iFace]);

	//	faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

        if( !faceImgArr[iFace] )
        {
            fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
            return 0;
        }
    }
	//memcpy(personNumTruthMat->data.i,mas,nFaces*sizeof(int));//+++

    return nFaces;
}

/*****************************************************************************/
//����� ���������� ������
int RecogTask::findNearestNeighbor(float* projectedTestFace)
{
    float leastDistSq = DBL_MAX;
    int i, iTrain, iNearest = 0;

	//printf("findNearestNeighbor\n");
    for(iTrain=0; iTrain<nTrainFaces; iTrain++) //�� ���� �����
    {
        float distSq=0;

        for(i=0; i<nEigens; i++)
        {
            float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
            distSq += d_i*d_i; // Euclidean
        }
//		printf("%.3f\n",distSq);

        if(distSq < leastDistSq)
        {
            leastDistSq = distSq;
            iNearest = iTrain;
        }

    }
//	printf("leastDistSq=%.3f\n",leastDistSq);

    if(rec_p_.face_thresh < leastDistSq) //������ ���� ������ ������
    {
        iNearest = -1;
    }

    return iNearest;
}

/*****************************************************************************/
//��������� ������ � facedata.xml
void RecogTask::storeTrainingData(void)
{
    CvFileStorage * fileStorage;
    int i;

    // create a file-storage interface
    fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

    // store all the data
    cvWriteInt( fileStorage, "nEigens", nEigens );
    cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
    cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
    cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
    cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
    cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));

    for(i=0; i<nEigens; i++)
    {
        char varname[200];
        sprintf( varname, "eigenVect_%d", i );
        cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
    }

    // release the file-storage interface
    cvReleaseFileStorage( &fileStorage );
}

/*****************************************************************************/
//doPCA, obtiene las caracteristicas principales del rostro
//coloca data normalizada en eigenValMat

void RecogTask::doPCA(void)
{
    int i;
    CvTermCriteria calcLimit;
    CvSize faceImgSize;

    // set the number of eigenvalues to use
    nEigens = nTrainFaces-1;

    // allocate the eigenvector images
    faceImgSize.width  = faceImgArr[0]->width;
    faceImgSize.height = faceImgArr[0]->height;
    eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);

    for(i=0; i<nEigens; i++) eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

    // allocate the eigenvalue array
    eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

    // allocate the averaged image
    pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

    // set the PCA termination criterion
    calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	//����� ������� ��������� (Principal Component Analysis, PCA) 
	//����������� ��� ������ ���������� ��� ������������ ������ ���������������.

    // compute average image, eigenvalues, and eigenvectors
	//nTrainFaces - ����� ��������, 
	//faceImgArr - ��������� �� ������ �����������-��������,
	//eigenVectArr � (����� �������) ��������� �� ������ ����������� �������� (����������� �������� 32 ���)
	//CV_EIGOBJ_NO_CALLBACK � ����� �����/������. ��� ������ � �������.
	//0-������ ������. ��� ������ � �������.
	//0-��������� �� ��������� ��� ������ � �������.
	//calcLimit-�������� ����������� ����������. ��� ��������: �� ���������� �������� � �� �� ��������
	//pAvgTrainImg-(����� �������) ����������� ����������� ��������
	//eigenValMat->data.fl-��������� �� ����������� ����� (����� ���� NULL)

    cvCalcEigenObjects(
        nTrainFaces,
        (void*)faceImgArr,
        (void*)eigenVectArr,
        CV_EIGOBJ_NO_CALLBACK,
        0,
        0,
        &calcLimit,
        pAvgTrainImg,
        eigenValMat->data.fl);

    cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

/*****************************************************************************/
//recognize
int RecogTask::recognize(void)
{
    int iNearest, nearest, truth;
    int i, nTestFaces  = 1;         // the number of test images
    float * projectedTestFace = 0;

    // project the test images onto the PCA subspace
    projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

//	printf("Start rec\n");

        // project the test image onto the PCA subspace
        cvEigenDecomposite(
            testImg,//faceImgArr[i],
            nEigens,
            eigenVectArr,
            0, 0,
            pAvgTrainImg,
            projectedTestFace);

        iNearest = findNearestNeighbor(projectedTestFace);
        truth    = personNumTruthMat->data.i[0];
        if(iNearest!= -1) nearest  = trainPersonNumMat->data.i[iNearest];
        else nearest = -1;

    return nearest;
}

/*****************************************************************************/
/** @brief learn, se encarga de sacar PCA al nuevo rostro grabado y pasa el control
  *        a StoreTrainingData.
  */
void RecogTask::learn(void)
{
     int i, offset;

    // load training data
    nTrainFaces = loadFaceImgArray();
	// nTrainFaces = loadBase();
	printf("nTrainFaces=%d\n",nTrainFaces);

    if( nTrainFaces < 2 )
    {
        fprintf(stderr, "Need 2 or more training faces\n" "Input file contains only %d\n", nTrainFaces);
        return;
    }

    // do PCA on the training faces
    doPCA();
	printf("PCA end\n");

    // project the training images onto the PCA subspace
    projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
    offset = projectedTrainFaceMat->step / sizeof(float);
    for(i=0; i<nTrainFaces; i++)
    {
        cvEigenDecomposite(
            faceImgArr[i],
            nEigens,
            eigenVectArr,
            0, 0,
            pAvgTrainImg,
            projectedTrainFaceMat->data.fl + i*offset);
    }

    // store the recognition data as an xml file
    storeTrainingData();
}

/*****************************************************************************/

void RecogTask::SendNewFace(int id, int x, int y)
{
 MessageFromServo msg;

 DatagramSocket send_s;
 SocketAddress sa("localhost", 11002);
 send_s.connect(sa);
 msg.id=id;
 msg.x=x;
 msg.y=y;
 send_s.sendBytes(&msg, sizeof(msg));
 send_s.close();
}

/*****************************************************************************/
//��������� ����������� ���� � �����
void RecogTask::detectFaces(IplImage* img, int ident)
{
    int i,j;
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH;	// Only search for 1 face.
    //CvSeq *faces = cvHaarDetectObjects(img, cascade_f, storage,	1.1, 5, 0, cvSize(130, 130));
	CvSeq *faces = cvHaarDetectObjects(img, cascade_f, storage,	1.1, 3, flags, cvSize(130, 130));

    if (faces->total == 0) return;


    CvRect *r = (CvRect*)cvGetSeqElem(faces, 0);
    cvRectangle(img, cvPoint(r->x, r->y),
                cvPoint(r->x + r->width, r->y + r->height),
                CV_RGB(255, 0, 0), 1, 8, 0);

    //int channels  = img->nChannels;
	int channels  = 3;
    IplImage* img1 = cvCreateImage(cvSize(r->height, r->width), img->depth, channels);

    int a, b, k;

    for (i = r->y, a = 0; i < r->y + r->height, a < r->height; ++i, ++a)
    {
        for (j = r->x, b = 0; j < r->x + r->width, b < r->width; ++j, ++b)
        {
            for ( k = 0; k < channels; ++k)
            {
                CvScalar tempo = cvGet2D(img, i, j);
                img1->imageData[a * img1->widthStep + b * channels + k] = (char)tempo.val[k];
            }
        }
    }


    char cadena[64];
	char imgFilename[128];
    int response; 

    if (ident>0)     // ���������� ������ ����
    {

		loadBase(); //��������� ����
	//	printf("base read ok\n");

		CvSize size = cvSize(IMG_WIDTH,IMG_HEIGHT);
	//	printf("size ok\n");
        IplImage* tmpsize = cvCreateImage(size,img->depth, channels);
	//	printf("create ok\n");
        cvResize(img1,tmpsize,CV_INTER_LINEAR);
	//	printf("resize ok\n");

		sprintf_s(imgFilename,128,"c:/robohead/img/%s%d.bmp",facesDB[ident-1].name,facesDB[ident-1].img_id);
		printf("save new img %s\n",imgFilename);

        cvSaveImage(imgFilename, tmpsize);
        learn(); //����������
		loadTrainingData();

		cvReleaseImage(&tmpsize);
    }


// ������������ ����

	int xf=0;
	int yf=0;

	xf=cvRound((double)r->width*0.5)+r->x;
	yf=cvRound((double)r->height*0.5)+r->y;
	xf=(100.0*(double)xf)/img->width;
	yf=(100.0*(double)yf)/img->height;


	CvSize size = cvSize(IMG_WIDTH, IMG_HEIGHT);
	IplImage *sizedImg=cvCreateImage(size,img->depth, channels);
	cvResize(img1, sizedImg, CV_INTER_LINEAR); //��������� ���� ��� ������
				
	IplImage *grayImg=cvCreateImage(size, IPL_DEPTH_8U,1);

	cvCvtColor(sizedImg,grayImg,CV_BGR2GRAY);
				
	//if (testImg==NULL) testImg = cvCreateImage(size, IPL_DEPTH_8U,1); //���������� � �����

	cvEqualizeHist(grayImg, testImg);

	cvReleaseImage(&sizedImg);
	cvReleaseImage(&grayImg);

	cvReleaseImage(&img1);

	response = recognize();

    sprintf_s(cadena, 64, "Unknown");

	if (response>0) sprintf_s(cadena, 64, "%s", facesDB[response-1].name);

    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
    cvPutText(img, cadena, cvPoint(r->x, r->y+r->height/2), &font, cvScalar(255, 255, 255, 0));

//	printf("Send %d, %d, %d\n",response,xf,yf);
	SendNewFace(response,xf,yf);

// ������������ ����

}

/*****************************************************************************/
void  RecogTask::run()
{ 


printf("Start recog\n");
 //printf("Start th%d\n",thNum);

    // Inicializando variables
    faceImgArr            = 0; // array of face images
    personNumTruthMat     = 0; // array of person numbers
    nTrainFaces           = 0; // the number of training images
    nEigens               = 0; // the number of eigenvalues
    pAvgTrainImg          = 0; // the average image
    eigenVectArr          = 0; // eigenvectors
    eigenValMat           = 0; // eigenvalues
    projectedTrainFaceMat = 0; // projected training faces
	trainPersonNumMat	  = 0;

    char *file1 = "haarcascade_frontalface_alt.xml";
    cascade_f = (CvHaarClassifierCascade*)cvLoad(file1, 0, 0, 0);
    storage = cvCreateMemStorage(0);

 

MessageFromServo fromDeq; //��������� ��� �������������, ���������� �� ����
fromDeq.id=0; //�����

     //CvCapture* capture = cvCreateCameraCapture(0);  // Inicio la captura por camara
	CvCapture* capture = cvCaptureFromCAM(1);

//	cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,320);
//	cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,240);

    IplImage* frame;
    cvNamedWindow( "CaptureWindow", CV_WINDOW_NORMAL );   // �������� ���� �������
    char key;

    cout << "   > > > > > > > > > >   M E N U   < < < < < < < < < " << endl << endl;
 //   cout << "   'l'   Learn new face "                         << endl;
 //   cout << "   'r'   Recognize Faces"                         << endl;
 //   cout << "   'e'   Reinicialize Process"                            << endl;
    cout << "  'ESC'  exit"                                        << endl;

	learn(); //����������
	loadTrainingData();

	testImg = cvCreateImage(cvSize(IMG_WIDTH,IMG_HEIGHT), IPL_DEPTH_8U,1);

//	writer = cvCreateVideoWriter("faces.avi", -1, 20, cvSize(640, 480), 1);
//    if (writer!=0) printf("writer OK\n");

	if( capture )
    {
		printf("Capture ready\n");
		while (true)
		{
			//frame = cvQueryFrame( capture );            // asigno la captura a frame

			if( !cvGrabFrame( capture )) break;
            frame = cvRetrieveFrame( capture );

			if( !frame ) {printf("Error frame\n");break;}           // Si no logro capturar nada, break
			
 
	 
			mut_new_data.lock();	
			if(!deq.empty()){  //��� �� ����

				fromDeq=deq.front(); //����� ������ �� ����
				deq.pop_front(); 
			}
			mut_new_data.unlock();


			key = cvWaitKey(1);                        // capturo la tecla

	//		if( key == 'l' ) fromDeq.id=10;

//			float t = (float)cvGetTickCount();

		//	printf("id=%d\n",fromDeq.id);
			detectFaces(frame, fromDeq.id);                    // LLamo a la funcion detectFaces

			fromDeq.id=0; //�����

//			cvWriteFrame(writer, frame);

//			float t1 = (float)cvGetTickCount() - t;
//			printf("< %.1f >\n", t1/(cvGetTickFrequency()*1000.));

			cvShowImage("CaptureWindow", frame);    // Muestro la captura en la ventana con marco al rostro si existe

			if( key == 27 )  break;                     // ESC = exit
            
		}//while (true)
	}

//	cvReleaseVideoWriter(&writer);

    cvReleaseCapture( &capture );
    cvDestroyWindow( "CaptureWindow" );

}