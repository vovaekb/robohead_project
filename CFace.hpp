/**
*
*   @Brief: Clase que implementa la funcionalidad de reconocimiento de rostros
*
*
**/

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include "opencv\cv.h"
#include "opencv\cvaux.h"
#include "opencv\highgui.h"

#include "sqlite3.h"

#include "Poco/Net/DatagramSocket.h"

using namespace std;

using namespace Poco::Net;
using Poco::Net::DatagramSocket;



#define MAX_IMG_COUNT 512	///<Максимальное кол-во изображений
#define FACE_THRESH 180000000.0	///<Порог распознавания

#define IMG_WIDTH 120
#define IMG_HEIGHT 120

//CvVideoWriter *writer = 0;

struct MessageToServo	
{

	int id;			///<идентификатор
	int x;			///<идентификатор
	int y;			///<идентификатор
};

///Структура, содержащая шаблон с определенным номером
struct faces_main
{
	int img_id;
	char name[64];
};

///Основной массив, содержащий базу изображений
faces_main facesDB[MAX_IMG_COUNT]; 




/**
*   @TODO: Actualmente todo el entorno es publico
*
**/
class CFace
{

public:

    //      MEMBERS       =======================
    IplImage ** faceImgArr        ; // array de rostros cargados desde el archivo train.txt
    CvMat    *  personNumTruthMat ; // array of person numbers
    int nTrainFaces               ; // the number of training images
    int nEigens                   ; // the number of eigenvalues
    IplImage * pAvgTrainImg       ; // the average image
    IplImage ** eigenVectArr      ; // eigenvectors
    CvMat * eigenValMat           ; // eigenvalues
    CvMat * projectedTrainFaceMat ; // projected training faces
    CvHaarClassifierCascade *cascade_f;
    CvMemStorage			*storage; // Almacenamiento de informacion
    int temporal                  ;  // variable para controlar aprendizaje o reiniciarlo
	IplImage* testImg;
	CvMat * trainPersonNumMat;
	MessageToServo msg;

    //      METHODS       ========================
    CFace();
    ~CFace();

    void    Execute_Capture(void);
    void    learn(void);
	int    recognize(void);
    void    doPCA(void);
    void    storeTrainingData(void);
    int     loadTrainingData();
    int     findNearestNeighbor(float* projectedTestFace);
    int     loadFaceImgArray();
    void    detectFaces(IplImage* img, char c);
	int	loadBase(); //загрузка базы изображений
	int addFace(char *name); //добавить новое лицо
	void SendNewFace(int id, int x, int y);
}; /// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


/** @brief loadBase
  *
  */
int CFace::loadBase()
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

/** @brief addFace
  *
  */
int CFace::addFace(char *name)
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

/** @brief loadTrainingData
  *
  */
int CFace::loadTrainingData()
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

/** @brief loadFaceImgArray, Загрузка изображений из базы
  *        
  *
  * @return numero de rostros en el archivo train.txt
  */
int CFace::loadFaceImgArray()
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

//Поиск ближайшего соседа
int CFace::findNearestNeighbor(float* projectedTestFace)
{
    float leastDistSq = DBL_MAX;
    int i, iTrain, iNearest = 0;

	//printf("findNearestNeighbor\n");
    for(iTrain=0; iTrain<nTrainFaces; iTrain++) //по всем лицам
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

    if(FACE_THRESH < leastDistSq) //должно быть меньше порога
    {
        iNearest = -1;
    }

    return iNearest;
}

//сохранить данные в facedata.xml
void CFace::storeTrainingData(void)
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

/** @brief doPCA, obtiene las caracteristicas principales del rostro
  *        coloca data normalizada en eigenValMat
  *
  */
void CFace::doPCA(void)
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

	//Метод главных компонент (Principal Component Analysis, PCA) 
	//применяется для сжатия информации без существенных потерь информативности.

    // compute average image, eigenvalues, and eigenvectors
	//nTrainFaces - число эталонов, 
	//faceImgArr - указатель на массив изображений-эталонов,
	//eigenVectArr – (выход функции) указатель на массив собственных объектов (изображения глубиной 32 бит)
	//CV_EIGOBJ_NO_CALLBACK – флаги ввода/вывода. Для работы с памятью.
	//0-размер буфера. Для работы с памятью.
	//0-указатель на структуру для работы с памятью.
	//calcLimit-критерий прекращения вычислений. Два варианта: по количеству итераций и по ко точности
	//pAvgTrainImg-(выход функции) усредненное изображение эталонов
	//eigenValMat->data.fl-указатель на собственные числа (может быть NULL)

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

/** @brief recognize
  *
  */
int CFace::recognize(void)
{
    int iNearest, nearest, truth;
    int i, nTestFaces  = 1;         // the number of test images
 //   CvMat * trainPersonNumMat = 0;  // the person numbers during training
    float * projectedTestFace = 0;

    // load test images and ground truth for person number
 //+++   nTestFaces = loadFaceImgArray("test.txt");
 //+++   printf("%d test faces loaded\n", nTestFaces);

    // load the saved training data
 //   if( !loadTrainingData( &trainPersonNumMat ) ) return 0;

    // project the test images onto the PCA subspace
    projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

  //  int* respNearest = new int[nTestFaces];
	int respNearest;

	printf("Start rec\n");

 //   for(i=0; i<nTestFaces; i++) //по всем лицам
   // {
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
   //     printf("iNearest=%d, nearest = %d, Truth = %d\n", iNearest,nearest, truth);
        respNearest =  nearest;
   // }
	
    return respNearest;
}

/** @brief learn, se encarga de sacar PCA al nuevo rostro grabado y pasa el control
  *        a StoreTrainingData.
  */
void CFace::learn(void)
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

/** @brief Constructor de la clase CFace
  *
  */
CFace::CFace()
{
    // Inicializando variables
    faceImgArr            = 0; // array of face images
    personNumTruthMat     = 0; // array of person numbers
    nTrainFaces           = 0; // the number of training images
    nEigens               = 0; // the number of eigenvalues
    pAvgTrainImg          = 0; // the average image
    eigenVectArr          = 0; // eigenvectors
    eigenValMat           = 0; // eigenvalues
    projectedTrainFaceMat = 0; // projected training faces
    temporal              = 0; // variable para controlar aprendizaje o reiniciarlo
	trainPersonNumMat	  = 0;

    char *file1 = "haarcascade_frontalface_alt.xml";
    cascade_f = (CvHaarClassifierCascade*)cvLoad(file1, 0, 0, 0);
    storage = cvCreateMemStorage(0);
}

/** @brief Захват с камеры
  *
  */
void CFace::Execute_Capture(void)
{
    //CvCapture* capture = cvCreateCameraCapture(0);  // Inicio la captura por camara
	CvCapture* capture = cvCaptureFromCAM(1);

    IplImage* frame;
    cvNamedWindow( "CaptureWindow", CV_WINDOW_AUTOSIZE );   // Основное Окно захвата
    char key;

    cout << "   > > > > > > > > > >   M E N U   < < < < < < < < < " << endl << endl;
    cout << "   's'   Learn new face "                         << endl;
    cout << "   'r'   Recognize Faces"                         << endl;
    cout << "   'e'   Reinicialize Process"                            << endl;
    cout << "  'ESC'  exit"                                        << endl;

	learn(); //дообучение
	loadTrainingData();
	temporal = 1;

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
			

			key = cvWaitKey(1);                        // capturo la tecla

			float t = (float)cvGetTickCount();

			detectFaces(frame, key);                    // LLamo a la funcion detectFaces

//			cvWriteFrame(writer, frame);

			float t1 = (float)cvGetTickCount() - t;
			printf("< %.1f >\n", t1/(cvGetTickFrequency()*1000.));

			cvShowImage("CaptureWindow", frame);    // Muestro la captura en la ventana con marco al rostro si existe

			if( key == 27 )  break;                     // ESC = exit
            
		}//while (true)
	}

//	cvReleaseVideoWriter(&writer);

    cvReleaseCapture( &capture );
    cvDestroyWindow( "Video" );
}


void CFace::SendNewFace(int id, int x, int y)
{
 DatagramSocket send_s;
 SocketAddress sa("localhost", 11002);
 send_s.connect(sa);
 msg.id=id;
 msg.x=x;
 msg.y=y;
 send_s.sendBytes(&msg, sizeof(msg));
 send_s.close();
}


/** @brief Реализует обнаружение лица в кадре
  *
  */
void CFace::detectFaces(IplImage* img, char key)
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
    int response; 
	//int *resp;
	char imgFilename[128];

    if (key == 's')     // s = добавить новое лицо
    {
        CvSize size = cvSize(IMG_WIDTH,IMG_HEIGHT);
        IplImage* tmpsize = cvCreateImage(size,img->depth, channels);
        cvResize(img1,tmpsize,CV_INTER_LINEAR);

        cout << "Enter new name: " << endl;
        cin >> cadena;
        
		addFace(cadena);

		sprintf_s(imgFilename,128,"c:/robohead/img/%s%d.bmp",facesDB[nTrainFaces-1].name,facesDB[nTrainFaces-1].img_id);
		printf("save new img %s\n",imgFilename);

        cvSaveImage(imgFilename, tmpsize);
        learn(); //дообучение
		loadTrainingData();

		cvReleaseImage(&tmpsize);
    }
	cvReleaseImage(&img1);
/*
    if (key == 'r')     // r = Режим распознавания лиц
    {
        temporal = 1;
    }

    if (key == 'e')     // e = tecla para reiniciar el proceso
    {
        temporal = 0;
    }*/

    if (temporal)   // распознавать лица
    {

		int maxheight=0;
		int bestface=-1;
		int xf=0;
		int yf=0;
        for (int nf = 0; nf < (faces ? faces->total : 0); ++nf) //по всем обнаруж. лицам
        {
            CvRect *r = (CvRect*)cvGetSeqElem(faces, nf); //отрисов. замеченные лица
            cvRectangle(img, cvPoint(r->x, r->y), cvPoint(r->x + r->width, r->y + r->height), CV_RGB(255, 0, 0), 1, 8, 0);

			xf=cvRound((double)r->width*0.5)+r->x;
			yf=cvRound((double)r->height*0.5)+r->y;

			xf=(100.0*(double)xf)/img->width;
			yf=(100.0*(double)yf)/img->height;


            //int channels  = img->nChannels;
            IplImage* img2 =cvCreateImage(cvSize(r->height, r->width), img->depth, channels);
            int a, b, k;

            for(i=r->y, a=0; i<r->y+r->height, a<r->height; ++i, ++a)
            {
                for(j=r->x, b=0; j<r->x+r->width, b<r->width; ++j, ++b)
                {
                    for(k=0; k<channels; k++)
                    {
                        CvScalar tempo = cvGet2D(img, i, j);
                        img2->imageData[a * img2->widthStep + b * channels + k] =
                            (char)tempo.val[k];
                    }
                }
            }
			//printf("rh=%d\n",r->height);

			if (maxheight<r->height) {
				bestface=nf;
				maxheight=r->height;

				CvSize size = cvSize(IMG_WIDTH, IMG_HEIGHT);
				IplImage *sizedImg=cvCreateImage(size,img->depth, channels);
				cvResize(img2, sizedImg, CV_INTER_LINEAR); //подгоняем лицо под размер
				
				IplImage *grayImg=cvCreateImage(size, IPL_DEPTH_8U,1);

				cvCvtColor(sizedImg,grayImg,CV_BGR2GRAY);
				
				//if (testImg==NULL) testImg = cvCreateImage(size, IPL_DEPTH_8U,1); //перегоняем в серый

				cvEqualizeHist(grayImg, testImg);

				cvReleaseImage(&sizedImg);
				cvReleaseImage(&grayImg);

			}

			cvReleaseImage(&img2);
        }//по всем обнаруж. лицам

		//printf("bestface=%d, maxheight=%d\n",bestface,maxheight);
       
		if (bestface>=0) {
			response = recognize();
		//	printf("resp=%d\n",response);
		//	cvReleaseImage(&testImg);
		}

        for(j = 0; j < faces->total; ++j) //по всем обнаруж. лицам
        {

            CvRect *r = (CvRect*)cvGetSeqElem(faces, j);

            sprintf_s(cadena, 64, "Unknown");

			if ((response>0)&&(j==bestface)) sprintf_s(cadena, 64, "%s", facesDB[response-1].name);

            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
            cvPutText(img, cadena, cvPoint(r->x, r->y+r->height/2), &font, cvScalar(255, 255, 255, 0));

			printf("Send %d, %d, %d\n",response,xf,yf);
			SendNewFace(response,xf,yf);
        }
	}//if (temporal)   // распознавать лица

}


/** @brief Destructor de la clase
  *
  */
CFace::~CFace()
{

}
