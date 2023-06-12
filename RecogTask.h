/**	\file RecogTask.h
	\brief ������������ ���� � ��������� ������ ������ �������������
*/

#ifndef RECOG_TASK
#define RECOG_TASK

#include <fstream>
#include <stdio.h>
#include <iostream>

#include "Poco/Runnable.h"
#include "Poco/Thread.h"
#include "Poco/Net/DatagramSocket.h"

#include "sqlite3.h"

#include "opencv/cv.h"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"

using std::string;
using Poco::Runnable;

using namespace Poco::Net;
using Poco::Net::DatagramSocket;


#define MAX_IMG_COUNT 512	///<������������ ���-�� �����������

#define IMG_WIDTH 120
#define IMG_HEIGHT 120

//CvVideoWriter *writer = 0;



///���������, ���������� ������ � ������������ �������
struct faces_main
{
	int img_id;
	char name[64];
};



/** \struct recog_pars	
       \brief ��������� ������������ 
 */
struct recog_pars
{

	double face_thresh;
/*
	int bort_num_height;	///<������� ������ ���� ������ � �������� �� �����
	int beam_num_height;	///<������� ������ ���� ������ � �������� �� �����
	int pl_num_height;		///<������� ������ ���� ������ � �������� �� ���������
	int openw;				///<������� ��������� � ������������ �����������
	int pl;					///<������� ��������� � ������������ ��������
	int cis;				///<������� ��������� � ������������ �������
	int box;				///<������� ��������� � ������������ ������ �������
	int other;				///<������� ��������� � ������������ ��������� ����� �������
	int beam;				///<������� ��������� � ������������ �����
	int hoppers;			///<������� ��������� ��������
	int open_and_box;		///<������� ��������� � ����������� � ������ (������������� ��� ���� ������)

	double bortt;	///<������� ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������ (����, ���� �����)
	double bortb;	///<������ ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������ (����, ���� �����)
	double beamt;	///<������� ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������(�����, ��� �����)
	double beamb;	///<������ ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������(�����, ��� �����)

	double bortt_dc;	///<������� ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������ (����, ���� �����)
	double bortb_dc;	///<������ ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������ (����, ���� �����)
	double beamt_dc;	///<������� ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������(�����, ��� �����)
	double beamb_dc;	///<������ ������� ������� ����� � ��������� �� ���������, � ������ ������� ������� ����� ������(�����, ��� �����)

	int templCount;			///<���������� ��������

	string host;			///<��� ��� IP-����� ����� �� ������� �������� ��
	string user;			///<��� ������������ ��� ���������� � ��
	string password;		///<������ ��� ���������� � ��
	string db_default;		///<���� ������ �� ���������
	int port;				///<���� ��� �������� � ��
	int sqliteBaseUsed;		///<������� ������������� ���� ������ SQLite
	string sqlitePath;		///<���� � �������� � ������ ������ SQLite
	int sqliteKeepTime;		///<������� ������� ������� �����

	int oneCadrForHighReliability; ///<���������� �� � ���� ������ ���� ���� � ������������ ������� c ������� �������� �������������(1-��, 0-���)
	int twoCadrsForLowReliability; ///<���������� �� � ���� ������ 2 ����� ��� ������, ������������� � ������ �������� �������������
	int allCadrsForNonRecognized; ///<���������� �� � ���� 4 ����� ��� ��������������� ������
	int inputOutput;			///<������ �� ������� ��������/�������� �������
	int inputDays;			///<�������� ������ ������� ������������ ���� �������� ������� �� ������� (� ����)
	int inputDir;			///<�������� ������, ����� ����������� �������� ������� ������� �������� �� ����������� (-1 - ������ ������, 1 - ����� �������)
	
	string img_save_path;	///<�������� ����������, ���� ����������� �����

	string cadr_application_ip;	///<IP-����� ��� ������ �������������
	int cadr_application_port;	///<����� ����� ��� ������ �������������

	int first_group[4];		///<������, ������� ��������� ������, �� ��� ������������ ������� ���������� ����� ������
*/
} ;

/**
	\class RecogTask
	\brief ����� ������ ��� �������������
*/
class  RecogTask : public Runnable
{
	recog_pars & rec_p_;

public:
	RecogTask:: RecogTask(recog_pars &cpr):rec_p_(cpr)
	{
	}


private:

	virtual void run();		///<������ ������ ������������� ������

    void    learn(void);
	int    recognize(void);
    void    doPCA(void);
    void    storeTrainingData(void);
    int     loadTrainingData();
    int     findNearestNeighbor(float* projectedTestFace);
    int     loadFaceImgArray();
    void    detectFaces(IplImage* img, int ident);
	int	loadBase(); //�������� ���� �����������
	void SendNewFace(int id, int x, int y);
	int addFace(char *name); //�������� ����� ����

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
	IplImage* testImg;
	CvMat * trainPersonNumMat;
	

};

#endif