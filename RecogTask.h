/**	\file RecogTask.h
	\brief Заголовочный файл с описанием класса потока распознавания
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


#define MAX_IMG_COUNT 512	///<Максимальное кол-во изображений

#define IMG_WIDTH 120
#define IMG_HEIGHT 120

//CvVideoWriter *writer = 0;



///Структура, содержащая шаблон с определенным номером
struct faces_main
{
	int img_id;
	char name[64];
};



/** \struct recog_pars	
       \brief Параметры конфигурации 
 */
struct recog_pars
{

	double face_thresh;
/*
	int bort_num_height;	///<средняя высота цифр номера в пикселях на борту
	int beam_num_height;	///<средняя высота цифр номера в пикселях на балке
	int pl_num_height;		///<средняя высота цифр номера в пикселях на платформе
	int openw;				///<признак включения в рассмотрение полувагонов
	int pl;					///<признак включения в рассмотрение платформ
	int cis;				///<признак включения в рассмотрение цистерн
	int box;				///<признак включения в рассмотрение крытых вагонов
	int other;				///<признак включения в рассмотрение остальных типов вагонов
	int beam;				///<признак включения в рассмотрение балки
	int hoppers;			///<признак включения хопперов
	int open_and_box;		///<признак включения и полувагонов и крытых (рассматриваем как одну группу)

	double bortt;	///<верхняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера (борт, верх кадра)
	double bortb;	///<нижняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера (борт, верх кадра)
	double beamt;	///<верхняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера(балка, низ кадра)
	double beamb;	///<нижняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера(балка, низ кадра)

	double bortt_dc;	///<верхняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера (борт, верх кадра)
	double bortb_dc;	///<нижняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера (борт, верх кадра)
	double beamt_dc;	///<верхняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера(балка, низ кадра)
	double beamb_dc;	///<нижняя граница области кадра в процентах от исходного, в рамках которой ведется поиск номера(балка, низ кадра)

	int templCount;			///<Количество шаблонов

	string host;			///<имя или IP-адрес хоста на котором хранится БД
	string user;			///<Имя пользователя для соединения с БД
	string password;		///<Пароль для соединения с БД
	string db_default;		///<База данных по умодчанию
	int port;				///<Порт для содиения с БД
	int sqliteBaseUsed;		///<Признак использования базы данных SQLite
	string sqlitePath;		///<Путь к каталогу с базами данных SQLite
	int sqliteKeepTime;		///<Сколько месяцев хранить кадры

	int oneCadrForHighReliability; ///<записывать ли в базу только один кадр с распознанным номером c высокой степенью достоверности(1-да, 0-нет)
	int twoCadrsForLowReliability; ///<записывать ли в базу только 2 кадра для номера, распознанного с низкой степенью достоверности
	int allCadrsForNonRecognized; ///<записывать ли в базу 4 кадра для нераспознанного номера
	int inputOutput;			///<Работа по таблице груженых/порожних вагонов
	int inputDays;			///<Параметр задает глубину сканирования базы входящих вагонов по времени (в днях)
	int inputDir;			///<Параметр задает, какое направление движения вагонов считать входящим на предприятие (-1 - справа налево, 1 - слева направо)
	
	string img_save_path;	///<основная директория, куда сохраняются файлы

	string cadr_application_ip;	///<IP-адрес для модуля распознавания
	int cadr_application_port;	///<номер порта для модуля распознавания

	int first_group[4];		///<Группа, которая считается первой, по ней производится попытка распознать номер вагона
*/
} ;

/**
	\class RecogTask
	\brief Класс потока для распознавания
*/
class  RecogTask : public Runnable
{
	recog_pars & rec_p_;

public:
	RecogTask:: RecogTask(recog_pars &cpr):rec_p_(cpr)
	{
	}


private:

	virtual void run();		///<Запуск потока распознавания кадров

    void    learn(void);
	int    recognize(void);
    void    doPCA(void);
    void    storeTrainingData(void);
    int     loadTrainingData();
    int     findNearestNeighbor(float* projectedTestFace);
    int     loadFaceImgArray();
    void    detectFaces(IplImage* img, int ident);
	int	loadBase(); //загрузка базы изображений
	void SendNewFace(int id, int x, int y);
	int addFace(char *name); //добавить новое лицо

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