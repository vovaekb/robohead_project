/**
*   @Brief: Reonocimiento de rostros, basado en "eigenface.c, by Robin Hewitt, 2007"
*
*
**/

#include <iostream>
#include <stdio.h>

#include "ReciverTask.h"
#include "RecogTask.h"


#include <Poco/AutoPtr.h>

#include <Poco/Util/IniFileConfiguration.h>

#include "Poco/Mutex.h"
#include "Poco/NamedMutex.h"
#include <deque>

#include "Poco/Runnable.h"
#include "Poco/Thread.h"



using std::string;
using Poco::AutoPtr;
using Poco::Util::IniFileConfiguration;
using Poco::Mutex;
using Poco::NamedMutex;
using Poco::Runnable;
using std::deque;

Mutex mut_new_data;				///<ћутекс, используетс€ дл€ последовательного доступа к деку потоков приема файлов и распознавани€ 
deque<MessageFromServo> deq;	///<ќсновной дек данных, передаваемых из потока приема файлов в распознавание


///ћаксимальное количество потоков, запускаемых программой
#define MAX_THREADS 8



int main( int argc, char** argv )
{

 NamedMutex StartSingleCopy("Face_Application");	
 if (!StartSingleCopy.tryLock()) return 0;					//ѕроверка на запуск единичного экземпл€ра приложени€

 recog_pars rec_par;  //параметры, используемые потоками распознавани€
 cadr_receive cadr_r; //параметры, используемые потоком приема кадров


//„тение файла конфигурации
 AutoPtr<IniFileConfiguration> cfg(new IniFileConfiguration);
 try
 {
	cfg->load("conf.ini");

	//threads=cfg->getInt("DEBUG.threads",1);  //считываем кол-во потоков распознавани€

	rec_par.face_thresh=cfg->getDouble("RECOG.FaceThresh",100000000.0);	//порог

	
/*
	cadr_r.recog_application_ip     = cfg->getString("IPC.ToRecognizeFromSelectorIp","localhost");
	cadr_r.recog_application_socket = cfg->getInt("IPC.ToRecognizeFromSelectorPort",14010);

	rec_par.cadr_application_ip     = cfg->getString("IPC.ToSelectorFromScalesIp","localhost");
	rec_par.cadr_application_port = cfg->getInt("IPC.ToSelectorFromRecognizePort",14005);
*/






	printf("Loading conf ok\n");

 }
 catch(...)
 {
	printf("Error loading config!\n");
	return -1;
 }





 Poco::Thread thread[MAX_THREADS];

 ReciverTask *synch_task = new ReciverTask(cadr_r);  //ѕоток приема команд
 thread[0].start(*synch_task);

 RecogTask *input_task = new RecogTask(rec_par); //1й поток распознавани€. ѕрисутствует всегда
 thread[1].start(*input_task);


 thread[0].join(); 
 thread[1].join(); 

    return 0;
}
