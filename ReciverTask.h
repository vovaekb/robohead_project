/**	\file ReciverTask.h
		\brief Файл с описанием класса потока приема кадров
*/
#ifndef RECIVER_TASK
#define RECIVER_TASK

#include "Poco/Runnable.h"
#include "Poco/Thread.h"

using std::string;
using Poco::Runnable;



/** \struct MessageFromSynch	
     \brief Структура сообщения, передаваемого в распознавание. 
 */
struct MessageFromServo	
{

	int id;			///<идентификатор
	int x;			///<идентификатор
	int y;			///<идентификатор
};

/**
	\struct cadr_receive
	\brief Параметры передачи данных
*/
typedef struct
{

	string servo_application_ip;	///<IP-адрес для модуля серво
	int servo_application_socket;	///<номер порта для модуля серво

} cadr_receive;


/**
	\class ReciverTask
	\brief Класс для приема данных от приложения серво
*/
class ReciverTask : public Runnable
{
	cadr_receive &c_r_;
public:
	ReciverTask(cadr_receive &cre):c_r_(cre)
	{
	}

private:

	MessageFromServo m_servo;  ///<Сообщение, передаваемое в распознавание

	
	virtual void run();	///<Запуск потока приема кадров
};

#endif