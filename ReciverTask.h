/**	\file ReciverTask.h
		\brief ���� � ��������� ������ ������ ������ ������
*/
#ifndef RECIVER_TASK
#define RECIVER_TASK

#include "Poco/Runnable.h"
#include "Poco/Thread.h"

using std::string;
using Poco::Runnable;



/** \struct MessageFromSynch	
     \brief ��������� ���������, ������������� � �������������. 
 */
struct MessageFromServo	
{

	int id;			///<�������������
	int x;			///<�������������
	int y;			///<�������������
};

/**
	\struct cadr_receive
	\brief ��������� �������� ������
*/
typedef struct
{

	string servo_application_ip;	///<IP-����� ��� ������ �����
	int servo_application_socket;	///<����� ����� ��� ������ �����

} cadr_receive;


/**
	\class ReciverTask
	\brief ����� ��� ������ ������ �� ���������� �����
*/
class ReciverTask : public Runnable
{
	cadr_receive &c_r_;
public:
	ReciverTask(cadr_receive &cre):c_r_(cre)
	{
	}

private:

	MessageFromServo m_servo;  ///<���������, ������������ � �������������

	
	virtual void run();	///<������ ������ ������ ������
};

#endif