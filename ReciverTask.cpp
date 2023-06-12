#include "ReciverTask.h"

#include "Poco/Net/DatagramSocket.h"
#include "Poco/Mutex.h"
#include <deque>

#include "Poco/Format.h" 

#include <fstream>

#include <stdio.h>

#include <sys/stat.h>
#include <memory.h>
#include <fcntl.h>

using namespace Poco::Net;
using Poco::Net::DatagramSocket;
using Poco::Mutex;
using std::deque;


extern Mutex mut_new_data;
extern deque<MessageFromServo> deq;




//-------------------------------------------------------------------------------------------
void ReciverTask::run()
{
 printf("Start receiver\n");

 //јдрес и порт приложени€ распознавани€
 SocketAddress addr("localhost",11001);
// SocketAddress addr(c_r_.servo_application_ip.c_str(),c_r_.servo_application_socket);
 DatagramSocket ds;
 ds.bind(addr);

 for(;;){
	ds.receiveBytes(&m_servo,sizeof(m_servo));

	mut_new_data.lock();
	deq.push_back(m_servo);
	mut_new_data.unlock();

	//Sleep(50); //ожидание
	
 }


}