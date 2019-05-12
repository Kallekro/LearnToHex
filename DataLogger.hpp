#ifndef DATALOGGER_HPP
#define DATALOGGER_HPP
#include <fstream>
class DataLogger {
public:
	std::ofstream OutStream;

	DataLogger(std::string fname) : m_filename(fname) {
		OutStream.open(m_filename);
	}
	~DataLogger() {
		OutStream.close();
	}
private:
	std::string m_filename;
};
#endif