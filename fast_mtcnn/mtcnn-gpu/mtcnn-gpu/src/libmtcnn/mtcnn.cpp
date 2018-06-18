#include "mtcnn.hpp"

typedef std::map<std::string, MtcnnFactory::creator> creator_map;

static creator_map& get_registery(void)
{
	static creator_map * instance_ptr=new creator_map();

	return *instance_ptr;
}

void MtcnnFactory::RegisterCreator(const std::string& name, creator& create_func)
{
	creator_map& registery=get_registery();

	registery[name]=create_func;
}

std::vector<std::string> MtcnnFactory::ListDetectorType(void)
{
	std::vector<std::string> ret;

	creator_map& registery=get_registery();

	creator_map::iterator it=registery.begin();

	while(it!=registery.end())
	{
		ret.push_back(it->first);
		it++;
	}

	return ret;
}


Mtcnn * MtcnnFactory::CreateDetector(const std::string& name)
{
	creator_map& registery=get_registery();

	if(registery.find(name)== registery.end())
		return nullptr;

	creator func=registery[name];

	return func();
}

