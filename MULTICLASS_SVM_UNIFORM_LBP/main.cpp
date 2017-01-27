#include "global.h"

int main()
{
	int x;
	Box B;
	cout << "Press 1 to train" << endl;
	cout << "Press 2 to test" << endl;
	cout << "Press 3 to exit" << endl;
	cin >> x;
	if (x == 1)
	{

		B.train1();
		return 0;
	}
	else if (x == 2)
	{
		B.test();
		return 0;
	}
	else
	{
		return -1;

	}


}