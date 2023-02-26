#include "Application.h"

int main()
{
  Application *app = new Application(800, 600);
  app->run();

  return 0;
}
