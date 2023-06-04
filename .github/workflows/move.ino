#include<Servo.h>
#include <Stepper.h>

const int stepsPerRevolution = 2038;
Stepper myStepper = Stepper(stepsPerRevolution, 8, 10, 9, 11);
Servo servo1;

int i;
int r;
int a;
void setup() {
  // put your setup code here, to run once:
  servo1.attach(7);
  delay(1000);
}

void loop() {

while(a==r){
a=random(0,3)*90;
}
r=a;
servo1.write(90);
	myStepper.setSpeed(14);
	myStepper.step(stepsPerRevolution);
  	myStepper.setSpeed(14);
	myStepper.step(stepsPerRevolution);
  	myStepper.setSpeed(14);
	myStepper.step(stepsPerRevolution);
	delay(1000);
}

